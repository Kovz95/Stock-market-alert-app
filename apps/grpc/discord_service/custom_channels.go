package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	"stockalert/expr"

	db "stockalert/database/generated"
	discordv1 "stockalert/gen/go/discord/v1"
)

const customChannelsDocKey = "custom_discord_channels"

var wsRE = regexp.MustCompile(`\s+`)

// customChannelEntry is the on-disk JSON shape for a single custom channel.
// Fields must match the Python document_store schema exactly.
type customChannelEntry struct {
	WebhookURL  string `json:"webhook_url"`
	ChannelName string `json:"channel_name"`
	Description string `json:"description"`
	Condition   string `json:"condition"`
	Enabled     bool   `json:"enabled"`
	Created     string `json:"created"`
}

// loadCustomChannels reads the raw custom_discord_channels document.
// Returns map[name]raw-entry-bytes so unknown keys are preserved on write.
func (s *Server) loadCustomChannels(ctx context.Context) (map[string]json.RawMessage, error) {
	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "acquire connection: %v", err)
	}
	defer conn.Release()

	doc, err := db.New(conn).GetAppDocument(ctx, customChannelsDocKey)
	if err != nil {
		if err == pgx.ErrNoRows {
			return map[string]json.RawMessage{}, nil
		}
		return nil, status.Errorf(codes.Internal, "load custom channels: %v", err)
	}

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(doc.Payload, &raw); err != nil {
		return nil, status.Errorf(codes.Internal, "parse custom channels: %v", err)
	}
	if raw == nil {
		raw = map[string]json.RawMessage{}
	}
	return raw, nil
}

// saveCustomChannels persists the raw document map back to the database.
func (s *Server) saveCustomChannels(ctx context.Context, raw map[string]json.RawMessage) error {
	payload, err := json.Marshal(raw)
	if err != nil {
		return err
	}
	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return err
	}
	defer conn.Release()
	return db.New(conn).UpsertAppDocument(ctx, db.UpsertAppDocumentParams{
		DocumentKey: customChannelsDocKey,
		Payload:     payload,
		SourcePath:  pgtype.Text{},
	})
}

// getEntry unmarshals a single entry from the raw map.
func getEntry(raw map[string]json.RawMessage, name string) (customChannelEntry, bool) {
	b, ok := raw[name]
	if !ok {
		return customChannelEntry{}, false
	}
	var e customChannelEntry
	_ = json.Unmarshal(b, &e)
	return e, true
}

// setEntry writes a single entry into the raw map, preserving unknown JSON keys.
func setEntry(raw map[string]json.RawMessage, name string, e customChannelEntry) error {
	// Start from existing raw bytes (so unknown keys from Python survive).
	existing := map[string]interface{}{}
	if b, ok := raw[name]; ok {
		_ = json.Unmarshal(b, &existing)
	}
	existing["webhook_url"] = e.WebhookURL
	existing["channel_name"] = e.ChannelName
	existing["description"] = e.Description
	existing["condition"] = e.Condition
	existing["enabled"] = e.Enabled
	existing["created"] = e.Created

	b, err := json.Marshal(existing)
	if err != nil {
		return err
	}
	raw[name] = b
	return nil
}

// toProtoChannel converts a persisted entry to the proto message.
func toProtoChannel(name string, e customChannelEntry) *discordv1.CustomDiscordChannel {
	ch := &discordv1.CustomDiscordChannel{
		Name:        name,
		ChannelName: e.ChannelName,
		Description: e.Description,
		WebhookUrl:  e.WebhookURL,
		Condition:   e.Condition,
		Enabled:     e.Enabled,
	}
	if e.Created != "" {
		if t, err := time.Parse(time.RFC3339, e.Created); err == nil {
			ch.CreatedAt = timestamppb.New(t)
		}
	}
	return ch
}

// deriveChannelName produces "#kebab-of-name" matching the Streamlit behaviour.
func deriveChannelName(name string) string {
	return "#" + wsRE.ReplaceAllString(strings.ToLower(name), "-")
}

// validateCondition returns ("", nil) for "price_level", runs expr.ParseCondition otherwise.
func validateCondition(condition string) (string, error) {
	if strings.EqualFold(condition, "price_level") {
		return "price_level", nil
	}
	if _, err := expr.ParseCondition(condition); err != nil {
		return "", fmt.Errorf("condition failed to parse: %w", err)
	}
	return condition, nil
}

// ---------------------------------------------------------------------------
// RPC implementations
// ---------------------------------------------------------------------------

func (s *Server) ListCustomDiscordChannels(ctx context.Context, _ *discordv1.ListCustomDiscordChannelsRequest) (*discordv1.ListCustomDiscordChannelsResponse, error) {
	raw, err := s.loadCustomChannels(ctx)
	if err != nil {
		return nil, err
	}
	channels := make([]*discordv1.CustomDiscordChannel, 0, len(raw))
	for name := range raw {
		e, _ := getEntry(raw, name)
		channels = append(channels, toProtoChannel(name, e))
	}
	return &discordv1.ListCustomDiscordChannelsResponse{Channels: channels}, nil
}

func (s *Server) CreateCustomDiscordChannel(ctx context.Context, req *discordv1.CreateCustomDiscordChannelRequest) (*discordv1.CreateCustomDiscordChannelResponse, error) {
	name := strings.TrimSpace(req.GetName())
	if name == "" {
		return nil, status.Error(codes.InvalidArgument, "name is required")
	}

	condition, err := validateCondition(strings.TrimSpace(req.GetCondition()))
	if err != nil {
		return &discordv1.CreateCustomDiscordChannelResponse{
			Success: false, ErrorMessage: err.Error(),
		}, nil
	}

	raw, loadErr := s.loadCustomChannels(ctx)
	if loadErr != nil {
		return nil, loadErr
	}

	if _, exists := raw[name]; exists {
		return &discordv1.CreateCustomDiscordChannelResponse{
			Success: false, ErrorMessage: fmt.Sprintf("channel '%s' already exists", name),
		}, nil
	}

	entry := customChannelEntry{
		WebhookURL:  req.GetWebhookUrl(),
		ChannelName: deriveChannelName(name),
		Description: req.GetDescription(),
		Condition:   condition,
		Enabled:     req.GetEnabled(),
		Created:     time.Now().UTC().Format(time.RFC3339),
	}
	if err := setEntry(raw, name, entry); err != nil {
		return nil, status.Errorf(codes.Internal, "marshal entry: %v", err)
	}
	if err := s.saveCustomChannels(ctx, raw); err != nil {
		return &discordv1.CreateCustomDiscordChannelResponse{
			Success: false, ErrorMessage: err.Error(),
		}, nil
	}
	return &discordv1.CreateCustomDiscordChannelResponse{
		Success: true,
		Channel: toProtoChannel(name, entry),
	}, nil
}

func (s *Server) UpdateCustomDiscordChannel(ctx context.Context, req *discordv1.UpdateCustomDiscordChannelRequest) (*discordv1.UpdateCustomDiscordChannelResponse, error) {
	name := strings.TrimSpace(req.GetName())
	if name == "" {
		return nil, status.Error(codes.InvalidArgument, "name is required")
	}

	raw, loadErr := s.loadCustomChannels(ctx)
	if loadErr != nil {
		return nil, loadErr
	}

	entry, ok := getEntry(raw, name)
	if !ok {
		return &discordv1.UpdateCustomDiscordChannelResponse{
			Success: false, ErrorMessage: fmt.Sprintf("channel '%s' not found", name),
		}, nil
	}

	if req.WebhookUrl != nil {
		entry.WebhookURL = req.GetWebhookUrl()
	}
	if req.Description != nil {
		entry.Description = req.GetDescription()
	}
	if req.Condition != nil {
		validated, err := validateCondition(strings.TrimSpace(req.GetCondition()))
		if err != nil {
			return &discordv1.UpdateCustomDiscordChannelResponse{
				Success: false, ErrorMessage: err.Error(),
			}, nil
		}
		entry.Condition = validated
	}
	if req.Enabled != nil {
		entry.Enabled = req.GetEnabled()
	}

	if err := setEntry(raw, name, entry); err != nil {
		return nil, status.Errorf(codes.Internal, "marshal entry: %v", err)
	}
	if err := s.saveCustomChannels(ctx, raw); err != nil {
		return &discordv1.UpdateCustomDiscordChannelResponse{
			Success: false, ErrorMessage: err.Error(),
		}, nil
	}
	return &discordv1.UpdateCustomDiscordChannelResponse{
		Success: true,
		Channel: toProtoChannel(name, entry),
	}, nil
}

func (s *Server) DeleteCustomDiscordChannel(ctx context.Context, req *discordv1.DeleteCustomDiscordChannelRequest) (*discordv1.DeleteCustomDiscordChannelResponse, error) {
	name := strings.TrimSpace(req.GetName())
	if name == "" {
		return nil, status.Error(codes.InvalidArgument, "name is required")
	}

	raw, loadErr := s.loadCustomChannels(ctx)
	if loadErr != nil {
		return nil, loadErr
	}

	if _, ok := raw[name]; !ok {
		return &discordv1.DeleteCustomDiscordChannelResponse{
			Success: false, ErrorMessage: fmt.Sprintf("channel '%s' not found", name),
		}, nil
	}

	delete(raw, name)
	if err := s.saveCustomChannels(ctx, raw); err != nil {
		return &discordv1.DeleteCustomDiscordChannelResponse{
			Success: false, ErrorMessage: err.Error(),
		}, nil
	}
	return &discordv1.DeleteCustomDiscordChannelResponse{Success: true}, nil
}

func (s *Server) SendCustomDiscordChannelTestMessage(ctx context.Context, req *discordv1.SendCustomDiscordChannelTestMessageRequest) (*discordv1.SendCustomDiscordChannelTestMessageResponse, error) {
	name := strings.TrimSpace(req.GetName())
	if name == "" {
		return nil, status.Error(codes.InvalidArgument, "name is required")
	}

	raw, loadErr := s.loadCustomChannels(ctx)
	if loadErr != nil {
		return nil, loadErr
	}

	entry, ok := getEntry(raw, name)
	if !ok {
		return &discordv1.SendCustomDiscordChannelTestMessageResponse{
			Success: false, ErrorMessage: fmt.Sprintf("channel '%s' not found", name),
		}, nil
	}
	if entry.WebhookURL == "" || isPlaceholderWebhook(entry.WebhookURL) {
		return &discordv1.SendCustomDiscordChannelTestMessageResponse{
			Success: false, ErrorMessage: "webhook not configured for this channel",
		}, nil
	}

	payload := map[string]interface{}{
		"embeds": []map[string]interface{}{
			{
				"title":       fmt.Sprintf("🧪 Test — %s", name),
				"description": fmt.Sprintf("Condition: `%s`\nSent at: %s", entry.Condition, time.Now().UTC().Format("2006-01-02 15:04:05 UTC")),
				"color":       0x00FF00,
			},
		},
	}
	body, _ := json.Marshal(payload)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, entry.WebhookURL, bytes.NewReader(body))
	if err != nil {
		return &discordv1.SendCustomDiscordChannelTestMessageResponse{
			Success: false, ErrorMessage: err.Error(),
		}, nil
	}
	httpReq.Header.Set("Content-Type", "application/json")
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return &discordv1.SendCustomDiscordChannelTestMessageResponse{
			Success: false, ErrorMessage: err.Error(),
		}, nil
	}
	defer resp.Body.Close()
	if resp.StatusCode != 204 && resp.StatusCode != 200 {
		return &discordv1.SendCustomDiscordChannelTestMessageResponse{
			Success: false, ErrorMessage: fmt.Sprintf("HTTP %d", resp.StatusCode),
		}, nil
	}
	return &discordv1.SendCustomDiscordChannelTestMessageResponse{Success: true}, nil
}

package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	db "stockalert/database/generated"
	discordv1 "stockalert/gen/go/discord/v1"
)

func (s *Server) GetHourlyDiscordConfig(ctx context.Context, req *discordv1.GetHourlyDiscordConfigRequest) (*discordv1.GetHourlyDiscordConfigResponse, error) {
	cfg, err := s.loadConfig(ctx)
	if err != nil {
		return nil, err
	}
	hourly := cfg.ChannelMappingsHourly
	if hourly == nil {
		hourly = make(map[string]ChannelEntry)
	}

	var channels []*discordv1.HourlyChannelInfo
	configured := 0
	for name, entry := range hourly {
		webhook := entry.WebhookURL
		ok := webhook != "" && !isPlaceholderWebhook(webhook)
		if ok {
			configured++
		}
		chName := entry.ChannelName
		if chName == "" {
			chName = name
		}
		channels = append(channels, &discordv1.HourlyChannelInfo{
			Name:        name,
			ChannelName: chName,
			Description: entry.Description,
			Configured:  ok,
		})
	}
	return &discordv1.GetHourlyDiscordConfigResponse{
		EnableIndustryRouting: cfg.EnableIndustryRouting,
		LogRoutingDecisions:   cfg.LogRoutingDecisions,
		Channels:              channels,
		ConfiguredCount:       int32(configured),
		TotalCount:            int32(len(channels)),
	}, nil
}

func (s *Server) CopyDailyToHourly(ctx context.Context, req *discordv1.CopyDailyToHourlyRequest) (*discordv1.CopyDailyToHourlyResponse, error) {
	cfg, err := s.loadConfig(ctx)
	if err != nil {
		return nil, err
	}
	daily := cfg.ChannelMappingsDaily
	hourly := cfg.ChannelMappingsHourly
	if daily == nil {
		daily = make(map[string]ChannelEntry)
	}
	if hourly == nil {
		hourly = make(map[string]ChannelEntry)
	}
	for name, dailyEntry := range daily {
		if h, ok := hourly[name]; ok && dailyEntry.WebhookURL != "" {
			h.WebhookURL = dailyEntry.WebhookURL
			hourly[name] = h
		}
	}
	cfg.ChannelMappingsHourly = hourly
	if err := s.saveConfig(ctx, cfg); err != nil {
		return &discordv1.CopyDailyToHourlyResponse{Success: false, ErrorMessage: err.Error()}, nil
	}
	return &discordv1.CopyDailyToHourlyResponse{Success: true}, nil
}

func (s *Server) UpdateHourlyChannelWebhook(ctx context.Context, req *discordv1.UpdateHourlyChannelWebhookRequest) (*discordv1.UpdateHourlyChannelWebhookResponse, error) {
	if req.GetChannelName() == "" {
		return nil, status.Error(codes.InvalidArgument, "channel_name is required")
	}
	cfg, err := s.loadConfig(ctx)
	if err != nil {
		return nil, err
	}
	hourly := cfg.ChannelMappingsHourly
	if hourly == nil {
		hourly = make(map[string]ChannelEntry)
	}
	entry, ok := hourly[req.GetChannelName()]
	if !ok {
		return &discordv1.UpdateHourlyChannelWebhookResponse{
			Success: false, ErrorMessage: "channel not found in hourly config",
		}, nil
	}
	entry.WebhookURL = req.GetWebhookUrl()
	hourly[req.GetChannelName()] = entry
	cfg.ChannelMappingsHourly = hourly
	if err := s.saveConfig(ctx, cfg); err != nil {
		return &discordv1.UpdateHourlyChannelWebhookResponse{Success: false, ErrorMessage: err.Error()}, nil
	}
	return &discordv1.UpdateHourlyChannelWebhookResponse{Success: true}, nil
}

func (s *Server) ResolveHourlyChannelForTicker(ctx context.Context, req *discordv1.ResolveHourlyChannelForTickerRequest) (*discordv1.ResolveHourlyChannelForTickerResponse, error) {
	ticker := strings.TrimSpace(req.GetTicker())
	if ticker == "" {
		return &discordv1.ResolveHourlyChannelForTickerResponse{}, nil
	}
	cfg, err := s.loadConfig(ctx)
	if err != nil {
		return nil, err
	}
	hourly := cfg.ChannelMappingsHourly
	if hourly == nil {
		hourly = make(map[string]ChannelEntry)
	}

	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()
	q := db.New(conn)

	// Try symbol as-is, then uppercase, then with -US suffix (if no hyphen).
	symbols := []string{ticker, strings.ToUpper(ticker)}
	if !strings.Contains(ticker, "-") && !strings.HasSuffix(ticker, ".") {
		symbols = append(symbols, ticker+"-US", strings.ToUpper(ticker)+"-US")
	}
	var economy string
	for _, sym := range symbols {
		row, err := q.GetStockEconomyBySymbol(ctx, sym)
		if err != nil {
			if err == pgx.ErrNoRows {
				continue
			}
			return nil, status.Errorf(codes.Internal, "lookup economy: %v", err)
		}
		if row.AssetType.Valid && row.AssetType.String == "ETF" {
			economy = "ETFs"
			break
		}
		if row.RbicsEconomy.Valid && row.RbicsEconomy.String != "" {
			economy = row.RbicsEconomy.String
			break
		}
	}
	if economy == "" {
		return &discordv1.ResolveHourlyChannelForTickerResponse{}, nil
	}
	entry, ok := hourly[economy]
	if !ok {
		return &discordv1.ResolveHourlyChannelForTickerResponse{Economy: economy}, nil
	}
	chName := entry.ChannelName
	if chName == "" {
		chName = economy
	}
	webhookOK := entry.WebhookURL != "" && !isPlaceholderWebhook(entry.WebhookURL)
	return &discordv1.ResolveHourlyChannelForTickerResponse{
		Economy:             economy,
		HourlyChannelName:   chName,
		WebhookConfigured:   webhookOK,
	}, nil
}

func (s *Server) SendHourlyTestMessage(ctx context.Context, req *discordv1.SendHourlyTestMessageRequest) (*discordv1.SendHourlyTestMessageResponse, error) {
	if req.GetChannelName() == "" {
		return nil, status.Error(codes.InvalidArgument, "channel_name is required")
	}
	cfg, err := s.loadConfig(ctx)
	if err != nil {
		return nil, err
	}
	hourly := cfg.ChannelMappingsHourly
	if hourly == nil {
		return &discordv1.SendHourlyTestMessageResponse{Success: false, ErrorMessage: "no hourly channels configured"}, nil
	}
	entry, ok := hourly[req.GetChannelName()]
	if !ok || entry.WebhookURL == "" || isPlaceholderWebhook(entry.WebhookURL) {
		return &discordv1.SendHourlyTestMessageResponse{Success: false, ErrorMessage: "webhook not configured for this channel"}, nil
	}
	chName := entry.ChannelName
	if chName == "" {
		chName = req.GetChannelName()
	}
	body := map[string]string{
		"content": fmt.Sprintf("**Test message – Hourly routing**\n📊 This is a test from **Hourly Discord Management** to verify the webhook for **%s**.\n⏰ Sent at: %s\n✅ If you see this, the channel is configured correctly.",
			chName, time.Now().UTC().Format("2006-01-02 15:04:05")),
		"username": "Hourly Alert Test",
	}
	payload, _ := json.Marshal(body)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, entry.WebhookURL, bytes.NewReader(payload))
	if err != nil {
		return &discordv1.SendHourlyTestMessageResponse{Success: false, ErrorMessage: err.Error()}, nil
	}
	httpReq.Header.Set("Content-Type", "application/json")
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(httpReq)
	if err != nil {
		return &discordv1.SendHourlyTestMessageResponse{Success: false, ErrorMessage: err.Error()}, nil
	}
	defer resp.Body.Close()
	if resp.StatusCode != 204 && resp.StatusCode != 200 {
		return &discordv1.SendHourlyTestMessageResponse{
			Success: false, ErrorMessage: fmt.Sprintf("HTTP %d", resp.StatusCode),
		}, nil
	}
	return &discordv1.SendHourlyTestMessageResponse{Success: true}, nil
}

func (s *Server) loadConfig(ctx context.Context) (*DiscordChannelsConfig, error) {
	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "acquire connection: %v", err)
	}
	defer conn.Release()
	doc, err := db.New(conn).GetAppDocument(ctx, discordConfigDocumentKey)
	if err != nil {
		if err == pgx.ErrNoRows {
			return &DiscordChannelsConfig{
				ChannelMappings:       make(map[string]ChannelEntry),
				ChannelMappingsDaily:  make(map[string]ChannelEntry),
				ChannelMappingsHourly: make(map[string]ChannelEntry),
				ChannelMappingsWeekly: make(map[string]ChannelEntry),
				DefaultChannel:        "General",
			}, nil
		}
		return nil, status.Errorf(codes.Internal, "load config: %v", err)
	}
	var cfg DiscordChannelsConfig
	if err := json.Unmarshal(doc.Payload, &cfg); err != nil {
		return nil, status.Errorf(codes.Internal, "parse config: %v", err)
	}
	if cfg.ChannelMappings == nil {
		cfg.ChannelMappings = make(map[string]ChannelEntry)
	}
	if cfg.ChannelMappingsDaily == nil {
		cfg.ChannelMappingsDaily = make(map[string]ChannelEntry)
	}
	if cfg.ChannelMappingsHourly == nil {
		cfg.ChannelMappingsHourly = make(map[string]ChannelEntry)
	}
	if cfg.ChannelMappingsWeekly == nil {
		cfg.ChannelMappingsWeekly = make(map[string]ChannelEntry)
	}
	return &cfg, nil
}

func (s *Server) saveConfig(ctx context.Context, cfg *DiscordChannelsConfig) error {
	payload, err := json.Marshal(cfg)
	if err != nil {
		return err
	}
	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return err
	}
	defer conn.Release()
	return db.New(conn).UpsertAppDocument(ctx, db.UpsertAppDocumentParams{
		DocumentKey: discordConfigDocumentKey,
		Payload:     payload,
		SourcePath:  pgtype.Text{},
	})
}

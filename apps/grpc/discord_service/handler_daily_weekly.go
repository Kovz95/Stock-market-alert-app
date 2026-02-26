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
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	db "stockalert/database/generated"
	discordv1 "stockalert/gen/go/discord/v1"
)

// getMapping returns the channel map for the given timeframe ("daily" or "weekly").
func (s *Server) getMapping(cfg *DiscordChannelsConfig, timeframe string) map[string]ChannelEntry {
	switch timeframe {
	case "daily":
		if cfg.ChannelMappingsDaily == nil {
			return make(map[string]ChannelEntry)
		}
		return cfg.ChannelMappingsDaily
	case "weekly":
		if cfg.ChannelMappingsWeekly == nil {
			return make(map[string]ChannelEntry)
		}
		return cfg.ChannelMappingsWeekly
	default:
		return make(map[string]ChannelEntry)
	}
}

func (s *Server) setMapping(cfg *DiscordChannelsConfig, timeframe string, m map[string]ChannelEntry) {
	switch timeframe {
	case "daily":
		cfg.ChannelMappingsDaily = m
	case "weekly":
		cfg.ChannelMappingsWeekly = m
	}
}

func buildConfigResponse(cfg *DiscordChannelsConfig, mapping map[string]ChannelEntry) *discordv1.GetHourlyDiscordConfigResponse {
	var channels []*discordv1.HourlyChannelInfo
	configured := 0
	for name, entry := range mapping {
		ok := entry.WebhookURL != "" && !isPlaceholderWebhook(entry.WebhookURL)
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
	}
}

func (s *Server) GetDailyDiscordConfig(ctx context.Context, req *discordv1.GetDailyDiscordConfigRequest) (*discordv1.GetHourlyDiscordConfigResponse, error) {
	cfg, err := s.loadConfig(ctx)
	if err != nil {
		return nil, err
	}
	m := s.getMapping(cfg, "daily")
	return buildConfigResponse(cfg, m), nil
}

func (s *Server) GetWeeklyDiscordConfig(ctx context.Context, req *discordv1.GetWeeklyDiscordConfigRequest) (*discordv1.GetHourlyDiscordConfigResponse, error) {
	cfg, err := s.loadConfig(ctx)
	if err != nil {
		return nil, err
	}
	m := s.getMapping(cfg, "weekly")
	return buildConfigResponse(cfg, m), nil
}

func (s *Server) CopyBaseToDaily(ctx context.Context, req *discordv1.CopyBaseToDailyRequest) (*discordv1.CopyDailyToHourlyResponse, error) {
	cfg, err := s.loadConfig(ctx)
	if err != nil {
		return nil, err
	}
	base := cfg.ChannelMappings
	daily := s.getMapping(cfg, "daily")
	if base == nil {
		base = make(map[string]ChannelEntry)
	}
	if daily == nil {
		daily = make(map[string]ChannelEntry)
	}
	for name, baseEntry := range base {
		if d, ok := daily[name]; ok && baseEntry.WebhookURL != "" {
			d.WebhookURL = baseEntry.WebhookURL
			daily[name] = d
		}
	}
	s.setMapping(cfg, "daily", daily)
	if err := s.saveConfig(ctx, cfg); err != nil {
		return &discordv1.CopyDailyToHourlyResponse{Success: false, ErrorMessage: err.Error()}, nil
	}
	return &discordv1.CopyDailyToHourlyResponse{Success: true}, nil
}

func (s *Server) CopyBaseToWeekly(ctx context.Context, req *discordv1.CopyBaseToWeeklyRequest) (*discordv1.CopyDailyToHourlyResponse, error) {
	cfg, err := s.loadConfig(ctx)
	if err != nil {
		return nil, err
	}
	base := cfg.ChannelMappings
	weekly := s.getMapping(cfg, "weekly")
	if base == nil {
		base = make(map[string]ChannelEntry)
	}
	if weekly == nil {
		weekly = make(map[string]ChannelEntry)
	}
	for name, baseEntry := range base {
		if w, ok := weekly[name]; ok && baseEntry.WebhookURL != "" {
			w.WebhookURL = baseEntry.WebhookURL
			weekly[name] = w
		}
	}
	s.setMapping(cfg, "weekly", weekly)
	if err := s.saveConfig(ctx, cfg); err != nil {
		return &discordv1.CopyDailyToHourlyResponse{Success: false, ErrorMessage: err.Error()}, nil
	}
	return &discordv1.CopyDailyToHourlyResponse{Success: true}, nil
}

func (s *Server) UpdateDailyChannelWebhook(ctx context.Context, req *discordv1.UpdateDailyChannelWebhookRequest) (*discordv1.UpdateHourlyChannelWebhookResponse, error) {
	if req.GetChannelName() == "" {
		return nil, status.Error(codes.InvalidArgument, "channel_name is required")
	}
	cfg, err := s.loadConfig(ctx)
	if err != nil {
		return nil, err
	}
	m := s.getMapping(cfg, "daily")
	entry, ok := m[req.GetChannelName()]
	if !ok {
		return &discordv1.UpdateHourlyChannelWebhookResponse{
			Success: false, ErrorMessage: "channel not found in daily config",
		}, nil
	}
	entry.WebhookURL = req.GetWebhookUrl()
	m[req.GetChannelName()] = entry
	s.setMapping(cfg, "daily", m)
	if err := s.saveConfig(ctx, cfg); err != nil {
		return &discordv1.UpdateHourlyChannelWebhookResponse{Success: false, ErrorMessage: err.Error()}, nil
	}
	return &discordv1.UpdateHourlyChannelWebhookResponse{Success: true}, nil
}

func (s *Server) UpdateWeeklyChannelWebhook(ctx context.Context, req *discordv1.UpdateWeeklyChannelWebhookRequest) (*discordv1.UpdateHourlyChannelWebhookResponse, error) {
	if req.GetChannelName() == "" {
		return nil, status.Error(codes.InvalidArgument, "channel_name is required")
	}
	cfg, err := s.loadConfig(ctx)
	if err != nil {
		return nil, err
	}
	m := s.getMapping(cfg, "weekly")
	entry, ok := m[req.GetChannelName()]
	if !ok {
		return &discordv1.UpdateHourlyChannelWebhookResponse{
			Success: false, ErrorMessage: "channel not found in weekly config",
		}, nil
	}
	entry.WebhookURL = req.GetWebhookUrl()
	m[req.GetChannelName()] = entry
	s.setMapping(cfg, "weekly", m)
	if err := s.saveConfig(ctx, cfg); err != nil {
		return &discordv1.UpdateHourlyChannelWebhookResponse{Success: false, ErrorMessage: err.Error()}, nil
	}
	return &discordv1.UpdateHourlyChannelWebhookResponse{Success: true}, nil
}

func (s *Server) resolveChannelForTicker(ctx context.Context, mapping map[string]ChannelEntry, ticker string) (*discordv1.ResolveHourlyChannelForTickerResponse, error) {
	ticker = strings.TrimSpace(ticker)
	if ticker == "" {
		return &discordv1.ResolveHourlyChannelForTickerResponse{}, nil
	}
	if mapping == nil {
		mapping = make(map[string]ChannelEntry)
	}
	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()
	q := db.New(conn)
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
	entry, ok := mapping[economy]
	if !ok {
		return &discordv1.ResolveHourlyChannelForTickerResponse{Economy: economy}, nil
	}
	chName := entry.ChannelName
	if chName == "" {
		chName = economy
	}
	webhookOK := entry.WebhookURL != "" && !isPlaceholderWebhook(entry.WebhookURL)
	return &discordv1.ResolveHourlyChannelForTickerResponse{
		Economy:           economy,
		HourlyChannelName: chName,
		WebhookConfigured: webhookOK,
	}, nil
}

func (s *Server) ResolveDailyChannelForTicker(ctx context.Context, req *discordv1.ResolveDailyChannelForTickerRequest) (*discordv1.ResolveHourlyChannelForTickerResponse, error) {
	cfg, err := s.loadConfig(ctx)
	if err != nil {
		return nil, err
	}
	return s.resolveChannelForTicker(ctx, s.getMapping(cfg, "daily"), req.GetTicker())
}

func (s *Server) ResolveWeeklyChannelForTicker(ctx context.Context, req *discordv1.ResolveWeeklyChannelForTickerRequest) (*discordv1.ResolveHourlyChannelForTickerResponse, error) {
	cfg, err := s.loadConfig(ctx)
	if err != nil {
		return nil, err
	}
	return s.resolveChannelForTicker(ctx, s.getMapping(cfg, "weekly"), req.GetTicker())
}

func (s *Server) sendTestMessage(ctx context.Context, mapping map[string]ChannelEntry, channelNameKey string, title string, username string) (*discordv1.SendHourlyTestMessageResponse, error) {
	if channelNameKey == "" {
		return nil, status.Error(codes.InvalidArgument, "channel_name is required")
	}
	if mapping == nil {
		return &discordv1.SendHourlyTestMessageResponse{Success: false, ErrorMessage: "no channels configured"}, nil
	}
	entry, ok := mapping[channelNameKey]
	if !ok || entry.WebhookURL == "" || isPlaceholderWebhook(entry.WebhookURL) {
		return &discordv1.SendHourlyTestMessageResponse{Success: false, ErrorMessage: "webhook not configured for this channel"}, nil
	}
	chName := entry.ChannelName
	if chName == "" {
		chName = channelNameKey
	}
	body := map[string]string{
		"content": fmt.Sprintf("**Test message – %s**\n📊 This is a test from **%s** to verify the webhook for **%s**.\n⏰ Sent at: %s\n✅ If you see this, the channel is configured correctly.",
			title, title, chName, time.Now().UTC().Format("2006-01-02 15:04:05")),
		"username": username,
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

func (s *Server) SendDailyTestMessage(ctx context.Context, req *discordv1.SendDailyTestMessageRequest) (*discordv1.SendHourlyTestMessageResponse, error) {
	cfg, err := s.loadConfig(ctx)
	if err != nil {
		return nil, err
	}
	return s.sendTestMessage(ctx, s.getMapping(cfg, "daily"), req.GetChannelName(), "Daily routing", "Daily Alert Test")
}

func (s *Server) SendWeeklyTestMessage(ctx context.Context, req *discordv1.SendWeeklyTestMessageRequest) (*discordv1.SendHourlyTestMessageResponse, error) {
	cfg, err := s.loadConfig(ctx)
	if err != nil {
		return nil, err
	}
	return s.sendTestMessage(ctx, s.getMapping(cfg, "weekly"), req.GetChannelName(), "Weekly routing", "Weekly Alert Test")
}

package discord

import (
	"context"
	"encoding/json"
	"log"
	"strings"

	"github.com/jackc/pgx/v5/pgtype"

	db "stockalert/database/generated"
)

// Router determines which Discord webhook URL to use for a given alert
// based on its economy classification, asset type (ETF), and timeframe.
//
// Configuration is loaded from the "discord_channels_config" app_documents row,
// which contains channel_mappings, channel_mappings_daily, channel_mappings_hourly,
// and channel_mappings_weekly entries.
type Router struct {
	queries  *db.Queries
	config   routerConfig
	metadata map[string]stockMeta // symbol -> metadata
}

type stockMeta struct {
	Name         string
	Economy      string
	AssetType    string
	ISIN         string
	Exchange     string
	Country      string
}

// channelMapping holds the Discord webhook URL and display name for one channel.
type channelMapping struct {
	WebhookURL  string `json:"webhook_url"`
	ChannelName string `json:"channel_name"`
	Description string `json:"description"`
}

type routerConfig struct {
	EnableIndustryRouting bool                      `json:"enable_industry_routing"`
	DefaultChannel        string                    `json:"default_channel"`
	LogRoutingDecisions   bool                      `json:"log_routing_decisions"`
	ChannelMappings       map[string]channelMapping `json:"channel_mappings"`
	ChannelMappingsDaily  map[string]channelMapping `json:"channel_mappings_daily"`
	ChannelMappingsWeekly map[string]channelMapping `json:"channel_mappings_weekly"`
	ChannelMappingsHourly map[string]channelMapping `json:"channel_mappings_hourly"`
}

// NewRouter creates a Router, loading config from app_documents and stock metadata.
func NewRouter(ctx context.Context, queries *db.Queries) (*Router, error) {
	r := &Router{
		queries:  queries,
		metadata: make(map[string]stockMeta),
	}

	// Load routing config
	if err := r.loadConfig(ctx); err != nil {
		log.Printf("discord/routing: failed to load config, using defaults: %v", err)
		r.config = defaultConfig()
	}

	// Load stock metadata for economy lookup
	if err := r.loadMetadata(ctx); err != nil {
		log.Printf("discord/routing: failed to load metadata: %v", err)
	}

	return r, nil
}

// ResolveWebhookURL determines the webhook URL for a triggered alert.
// Returns empty string if no valid webhook is configured.
func (r *Router) ResolveWebhookURL(ticker, timeframe, exchange string, isRatio bool) string {
	if !r.config.EnableIndustryRouting {
		_, url := r.getChannelInfo(r.config.DefaultChannel, "")
		return url
	}

	normalizedTF := normalizeTimeframe(timeframe)

	// Ratio alerts go to Pairs channel
	if isRatio {
		_, url := r.getChannelInfo("Pairs", normalizedTF)
		if url != "" {
			return url
		}
	}

	// Check if ETF
	if meta, ok := r.metadata[ticker]; ok {
		if strings.EqualFold(meta.AssetType, "ETF") {
			_, url := r.getChannelInfo("ETFs", normalizedTF)
			if url != "" {
				return url
			}
		}
	}

	// Route by economy
	economy := r.GetEconomy(ticker)
	if economy != "" {
		mapping := r.getMappingForTimeframe(normalizedTF)
		if _, ok := mapping[economy]; ok {
			_, url := r.getChannelInfo(economy, normalizedTF)
			if url != "" {
				return url
			}
		}
	}

	// Fallback to default
	_, url := r.getChannelInfo(r.config.DefaultChannel, normalizedTF)
	return url
}

// GetEconomy returns the RBICS economy classification for a ticker.
func (r *Router) GetEconomy(ticker string) string {
	if meta, ok := r.metadata[ticker]; ok && meta.Economy != "" {
		return meta.Economy
	}
	// Try with -US suffix
	if !strings.Contains(ticker, "-") {
		if meta, ok := r.metadata[ticker+"-US"]; ok && meta.Economy != "" {
			return meta.Economy
		}
	}
	return ""
}

// GetISIN returns the ISIN for a ticker from cached metadata.
func (r *Router) GetISIN(ticker string) string {
	if meta, ok := r.metadata[ticker]; ok {
		return meta.ISIN
	}
	if !strings.Contains(ticker, "-") {
		if meta, ok := r.metadata[ticker+"-US"]; ok {
			return meta.ISIN
		}
	}
	return ""
}

// GetMetadata returns the cached stock metadata for a ticker.
func (r *Router) GetMetadata(ticker string) (stockMeta, bool) {
	m, ok := r.metadata[ticker]
	return m, ok
}

// MetadataMap returns the full metadata map (for alert formatting).
func (r *Router) MetadataMap() map[string]stockMeta {
	return r.metadata
}

func (r *Router) loadConfig(ctx context.Context) error {
	doc, err := r.queries.GetAppDocument(ctx, "discord_channels_config")
	if err != nil {
		return err
	}

	var cfg routerConfig
	if err := json.Unmarshal(doc.Payload, &cfg); err != nil {
		return err
	}

	// Inherit base mappings into daily/weekly/hourly if not set
	if cfg.ChannelMappingsDaily == nil {
		cfg.ChannelMappingsDaily = inheritMappings(cfg.ChannelMappings, "daily")
	}
	if cfg.ChannelMappingsWeekly == nil {
		cfg.ChannelMappingsWeekly = inheritMappings(cfg.ChannelMappings, "weekly")
	}
	if cfg.ChannelMappingsHourly == nil {
		cfg.ChannelMappingsHourly = inheritMappings(cfg.ChannelMappings, "hourly")
	}
	if cfg.DefaultChannel == "" {
		cfg.DefaultChannel = "General"
	}

	r.config = cfg
	return nil
}

func (r *Router) loadMetadata(ctx context.Context) error {
	rows, err := r.queries.ListStockMetadataForAlerts(ctx)
	if err != nil {
		return err
	}

	for _, row := range rows {
		r.metadata[row.Symbol] = stockMeta{
			Name:      textVal(row.Name),
			Economy:   textVal(row.RbicsEconomy),
			AssetType: textVal(row.AssetType),
			ISIN:      textVal(row.Isin),
			Exchange:  textVal(row.Exchange),
			Country:   textVal(row.Country),
		}
	}

	log.Printf("discord/routing: loaded %d stock metadata entries", len(r.metadata))
	return nil
}

func (r *Router) getMappingForTimeframe(normalizedTF string) map[string]channelMapping {
	switch normalizedTF {
	case "daily":
		if len(r.config.ChannelMappingsDaily) > 0 {
			return r.config.ChannelMappingsDaily
		}
	case "weekly":
		if len(r.config.ChannelMappingsWeekly) > 0 {
			return r.config.ChannelMappingsWeekly
		}
	case "hourly":
		if len(r.config.ChannelMappingsHourly) > 0 {
			return r.config.ChannelMappingsHourly
		}
	}
	return r.config.ChannelMappings
}

func (r *Router) getChannelInfo(channelName, normalizedTF string) (string, string) {
	mapping := r.getMappingForTimeframe(normalizedTF)

	if info, ok := mapping[channelName]; ok {
		if info.WebhookURL != "" && !strings.Contains(info.WebhookURL, "YOUR_") {
			return info.ChannelName, info.WebhookURL
		}
	}

	// Fallback to base mapping
	if info, ok := r.config.ChannelMappings[channelName]; ok {
		if info.WebhookURL != "" && !strings.Contains(info.WebhookURL, "YOUR_") {
			return info.ChannelName, info.WebhookURL
		}
	}

	// Ultimate fallback to default channel in base mapping
	if channelName != r.config.DefaultChannel {
		return r.getChannelInfo(r.config.DefaultChannel, normalizedTF)
	}

	return "#general-alerts", ""
}

func defaultConfig() routerConfig {
	return routerConfig{
		DefaultChannel:       "General",
		EnableIndustryRouting: false,
		ChannelMappings: map[string]channelMapping{
			"General": {
				ChannelName: "#general-alerts",
				Description: "General alerts and fallback",
			},
		},
	}
}

func inheritMappings(base map[string]channelMapping, suffix string) map[string]channelMapping {
	if len(base) == 0 {
		return nil
	}
	out := make(map[string]channelMapping, len(base))
	for key, val := range base {
		out[key] = channelMapping{
			WebhookURL:  val.WebhookURL,
			ChannelName: val.ChannelName + "-" + suffix,
			Description: val.Description,
		}
	}
	return out
}

func normalizeTimeframe(tf string) string {
	switch strings.ToLower(strings.TrimSpace(tf)) {
	case "1wk", "1w", "weekly", "week":
		return "weekly"
	case "1h", "1hr", "hourly", "hour":
		return "hourly"
	default:
		return "daily"
	}
}

func textVal(t pgtype.Text) string {
	if t.Valid {
		return t.String
	}
	return ""
}

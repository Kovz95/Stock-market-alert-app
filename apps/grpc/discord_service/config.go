package main

// DiscordChannelsConfig matches the JSON stored in app_documents for key "discord_channels_config".
type DiscordChannelsConfig struct {
	ChannelMappings        map[string]ChannelEntry `json:"channel_mappings"`
	ChannelMappingsDaily   map[string]ChannelEntry `json:"channel_mappings_daily"`
	ChannelMappingsHourly  map[string]ChannelEntry `json:"channel_mappings_hourly"`
	ChannelMappingsWeekly map[string]ChannelEntry `json:"channel_mappings_weekly"`
	DefaultChannel         string                  `json:"default_channel"`
	EnableIndustryRouting  bool                    `json:"enable_industry_routing"`
	LogRoutingDecisions    bool                    `json:"log_routing_decisions"`
}

type ChannelEntry struct {
	WebhookURL  string `json:"webhook_url"`
	ChannelName string `json:"channel_name"`
	Description string `json:"description"`
}

const discordConfigDocumentKey = "discord_channels_config"

// Special channel keys (not economy-based); same as Streamlit.
var specialChannelNames = map[string]bool{
	"ETFs": true, "Pairs": true, "General": true, "Futures": true, "Failed_Price_Updates": true,
}

func isPlaceholderWebhook(url string) bool {
	return url == "" || len(url) > 4 && url[:5] == "YOUR_"
}

package discord

import (
	"fmt"
	"strings"
	"time"
)

// Embed represents a Discord embed object.
type Embed struct {
	Title       string       `json:"title,omitempty"`
	Description string       `json:"description,omitempty"`
	Color       int          `json:"color,omitempty"`
	Fields      []EmbedField `json:"fields,omitempty"`
	Footer      *EmbedFooter `json:"footer,omitempty"`
	Timestamp   string       `json:"timestamp,omitempty"`
}

// EmbedField is a single field within a Discord embed.
type EmbedField struct {
	Name   string `json:"name"`
	Value  string `json:"value"`
	Inline bool   `json:"inline,omitempty"`
}

// EmbedFooter is the footer section of a Discord embed.
type EmbedFooter struct {
	Text string `json:"text,omitempty"`
}

const (
	ColorGreen = 0x00FF00 // Buy action
	ColorRed   = 0xFF0000 // Sell action
)

// AlertInfo holds the data needed to format an alert as a Discord embed.
type AlertInfo struct {
	Ticker     string
	StockName  string
	Action     string   // "Buy", "Sell", "on"
	Timeframe  string   // "1d", "1wk", "1h"
	Exchange   string
	Economy    string
	ISIN       string
	Conditions []string // condition strings
}

// FormatAlertEmbed creates a Discord embed for a triggered alert.
func FormatAlertEmbed(info AlertInfo) Embed {
	// Determine color
	color := ColorGreen
	if strings.EqualFold(info.Action, "Sell") {
		color = ColorRed
	}

	// Build conditions text
	var condLines []string
	for _, c := range info.Conditions {
		if c != "" {
			condLines = append(condLines, fmt.Sprintf("• %s", c))
		}
	}
	condText := strings.Join(condLines, "\n")
	if condText == "" {
		condText = "—"
	}
	// Discord embed description limit is 4096 chars
	if len(condText) > 4096 {
		condText = condText[:4093] + "..."
	}

	now := time.Now().UTC()
	et := toET(now)

	envTag := EnvironmentTag()
	title := fmt.Sprintf("%s — %s", info.Ticker, info.StockName)
	if envTag != "" {
		title = strings.TrimSpace(envTag) + " " + title
	}
	// Discord embed title limit is 256 chars
	if len(title) > 256 {
		title = title[:256]
	}

	embed := Embed{
		Title:       title,
		Description: condText,
		Color:       color,
		Fields: []EmbedField{
			{Name: "Action", Value: info.Action, Inline: true},
			{Name: "Timeframe", Value: info.Timeframe, Inline: true},
		},
		Footer: &EmbedFooter{
			Text: fmt.Sprintf("Triggered at %s", et.Format("2006-01-02 03:04:05 PM ET")),
		},
		Timestamp: now.Format(time.RFC3339),
	}

	if info.Exchange != "" {
		embed.Fields = append(embed.Fields, EmbedField{
			Name: "Exchange", Value: info.Exchange, Inline: true,
		})
	}
	if info.Economy != "" {
		embed.Fields = append(embed.Fields, EmbedField{
			Name: "Economy", Value: info.Economy, Inline: true,
		})
	}
	if info.ISIN != "" {
		embed.Fields = append(embed.Fields, EmbedField{
			Name: "ISIN", Value: info.ISIN, Inline: true,
		})
	}

	return embed
}

// FormatDuration converts seconds to a human-readable duration string.
// Examples: "1h 23m 45s", "5m 30s", "45s".
func FormatDuration(seconds float64) string {
	if seconds <= 0 {
		return "0s"
	}
	d := time.Duration(seconds * float64(time.Second))
	h := int(d.Hours())
	m := int(d.Minutes()) % 60
	s := int(d.Seconds()) % 60

	if h > 0 {
		return fmt.Sprintf("%dh %dm %ds", h, m, s)
	}
	if m > 0 {
		return fmt.Sprintf("%dm %ds", m, s)
	}
	return fmt.Sprintf("%ds", s)
}

// FormatESTTime formats a UTC time as Eastern Time.
func FormatESTTime(t time.Time) string {
	return toET(t).Format("2006-01-02 03:04:05 PM")
}

// toET converts a time to America/New_York timezone.
func toET(t time.Time) time.Time {
	loc, err := time.LoadLocation("America/New_York")
	if err != nil {
		return t
	}
	return t.In(loc)
}

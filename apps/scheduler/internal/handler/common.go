package handler

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"time"

	"github.com/hibiken/asynq"
	"github.com/jackc/pgx/v5/pgtype"

	"stockalert/alert"
	"stockalert/calendar"
	"stockalert/discord"
	db "stockalert/database/generated"

	"stockalert/apps/scheduler/internal/price"
	"stockalert/apps/scheduler/internal/status"
)

// ShadowRecord is one triggered alert written for shadow comparison.
type ShadowRecord struct {
	AlertID string `json:"alert_id"`
	Ticker  string `json:"ticker"`
}

// ShadowOutput is the JSON written when SCHEDULER_SHADOW_MODE is set.
type ShadowOutput struct {
	Exchange   string         `json:"exchange"`
	Timeframe  string         `json:"timeframe"`
	UTC        string         `json:"utc_timestamp"`
	Triggered  []ShadowRecord `json:"triggered"`
	Total      int            `json:"total"`
	TriggeredN int            `json:"triggered_count"`
	Errors     int            `json:"errors"`
	Skipped    int            `json:"skipped"`
	NoData     int            `json:"no_data"`
}

// Common holds shared dependencies for all job handlers.
type Common struct {
	Queries    *db.Queries
	Checker    *alert.Checker
	Router     *discord.Router
	Accum      *discord.Accumulator
	Notifier   *discord.Notifier
	Updater    *price.Updater
	Status     *status.Manager
	JobTimeout time.Duration

	// Discord webhooks for status (start/complete/error). Empty = no status notifications.
	WebhookDaily  string
	WebhookWeekly string
	WebhookHourly string

	// ShadowDir: when non-empty, write trigger results to this dir for Go vs Python comparison.
	ShadowDir string
}

// statusNotifierFor returns a StatusNotifier for the given job type, or nil if webhook is not set.
func (c *Common) statusNotifierFor(jobLabel, timeframe string) *discord.StatusNotifier {
	var url string
	switch jobLabel {
	case "daily":
		url = c.WebhookDaily
	case "weekly":
		url = c.WebhookWeekly
	case "hourly":
		url = c.WebhookHourly
	default:
		return nil
	}
	if url == "" {
		return nil
	}
	return discord.NewStatusNotifier(c.Notifier, url, jobLabel, timeframe)
}

// Execute runs the full job lifecycle for a single exchange/timeframe:
// check should-run (hourly → calendar.IsExchangeOpen), notify start, update prices,
// check alerts and accumulate Discord embeds, flush embeds, notify complete/error.
func (c *Common) Execute(ctx context.Context, exchange, timeframe string, statusNotifier *discord.StatusNotifier) (priceStats *discord.PriceStats, alertStats *discord.AlertStats, err error) {
	runTime := time.Now().UTC()
	start := runTime

	// Hourly: skip if exchange is closed
	if timeframe == "hourly" {
		if !calendar.IsExchangeOpen(exchange, runTime) {
			log.Printf("[%s/%s] exchange closed, skipping", exchange, timeframe)
			if statusNotifier != nil {
				statusNotifier.NotifySkipped(runTime, exchange+" is closed")
			}
			return nil, nil, fmt.Errorf("%s is closed: %w", exchange, asynq.SkipRetry)
		}
	}

	if statusNotifier != nil {
		statusNotifier.NotifyStart(runTime, exchange)
	}

	if err := c.Status.UpdateRunning(ctx, exchange, timeframe); err != nil {
		log.Printf("status update running: %v", err)
	}

	// 1. Update prices
	priceStats, err = c.Updater.UpdateForExchange(ctx, exchange, timeframe)
	if err != nil {
		log.Printf("[%s/%s] price update error: %v", exchange, timeframe, err)
		c.reportError(ctx, runTime, exchange, timeframe, err, statusNotifier)
		return nil, nil, err
	}

	// 2. Load alerts for this exchange + timeframe
	alerts, err := c.Queries.ListAlertsByExchangeAndTimeframe(ctx, db.ListAlertsByExchangeAndTimeframeParams{
		Exchanges: []string{exchange},
		Timeframe: pgtype.Text{String: timeframe, Valid: true},
	})
	if err != nil {
		log.Printf("[%s/%s] list alerts: %v", exchange, timeframe, err)
		c.reportError(ctx, runTime, exchange, timeframe, err, statusNotifier)
		return nil, nil, err
	}

	alertStats = &discord.AlertStats{Total: len(alerts)}
	if len(alerts) == 0 {
		c.reportSuccess(ctx, runTime, start, exchange, timeframe, priceStats, alertStats, statusNotifier)
		return priceStats, alertStats, nil
	}

	// Pre-warm cache and evaluate
	since := sinceDateForTimeframe(timeframe)
	if err := c.Checker.PreWarmCache(ctx, alerts, timeframe, since); err != nil {
		log.Printf("[%s/%s] prewarm cache: %v", exchange, timeframe, err)
	}

	var shadowTriggered []ShadowRecord
	onTriggered := func(a db.Alert, result alert.CheckResult) {
		alertStats.Triggered++
		ticker := alertPrimaryTicker(&a)
		if c.ShadowDir != "" {
			shadowTriggered = append(shadowTriggered, ShadowRecord{AlertID: result.AlertID, Ticker: ticker})
		}
		webhookURL := c.Router.ResolveWebhookURL(
			ticker,
			timeframe,
			exchange,
			a.IsRatio.Bool && a.IsRatio.Valid,
		)
		conditions := []string{}
		if result.Triggered {
			conditions = []string{"Conditions met"}
		}
		embed := discord.FormatAlertEmbed(discord.AlertInfo{
			Ticker:     ticker,
			StockName:  textStr(a.StockName),
			Action:     textStr(a.Action),
			Timeframe:  timeframe,
			Exchange:   textStr(a.Exchange),
			Economy:    c.Router.GetEconomy(ticker),
			ISIN:       c.Router.GetISIN(ticker),
			Conditions: conditions,
		})
		c.Accum.Add(webhookURL, embed)
	}
	stats, err := c.Checker.CheckAlerts(ctx, alerts, timeframe, onTriggered)
	if err != nil {
		c.reportError(ctx, runTime, exchange, timeframe, err, statusNotifier)
		return nil, nil, err
	}
	alertStats.Triggered = stats.Triggered
	alertStats.NotTriggered = stats.Success - stats.Triggered
	alertStats.Skipped = stats.Skipped
	alertStats.NoData = stats.NoData
	alertStats.Errors = stats.Errors

	if c.ShadowDir != "" {
		writeShadowOutput(c.ShadowDir, exchange, timeframe, runTime, shadowTriggered, stats)
	}

	c.Accum.FlushAll()
	c.reportSuccess(ctx, runTime, start, exchange, timeframe, priceStats, alertStats, statusNotifier)
	return priceStats, alertStats, nil
}

func (c *Common) reportSuccess(ctx context.Context, runTime, start time.Time, exchange, timeframe string, priceStats *discord.PriceStats, alertStats *discord.AlertStats, statusNotifier *discord.StatusNotifier) {
	durationSec := time.Since(start).Seconds()
	c.Status.UpdateSuccess(ctx, exchange, timeframe, priceStats, alertStats)
	if statusNotifier != nil {
		statusNotifier.NotifyComplete(runTime, durationSec, exchange, priceStats, alertStats)
	}
}

func (c *Common) reportError(ctx context.Context, runTime time.Time, exchange, timeframe string, err error, statusNotifier *discord.StatusNotifier) {
	c.Status.UpdateError(ctx, exchange, timeframe, err.Error())
	if statusNotifier != nil {
		statusNotifier.NotifyError(runTime, err.Error())
	}
}

func sinceDateForTimeframe(timeframe string) time.Time {
	now := time.Now().UTC()
	switch timeframe {
	case "hourly":
		return now.AddDate(0, 0, -7)
	case "weekly":
		return now.AddDate(0, 0, -365)
	default:
		return now.AddDate(0, 0, -365)
	}
}

func alertPrimaryTicker(a *db.Alert) string {
	if a.IsRatio.Bool && a.IsRatio.Valid {
		if a.Ticker1.Valid {
			return a.Ticker1.String
		}
		return ""
	}
	if a.Ticker.Valid {
		return a.Ticker.String
	}
	return ""
}

func textStr(t pgtype.Text) string {
	if t.Valid {
		return t.String
	}
	return ""
}

func writeShadowOutput(dir, exchange, timeframe string, runTime time.Time, triggered []ShadowRecord, stats alert.CheckStats) {
	_ = os.MkdirAll(dir, 0755)
	ts := runTime.UTC().Format("20060102_150405")
	name := fmt.Sprintf("go_%s_%s_%s.json", exchange, timeframe, ts)
	path := filepath.Join(dir, name)
	out := ShadowOutput{
		Exchange:   exchange,
		Timeframe:  timeframe,
		UTC:        runTime.UTC().Format(time.RFC3339),
		Triggered:  triggered,
		Total:      stats.Total,
		TriggeredN: stats.Triggered,
		Errors:     stats.Errors,
		Skipped:    stats.Skipped,
		NoData:     stats.NoData,
	}
	b, err := json.MarshalIndent(out, "", "  ")
	if err != nil {
		log.Printf("shadow write marshal: %v", err)
		return
	}
	if err := os.WriteFile(path, b, 0644); err != nil {
		log.Printf("shadow write %s: %v", path, err)
		return
	}
	log.Printf("shadow wrote %s (%d triggered)", path, len(triggered))
}

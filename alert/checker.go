package alert

import (
	"context"
	"fmt"
	"log"
	"strings"
	"sync"
	"time"

	"github.com/jackc/pgx/v5/pgtype"

	db "stockalert/database/generated"
	"stockalert/indicator"
)

// CheckResult holds the outcome of checking a single alert.
type CheckResult struct {
	AlertID    string
	Ticker     string
	Triggered  bool
	Skipped    bool
	SkipReason string
	Error      error
	ExecTimeMs int
}

// CheckStats aggregates results across all alerts in a run.
type CheckStats struct {
	Total     int `json:"total"`
	Triggered int `json:"triggered"`
	Errors    int `json:"errors"`
	Skipped   int `json:"skipped"`
	NoData    int `json:"no_data"`
	Success   int `json:"success"`
}

// DeferredAudit collects the data needed for a single audit row,
// written in bulk after all alerts finish.
type DeferredAudit struct {
	Timestamp           time.Time
	AlertID             string
	Ticker              string
	StockName           string
	Exchange            string
	Timeframe           string
	Action              string
	EvaluationType      string
	PriceDataPulled     bool
	PriceDataSource     string
	ConditionsEvaluated bool
	AlertTriggered      bool
	TriggerReason       string
	ExecutionTimeMs     int
	CacheHit            bool
	ErrorMessage        string
}

// DeferredTrigger records an alert that was triggered and needs its
// last_triggered timestamp updated in the database.
type DeferredTrigger struct {
	AlertID   pgtype.UUID
	Timestamp time.Time
}

// PriceCache is a thread-safe in-memory cache of OHLCV data keyed by
// "TICKER_TIMEFRAME" (e.g. "AAPL_daily").
type PriceCache struct {
	mu   sync.RWMutex
	data map[string]*indicator.OHLCV
}

// NewPriceCache creates an empty price cache.
func NewPriceCache() *PriceCache {
	return &PriceCache{data: make(map[string]*indicator.OHLCV)}
}

// Get returns cached OHLCV data and true, or nil and false.
func (c *PriceCache) Get(ticker, timeframe string) (*indicator.OHLCV, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	d, ok := c.data[cacheKey(ticker, timeframe)]
	return d, ok
}

// Set stores OHLCV data in the cache.
func (c *PriceCache) Set(ticker, timeframe string, data *indicator.OHLCV) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.data[cacheKey(ticker, timeframe)] = data
}

// Len returns the number of entries in the cache.
func (c *PriceCache) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.data)
}

func cacheKey(ticker, timeframe string) string {
	return ticker + "_" + timeframe
}

// Checker orchestrates checking alerts: load alerts from DB, pre-warm
// the price cache, evaluate each alert, and batch-flush audit records
// and last_triggered updates.
type Checker struct {
	queries   *db.Queries
	evaluator *Evaluator
	cache     *PriceCache

	// Deferred collections — flushed after all alerts complete.
	mu              sync.Mutex
	deferredAudits  []DeferredAudit
	deferredTrigger []DeferredTrigger
}

// NewChecker creates a Checker with the given DB queries and indicator registry.
func NewChecker(queries *db.Queries, registry *indicator.Registry) *Checker {
	return &Checker{
		queries:   queries,
		evaluator: NewEvaluator(registry),
		cache:     NewPriceCache(),
	}
}

// CheckAlerts evaluates a slice of alerts and returns aggregate stats.
// It pre-warms the price cache, then checks each alert sequentially
// (or concurrently if maxWorkers > 1), then flushes deferred DB writes.
//
// The onTriggered callback is invoked for each triggered alert so the
// caller can route it to Discord. It receives the alert and the formatted
// trigger info. If nil, triggered alerts are only recorded in the audit.
func (ch *Checker) CheckAlerts(
	ctx context.Context,
	alerts []db.Alert,
	timeframe string,
	onTriggered func(alert db.Alert, result CheckResult),
) (CheckStats, error) {
	if len(alerts) == 0 {
		return CheckStats{}, nil
	}

	// Reset deferred collections
	ch.mu.Lock()
	ch.deferredAudits = ch.deferredAudits[:0]
	ch.deferredTrigger = ch.deferredTrigger[:0]
	ch.mu.Unlock()

	normalizedTF := NormalizeTimeframe(timeframe)

	var stats CheckStats
	stats.Total = len(alerts)

	for i := range alerts {
		result := ch.checkSingleAlert(ctx, &alerts[i], normalizedTF)
		ch.aggregateResult(&stats, result)
		if result.Triggered && onTriggered != nil {
			onTriggered(alerts[i], result)
		}
	}

	// Flush deferred writes
	if err := ch.flushAudits(ctx); err != nil {
		log.Printf("alert/checker: failed to flush audits: %v", err)
	}
	if err := ch.flushTriggers(ctx); err != nil {
		log.Printf("alert/checker: failed to flush triggers: %v", err)
	}

	return stats, nil
}

// PreWarmCache loads price data from the database into the in-memory cache
// for all unique tickers in the given alerts. This turns N individual
// queries into a single batch query per timeframe.
func (ch *Checker) PreWarmCache(ctx context.Context, alerts []db.Alert, timeframe string, sinceDate time.Time) error {
	normalizedTF := NormalizeTimeframe(timeframe)

	// Collect unique tickers
	seen := make(map[string]bool)
	var tickers []string
	for _, a := range alerts {
		for _, t := range alertTickers(a) {
			if t != "" && !seen[t] {
				seen[t] = true
				tickers = append(tickers, t)
			}
		}
	}

	if len(tickers) == 0 {
		return nil
	}

	switch normalizedTF {
	case "daily":
		rows, err := ch.queries.GetDailyPricesBatch(ctx, db.GetDailyPricesBatchParams{
			Tickers:   tickers,
			SinceDate: pgtype.Date{Time: sinceDate, Valid: true},
		})
		if err != nil {
			return fmt.Errorf("batch load daily prices: %w", err)
		}
		ch.cacheDailyRows(rows, normalizedTF)

	case "hourly":
		rows, err := ch.queries.GetHourlyPricesBatch(ctx, db.GetHourlyPricesBatchParams{
			Tickers: tickers,
			SinceTs: pgtype.Timestamptz{Time: sinceDate, Valid: true},
		})
		if err != nil {
			return fmt.Errorf("batch load hourly prices: %w", err)
		}
		ch.cacheHourlyRows(rows, normalizedTF)

	case "weekly":
		rows, err := ch.queries.GetWeeklyPricesBatch(ctx, db.GetWeeklyPricesBatchParams{
			Tickers:   tickers,
			SinceDate: pgtype.Date{Time: sinceDate, Valid: true},
		})
		if err != nil {
			return fmt.Errorf("batch load weekly prices: %w", err)
		}
		ch.cacheWeeklyRows(rows, normalizedTF)
	}

	log.Printf("alert/checker: pre-warmed cache with %d entries for %s", ch.cache.Len(), normalizedTF)
	return nil
}

// Cache returns the underlying PriceCache for external population (e.g., FMP fallback).
func (ch *Checker) Cache() *PriceCache {
	return ch.cache
}

// checkSingleAlert evaluates one alert against cached price data.
func (ch *Checker) checkSingleAlert(ctx context.Context, alert *db.Alert, normalizedTF string) CheckResult {
	start := time.Now()

	ticker := alertPrimaryTicker(alert)
	result := CheckResult{
		AlertID: uuidStr(alert.AlertID),
		Ticker:  ticker,
	}

	audit := DeferredAudit{
		Timestamp:      time.Now().UTC(),
		AlertID:        result.AlertID,
		Ticker:         ticker,
		StockName:      textStr(alert.StockName),
		Exchange:       textStr(alert.Exchange),
		Timeframe:      textStr(alert.Timeframe),
		Action:         textStr(alert.Action),
		EvaluationType: normalizedTF,
	}

	defer func() {
		audit.ExecutionTimeMs = int(time.Since(start).Milliseconds())
		result.ExecTimeMs = audit.ExecutionTimeMs
		ch.mu.Lock()
		ch.deferredAudits = append(ch.deferredAudits, audit)
		ch.mu.Unlock()
	}()

	// Check if disabled
	if textStr(alert.Action) == "off" {
		result.Skipped = true
		result.SkipReason = "disabled"
		return result
	}

	// Check if already triggered today
	if shouldSkipAlreadyTriggered(alert.LastTriggered) {
		result.Skipped = true
		result.SkipReason = "already_triggered_today"
		return result
	}

	// Get price data from cache
	data, ok := ch.cache.Get(ticker, normalizedTF)
	if !ok || data == nil || data.Len() == 0 {
		audit.ErrorMessage = "no price data available"
		result.Error = fmt.Errorf("no price data for %s (%s)", ticker, normalizedTF)
		return result
	}

	audit.PriceDataPulled = true
	audit.PriceDataSource = "cache"
	audit.CacheHit = true

	// Evaluate conditions
	combinationLogic := textStr(alert.CombinationLogic)
	triggered, err := ch.evaluator.EvaluateAlert(data, alert.Conditions, combinationLogic, ticker)
	audit.ConditionsEvaluated = true

	if err != nil {
		audit.ErrorMessage = err.Error()
		result.Error = err
		return result
	}

	audit.AlertTriggered = triggered
	result.Triggered = triggered

	if triggered {
		audit.TriggerReason = "conditions_met"
		now := time.Now().UTC()
		ch.mu.Lock()
		ch.deferredTrigger = append(ch.deferredTrigger, DeferredTrigger{
			AlertID:   alert.AlertID,
			Timestamp: now,
		})
		ch.mu.Unlock()
	}

	return result
}

// aggregateResult updates stats from a single check result.
func (ch *Checker) aggregateResult(stats *CheckStats, r CheckResult) {
	if r.Skipped {
		stats.Skipped++
		return
	}
	if r.Error != nil {
		if strings.Contains(r.Error.Error(), "no price data") {
			stats.NoData++
		} else {
			stats.Errors++
		}
		return
	}
	if r.Triggered {
		stats.Triggered++
	}
	stats.Success++
}

// flushAudits bulk-inserts all deferred audit records using the COPY protocol.
func (ch *Checker) flushAudits(ctx context.Context) error {
	ch.mu.Lock()
	audits := make([]DeferredAudit, len(ch.deferredAudits))
	copy(audits, ch.deferredAudits)
	ch.mu.Unlock()

	if len(audits) == 0 {
		return nil
	}

	params := make([]db.CopyAlertAuditsParams, 0, len(audits))
	for _, a := range audits {
		params = append(params, db.CopyAlertAuditsParams{
			Timestamp:           pgtype.Timestamptz{Time: a.Timestamp, Valid: true},
			AlertID:             a.AlertID,
			Ticker:              a.Ticker,
			StockName:           pgtype.Text{String: a.StockName, Valid: a.StockName != ""},
			Exchange:            pgtype.Text{String: a.Exchange, Valid: a.Exchange != ""},
			Timeframe:           pgtype.Text{String: a.Timeframe, Valid: a.Timeframe != ""},
			Action:              pgtype.Text{String: a.Action, Valid: a.Action != ""},
			EvaluationType:      a.EvaluationType,
			PriceDataPulled:     pgtype.Bool{Bool: a.PriceDataPulled, Valid: true},
			PriceDataSource:     pgtype.Text{String: a.PriceDataSource, Valid: a.PriceDataSource != ""},
			ConditionsEvaluated: pgtype.Bool{Bool: a.ConditionsEvaluated, Valid: true},
			AlertTriggered:      pgtype.Bool{Bool: a.AlertTriggered, Valid: true},
			TriggerReason:       pgtype.Text{String: a.TriggerReason, Valid: a.TriggerReason != ""},
			ExecutionTimeMs:     pgtype.Int4{Int32: int32(a.ExecutionTimeMs), Valid: true},
			CacheHit:            pgtype.Bool{Bool: a.CacheHit, Valid: true},
			ErrorMessage:        pgtype.Text{String: a.ErrorMessage, Valid: a.ErrorMessage != ""},
		})
	}

	n, err := ch.queries.CopyAlertAudits(ctx, params)
	if err != nil {
		return fmt.Errorf("copy alert audits (%d rows): %w", len(params), err)
	}
	log.Printf("alert/checker: flushed %d audit records", n)
	return nil
}

// flushTriggers updates last_triggered for all triggered alerts.
func (ch *Checker) flushTriggers(ctx context.Context) error {
	ch.mu.Lock()
	triggers := make([]DeferredTrigger, len(ch.deferredTrigger))
	copy(triggers, ch.deferredTrigger)
	ch.mu.Unlock()

	if len(triggers) == 0 {
		return nil
	}

	for _, t := range triggers {
		err := ch.queries.BulkUpdateLastTriggered(ctx, db.BulkUpdateLastTriggeredParams{
			AlertID:       t.AlertID,
			LastTriggered: pgtype.Timestamptz{Time: t.Timestamp, Valid: true},
		})
		if err != nil {
			log.Printf("alert/checker: failed to update last_triggered for %s: %v", uuidStr(t.AlertID), err)
		}
	}

	log.Printf("alert/checker: updated last_triggered for %d alerts", len(triggers))
	return nil
}

// --- Cache population helpers ---

func (ch *Checker) cacheDailyRows(rows []db.GetDailyPricesBatchRow, tf string) {
	grouped := make(map[string][]db.GetDailyPricesBatchRow)
	for _, r := range rows {
		grouped[r.Ticker] = append(grouped[r.Ticker], r)
	}
	for ticker, rowGroup := range grouped {
		ohlcv := &indicator.OHLCV{
			Open:   make([]float64, len(rowGroup)),
			High:   make([]float64, len(rowGroup)),
			Low:    make([]float64, len(rowGroup)),
			Close:  make([]float64, len(rowGroup)),
			Volume: make([]float64, len(rowGroup)),
		}
		for i, r := range rowGroup {
			ohlcv.Open[i] = r.Open.Float64
			ohlcv.High[i] = r.High.Float64
			ohlcv.Low[i] = r.Low.Float64
			ohlcv.Close[i] = r.Close
			ohlcv.Volume[i] = float64(r.Volume.Int64)
		}
		ch.cache.Set(ticker, tf, ohlcv)
	}
}

func (ch *Checker) cacheHourlyRows(rows []db.GetHourlyPricesBatchRow, tf string) {
	grouped := make(map[string][]db.GetHourlyPricesBatchRow)
	for _, r := range rows {
		grouped[r.Ticker] = append(grouped[r.Ticker], r)
	}
	for ticker, rowGroup := range grouped {
		ohlcv := &indicator.OHLCV{
			Open:   make([]float64, len(rowGroup)),
			High:   make([]float64, len(rowGroup)),
			Low:    make([]float64, len(rowGroup)),
			Close:  make([]float64, len(rowGroup)),
			Volume: make([]float64, len(rowGroup)),
		}
		for i, r := range rowGroup {
			ohlcv.Open[i] = r.Open.Float64
			ohlcv.High[i] = r.High.Float64
			ohlcv.Low[i] = r.Low.Float64
			ohlcv.Close[i] = r.Close
			ohlcv.Volume[i] = float64(r.Volume.Int64)
		}
		ch.cache.Set(ticker, tf, ohlcv)
	}
}

func (ch *Checker) cacheWeeklyRows(rows []db.GetWeeklyPricesBatchRow, tf string) {
	grouped := make(map[string][]db.GetWeeklyPricesBatchRow)
	for _, r := range rows {
		grouped[r.Ticker] = append(grouped[r.Ticker], r)
	}
	for ticker, rowGroup := range grouped {
		ohlcv := &indicator.OHLCV{
			Open:   make([]float64, len(rowGroup)),
			High:   make([]float64, len(rowGroup)),
			Low:    make([]float64, len(rowGroup)),
			Close:  make([]float64, len(rowGroup)),
			Volume: make([]float64, len(rowGroup)),
		}
		for i, r := range rowGroup {
			ohlcv.Open[i] = r.Open.Float64
			ohlcv.High[i] = r.High.Float64
			ohlcv.Low[i] = r.Low.Float64
			ohlcv.Close[i] = r.Close
			ohlcv.Volume[i] = float64(r.Volume.Int64)
		}
		ch.cache.Set(ticker, tf, ohlcv)
	}
}

// --- Alert field helpers ---

// alertPrimaryTicker returns the main ticker for an alert.
// For ratio alerts it returns ticker1.
func alertPrimaryTicker(a *db.Alert) string {
	if a.IsRatio.Bool && a.IsRatio.Valid {
		return textStr(a.Ticker1)
	}
	return textStr(a.Ticker)
}

// alertTickers returns all tickers referenced by an alert.
func alertTickers(a db.Alert) []string {
	if a.IsRatio.Bool && a.IsRatio.Valid {
		var out []string
		if t := textStr(a.Ticker1); t != "" {
			out = append(out, t)
		}
		if t := textStr(a.Ticker2); t != "" {
			out = append(out, t)
		}
		return out
	}
	if t := textStr(a.Ticker); t != "" {
		return []string{t}
	}
	return nil
}

// shouldSkipAlreadyTriggered returns true if the alert was already triggered
// today (UTC date comparison).
func shouldSkipAlreadyTriggered(lastTriggered pgtype.Timestamptz) bool {
	if !lastTriggered.Valid {
		return false
	}
	today := time.Now().UTC().Truncate(24 * time.Hour)
	triggered := lastTriggered.Time.UTC().Truncate(24 * time.Hour)
	return triggered.Equal(today)
}

// textStr extracts the string from a pgtype.Text, returning "" if not valid.
func textStr(t pgtype.Text) string {
	if t.Valid {
		return t.String
	}
	return ""
}

// uuidStr formats a pgtype.UUID as a hex string.
func uuidStr(u pgtype.UUID) string {
	if !u.Valid {
		return ""
	}
	b := u.Bytes
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}


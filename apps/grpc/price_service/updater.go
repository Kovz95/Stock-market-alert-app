package main

import (
	"context"
	"log/slog"
	"math"
	"sync"
	"sync/atomic"
	"time"

	"github.com/jackc/pgx/v5/pgtype"
	"github.com/jackc/pgx/v5/pgxpool"

	db "stockalert/database/generated"
)

const (
	backfillLimit  = 750 // rows to fetch when ticker has no existing data
	bufferDays     = 5   // extra days to re-fetch for corrections/adjustments
	defaultWorkers = 10  // concurrent workers for ticker processing
)

// priceUpdater runs FMP fetches and DB copy/upsert for a given set of tickers.
type priceUpdater struct {
	fmp    fmpFetcher
	pool   *pgxpool.Pool
	logger *slog.Logger
}

func newPriceUpdater(fmp fmpFetcher, pool *pgxpool.Pool, logger *slog.Logger) *priceUpdater {
	if logger == nil {
		logger = slog.Default()
	}
	return &priceUpdater{fmp: fmp, pool: pool, logger: logger.With("component", "price_updater")}
}

// tickerResult carries the outcome of processing a single ticker.
type tickerResult struct {
	ticker string
	ok     bool
}

// runWorkers fans out ticker processing across defaultWorkers goroutines.
// Each worker acquires its own DB connection from the pool.
func (u *priceUpdater) runWorkers(ctx context.Context, tickers []string, process func(ctx context.Context, q *db.Queries, ticker string) bool) (updated, failed int) {
	tickerCh := make(chan string, len(tickers))
	for _, t := range tickers {
		tickerCh <- t
	}
	close(tickerCh)

	resultCh := make(chan tickerResult, len(tickers))
	var wg sync.WaitGroup

	workers := defaultWorkers
	if len(tickers) < workers {
		workers = len(tickers)
	}

	var idx atomic.Int64
	total := len(tickers)

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			conn, err := u.pool.Acquire(ctx)
			if err != nil {
				u.logger.Error("worker failed to acquire DB connection", "error", err)
				// Drain remaining tickers as failures.
				for ticker := range tickerCh {
					resultCh <- tickerResult{ticker: ticker, ok: false}
				}
				return
			}
			defer conn.Release()
			q := db.New(conn)

			for ticker := range tickerCh {
				i := idx.Add(1)
				if i%int64(progressLogInterval) == 0 || i == 1 {
					u.logger.Info("worker progress", "ticker_index", i, "total_tickers", total, "current", ticker)
				}
				ok := process(ctx, q, ticker)
				resultCh <- tickerResult{ticker: ticker, ok: ok}
			}
		}()
	}

	// Close results channel once all workers finish.
	go func() {
		wg.Wait()
		close(resultCh)
	}()

	for r := range resultCh {
		if r.ok {
			updated++
		} else {
			failed++
		}
	}
	return updated, failed
}

// updateForTickers updates prices for the given tickers and timeframe using the provided queries.
// Returns updated count, failed count, and any error.
func (u *priceUpdater) updateForTickers(ctx context.Context, q *db.Queries, exchange, timeframe string, tickers []string) (updated, failed int, err error) {
	if len(tickers) == 0 {
		u.logger.Info("updateForTickers skipped: no tickers", "exchange", exchange, "timeframe", timeframe)
		return 0, 0, nil
	}
	start := time.Now()
	u.logger.Info("updateForTickers started", "exchange", exchange, "timeframe", timeframe, "ticker_count", len(tickers))
	defer func() {
		u.logger.Info("updateForTickers finished", "exchange", exchange, "timeframe", timeframe, "updated", updated, "failed", failed, "duration_ms", time.Since(start).Milliseconds())
	}()
	switch timeframe {
	case "daily":
		updated, failed = u.updateDaily(ctx, q, tickers)
		return updated, failed, nil
	case "weekly":
		updated, failed = u.updateWeekly(ctx, q, tickers)
		return updated, failed, nil
	case "hourly":
		updated, failed = u.updateHourly(ctx, q, exchange, tickers)
		return updated, failed, nil
	default:
		return 0, 0, nil
	}
}

const progressLogInterval = 50 // log progress every N tickers

func (u *priceUpdater) updateDaily(ctx context.Context, q *db.Queries, tickers []string) (updated, failed int) {
	// Pre-fetch latest daily dates for all tickers to compute smart limits.
	lastDates := make(map[string]time.Time)
	if rows, err := q.LastDailyPerTicker(ctx); err == nil {
		for _, r := range rows {
			if r.LastDate.Valid {
				lastDates[r.Ticker] = r.LastDate.Time
			}
		}
		u.logger.Info("daily smart limits loaded", "tickers_with_data", len(lastDates))
	} else {
		u.logger.Warn("LastDailyPerTicker failed, falling back to full backfill", "error", err)
	}

	now := time.Now()
	return u.runWorkers(ctx, tickers, func(ctx context.Context, wq *db.Queries, ticker string) bool {
		limit := backfillLimit
		if lastDate, ok := lastDates[ticker]; ok {
			daysMissing := int(math.Ceil(now.Sub(lastDate).Hours() / 24))
			limit = daysMissing + bufferDays
			if limit < bufferDays {
				limit = bufferDays
			}
			if limit > backfillLimit {
				limit = backfillLimit
			}
		}

		rows, err := u.fmp.FetchDaily(ticker, limit)
		if err != nil {
			u.logger.Warn("FMP daily fetch error", "ticker", ticker, "error", err)
			return false
		}
		for _, r := range rows {
			date, err := time.Parse("2006-01-02", r.Date)
			if err != nil {
				continue
			}
			arg := db.UpsertDailyPriceParams{
				Ticker: ticker,
				Date:   pgtype.Date{Time: date, Valid: true},
				Open:   pgtype.Float8{Float64: r.Open, Valid: true},
				High:   pgtype.Float8{Float64: r.High, Valid: true},
				Low:    pgtype.Float8{Float64: r.Low, Valid: true},
				Close:  r.Close,
				Volume: pgtype.Int8{Int64: r.Volume, Valid: true},
			}
			if err := wq.UpsertDailyPrice(ctx, arg); err != nil {
				u.logger.Error("UpsertDailyPrice failed", "ticker", ticker, "date", r.Date, "error", err)
				return false
			}
		}
		return true
	})
}

func (u *priceUpdater) updateWeekly(ctx context.Context, q *db.Queries, tickers []string) (updated, failed int) {
	// Pre-fetch latest weekly dates for all tickers to compute smart limits.
	lastDates := make(map[string]time.Time)
	if rows, err := q.LastWeeklyPerTicker(ctx); err == nil {
		for _, r := range rows {
			if r.LastDate.Valid {
				lastDates[r.Ticker] = r.LastDate.Time
			}
		}
		u.logger.Info("weekly smart limits loaded", "tickers_with_data", len(lastDates))
	} else {
		u.logger.Warn("LastWeeklyPerTicker failed, falling back to full backfill", "error", err)
	}

	now := time.Now()
	return u.runWorkers(ctx, tickers, func(ctx context.Context, wq *db.Queries, ticker string) bool {
		limit := backfillLimit
		if lastDate, ok := lastDates[ticker]; ok {
			daysMissing := int(math.Ceil(now.Sub(lastDate).Hours() / 24))
			limit = daysMissing + bufferDays
			if limit < bufferDays {
				limit = bufferDays
			}
			if limit > backfillLimit {
				limit = backfillLimit
			}
		}

		rows, err := u.fmp.FetchDaily(ticker, limit)
		if err != nil {
			u.logger.Warn("FMP daily (weekly) fetch error", "ticker", ticker, "error", err)
			return false
		}
		weekly := resampleDailyToWeekly(rows)
		for _, w := range weekly {
			arg := db.UpsertWeeklyPriceParams{
				Ticker:     ticker,
				WeekEnding: pgtype.Date{Time: w.WeekEnding, Valid: true},
				Open:       pgtype.Float8{Float64: w.Open, Valid: true},
				High:       pgtype.Float8{Float64: w.High, Valid: true},
				Low:        pgtype.Float8{Float64: w.Low, Valid: true},
				Close:      w.Close,
				Volume:     pgtype.Int8{Int64: w.Volume, Valid: true},
			}
			if err := wq.UpsertWeeklyPrice(ctx, arg); err != nil {
				u.logger.Error("UpsertWeeklyPrice failed", "ticker", ticker, "error", err)
				return false
			}
		}
		return true
	})
}

type weeklyBar struct {
	WeekEnding time.Time
	Open       float64
	High       float64
	Low        float64
	Close      float64
	Volume     int64
}

func resampleDailyToWeekly(rows []dailyRow) []weeklyBar {
	if len(rows) == 0 {
		return nil
	}
	type dated struct {
		date time.Time
		r    dailyRow
	}
	var list []dated
	for _, r := range rows {
		date, err := time.Parse("2006-01-02", r.Date)
		if err != nil {
			continue
		}
		list = append(list, dated{date, r})
	}
	for i := 0; i < len(list); i++ {
		for j := i + 1; j < len(list); j++ {
			if list[j].date.Before(list[i].date) {
				list[i], list[j] = list[j], list[i]
			}
		}
	}
	byWeek := make(map[string]*weeklyBar)
	for _, d := range list {
		date, r := d.date, d.r
		weekday := int(date.Weekday())
		if weekday == 0 {
			weekday = 7
		}
		friday := date.AddDate(0, 0, 5-weekday)
		if friday.Before(date) {
			friday = date.AddDate(0, 0, 7)
		}
		key := friday.Format("2006-01-02")
		if b, ok := byWeek[key]; ok {
			if r.High > b.High {
				b.High = r.High
			}
			if r.Low < b.Low || b.Low == 0 {
				b.Low = r.Low
			}
			b.Close = r.Close
			b.Volume += r.Volume
		} else {
			byWeek[key] = &weeklyBar{
				WeekEnding: friday,
				Open:       r.Open,
				High:       r.High,
				Low:        r.Low,
				Close:      r.Close,
				Volume:     r.Volume,
			}
		}
	}
	var out []weeklyBar
	for _, b := range byWeek {
		out = append(out, *b)
	}
	return out
}

func (u *priceUpdater) updateHourly(ctx context.Context, q *db.Queries, exchange string, tickers []string) (updated, failed int) {
	// Pre-fetch latest hourly datetimes for all tickers to skip already-stored rows.
	lastDTs := make(map[string]time.Time)
	if rows, err := q.LastHourlyPerTicker(ctx); err == nil {
		for _, r := range rows {
			if t, ok := r.LastDt.(time.Time); ok {
				lastDTs[r.Ticker] = t
			}
		}
		u.logger.Info("hourly smart filter loaded", "tickers_with_data", len(lastDTs))
	} else {
		u.logger.Warn("LastHourlyPerTicker failed, upserting all rows", "error", err)
	}

	return u.runWorkers(ctx, tickers, func(ctx context.Context, wq *db.Queries, ticker string) bool {
		rows, err := u.fmp.FetchHourly(ticker)
		if err != nil {
			u.logger.Warn("FMP hourly fetch error", "ticker", ticker, "error", err)
			return false
		}

		cutoff := time.Time{}
		if lastDT, ok := lastDTs[ticker]; ok {
			cutoff = lastDT.AddDate(0, 0, -bufferDays)
		}

		for _, r := range rows {
			t, err := time.Parse("2006-01-02 15:04:05", r.Date)
			if err != nil {
				t, _ = time.Parse(time.RFC3339, r.Date)
			}
			if !cutoff.IsZero() && t.Before(cutoff) {
				continue
			}
			arg := db.UpsertHourlyPriceParams{
				Ticker:   ticker,
				Datetime: pgtype.Timestamptz{Time: t.UTC(), Valid: true},
				Open:     pgtype.Float8{Float64: r.Open, Valid: true},
				High:     pgtype.Float8{Float64: r.High, Valid: true},
				Low:      pgtype.Float8{Float64: r.Low, Valid: true},
				Close:    r.Close,
				Volume:   pgtype.Int8{Int64: r.Volume, Valid: true},
			}
			if err := wq.UpsertHourlyPrice(ctx, arg); err != nil {
				u.logger.Error("UpsertHourlyPrice failed", "ticker", ticker, "error", err)
				return false
			}
		}
		return true
	})
}

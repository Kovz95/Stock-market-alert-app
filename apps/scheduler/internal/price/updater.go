package price

import (
	"context"
	"log/slog"
	"time"

	"github.com/jackc/pgx/v5/pgtype"

	"stockalert/calendar"
	db "stockalert/database/generated"
	"stockalert/discord"
)

const dailyLimit = 750

// Updater orchestrates FMP API calls and upserts prices via sqlc.
type Updater struct {
	queries *db.Queries
	fmp     FMPFetcher
	logger  *slog.Logger
}

// NewUpdater creates an Updater.
func NewUpdater(queries *db.Queries, fmp FMPFetcher, logger *slog.Logger) *Updater {
	if logger == nil {
		logger = slog.Default()
	}
	return &Updater{queries: queries, fmp: fmp, logger: logger.With("component", "price_updater")}
}

// UpdateForExchange updates prices for all tickers on the given exchange and timeframe.
// Returns price stats for Discord. For hourly, only runs when exchange is open (caller checks).
func (u *Updater) UpdateForExchange(ctx context.Context, exchange, timeframe string) (*discord.PriceStats, error) {
	start := time.Now()
	tickers, err := u.tickersForExchange(ctx, exchange)
	if err != nil {
		return nil, err
	}
	stats := &discord.PriceStats{Total: len(tickers)}

	u.logger.Info("starting price update",
		"exchange", exchange,
		"timeframe", timeframe,
		"ticker_count", len(tickers),
	)

	if len(tickers) == 0 {
		u.logger.Info("no tickers found, skipping price update", "exchange", exchange)
		return stats, nil
	}

	switch timeframe {
	case "daily":
		updated, failed := u.updateDaily(ctx, tickers)
		stats.Updated = updated
		stats.Failed = failed
	case "weekly":
		updated, failed := u.updateWeekly(ctx, tickers)
		stats.Updated = updated
		stats.Failed = failed
	case "hourly":
		updated, failed := u.updateHourly(ctx, exchange, tickers)
		stats.Updated = updated
		stats.Failed = failed
	}

	u.logger.Info("price update complete",
		"exchange", exchange,
		"timeframe", timeframe,
		"updated", stats.Updated,
		"failed", stats.Failed,
		"total", stats.Total,
		"duration_ms", time.Since(start).Milliseconds(),
	)
	return stats, nil
}

func (u *Updater) tickersForExchange(ctx context.Context, exchange string) ([]string, error) {
	rows, err := u.queries.ListStockMetadataForAlerts(ctx)
	if err != nil {
		return nil, err
	}
	var tickers []string
	for _, r := range rows {
		if !r.Exchange.Valid {
			continue
		}
		if r.Exchange.String == exchange {
			tickers = append(tickers, r.Symbol)
		}
	}
	return tickers, nil
}

func (u *Updater) updateDaily(ctx context.Context, tickers []string) (updated, failed int) {
	var allRows []db.CopyDailyPricesParams
	for _, ticker := range tickers {
		rows, err := u.fmp.FetchDaily(ticker, dailyLimit)
		if err != nil {
			u.logger.Warn("FMP daily fetch error", "ticker", ticker, "error", err)
			failed++
			continue
		}
		for _, r := range rows {
			date, err := time.Parse("2006-01-02", r.Date)
			if err != nil {
				continue
			}
			allRows = append(allRows, db.CopyDailyPricesParams{
				Ticker: ticker,
				Date:   pgtype.Date{Time: date, Valid: true},
				Open:   pgtype.Float8{Float64: r.Open, Valid: true},
				High:   pgtype.Float8{Float64: r.High, Valid: true},
				Low:    pgtype.Float8{Float64: r.Low, Valid: true},
				Close:  r.Close,
				Volume: pgtype.Int8{Int64: r.Volume, Valid: true},
			})
		}
		updated++
	}
	if len(allRows) == 0 {
		return updated, failed
	}
	n, err := u.queries.CopyDailyPrices(ctx, allRows)
	if err != nil {
		u.logger.Error("CopyDailyPrices failed", "error", err)
		return 0, len(tickers)
	}
	u.logger.Debug("daily prices copied", "rows_copied", n, "tickers_processed", updated)
	return updated, failed
}

func (u *Updater) updateWeekly(ctx context.Context, tickers []string) (updated, failed int) {
	// Fetch daily and resample to week-ending (Friday)
	var allRows []db.CopyWeeklyPricesParams
	for _, ticker := range tickers {
		rows, err := u.fmp.FetchDaily(ticker, dailyLimit)
		if err != nil {
			u.logger.Warn("FMP daily (weekly) fetch error", "ticker", ticker, "error", err)
			failed++
			continue
		}
		weekly := resampleDailyToWeekly(rows)
		for _, w := range weekly {
			allRows = append(allRows, db.CopyWeeklyPricesParams{
				Ticker:     ticker,
				WeekEnding: pgtype.Date{Time: w.WeekEnding, Valid: true},
				Open:       pgtype.Float8{Float64: w.Open, Valid: true},
				High:       pgtype.Float8{Float64: w.High, Valid: true},
				Low:        pgtype.Float8{Float64: w.Low, Valid: true},
				Close:      w.Close,
				Volume:     pgtype.Int8{Int64: w.Volume, Valid: true},
			})
		}
		updated++
	}
	if len(allRows) == 0 {
		return updated, failed
	}
	n, err := u.queries.CopyWeeklyPrices(ctx, allRows)
	if err != nil {
		u.logger.Error("CopyWeeklyPrices failed", "error", err)
		return 0, len(tickers)
	}
	u.logger.Debug("weekly prices copied", "rows_copied", n, "tickers_processed", updated)
	return updated, failed
}

type weeklyBar struct {
	WeekEnding time.Time
	Open       float64
	High       float64
	Low        float64
	Close      float64
	Volume     int64
}

func resampleDailyToWeekly(rows []DailyRow) []weeklyBar {
	if len(rows) == 0 {
		return nil
	}
	// Sort by date (FMP may return newest first)
	type dated struct {
		date time.Time
		r    DailyRow
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
	// Group by week ending Friday: Open = first day of week, Close = last, High/Low = max/min
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

func (u *Updater) updateHourly(ctx context.Context, exchange string, tickers []string) (updated, failed int) {
	// Optional: respect calendar hourly alignment to only fetch when exchange is open
	_ = calendar.GetHourlyAlignment(exchange)

	var allRows []db.CopyHourlyPricesParams
	for _, ticker := range tickers {
		rows, err := u.fmp.FetchHourly(ticker)
		if err != nil {
			u.logger.Warn("FMP hourly fetch error", "ticker", ticker, "error", err)
			failed++
			continue
		}
		for _, r := range rows {
			// FMP returns "2024-01-15 14:00:00" or RFC3339
			t, err := time.Parse("2006-01-02 15:04:05", r.Date)
			if err != nil {
				t, _ = time.Parse(time.RFC3339, r.Date)
			}
			allRows = append(allRows, db.CopyHourlyPricesParams{
				Ticker:   ticker,
				Datetime: pgtype.Timestamptz{Time: t.UTC(), Valid: true},
				Open:     pgtype.Float8{Float64: r.Open, Valid: true},
				High:     pgtype.Float8{Float64: r.High, Valid: true},
				Low:      pgtype.Float8{Float64: r.Low, Valid: true},
				Close:    r.Close,
				Volume:   pgtype.Int8{Int64: r.Volume, Valid: true},
			})
		}
		updated++
	}
	if len(allRows) == 0 {
		return updated, failed
	}
	n, err := u.queries.CopyHourlyPrices(ctx, allRows)
	if err != nil {
		u.logger.Error("CopyHourlyPrices failed", "error", err)
		return 0, len(tickers)
	}
	u.logger.Debug("hourly prices copied", "rows_copied", n, "tickers_processed", updated)
	return updated, failed
}

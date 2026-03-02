package main

import (
	"context"
	"log"
	"sync"
	"time"

	"github.com/jackc/pgx/v5/pgtype"
	"github.com/jackc/pgx/v5/pgxpool"

	"stockalert/calendar"
	db "stockalert/database/generated"
)

const (
	dailyFetchLimit    = 750
	fmpConcurrency     = 10
	upsertDailySQL     = `INSERT INTO daily_prices (ticker, date, open, high, low, close, volume)
SELECT unnest($1::text[]), unnest($2::date[]), unnest($3::float8[]), unnest($4::float8[]),
       unnest($5::float8[]), unnest($6::float8[]), unnest($7::bigint[])
ON CONFLICT (ticker, date) DO UPDATE SET
    open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
    close = EXCLUDED.close, volume = EXCLUDED.volume, updated_at = NOW()`
	upsertWeeklySQL    = `INSERT INTO weekly_prices (ticker, week_ending, open, high, low, close, volume)
SELECT unnest($1::text[]), unnest($2::date[]), unnest($3::float8[]), unnest($4::float8[]),
       unnest($5::float8[]), unnest($6::float8[]), unnest($7::bigint[])
ON CONFLICT (ticker, week_ending) DO UPDATE SET
    open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
    close = EXCLUDED.close, volume = EXCLUDED.volume, updated_at = NOW()`
	upsertHourlySQL    = `INSERT INTO hourly_prices (ticker, datetime, open, high, low, close, volume)
SELECT unnest($1::text[]), unnest($2::timestamptz[]), unnest($3::float8[]), unnest($4::float8[]),
       unnest($5::float8[]), unnest($6::float8[]), unnest($7::bigint[])
ON CONFLICT (ticker, datetime) DO UPDATE SET
    open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low,
    close = EXCLUDED.close, volume = EXCLUDED.volume, updated_at = NOW()`
)

// priceStats summarises a price-update run.
type priceStats struct {
	Total   int
	Updated int
	Failed  int
	Skipped int
}

// priceUpdater orchestrates FMP API calls and upserts prices into the DB.
type priceUpdater struct {
	pool    *pgxpool.Pool
	queries *db.Queries
	fmp     fmpFetcher
}

// newPriceUpdater creates a priceUpdater.
func newPriceUpdater(pool *pgxpool.Pool, queries *db.Queries, fmp fmpFetcher) *priceUpdater {
	return &priceUpdater{pool: pool, queries: queries, fmp: fmp}
}

// UpdateForExchange updates prices for all tickers on the given exchange and timeframe.
func (u *priceUpdater) UpdateForExchange(ctx context.Context, exchange, timeframe string) (*priceStats, error) {
	tickers, err := u.tickersForExchange(ctx, exchange)
	if err != nil {
		return nil, err
	}
	stats := &priceStats{Total: len(tickers)}
	if len(tickers) == 0 {
		return stats, nil
	}

	switch timeframe {
	case "daily":
		stats.Updated, stats.Failed = u.updateDaily(ctx, tickers)
	case "weekly":
		stats.Updated, stats.Failed = u.updateWeekly(ctx, tickers)
	case "hourly":
		stats.Updated, stats.Failed = u.updateHourly(ctx, exchange, tickers)
	}
	return stats, nil
}

func (u *priceUpdater) tickersForExchange(ctx context.Context, exchange string) ([]string, error) {
	rows, err := u.queries.ListStockMetadataForAlerts(ctx)
	if err != nil {
		return nil, err
	}
	var tickers []string
	for _, r := range rows {
		if r.Exchange.Valid && r.Exchange.String == exchange {
			tickers = append(tickers, r.Symbol)
		}
	}
	return tickers, nil
}

type dailyFetchResult struct {
	ticker string
	rows   []db.CopyDailyPricesParams
	err    error
}

func (u *priceUpdater) updateDaily(ctx context.Context, tickers []string) (updated, failed int) {
	results := make(chan dailyFetchResult, len(tickers))
	sem := make(chan struct{}, fmpConcurrency)
	var wg sync.WaitGroup

	for _, ticker := range tickers {
		if ctx.Err() != nil {
			break
		}
		wg.Add(1)
		t := ticker
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			apiRows, err := u.fmp.FetchDaily(t, dailyFetchLimit)
			if err != nil {
				results <- dailyFetchResult{ticker: t, err: err}
				return
			}
			var params []db.CopyDailyPricesParams
			for _, r := range apiRows {
				date, err := time.Parse("2006-01-02", r.Date)
				if err != nil {
					continue
				}
				params = append(params, db.CopyDailyPricesParams{
					Ticker: t,
					Date:   pgtype.Date{Time: date, Valid: true},
					Open:   pgtype.Float8{Float64: r.Open, Valid: true},
					High:   pgtype.Float8{Float64: r.High, Valid: true},
					Low:    pgtype.Float8{Float64: r.Low, Valid: true},
					Close:  r.Close,
					Volume: pgtype.Int8{Int64: r.Volume, Valid: true},
				})
			}
			results <- dailyFetchResult{ticker: t, rows: params}
		}()
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	var allRows []db.CopyDailyPricesParams
	for r := range results {
		if r.err != nil {
			log.Printf("fmp daily %s: %v", r.ticker, r.err)
			failed++
		} else {
			allRows = append(allRows, r.rows...)
			updated++
		}
	}

	if len(allRows) == 0 {
		return updated, failed
	}
	if err := u.bulkUpsertDaily(ctx, allRows); err != nil {
		log.Printf("bulkUpsertDaily: %v", err)
		return 0, len(tickers)
	}
	log.Printf("price/daily: upserted %d rows for %d tickers", len(allRows), updated)
	return updated, failed
}

func (u *priceUpdater) bulkUpsertDaily(ctx context.Context, rows []db.CopyDailyPricesParams) error {
	tickers := make([]string, len(rows))
	dates := make([]pgtype.Date, len(rows))
	opens := make([]pgtype.Float8, len(rows))
	highs := make([]pgtype.Float8, len(rows))
	lows := make([]pgtype.Float8, len(rows))
	closes := make([]float64, len(rows))
	volumes := make([]pgtype.Int8, len(rows))
	for i, r := range rows {
		tickers[i] = r.Ticker
		dates[i] = r.Date
		opens[i] = r.Open
		highs[i] = r.High
		lows[i] = r.Low
		closes[i] = r.Close
		volumes[i] = r.Volume
	}
	_, err := u.pool.Exec(ctx, upsertDailySQL, tickers, dates, opens, highs, lows, closes, volumes)
	return err
}

type weeklyFetchResult struct {
	ticker string
	rows   []db.CopyWeeklyPricesParams
	err    error
}

func (u *priceUpdater) updateWeekly(ctx context.Context, tickers []string) (updated, failed int) {
	results := make(chan weeklyFetchResult, len(tickers))
	sem := make(chan struct{}, fmpConcurrency)
	var wg sync.WaitGroup

	for _, ticker := range tickers {
		if ctx.Err() != nil {
			break
		}
		wg.Add(1)
		t := ticker
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			apiRows, err := u.fmp.FetchDaily(t, dailyFetchLimit)
			if err != nil {
				results <- weeklyFetchResult{ticker: t, err: err}
				return
			}
			var params []db.CopyWeeklyPricesParams
			for _, w := range resampleToWeekly(apiRows) {
				params = append(params, db.CopyWeeklyPricesParams{
					Ticker:     t,
					WeekEnding: pgtype.Date{Time: w.weekEnding, Valid: true},
					Open:       pgtype.Float8{Float64: w.open, Valid: true},
					High:       pgtype.Float8{Float64: w.high, Valid: true},
					Low:        pgtype.Float8{Float64: w.low, Valid: true},
					Close:      w.close,
					Volume:     pgtype.Int8{Int64: w.volume, Valid: true},
				})
			}
			results <- weeklyFetchResult{ticker: t, rows: params}
		}()
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	var allRows []db.CopyWeeklyPricesParams
	for r := range results {
		if r.err != nil {
			log.Printf("fmp daily (weekly) %s: %v", r.ticker, r.err)
			failed++
		} else {
			allRows = append(allRows, r.rows...)
			updated++
		}
	}

	if len(allRows) == 0 {
		return updated, failed
	}
	if err := u.bulkUpsertWeekly(ctx, allRows); err != nil {
		log.Printf("bulkUpsertWeekly: %v", err)
		return 0, len(tickers)
	}
	log.Printf("price/weekly: upserted %d rows for %d tickers", len(allRows), updated)
	return updated, failed
}

func (u *priceUpdater) bulkUpsertWeekly(ctx context.Context, rows []db.CopyWeeklyPricesParams) error {
	tickers := make([]string, len(rows))
	weekEndings := make([]pgtype.Date, len(rows))
	opens := make([]pgtype.Float8, len(rows))
	highs := make([]pgtype.Float8, len(rows))
	lows := make([]pgtype.Float8, len(rows))
	closes := make([]float64, len(rows))
	volumes := make([]pgtype.Int8, len(rows))
	for i, r := range rows {
		tickers[i] = r.Ticker
		weekEndings[i] = r.WeekEnding
		opens[i] = r.Open
		highs[i] = r.High
		lows[i] = r.Low
		closes[i] = r.Close
		volumes[i] = r.Volume
	}
	_, err := u.pool.Exec(ctx, upsertWeeklySQL, tickers, weekEndings, opens, highs, lows, closes, volumes)
	return err
}

type hourlyFetchResult struct {
	ticker string
	rows   []db.CopyHourlyPricesParams
	err    error
}

func (u *priceUpdater) updateHourly(ctx context.Context, exchange string, tickers []string) (updated, failed int) {
	_ = calendar.GetHourlyAlignment(exchange)

	results := make(chan hourlyFetchResult, len(tickers))
	sem := make(chan struct{}, fmpConcurrency)
	var wg sync.WaitGroup

	for _, ticker := range tickers {
		if ctx.Err() != nil {
			break
		}
		wg.Add(1)
		t := ticker
		go func() {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()

			apiRows, err := u.fmp.FetchHourly(t)
			if err != nil {
				results <- hourlyFetchResult{ticker: t, err: err}
				return
			}
			var params []db.CopyHourlyPricesParams
			for _, r := range apiRows {
				ts, err := time.Parse("2006-01-02 15:04:05", r.Date)
				if err != nil {
					ts, _ = time.Parse(time.RFC3339, r.Date)
				}
				params = append(params, db.CopyHourlyPricesParams{
					Ticker:   t,
					Datetime: pgtype.Timestamptz{Time: ts.UTC(), Valid: true},
					Open:     pgtype.Float8{Float64: r.Open, Valid: true},
					High:     pgtype.Float8{Float64: r.High, Valid: true},
					Low:      pgtype.Float8{Float64: r.Low, Valid: true},
					Close:    r.Close,
					Volume:   pgtype.Int8{Int64: r.Volume, Valid: true},
				})
			}
			results <- hourlyFetchResult{ticker: t, rows: params}
		}()
	}

	go func() {
		wg.Wait()
		close(results)
	}()

	var allRows []db.CopyHourlyPricesParams
	for r := range results {
		if r.err != nil {
			log.Printf("fmp hourly %s: %v", r.ticker, r.err)
			failed++
		} else {
			allRows = append(allRows, r.rows...)
			updated++
		}
	}

	if len(allRows) == 0 {
		return updated, failed
	}
	if err := u.bulkUpsertHourly(ctx, allRows); err != nil {
		log.Printf("bulkUpsertHourly: %v", err)
		return 0, len(tickers)
	}
	log.Printf("price/hourly: upserted %d rows for %d tickers", len(allRows), updated)
	return updated, failed
}

func (u *priceUpdater) bulkUpsertHourly(ctx context.Context, rows []db.CopyHourlyPricesParams) error {
	tickers := make([]string, len(rows))
	datetimes := make([]pgtype.Timestamptz, len(rows))
	opens := make([]pgtype.Float8, len(rows))
	highs := make([]pgtype.Float8, len(rows))
	lows := make([]pgtype.Float8, len(rows))
	closes := make([]float64, len(rows))
	volumes := make([]pgtype.Int8, len(rows))
	for i, r := range rows {
		tickers[i] = r.Ticker
		datetimes[i] = r.Datetime
		opens[i] = r.Open
		highs[i] = r.High
		lows[i] = r.Low
		closes[i] = r.Close
		volumes[i] = r.Volume
	}
	_, err := u.pool.Exec(ctx, upsertHourlySQL, tickers, datetimes, opens, highs, lows, closes, volumes)
	return err
}

type weeklyBar struct {
	weekEnding time.Time
	open       float64
	high       float64
	low        float64
	close      float64
	volume     int64
}

func resampleToWeekly(rows []dailyRow) []weeklyBar {
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
			if r.High > b.high {
				b.high = r.High
			}
			if r.Low < b.low || b.low == 0 {
				b.low = r.Low
			}
			b.close = r.Close
			b.volume += r.Volume
		} else {
			byWeek[key] = &weeklyBar{
				weekEnding: friday,
				open:       r.Open,
				high:       r.High,
				low:        r.Low,
				close:      r.Close,
				volume:     r.Volume,
			}
		}
	}
	var out []weeklyBar
	for _, b := range byWeek {
		out = append(out, *b)
	}
	return out
}

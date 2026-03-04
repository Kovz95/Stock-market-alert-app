package price

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/jackc/pgx/v5/pgtype"

	"stockalert/calendar"
	db "stockalert/database/generated"
	"stockalert/discord"
)

const (
	dailyLimit          = 750
	hourlyUpsertChunk   = 10000 // rows per BulkUpsertHourlyPrices call
	defaultDailyConc   = 25    // used when DailyConcurrency <= 0
	defaultWeeklyConc  = 25    // used when WeeklyConcurrency <= 0
	defaultHourlyConc   = 25    // used when HourlyConcurrency <= 0
)

// Updater orchestrates FMP API calls and upserts prices via sqlc.
type Updater struct {
	queries             *db.Queries
	fmp                 FMPFetcher
	logger              *slog.Logger
	dailyConcurrency    int
	weeklyConcurrency   int
	hourlyConcurrency   int
}

// NewUpdater creates an Updater. dailyConc, weeklyConc, hourlyConc are max parallel FMP fetches per timeframe (0 = use defaults).
func NewUpdater(queries *db.Queries, fmp FMPFetcher, logger *slog.Logger, dailyConc, weeklyConc, hourlyConc int) *Updater {
	if logger == nil {
		logger = slog.Default()
	}
	d := dailyConc
	if d <= 0 {
		d = defaultDailyConc
	}
	w := weeklyConc
	if w <= 0 {
		w = defaultWeeklyConc
	}
	h := hourlyConc
	if h <= 0 {
		h = defaultHourlyConc
	}
	return &Updater{
		queries:           queries,
		fmp:               fmp,
		logger:             logger.With("component", "price_updater"),
		dailyConcurrency:   d,
		weeklyConcurrency: w,
		hourlyConcurrency: h,
	}
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

type dailyFetchResult struct {
	ticker string
	rows   []DailyRow
	err    error
}

func (u *Updater) updateDaily(ctx context.Context, tickers []string) (updated, failed int) {
	conc := u.dailyConcurrency
	if conc <= 0 {
		conc = defaultDailyConc
	}
	results := make(chan dailyFetchResult, len(tickers))
	sem := make(chan struct{}, conc)
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
			rows, err := u.fmp.FetchDaily(t, dailyLimit)
			if err != nil {
				results <- dailyFetchResult{ticker: t, err: err}
				return
			}
			results <- dailyFetchResult{ticker: t, rows: rows}
		}()
	}
	go func() {
		wg.Wait()
		close(results)
	}()

	for r := range results {
		if r.err != nil {
			u.logger.Warn("FMP daily fetch error", "ticker", r.ticker, "error", r.err)
			failed++
			continue
		}
		ok := true
		for _, row := range r.rows {
			date, err := time.Parse("2006-01-02", row.Date)
			if err != nil {
				continue
			}
			arg := db.UpsertDailyPriceParams{
				Ticker: r.ticker,
				Date:   pgtype.Date{Time: date, Valid: true},
				Open:   pgtype.Float8{Float64: row.Open, Valid: true},
				High:   pgtype.Float8{Float64: row.High, Valid: true},
				Low:    pgtype.Float8{Float64: row.Low, Valid: true},
				Close:  row.Close,
				Volume: pgtype.Int8{Int64: row.Volume, Valid: true},
			}
			if err := u.queries.UpsertDailyPrice(ctx, arg); err != nil {
				u.logger.Warn("UpsertDailyPrice failed", "ticker", r.ticker, "date", row.Date, "error", err)
				failed++
				ok = false
				break
			}
		}
		if ok {
			updated++
		}
	}
	return updated, failed
}

type weeklyFetchResult struct {
	ticker string
	rows   []DailyRow
	err    error
}

func (u *Updater) updateWeekly(ctx context.Context, tickers []string) (updated, failed int) {
	conc := u.weeklyConcurrency
	if conc <= 0 {
		conc = defaultWeeklyConc
	}
	results := make(chan weeklyFetchResult, len(tickers))
	sem := make(chan struct{}, conc)
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
			rows, err := u.fmp.FetchDaily(t, dailyLimit)
			if err != nil {
				results <- weeklyFetchResult{ticker: t, err: err}
				return
			}
			results <- weeklyFetchResult{ticker: t, rows: rows}
		}()
	}
	go func() {
		wg.Wait()
		close(results)
	}()

	for r := range results {
		if r.err != nil {
			u.logger.Warn("FMP daily (weekly) fetch error", "ticker", r.ticker, "error", r.err)
			failed++
			continue
		}
		weekly := resampleDailyToWeekly(r.rows)
		ok := true
		for _, w := range weekly {
			arg := db.UpsertWeeklyPriceParams{
				Ticker:     r.ticker,
				WeekEnding: pgtype.Date{Time: w.WeekEnding, Valid: true},
				Open:       pgtype.Float8{Float64: w.Open, Valid: true},
				High:       pgtype.Float8{Float64: w.High, Valid: true},
				Low:        pgtype.Float8{Float64: w.Low, Valid: true},
				Close:      w.Close,
				Volume:     pgtype.Int8{Int64: w.Volume, Valid: true},
			}
			if err := u.queries.UpsertWeeklyPrice(ctx, arg); err != nil {
				u.logger.Warn("UpsertWeeklyPrice failed", "ticker", r.ticker, "week_ending", w.WeekEnding.Format("2006-01-02"), "error", err)
				failed++
				ok = false
				break
			}
		}
		if ok {
			updated++
		}
	}
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

type hourlyFetchResult struct {
	ticker string
	rows   []HourlyRow
	err    error
}

func (u *Updater) updateHourly(ctx context.Context, exchange string, tickers []string) (updated, failed int) {
	_ = calendar.GetHourlyAlignment(exchange)

	conc := u.hourlyConcurrency
	if conc <= 0 {
		conc = defaultHourlyConc
	}
	results := make(chan hourlyFetchResult, len(tickers))
	sem := make(chan struct{}, conc)
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
			rows, err := u.fmp.FetchHourly(t)
			if err != nil {
				results <- hourlyFetchResult{ticker: t, err: err}
				return
			}
			results <- hourlyFetchResult{ticker: t, rows: rows}
		}()
	}
	go func() {
		wg.Wait()
		close(results)
	}()

	var tickersCol []string
	var datetimesCol []pgtype.Timestamptz
	var opensCol, highsCol, lowsCol []float64
	var closesCol []float64
	var volumesCol []int64

	for r := range results {
		if r.err != nil {
			u.logger.Warn("FMP hourly fetch error", "ticker", r.ticker, "error", r.err)
			failed++
			continue
		}
		for _, row := range r.rows {
			t, err := time.Parse("2006-01-02 15:04:05", row.Date)
			if err != nil {
				t, _ = time.Parse(time.RFC3339, row.Date)
			}
			tickersCol = append(tickersCol, r.ticker)
			datetimesCol = append(datetimesCol, pgtype.Timestamptz{Time: t.UTC(), Valid: true})
			opensCol = append(opensCol, row.Open)
			highsCol = append(highsCol, row.High)
			lowsCol = append(lowsCol, row.Low)
			closesCol = append(closesCol, row.Close)
			volumesCol = append(volumesCol, int64(row.Volume))
		}
		updated++
	}

	if len(tickersCol) == 0 {
		return updated, failed
	}

	// Bulk upsert in chunks to avoid huge single query
	for i := 0; i < len(tickersCol); i += hourlyUpsertChunk {
		end := i + hourlyUpsertChunk
		if end > len(tickersCol) {
			end = len(tickersCol)
		}
		arg := db.BulkUpsertHourlyPricesParams{
			Column1: tickersCol[i:end],
			Column2: datetimesCol[i:end],
			Column3: opensCol[i:end],
			Column4: highsCol[i:end],
			Column5: lowsCol[i:end],
			Column6: closesCol[i:end],
			Column7: volumesCol[i:end],
		}
		if err := u.queries.BulkUpsertHourlyPrices(ctx, arg); err != nil {
			u.logger.Warn("BulkUpsertHourlyPrices failed", "offset", i, "count", end-i, "error", err)
			return updated, failed
		}
	}
	return updated, failed
}

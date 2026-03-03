package main

import (
	"context"
	"log/slog"
	"strings"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"stockalert/calendar"
	schedulerv1 "stockalert/gen/go/scheduler/v1"
)

func (s *Server) GetExchangeSchedule(ctx context.Context, req *schedulerv1.GetExchangeScheduleRequest) (*schedulerv1.GetExchangeScheduleResponse, error) {
	timeframe := req.GetTimeframe()
	if timeframe == "" {
		timeframe = "daily"
		s.logger.Debug("GetExchangeSchedule: timeframe defaulted", "timeframe", timeframe)
	} else {
		s.logger.Debug("GetExchangeSchedule: request received", "timeframe", timeframe)
	}

	now := time.Now()
	eastern, _ := time.LoadLocation("America/New_York")
	nowET := now.In(eastern)
	s.logger.Debug("GetExchangeSchedule: reference time set",
		"now_utc", now.UTC().Format(time.RFC3339),
		"now_et", nowET.Format(time.RFC3339),
	)

	symbols := make([]string, 0, len(calendar.ExchangeSchedules))
	for sym := range calendar.ExchangeSchedules {
		symbols = append(symbols, sym)
	}
	s.logger.Debug("GetExchangeSchedule: fetching last runs",
		"exchange_count", len(symbols),
		"timeframe", timeframe,
	)

	queryStart := time.Now()
	lastRuns := fetchLastRuns(ctx, s.pool, symbols, timeframe)
	s.logger.Info("GetExchangeSchedule: last runs fetched",
		"timeframe", timeframe,
		"exchanges_with_data", len(lastRuns),
		"total_exchanges", len(symbols),
		"query_duration_ms", time.Since(queryStart).Milliseconds(),
	)

	rows := make([]*schedulerv1.ExchangeScheduleRow, 0, len(calendar.ExchangeSchedules))

	buildStart := time.Now()
	switch timeframe {
	case "hourly":
		s.logger.Debug("GetExchangeSchedule: building hourly rows")
		rows = buildHourlyRows(nowET, now, eastern, lastRuns)
	case "weekly":
		s.logger.Debug("GetExchangeSchedule: building weekly rows")
		rows = buildWeeklyRows(nowET, now, eastern, lastRuns)
	default:
		s.logger.Debug("GetExchangeSchedule: building daily rows")
		rows = buildDailyRows(nowET, now, eastern, lastRuns)
	}
	s.logger.Info("GetExchangeSchedule: rows built",
		"timeframe", timeframe,
		"row_count", len(rows),
		"build_duration_ms", time.Since(buildStart).Milliseconds(),
	)

	return &schedulerv1.GetExchangeScheduleResponse{Rows: rows}, nil
}

// buildDailyRows computes the daily schedule: run at session close + 40 min.
func buildDailyRows(nowET, now time.Time, eastern *time.Location, lastRuns map[string]lastRunInfo) []*schedulerv1.ExchangeScheduleRow {
	rows := make([]*schedulerv1.ExchangeScheduleRow, 0, len(calendar.ExchangeSchedules))
	for sym, sched := range calendar.ExchangeSchedules {
		tz := calendar.GetCalendarTimezone(sym)
		region := regionFromTZ(tz)

		// Derive display times from the actual next run so they match the countdown.
		nextRun := calendar.GetNextDailyRunTime(sym, now)
		remaining := int64(nextRun.Sub(now).Seconds())
		if remaining < 0 {
			remaining = 0
		}

		nextRunET := nextRun.In(eastern)
		nextCloseET := nextRunET.Add(-40 * time.Minute)

		// Close (ET): exchange close time in Eastern
		closeEt := nextCloseET.Format("3:04 PM")

		row := &schedulerv1.ExchangeScheduleRow{
			Exchange:             sched.Name,
			Symbol:               sym,
			Region:               region,
			RunTimeEt:            nextRunET.Format("3:04 PM"),
			RunTimeUtc:           nextRun.UTC().Format("3:04 PM"),
			LocalClose:           closeEt,
			LocalTz:              tz,
			TimeRemainingSeconds: remaining,
		}
		if lr, ok := lastRuns[sym]; ok {
			row.LastRunDate = lr.day
			row.LastRunStart = lr.start
			row.LastRunEnd = lr.end
		}
		rows = append(rows, row)
	}
	return rows
}

// buildWeeklyRows: same close times as daily but time-remaining is to next Friday's run.
func buildWeeklyRows(nowET, now time.Time, eastern *time.Location, lastRuns map[string]lastRunInfo) []*schedulerv1.ExchangeScheduleRow {
	rows := make([]*schedulerv1.ExchangeScheduleRow, 0, len(calendar.ExchangeSchedules))
	for sym, sched := range calendar.ExchangeSchedules {
		tz := calendar.GetCalendarTimezone(sym)
		region := regionFromTZ(tz)

		closeHour, closeMin := calendar.GetExchangeCloseTime(sym, nowET)
		closeET := time.Date(nowET.Year(), nowET.Month(), nowET.Day(), closeHour, closeMin, 0, 0, eastern)
		runET := closeET.Add(40 * time.Minute)
		runUTC := runET.UTC()

		// Close (ET): exchange close time in Eastern
		closeEt := closeET.Format("3:04 PM")

		// For weekly, find the next Friday at the daily run time.
		nextFriday := nextWeeklyRunTime(sym, now, eastern)
		remaining := int64(nextFriday.Sub(now).Seconds())
		if remaining < 0 {
			remaining = 0
		}

		row := &schedulerv1.ExchangeScheduleRow{
			Exchange:             sched.Name,
			Symbol:               sym,
			Region:               region,
			RunTimeEt:            runET.Format("3:04 PM"),
			RunTimeUtc:           runUTC.Format("3:04 PM"),
			LocalClose:           closeEt,
			LocalTz:              tz,
			TimeRemainingSeconds: remaining,
		}
		if lr, ok := lastRuns[sym]; ok {
			row.LastRunDate = lr.day
			row.LastRunStart = lr.start
			row.LastRunEnd = lr.end
		}
		rows = append(rows, row)
	}
	return rows
}

// nextWeeklyRunTime returns the next Friday's daily run time (close+40min) in UTC.
func nextWeeklyRunTime(exchange string, now time.Time, eastern *time.Location) time.Time {
	ref := now.In(eastern)
	// Find the next Friday (could be today if it's Friday and run hasn't happened yet).
	daysUntilFriday := (int(time.Friday) - int(ref.Weekday()) + 7) % 7
	candidate := ref.AddDate(0, 0, daysUntilFriday)
	closeHour, closeMin := calendar.GetExchangeCloseTime(exchange, candidate)
	runTime := time.Date(candidate.Year(), candidate.Month(), candidate.Day(), closeHour, closeMin+40, 0, 0, eastern)
	// If today is Friday but the run time has already passed, advance by one week.
	if daysUntilFriday == 0 && now.After(runTime) {
		candidate = ref.AddDate(0, 0, 7)
		closeHour, closeMin = calendar.GetExchangeCloseTime(exchange, candidate)
		runTime = time.Date(candidate.Year(), candidate.Month(), candidate.Day(), closeHour, closeMin+40, 0, 0, eastern)
	}
	return runTime.UTC()
}

// buildHourlyRows computes the hourly schedule: next aligned candle time while exchange is open.
// LocalClose shows Close (ET): exchange session close time in Eastern.
// RunTimeEt shows the next candle time; TimeRemaining counts down to it.
// Exchanges that are currently closed show TimeRemaining = 0.
func buildHourlyRows(nowET, now time.Time, eastern *time.Location, lastRuns map[string]lastRunInfo) []*schedulerv1.ExchangeScheduleRow {
	rows := make([]*schedulerv1.ExchangeScheduleRow, 0, len(calendar.ExchangeSchedules))
	for sym, sched := range calendar.ExchangeSchedules {
		tz := calendar.GetCalendarTimezone(sym)
		region := regionFromTZ(tz)

		isOpen := calendar.IsExchangeOpen(sym, now)
		var nextCandleET time.Time
		var remaining int64
		alignment := calendar.GetHourlyAlignment(sym)

		if isOpen {
			nextCandleET = nextHourlyCandle(now, eastern, alignment)
			remaining = int64(nextCandleET.Sub(now).Seconds())
			if remaining < 0 {
				remaining = 0
			}
		}

		runTimeEt := ""
		runTimeUtc := ""
		if isOpen {
			runTimeEt = nextCandleET.In(eastern).Format("3:04 PM")
			runTimeUtc = nextCandleET.UTC().Format("3:04 PM")
		}

		// Close (ET): exchange session close time in Eastern
		closeHour, closeMin := calendar.GetExchangeCloseTime(sym, nowET)
		closeET := time.Date(nowET.Year(), nowET.Month(), nowET.Day(), closeHour, closeMin, 0, 0, eastern)
		closeEt := closeET.Format("3:04 PM")

		row := &schedulerv1.ExchangeScheduleRow{
			Exchange:             sched.Name,
			Symbol:               sym,
			Region:               region,
			RunTimeEt:            runTimeEt,
			RunTimeUtc:           runTimeUtc,
			LocalClose:           closeEt,
			LocalTz:              tz,
			TimeRemainingSeconds: remaining,
		}
		if lr, ok := lastRuns[sym]; ok {
			row.LastRunDate = lr.day
			row.LastRunStart = lr.start
			row.LastRunEnd = lr.end
		}
		rows = append(rows, row)
	}
	return rows
}

// nextHourlyCandle returns the next candle boundary (in eastern) after now,
// based on alignment: "hour" → next :00, "quarter" → next :15, "half" → next :30.
func nextHourlyCandle(now time.Time, eastern *time.Location, alignment string) time.Time {
	et := now.In(eastern)
	y, mo, d := et.Date()
	h, m := et.Hour(), et.Minute()

	switch alignment {
	case "quarter":
		// Next multiple of 15 minutes after current minute
		nextM := ((m / 15) + 1) * 15
		extraH := nextM / 60
		nextM = nextM % 60
		return time.Date(y, mo, d, h+extraH, nextM, 0, 0, eastern)
	case "half":
		// Next :00 or :30
		if m < 30 {
			return time.Date(y, mo, d, h, 30, 0, 0, eastern)
		}
		return time.Date(y, mo, d, h+1, 0, 0, 0, eastern)
	default: // "hour"
		return time.Date(y, mo, d, h+1, 0, 0, 0, eastern)
	}
}

func regionFromTZ(tz string) string {
	if tz == "" {
		return "Unknown"
	}
	if strings.HasPrefix(tz, "Asia/") || strings.HasPrefix(tz, "Australia/") || strings.HasPrefix(tz, "Pacific/") {
		return "Asia-Pacific"
	}
	if strings.HasPrefix(tz, "America/") || strings.HasPrefix(tz, "Atlantic/") {
		return "Americas"
	}
	return "Europe"
}

type lastRunInfo struct {
	day   string
	start string
	end   string
}

func fetchLastRuns(ctx context.Context, pool *pgxpool.Pool, exchanges []string, timeframe string) map[string]lastRunInfo {
	results := make(map[string]lastRunInfo)
	logger := slog.Default()

	logger.Debug("fetchLastRuns: querying alert_audits",
		"exchange_count", len(exchanges),
		"timeframe", timeframe,
	)

	query := `
		WITH per_day AS (
			SELECT
				exchange,
				DATE(timestamp) AS day,
				MIN(timestamp) AS start_ts,
				MAX(timestamp) AS end_ts
			FROM alert_audits
			WHERE exchange = ANY($1)
			  AND evaluation_type IN ('scheduled', 'parallel')
			  AND timeframe = $2
			GROUP BY exchange, DATE(timestamp)
		),
		latest AS (
			SELECT DISTINCT ON (exchange)
				exchange, day, start_ts, end_ts
			FROM per_day
			ORDER BY exchange, day DESC
		)
		SELECT exchange, day, start_ts, end_ts
		FROM latest`

	rows, err := pool.Query(ctx, query, exchanges, timeframe)
	if err != nil {
		logger.Error("fetchLastRuns: query failed",
			"timeframe", timeframe,
			"error", err,
		)
		return results
	}
	defer rows.Close()

	scanErrors := 0
	for rows.Next() {
		var exchange string
		var day time.Time
		var startTS, endTS time.Time
		if err := rows.Scan(&exchange, &day, &startTS, &endTS); err != nil {
			logger.Warn("fetchLastRuns: failed to scan row", "error", err)
			scanErrors++
			continue
		}
		results[exchange] = lastRunInfo{
			day:   day.Format("2006-01-02"),
			start: startTS.UTC().Format(time.RFC3339),
			end:   endTS.UTC().Format(time.RFC3339),
		}
		logger.Debug("fetchLastRuns: row scanned",
			"exchange", exchange,
			"day", day.Format("2006-01-02"),
			"start", startTS.UTC().Format(time.RFC3339),
			"end", endTS.UTC().Format(time.RFC3339),
		)
	}

	if scanErrors > 0 {
		logger.Warn("fetchLastRuns: completed with scan errors",
			"scan_errors", scanErrors,
			"results_count", len(results),
		)
	} else {
		logger.Debug("fetchLastRuns: completed",
			"results_count", len(results),
			"timeframe", timeframe,
		)
	}

	return results
}

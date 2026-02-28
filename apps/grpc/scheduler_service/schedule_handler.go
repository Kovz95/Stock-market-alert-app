package main

import (
	"context"
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
	}

	now := time.Now()
	eastern, _ := time.LoadLocation("America/New_York")
	nowET := now.In(eastern)

	symbols := make([]string, 0, len(calendar.ExchangeSchedules))
	for sym := range calendar.ExchangeSchedules {
		symbols = append(symbols, sym)
	}

	lastRuns := fetchLastRuns(ctx, s.pool, symbols, timeframe)

	rows := make([]*schedulerv1.ExchangeScheduleRow, 0, len(calendar.ExchangeSchedules))

	switch timeframe {
	case "hourly":
		rows = buildHourlyRows(nowET, now, eastern, lastRuns)
	case "weekly":
		rows = buildWeeklyRows(nowET, now, eastern, lastRuns)
	default:
		rows = buildDailyRows(nowET, now, eastern, lastRuns)
	}

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

		localClose := ""
		if tz != "" {
			if loc, err := time.LoadLocation(tz); err == nil {
				localClose = nextCloseET.In(loc).Format("15:04")
			}
		}

		row := &schedulerv1.ExchangeScheduleRow{
			Exchange:             sched.Name,
			Symbol:               sym,
			Region:               region,
			RunTimeEt:            nextRunET.Format("15:04"),
			RunTimeUtc:           nextRun.UTC().Format("15:04"),
			LocalClose:           localClose,
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

		localClose := ""
		if tz != "" {
			if loc, err := time.LoadLocation(tz); err == nil {
				localClose = closeET.In(loc).Format("15:04")
			}
		}

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
			RunTimeEt:            runET.Format("15:04"),
			RunTimeUtc:           runUTC.Format("15:04"),
			LocalClose:           localClose,
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
// LocalClose shows the hourly alignment ("hour", "quarter", or "half").
// RunTimeEt shows the next candle time; TimeRemaining counts down to it.
// Exchanges that are currently closed show TimeRemaining = 0.
func buildHourlyRows(nowET, now time.Time, eastern *time.Location, lastRuns map[string]lastRunInfo) []*schedulerv1.ExchangeScheduleRow {
	rows := make([]*schedulerv1.ExchangeScheduleRow, 0, len(calendar.ExchangeSchedules))
	for sym, sched := range calendar.ExchangeSchedules {
		tz := calendar.GetCalendarTimezone(sym)
		region := regionFromTZ(tz)
		alignment := calendar.GetHourlyAlignment(sym)

		isOpen := calendar.IsExchangeOpen(sym, now)
		var nextCandleET time.Time
		var remaining int64

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
			runTimeEt = nextCandleET.In(eastern).Format("15:04")
			runTimeUtc = nextCandleET.UTC().Format("15:04")
		}

		row := &schedulerv1.ExchangeScheduleRow{
			Exchange:             sched.Name,
			Symbol:               sym,
			Region:               region,
			RunTimeEt:            runTimeEt,
			RunTimeUtc:           runTimeUtc,
			LocalClose:           alignment, // repurposed: shows candle alignment
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
		return results
	}
	defer rows.Close()

	for rows.Next() {
		var exchange string
		var day time.Time
		var startTS, endTS time.Time
		if err := rows.Scan(&exchange, &day, &startTS, &endTS); err != nil {
			continue
		}
		results[exchange] = lastRunInfo{
			day:   day.Format("2006-01-02"),
			start: startTS.UTC().Format(time.RFC3339),
			end:   endTS.UTC().Format(time.RFC3339),
		}
	}

	return results
}

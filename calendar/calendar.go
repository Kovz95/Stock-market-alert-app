// Package calendar provides exchange market hours validation and DST handling.
// It determines whether a given exchange is open at a specific time,
// accounting for time zones and daylight saving transitions.
package calendar

import (
	"fmt"
	"log"
	"time"

	_ "time/tzdata" // embed IANA database so LoadLocation works in minimal containers
)

// Eastern timezone (America/New_York). Loaded at init.
var eastern *time.Location

func init() {
	var err error
	eastern, err = time.LoadLocation("America/New_York")
	if err != nil {
		// Never fall back to UTC: that would make "4:40 PM ET" be stored as 16:40 UTC,
		// which displays as 12:20 PM ET (EST) and mis-schedules daily runs by ~5 hours.
		log.Printf("calendar: LoadLocation America/New_York failed: %v; using EST fixed offset", err)
		eastern = time.FixedZone("EST", -5*3600) // EST fallback so ET times are not treated as UTC
	}
}

// isLocationInDST returns whether the location is in DST at ref. Returns nil if unknown.
func isLocationInDST(tzName string, ref time.Time) *bool {
	loc, err := time.LoadLocation(tzName)
	if err != nil {
		return nil
	}
	// Compare offset at ref with offset in mid-winter (Jan) and mid-summer (Jul).
	jan := time.Date(ref.Year(), 1, 15, 12, 0, 0, 0, loc)
	jul := time.Date(ref.Year(), 7, 15, 12, 0, 0, 0, loc)
	_, offJan := jan.Zone()
	_, offJul := jul.Zone()
	_, offNow := ref.In(loc).Zone()
	// If winter and summer offsets differ, DST is used. Current is DST iff it matches summer.
	if offJan != offJul {
		b := offNow == offJul
		return &b
	}
	b := false
	return &b
}

// IsDSTActive reports whether US Eastern is currently in Daylight Saving Time.
func IsDSTActive() bool {
	return isDSTActiveAt(time.Now())
}

func isDSTActiveAt(t time.Time) bool {
	et := t.In(eastern)
	_, offset := et.Zone()
	// EST = -5*3600, EDT = -4*3600
	return offset == -4*3600
}

// IsInDSTTransitionPeriod reports whether US and European DST are misaligned:
// - 2nd Sunday in March to last Sunday in March (US in EDT, Europe still standard)
// - Last Sunday in October to 1st Sunday in November (Europe back to standard, US still EDT)
func IsInDSTTransitionPeriod() bool {
	return isInDSTTransitionPeriodAt(time.Now())
}

func isInDSTTransitionPeriodAt(t time.Time) bool {
	usDST := isDSTActiveAt(t)
	if !usDST {
		return false
	}
	// UK and Germany switch on same dates (last Sun Mar/Oct). Check Europe/London.
	london, err := time.LoadLocation("Europe/London")
	if err != nil {
		return false
	}
	uk := t.In(london)
	_, ukOffset := uk.Zone()
	// GMT = 0, BST = +1
	ukDST := ukOffset == 3600
	return usDST && !ukDST
}

// GetExchangeCloseTime returns the close time (hour, minute) in Eastern for the given exchange
// at ref, accounting for EST/EDT and DST transition overrides.
func GetExchangeCloseTime(exchange string, ref time.Time) (hour, minute int) {
	sched, ok := ExchangeSchedules[exchange]
	if !ok {
		return 16, 40
	}
	useEDT := isDSTActiveAt(ref)
	if !useEDT {
		return sched.ESTCloseHour, sched.ESTCloseMinute
	}
	hour = sched.EDTCloseHour
	minute = sched.EDTCloseMinute
	override := sched.DSTOverride
	if override == nil {
		return hour, minute
	}

	refET := ref.In(eastern)
	currentDate := refET
	inTransition := isInDSTTransitionPeriodAt(ref)

	tzName := sched.Timezone
	if tzName == "" {
		tzName = ExchangeTimezones[exchange]
	}
	var isLocalDST *bool
	if tzName != "" {
		if dst := isLocationInDST(tzName, ref); dst != nil {
			isLocalDST = dst
		}
	}

	apply := false
	switch override.Condition {
	case "both_dst":
		if isLocalDST == nil {
			apply = useEDT
		} else {
			apply = useEDT && *isLocalDST
		}
	case "misaligned_dst":
		if isLocalDST == nil {
			apply = inTransition
		} else {
			apply = useEDT != *isLocalDST
		}
	default:
		if len(override.Periods) > 0 {
			for _, p := range override.Periods {
				if IsDateInPeriod(currentDate, p.StartMonth, p.StartDay, p.EndMonth, p.EndDay) {
					apply = true
					break
				}
			}
		} else if isLocalDST != nil {
			apply = useEDT != *isLocalDST
		} else {
			apply = inTransition
		}
	}
	if apply {
		hour = override.EDTCloseHour
		minute = override.EDTCloseMin
	}
	return hour, minute
}

// GetHourlyAlignment returns the hourly alignment for the exchange: "hour", "quarter", or "half".
func GetHourlyAlignment(exchange string) string {
	if a, ok := HourlyAlignment[exchange]; ok {
		return a
	}
	return "hour"
}

// GetCalendarTimezone returns the IANA timezone for the exchange, or empty string if unknown.
func GetCalendarTimezone(exchange string) string {
	if sched, ok := ExchangeSchedules[exchange]; ok && sched.Timezone != "" {
		return sched.Timezone
	}
	if tz, ok := ExchangeTimezones[exchange]; ok {
		return tz
	}
	// US exchanges
	switch exchange {
	case ExchangeNYSE, ExchangeNASDAQ, ExchangeNYSEAmerican, ExchangeNYSEArca, ExchangeCBOEBZX:
		return "America/New_York"
	case ExchangeToronto:
		return "America/Toronto"
	case ExchangeMexico:
		return "America/Mexico_City"
	case ExchangeSaoPaulo:
		return "America/Sao_Paulo"
	case ExchangeBuenosAires:
		return "America/Argentina/Buenos_Aires"
	case ExchangeSantiago:
		return "America/Santiago"
	case ExchangeTokyo:
		return "Asia/Tokyo"
	case ExchangeHongKong:
		return "Asia/Hong_Kong"
	case ExchangeSingapore:
		return "Asia/Singapore"
	case ExchangeTaiwan:
		return "Asia/Taipei"
	case ExchangeNSEIndia, ExchangeBSEIndia:
		return "Asia/Kolkata"
	case ExchangeIndonesia:
		return "Asia/Jakarta"
	case ExchangeThailand:
		return "Asia/Bangkok"
	case ExchangeMalaysia:
		return "Asia/Kuala_Lumpur"
	case ExchangeIstanbul:
		return "Europe/Istanbul"
	case ExchangeJSE:
		return "Africa/Johannesburg"
	}
	return ""
}

// Local session open/close for IsExchangeOpen fallback (hour, minute in exchange local time).
var localSessionOpen = map[string]struct{ H, M int }{
	ExchangeLondon: {8, 0},
	ExchangeXetra:  {8, 0},
	ExchangeTokyo:  {9, 0},
}
var localSessionClose = map[string]struct{ H, M int }{
	ExchangeLondon: {16, 30},
	ExchangeXetra:  {16, 30},
	ExchangeTokyo:  {15, 0},
}

const defaultOpenH, defaultOpenM = 9, 30
const defaultCloseH, defaultCloseM = 16, 0

// IsExchangeOpen reports whether the exchange is open at t (schedule-based fallback: weekday + local open/close).
// t should be the current time as a UTC instant (e.g. time.Now().UTC()); it is converted to the exchange's
// local timezone to check session hours.
func IsExchangeOpen(exchange string, t time.Time) bool {
	_, _, open := isExchangeOpenDetail(exchange, t)
	return open
}

// ExchangeOpenStatus returns a short diagnostic string for why an exchange is open or closed at t.
// Useful for debugging UTC vs local timing: e.g. "NYSE local=2025-03-04T15:00:00-04:00 session=09:30-16:00 => open".
func ExchangeOpenStatus(exchange string, t time.Time) string {
	localStr, sessionStr, open := isExchangeOpenDetail(exchange, t)
	if open {
		return localStr + " " + sessionStr + " => open"
	}
	return localStr + " " + sessionStr + " => closed"
}

// isExchangeOpenDetail returns local time string, session string, and open result.
func isExchangeOpenDetail(exchange string, t time.Time) (localStr, sessionStr string, open bool) {
	if _, ok := ExchangeSchedules[exchange]; !ok {
		return "", "", false
	}
	tzName := GetCalendarTimezone(exchange)
	if tzName == "" {
		tzName = "UTC"
	}
	loc, err := time.LoadLocation(tzName)
	if err != nil {
		return "tz_error=" + tzName, "", false
	}
	// Normalize to UTC instant so wall clock is unambiguous, then convert to exchange local.
	utc := t.UTC()
	local := utc.In(loc)
	wd := local.Weekday()
	if wd == time.Saturday || wd == time.Sunday {
		localStr := "utc=" + utc.Format(time.RFC3339) + " local=" + local.Format(time.RFC3339) + " weekday=" + wd.String()
		return localStr, "", false
	}
	openH, openM := defaultOpenH, defaultOpenM
	if o, ok := localSessionOpen[exchange]; ok {
		openH, openM = o.H, o.M
	}
	closeH, closeM := defaultCloseH, defaultCloseM
	if c, ok := localSessionClose[exchange]; ok {
		closeH, closeM = c.H, c.M
	}
	openT := time.Date(local.Year(), local.Month(), local.Day(), openH, openM, 0, 0, loc)
	closeT := time.Date(local.Year(), local.Month(), local.Day(), closeH, closeM, 0, 0, loc)
	open = !local.Before(openT) && local.Before(closeT)
	localStr = "utc=" + utc.Format(time.RFC3339) + " " + exchange + "_local=" + local.Format(time.RFC3339)
	sessionStr = "session=" + fmtHHMM(openH, openM) + "-" + fmtHHMM(closeH, closeM)
	return localStr, sessionStr, open
}

func fmtHHMM(h, m int) string {
	return fmt.Sprintf("%02d:%02d", h, m)
}

// GetNextDailyRunTime returns the next run time (session close + 40 minutes) in Eastern for the exchange.
// ref is the reference time; the returned time is in UTC.
func GetNextDailyRunTime(exchange string, ref time.Time) time.Time {
	if _, ok := ExchangeSchedules[exchange]; !ok {
		return ref.Add(24 * time.Hour)
	}
	refET := ref.In(eastern)

	// Start from today, skipping weekend days (no market on Sat/Sun).
	candidate := refET
	for candidate.Weekday() == time.Saturday || candidate.Weekday() == time.Sunday {
		candidate = candidate.AddDate(0, 0, 1)
	}

	hour, minute := GetExchangeCloseTime(exchange, candidate)
	closeET := time.Date(candidate.Year(), candidate.Month(), candidate.Day(), hour, minute, 0, 0, eastern)
	runET := closeET.Add(40 * time.Minute)

	// Advance to the next business day only after the run time (close+40min) has passed.
	if !refET.Before(runET) {
		nextDay := candidate.AddDate(0, 0, 1)
		for nextDay.Weekday() == time.Saturday || nextDay.Weekday() == time.Sunday {
			nextDay = nextDay.AddDate(0, 0, 1)
		}
		hour, minute = GetExchangeCloseTime(exchange, nextDay)
		closeET = time.Date(nextDay.Year(), nextDay.Month(), nextDay.Day(), hour, minute, 0, 0, eastern)
		runET = closeET.Add(40 * time.Minute)
	}

	return runET.UTC()
}

// ExpectedLastTradingDate returns the calendar date (as UTC midnight) of the most recent trading day
// that has closed for the exchange (i.e. we expect daily data to be available for this date).
// Uses the same close+40min convention as GetNextDailyRunTime.
func ExpectedLastTradingDate(exchange string, now time.Time) time.Time {
	if _, ok := ExchangeSchedules[exchange]; !ok {
		return now.UTC().Truncate(24 * time.Hour)
	}
	refET := now.In(eastern)
	candidate := refET
	for candidate.Weekday() == time.Saturday || candidate.Weekday() == time.Sunday {
		candidate = candidate.AddDate(0, 0, 1)
	}
	hour, minute := GetExchangeCloseTime(exchange, candidate)
	closeET := time.Date(candidate.Year(), candidate.Month(), candidate.Day(), hour, minute, 0, 0, eastern)
	runET := closeET.Add(40 * time.Minute)
	var lastCompleted time.Time
	if refET.Before(runET) {
		// Today's run hasn't happened; last completed = previous trading day
		prev := candidate.AddDate(0, 0, -1)
		for prev.Weekday() == time.Saturday || prev.Weekday() == time.Sunday {
			prev = prev.AddDate(0, 0, -1)
		}
		lastCompleted = prev
	} else {
		lastCompleted = candidate
	}
	return time.Date(lastCompleted.Year(), lastCompleted.Month(), lastCompleted.Day(), 0, 0, 0, 0, time.UTC)
}

// ExpectedLastWeekEnding returns the calendar date (UTC midnight) of the most recent completed
// week-ending Friday for the exchange. After Friday's close+40min we consider that week complete.
func ExpectedLastWeekEnding(exchange string, now time.Time) time.Time {
	if _, ok := ExchangeSchedules[exchange]; !ok {
		return now.UTC().Truncate(24 * time.Hour)
	}
	refET := now.In(eastern)
	friday := refET
	for friday.Weekday() != time.Friday {
		friday = friday.AddDate(0, 0, -1)
	}
	hour, minute := GetExchangeCloseTime(exchange, friday)
	closeET := time.Date(friday.Year(), friday.Month(), friday.Day(), hour, minute, 0, 0, eastern)
	runET := closeET.Add(40 * time.Minute)
	if refET.Before(runET) {
		friday = friday.AddDate(0, 0, -7)
	}
	return time.Date(friday.Year(), friday.Month(), friday.Day(), 0, 0, 0, 0, time.UTC)
}

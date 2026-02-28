// Package calendar provides exchange market hours validation and DST handling.
// It determines whether a given exchange is open at a specific time,
// accounting for time zones and daylight saving transitions.
package calendar

import (
	"time"
)

// Eastern timezone (America/New_York). Loaded at init.
var eastern *time.Location

func init() {
	var err error
	eastern, err = time.LoadLocation("America/New_York")
	if err != nil {
		eastern = time.UTC
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
func IsExchangeOpen(exchange string, t time.Time) bool {
	if _, ok := ExchangeSchedules[exchange]; !ok {
		return false
	}
	tzName := GetCalendarTimezone(exchange)
	if tzName == "" {
		tzName = "UTC"
	}
	loc, err := time.LoadLocation(tzName)
	if err != nil {
		return false
	}
	local := t.In(loc)
	wd := local.Weekday()
	if wd == time.Saturday || wd == time.Sunday {
		return false
	}
	openH, openM := defaultOpenH, defaultOpenM
	if o, ok := localSessionOpen[exchange]; ok {
		openH, openM = o.H, o.M
	}
	closeH, closeM := defaultCloseH, defaultCloseM
	if c, ok := localSessionClose[exchange]; ok {
		closeH, closeM = c.H, c.M
	}
	open := time.Date(local.Year(), local.Month(), local.Day(), openH, openM, 0, 0, loc)
	close := time.Date(local.Year(), local.Month(), local.Day(), closeH, closeM, 0, 0, loc)
	return !local.Before(open) && local.Before(close)
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

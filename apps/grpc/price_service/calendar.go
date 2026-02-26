package main

import (
	"time"
)

// expectedLastTradingDayUTC returns the expected last trading day (date only, UTC)
// using US market convention: after 21:00 UTC (≈4pm ET) we expect today's bar;
// before that, yesterday; weekends use previous Friday.
func expectedLastTradingDayUTC(now time.Time) time.Time {
	y, m, d := now.Date()
	today := time.Date(y, m, d, 0, 0, 0, 0, time.UTC)
	weekday := now.Weekday() // 0=Sun, 4=Fri, 6=Sat
	hour := now.Hour() + now.Minute()/60

	if weekday == 0 { // Sunday -> Friday
		return today.AddDate(0, 0, -2)
	}
	if weekday == 6 { // Saturday -> Friday
		return today.AddDate(0, 0, -1)
	}
	// Monday=1 .. Friday=5
	if hour >= 21 { // After ~4pm ET
		return today
	}
	// Before close: expect yesterday
	if weekday == 1 { // Monday -> Friday
		return today.AddDate(0, 0, -3)
	}
	return today.AddDate(0, 0, -1)
}

// expectedWeekEndingUTC returns the expected last completed week-ending (Friday)
// in UTC. After Friday 21:40 UTC we consider this week's Friday complete.
func expectedWeekEndingUTC(now time.Time) time.Time {
	y, m, d := now.Date()
	today := time.Date(y, m, d, 0, 0, 0, 0, time.UTC)
	weekday := now.Weekday() // 0=Sun, 1=Mon, ..., 5=Fri, 6=Sat

	if weekday == 6 { // Saturday -> previous Friday
		return today.AddDate(0, 0, -1)
	}
	if weekday == 0 { // Sunday -> previous Friday
		return today.AddDate(0, 0, -2)
	}
	if weekday == 5 { // Friday: after 21:40 UTC then today else previous Friday
		if now.Hour() > 21 || (now.Hour() == 21 && now.Minute() >= 40) {
			return today
		}
		return today.AddDate(0, 0, -7)
	}
	// Mon=1: -3, Tue=2: -4, Wed=3: -5, Thu=4: -6
	daysBack := int(weekday) + 3
	return today.AddDate(0, 0, -daysBack)
}

// referenceHourUTC returns a reference "expected" hour for hourly staleness:
// the latest non-future hour (truncate to hour, cap at now).
func referenceHourUTC(now time.Time) time.Time {
	return now.UTC().Truncate(time.Hour)
}

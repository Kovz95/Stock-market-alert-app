package main

import "time"

// referenceHourUTC returns a reference "expected" hour for hourly staleness:
// the latest non-future hour (truncate to hour, cap at now).
// Exchange-agnostic; for exchange-aware session logic use stockalert/calendar.
func referenceHourUTC(now time.Time) time.Time {
	return now.UTC().Truncate(time.Hour)
}

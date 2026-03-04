package calendar

import (
	"strings"
	"testing"
	"time"
)

func TestIsDSTActive(t *testing.T) {
	// Mid-January: EST (no DST)
	jan := time.Date(2025, 1, 15, 12, 0, 0, 0, time.UTC)
	if isDSTActiveAt(jan) {
		t.Error("January should not be DST in Eastern")
	}
	// Mid-July: EDT
	jul := time.Date(2025, 7, 15, 12, 0, 0, 0, time.UTC)
	if !isDSTActiveAt(jul) {
		t.Error("July should be DST in Eastern")
	}
}

func TestIsInDSTTransitionPeriod(t *testing.T) {
	// March 15, 2025: US already EDT (2nd Sun March 2025 = 9th), Europe still CET -> misaligned
	march := time.Date(2025, 3, 15, 12, 0, 0, 0, time.UTC)
	if !isInDSTTransitionPeriodAt(march) {
		t.Error("March 15 should be in DST transition (US EDT, Europe not yet)")
	}
	// April: both in DST -> not transition
	april := time.Date(2025, 4, 15, 12, 0, 0, 0, time.UTC)
	if isInDSTTransitionPeriodAt(april) {
		t.Error("April should not be in DST transition")
	}
	// Late October 2025: Europe back to standard, US still EDT -> misaligned
	oct := time.Date(2025, 10, 28, 12, 0, 0, 0, time.UTC)
	if !isInDSTTransitionPeriodAt(oct) {
		t.Error("Oct 28 should be in DST transition (Europe standard, US EDT)")
	}
	// December: neither in DST -> not transition
	dec := time.Date(2025, 12, 1, 12, 0, 0, 0, time.UTC)
	if isInDSTTransitionPeriodAt(dec) {
		t.Error("December should not be in DST transition")
	}
}

func TestGetExchangeCloseTime(t *testing.T) {
	// US exchange: same 16:40 EST and EDT
	refEST := time.Date(2025, 1, 15, 18, 0, 0, 0, time.UTC) // 1 PM ET
	refEDT := time.Date(2025, 7, 15, 18, 0, 0, 0, time.UTC) // 1 PM ET
	h, m := GetExchangeCloseTime(ExchangeNYSE, refEST)
	if h != 16 || m != 40 {
		t.Errorf("NYSE close in EST: got %d:%02d, want 16:40", h, m)
	}
	h, m = GetExchangeCloseTime(ExchangeNYSE, refEDT)
	if h != 16 || m != 40 {
		t.Errorf("NYSE close in EDT: got %d:%02d, want 16:40", h, m)
	}

	// London: 11:50 normally, 12:50 during transition
	refMarch := time.Date(2025, 3, 15, 15, 0, 0, 0, time.UTC) // 11 AM ET, in transition
	h, m = GetExchangeCloseTime(ExchangeLondon, refMarch)
	if h != 13 || m != 10 {
		t.Errorf("LONDON close during transition: got %d:%02d, want 13:10", h, m)
	}
	refApril := time.Date(2025, 4, 15, 15, 0, 0, 0, time.UTC)
	h, m = GetExchangeCloseTime(ExchangeLondon, refApril)
	if h != 12 || m != 10 {
		t.Errorf("LONDON close in April: got %d:%02d, want 12:10", h, m)
	}
}

func TestGetHourlyAlignment(t *testing.T) {
	tests := []struct {
		exchange string
		want     string
	}{
		{ExchangeNYSE, "hour"},
		{ExchangeBSEIndia, "quarter"},
		{ExchangeNSEIndia, "quarter"},
		{ExchangeHongKong, "half"},
		{ExchangeEuronextParis, "half"},
		{ExchangeAthens, "half"},
		{ExchangeOMXNordicIceland, "half"},
		{"UNKNOWN", "hour"},
	}
	for _, tt := range tests {
		got := GetHourlyAlignment(tt.exchange)
		if got != tt.want {
			t.Errorf("GetHourlyAlignment(%q) = %q, want %q", tt.exchange, got, tt.want)
		}
	}
}

func TestGetCalendarTimezone(t *testing.T) {
	tests := []struct {
		exchange string
		want     string
	}{
		{ExchangeNYSE, "America/New_York"},
		{ExchangeLondon, "Europe/London"},
		{ExchangeTokyo, "Asia/Tokyo"},
		{ExchangeBucharestSpot, "Europe/Bucharest"},
		{ExchangeColombia, "America/Bogota"},
	}
	for _, tt := range tests {
		got := GetCalendarTimezone(tt.exchange)
		if got != tt.want {
			t.Errorf("GetCalendarTimezone(%q) = %q, want %q", tt.exchange, got, tt.want)
		}
	}
}

// TestGetCalendarTimezone_AllExchanges ensures every exchange has a non-empty IANA timezone
// so IsExchangeOpen can convert UTC to local time correctly.
func TestGetCalendarTimezone_AllExchanges(t *testing.T) {
	for exchange := range ExchangeSchedules {
		tz := GetCalendarTimezone(exchange)
		if tz == "" {
			t.Errorf("exchange %q: GetCalendarTimezone returned empty string", exchange)
		}
		// LoadLocation must succeed for IsExchangeOpen to work
		loc, err := time.LoadLocation(tz)
		if err != nil {
			t.Errorf("exchange %q: timezone %q LoadLocation failed: %v", exchange, tz, err)
		}
		if loc == nil {
			t.Errorf("exchange %q: LoadLocation returned nil", exchange)
		}
	}
}

func TestIsExchangeOpen(t *testing.T) {
	// Monday 10:00 AM New York -> NYSE open
	mon := time.Date(2025, 1, 6, 15, 0, 0, 0, time.UTC) // 10 AM ET
	if !IsExchangeOpen(ExchangeNYSE, mon) {
		t.Error("NYSE should be open Monday 10 AM ET")
	}
	// Saturday
	sat := time.Date(2025, 1, 4, 15, 0, 0, 0, time.UTC)
	if IsExchangeOpen(ExchangeNYSE, sat) {
		t.Error("NYSE should be closed Saturday")
	}
	// Sunday
	sun := time.Date(2025, 1, 5, 15, 0, 0, 0, time.UTC)
	if IsExchangeOpen(ExchangeNYSE, sun) {
		t.Error("NYSE should be closed Sunday")
	}
	// Before open (7 AM ET)
	early := time.Date(2025, 1, 6, 12, 0, 0, 0, time.UTC)
	if IsExchangeOpen(ExchangeNYSE, early) {
		t.Error("NYSE should be closed at 7 AM ET")
	}
	// Tuesday 3:00 PM ET (EDT in March) = 19:00 UTC -> NYSE open
	marchOpen := time.Date(2025, 3, 4, 19, 0, 0, 0, time.UTC)
	if !IsExchangeOpen(ExchangeNYSE, marchOpen) {
		t.Error("NYSE should be open Tuesday 3 PM ET (19:00 UTC in March)")
	}
}

func TestExchangeOpenStatus(t *testing.T) {
	// Monday 10 AM ET -> open; status should contain "=> open"
	mon := time.Date(2025, 1, 6, 15, 0, 0, 0, time.UTC)
	status := ExchangeOpenStatus(ExchangeNYSE, mon)
	if status == "" {
		t.Error("ExchangeOpenStatus should not be empty")
	}
	if !IsExchangeOpen(ExchangeNYSE, mon) {
		t.Error("NYSE should be open Monday 10 AM ET")
	}
	if status != "" && status[len(status)-7:] != "=> open" {
		t.Errorf("ExchangeOpenStatus for open NYSE should end with '=> open', got %q", status)
	}
	// Saturday -> closed
	sat := time.Date(2025, 1, 4, 15, 0, 0, 0, time.UTC)
	statusSat := ExchangeOpenStatus(ExchangeNYSE, sat)
	if statusSat != "" && statusSat[len(statusSat)-9:] != "=> closed" {
		t.Errorf("ExchangeOpenStatus for Saturday should end with '=> closed', got %q", statusSat)
	}
}

// TestIsExchangeOpen_AllExchanges calls IsExchangeOpen for every exchange at a few fixed
// UTC times to ensure no panic and consistent behavior (weekday vs weekend, open vs closed).
func TestIsExchangeOpen_AllExchanges(t *testing.T) {
	// Tuesday 12:00 UTC: Europe afternoon, US morning, Asia evening — mix of open/closed.
	tueNoonUTC := time.Date(2025, 3, 4, 12, 0, 0, 0, time.UTC)
	// Saturday same time — all should be closed (weekend).
	satNoonUTC := time.Date(2025, 3, 1, 12, 0, 0, 0, time.UTC)

	for exchange := range ExchangeSchedules {
		// Weekday: no panic; result is either open or closed
		_ = IsExchangeOpen(exchange, tueNoonUTC)
		// Saturday: all exchanges should be closed
		openSat := IsExchangeOpen(exchange, satNoonUTC)
		if openSat {
			t.Errorf("exchange %q: should be closed on Saturday 12:00 UTC", exchange)
		}
	}
	// Sanity: at least one exchange open on Tuesday noon UTC (e.g. European)
	openCount := 0
	for exchange := range ExchangeSchedules {
		if IsExchangeOpen(exchange, tueNoonUTC) {
			openCount++
		}
	}
	if openCount == 0 {
		t.Error("at least one exchange should be open on Tuesday 12:00 UTC (e.g. European)")
	}
}

// TestExchangeOpenStatus_AllExchanges ensures ExchangeOpenStatus returns a valid string
// for every exchange (no panic, ends with "=> open" or "=> closed").
func TestExchangeOpenStatus_AllExchanges(t *testing.T) {
	tueNoonUTC := time.Date(2025, 3, 4, 12, 0, 0, 0, time.UTC)
	for exchange := range ExchangeSchedules {
		status := ExchangeOpenStatus(exchange, tueNoonUTC)
		if status == "" {
			t.Errorf("exchange %q: ExchangeOpenStatus returned empty string", exchange)
		}
		if !strings.HasSuffix(status, "=> open") && !strings.HasSuffix(status, "=> closed") {
			t.Errorf("exchange %q: ExchangeOpenStatus should end with '=> open' or '=> closed', got %q", exchange, status)
		}
	}
}

func TestGetNextDailyRunTime(t *testing.T) {
	// Tuesday 10 AM ET -> next run is same day 16:40 + 40 min = 17:20 ET
	tue := time.Date(2025, 1, 7, 15, 0, 0, 0, time.UTC) // 10 AM ET
	run := GetNextDailyRunTime(ExchangeNYSE, tue)
	runET := run.In(eastern)
	if runET.Hour() != 17 || runET.Minute() != 20 {
		t.Errorf("Next run for NYSE: got %d:%02d ET, want 17:20", runET.Hour(), runET.Minute())
	}
	if runET.YearDay() != tue.In(eastern).YearDay() {
		t.Error("Next run should be same calendar day for NYSE when before close")
	}
}

func TestIsDateInPeriod(t *testing.T) {
	// Period March 10 - March 20 (no wrap)
	d := time.Date(2025, 3, 15, 12, 0, 0, 0, time.UTC)
	if !IsDateInPeriod(d, 3, 10, 3, 20) {
		t.Error("March 15 should be in March 10-20")
	}
	d = time.Date(2025, 3, 5, 12, 0, 0, 0, time.UTC)
	if IsDateInPeriod(d, 3, 10, 3, 20) {
		t.Error("March 5 should not be in March 10-20")
	}
	// Wrap-around: Dec 20 - Jan 10
	d = time.Date(2025, 12, 25, 12, 0, 0, 0, time.UTC)
	if !IsDateInPeriod(d, 12, 20, 1, 10) {
		t.Error("Dec 25 should be in Dec 20 - Jan 10")
	}
	d = time.Date(2025, 1, 5, 12, 0, 0, 0, time.UTC)
	if !IsDateInPeriod(d, 12, 20, 1, 10) {
		t.Error("Jan 5 should be in Dec 20 - Jan 10")
	}
	d = time.Date(2025, 1, 15, 12, 0, 0, 0, time.UTC)
	if IsDateInPeriod(d, 12, 20, 1, 10) {
		t.Error("Jan 15 should not be in Dec 20 - Jan 10")
	}
}

func TestExchangeCount(t *testing.T) {
	// 42 exchanges per plan; config may include a couple more (e.g. BSE INDIA, BUCHAREST SPOT, COLOMBIA)
	if n := len(ExchangeSchedules); n < 42 {
		t.Errorf("expected at least 42 exchanges, got %d", n)
	}
}

func TestExpectedLastTradingDate(t *testing.T) {
	// Tuesday 10 AM ET: before close -> last completed = Monday
	tue10amET := time.Date(2025, 1, 7, 15, 0, 0, 0, time.UTC) // 10 AM ET
	got := ExpectedLastTradingDate(ExchangeNYSE, tue10amET)
	// Monday Jan 6, 2025
	if got.Year() != 2025 || got.Month() != 1 || got.Day() != 6 {
		t.Errorf("ExpectedLastTradingDate(Tue 10am ET) = %s, want 2025-01-06", got.Format("2006-01-02"))
	}
	// Tuesday 6 PM ET: after close+40 -> last completed = Tuesday
	tue6pmET := time.Date(2025, 1, 7, 23, 0, 0, 0, time.UTC) // 6 PM ET
	got2 := ExpectedLastTradingDate(ExchangeNYSE, tue6pmET)
	if got2.Year() != 2025 || got2.Month() != 1 || got2.Day() != 7 {
		t.Errorf("ExpectedLastTradingDate(Tue 6pm ET) = %s, want 2025-01-07", got2.Format("2006-01-02"))
	}
}

func TestExpectedLastWeekEnding(t *testing.T) {
	// Friday before close -> previous Friday
	friNoonET := time.Date(2025, 1, 10, 17, 0, 0, 0, time.UTC) // Fri noon ET
	got := ExpectedLastWeekEnding(ExchangeNYSE, friNoonET)
	if got.Year() != 2025 || got.Month() != 1 || got.Day() != 3 {
		t.Errorf("ExpectedLastWeekEnding(Fri noon ET) = %s, want 2025-01-03", got.Format("2006-01-02"))
	}
	// Friday after close+40 -> this Friday
	fri6pmET := time.Date(2025, 1, 11, 0, 0, 0, 0, time.UTC) // Fri 7pm ET = after 16:40+40
	got2 := ExpectedLastWeekEnding(ExchangeNYSE, fri6pmET)
	if got2.Year() != 2025 || got2.Month() != 1 || got2.Day() != 10 {
		t.Errorf("ExpectedLastWeekEnding(Fri 7pm ET) = %s, want 2025-01-10", got2.Format("2006-01-02"))
	}
}

func TestIsExchangeOpenFromSnapshot(t *testing.T) {
	// Nil snapshot -> false
	if IsExchangeOpenFromSnapshot(ExchangeNYSE, nil) {
		t.Error("nil snapshot should return false")
	}
	// Empty snapshot -> false
	snap := &MarketHoursSnapshot{ByExchange: map[string]FMPExchangeHours{}}
	if IsExchangeOpenFromSnapshot(ExchangeNYSE, snap) {
		t.Error("empty snapshot should return false")
	}
	// NYSE in snapshot, open -> true
	snap.ByExchange[ExchangeNYSE] = FMPExchangeHours{Exchange: "NYSE", IsMarketOpen: true}
	if !IsExchangeOpenFromSnapshot(ExchangeNYSE, snap) {
		t.Error("NYSE open in snapshot should return true")
	}
	// NYSE in snapshot, closed -> false
	snap.ByExchange[ExchangeNYSE] = FMPExchangeHours{Exchange: "NYSE", IsMarketOpen: false}
	if IsExchangeOpenFromSnapshot(ExchangeNYSE, snap) {
		t.Error("NYSE closed in snapshot should return false")
	}
	// Unknown exchange in snapshot -> false
	if IsExchangeOpenFromSnapshot("UNKNOWN_EXCHANGE", snap) {
		t.Error("unknown exchange should return false")
	}
}

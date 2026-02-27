package calendar

import (
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

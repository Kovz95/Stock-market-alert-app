package price

import (
	"fmt"
	"time"

	"stockalert/calendar"
)

// ExchangeRepresentativeTicker maps each exchange to a liquid FMP ticker used
// to probe current hourly candle timing. The ticker must be traded on that
// exchange so its candle boundaries reflect actual market hours.
//
// Note: FMP ticker formats vary by market (e.g. ".L" suffix for LSE, ".DE"
// for Xetra). Verify against the FMP symbol search if a ticker stops returning
// data.
var ExchangeRepresentativeTicker = map[string]string{
	// North America
	calendar.ExchangeNYSE:           "AAPL",
	calendar.ExchangeNASDAQ:         "AAPL",
	calendar.ExchangeNYSEAmerican:   "SPY",
	calendar.ExchangeNYSEArca:       "SPY",
	calendar.ExchangeCBOEBZX:        "SPY",
	calendar.ExchangeToronto:        "SHOP.TO",
	calendar.ExchangeSaoPaulo:       "PETR4.SA",
	calendar.ExchangeMexico:         "AMX.MX",
	calendar.ExchangeBuenosAires:    "GGAL.BA",
	calendar.ExchangeSantiago:       "ITAUCL.SN",

	// Europe
	calendar.ExchangeLondon:              "SHEL.L",
	calendar.ExchangeEuronextDublin:      "CRH.IE",
	calendar.ExchangeXetra:               "SAP.DE",
	calendar.ExchangeEuronextParis:       "MC.PA",
	calendar.ExchangeEuronextAmsterdam:   "ASML.AS",
	calendar.ExchangeEuronextBrussels:    "AB.BR",
	calendar.ExchangeEuronextLisbon:      "EDP.LS",
	calendar.ExchangeMilan:               "ENI.MI",
	calendar.ExchangeSpain:               "ITX.MC",
	calendar.ExchangeSixSwiss:            "NESN.SW",
	calendar.ExchangeVienna:              "OMV.VI",
	calendar.ExchangeOMXNordicStockholm:  "ERIC-B.ST",
	calendar.ExchangeOMXNordicCopenhagen: "NOVO-B.CO",
	calendar.ExchangeOMXNordicHelsinki:   "NOKIA.HE",
	calendar.ExchangeOMXNordicIceland:    "MAREL.IC",
	calendar.ExchangeOslo:                "EQNR.OL",
	calendar.ExchangeWarsaw:              "PKN.WA",
	calendar.ExchangePrague:              "CEZ.PR",
	calendar.ExchangeBudapest:            "OTP.BD",
	calendar.ExchangeBucharestSpot:       "TLV.RO",
	calendar.ExchangeAthens:              "OPAP.AT",
	calendar.ExchangeIstanbul:            "GARAN.IS",

	// Africa
	calendar.ExchangeJSE: "NPN.JO",

	// Asia / Pacific
	calendar.ExchangeTokyo:     "7203.T",
	calendar.ExchangeHongKong:  "0700.HK",
	calendar.ExchangeSingapore: "D05.SI",
	calendar.ExchangeASX:       "BHP.AX",
	calendar.ExchangeTaiwan:    "2330.TW",
	calendar.ExchangeNSEIndia:  "RELIANCE.NS",
	calendar.ExchangeBSEIndia:  "RELIANCE.BO",
	calendar.ExchangeIndonesia: "BBCA.JK",
	calendar.ExchangeThailand:  "AOT.BK",
	calendar.ExchangeMalaysia:  "PBBANK.KL",

	// Other
	calendar.ExchangeColombia: "PFBCOLOM.CL",
}

// fmpCandleTimezone is the IANA timezone name used to interpret FMP candle
// date strings for each exchange. FMP returns candle timestamps in the
// exchange's local time (not UTC).
var fmpCandleTimezone = map[string]string{
	calendar.ExchangeNYSE:                "America/New_York",
	calendar.ExchangeNASDAQ:              "America/New_York",
	calendar.ExchangeNYSEAmerican:        "America/New_York",
	calendar.ExchangeNYSEArca:            "America/New_York",
	calendar.ExchangeCBOEBZX:             "America/New_York",
	calendar.ExchangeToronto:             "America/New_York",
	calendar.ExchangeMexico:              "America/Mexico_City",
	calendar.ExchangeSaoPaulo:            "America/Sao_Paulo",
	calendar.ExchangeBuenosAires:         "America/Argentina/Buenos_Aires",
	calendar.ExchangeSantiago:            "America/Santiago",
	calendar.ExchangeColombia:            "America/Bogota",
	calendar.ExchangeLondon:              "Europe/London",
	calendar.ExchangeEuronextDublin:      "Europe/Dublin",
	calendar.ExchangeXetra:               "Europe/Berlin",
	calendar.ExchangeEuronextParis:       "Europe/Paris",
	calendar.ExchangeEuronextAmsterdam:   "Europe/Amsterdam",
	calendar.ExchangeEuronextBrussels:    "Europe/Brussels",
	calendar.ExchangeEuronextLisbon:      "Europe/Lisbon",
	calendar.ExchangeMilan:               "Europe/Rome",
	calendar.ExchangeSpain:               "Europe/Madrid",
	calendar.ExchangeSixSwiss:            "Europe/Zurich",
	calendar.ExchangeVienna:              "Europe/Vienna",
	calendar.ExchangeOMXNordicStockholm:  "Europe/Stockholm",
	calendar.ExchangeOMXNordicCopenhagen: "Europe/Copenhagen",
	calendar.ExchangeOMXNordicHelsinki:   "Europe/Helsinki",
	calendar.ExchangeOMXNordicIceland:    "Atlantic/Reykjavik",
	calendar.ExchangeOslo:                "Europe/Oslo",
	calendar.ExchangeWarsaw:              "Europe/Warsaw",
	calendar.ExchangePrague:              "Europe/Prague",
	calendar.ExchangeBudapest:            "Europe/Budapest",
	calendar.ExchangeBucharestSpot:       "Europe/Bucharest",
	calendar.ExchangeAthens:              "Europe/Athens",
	calendar.ExchangeIstanbul:            "Europe/Istanbul",
	calendar.ExchangeJSE:                 "Africa/Johannesburg",
	calendar.ExchangeTokyo:               "Asia/Tokyo",
	calendar.ExchangeHongKong:            "Asia/Hong_Kong",
	calendar.ExchangeSingapore:           "Asia/Singapore",
	calendar.ExchangeASX:                 "Australia/Sydney",
	calendar.ExchangeTaiwan:              "Asia/Taipei",
	calendar.ExchangeNSEIndia:            "Asia/Kolkata",
	calendar.ExchangeBSEIndia:            "Asia/Kolkata",
	calendar.ExchangeIndonesia:           "Asia/Jakarta",
	calendar.ExchangeThailand:            "Asia/Bangkok",
	calendar.ExchangeMalaysia:            "Asia/Kuala_Lumpur",
}

// NextHourlyCandleEnd detects the true candle duration for the exchange by
// examining the gaps between the most recent candles returned by FMP's
// /historical-chart/1hour endpoint, then returns when the next candle closes
// in UTC.
//
// Why look at multiple candles instead of assuming 1 hour:
//   - Some exchanges (e.g. Hong Kong, Euronext Paris) publish 30-minute bars.
//   - Lunch breaks create a 90+ minute gap between the last pre-lunch candle
//     and the first post-lunch candle, which would inflate a naïve 2-row diff.
//
// The standard candle duration is the minimum gap (≥ 15 min) found across the
// last candleLookback consecutive pairs. Lunch-break gaps are always ≥ 90 min
// and are never the minimum, so they are naturally filtered out.
//
// Returns an error if the exchange has no representative ticker, FMP returns
// fewer than candleLookback+1 rows, the dates cannot be parsed, or no valid
// gap is found.
const candleLookback = 5 // number of recent candles inspected to find the duration

func NextHourlyCandleEnd(exchange string, now time.Time, fetcher FMPFetcher) (time.Time, error) {
	ticker, ok := ExchangeRepresentativeTicker[exchange]
	if !ok {
		return time.Time{}, fmt.Errorf("no representative ticker configured for exchange %q", exchange)
	}

	rows, err := fetcher.FetchHourly(ticker)
	if err != nil {
		return time.Time{}, fmt.Errorf("FMP fetch %s (%s): %w", exchange, ticker, err)
	}
	need := candleLookback + 1
	if len(rows) < need {
		return time.Time{}, fmt.Errorf("FMP returned %d candle(s) for %s (%s), need ≥%d", len(rows), exchange, ticker, need)
	}

	loc := fmpTimezoneFor(exchange)

	// Parse the first (need) candle timestamps. FMP returns newest-first.
	times := make([]time.Time, need)
	for i := range times {
		t, err := parseHourlyCandleDate(rows[i].Date, loc)
		if err != nil {
			return time.Time{}, fmt.Errorf("parse candle[%d] date for %s: %w", i, exchange, err)
		}
		times[i] = t
	}

	// Find the standard candle duration: the smallest gap between consecutive
	// candles that is ≥ 15 min. Lunch-break gaps (90+ min) will never win.
	const minCandle = 15 * time.Minute
	candleDur := time.Duration(0)
	for i := 0; i < len(times)-1; i++ {
		gap := times[i].Sub(times[i+1]) // positive because newest-first
		if gap >= minCandle && (candleDur == 0 || gap < candleDur) {
			candleDur = gap
		}
	}
	if candleDur == 0 {
		return time.Time{}, fmt.Errorf("could not determine candle duration for %s: no gap ≥ %s found", exchange, minCandle)
	}

	// times[0] is the open of the most recently started candle.
	// It closes at times[0] + candleDur.
	candleEnd := times[0].Add(candleDur).UTC()

	if candleEnd.After(now) {
		return candleEnd, nil
	}
	// That candle has already closed; the next one closes one duration later.
	return candleEnd.Add(candleDur), nil
}

func fmpTimezoneFor(exchange string) *time.Location {
	tzName, ok := fmpCandleTimezone[exchange]
	if !ok {
		tzName = "America/New_York"
	}
	loc, err := time.LoadLocation(tzName)
	if err != nil {
		return time.UTC
	}
	return loc
}

func parseHourlyCandleDate(s string, loc *time.Location) (time.Time, error) {
	for _, format := range []string{
		"2006-01-02 15:04:05",
		"2006-01-02T15:04:05",
		time.RFC3339,
	} {
		if t, err := time.ParseInLocation(format, s, loc); err == nil {
			return t, nil
		}
	}
	return time.Time{}, fmt.Errorf("cannot parse %q as candle date", s)
}

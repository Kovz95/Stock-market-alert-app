// Package calendar defines exchange schedules, timezones, and hourly alignment.
package calendar

import "time"

// DSTOverride defines optional close-time overrides during DST transition periods.
// Condition may be "both_dst" (apply when US and local exchange are both in DST),
// "misaligned_dst" (apply when US/Europe DST is misaligned), or empty for date-period-based logic.
type DSTOverride struct {
	Condition    string       // "both_dst", "misaligned_dst", or "" for period-based
	EDTCloseHour int          // override hour when condition matches
	EDTCloseMin  int          // override minute when condition matches
	Periods      []DSTPeriod  // optional date ranges (e.g. March 2nd Sun - last Sun March)
}

// DSTPeriod defines a date range for DST override (month/day, 1-based).
type DSTPeriod struct {
	StartMonth, StartDay int
	EndMonth, EndDay     int
}

// ExchangeSchedule defines closing times in ET (EST/EDT) and metadata for an exchange.
type ExchangeSchedule struct {
	ESTCloseHour   int
	ESTCloseMinute int
	EDTCloseHour   int
	EDTCloseMinute int
	Name           string
	Notes          string
	DayOffset      int            // -1 for Asia/Pacific (close after midnight ET = previous US day)
	Timezone       string         // optional IANA timezone
	DSTOverride    *DSTOverride   // optional
}

// Exchange name constants for lookup.
const (
	ExchangeNASDAQ         = "NASDAQ"
	ExchangeNYSE           = "NYSE"
	ExchangeNYSEAmerican   = "NYSE AMERICAN"
	ExchangeNYSEArca       = "NYSE ARCA"
	ExchangeCBOEBZX        = "CBOE BZX"
	ExchangeToronto        = "TORONTO"
	ExchangeSaoPaulo       = "SAO PAULO"
	ExchangeMexico         = "MEXICO"
	ExchangeBuenosAires    = "BUENOS AIRES"
	ExchangeSantiago       = "SANTIAGO"
	ExchangeLondon         = "LONDON"
	ExchangeEuronextDublin = "EURONEXT DUBLIN"
	ExchangeXetra          = "XETRA"
	ExchangeEuronextParis  = "EURONEXT PARIS"
	ExchangeEuronextAmsterdam = "EURONEXT AMSTERDAM"
	ExchangeEuronextBrussels = "EURONEXT BRUSSELS"
	ExchangeMilan          = "MILAN"
	ExchangeSpain          = "SPAIN"
	ExchangeSixSwiss       = "SIX SWISS"
	ExchangeVienna         = "VIENNA"
	ExchangeEuronextLisbon = "EURONEXT LISBON"
	ExchangeOMXNordicStockholm = "OMX NORDIC STOCKHOLM"
	ExchangeOMXNordicCopenhagen = "OMX NORDIC COPENHAGEN"
	ExchangeOMXNordicHelsinki  = "OMX NORDIC HELSINKI"
	ExchangeOMXNordicIceland   = "OMX NORDIC ICELAND"
	ExchangeOslo           = "OSLO"
	ExchangeWarsaw         = "WARSAW"
	ExchangePrague         = "PRAGUE"
	ExchangeBudapest       = "BUDAPEST"
	ExchangeAthens         = "ATHENS"
	ExchangeIstanbul       = "ISTANBUL"
	ExchangeJSE            = "JSE"
	ExchangeTokyo          = "TOKYO"
	ExchangeHongKong       = "HONG KONG"
	ExchangeSingapore      = "SINGAPORE"
	ExchangeASX            = "ASX"
	ExchangeTaiwan         = "TAIWAN"
	ExchangeNSEIndia       = "NSE INDIA"
	ExchangeIndonesia      = "INDONESIA"
	ExchangeThailand       = "THAILAND"
	ExchangeMalaysia       = "MALAYSIA"
	ExchangeBSEIndia       = "BSE INDIA"
	ExchangeBucharestSpot  = "BUCHAREST SPOT"
	ExchangeColombia       = "COLOMBIA"
)

// misalignedOverride is the common European override: 12:50 or 13:10 etc. EDT during transition.
var misalignedOverride = &DSTOverride{
	Condition:    "misaligned_dst",
	EDTCloseHour: 13,
	EDTCloseMin:  10,
}

// ExchangeSchedules holds all 42 exchange definitions (EST/EDT close times, DST overrides, day offsets).
var ExchangeSchedules = map[string]ExchangeSchedule{
	ExchangeNASDAQ: {
		ESTCloseHour: 16, ESTCloseMinute: 40,
		EDTCloseHour: 16, EDTCloseMinute: 40,
		Name: "NASDAQ Stock Market",
		Notes: "4:20 PM ET (4:00 PM close + 40 min delay)",
	},
	ExchangeNYSE: {
		ESTCloseHour: 16, ESTCloseMinute: 40,
		EDTCloseHour: 16, EDTCloseMinute: 40,
		Name: "New York Stock Exchange",
		Notes: "4:20 PM ET (4:00 PM close + 40 min delay)",
	},
	ExchangeNYSEAmerican: {
		ESTCloseHour: 16, ESTCloseMinute: 40,
		EDTCloseHour: 16, EDTCloseMinute: 40,
		Name: "NYSE American",
		Notes: "4:20 PM ET (4:00 PM close + 40 min delay)",
	},
	ExchangeNYSEArca: {
		ESTCloseHour: 16, ESTCloseMinute: 40,
		EDTCloseHour: 16, EDTCloseMinute: 40,
		Name: "NYSE Arca",
		Notes: "4:20 PM ET (4:00 PM close + 40 min delay)",
	},
	ExchangeCBOEBZX: {
		ESTCloseHour: 16, ESTCloseMinute: 40,
		EDTCloseHour: 16, EDTCloseMinute: 40,
		Name: "Cboe BZX Exchange",
		Notes: "4:20 PM ET (4:00 PM close + 40 min delay)",
	},
	ExchangeToronto: {
		ESTCloseHour: 16, ESTCloseMinute: 40,
		EDTCloseHour: 16, EDTCloseMinute: 40,
		Name: "Toronto Stock Exchange",
		Notes: "4:20 PM ET (4:00 PM close + 40 min delay)",
	},
	ExchangeSaoPaulo: {
		ESTCloseHour: 15, ESTCloseMinute: 35,
		EDTCloseHour: 16, EDTCloseMinute: 35,
		Name: "B3 Sao Paulo Stock Exchange",
		Notes: "3:15 PM EST / 4:15 PM EDT",
	},
	ExchangeMexico: {
		ESTCloseHour: 16, ESTCloseMinute: 40,
		EDTCloseHour: 16, EDTCloseMinute: 40,
		Name: "Mexican Stock Exchange",
		Notes: "4:20 PM ET (same as NYSE)",
	},
	ExchangeBuenosAires: {
		ESTCloseHour: 15, ESTCloseMinute: 40,
		EDTCloseHour: 16, EDTCloseMinute: 40,
		Name: "Buenos Aires Stock Exchange",
		Notes: "3:20 PM EST / 4:20 PM EDT",
	},
	ExchangeSantiago: {
		ESTCloseHour: 15, ESTCloseMinute: 40,
		EDTCloseHour: 16, EDTCloseMinute: 40,
		Name: "Santiago Stock Exchange",
		Notes: "3:20 PM EST / 4:20 PM EDT",
	},
	ExchangeLondon: {
		ESTCloseHour: 12, ESTCloseMinute: 10,
		EDTCloseHour: 12, EDTCloseMinute: 10,
		Name: "London Stock Exchange",
		Notes: "11:50 AM EST / 11:50 AM EDT (12:50 PM EDT during DST transition periods)",
		DSTOverride: misalignedOverride,
	},
	ExchangeEuronextDublin: {
		ESTCloseHour: 12, ESTCloseMinute: 10,
		EDTCloseHour: 12, EDTCloseMinute: 10,
		Name: "Euronext Dublin",
		Notes: "11:50 AM EST / 11:50 AM EDT (12:50 PM EDT during DST transition periods)",
		DSTOverride: misalignedOverride,
	},
	ExchangeXetra: {
		ESTCloseHour: 12, ESTCloseMinute: 10,
		EDTCloseHour: 12, EDTCloseMinute: 10,
		Name: "Frankfurt Stock Exchange (Xetra)",
		Notes: "11:50 AM EST / 11:50 AM EDT (12:50 PM EDT during DST transition periods)",
		DSTOverride: misalignedOverride,
	},
	ExchangeEuronextParis: {
		ESTCloseHour: 12, ESTCloseMinute: 10,
		EDTCloseHour: 12, EDTCloseMinute: 10,
		Name: "Euronext Paris",
		Notes: "11:50 AM EST / 11:50 AM EDT (12:50 PM EDT during DST transition periods)",
		DSTOverride: misalignedOverride,
	},
	ExchangeEuronextAmsterdam: {
		ESTCloseHour: 12, ESTCloseMinute: 10,
		EDTCloseHour: 12, EDTCloseMinute: 10,
		Name: "Euronext Amsterdam",
		Notes: "11:50 AM EST / 11:50 AM EDT (12:50 PM EDT during DST transition periods)",
		DSTOverride: misalignedOverride,
	},
	ExchangeEuronextBrussels: {
		ESTCloseHour: 12, ESTCloseMinute: 10,
		EDTCloseHour: 12, EDTCloseMinute: 10,
		Name: "Euronext Brussels",
		Notes: "11:50 AM EST / 11:50 AM EDT (12:50 PM EDT during DST transition periods)",
		DSTOverride: misalignedOverride,
	},
	ExchangeMilan: {
		ESTCloseHour: 12, ESTCloseMinute: 10,
		EDTCloseHour: 12, EDTCloseMinute: 10,
		Name: "Borsa Italiana Milan",
		Notes: "11:50 AM EST / 11:50 AM EDT (12:50 PM EDT during DST transition periods)",
		DSTOverride: misalignedOverride,
	},
	ExchangeSpain: {
		ESTCloseHour: 12, ESTCloseMinute: 10,
		EDTCloseHour: 12, EDTCloseMinute: 10,
		Name: "Madrid Stock Exchange",
		Notes: "11:50 AM EST / 11:50 AM EDT (12:50 PM EDT during DST transition periods)",
		DSTOverride: misalignedOverride,
	},
	ExchangeSixSwiss: {
		ESTCloseHour: 12, ESTCloseMinute: 10,
		EDTCloseHour: 12, EDTCloseMinute: 10,
		Name: "SIX Swiss Exchange",
		Notes: "11:50 AM EST / 11:50 AM EDT (12:50 PM EDT during DST transition periods)",
		DSTOverride: misalignedOverride,
	},
	ExchangeVienna: {
		ESTCloseHour: 12, ESTCloseMinute: 10,
		EDTCloseHour: 12, EDTCloseMinute: 10,
		Name: "Vienna Stock Exchange",
		Notes: "11:50 AM EST / 11:50 AM EDT (12:50 PM EDT during DST transition periods)",
		DSTOverride: misalignedOverride,
	},
	ExchangeEuronextLisbon: {
		ESTCloseHour: 12, ESTCloseMinute: 10,
		EDTCloseHour: 12, EDTCloseMinute: 10,
		Name: "Euronext Lisbon",
		Notes: "11:50 AM EST / 11:50 AM EDT (12:50 PM EDT during DST transition periods)",
		DSTOverride: misalignedOverride,
	},
	ExchangeOMXNordicStockholm: {
		ESTCloseHour: 12, ESTCloseMinute: 10,
		EDTCloseHour: 12, EDTCloseMinute: 10,
		Name: "Nasdaq Stockholm",
		Notes: "11:30 AM EST / 11:30 AM EDT (12:30 PM EDT during DST transition periods)",
		DSTOverride: &DSTOverride{Condition: "misaligned_dst", EDTCloseHour: 13, EDTCloseMin: 10},
	},
	ExchangeOMXNordicCopenhagen: {
		ESTCloseHour: 11, ESTCloseMinute: 40,
		EDTCloseHour: 11, EDTCloseMinute: 40,
		Name: "Nasdaq Copenhagen",
		Notes: "11:00 AM EST / 11:00 AM EDT (12:00 PM EDT during DST transition periods)",
		DSTOverride: &DSTOverride{Condition: "misaligned_dst", EDTCloseHour: 12, EDTCloseMin: 40},
	},
	ExchangeOMXNordicHelsinki: {
		ESTCloseHour: 12, ESTCloseMinute: 10,
		EDTCloseHour: 12, EDTCloseMinute: 10,
		Name: "Nasdaq Helsinki",
		Notes: "11:30 AM EST / 11:30 AM EDT (12:30 PM EDT during DST transition periods)",
		DSTOverride: &DSTOverride{Condition: "misaligned_dst", EDTCloseHour: 13, EDTCloseMin: 10},
	},
	ExchangeOMXNordicIceland: {
		ESTCloseHour: 11, ESTCloseMinute: 10,
		EDTCloseHour: 12, EDTCloseMinute: 10,
		Name: "Nasdaq Iceland",
		Notes: "10:50 AM EST / 11:50 AM EDT",
		// No DST override - Iceland doesn't use DST
	},
	ExchangeOslo: {
		ESTCloseHour: 11, ESTCloseMinute: 5,
		EDTCloseHour: 11, EDTCloseMinute: 5,
		Name: "Oslo Stock Exchange",
		Notes: "10:25 AM EST / 10:25 AM EDT (11:25 AM EDT during DST transition periods)",
		DSTOverride: &DSTOverride{Condition: "misaligned_dst", EDTCloseHour: 12, EDTCloseMin: 5},
	},
	ExchangeWarsaw: {
		ESTCloseHour: 11, ESTCloseMinute: 40,
		EDTCloseHour: 11, EDTCloseMinute: 40,
		Name: "Warsaw Stock Exchange",
		Notes: "11:00 AM EST / 11:00 AM EDT (12:00 PM EDT during DST transition periods)",
		DSTOverride: &DSTOverride{Condition: "misaligned_dst", EDTCloseHour: 12, EDTCloseMin: 40},
	},
	ExchangePrague: {
		ESTCloseHour: 10, ESTCloseMinute: 55,
		EDTCloseHour: 10, EDTCloseMinute: 55,
		Name: "Prague Stock Exchange",
		Notes: "10:15 AM EST / 10:15 AM EDT (11:15 AM EDT during DST transition periods)",
		DSTOverride: &DSTOverride{Condition: "misaligned_dst", EDTCloseHour: 11, EDTCloseMin: 55},
	},
	ExchangeBudapest: {
		ESTCloseHour: 11, ESTCloseMinute: 55,
		EDTCloseHour: 11, EDTCloseMinute: 55,
		Name: "Budapest Stock Exchange",
		Notes: "11:05 AM EST / 11:05 AM EDT (12:05 PM EDT during DST transition periods)",
		DSTOverride: &DSTOverride{Condition: "misaligned_dst", EDTCloseHour: 12, EDTCloseMin: 55},
	},
	ExchangeAthens: {
		ESTCloseHour: 10, ESTCloseMinute: 50,
		EDTCloseHour: 10, EDTCloseMinute: 50,
		Name: "Athens Stock Exchange",
		Notes: "10:10 AM EST / 10:10 AM EDT (11:10 AM EDT during DST transition periods)",
		DSTOverride: &DSTOverride{Condition: "misaligned_dst", EDTCloseHour: 11, EDTCloseMin: 50},
	},
	ExchangeIstanbul: {
		ESTCloseHour: 11, ESTCloseMinute: 0,
		EDTCloseHour: 12, EDTCloseMinute: 0,
		Name: "Borsa Istanbul",
		Notes: "10:20 AM EST / 11:20 AM EDT (Turkey doesn't observe DST)",
	},
	ExchangeJSE: {
		ESTCloseHour: 11, ESTCloseMinute: 0,
		EDTCloseHour: 12, EDTCloseMinute: 0,
		Name: "Johannesburg Stock Exchange",
		Notes: "10:20 AM EST / 11:20 AM EDT (South Africa doesn't observe DST)",
	},
	ExchangeTokyo: {
		ESTCloseHour: 2, ESTCloseMinute: 15,
		EDTCloseHour: 3, EDTCloseMinute: 15,
		Name: "Tokyo Stock Exchange",
		Notes: "1:20 AM EST / 2:20 AM EDT",
		DayOffset: -1,
	},
	ExchangeHongKong: {
		ESTCloseHour: 3, ESTCloseMinute: 50,
		EDTCloseHour: 4, EDTCloseMinute: 50,
		Name: "Hong Kong Stock Exchange",
		Notes: "3:20 AM EST / 4:20 AM EDT",
		DayOffset: -1,
	},
	ExchangeSingapore: {
		ESTCloseHour: 4, ESTCloseMinute: 50,
		EDTCloseHour: 5, EDTCloseMinute: 50,
		Name: "Singapore Exchange",
		Notes: "4:20 AM EST / 5:20 AM EDT",
		DayOffset: -1,
	},
	ExchangeASX: {
		ESTCloseHour: 1, ESTCloseMinute: 0,
		EDTCloseHour: 3, EDTCloseMinute: 0,
		Name: "Australian Securities Exchange",
		Notes: "12:40 AM EST / 2:40 AM EDT (1:40 AM EDT when both regions observe DST)",
		Timezone: "Australia/Sydney",
		DayOffset: -1,
		DSTOverride: &DSTOverride{
			Condition:    "both_dst",
			EDTCloseHour: 2,
			EDTCloseMin:  0,
		},
	},
	ExchangeTaiwan: {
		ESTCloseHour: 1, ESTCloseMinute: 10,
		EDTCloseHour: 2, EDTCloseMinute: 10,
		Name: "Taiwan Stock Exchange",
		Notes: "12:50 AM EST / 1:50 AM EDT",
		DayOffset: -1,
	},
	ExchangeNSEIndia: {
		ESTCloseHour: 5, ESTCloseMinute: 40,
		EDTCloseHour: 6, EDTCloseMinute: 40,
		Name: "National Stock Exchange of India",
		Notes: "5:20 AM EST / 6:20 AM EDT",
		DayOffset: -1,
	},
	ExchangeIndonesia: {
		ESTCloseHour: 4, ESTCloseMinute: 40,
		EDTCloseHour: 5, EDTCloseMinute: 40,
		Name: "Indonesia Stock Exchange",
		Notes: "4:20 AM EST / 5:20 AM EDT",
		DayOffset: -1,
	},
	ExchangeThailand: {
		ESTCloseHour: 5, ESTCloseMinute: 20,
		EDTCloseHour: 6, EDTCloseMinute: 20,
		Name: "Stock Exchange of Thailand",
		Notes: "4:50 AM EST / 5:50 AM EDT",
		DayOffset: -1,
	},
	ExchangeMalaysia: {
		ESTCloseHour: 4, ESTCloseMinute: 40,
		EDTCloseHour: 5, EDTCloseMinute: 40,
		Name: "Bursa Malaysia",
		Notes: "4:20 AM EST / 5:20 AM EDT",
		DayOffset: -1,
	},
	ExchangeBSEIndia: {
		ESTCloseHour: 5, ESTCloseMinute: 40,
		EDTCloseHour: 6, EDTCloseMinute: 40,
		Name: "Bombay Stock Exchange",
		Notes: "5:40 AM EST / 6:40 AM EDT - Same hours as NSE",
		DayOffset: -1,
	},
	ExchangeBucharestSpot: {
		ESTCloseHour: 11, ESTCloseMinute: 0,
		EDTCloseHour: 11, EDTCloseMinute: 0,
		Name: "Bucharest Stock Exchange",
		Notes: "11:00 AM EST / 11:00 AM EDT",
		DayOffset: 0,
	},
	ExchangeColombia: {
		ESTCloseHour: 16, ESTCloseMinute: 40,
		EDTCloseHour: 15, EDTCloseMinute: 40,
		Name: "Colombia Stock Exchange (BVC)",
		Notes: "4:40 PM EST / 3:40 PM EDT",
		DayOffset: 0,
	},
}

// EXCHANGE_TIMEZONES maps exchange name to IANA timezone for European and other exchanges.
var ExchangeTimezones = map[string]string{
	ExchangeLondon:             "Europe/London",
	ExchangeXetra:              "Europe/Berlin",
	ExchangeEuronextParis:      "Europe/Paris",
	ExchangeEuronextAmsterdam:  "Europe/Amsterdam",
	ExchangeEuronextBrussels:   "Europe/Brussels",
	ExchangeEuronextLisbon:     "Europe/Lisbon",
	ExchangeEuronextDublin:     "Europe/Dublin",
	ExchangeMilan:              "Europe/Rome",
	ExchangeSpain:              "Europe/Madrid",
	ExchangeSixSwiss:           "Europe/Zurich",
	"SIX":                      "Europe/Zurich",
	ExchangeOMXNordicStockholm: "Europe/Stockholm",
	ExchangeOMXNordicCopenhagen: "Europe/Copenhagen",
	ExchangeOMXNordicHelsinki:  "Europe/Helsinki",
	ExchangeOMXNordicIceland:   "Atlantic/Reykjavik",
	ExchangeOslo:               "Europe/Oslo",
	ExchangeWarsaw:             "Europe/Warsaw",
	ExchangeVienna:             "Europe/Vienna",
	ExchangeAthens:             "Europe/Athens",
	ExchangePrague:             "Europe/Prague",
	ExchangeBudapest:           "Europe/Budapest",
	ExchangeBucharestSpot:      "Europe/Bucharest",
	ExchangeColombia:           "America/Bogota",
}

// HourlyOpenMinute maps exchange name to the minute-of-hour at which its
// session opens, which is also the minute at which 1-hour candles close.
// Exchanges that open at :30 (e.g. NYSE at 9:30 ET) have candles closing at
// :30 past each hour; exchanges that open on the hour close on the hour.
// Only relevant for "hour"-aligned exchanges; "half" and "quarter" exchanges
// always use :00/:30 or :00/:15/:30/:45 boundaries regardless of open time.
var HourlyOpenMinute = map[string]int{
	ExchangeNYSE:         30, // 9:30 ET
	ExchangeNASDAQ:       30, // 9:30 ET
	ExchangeNYSEAmerican: 30, // 9:30 ET
	ExchangeNYSEArca:     30, // 9:30 ET
	ExchangeCBOEBZX:      30, // 9:30 ET
	ExchangeToronto:      30, // 9:30 ET
	ExchangeSantiago:     30, // 9:30 CLT
	ExchangeMexico:       30, // 8:30 CT
	ExchangeColombia:     30, // 9:30 COT
	// All other exchanges open on the hour and default to 0.
}

// HourlyAlignment maps exchange name to alignment: "hour", "quarter", or "half".
// "hour" = :00 candles (most), "quarter" = :15 (BSE/NSE India), "half" = :30 (Hong Kong, Euronext Paris, Athens, Iceland).
var HourlyAlignment = map[string]string{
	ExchangeNYSE:               "hour",
	ExchangeNASDAQ:             "hour",
	ExchangeNYSEAmerican:       "hour",
	ExchangeNYSEArca:           "hour",
	ExchangeCBOEBZX:            "hour",
	ExchangeToronto:            "hour",
	ExchangeMexico:             "hour",
	ExchangeSantiago:           "hour",
	ExchangeBuenosAires:        "hour",
	ExchangeLondon:             "hour",
	ExchangeXetra:              "hour",
	ExchangeEuronextAmsterdam:  "hour",
	ExchangeEuronextBrussels:   "hour",
	ExchangeEuronextLisbon:     "hour",
	ExchangeEuronextDublin:     "hour",
	ExchangeMilan:              "hour",
	ExchangeSpain:              "hour",
	ExchangeSixSwiss:           "hour",
	ExchangeOMXNordicStockholm: "hour",
	ExchangeOMXNordicCopenhagen: "hour",
	ExchangeOMXNordicHelsinki:  "hour",
	ExchangeOslo:               "hour",
	ExchangeWarsaw:             "hour",
	ExchangeVienna:             "hour",
	ExchangePrague:             "hour",
	ExchangeBudapest:           "hour",
	ExchangeIstanbul:           "hour",
	ExchangeTokyo:              "hour",
	ExchangeTaiwan:             "hour",
	ExchangeASX:                "hour",
	ExchangeSingapore:          "hour",
	ExchangeMalaysia:           "hour",
	ExchangeThailand:           "hour",
	ExchangeIndonesia:          "hour",
	ExchangeSaoPaulo:           "hour",
	ExchangeJSE:                "hour",
	ExchangeBSEIndia:           "quarter",
	ExchangeNSEIndia:           "quarter",
	ExchangeHongKong:           "half",
	ExchangeEuronextParis:      "half",
	ExchangeAthens:             "half",
	ExchangeOMXNordicIceland:   "half",
	"SIX":                      "hour", // alias for SIX SWISS
}

// IsDateInPeriod reports whether d falls within the period (handles wrap-around across year).
func IsDateInPeriod(d time.Time, startMonth, startDay, endMonth, endDay int) bool {
	y := d.Year()
	start := time.Date(y, time.Month(startMonth), startDay, 0, 0, 0, 0, time.UTC)
	end := time.Date(y, time.Month(endMonth), endDay, 23, 59, 59, 0, time.UTC)
	if !start.After(end) {
		return !d.Before(start) && !d.After(end)
	}
	return !d.Before(start) || !d.After(end)
}

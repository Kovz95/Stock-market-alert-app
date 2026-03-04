// Package calendar: FMP All Exchange Market Hours API integration.
// Uses FMP_API_KEY from environment (passed by caller) for auth.
package calendar

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

const fmpStableBaseURL = "https://financialmodelingprep.com/stable"

// FMPExchangeHours is one exchange's row from the All Exchange Market Hours API.
type FMPExchangeHours struct {
	Exchange    string `json:"exchange"`
	Name        string `json:"name"`
	OpeningHour string `json:"openingHour"`
	ClosingHour string `json:"closingHour"`
	Timezone    string `json:"timezone"`
	IsMarketOpen bool  `json:"isMarketOpen"`
}

// MarketHoursSnapshot is the result of one call to All Exchange Market Hours.
// ByExchange is keyed by our internal exchange constant (e.g. "NYSE", "LONDON").
type MarketHoursSnapshot struct {
	ByExchange map[string]FMPExchangeHours
	FetchedAt  time.Time
}

// FMPToInternal maps FMP API exchange symbols to our internal exchange constants.
// FMP returns symbols like "NASDAQ", "NYSE", "LSE"; we use "NASDAQ", "NYSE", "LONDON".
// Refine this map after inspecting the actual API response.
var FMPToInternal = map[string]string{
	"NASDAQ":   ExchangeNASDAQ,
	"NYSE":     ExchangeNYSE,
	"NYSE MKT": ExchangeNYSEAmerican,
	"AMEX":     ExchangeNYSEAmerican,
	"ARCA":     ExchangeNYSEArca,
	"NYSEARCA": ExchangeNYSEArca,
	"BATS":     ExchangeCBOEBZX,
	"CBOE":     ExchangeCBOEBZX,
	"TSX":      ExchangeToronto,
	"TSXV":     ExchangeToronto,
	"BVMF":     ExchangeSaoPaulo,
	"BMV":      ExchangeMexico,
	"MEXICO":   ExchangeMexico,
	"BCBA":     ExchangeBuenosAires,
	"BCS":      ExchangeSantiago,
	"LSE":      ExchangeLondon,
	"LONDON":   ExchangeLondon,
	"XETR":     ExchangeXetra,
	"XETRA":    ExchangeXetra,
	"EPA":      ExchangeEuronextParis,
	"PARIS":    ExchangeEuronextParis,
	"EURONEXT PARIS": ExchangeEuronextParis,
	"AEA":      ExchangeEuronextAmsterdam,
	"AMS":      ExchangeEuronextAmsterdam,
	"BRU":      ExchangeEuronextBrussels,
	"LIS":      ExchangeEuronextLisbon,
	"DUB":      ExchangeEuronextDublin,
	"ISE":      ExchangeEuronextDublin,
	"BIT":      ExchangeMilan,
	"MILAN":    ExchangeMilan,
	"MAD":      ExchangeSpain,
	"MADRID":   ExchangeSpain,
	"SIX":      ExchangeSixSwiss,
	"SWX":      ExchangeSixSwiss,
	"VIENNA":   ExchangeVienna,
	"WBAG":     ExchangeVienna,
	"STO":      ExchangeOMXNordicStockholm,
	"OMXSTO":   ExchangeOMXNordicStockholm,
	"CPH":      ExchangeOMXNordicCopenhagen,
	"OMXCPH":   ExchangeOMXNordicCopenhagen,
	"HEL":      ExchangeOMXNordicHelsinki,
	"OMXHEL":   ExchangeOMXNordicHelsinki,
	"ICE":      ExchangeOMXNordicIceland,
	"OMXICE":   ExchangeOMXNordicIceland,
	"OSE":      ExchangeOslo,
	"OSLO":     ExchangeOslo,
	"WAR":      ExchangeWarsaw,
	"WSE":      ExchangeWarsaw,
	"PRA":      ExchangePrague,
	"BUD":      ExchangeBudapest,
	"BUX":      ExchangeBudapest,
	"ASE":      ExchangeAthens,
	"ATHENS":   ExchangeAthens,
	"BIST":     ExchangeIstanbul,
	"XIST":     ExchangeIstanbul,
	"ISTANBUL": ExchangeIstanbul,
	"JSE":      ExchangeJSE,
	"JOHANNESBURG": ExchangeJSE,
	"TSE":      ExchangeTokyo,
	"JPX":      ExchangeTokyo,
	"TOKYO":    ExchangeTokyo,
	"HKG":      ExchangeHongKong,
	"HKEX":     ExchangeHongKong,
	"HONG KONG": ExchangeHongKong,
	"SES":      ExchangeSingapore,
	"SGX":      ExchangeSingapore,
	"SINGAPORE": ExchangeSingapore,
	"ASX":      ExchangeASX,
	"AUSTRALIA": ExchangeASX,
	"TWSE":     ExchangeTaiwan,
	"TAIWAN":   ExchangeTaiwan,
	"NSE":      ExchangeNSEIndia,
	"NSE INDIA": ExchangeNSEIndia,
	"BSE":      ExchangeBSEIndia,
	"BSE INDIA": ExchangeBSEIndia,
	"IDX":      ExchangeIndonesia,
	"INDONESIA": ExchangeIndonesia,
	"SET":      ExchangeThailand,
	"THAILAND": ExchangeThailand,
	"KLSE":     ExchangeMalaysia,
	"MALAYSIA": ExchangeMalaysia,
	"BVB":      ExchangeBucharestSpot,
	"BUCHAREST": ExchangeBucharestSpot,
	"BVC":      ExchangeColombia,
	"COLOMBIA": ExchangeColombia,
}

// FetchAllExchangeMarketHours calls FMP's all-exchange-market-hours endpoint.
// apiKey is read from environment by the caller (e.g. FMP_API_KEY from .env).
// If client is nil, http.DefaultClient is used. If apiKey is empty, returns (nil, nil).
// On 429, retries up to 3 times with backoff (Retry-After or 60s).
func FetchAllExchangeMarketHours(ctx context.Context, apiKey string, client *http.Client) (*MarketHoursSnapshot, error) {
	if apiKey == "" {
		return nil, nil
	}
	if client == nil {
		client = http.DefaultClient
	}
	u, err := url.Parse(fmpStableBaseURL + "/all-exchange-market-hours")
	if err != nil {
		return nil, err
	}
	u.RawQuery = url.Values{"apikey": {apiKey}}.Encode()

	const maxRetries = 3
	var lastErr error
	for attempt := 0; attempt < maxRetries; attempt++ {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
		if err != nil {
			return nil, err
		}
		resp, err := client.Do(req)
		if err != nil {
			lastErr = err
			continue
		}

		if resp.StatusCode == http.StatusTooManyRequests {
			backoff := parseFMPRetryAfter(resp)
			resp.Body.Close()
			if attempt+1 < maxRetries && backoff > 0 {
				select {
				case <-ctx.Done():
					return nil, ctx.Err()
				case <-time.After(backoff):
				}
				continue
			}
			lastErr = fmt.Errorf("FMP all-exchange-market-hours: 429 rate limited after %d retries", attempt+1)
			break
		}

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
			resp.Body.Close()
			lastErr = fmt.Errorf("FMP all-exchange-market-hours: %s %s", resp.Status, string(body))
			break
		}

		var raw []FMPExchangeHours
		if err := json.NewDecoder(resp.Body).Decode(&raw); err != nil {
			resp.Body.Close()
			return nil, err
		}
		resp.Body.Close()
		snap := &MarketHoursSnapshot{
			ByExchange: make(map[string]FMPExchangeHours),
			FetchedAt:  time.Now().UTC(),
		}
		for _, h := range raw {
			internal := FMPToInternal[h.Exchange]
			if internal == "" {
				internal = h.Exchange
			}
			snap.ByExchange[internal] = h
		}
		return snap, nil
	}
	return nil, lastErr
}

func parseFMPRetryAfter(resp *http.Response) time.Duration {
	if s := resp.Header.Get("Retry-After"); s != "" {
		if sec, err := strconv.ParseFloat(strings.TrimSpace(s), 64); err == nil && sec > 0 {
			return time.Duration(sec * float64(time.Second))
		}
	}
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
	var v struct {
		RetryAfter float64 `json:"retry_after"`
	}
	if json.Unmarshal(raw, &v) == nil && v.RetryAfter > 0 {
		return time.Duration(v.RetryAfter * float64(time.Second))
	}
	return 60 * time.Second
}

// IsExchangeOpenFromSnapshot returns whether the exchange is open according to the FMP snapshot.
// If snapshot is nil or the exchange is not in the snapshot, returns false (caller should fall back to IsExchangeOpen).
func IsExchangeOpenFromSnapshot(exchange string, snapshot *MarketHoursSnapshot) bool {
	if snapshot == nil || snapshot.ByExchange == nil {
		return false
	}
	h, ok := snapshot.ByExchange[exchange]
	if !ok {
		return false
	}
	return h.IsMarketOpen
}

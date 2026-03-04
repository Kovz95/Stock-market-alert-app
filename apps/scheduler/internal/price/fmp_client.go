package price

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

const fmpBaseURL = "https://financialmodelingprep.com/api/v3"

// FMPFetcher is the interface for fetching price data (real API or mock for tests).
type FMPFetcher interface {
	FetchDaily(ticker string, limit int) ([]DailyRow, error)
	FetchHourly(ticker string) ([]HourlyRow, error)
}

// FMPClient calls the Financial Modeling Prep API for historical prices.
// It applies a global rate limit (min interval between requests) and retries on 429.
type FMPClient struct {
	apiKey      string
	client      *http.Client
	mu          sync.Mutex
	lastRequest time.Time
	minInterval time.Duration
}

// NewFMPClient creates an FMP API client with rate limiting and 429 retry.
// Min interval between requests is read from FMP_MIN_INTERVAL_MS (default 200ms = 5 req/s, ~300/min).
func NewFMPClient(apiKey string) *FMPClient {
	interval := 200 * time.Millisecond
	if s := os.Getenv("FMP_MIN_INTERVAL_MS"); s != "" {
		if ms, err := strconv.Atoi(strings.TrimSpace(s)); err == nil && ms > 0 {
			interval = time.Duration(ms) * time.Millisecond
		}
	}
	return &FMPClient{
		apiKey:      apiKey,
		client:      &http.Client{Timeout: 15 * time.Second},
		minInterval: interval,
	}
}

func (c *FMPClient) rateLimit() {
	c.mu.Lock()
	defer c.mu.Unlock()
	if elapsed := time.Since(c.lastRequest); elapsed < c.minInterval {
		time.Sleep(c.minInterval - elapsed)
	}
	c.lastRequest = time.Now()
}

// DailyRow is one day of OHLCV from FMP historical-price-full.
type DailyRow struct {
	Date   string  `json:"date"`
	Open   float64 `json:"open"`
	High   float64 `json:"high"`
	Low    float64 `json:"low"`
	Close  float64 `json:"close"`
	Volume int64   `json:"volume"`
}

// HourlyRow is one bar from FMP historical-chart/1hour.
// Volume is float64 because FMP sometimes returns volume as a float (e.g. 1111905.34).
type HourlyRow struct {
	Date   string  `json:"date"`
	Open   float64 `json:"open"`
	High   float64 `json:"high"`
	Low    float64 `json:"low"`
	Close  float64 `json:"close"`
	Volume float64 `json:"volume"`
}

// FetchDaily fetches daily historical prices for a ticker (up to limit days).
func (c *FMPClient) FetchDaily(ticker string, limit int) ([]DailyRow, error) {
	if c.apiKey == "" {
		return nil, fmt.Errorf("FMP_API_KEY not set")
	}
	u, _ := url.Parse(fmpBaseURL + "/historical-price-full/" + url.PathEscape(ticker))
	q := u.Query()
	q.Set("apikey", c.apiKey)
	if limit > 0 {
		q.Set("timeseries", fmt.Sprintf("%d", limit)) // FMP: timeseries = number of days
	}
	u.RawQuery = q.Encode()

	const maxRetries = 3
	var lastErr error
	for attempt := 0; attempt < maxRetries; attempt++ {
		c.rateLimit()
		resp, err := c.client.Get(u.String())
		if err != nil {
			lastErr = err
			continue
		}

		if resp.StatusCode == http.StatusTooManyRequests {
			backoff := c.parseFMPRetryAfter(resp)
			_ = resp.Body.Close()
			if attempt+1 < maxRetries && backoff > 0 {
				time.Sleep(backoff)
				continue
			}
			lastErr = fmt.Errorf("FMP API 429 rate limited after %d retries", attempt+1)
			break
		}

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
			_ = resp.Body.Close()
			lastErr = fmt.Errorf("FMP API %s %s", resp.Status, string(body))
			break
		}

		var out struct {
			Historical []DailyRow `json:"historical"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
			_ = resp.Body.Close()
			return nil, err
		}
		_ = resp.Body.Close()
		if len(out.Historical) == 0 {
			return nil, nil
		}
		return out.Historical, nil
	}
	return nil, lastErr
}

// FetchHourly fetches hourly (1h) historical chart for a ticker.
func (c *FMPClient) FetchHourly(ticker string) ([]HourlyRow, error) {
	if c.apiKey == "" {
		return nil, fmt.Errorf("FMP_API_KEY not set")
	}
	u, _ := url.Parse(fmpBaseURL + "/historical-chart/1hour/" + url.PathEscape(ticker))
	u.RawQuery = url.Values{"apikey": {c.apiKey}}.Encode()

	const maxRetries = 3
	var lastErr error
	for attempt := 0; attempt < maxRetries; attempt++ {
		c.rateLimit()
		resp, err := c.client.Get(u.String())
		if err != nil {
			lastErr = err
			continue
		}

		if resp.StatusCode == http.StatusTooManyRequests {
			backoff := c.parseFMPRetryAfter(resp)
			_ = resp.Body.Close()
			if attempt+1 < maxRetries && backoff > 0 {
				time.Sleep(backoff)
				continue
			}
			lastErr = fmt.Errorf("FMP API 429 rate limited after %d retries", attempt+1)
			break
		}

		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
			_ = resp.Body.Close()
			lastErr = fmt.Errorf("FMP API %s %s", resp.Status, string(body))
			break
		}

		var rows []HourlyRow
		if err := json.NewDecoder(resp.Body).Decode(&rows); err != nil {
			_ = resp.Body.Close()
			return nil, err
		}
		_ = resp.Body.Close()
		return rows, nil
	}
	return nil, lastErr
}

// parseFMPRetryAfter returns a backoff duration from 429 response (Retry-After header or default 60s).
func (c *FMPClient) parseFMPRetryAfter(resp *http.Response) time.Duration {
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

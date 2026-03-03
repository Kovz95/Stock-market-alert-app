package price

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"time"
)

const fmpBaseURL = "https://financialmodelingprep.com/api/v3"

// FMPFetcher is the interface for fetching price data (real API or mock for tests).
type FMPFetcher interface {
	FetchDaily(ticker string, limit int) ([]DailyRow, error)
	FetchHourly(ticker string) ([]HourlyRow, error)
}

// FMPClient calls the Financial Modeling Prep API for historical prices.
type FMPClient struct {
	apiKey string
	client *http.Client
}

// NewFMPClient creates an FMP API client.
func NewFMPClient(apiKey string) *FMPClient {
	return &FMPClient{
		apiKey: apiKey,
		client: &http.Client{Timeout: 15 * time.Second},
	}
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

	resp, err := c.client.Get(u.String())
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("FMP API %s", resp.Status)
	}

	var out struct {
		Historical []DailyRow `json:"historical"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	if len(out.Historical) == 0 {
		return nil, nil
	}
	return out.Historical, nil
}

// FetchHourly fetches hourly (1h) historical chart for a ticker.
func (c *FMPClient) FetchHourly(ticker string) ([]HourlyRow, error) {
	if c.apiKey == "" {
		return nil, fmt.Errorf("FMP_API_KEY not set")
	}
	u, _ := url.Parse(fmpBaseURL + "/historical-chart/1hour/" + url.PathEscape(ticker))
	u.RawQuery = url.Values{"apikey": {c.apiKey}}.Encode()

	resp, err := c.client.Get(u.String())
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("FMP API %s", resp.Status)
	}

	var rows []HourlyRow
	if err := json.NewDecoder(resp.Body).Decode(&rows); err != nil {
		return nil, err
	}
	return rows, nil
}

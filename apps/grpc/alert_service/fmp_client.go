package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"time"
)

const fmpBaseURL = "https://financialmodelingprep.com/api/v3"

// fmpFetcher is the interface for fetching price data.
type fmpFetcher interface {
	FetchDaily(ticker string, limit int) ([]dailyRow, error)
	FetchHourly(ticker string) ([]hourlyRow, error)
}

// fmpClient calls the Financial Modeling Prep API for historical prices.
type fmpClient struct {
	apiKey string
	client *http.Client
}

// newFMPClient creates an FMP API client.
func newFMPClient(apiKey string) *fmpClient {
	return &fmpClient{
		apiKey: apiKey,
		client: &http.Client{Timeout: 15 * time.Second},
	}
}

// dailyRow is one day of OHLCV from FMP historical-price-full.
type dailyRow struct {
	Date   string  `json:"date"`
	Open   float64 `json:"open"`
	High   float64 `json:"high"`
	Low    float64 `json:"low"`
	Close  float64 `json:"close"`
	Volume int64   `json:"volume"`
}

// hourlyRow is one bar from FMP historical-chart/1hour.
// Volume is float64 because FMP sometimes returns volume as a float.
type hourlyRow struct {
	Date   string  `json:"date"`
	Open   float64 `json:"open"`
	High   float64 `json:"high"`
	Low    float64 `json:"low"`
	Close  float64 `json:"close"`
	Volume float64 `json:"volume"`
}

// FetchDaily fetches daily historical prices for a ticker (up to limit days).
func (c *fmpClient) FetchDaily(ticker string, limit int) ([]dailyRow, error) {
	if c.apiKey == "" {
		return nil, fmt.Errorf("FMP_API_KEY not set")
	}
	u, _ := url.Parse(fmpBaseURL + "/historical-price-full/" + url.PathEscape(ticker))
	q := u.Query()
	q.Set("apikey", c.apiKey)
	if limit > 0 {
		q.Set("timeseries", fmt.Sprintf("%d", limit))
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
		Historical []dailyRow `json:"historical"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	return out.Historical, nil
}

// FetchHourly fetches hourly (1h) historical chart for a ticker.
func (c *fmpClient) FetchHourly(ticker string) ([]hourlyRow, error) {
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

	var rows []hourlyRow
	if err := json.NewDecoder(resp.Body).Decode(&rows); err != nil {
		return nil, err
	}
	return rows, nil
}

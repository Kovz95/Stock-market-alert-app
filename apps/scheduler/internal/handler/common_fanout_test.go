package handler

import (
	"context"
	"encoding/json"
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgtype"

	"stockalert/alert"
	db "stockalert/database/generated"
	"stockalert/discord"
	"stockalert/indicator"
)

// ---- Recording accumulator ----

type recordingAccum struct {
	mu    sync.Mutex
	calls []accumCall
}

type accumCall struct {
	url   string
	embed discord.Embed
}

func (a *recordingAccum) Add(url string, embed discord.Embed) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.calls = append(a.calls, accumCall{url: url, embed: embed})
}

func (a *recordingAccum) FlushAll() (int, int) { return 0, 0 }

func (a *recordingAccum) URLs() []string {
	a.mu.Lock()
	defer a.mu.Unlock()
	out := make([]string, len(a.calls))
	for i, c := range a.calls {
		out[i] = c.url
	}
	return out
}

// ---- Fake WebhookRouter ----

type fixedRouter struct{ url string }

func (f *fixedRouter) ResolveWebhookURL(_, _, _ string, _ bool) string { return f.url }
func (f *fixedRouter) GetEconomy(_ string) string                       { return "" }
func (f *fixedRouter) GetISIN(_ string) string                          { return "" }

// ---- Fake portfolio queries ----

type fakePortfolioQueriesFanout struct {
	portfolios []db.ListPortfoliosForFanoutRow
	stocks     []db.PortfolioStock
}

func (f *fakePortfolioQueriesFanout) ListPortfoliosForFanout(_ context.Context) ([]db.ListPortfoliosForFanoutRow, error) {
	return f.portfolios, nil
}
func (f *fakePortfolioQueriesFanout) ListPortfolioStocks(_ context.Context) ([]db.PortfolioStock, error) {
	return f.stocks, nil
}

// ---- Fake custom channel queries ----

type fakeCustomQueriesFanout struct {
	payload map[string]interface{}
}

func (f *fakeCustomQueriesFanout) GetAppDocument(_ context.Context, _ string) (db.AppDocument, error) {
	if f.payload == nil {
		return db.AppDocument{}, pgx.ErrNoRows
	}
	b, _ := json.Marshal(f.payload)
	return db.AppDocument{Payload: b}, nil
}

// ---- Helpers ----

func textPF(s string) pgtype.Text { return pgtype.Text{String: s, Valid: true} }
func boolPF(b bool) pgtype.Bool   { return pgtype.Bool{Bool: b, Valid: true} }

func makeAlert(ticker, condition string) db.Alert {
	conds, _ := json.Marshal([]string{condition})
	return db.Alert{
		Ticker:     pgtype.Text{String: ticker, Valid: true},
		StockName:  pgtype.Text{String: "Test Stock", Valid: true},
		Action:     pgtype.Text{String: "Buy", Valid: true},
		Timeframe:  pgtype.Text{String: "daily", Valid: true},
		Exchange:   pgtype.Text{String: "NYSE", Valid: true},
		Conditions: conds,
	}
}

func buildFanoutCommon(
	economyURL string,
	pq discord.PortfolioQueriesDB,
	cq discord.CustomChannelQueriesDB,
	accum discord.AccumFlusher,
) *Common {
	registry := indicator.NewDefaultRegistry()
	// Use a stub Queries that never executes real SQL (alert checker is not called in these tests).
	queries := db.New(&fakeQueriesDBTX{})

	return &Common{
		Queries:        queries,
		Checker:        alert.NewChecker(queries, registry),
		Router:         &fixedRouter{url: economyURL},
		Accum:          accum,
		Notifier:       discord.NewNotifier(),
		Portfolios:     discord.NewPortfolioResolverFromQueries(pq, time.Hour, slog.Default()),
		CustomChannels: discord.NewCustomChannelResolverFromQueries(cq, time.Hour, slog.Default()),
		Logger:         slog.Default(),
		JobTimeout:     30 * time.Second,
	}
}

func countURL(urls []string, target string) int {
	n := 0
	for _, u := range urls {
		if u == target {
			n++
		}
	}
	return n
}

// ---- Tests ----

// TestFanout_ThreeDestinations: economy W1, portfolio W2, custom W3 — all unique.
// Expects exactly 3 Add calls.
func TestFanout_ThreeDestinations(t *testing.T) {
	const (
		economyURL   = "https://discord.com/webhooks/W1"
		portfolioURL = "https://discord.com/webhooks/W2"
		customURL    = "https://discord.com/webhooks/W3"
		testTicker   = "AAPL"
		testCond     = "close > 100"
	)

	accum := &recordingAccum{}
	pq := &fakePortfolioQueriesFanout{
		portfolios: []db.ListPortfoliosForFanoutRow{
			{ID: "p1", Name: "P1", DiscordWebhook: textPF(portfolioURL), Enabled: boolPF(true)},
		},
		stocks: []db.PortfolioStock{{PortfolioID: "p1", Ticker: testTicker}},
	}
	cq := &fakeCustomQueriesFanout{
		payload: map[string]interface{}{
			"ch1": map[string]interface{}{
				"webhook_url": customURL,
				"condition":   testCond,
				"enabled":     true,
			},
		},
	}

	c := buildFanoutCommon(economyURL, pq, cq, accum)

	a := makeAlert(testTicker, testCond)
	result := alert.CheckResult{AlertID: "test-id", Triggered: true}
	c.dispatchTriggered(context.Background(), a, result, "NYSE", "daily")

	urls := accum.URLs()
	if len(urls) != 3 {
		t.Fatalf("expected 3 Add calls, got %d: %v", len(urls), urls)
	}
	for _, want := range []string{economyURL, portfolioURL, customURL} {
		if countURL(urls, want) != 1 {
			t.Errorf("expected %s exactly once, got %d occurrences in %v", want, countURL(urls, want), urls)
		}
	}

	// All calls should have received the same embed pointer value (built once).
	if len(accum.calls) == 3 {
		e0 := accum.calls[0].embed
		for i, call := range accum.calls[1:] {
			if call.embed.Title != e0.Title || call.embed.Color != e0.Color {
				t.Errorf("embed at index %d differs from embed[0]", i+1)
			}
		}
	}
}

// TestFanout_Dedup: economy=W1, portfolio p1=W1 (shared), portfolio p2=W2, custom=W1 (shared).
// Expects exactly 2 Add calls: W1 once, W2 once.
func TestFanout_Dedup(t *testing.T) {
	const (
		sharedURL    = "https://discord.com/webhooks/W1"
		portfolioURL = "https://discord.com/webhooks/W2"
		testTicker   = "AAPL"
		testCond     = "close > 100"
	)

	accum := &recordingAccum{}
	pq := &fakePortfolioQueriesFanout{
		portfolios: []db.ListPortfoliosForFanoutRow{
			{ID: "p1", Name: "P1", DiscordWebhook: textPF(sharedURL), Enabled: boolPF(true)},   // same as economy
			{ID: "p2", Name: "P2", DiscordWebhook: textPF(portfolioURL), Enabled: boolPF(true)}, // unique
		},
		stocks: []db.PortfolioStock{
			{PortfolioID: "p1", Ticker: testTicker},
			{PortfolioID: "p2", Ticker: testTicker},
		},
	}
	cq := &fakeCustomQueriesFanout{
		payload: map[string]interface{}{
			"ch1": map[string]interface{}{
				"webhook_url": sharedURL, // same as economy — should be deduped
				"condition":   testCond,
				"enabled":     true,
			},
		},
	}

	c := buildFanoutCommon(sharedURL, pq, cq, accum)

	a := makeAlert(testTicker, testCond)
	result := alert.CheckResult{AlertID: "test-id", Triggered: true}
	c.dispatchTriggered(context.Background(), a, result, "NYSE", "daily")

	urls := accum.URLs()
	if len(urls) != 2 {
		t.Fatalf("expected 2 Add calls (dedup), got %d: %v", len(urls), urls)
	}
	if countURL(urls, sharedURL) != 1 {
		t.Errorf("W1 should appear exactly once, got %d in %v", countURL(urls, sharedURL), urls)
	}
	if countURL(urls, portfolioURL) != 1 {
		t.Errorf("W2 should appear exactly once, got %d in %v", countURL(urls, portfolioURL), urls)
	}
}

// fakeQueriesDBTX implements db.DBTX for creating a stub *db.Queries.
// All methods panic — they should never be called in the fanout tests.
type fakeQueriesDBTX struct{}

func (f *fakeQueriesDBTX) Exec(_ context.Context, _ string, _ ...interface{}) (pgconn.CommandTag, error) {
	panic("fakeQueriesDBTX.Exec should not be called in fanout test")
}
func (f *fakeQueriesDBTX) Query(_ context.Context, _ string, _ ...interface{}) (pgx.Rows, error) {
	panic("fakeQueriesDBTX.Query should not be called in fanout test")
}
func (f *fakeQueriesDBTX) QueryRow(_ context.Context, _ string, _ ...interface{}) pgx.Row {
	panic("fakeQueriesDBTX.QueryRow should not be called in fanout test")
}
func (f *fakeQueriesDBTX) CopyFrom(_ context.Context, _ pgx.Identifier, _ []string, _ pgx.CopyFromSource) (int64, error) {
	panic("fakeQueriesDBTX.CopyFrom should not be called in fanout test")
}

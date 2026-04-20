package discord

import (
	"context"
	"errors"
	"log/slog"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgtype"

	db "stockalert/database/generated"
)

// fakePortfolioQueries is a test double for PortfolioQueriesDB that counts calls.
type fakePortfolioQueries struct {
	portfolios  []db.ListPortfoliosForFanoutRow
	stocks      []db.PortfolioStock
	loadCount   int
	returnError error
}

func (f *fakePortfolioQueries) ListPortfoliosForFanout(_ context.Context) ([]db.ListPortfoliosForFanoutRow, error) {
	f.loadCount++
	if f.returnError != nil {
		return nil, f.returnError
	}
	return f.portfolios, nil
}

func (f *fakePortfolioQueries) ListPortfolioStocks(_ context.Context) ([]db.PortfolioStock, error) {
	if f.returnError != nil {
		return nil, f.returnError
	}
	return f.stocks, nil
}

func textPgType(s string) pgtype.Text {
	return pgtype.Text{String: s, Valid: true}
}

func boolPgType(b bool) pgtype.Bool {
	return pgtype.Bool{Bool: b, Valid: true}
}

func newTestPortfolioResolver(q PortfolioQueriesDB, ttl time.Duration) *PortfolioResolver {
	return NewPortfolioResolverFromQueries(q, ttl, slog.Default())
}

func TestPortfolioResolver_Empty(t *testing.T) {
	fake := &fakePortfolioQueries{}
	r := newTestPortfolioResolver(fake, time.Hour)

	urls, err := r.ResolveWebhooks(context.Background(), "AAPL")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(urls) != 0 {
		t.Errorf("expected empty slice, got %v", urls)
	}
}

func TestPortfolioResolver_SingleMatch(t *testing.T) {
	fake := &fakePortfolioQueries{
		portfolios: []db.ListPortfoliosForFanoutRow{
			{ID: "p1", Name: "MyPortfolio", DiscordWebhook: textPgType("W1"), Enabled: boolPgType(true)},
		},
		stocks: []db.PortfolioStock{
			{PortfolioID: "p1", Ticker: "AAPL"},
		},
	}
	r := newTestPortfolioResolver(fake, time.Hour)

	urls, err := r.ResolveWebhooks(context.Background(), "AAPL")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(urls) != 1 || urls[0] != "W1" {
		t.Errorf("expected [W1], got %v", urls)
	}
}

func TestPortfolioResolver_MultiMatch(t *testing.T) {
	fake := &fakePortfolioQueries{
		portfolios: []db.ListPortfoliosForFanoutRow{
			{ID: "p1", Name: "Port1", DiscordWebhook: textPgType("W1"), Enabled: boolPgType(true)},
			{ID: "p2", Name: "Port2", DiscordWebhook: textPgType("W2"), Enabled: boolPgType(true)},
		},
		stocks: []db.PortfolioStock{
			{PortfolioID: "p1", Ticker: "AAPL"},
			{PortfolioID: "p2", Ticker: "AAPL"},
		},
	}
	r := newTestPortfolioResolver(fake, time.Hour)

	urls, err := r.ResolveWebhooks(context.Background(), "AAPL")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(urls) != 2 {
		t.Errorf("expected 2 URLs, got %v", urls)
	}
}

func TestPortfolioResolver_CaseInsensitive(t *testing.T) {
	fake := &fakePortfolioQueries{
		portfolios: []db.ListPortfoliosForFanoutRow{
			{ID: "p1", Name: "P1", DiscordWebhook: textPgType("W1"), Enabled: boolPgType(true)},
		},
		stocks: []db.PortfolioStock{
			{PortfolioID: "p1", Ticker: "aapl"}, // lowercase in DB
		},
	}
	r := newTestPortfolioResolver(fake, time.Hour)

	// Uppercase query should match lowercase stored ticker.
	urls, err := r.ResolveWebhooks(context.Background(), "AAPL")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(urls) != 1 || urls[0] != "W1" {
		t.Errorf("expected [W1] for uppercase query, got %v", urls)
	}

	// Also test reverse: stored uppercase, query lowercase.
	fake2 := &fakePortfolioQueries{
		portfolios: []db.ListPortfoliosForFanoutRow{
			{ID: "p1", Name: "P1", DiscordWebhook: textPgType("W1"), Enabled: boolPgType(true)},
		},
		stocks: []db.PortfolioStock{
			{PortfolioID: "p1", Ticker: "AAPL"},
		},
	}
	r2 := newTestPortfolioResolver(fake2, time.Hour)
	urls2, err := r2.ResolveWebhooks(context.Background(), "aapl")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(urls2) != 1 || urls2[0] != "W1" {
		t.Errorf("expected [W1] for lowercase query, got %v", urls2)
	}
}

func TestPortfolioResolver_ExchangeSuffix(t *testing.T) {
	fake := &fakePortfolioQueries{
		portfolios: []db.ListPortfoliosForFanoutRow{
			{ID: "p1", Name: "P1", DiscordWebhook: textPgType("W1"), Enabled: boolPgType(true)},
		},
		stocks: []db.PortfolioStock{
			{PortfolioID: "p1", Ticker: "SHEL.L"},
		},
	}
	r := newTestPortfolioResolver(fake, time.Hour)

	// Bare ticker should NOT match exchange-suffixed stored ticker.
	urls, err := r.ResolveWebhooks(context.Background(), "SHEL")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(urls) != 0 {
		t.Errorf("expected no match for SHEL, got %v", urls)
	}

	// Full exchange-suffixed ticker SHOULD match.
	urls2, err := r.ResolveWebhooks(context.Background(), "SHEL.L")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(urls2) != 1 || urls2[0] != "W1" {
		t.Errorf("expected [W1] for SHEL.L, got %v", urls2)
	}
}

func TestPortfolioResolver_DisabledExcluded(t *testing.T) {
	// The SQL WHERE clause filters out disabled portfolios; we verify the resolver
	// respects the absence of disabled portfolios in the query result.
	fake := &fakePortfolioQueries{
		// ListPortfoliosForFanout returns only enabled portfolios (SQL filters them).
		portfolios: []db.ListPortfoliosForFanoutRow{},
		stocks: []db.PortfolioStock{
			{PortfolioID: "p1", Ticker: "AAPL"},
		},
	}
	r := newTestPortfolioResolver(fake, time.Hour)

	urls, err := r.ResolveWebhooks(context.Background(), "AAPL")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(urls) != 0 {
		t.Errorf("expected no URLs for disabled portfolio, got %v", urls)
	}
}

func TestPortfolioResolver_BlankWebhookExcluded(t *testing.T) {
	// SQL WHERE clause also filters blank webhooks; verify resolver handles absent webhook.
	fake := &fakePortfolioQueries{
		// Portfolio with blank webhook is filtered by SQL, so query returns nothing.
		portfolios: []db.ListPortfoliosForFanoutRow{},
		stocks:     []db.PortfolioStock{{PortfolioID: "p1", Ticker: "AAPL"}},
	}
	r := newTestPortfolioResolver(fake, time.Hour)

	urls, err := r.ResolveWebhooks(context.Background(), "AAPL")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(urls) != 0 {
		t.Errorf("expected no URLs for blank webhook, got %v", urls)
	}
}

func TestPortfolioResolver_CacheHit(t *testing.T) {
	fake := &fakePortfolioQueries{
		portfolios: []db.ListPortfoliosForFanoutRow{
			{ID: "p1", Name: "P1", DiscordWebhook: textPgType("W1"), Enabled: boolPgType(true)},
		},
		stocks: []db.PortfolioStock{{PortfolioID: "p1", Ticker: "AAPL"}},
	}
	r := newTestPortfolioResolver(fake, time.Hour) // 1-hour TTL

	// First call: loads from DB.
	_, _ = r.ResolveWebhooks(context.Background(), "AAPL")
	countAfterFirst := fake.loadCount

	// Second call within TTL: must NOT hit DB again.
	_, _ = r.ResolveWebhooks(context.Background(), "AAPL")
	if fake.loadCount != countAfterFirst {
		t.Errorf("expected no reload within TTL, but loadCount went from %d to %d", countAfterFirst, fake.loadCount)
	}
}

func TestPortfolioResolver_CacheReload(t *testing.T) {
	fake := &fakePortfolioQueries{
		portfolios: []db.ListPortfoliosForFanoutRow{
			{ID: "p1", Name: "P1", DiscordWebhook: textPgType("W1"), Enabled: boolPgType(true)},
		},
		stocks: []db.PortfolioStock{{PortfolioID: "p1", Ticker: "AAPL"}},
	}
	r := newTestPortfolioResolver(fake, 1) // TTL = 1 nanosecond

	// First call loads.
	_, _ = r.ResolveWebhooks(context.Background(), "AAPL")
	countAfterFirst := fake.loadCount

	// After TTL has expired (1ns ago), next call must reload.
	time.Sleep(2 * time.Nanosecond)
	_, _ = r.ResolveWebhooks(context.Background(), "AAPL")
	if fake.loadCount <= countAfterFirst {
		t.Errorf("expected reload after TTL expired, but loadCount stayed at %d", fake.loadCount)
	}
}

func TestPortfolioResolver_LoadError(t *testing.T) {
	fake := &fakePortfolioQueries{
		returnError: errors.New("db down"),
	}
	r := newTestPortfolioResolver(fake, time.Hour)

	urls, err := r.ResolveWebhooks(context.Background(), "AAPL")
	if err == nil {
		t.Error("expected error from failed load")
	}
	if len(urls) != 0 {
		t.Errorf("expected empty urls on error, got %v", urls)
	}
}

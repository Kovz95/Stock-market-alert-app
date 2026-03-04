package price

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	db "stockalert/database/generated"
)

// Integration test: full price update flow with mocked FMP against a real database.
// Skip unless DATABASE_URL is set (e.g. CI or local postgres).
func TestUpdater_UpdateForExchange_Integration(t *testing.T) {
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		t.Skip("DATABASE_URL not set, skipping integration test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	pool, err := pgxpool.New(ctx, dbURL)
	if err != nil {
		t.Fatalf("connect: %v", err)
	}
	defer pool.Close()
	if err := pool.Ping(ctx); err != nil {
		t.Fatalf("ping: %v", err)
	}

	queries := db.New(pool)

	// Mock FMP: return a few daily rows for a ticker so CopyDailyPrices has something to insert.
	mock := &MockFMP{
		Daily: map[string][]DailyRow{
			"TEST_TICKER": {
				{Date: "2025-01-02", Open: 100, High: 101, Low: 99, Close: 100.5, Volume: 1e6},
				{Date: "2025-01-03", Open: 100.5, High: 102, Low: 100, Close: 101, Volume: 1.1e6},
			},
		},
	}

	updater := NewUpdater(queries, mock, nil, 0)

	// UpdateForExchange loads tickers from DB; if none for "NYSE" we get Total=0 and no FMP calls.
	// If the test DB has tickers for NYSE, mock will be used for those tickers (we only seeded TEST_TICKER).
	stats, err := updater.UpdateForExchange(ctx, "NYSE", "daily")
	if err != nil {
		t.Fatalf("UpdateForExchange: %v", err)
	}
	if stats == nil {
		t.Fatal("expected non-nil stats")
	}
	// Just assert no crash; Total depends on test DB contents
	_ = stats.Total
	_ = stats.Updated
	_ = stats.Failed
}

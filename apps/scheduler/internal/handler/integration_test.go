package handler

import (
	"context"
	"log/slog"
	"os"
	"testing"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	"stockalert/alert"
	db "stockalert/database/generated"
	"stockalert/discord"
	"stockalert/indicator"

	"stockalert/apps/scheduler/internal/price"
	"stockalert/apps/scheduler/internal/status"
)

// Integration test: full job execution (Execute) with mocked FMP against real DB.
// Skip unless DATABASE_URL is set.
func TestCommon_Execute_Integration(t *testing.T) {
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		t.Skip("DATABASE_URL not set, skipping integration test")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
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
	router, _ := discord.NewRouter(ctx, queries)
	registry := indicator.NewDefaultRegistry()
	checker := alert.NewChecker(queries, registry)
	notifier := discord.NewNotifier()
	accum := discord.NewAccumulator(notifier)
	mockFMP := &price.MockFMP{Daily: map[string][]price.DailyRow{}}
	updater := price.NewUpdater(queries, mockFMP, nil, 0)
	statusMgr := status.NewManager(queries, nil)

	common := &Common{
		Queries:       queries,
		Checker:       checker,
		Router:        router,
		Accum:         accum,
		Notifier:      notifier,
		Updater:       updater,
		Status:        statusMgr,
		Logger:        slog.Default(),
		JobTimeout:    5 * time.Minute,
		WebhookDaily:  "",
		WebhookWeekly: "",
		WebhookHourly: "",
		ShadowDir:     "",
	}

	// Run daily job for NYSE; mock FMP returns no data so no prices updated, alerts still run from DB cache.
	_, _, err = common.Execute(ctx, "NYSE", "daily", nil)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
}

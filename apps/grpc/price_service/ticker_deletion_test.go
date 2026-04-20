package main

import (
	"context"
	"fmt"
	"os"
	"testing"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"
	"github.com/jackc/pgx/v5/pgxpool"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	db "stockalert/database/generated"
	pricev1 "stockalert/gen/go/price/v1"
)

// testPool returns a pgxpool connected to the DATABASE_URL env var, or skips the test.
func testPool(t *testing.T) *pgxpool.Pool {
	t.Helper()
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		t.Skip("DATABASE_URL not set; skipping integration test")
	}
	pool, err := pgxpool.New(context.Background(), dbURL)
	if err != nil {
		t.Fatalf("failed to create pool: %v", err)
	}
	t.Cleanup(pool.Close)
	return pool
}

// testServer returns a minimal *Server backed by the test pool (no updater/logger needed).
func testServer(pool *pgxpool.Pool) *Server {
	return &Server{pool: pool}
}

// seedZZDEAD inserts one row for ticker "ZZDEAD" into every reference table inside tx.
// Returns the expected counts we seeded.
func seedZZDEAD(t *testing.T, ctx context.Context, tx pgx.Tx) {
	t.Helper()
	ticker := "ZZDEAD"
	ttext := pgtype.Text{String: ticker, Valid: true}

	queries := []struct {
		name string
		sql  string
		args []interface{}
	}{
		{"stock_metadata", `INSERT INTO stock_metadata (symbol, name) VALUES ($1, 'Dead Ticker') ON CONFLICT DO NOTHING`, []interface{}{ticker}},
		{"ticker_metadata", `INSERT INTO ticker_metadata (ticker, exchange) VALUES ($1, 'TEST') ON CONFLICT DO NOTHING`, []interface{}{ticker}},
		{"daily_prices", `INSERT INTO daily_prices (ticker, date, close) VALUES ($1, '2020-01-01', 100.0) ON CONFLICT DO NOTHING`, []interface{}{ticker}},
		{"hourly_prices", `INSERT INTO hourly_prices (ticker, datetime, close) VALUES ($1, '2020-01-01 10:00:00+00', 100.0) ON CONFLICT DO NOTHING`, []interface{}{ticker}},
		{"weekly_prices", `INSERT INTO weekly_prices (ticker, week_ending, close) VALUES ($1, '2020-01-05', 100.0) ON CONFLICT DO NOTHING`, []interface{}{ticker}},
		{"continuous_prices", `INSERT INTO continuous_prices (symbol, date, close) VALUES ($1, '2020-01-01', 100.0) ON CONFLICT DO NOTHING`, []interface{}{ticker}},
		{"daily_move_stats", `INSERT INTO daily_move_stats (ticker, date) VALUES ($1, '2020-01-01') ON CONFLICT DO NOTHING`, []interface{}{ticker}},
		{"futures_metadata", `INSERT INTO futures_metadata (symbol, name) VALUES ($1, 'Dead Future') ON CONFLICT DO NOTHING`, []interface{}{ticker}},
		{"alerts_direct", `INSERT INTO alerts (alert_id, name, ticker) VALUES (gen_random_uuid(), 'Dead Alert', $1) ON CONFLICT DO NOTHING`, []interface{}{ttext}},
		{"alerts_ratio", `INSERT INTO alerts (alert_id, name, ticker1) VALUES (gen_random_uuid(), 'Dead Ratio', $1) ON CONFLICT DO NOTHING`, []interface{}{ttext}},
		{"alert_audits", `INSERT INTO alert_audits (timestamp, alert_id, ticker, evaluation_type) VALUES (NOW(), gen_random_uuid()::text, $1, 'test') ON CONFLICT DO NOTHING`, []interface{}{ticker}},
		{"portfolio_stocks", `INSERT INTO portfolios (id, name) VALUES ('test-portfolio-zzdead', 'Test') ON CONFLICT DO NOTHING`, []interface{}{}},
		{"portfolio_stocks_row", `INSERT INTO portfolio_stocks (portfolio_id, ticker) VALUES ('test-portfolio-zzdead', $1) ON CONFLICT DO NOTHING`, []interface{}{ticker}},
	}

	for _, q := range queries {
		if _, err := tx.Exec(ctx, q.sql, q.args...); err != nil {
			t.Fatalf("seed %s: %v", q.name, err)
		}
	}
}

// countZZDEAD returns the total number of rows across all tables for "ZZDEAD".
func countZZDEAD(t *testing.T, ctx context.Context, pool *pgxpool.Pool) int64 {
	t.Helper()
	conn, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	defer conn.Release()
	row, err := db.New(conn).CountTickerReferences(ctx, "ZZDEAD")
	if err != nil {
		t.Fatalf("count: %v", err)
	}
	return row.StockMetadata + row.TickerMetadata + row.DailyPrices + row.HourlyPrices +
		row.WeeklyPrices + row.ContinuousPrices + row.DailyMoveStats + row.FuturesMetadata +
		row.AlertsDirect + row.AlertsRatio + row.AlertAudits + row.PortfolioStocks
}

func TestDeleteTicker_FullCascade(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()

	// Seed inside a tx, then commit so the service can see the rows.
	tx, err := pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		t.Fatalf("begin: %v", err)
	}
	seedZZDEAD(t, ctx, tx)
	if err := tx.Commit(ctx); err != nil {
		t.Fatalf("commit seed: %v", err)
	}

	// Verify seed is visible.
	if total := countZZDEAD(t, ctx, pool); total == 0 {
		t.Fatal("seed produced zero rows")
	}

	t.Cleanup(func() {
		// Best-effort cleanup in case DeleteTicker didn't get them all.
		conn, _ := pool.Acquire(ctx)
		if conn != nil {
			conn.Exec(ctx, "DELETE FROM portfolio_stocks WHERE ticker = 'ZZDEAD'")
			conn.Exec(ctx, "DELETE FROM portfolios WHERE id = 'test-portfolio-zzdead'")
			conn.Exec(ctx, "DELETE FROM alert_audits WHERE ticker = 'ZZDEAD'")
			conn.Exec(ctx, "DELETE FROM alerts WHERE ticker = 'ZZDEAD' OR ticker1 = 'ZZDEAD' OR ticker2 = 'ZZDEAD'")
			conn.Exec(ctx, "DELETE FROM daily_move_stats WHERE ticker = 'ZZDEAD'")
			conn.Exec(ctx, "DELETE FROM daily_prices WHERE ticker = 'ZZDEAD'")
			conn.Exec(ctx, "DELETE FROM hourly_prices WHERE ticker = 'ZZDEAD'")
			conn.Exec(ctx, "DELETE FROM weekly_prices WHERE ticker = 'ZZDEAD'")
			conn.Exec(ctx, "DELETE FROM continuous_prices WHERE symbol = 'ZZDEAD'")
			conn.Exec(ctx, "DELETE FROM futures_metadata WHERE symbol = 'ZZDEAD'")
			conn.Exec(ctx, "DELETE FROM ticker_metadata WHERE ticker = 'ZZDEAD'")
			conn.Exec(ctx, "DELETE FROM stock_metadata WHERE symbol = 'ZZDEAD'")
			conn.Release()
		}
	})

	s := testServer(pool)
	resp, err := s.DeleteTicker(ctx, &pricev1.DeleteTickerRequest{Ticker: "ZZDEAD"})
	if err != nil {
		t.Fatalf("DeleteTicker rpc error: %v", err)
	}
	if !resp.Success {
		t.Fatalf("expected success=true, got error: %s", resp.ErrorMessage)
	}
	if resp.Ticker != "ZZDEAD" {
		t.Errorf("ticker mismatch: got %q", resp.Ticker)
	}

	// All tables should now have zero rows for ZZDEAD.
	if total := countZZDEAD(t, ctx, pool); total != 0 {
		t.Errorf("expected 0 rows after delete, got %d", total)
	}

	// Counts returned should be > 0 (we seeded at least one row in each).
	if resp.Counts == nil {
		t.Fatal("expected non-nil counts")
	}
	if resp.Counts.StockMetadata != 1 {
		t.Errorf("stock_metadata count: want 1, got %d", resp.Counts.StockMetadata)
	}
	if resp.Counts.TickerMetadata != 1 {
		t.Errorf("ticker_metadata count: want 1, got %d", resp.Counts.TickerMetadata)
	}
	if resp.Counts.AlertsDirect != 1 {
		t.Errorf("alerts_direct count: want 1, got %d", resp.Counts.AlertsDirect)
	}
	if resp.Counts.AlertsRatio != 1 {
		t.Errorf("alerts_ratio count: want 1, got %d", resp.Counts.AlertsRatio)
	}
}

func TestDeleteTicker_NonExistent(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()

	s := testServer(pool)
	resp, err := s.DeleteTicker(ctx, &pricev1.DeleteTickerRequest{Ticker: "ZZNONEXISTENT999"})
	if err != nil {
		t.Fatalf("unexpected rpc error: %v", err)
	}
	if !resp.Success {
		t.Errorf("expected success=true for non-existent ticker, got: %s", resp.ErrorMessage)
	}
	if resp.Counts == nil {
		t.Fatal("expected non-nil counts")
	}
	total := resp.Counts.StockMetadata + resp.Counts.TickerMetadata + resp.Counts.DailyPrices +
		resp.Counts.HourlyPrices + resp.Counts.WeeklyPrices + resp.Counts.ContinuousPrices +
		resp.Counts.DailyMoveStats + resp.Counts.FuturesMetadata + resp.Counts.AlertsDirect +
		resp.Counts.AlertsRatio + resp.Counts.AlertAudits + resp.Counts.PortfolioStocks
	if total != 0 {
		t.Errorf("expected all-zero counts for non-existent ticker, got total=%d", total)
	}
}

func TestDeleteTicker_InvalidArgument(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()

	s := testServer(pool)
	_, err := s.DeleteTicker(ctx, &pricev1.DeleteTickerRequest{Ticker: ""})
	if err == nil {
		t.Fatal("expected error for empty ticker, got nil")
	}
	st, ok := status.FromError(err)
	if !ok {
		t.Fatalf("expected grpc status error, got %T: %v", err, err)
	}
	if st.Code() != codes.InvalidArgument {
		t.Errorf("expected codes.InvalidArgument, got %v", st.Code())
	}
}

func TestDeleteTicker_RollbackOnError(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()

	// Seed ZZDEAD2 so there's something to delete.
	tx, err := pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		t.Fatalf("begin: %v", err)
	}
	ticker := "ZZDEAD2"
	if _, err := tx.Exec(ctx, `INSERT INTO stock_metadata (symbol, name) VALUES ($1, 'Test') ON CONFLICT DO NOTHING`, ticker); err != nil {
		tx.Rollback(ctx)
		t.Fatalf("seed stock_metadata: %v", err)
	}
	if _, err := tx.Exec(ctx, `INSERT INTO ticker_metadata (ticker) VALUES ($1) ON CONFLICT DO NOTHING`, ticker); err != nil {
		tx.Rollback(ctx)
		t.Fatalf("seed ticker_metadata: %v", err)
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatalf("commit seed: %v", err)
	}
	t.Cleanup(func() {
		conn, _ := pool.Acquire(ctx)
		if conn != nil {
			conn.Exec(ctx, "DELETE FROM ticker_metadata WHERE ticker = $1", ticker)
			conn.Exec(ctx, "DELETE FROM stock_metadata WHERE symbol = $1", ticker)
			conn.Release()
		}
	})

	// Use an errorInjectingServer that wraps the pool to fail mid-cascade.
	s := &errorInjectingServer{pool: pool, failAfterStep: 3} // fail at DailyMoveStats step
	resp, err := s.DeleteTicker(ctx, &pricev1.DeleteTickerRequest{Ticker: ticker})
	if err != nil {
		// An RPC-level error (not a response error) means something unexpected happened.
		t.Fatalf("unexpected rpc error: %v", err)
	}
	if resp.Success {
		t.Fatal("expected success=false when error injected mid-cascade")
	}
	if resp.ErrorMessage == "" {
		t.Error("expected non-empty error_message on failure")
	}

	// Verify rollback: rows should still exist.
	conn, err := pool.Acquire(ctx)
	if err != nil {
		t.Fatalf("acquire: %v", err)
	}
	defer conn.Release()
	var cnt int
	if err := conn.QueryRow(ctx, "SELECT COUNT(*) FROM stock_metadata WHERE symbol = $1", ticker).Scan(&cnt); err != nil {
		t.Fatalf("count stock_metadata: %v", err)
	}
	if cnt == 0 {
		t.Error("rollback failed: stock_metadata row was deleted despite error mid-cascade")
	}
}

func TestPreviewDeleteTicker_Counts(t *testing.T) {
	pool := testPool(t)
	ctx := context.Background()

	ticker := "ZZPREVIEW1"
	tx, err := pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		t.Fatalf("begin: %v", err)
	}
	ttext := pgtype.Text{String: ticker, Valid: true}
	for _, q := range []struct {
		sql  string
		args []interface{}
	}{
		{`INSERT INTO stock_metadata (symbol, name) VALUES ($1, 'Preview') ON CONFLICT DO NOTHING`, []interface{}{ticker}},
		{`INSERT INTO ticker_metadata (ticker) VALUES ($1) ON CONFLICT DO NOTHING`, []interface{}{ticker}},
		{`INSERT INTO daily_prices (ticker, date, close) VALUES ($1, '2020-01-01', 1.0) ON CONFLICT DO NOTHING`, []interface{}{ticker}},
		{`INSERT INTO alerts (alert_id, name, ticker) VALUES (gen_random_uuid(), 'PA', $1) ON CONFLICT DO NOTHING`, []interface{}{ttext}},
	} {
		if _, err := tx.Exec(ctx, q.sql, q.args...); err != nil {
			tx.Rollback(ctx)
			t.Fatalf("seed: %v", err)
		}
	}
	if err := tx.Commit(ctx); err != nil {
		t.Fatalf("commit: %v", err)
	}
	t.Cleanup(func() {
		conn, _ := pool.Acquire(ctx)
		if conn != nil {
			conn.Exec(ctx, "DELETE FROM alerts WHERE ticker = $1", ttext)
			conn.Exec(ctx, "DELETE FROM daily_prices WHERE ticker = $1", ticker)
			conn.Exec(ctx, "DELETE FROM ticker_metadata WHERE ticker = $1", ticker)
			conn.Exec(ctx, "DELETE FROM stock_metadata WHERE symbol = $1", ticker)
			conn.Release()
		}
	})

	s := testServer(pool)
	resp, err := s.PreviewDeleteTicker(ctx, &pricev1.PreviewDeleteTickerRequest{Ticker: ticker})
	if err != nil {
		t.Fatalf("rpc error: %v", err)
	}
	if !resp.Exists {
		t.Error("expected exists=true")
	}
	if resp.Counts == nil {
		t.Fatal("nil counts")
	}
	if resp.Counts.StockMetadata != 1 {
		t.Errorf("stock_metadata: want 1, got %d", resp.Counts.StockMetadata)
	}
	if resp.Counts.TickerMetadata != 1 {
		t.Errorf("ticker_metadata: want 1, got %d", resp.Counts.TickerMetadata)
	}
	if resp.Counts.DailyPrices != 1 {
		t.Errorf("daily_prices: want 1, got %d", resp.Counts.DailyPrices)
	}
	if resp.Counts.AlertsDirect != 1 {
		t.Errorf("alerts_direct: want 1, got %d", resp.Counts.AlertsDirect)
	}

	// exists=false for ticker not in metadata tables.
	resp2, err := s.PreviewDeleteTicker(ctx, &pricev1.PreviewDeleteTickerRequest{Ticker: "ZZNOTHERE999"})
	if err != nil {
		t.Fatalf("rpc error: %v", err)
	}
	if resp2.Exists {
		t.Error("expected exists=false for unknown ticker")
	}
}

// errorInjectingServer wraps Server.DeleteTicker to fail at a specific step by closing
// the transaction connection after a certain number of executions.
// This is a minimal inline test helper — not production code.
type errorInjectingServer struct {
	pool          *pgxpool.Pool
	failAfterStep int
}

func (s *errorInjectingServer) DeleteTicker(ctx context.Context, req *pricev1.DeleteTickerRequest) (*pricev1.DeleteTickerResponse, error) {
	if req.Ticker == "" {
		return nil, status.Errorf(codes.InvalidArgument, "ticker is required")
	}

	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return nil, status.Errorf(codes.Internal, "begin_tx: %v", err)
	}
	defer tx.Rollback(ctx) //nolint:errcheck

	ticker := req.Ticker
	tickerText := pgtype.Text{String: ticker, Valid: true}
	q := db.New(tx)
	step := 0

	maybeError := func(name string) error {
		step++
		if step == s.failAfterStep {
			return fmt.Errorf("injected error at step %d (%s)", step, name)
		}
		return nil
	}

	if err := maybeError("portfolio_stocks"); err != nil {
		return &pricev1.DeleteTickerResponse{Success: false, ErrorMessage: err.Error()}, nil
	}
	if _, err := q.DeleteTickerFromPortfolioStocks(ctx, ticker); err != nil {
		return &pricev1.DeleteTickerResponse{Success: false, ErrorMessage: "delete_from_portfolio_stocks: " + err.Error()}, nil
	}

	if err := maybeError("alert_audits"); err != nil {
		return &pricev1.DeleteTickerResponse{Success: false, ErrorMessage: err.Error()}, nil
	}
	if _, err := q.DeleteTickerFromAlertAudits(ctx, ticker); err != nil {
		return &pricev1.DeleteTickerResponse{Success: false, ErrorMessage: "delete_from_alert_audits: " + err.Error()}, nil
	}

	if err := maybeError("alerts_direct"); err != nil {
		return &pricev1.DeleteTickerResponse{Success: false, ErrorMessage: err.Error()}, nil
	}
	if _, err := q.DeleteTickerFromAlertsDirect(ctx, tickerText); err != nil {
		return &pricev1.DeleteTickerResponse{Success: false, ErrorMessage: "delete_from_alerts_direct: " + err.Error()}, nil
	}

	if err := maybeError("daily_move_stats"); err != nil {
		return &pricev1.DeleteTickerResponse{Success: false, ErrorMessage: err.Error()}, nil
	}
	if _, err := q.DeleteTickerFromDailyMoveStats(ctx, ticker); err != nil {
		return &pricev1.DeleteTickerResponse{Success: false, ErrorMessage: "delete_from_daily_move_stats: " + err.Error()}, nil
	}

	// Remaining steps not needed for rollback test.
	if _, err := q.DeleteTickerFromAlertsRatio(ctx, tickerText); err != nil {
		return &pricev1.DeleteTickerResponse{Success: false, ErrorMessage: err.Error()}, nil
	}
	if _, err := q.DeleteTickerFromDailyPrices(ctx, ticker); err != nil {
		return &pricev1.DeleteTickerResponse{Success: false, ErrorMessage: err.Error()}, nil
	}
	if _, err := q.DeleteTickerFromHourlyPrices(ctx, ticker); err != nil {
		return &pricev1.DeleteTickerResponse{Success: false, ErrorMessage: err.Error()}, nil
	}
	if _, err := q.DeleteTickerFromWeeklyPrices(ctx, ticker); err != nil {
		return &pricev1.DeleteTickerResponse{Success: false, ErrorMessage: err.Error()}, nil
	}
	if _, err := q.DeleteTickerFromContinuousPrices(ctx, ticker); err != nil {
		return &pricev1.DeleteTickerResponse{Success: false, ErrorMessage: err.Error()}, nil
	}
	if _, err := q.DeleteTickerFromFuturesMetadata(ctx, ticker); err != nil {
		return &pricev1.DeleteTickerResponse{Success: false, ErrorMessage: err.Error()}, nil
	}
	if _, err := q.DeleteTickerFromTickerMetadata(ctx, ticker); err != nil {
		return &pricev1.DeleteTickerResponse{Success: false, ErrorMessage: err.Error()}, nil
	}
	if _, err := q.DeleteTickerFromStockMetadata(ctx, ticker); err != nil {
		return &pricev1.DeleteTickerResponse{Success: false, ErrorMessage: err.Error()}, nil
	}

	if err := tx.Commit(ctx); err != nil {
		return &pricev1.DeleteTickerResponse{Success: false, ErrorMessage: "commit: " + err.Error()}, nil
	}
	return &pricev1.DeleteTickerResponse{Success: true, Ticker: ticker}, nil
}

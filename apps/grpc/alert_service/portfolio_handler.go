package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"strings"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	alertv1 "stockalert/gen/go/alert/v1"
)

func shortID() string {
	b := make([]byte, 4)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// loadPortfolio is a helper that loads a single portfolio (with tickers) from the DB.
func (s *Server) loadPortfolio(ctx context.Context, portfolioID string) (*alertv1.Portfolio, error) {
	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	var name, webhook, createdDate, lastUpdated string
	var enabled bool
	err = conn.QueryRow(ctx, `
		SELECT name, COALESCE(discord_webhook,''), COALESCE(enabled, true),
		       COALESCE(created_date::text,''), COALESCE(last_updated::text,'')
		FROM portfolios WHERE id = $1
	`, portfolioID).Scan(&name, &webhook, &enabled, &createdDate, &lastUpdated)
	if err != nil {
		return nil, status.Errorf(codes.NotFound, "portfolio %q not found", portfolioID)
	}

	tickerRows, err := conn.Query(ctx, `
		SELECT ticker FROM portfolio_stocks WHERE portfolio_id = $1 ORDER BY ticker
	`, portfolioID)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "portfolio tickers query: %v", err)
	}
	defer tickerRows.Close()

	var tickers []string
	for tickerRows.Next() {
		var t string
		if err := tickerRows.Scan(&t); err != nil {
			return nil, status.Errorf(codes.Internal, "scan portfolio ticker: %v", err)
		}
		tickers = append(tickers, t)
	}
	if err := tickerRows.Err(); err != nil {
		return nil, status.Errorf(codes.Internal, "portfolio tickers rows: %v", err)
	}

	return &alertv1.Portfolio{
		PortfolioId:    portfolioID,
		Name:           name,
		Tickers:        tickers,
		DiscordWebhook: webhook,
		Enabled:        enabled,
		CreatedDate:    createdDate,
		LastUpdated:    lastUpdated,
	}, nil
}

func (s *Server) GetPortfolio(ctx context.Context, req *alertv1.GetPortfolioRequest) (*alertv1.GetPortfolioResponse, error) {
	pid := strings.TrimSpace(req.GetPortfolioId())
	if pid == "" {
		return nil, status.Error(codes.InvalidArgument, "portfolio_id is required")
	}

	p, err := s.loadPortfolio(ctx, pid)
	if err != nil {
		return nil, err
	}
	return &alertv1.GetPortfolioResponse{Portfolio: p}, nil
}

func (s *Server) CreatePortfolio(ctx context.Context, req *alertv1.CreatePortfolioRequest) (*alertv1.CreatePortfolioResponse, error) {
	name := strings.TrimSpace(req.GetName())
	if name == "" {
		return nil, status.Error(codes.InvalidArgument, "name is required")
	}
	webhook := strings.TrimSpace(req.GetDiscordWebhook())

	portfolioID := shortID()
	now := time.Now().UTC().Format(time.RFC3339)

	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	_, err = conn.Exec(ctx, `
		INSERT INTO portfolios (id, name, discord_webhook, enabled, created_date, last_updated)
		VALUES ($1, $2, $3, true, $4, $5)
	`, portfolioID, name, webhook, now, now)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "create portfolio: %v", err)
	}

	p, err := s.loadPortfolio(ctx, portfolioID)
	if err != nil {
		return nil, err
	}
	return &alertv1.CreatePortfolioResponse{Portfolio: p}, nil
}

func (s *Server) UpdatePortfolio(ctx context.Context, req *alertv1.UpdatePortfolioRequest) (*alertv1.UpdatePortfolioResponse, error) {
	pid := strings.TrimSpace(req.GetPortfolioId())
	if pid == "" {
		return nil, status.Error(codes.InvalidArgument, "portfolio_id is required")
	}

	name := strings.TrimSpace(req.GetName())
	webhook := strings.TrimSpace(req.GetDiscordWebhook())
	enabled := req.GetEnabled()
	now := time.Now().UTC().Format(time.RFC3339)

	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	tag, err := conn.Exec(ctx, `
		UPDATE portfolios
		SET name = $2, discord_webhook = $3, enabled = $4, last_updated = $5
		WHERE id = $1
	`, pid, name, webhook, enabled, now)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "update portfolio: %v", err)
	}
	if tag.RowsAffected() == 0 {
		return nil, status.Errorf(codes.NotFound, "portfolio %q not found", pid)
	}

	p, err := s.loadPortfolio(ctx, pid)
	if err != nil {
		return nil, err
	}
	return &alertv1.UpdatePortfolioResponse{Portfolio: p}, nil
}

func (s *Server) DeletePortfolio(ctx context.Context, req *alertv1.DeletePortfolioRequest) (*alertv1.DeletePortfolioResponse, error) {
	pid := strings.TrimSpace(req.GetPortfolioId())
	if pid == "" {
		return nil, status.Error(codes.InvalidArgument, "portfolio_id is required")
	}

	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	_, err = conn.Exec(ctx, `DELETE FROM portfolio_stocks WHERE portfolio_id = $1`, pid)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "delete portfolio stocks: %v", err)
	}

	tag, err := conn.Exec(ctx, `DELETE FROM portfolios WHERE id = $1`, pid)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "delete portfolio: %v", err)
	}
	if tag.RowsAffected() == 0 {
		return nil, status.Errorf(codes.NotFound, "portfolio %q not found", pid)
	}

	return &alertv1.DeletePortfolioResponse{}, nil
}

func (s *Server) AddStocksToPortfolio(ctx context.Context, req *alertv1.AddStocksToPortfolioRequest) (*alertv1.AddStocksToPortfolioResponse, error) {
	pid := strings.TrimSpace(req.GetPortfolioId())
	if pid == "" {
		return nil, status.Error(codes.InvalidArgument, "portfolio_id is required")
	}
	tickers := req.GetTickers()
	if len(tickers) == 0 {
		return nil, status.Error(codes.InvalidArgument, "tickers is required")
	}

	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	// Verify portfolio exists
	var exists bool
	err = conn.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM portfolios WHERE id = $1)`, pid).Scan(&exists)
	if err != nil || !exists {
		return nil, status.Errorf(codes.NotFound, "portfolio %q not found", pid)
	}

	// Insert tickers, ignoring duplicates
	for _, ticker := range tickers {
		t := strings.TrimSpace(strings.ToUpper(ticker))
		if t == "" {
			continue
		}
		_, err = conn.Exec(ctx, `
			INSERT INTO portfolio_stocks (portfolio_id, ticker)
			VALUES ($1, $2)
			ON CONFLICT (portfolio_id, ticker) DO NOTHING
		`, pid, t)
		if err != nil {
			return nil, status.Errorf(codes.Internal, "add stock %q: %v", t, err)
		}
	}

	// Update last_updated
	now := time.Now().UTC().Format(time.RFC3339)
	_, err = conn.Exec(ctx, `UPDATE portfolios SET last_updated = $2 WHERE id = $1`, pid, now)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "update last_updated: %v", err)
	}

	p, loadErr := s.loadPortfolio(ctx, pid)
	if loadErr != nil {
		return nil, loadErr
	}
	return &alertv1.AddStocksToPortfolioResponse{Portfolio: p}, nil
}

func (s *Server) RemoveStocksFromPortfolio(ctx context.Context, req *alertv1.RemoveStocksFromPortfolioRequest) (*alertv1.RemoveStocksFromPortfolioResponse, error) {
	pid := strings.TrimSpace(req.GetPortfolioId())
	if pid == "" {
		return nil, status.Error(codes.InvalidArgument, "portfolio_id is required")
	}
	tickers := req.GetTickers()
	if len(tickers) == 0 {
		return nil, status.Error(codes.InvalidArgument, "tickers is required")
	}

	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	// Build parameterized DELETE
	args := []interface{}{pid}
	placeholders := make([]string, len(tickers))
	for i, ticker := range tickers {
		t := strings.TrimSpace(strings.ToUpper(ticker))
		args = append(args, t)
		placeholders[i] = fmt.Sprintf("$%d", i+2)
	}

	query := fmt.Sprintf(`DELETE FROM portfolio_stocks WHERE portfolio_id = $1 AND ticker IN (%s)`,
		strings.Join(placeholders, ","))
	_, err = conn.Exec(ctx, query, args...)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "remove stocks: %v", err)
	}

	// Update last_updated
	now := time.Now().UTC().Format(time.RFC3339)
	_, err = conn.Exec(ctx, `UPDATE portfolios SET last_updated = $2 WHERE id = $1`, pid, now)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "update last_updated: %v", err)
	}

	p, loadErr := s.loadPortfolio(ctx, pid)
	if loadErr != nil {
		return nil, loadErr
	}
	return &alertv1.RemoveStocksFromPortfolioResponse{Portfolio: p}, nil
}

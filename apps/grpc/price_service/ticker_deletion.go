package main

import (
	"context"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	db "stockalert/database/generated"
	pricev1 "stockalert/gen/go/price/v1"
)

func (s *Server) PreviewDeleteTicker(ctx context.Context, req *pricev1.PreviewDeleteTickerRequest) (*pricev1.PreviewDeleteTickerResponse, error) {
	if req.Ticker == "" {
		return nil, status.Errorf(codes.InvalidArgument, "ticker is required")
	}

	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	q := db.New(conn)

	exists, err := q.TickerExists(ctx, req.Ticker)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "ticker_exists: %v", err)
	}

	counts, err := q.CountTickerReferences(ctx, req.Ticker)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "count_ticker_references: %v", err)
	}

	return &pricev1.PreviewDeleteTickerResponse{
		Ticker: req.Ticker,
		Exists: exists,
		Counts: countRowToCounts(counts),
	}, nil
}

func (s *Server) DeleteTicker(ctx context.Context, req *pricev1.DeleteTickerRequest) (*pricev1.DeleteTickerResponse, error) {
	if req.Ticker == "" {
		return nil, status.Errorf(codes.InvalidArgument, "ticker is required")
	}

	tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})
	if err != nil {
		return nil, status.Errorf(codes.Internal, "begin_tx: %v", err)
	}
	defer tx.Rollback(ctx) //nolint:errcheck

	q := db.New(tx)
	var c pricev1.TickerDeletionCounts

	ticker := req.Ticker
	tickerText := pgtype.Text{String: ticker, Valid: true}

	n, err := q.DeleteTickerFromPortfolioStocks(ctx, ticker)
	if err != nil {
		return failDelete("delete_from_portfolio_stocks", err), nil
	}
	c.PortfolioStocks = n

	n, err = q.DeleteTickerFromAlertAudits(ctx, ticker)
	if err != nil {
		return failDelete("delete_from_alert_audits", err), nil
	}
	c.AlertAudits = n

	n, err = q.DeleteTickerFromAlertsDirect(ctx, tickerText)
	if err != nil {
		return failDelete("delete_from_alerts_direct", err), nil
	}
	c.AlertsDirect = n

	n, err = q.DeleteTickerFromAlertsRatio(ctx, tickerText)
	if err != nil {
		return failDelete("delete_from_alerts_ratio", err), nil
	}
	c.AlertsRatio = n

	n, err = q.DeleteTickerFromDailyMoveStats(ctx, ticker)
	if err != nil {
		return failDelete("delete_from_daily_move_stats", err), nil
	}
	c.DailyMoveStats = n

	n, err = q.DeleteTickerFromDailyPrices(ctx, ticker)
	if err != nil {
		return failDelete("delete_from_daily_prices", err), nil
	}
	c.DailyPrices = n

	n, err = q.DeleteTickerFromHourlyPrices(ctx, ticker)
	if err != nil {
		return failDelete("delete_from_hourly_prices", err), nil
	}
	c.HourlyPrices = n

	n, err = q.DeleteTickerFromWeeklyPrices(ctx, ticker)
	if err != nil {
		return failDelete("delete_from_weekly_prices", err), nil
	}
	c.WeeklyPrices = n

	n, err = q.DeleteTickerFromContinuousPrices(ctx, ticker)
	if err != nil {
		return failDelete("delete_from_continuous_prices", err), nil
	}
	c.ContinuousPrices = n

	n, err = q.DeleteTickerFromFuturesMetadata(ctx, ticker)
	if err != nil {
		return failDelete("delete_from_futures_metadata", err), nil
	}
	c.FuturesMetadata = n

	n, err = q.DeleteTickerFromTickerMetadata(ctx, ticker)
	if err != nil {
		return failDelete("delete_from_ticker_metadata", err), nil
	}
	c.TickerMetadata = n

	n, err = q.DeleteTickerFromStockMetadata(ctx, ticker)
	if err != nil {
		return failDelete("delete_from_stock_metadata", err), nil
	}
	c.StockMetadata = n

	if err := tx.Commit(ctx); err != nil {
		return &pricev1.DeleteTickerResponse{
			Success:      false,
			ErrorMessage: fmt.Sprintf("commit: %v", err),
			Ticker:       ticker,
		}, nil
	}

	return &pricev1.DeleteTickerResponse{
		Success: true,
		Ticker:  ticker,
		Counts:  &c,
	}, nil
}

func failDelete(step string, err error) *pricev1.DeleteTickerResponse {
	return &pricev1.DeleteTickerResponse{
		Success:      false,
		ErrorMessage: fmt.Sprintf("%s: %v", step, err),
	}
}

func countRowToCounts(r db.CountTickerReferencesRow) *pricev1.TickerDeletionCounts {
	return &pricev1.TickerDeletionCounts{
		StockMetadata:   r.StockMetadata,
		TickerMetadata:  r.TickerMetadata,
		DailyPrices:     r.DailyPrices,
		HourlyPrices:    r.HourlyPrices,
		WeeklyPrices:    r.WeeklyPrices,
		ContinuousPrices: r.ContinuousPrices,
		DailyMoveStats:  r.DailyMoveStats,
		FuturesMetadata: r.FuturesMetadata,
		AlertsDirect:    r.AlertsDirect,
		AlertsRatio:     r.AlertsRatio,
		AlertAudits:     r.AlertAudits,
		PortfolioStocks: r.PortfolioStocks,
	}
}

package main

import (
	"context"
	"fmt"
	"log"
	"time"

	"github.com/jackc/pgx/v5/pgtype"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"stockalert/alert"
	"stockalert/calendar"
	db "stockalert/database/generated"
	"stockalert/discord"
	alertv1 "stockalert/gen/go/alert/v1"
)

func (s *Server) EvaluateExchange(ctx context.Context, req *alertv1.EvaluateExchangeRequest) (*alertv1.EvaluateExchangeResponse, error) {
	exchange := req.GetExchange()
	if exchange == "" {
		return nil, status.Error(codes.InvalidArgument, "exchange is required")
	}
	timeframe := req.GetTimeframe()
	if timeframe == "" {
		timeframe = "daily"
	}

	start := time.Now()
	log.Printf("[evaluate] %s/%s starting", exchange, timeframe)

	// Hourly: skip if exchange is closed
	if timeframe == "hourly" && !calendar.IsExchangeOpen(exchange, time.Now()) {
		log.Printf("[evaluate] %s/%s skipped: exchange closed", exchange, timeframe)
		return &alertv1.EvaluateExchangeResponse{
			Success: false,
			Message: fmt.Sprintf("%s is currently closed", exchange),
		}, nil
	}

	// 1. Update prices
	if s.updater == nil {
		return &alertv1.EvaluateExchangeResponse{
			Success: false,
			Message: "price updater unavailable: FMP_API_KEY may not be set",
		}, nil
	}
	// Use a background context with a generous timeout so that a gRPC client
	// disconnect or deadline does not abort a mid-flight price update.
	priceCtx, priceCancel := context.WithTimeout(context.Background(), 20*time.Minute)
	defer priceCancel()
	pStats, err := s.updater.UpdateForExchange(priceCtx, exchange, timeframe)
	if err != nil {
		log.Printf("[evaluate] %s/%s price update error: %v", exchange, timeframe, err)
		return &alertv1.EvaluateExchangeResponse{
			Success: false,
			Message: fmt.Sprintf("price update failed: %v", err),
		}, nil
	}
	log.Printf("[evaluate] %s/%s prices updated: %d/%d (failed: %d)",
		exchange, timeframe, pStats.Updated, pStats.Total, pStats.Failed)

	// 2. Load alerts for this exchange + timeframe
	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "acquire connection: %v", err)
	}
	defer conn.Release()

	q := db.New(conn)
	alerts, err := q.ListAlertsByExchangeAndTimeframe(ctx, db.ListAlertsByExchangeAndTimeframeParams{
		Exchanges: []string{exchange},
		Timeframe: pgtype.Text{String: timeframe, Valid: true},
	})
	if err != nil {
		return nil, status.Errorf(codes.Internal, "list alerts: %v", err)
	}
	log.Printf("[evaluate] %s/%s evaluating %d alerts", exchange, timeframe, len(alerts))

	if len(alerts) == 0 {
		dur := time.Since(start).Seconds()
		return &alertv1.EvaluateExchangeResponse{
			Success:         true,
			Message:         fmt.Sprintf("No alerts configured for %s/%s", exchange, timeframe),
			AlertsTotal:     0,
			AlertsTriggered: 0,
			PricesUpdated:   int32(pStats.Updated),
			DurationSeconds: dur,
		}, nil
	}

	// 3. Pre-warm alert checker cache
	since := sinceDate(timeframe)
	if err := s.checker.PreWarmCache(ctx, alerts, timeframe, since); err != nil {
		log.Printf("[evaluate] %s/%s prewarm cache: %v", exchange, timeframe, err)
	}

	// 4. Evaluate alerts; accumulate Discord embeds for triggered ones
	var triggered int32
	onTriggered := func(a db.Alert, result alert.CheckResult) {
		triggered++
		ticker := primaryTicker(&a)
		webhookURL := s.router.ResolveWebhookURL(
			ticker,
			timeframe,
			exchange,
			a.IsRatio.Bool && a.IsRatio.Valid,
		)
		embed := discord.FormatAlertEmbed(discord.AlertInfo{
			Ticker:     ticker,
			StockName:  pgText(a.StockName),
			Action:     pgText(a.Action),
			Timeframe:  timeframe,
			Exchange:   pgText(a.Exchange),
			Economy:    s.router.GetEconomy(ticker),
			ISIN:       s.router.GetISIN(ticker),
			Conditions: []string{"Conditions met"},
		})
		s.accum.Add(webhookURL, embed)
	}

	stats, err := s.checker.CheckAlerts(ctx, alerts, timeframe, onTriggered)
	if err != nil {
		log.Printf("[evaluate] %s/%s check alerts error: %v", exchange, timeframe, err)
		return &alertv1.EvaluateExchangeResponse{
			Success: false,
			Message: fmt.Sprintf("alert evaluation failed: %v", err),
		}, nil
	}

	// 5. Flush Discord notifications
	s.accum.FlushAll()

	dur := time.Since(start).Seconds()
	log.Printf("[evaluate] %s/%s complete in %.1fs: %d/%d triggered",
		exchange, timeframe, dur, stats.Triggered, stats.Total)

	return &alertv1.EvaluateExchangeResponse{
		Success:         true,
		Message:         fmt.Sprintf("Evaluated %d alerts for %s/%s: %d triggered", stats.Total, exchange, timeframe, stats.Triggered),
		AlertsTotal:     int32(stats.Total),
		AlertsTriggered: int32(stats.Triggered),
		PricesUpdated:   int32(pStats.Updated),
		DurationSeconds: dur,
	}, nil
}

func sinceDate(timeframe string) time.Time {
	now := time.Now().UTC()
	switch timeframe {
	case "hourly":
		return now.AddDate(0, 0, -7)
	default:
		return now.AddDate(0, 0, -365)
	}
}

func primaryTicker(a *db.Alert) string {
	if a.IsRatio.Bool && a.IsRatio.Valid {
		if a.Ticker1.Valid {
			return a.Ticker1.String
		}
		return ""
	}
	if a.Ticker.Valid {
		return a.Ticker.String
	}
	return ""
}

func pgText(t pgtype.Text) string {
	if t.Valid {
		return t.String
	}
	return ""
}

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

	// 1. Load alerts for this exchange + timeframe.
	// Exchange: alerts may be stored with calendar symbol (e.g. "ATHENS") or display name (e.g. "Athens Stock Exchange").
	// Timeframe: alerts may be stored as "1d"/"1wk"/"1h" (Streamlit) or "daily"/"weekly"/"hourly".
	exchangesToQuery := []string{exchange}
	if sched, ok := calendar.ExchangeSchedules[exchange]; ok && sched.Name != "" && sched.Name != exchange {
		exchangesToQuery = append(exchangesToQuery, sched.Name)
	}
	timeframesToQuery := alert.TimeframeQueryVariants(timeframe)

	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "acquire connection: %v", err)
	}
	defer conn.Release()

	q := db.New(conn)
	alerts, err := q.ListAlertsByExchangeAndTimeframes(ctx, db.ListAlertsByExchangeAndTimeframesParams{
		Exchanges:  exchangesToQuery,
		Timeframes: timeframesToQuery,
	})
	if err != nil {
		return nil, status.Errorf(codes.Internal, "list alerts: %v", err)
	}
	log.Printf("[evaluate] %s/%s evaluating %d alerts", exchange, timeframe, len(alerts))

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
		PricesUpdated:   int32(0),
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

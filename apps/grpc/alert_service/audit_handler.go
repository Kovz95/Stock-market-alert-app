package main

import (
	"context"
	"time"

	"github.com/jackc/pgx/v5"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	alertv1 "stockalert/gen/go/alert/v1"
	db "stockalert/database/generated"
)

// GetDashboardStats returns lightweight aggregates for dashboard KPI cards (no large payloads).
func (s *Server) GetDashboardStats(ctx context.Context, _ *alertv1.GetDashboardStatsRequest) (*alertv1.GetDashboardStatsResponse, error) {
	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	var activeAlerts, triggeredToday, triggersLast7d int64
	err = conn.QueryRow(ctx, `
		SELECT
			(SELECT COUNT(*) FROM alerts),
			(SELECT COUNT(*) FROM alert_audits WHERE alert_triggered = true AND timestamp >= (date_trunc('day', NOW() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC')),
			(SELECT COUNT(*) FROM alert_audits WHERE alert_triggered = true AND timestamp >= (NOW() AT TIME ZONE 'UTC') - INTERVAL '7 days')
	`).Scan(&activeAlerts, &triggeredToday, &triggersLast7d)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "dashboard stats query: %v", err)
	}

	var watchedSymbols int64
	err = conn.QueryRow(ctx, `
		SELECT COUNT(*) FROM (
			SELECT DISTINCT COALESCE(CASE WHEN is_ratio AND NULLIF(TRIM(ratio), '') IS NOT NULL THEN TRIM(ratio) ELSE NULL END, NULLIF(TRIM(ticker), '')) AS sym
			FROM alerts
			WHERE (is_ratio AND NULLIF(TRIM(ratio), '') IS NOT NULL) OR (NOT is_ratio AND NULLIF(TRIM(ticker), '') IS NOT NULL)
		) t WHERE sym IS NOT NULL AND sym != ''
	`).Scan(&watchedSymbols)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "dashboard watched symbols: %v", err)
	}

	// Active alerts by timeframe (alerts.timeframe may be '1d'/'daily', '1wk'/'weekly', '1h'/'hourly')
	var activeHourly, activeDaily, activeWeekly int64
	err = conn.QueryRow(ctx, `
		SELECT
			COUNT(*) FILTER (WHERE LOWER(TRIM(COALESCE(timeframe, ''))) IN ('hourly', '1h', '1hr')) AS hourly,
			COUNT(*) FILTER (WHERE LOWER(TRIM(COALESCE(timeframe, ''))) IN ('daily', '1d')) AS daily,
			COUNT(*) FILTER (WHERE LOWER(TRIM(COALESCE(timeframe, ''))) IN ('weekly', '1wk', '1w')) AS weekly
		FROM alerts
	`).Scan(&activeHourly, &activeDaily, &activeWeekly)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "dashboard active by timeframe: %v", err)
	}

	// Triggered today by timeframe (alert_audits.evaluation_type is normalized: daily/weekly/hourly)
	var todayHourly, todayDaily, todayWeekly int64
	err = conn.QueryRow(ctx, `
		SELECT
			COUNT(*) FILTER (WHERE LOWER(TRIM(COALESCE(evaluation_type, ''))) = 'hourly') AS hourly,
			COUNT(*) FILTER (WHERE LOWER(TRIM(COALESCE(evaluation_type, ''))) = 'daily') AS daily,
			COUNT(*) FILTER (WHERE LOWER(TRIM(COALESCE(evaluation_type, ''))) = 'weekly') AS weekly
		FROM alert_audits
		WHERE alert_triggered = true AND timestamp >= (date_trunc('day', NOW() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC')
	`).Scan(&todayHourly, &todayDaily, &todayWeekly)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "dashboard triggered today by timeframe: %v", err)
	}

	// Triggers last 7d by timeframe
	var last7dHourly, last7dDaily, last7dWeekly int64
	err = conn.QueryRow(ctx, `
		SELECT
			COUNT(*) FILTER (WHERE LOWER(TRIM(COALESCE(evaluation_type, ''))) = 'hourly') AS hourly,
			COUNT(*) FILTER (WHERE LOWER(TRIM(COALESCE(evaluation_type, ''))) = 'daily') AS daily,
			COUNT(*) FILTER (WHERE LOWER(TRIM(COALESCE(evaluation_type, ''))) = 'weekly') AS weekly
		FROM alert_audits
		WHERE alert_triggered = true AND timestamp >= (NOW() AT TIME ZONE 'UTC') - INTERVAL '7 days'
	`).Scan(&last7dHourly, &last7dDaily, &last7dWeekly)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "dashboard triggers 7d by timeframe: %v", err)
	}

	return &alertv1.GetDashboardStatsResponse{
		ActiveAlerts:    int32(activeAlerts),
		TriggeredToday:  int32(triggeredToday),
		WatchedSymbols:  int32(watchedSymbols),
		TriggersLast_7D: int32(triggersLast7d),
		ActiveAlertsByTimeframe: &alertv1.DashboardTimeframeBreakdown{
			Hourly: int32(activeHourly),
			Daily:  int32(activeDaily),
			Weekly: int32(activeWeekly),
		},
		TriggeredTodayByTimeframe: &alertv1.DashboardTimeframeBreakdown{
			Hourly: int32(todayHourly),
			Daily:  int32(todayDaily),
			Weekly: int32(todayWeekly),
		},
		TriggersLast_7DByTimeframe: &alertv1.DashboardTimeframeBreakdown{
			Hourly: int32(last7dHourly),
			Daily:  int32(last7dDaily),
			Weekly: int32(last7dWeekly),
		},
	}, nil
}

// GetTriggerCountByDay returns trigger counts per day from alert_audits (same source as dashboard stats).
func (s *Server) GetTriggerCountByDay(ctx context.Context, req *alertv1.GetTriggerCountByDayRequest) (*alertv1.GetTriggerCountByDayResponse, error) {
	days := req.GetDays()
	if days < 7 {
		days = 7
	}
	if days > 90 {
		days = 90
	}
	cutoff := time.Now().UTC().AddDate(0, 0, -int(days))
	cutoff = time.Date(cutoff.Year(), cutoff.Month(), cutoff.Day(), 0, 0, 0, 0, time.UTC)

	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	query := `
		SELECT (date_trunc('day', timestamp AT TIME ZONE 'UTC'))::date::text AS day, COUNT(*)::bigint
		FROM alert_audits
		WHERE alert_triggered = true AND timestamp >= $1
		GROUP BY (date_trunc('day', timestamp AT TIME ZONE 'UTC'))::date
		ORDER BY day
	`
	rows, err := conn.Query(ctx, query, cutoff)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "trigger count by day: %v", err)
	}
	defer rows.Close()

	var out []*alertv1.TriggerCountRow
	for rows.Next() {
		var day string
		var count int64
		if err := rows.Scan(&day, &count); err != nil {
			return nil, status.Errorf(codes.Internal, "scan trigger count row: %v", err)
		}
		out = append(out, &alertv1.TriggerCountRow{Date: day, Count: count})
	}
	if err := rows.Err(); err != nil {
		return nil, status.Errorf(codes.Internal, "trigger count rows: %v", err)
	}

	// Fill missing days with 0 so the chart has a point for every day
	daySet := make(map[string]int64)
	for _, r := range out {
		daySet[r.Date] = r.Count
	}
	var filled []*alertv1.TriggerCountRow
	for d := 0; d < int(days); d++ {
		dt := cutoff.AddDate(0, 0, d)
		key := dt.Format("2006-01-02")
		c := int64(0)
		if n, ok := daySet[key]; ok {
			c = n
		}
		filled = append(filled, &alertv1.TriggerCountRow{Date: key, Count: c})
	}
	return &alertv1.GetTriggerCountByDayResponse{Rows: filled}, nil
}

func (s *Server) GetAuditSummary(ctx context.Context, req *alertv1.GetAuditSummaryRequest) (*alertv1.GetAuditSummaryResponse, error) {
	days := req.GetDays()
	if days <= 0 {
		days = 7
	}
	if days > 90 {
		days = 90
	}
	cutoff := time.Now().UTC().AddDate(0, 0, -int(days))
	limit := req.GetLimit()

	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	baseQuery := `
		SELECT
			alert_id,
			ticker,
			COALESCE(stock_name, ''),
			COALESCE(exchange, ''),
			COALESCE(timeframe, ''),
			COALESCE(action, ''),
			evaluation_type,
			COUNT(*)::bigint AS total_checks,
			SUM(CASE WHEN price_data_pulled THEN 1 ELSE 0 END)::bigint AS successful_price_pulls,
			SUM(CASE WHEN conditions_evaluated THEN 1 ELSE 0 END)::bigint AS successful_evaluations,
			SUM(CASE WHEN alert_triggered THEN 1 ELSE 0 END)::bigint AS total_triggers,
			AVG(execution_time_ms)::float8 AS avg_execution_time_ms,
			MAX(timestamp) AS last_check,
			MIN(timestamp) AS first_check
		FROM alert_audits
		WHERE timestamp >= $1
		GROUP BY alert_id, ticker, stock_name, exchange, timeframe, action, evaluation_type
	`
	var rows pgx.Rows
	if limit > 0 {
		query := `SELECT * FROM (` + baseQuery + `) t ORDER BY total_triggers DESC LIMIT $2`
		rows, err = conn.Query(ctx, query, cutoff, limit)
	} else {
		query := baseQuery + ` ORDER BY last_check DESC`
		rows, err = conn.Query(ctx, query, cutoff)
	}
	if err != nil {
		return nil, status.Errorf(codes.Internal, "audit summary query: %v", err)
	}
	defer rows.Close()

	var out []*alertv1.AuditSummaryRow
	for rows.Next() {
		var (
			alertID, ticker, stockName, exchange, timeframe, action, evalType string
			totalChecks, pricePulls, evaluations, triggers                    int64
			avgMs                                                             *float64
			lastCheck, firstCheck                                             time.Time
		)
		err := rows.Scan(
			&alertID, &ticker, &stockName, &exchange, &timeframe, &action, &evalType,
			&totalChecks, &pricePulls, &evaluations, &triggers,
			&avgMs, &lastCheck, &firstCheck,
		)
		if err != nil {
			return nil, status.Errorf(codes.Internal, "scan audit summary row: %v", err)
		}
		avgVal := 0.0
		if avgMs != nil {
			avgVal = *avgMs
		}
		out = append(out, &alertv1.AuditSummaryRow{
			AlertId:                 alertID,
			Ticker:                  ticker,
			StockName:               stockName,
			Exchange:                exchange,
			Timeframe:               timeframe,
			Action:                  action,
			EvaluationType:          evalType,
			TotalChecks:             totalChecks,
			SuccessfulPricePulls:    pricePulls,
			SuccessfulEvaluations:   evaluations,
			TotalTriggers:           triggers,
			AvgExecutionTimeMs:     avgVal,
			LastCheck:               timestamppb.New(lastCheck),
			FirstCheck:              timestamppb.New(firstCheck),
		})
	}
	if err := rows.Err(); err != nil {
		return nil, status.Errorf(codes.Internal, "audit summary rows: %v", err)
	}
	return &alertv1.GetAuditSummaryResponse{Rows: out}, nil
}

// GetTopTriggeredAlerts returns the top N alerts by count of rows in alert_audits where alert_triggered = true.
// Only alerts that still exist in the alerts table are returned (INNER JOIN).
func (s *Server) GetTopTriggeredAlerts(ctx context.Context, req *alertv1.GetTopTriggeredAlertsRequest) (*alertv1.GetTopTriggeredAlertsResponse, error) {
	days := req.GetDays()
	if days <= 0 {
		days = 30
	}
	if days > 90 {
		days = 90
	}
	limit := req.GetLimit()
	if limit <= 0 {
		limit = 10
	}
	if limit > 50 {
		limit = 50
	}
	cutoff := time.Now().UTC().AddDate(0, 0, -int(days))

	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	query := `
		WITH top AS (
			SELECT alert_id, COUNT(*) AS trigger_count
			FROM alert_audits
			WHERE alert_triggered = true
			AND timestamp >= $1
			GROUP BY alert_id
			ORDER BY trigger_count DESC
			LIMIT $2
		)
		SELECT
			top.trigger_count,
			a.alert_id, a.name, a.stock_name, a.ticker, a.ticker1, a.ticker2,
			a.conditions, a.combination_logic, a.last_triggered, a.action,
			a.timeframe, a.exchange, a.country, a.ratio, a.is_ratio,
			a.adjustment_method, a.dtp_params, a.multi_timeframe_params,
			a.mixed_timeframe_params, a.raw_payload, a.created_at, a.updated_at
		FROM alerts a
		INNER JOIN top ON top.alert_id = a.alert_id::text
		ORDER BY top.trigger_count DESC
	`
	rows, err := conn.Query(ctx, query, cutoff, limit)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "top triggered alerts query: %v", err)
	}
	defer rows.Close()

	var out []*alertv1.TopTriggeredAlert
	for rows.Next() {
		var triggerCount int64
		var a db.Alert
		err := rows.Scan(
			&triggerCount,
			&a.AlertID,
			&a.Name,
			&a.StockName,
			&a.Ticker,
			&a.Ticker1,
			&a.Ticker2,
			&a.Conditions,
			&a.CombinationLogic,
			&a.LastTriggered,
			&a.Action,
			&a.Timeframe,
			&a.Exchange,
			&a.Country,
			&a.Ratio,
			&a.IsRatio,
			&a.AdjustmentMethod,
			&a.DtpParams,
			&a.MultiTimeframeParams,
			&a.MixedTimeframeParams,
			&a.RawPayload,
			&a.CreatedAt,
			&a.UpdatedAt,
		)
		if err != nil {
			return nil, status.Errorf(codes.Internal, "scan top triggered alert: %v", err)
		}
		out = append(out, &alertv1.TopTriggeredAlert{
			Alert:        dbAlertToProto(a),
			TriggerCount: triggerCount,
		})
	}
	if err := rows.Err(); err != nil {
		return nil, status.Errorf(codes.Internal, "top triggered alerts rows: %v", err)
	}

	var totalCount int64
	if err := conn.QueryRow(ctx, "SELECT COUNT(*) FROM alerts").Scan(&totalCount); err != nil {
		return nil, status.Errorf(codes.Internal, "count alerts: %v", err)
	}
	return &alertv1.GetTopTriggeredAlertsResponse{
		Alerts:     out,
		TotalCount: int32(totalCount),
	}, nil
}

func (s *Server) GetPerformanceMetrics(ctx context.Context, req *alertv1.GetPerformanceMetricsRequest) (*alertv1.GetPerformanceMetricsResponse, error) {
	days := req.GetDays()
	if days <= 0 {
		days = 7
	}
	if days > 90 {
		days = 90
	}
	cutoff := time.Now().UTC().AddDate(0, 0, -int(days))

	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	var totalChecks int64
	err = conn.QueryRow(ctx, "SELECT COUNT(*) FROM alert_audits WHERE timestamp >= $1", cutoff).Scan(&totalChecks)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "total checks: %v", err)
	}

	var successfulPulls int64
	err = conn.QueryRow(ctx,
		"SELECT COUNT(*) FROM alert_audits WHERE timestamp >= $1 AND price_data_pulled = true",
		cutoff).Scan(&successfulPulls)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "successful pulls: %v", err)
	}

	var cacheHits, totalPulls int64
	err = conn.QueryRow(ctx, `
		SELECT
			SUM(CASE WHEN cache_hit THEN 1 ELSE 0 END),
			COUNT(*)
		FROM alert_audits
		WHERE timestamp >= $1 AND price_data_pulled = true
	`, cutoff).Scan(&cacheHits, &totalPulls)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "cache stats: %v", err)
	}
	cacheHitRate := 0.0
	if totalPulls > 0 {
		cacheHitRate = float64(cacheHits) / float64(totalPulls) * 100
	}

	var avgExecutionMs *float64
	err = conn.QueryRow(ctx, `
		SELECT AVG(execution_time_ms)::float8 FROM alert_audits
		WHERE timestamp >= $1 AND execution_time_ms IS NOT NULL
	`, cutoff).Scan(&avgExecutionMs)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "avg execution: %v", err)
	}
	avgMs := 0.0
	if avgExecutionMs != nil {
		avgMs = *avgExecutionMs
	}

	var totalErrors int64
	err = conn.QueryRow(ctx, `
		SELECT COUNT(*) FROM alert_audits
		WHERE timestamp >= $1
		AND (
			price_data_pulled = false
			OR error_message ILIKE '%No data available%'
			OR error_message ILIKE '%FMP API%'
		)
	`, cutoff).Scan(&totalErrors)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "total errors: %v", err)
	}

	successRate := 0.0
	if totalChecks > 0 {
		successRate = float64(successfulPulls) / float64(totalChecks) * 100
	}
	errorRate := 0.0
	if totalChecks > 0 {
		errorRate = float64(totalErrors) / float64(totalChecks) * 100
	}

	return &alertv1.GetPerformanceMetricsResponse{
		TotalChecks:             totalChecks,
		SuccessfulPricePulls:    successfulPulls,
		SuccessRate:             successRate,
		CacheHitRate:            cacheHitRate,
		AvgExecutionTimeMs:      avgMs,
		TotalErrors:             totalErrors,
		ErrorRate:               errorRate,
		AnalysisPeriodDays:      days,
	}, nil
}

func (s *Server) GetAlertHistory(ctx context.Context, req *alertv1.GetAlertHistoryRequest) (*alertv1.GetAlertHistoryResponse, error) {
	alertID := req.GetAlertId()
	if alertID == "" {
		return nil, status.Error(codes.InvalidArgument, "alert_id is required")
	}
	limit := req.GetLimit()
	if limit <= 0 {
		limit = 100
	}
	if limit > 1000 {
		limit = 1000
	}

	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	query := `
		SELECT id, timestamp, alert_id, ticker, COALESCE(stock_name,''), COALESCE(exchange,''), COALESCE(timeframe,''),
			COALESCE(action,''), evaluation_type, COALESCE(price_data_pulled, false), COALESCE(price_data_source,''),
			COALESCE(conditions_evaluated, false), COALESCE(alert_triggered, false), COALESCE(trigger_reason,''),
			execution_time_ms, COALESCE(cache_hit, false), COALESCE(error_message,'')
		FROM alert_audits
		WHERE alert_id = $1
		ORDER BY timestamp DESC
		LIMIT $2
	`
	rows, err := conn.Query(ctx, query, alertID, limit)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "alert history query: %v", err)
	}
	defer rows.Close()

	var out []*alertv1.AuditHistoryRow
	for rows.Next() {
		var (
			id                                                                  int64
			ts                                                                  time.Time
			aid, ticker, stockName, exchange, timeframe, action, evalType       string
			pricePulled, conditionsEval, triggered                             bool
			priceSource, triggerReason                                          string
			execMs                                                               *int32
			cacheHit                                                            bool
			errorMsg                                                            string
		)
		err := rows.Scan(
			&id, &ts, &aid, &ticker, &stockName, &exchange, &timeframe, &action, &evalType,
			&pricePulled, &priceSource, &conditionsEval, &triggered, &triggerReason,
			&execMs, &cacheHit, &errorMsg,
		)
		if err != nil {
			return nil, status.Errorf(codes.Internal, "scan history row: %v", err)
		}
		execVal := int32(0)
		if execMs != nil {
			execVal = *execMs
		}
		out = append(out, &alertv1.AuditHistoryRow{
			Id:                   id,
			Timestamp:            timestamppb.New(ts),
			AlertId:              aid,
			Ticker:               ticker,
			StockName:            stockName,
			Exchange:             exchange,
			Timeframe:            timeframe,
			Action:               action,
			EvaluationType:      evalType,
			PriceDataPulled:      pricePulled,
			PriceDataSource:      priceSource,
			ConditionsEvaluated:  conditionsEval,
			AlertTriggered:       triggered,
			TriggerReason:        triggerReason,
			ExecutionTimeMs:     execVal,
			CacheHit:             cacheHit,
			ErrorMessage:        errorMsg,
		})
	}
	if err := rows.Err(); err != nil {
		return nil, status.Errorf(codes.Internal, "alert history rows: %v", err)
	}
	return &alertv1.GetAlertHistoryResponse{Rows: out}, nil
}

func (s *Server) GetFailedPriceData(ctx context.Context, req *alertv1.GetFailedPriceDataRequest) (*alertv1.GetFailedPriceDataResponse, error) {
	days := req.GetDays()
	if days <= 0 {
		days = 7
	}
	if days > 90 {
		days = 90
	}

	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	cutoff := time.Now().UTC().AddDate(0, 0, -int(days))
	// Failed alerts: price_data_pulled = false OR error like No data / FMP API
	failedQuery := `
		SELECT
			f.alert_id,
			f.ticker,
			COALESCE(f.stock_name, ''),
			COALESCE(f.exchange, ''),
			COALESCE(f.timeframe, ''),
			COUNT(*)::bigint AS failure_count,
			MAX(f.timestamp) AS last_failure,
			MIN(f.timestamp) AS first_failure,
			AVG(f.execution_time_ms)::float8 AS avg_execution_time
		FROM alert_audits f
		WHERE (f.price_data_pulled = false OR f.error_message ILIKE '%No data available%' OR f.error_message ILIKE '%FMP API%')
		AND f.timestamp >= $1
		GROUP BY f.alert_id, f.ticker, f.stock_name, f.exchange, f.timeframe
		ORDER BY failure_count DESC
	`
	rows, err := conn.Query(ctx, failedQuery, cutoff)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed price data query: %v", err)
	}
	defer rows.Close()

	type failedRow struct {
		alertID      string
		ticker       string
		stockName    string
		exchange     string
		timeframe    string
		failureCount int64
		lastFail     time.Time
		firstFail    time.Time
		avgExec      *float64
	}
	var failedRows []failedRow
	for rows.Next() {
		var r failedRow
		err := rows.Scan(&r.alertID, &r.ticker, &r.stockName, &r.exchange, &r.timeframe,
			&r.failureCount, &r.lastFail, &r.firstFail, &r.avgExec)
		if err != nil {
			return nil, status.Errorf(codes.Internal, "scan failed row: %v", err)
		}
		failedRows = append(failedRows, r)
	}
	if err := rows.Err(); err != nil {
		return nil, status.Errorf(codes.Internal, "failed rows: %v", err)
	}

	// Asset type from stock_metadata (symbol = ticker or symbol = ticker || '-US')
	assetTypeQuery := `
		SELECT COALESCE(sm.asset_type, 'Unknown')
		FROM stock_metadata sm
		WHERE sm.symbol = $1 OR sm.symbol = $2
		LIMIT 1
	`
	getAssetType := func(ticker string) string {
		var at string
		err := conn.QueryRow(ctx, assetTypeQuery, ticker, ticker+"-US").Scan(&at)
		if err == pgx.ErrNoRows || err != nil {
			return "Unknown"
		}
		if at == "" {
			return "Unknown"
		}
		return at
	}

	var outRows []*alertv1.FailedAlertRow
	assetCount := make(map[string]struct{ failedAlerts, failureCount int64 })
	exchangeCount := make(map[string]struct{ failedAlerts, failureCount int64 })

	for _, r := range failedRows {
		assetType := getAssetType(r.ticker)
		avgExec := 0.0
		if r.avgExec != nil {
			avgExec = *r.avgExec
		}
		outRows = append(outRows, &alertv1.FailedAlertRow{
			AlertId:        r.alertID,
			Ticker:         r.ticker,
			StockName:      r.stockName,
			Exchange:       r.exchange,
			Timeframe:      r.timeframe,
			AssetType:      assetType,
			FailureCount:   r.failureCount,
			LastFailure:    timestamppb.New(r.lastFail),
			FirstFailure:   timestamppb.New(r.firstFail),
			AvgExecutionTime: avgExec,
		})
		a := assetCount[assetType]
		a.failedAlerts++
		a.failureCount += r.failureCount
		assetCount[assetType] = a
		ex := r.exchange
		if ex == "" {
			ex = "(empty)"
		}
		e := exchangeCount[ex]
		e.failedAlerts++
		e.failureCount += r.failureCount
		exchangeCount[ex] = e
	}

	var totalFailures int64
	for _, r := range failedRows {
		totalFailures += r.failureCount
	}
	totalFailedAlerts := int64(len(failedRows))

	var totalChecks int64
	err = conn.QueryRow(ctx, "SELECT COUNT(*) FROM alert_audits WHERE timestamp >= $1", cutoff).Scan(&totalChecks)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "total checks for failure rate: %v", err)
	}
	failureRate := 0.0
	if totalChecks > 0 {
		failureRate = float64(totalFailures) / float64(totalChecks) * 100
	}

	var assetBreakdown []*alertv1.AssetTypeBreakdownRow
	for at, c := range assetCount {
		assetBreakdown = append(assetBreakdown, &alertv1.AssetTypeBreakdownRow{
			AssetType:    at,
			FailedAlerts: c.failedAlerts,
			FailureCount: c.failureCount,
		})
	}
	var exchangeBreakdown []*alertv1.ExchangeBreakdownRow
	for ex, c := range exchangeCount {
		exchangeBreakdown = append(exchangeBreakdown, &alertv1.ExchangeBreakdownRow{
			Exchange:     ex,
			FailedAlerts: c.failedAlerts,
			FailureCount: c.failureCount,
		})
	}

	return &alertv1.GetFailedPriceDataResponse{
		Rows:                outRows,
		TotalFailedAlerts:   totalFailedAlerts,
		TotalFailures:       totalFailures,
		FailureRate:         failureRate,
		AssetTypeBreakdown:  assetBreakdown,
		ExchangeBreakdown:   exchangeBreakdown,
	}, nil
}

func (s *Server) ClearAuditData(ctx context.Context, req *alertv1.ClearAuditDataRequest) (*alertv1.ClearAuditDataResponse, error) {
	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	tag, err := conn.Exec(ctx, "DELETE FROM alert_audits")
	if err != nil {
		return nil, status.Errorf(codes.Internal, "clear audit data: %v", err)
	}
	return &alertv1.ClearAuditDataResponse{DeletedCount: tag.RowsAffected()}, nil
}

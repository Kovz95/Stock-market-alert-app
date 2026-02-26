package main

import (
	"context"
	"strings"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	alertv1 "stockalert/gen/go/alert/v1"
)

func (s *Server) GetTriggerHistoryByTicker(ctx context.Context, req *alertv1.GetTriggerHistoryByTickerRequest) (*alertv1.GetTriggerHistoryByTickerResponse, error) {
	ticker := strings.TrimSpace(req.GetTicker())
	if ticker == "" {
		return nil, status.Error(codes.InvalidArgument, "ticker is required")
	}
	limit := req.GetLimit()
	if limit <= 0 {
		limit = 50
	}
	if limit > 500 {
		limit = 500
	}
	includeAll := req.GetIncludeAllEvaluations()
	daysBack := req.GetDaysBack()

	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	// Build query: join alerts to get alert name; filter by ticker, optional trigger, optional days_back
	query := `
		SELECT
			aa.id,
			aa.timestamp,
			aa.alert_id,
			aa.ticker,
			COALESCE(aa.stock_name,''),
			COALESCE(aa.exchange,''),
			COALESCE(aa.timeframe,''),
			COALESCE(aa.action,''),
			aa.evaluation_type,
			COALESCE(aa.price_data_pulled, false),
			COALESCE(aa.price_data_source,''),
			COALESCE(aa.conditions_evaluated, false),
			COALESCE(aa.alert_triggered, false),
			COALESCE(aa.trigger_reason,''),
			aa.execution_time_ms,
			COALESCE(aa.cache_hit, false),
			COALESCE(aa.error_message,''),
			COALESCE(a.name, '')
		FROM alert_audits aa
		LEFT JOIN alerts a ON a.alert_id = aa.alert_id
		WHERE aa.ticker = $1
	`
	args := []interface{}{ticker}
	if !includeAll {
		query += ` AND aa.alert_triggered = true`
	}
	if daysBack > 0 {
		cutoff := time.Now().UTC().AddDate(0, 0, -int(daysBack))
		args = append(args, cutoff)
		query += ` AND aa.timestamp >= $2 ORDER BY aa.timestamp DESC LIMIT $3`
		args = append(args, limit)
	} else {
		query += ` ORDER BY aa.timestamp DESC LIMIT $2`
		args = append(args, limit)
	}

	rows, err := conn.Query(ctx, query, args...)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "trigger history by ticker query: %v", err)
	}
	defer rows.Close()

	var out []*alertv1.AuditHistoryRow
	for rows.Next() {
		var (
			id                                                                  int64
			ts                                                                  time.Time
			aid, tickerVal, stockName, exchange, timeframe, action, evalType   string
			pricePulled, conditionsEval, triggered                             bool
			priceSource, triggerReason                                          string
			execMs                                                               *int32
			cacheHit                                                            bool
			errorMsg                                                            string
			alertName                                                           string
		)
		err := rows.Scan(
			&id, &ts, &aid, &tickerVal, &stockName, &exchange, &timeframe, &action, &evalType,
			&pricePulled, &priceSource, &conditionsEval, &triggered, &triggerReason,
			&execMs, &cacheHit, &errorMsg, &alertName,
		)
		if err != nil {
			return nil, status.Errorf(codes.Internal, "scan trigger history row: %v", err)
		}
		execVal := int32(0)
		if execMs != nil {
			execVal = *execMs
		}
		out = append(out, &alertv1.AuditHistoryRow{
			Id:                   id,
			Timestamp:            timestamppb.New(ts),
			AlertId:              aid,
			Ticker:               tickerVal,
			StockName:            stockName,
			Exchange:             exchange,
			Timeframe:            timeframe,
			Action:               action,
			EvaluationType:       evalType,
			PriceDataPulled:     pricePulled,
			PriceDataSource:     priceSource,
			ConditionsEvaluated:  conditionsEval,
			AlertTriggered:      triggered,
			TriggerReason:       triggerReason,
			ExecutionTimeMs:    execVal,
			CacheHit:            cacheHit,
			ErrorMessage:        errorMsg,
			AlertName:           alertName,
		})
	}
	if err := rows.Err(); err != nil {
		return nil, status.Errorf(codes.Internal, "trigger history rows: %v", err)
	}
	return &alertv1.GetTriggerHistoryByTickerResponse{Rows: out}, nil
}

func (s *Server) SearchStocks(ctx context.Context, req *alertv1.SearchStocksRequest) (*alertv1.SearchStocksResponse, error) {
	query := strings.TrimSpace(req.GetQuery())
	if query == "" {
		return &alertv1.SearchStocksResponse{Results: nil}, nil
	}
	limit := req.GetLimit()
	if limit <= 0 {
		limit = 20
	}
	if limit > 50 {
		limit = 50
	}

	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	// Search by symbol or name (case-insensitive); order exact ticker match first, then symbol prefix, then name
	pattern := "%" + query + "%"
	queryUpper := strings.ToUpper(query)
	prefixPattern := queryUpper + "%"
	sql := `
		SELECT symbol, COALESCE(name,''), COALESCE(exchange,''), COALESCE(asset_type,'Stock'), COALESCE(rbics_economy,'')
		FROM stock_metadata
		WHERE symbol ILIKE $1 OR name ILIKE $1
		ORDER BY
			CASE WHEN UPPER(symbol) = $2 THEN 0 WHEN UPPER(symbol) LIKE $3 THEN 1 ELSE 2 END,
			LENGTH(symbol),
			symbol
		LIMIT $4
	`
	rows, err := conn.Query(ctx, sql, pattern, queryUpper, prefixPattern, limit)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "search stocks query: %v", err)
	}
	defer rows.Close()

	var results []*alertv1.StockSearchResult
	for rows.Next() {
		var symbol, name, exchange, assetType, rbicsEconomy string
		if err := rows.Scan(&symbol, &name, &exchange, &assetType, &rbicsEconomy); err != nil {
			return nil, status.Errorf(codes.Internal, "scan stock search row: %v", err)
		}
		results = append(results, &alertv1.StockSearchResult{
			Ticker:       symbol,
			Name:         name,
			Exchange:     exchange,
			Type:         assetType,
			RbicsEconomy: rbicsEconomy,
		})
	}
	if err := rows.Err(); err != nil {
		return nil, status.Errorf(codes.Internal, "search stocks rows: %v", err)
	}
	return &alertv1.SearchStocksResponse{Results: results}, nil
}

func (s *Server) ListPortfolios(ctx context.Context, req *alertv1.ListPortfoliosRequest) (*alertv1.ListPortfoliosResponse, error) {
	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	rows, err := conn.Query(ctx, `
		SELECT p.id, p.name
		FROM portfolios p
		ORDER BY p.name
	`)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "list portfolios query: %v", err)
	}
	defer rows.Close()

	var portfolios []*alertv1.Portfolio
	for rows.Next() {
		var id, name string
		if err := rows.Scan(&id, &name); err != nil {
			return nil, status.Errorf(codes.Internal, "scan portfolio row: %v", err)
		}
		portfolios = append(portfolios, &alertv1.Portfolio{
			PortfolioId: id,
			Name:        name,
			Tickers:     nil, // filled below
		})
	}
	if err := rows.Err(); err != nil {
		return nil, status.Errorf(codes.Internal, "list portfolios rows: %v", err)
	}

	// Load tickers for each portfolio
	for _, p := range portfolios {
		tickerRows, err := conn.Query(ctx, `
			SELECT ticker FROM portfolio_stocks WHERE portfolio_id = $1 ORDER BY ticker
		`, p.PortfolioId)
		if err != nil {
			return nil, status.Errorf(codes.Internal, "portfolio tickers query: %v", err)
		}
		var tickers []string
		for tickerRows.Next() {
			var t string
			if err := tickerRows.Scan(&t); err != nil {
				tickerRows.Close()
				return nil, status.Errorf(codes.Internal, "scan portfolio ticker: %v", err)
			}
			tickers = append(tickers, t)
		}
		tickerRows.Close()
		if err := tickerRows.Err(); err != nil {
			return nil, status.Errorf(codes.Internal, "portfolio tickers rows: %v", err)
		}
		p.Tickers = tickers
	}

	return &alertv1.ListPortfoliosResponse{Portfolios: portfolios}, nil
}

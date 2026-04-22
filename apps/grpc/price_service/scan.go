package main

import (
	"context"
	"sync"
	"time"

	"github.com/jackc/pgx/v5/pgtype"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	db "stockalert/database/generated"
	"stockalert/expr"
	pricev1 "stockalert/gen/go/price/v1"
	"stockalert/indicator"
)

const (
	scanLookbackDaysDaily  = 400  // ~280 trading days; supports SMA(200) + buffer
	// keep aligned with sinceDateForTimeframe("weekly") in apps/scheduler/internal/handler/common.go
	scanLookbackDaysWeekly = 2000 // ~285 weekly bars; supports SMA(200) weekly + buffer
	scanLookbackDaysHourly = 90   // ~60 trading days × 6.5h ≈ 390 hourly bars
	scanMinBars            = 50
	scanMaxConcurrency  = 20
	scanMaxTickersCap   = 20000
	scanMaxLookback     = 250
	scanMaxTotalMatches = 50000
)

// metaRow is a convenience view of ListFullStockMetadataRow for filtering and building ScanMatch.
type metaRow struct {
	symbol             string
	name               string
	exchange           string
	country            string
	assetType          string
	rbicsEconomy       string
	rbicsSector        string
	rbicsSubsector     string
	rbicsIndustryGroup string
	rbicsIndustry      string
	rbicsSubindustry   string
}

func (s *Server) RunScan(ctx context.Context, req *pricev1.RunScanRequest) (resp *pricev1.RunScanResponse, err error) {
	defer func() {
		if r := recover(); r != nil {
			resp = nil
			err = status.Errorf(codes.Internal, "RunScan panic: %v", r)
		}
	}()
	if len(req.Conditions) == 0 {
		return &pricev1.RunScanResponse{}, nil
	}

	// Use a dedicated connection for metadata only; workers acquire their own (pgx conns are not concurrent-safe).
	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	metaRows, err := db.New(conn).ListFullStockMetadata(ctx)
	conn.Release()
	if err != nil {
		return nil, status.Errorf(codes.Internal, "list full stock metadata: %v", err)
	}

	metaBySymbol := make(map[string]metaRow)
	for _, r := range metaRows {
		metaBySymbol[r.Symbol] = metaRow{
			symbol:             r.Symbol,
			name:               pgTextString(r.Name),
			exchange:           pgTextString(r.Exchange),
			country:            pgTextString(r.Country),
			assetType:          pgTextString(r.AssetType),
			rbicsEconomy:       pgTextString(r.RbicsEconomy),
			rbicsSector:        pgTextString(r.RbicsSector),
			rbicsSubsector:     pgTextString(r.RbicsSubsector),
			rbicsIndustryGroup: pgTextString(r.RbicsIndustryGroup),
			rbicsIndustry:      pgTextString(r.RbicsIndustry),
			rbicsSubindustry:   pgTextString(r.RbicsSubindustry),
		}
	}

	var tickers []string
	if len(req.Tickers) > 0 {
		tickers = req.Tickers
	} else {
		tickers = resolveTickersFromFilter(metaRows, req.GetSymbolFilter())
	}

	maxTickers := int(req.MaxTickers)
	if maxTickers <= 0 {
		maxTickers = scanMaxTickersCap
	}
	if len(tickers) > maxTickers {
		tickers = tickers[:maxTickers]
	}

	registry := indicator.NewDefaultRegistry()
	eval := expr.NewEvaluator(registry)
	combinationLogic := req.CombinationLogic
	if combinationLogic == "" {
		combinationLogic = "AND"
	}

	lookbackDays := int(req.LookbackDays)
	if lookbackDays < 0 {
		lookbackDays = 0
	}
	if lookbackDays > scanMaxLookback {
		lookbackDays = scanMaxLookback
	}

	var mu sync.Mutex
	var matches []*pricev1.ScanMatch
	truncated := false
	sem := make(chan struct{}, scanMaxConcurrency)
	var wg sync.WaitGroup

	for _, ticker := range tickers {
		ticker := ticker
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer func() { <-sem; wg.Done() }()
			// Check if we've already hit the cap.
			mu.Lock()
			if len(matches) >= scanMaxTotalMatches {
				truncated = true
				mu.Unlock()
				return
			}
			mu.Unlock()

			workerConn, err := s.pool.Acquire(ctx)
			if err != nil {
				return
			}
			defer workerConn.Release()
			q := db.New(workerConn)
			tickerMatches := s.scanOneTicker(ctx, q, eval, ticker, metaBySymbol[ticker], req.Timeframe, req.Conditions, combinationLogic, lookbackDays)
			if len(tickerMatches) > 0 {
				mu.Lock()
				matches = append(matches, tickerMatches...)
				if len(matches) >= scanMaxTotalMatches {
					truncated = true
				}
				mu.Unlock()
			}
		}()
	}

	wg.Wait()

	resp = &pricev1.RunScanResponse{Matches: matches}
	if truncated {
		resp.ErrorMessage = "Results truncated: exceeded 50,000 match limit."
	}
	return resp, nil
}

func resolveTickersFromFilter(rows []db.ListFullStockMetadataRow, filter *pricev1.SymbolFilter) []string {
	if filter == nil {
		out := make([]string, 0, len(rows))
		for _, r := range rows {
			out = append(out, r.Symbol)
		}
		return out
	}
	assetTypes := stringSet(filter.AssetTypes)
	countries := stringSet(filter.Countries)
	exchanges := stringSet(filter.Exchanges)
	rbicsEconomy := stringSet(filter.RbicsEconomy)
	rbicsSector := stringSet(filter.RbicsSector)
	rbicsSubsector := stringSet(filter.RbicsSubsector)
	rbicsIndustryGroup := stringSet(filter.RbicsIndustryGroup)
	rbicsIndustry := stringSet(filter.RbicsIndustry)
	rbicsSubindustry := stringSet(filter.RbicsSubindustry)

	out := make([]string, 0)
	for _, r := range rows {
		if len(assetTypes) > 0 && !assetTypes[pgTextString(r.AssetType)] {
			continue
		}
		if len(countries) > 0 && !countries[pgTextString(r.Country)] {
			continue
		}
		if len(exchanges) > 0 && !exchanges[pgTextString(r.Exchange)] {
			continue
		}
		if len(rbicsEconomy) > 0 && !rbicsEconomy[pgTextString(r.RbicsEconomy)] {
			continue
		}
		if len(rbicsSector) > 0 && !rbicsSector[pgTextString(r.RbicsSector)] {
			continue
		}
		if len(rbicsSubsector) > 0 && !rbicsSubsector[pgTextString(r.RbicsSubsector)] {
			continue
		}
		if len(rbicsIndustryGroup) > 0 && !rbicsIndustryGroup[pgTextString(r.RbicsIndustryGroup)] {
			continue
		}
		if len(rbicsIndustry) > 0 && !rbicsIndustry[pgTextString(r.RbicsIndustry)] {
			continue
		}
		if len(rbicsSubindustry) > 0 && !rbicsSubindustry[pgTextString(r.RbicsSubindustry)] {
			continue
		}
		out = append(out, r.Symbol)
	}
	return out
}

func stringSet(s []string) map[string]bool {
	m := make(map[string]bool, len(s))
	for _, v := range s {
		m[v] = true
	}
	return m
}

func (s *Server) scanOneTicker(
	ctx context.Context,
	q *db.Queries,
	eval *expr.Evaluator,
	ticker string,
	meta metaRow,
	timeframe pricev1.Timeframe,
	conditions []string,
	combinationLogic string,
	lookbackDays int,
) []*pricev1.ScanMatch {
	var ohlcv *indicator.OHLCV
	switch timeframe {
	case pricev1.Timeframe_TIMEFRAME_HOURLY:
		ohlcv = s.loadHourlyOHLCV(ctx, q, ticker)
	case pricev1.Timeframe_TIMEFRAME_WEEKLY:
		ohlcv = s.loadWeeklyOHLCV(ctx, q, ticker)
	default:
		ohlcv = s.loadDailyOHLCV(ctx, q, ticker)
	}
	if ohlcv == nil || ohlcv.Len() < scanMinBars {
		return nil
	}

	totalBars := ohlcv.Len()

	// Determine the range of bars to evaluate.
	// startBar is the earliest bar index (inclusive) to use as "last bar" of a sub-slice.
	// endBar is the latest bar index (inclusive) — always totalBars-1.
	startBar := totalBars - 1
	if lookbackDays > 0 {
		startBar = totalBars - lookbackDays
		if startBar < scanMinBars {
			startBar = scanMinBars
		}
	}

	var matches []*pricev1.ScanMatch
	ctxMap := map[string]interface{}{"ticker": ticker}

	for barIdx := startBar; barIdx < totalBars; barIdx++ {
		var evalOHLCV *indicator.OHLCV
		if barIdx == totalBars-1 {
			evalOHLCV = ohlcv // no slicing needed for the last bar
		} else {
			evalOHLCV = ohlcv.Slice(barIdx + 1)
		}

		ok, err := eval.EvalConditionList(evalOHLCV, conditions, combinationLogic, ctxMap)
		if err != nil || !ok {
			continue
		}

		lastClose := evalOHLCV.Close[evalOHLCV.Len()-1]
		matchDate := ""
		if lookbackDays > 0 && len(ohlcv.Dates) > barIdx {
			matchDate = ohlcv.Dates[barIdx]
		}

		matches = append(matches, &pricev1.ScanMatch{
			Ticker:             ticker,
			Name:               meta.name,
			Exchange:           meta.exchange,
			Country:            meta.country,
			AssetType:          meta.assetType,
			Price:              lastClose,
			RbicsEconomy:       meta.rbicsEconomy,
			RbicsSector:        meta.rbicsSector,
			RbicsSubsector:     meta.rbicsSubsector,
			RbicsIndustryGroup: meta.rbicsIndustryGroup,
			RbicsIndustry:      meta.rbicsIndustry,
			RbicsSubindustry:   meta.rbicsSubindustry,
			MatchDate:          matchDate,
		})
	}
	return matches
}

func (s *Server) loadDailyOHLCV(ctx context.Context, q *db.Queries, ticker string) *indicator.OHLCV {
	since := time.Now().AddDate(0, 0, -scanLookbackDaysDaily)
	rows, err := q.GetDailyPricesBatch(ctx, db.GetDailyPricesBatchParams{
		Tickers:   []string{ticker},
		SinceDate: pgtype.Date{Time: since, Valid: true},
	})
	if err != nil || len(rows) == 0 {
		return nil
	}
	return rowsToOHLCVDaily(rows)
}

func (s *Server) loadHourlyOHLCV(ctx context.Context, q *db.Queries, ticker string) *indicator.OHLCV {
	since := time.Now().AddDate(0, 0, -scanLookbackDaysHourly)
	rows, err := q.GetHourlyPricesBatch(ctx, db.GetHourlyPricesBatchParams{
		Tickers: []string{ticker},
		SinceTs: pgtype.Timestamptz{Time: since, Valid: true},
	})
	if err != nil || len(rows) == 0 {
		return nil
	}
	return rowsToOHLCVHourly(rows)
}

func (s *Server) loadWeeklyOHLCV(ctx context.Context, q *db.Queries, ticker string) *indicator.OHLCV {
	since := time.Now().AddDate(0, 0, -scanLookbackDaysWeekly)
	rows, err := q.GetWeeklyPricesBatch(ctx, db.GetWeeklyPricesBatchParams{
		Tickers:   []string{ticker},
		SinceDate: pgtype.Date{Time: since, Valid: true},
	})
	if err != nil || len(rows) == 0 {
		return nil
	}
	today := time.Now().UTC().Truncate(24 * time.Hour)
	filtered := rows[:0]
	for _, r := range rows {
		if r.WeekEnding.Valid && r.WeekEnding.Time.After(today) {
			continue
		}
		filtered = append(filtered, r)
	}
	rows = filtered
	if len(rows) == 0 {
		return nil
	}
	return rowsToOHLCVWeekly(rows)
}

func rowsToOHLCVDaily(rows []db.GetDailyPricesBatchRow) *indicator.OHLCV {
	n := len(rows)
	ohlcv := &indicator.OHLCV{
		Open:   make([]float64, n),
		High:   make([]float64, n),
		Low:    make([]float64, n),
		Close:  make([]float64, n),
		Volume: make([]float64, n),
		Dates:  make([]string, n),
	}
	for i, r := range rows {
		ohlcv.Open[i] = r.Open.Float64
		ohlcv.High[i] = r.High.Float64
		ohlcv.Low[i] = r.Low.Float64
		ohlcv.Close[i] = r.Close
		ohlcv.Volume[i] = float64(r.Volume.Int64)
		if r.Date.Valid {
			ohlcv.Dates[i] = r.Date.Time.Format("2006-01-02")
		}
	}
	return ohlcv
}

func rowsToOHLCVHourly(rows []db.GetHourlyPricesBatchRow) *indicator.OHLCV {
	n := len(rows)
	ohlcv := &indicator.OHLCV{
		Open:   make([]float64, n),
		High:   make([]float64, n),
		Low:    make([]float64, n),
		Close:  make([]float64, n),
		Volume: make([]float64, n),
		Dates:  make([]string, n),
	}
	for i, r := range rows {
		ohlcv.Open[i] = r.Open.Float64
		ohlcv.High[i] = r.High.Float64
		ohlcv.Low[i] = r.Low.Float64
		ohlcv.Close[i] = r.Close
		ohlcv.Volume[i] = float64(r.Volume.Int64)
		if r.Datetime.Valid {
			ohlcv.Dates[i] = r.Datetime.Time.Format(time.RFC3339)
		}
	}
	return ohlcv
}

func rowsToOHLCVWeekly(rows []db.GetWeeklyPricesBatchRow) *indicator.OHLCV {
	n := len(rows)
	ohlcv := &indicator.OHLCV{
		Open:   make([]float64, n),
		High:   make([]float64, n),
		Low:    make([]float64, n),
		Close:  make([]float64, n),
		Volume: make([]float64, n),
		Dates:  make([]string, n),
	}
	for i, r := range rows {
		ohlcv.Open[i] = r.Open.Float64
		ohlcv.High[i] = r.High.Float64
		ohlcv.Low[i] = r.Low.Float64
		ohlcv.Close[i] = r.Close
		ohlcv.Volume[i] = float64(r.Volume.Int64)
		if r.WeekEnding.Valid {
			ohlcv.Dates[i] = r.WeekEnding.Time.Format("2006-01-02")
		}
	}
	return ohlcv
}

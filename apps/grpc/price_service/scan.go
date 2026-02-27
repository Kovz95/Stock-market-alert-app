package main

import (
	"context"
	"log"
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
	scanLookbackDays   = 250
	scanMinBars        = 50
	scanMaxConcurrency = 20
	scanMaxTickersCap  = 20000
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

	var mu sync.Mutex
	var matches []*pricev1.ScanMatch
	sem := make(chan struct{}, scanMaxConcurrency)
	var wg sync.WaitGroup

	for _, ticker := range tickers {
		ticker := ticker
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer func() { <-sem; wg.Done() }()
			workerConn, err := s.pool.Acquire(ctx)
			if err != nil {
				return
			}
			defer workerConn.Release()
			q := db.New(workerConn)
			m := s.scanOneTicker(ctx, q, eval, ticker, metaBySymbol[ticker], req.Timeframe, req.Conditions, combinationLogic)
			if m != nil {
				mu.Lock()
				matches = append(matches, m)
				mu.Unlock()
			}
		}()
	}

	wg.Wait()

	return &pricev1.RunScanResponse{Matches: matches}, nil
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
) *pricev1.ScanMatch {
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

	log.Printf("[scanOneTicker] ticker=%s timeframe=%v conditions=%v combinationLogic=%s", ticker, timeframe, conditions, combinationLogic)

	ctxMap := map[string]interface{}{"ticker": ticker}
	ok, err := eval.EvalConditionList(ohlcv, conditions, combinationLogic, ctxMap)
	if err != nil || !ok {
		return nil
	}

	lastClose := ohlcv.Close[ohlcv.Len()-1]
	return &pricev1.ScanMatch{
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
	}
}

func (s *Server) loadDailyOHLCV(ctx context.Context, q *db.Queries, ticker string) *indicator.OHLCV {
	since := time.Now().AddDate(0, 0, -scanLookbackDays)
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
	since := time.Now().AddDate(0, 0, -scanLookbackDays)
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
	since := time.Now().AddDate(0, 0, -scanLookbackDays)
	rows, err := q.GetWeeklyPricesBatch(ctx, db.GetWeeklyPricesBatchParams{
		Tickers:   []string{ticker},
		SinceDate: pgtype.Date{Time: since, Valid: true},
	})
	if err != nil || len(rows) == 0 {
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
	}
	for i, r := range rows {
		ohlcv.Open[i] = r.Open.Float64
		ohlcv.High[i] = r.High.Float64
		ohlcv.Low[i] = r.Low.Float64
		ohlcv.Close[i] = r.Close
		ohlcv.Volume[i] = float64(r.Volume.Int64)
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
	}
	for i, r := range rows {
		ohlcv.Open[i] = r.Open.Float64
		ohlcv.High[i] = r.High.Float64
		ohlcv.Low[i] = r.Low.Float64
		ohlcv.Close[i] = r.Close
		ohlcv.Volume[i] = float64(r.Volume.Int64)
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
	}
	for i, r := range rows {
		ohlcv.Open[i] = r.Open.Float64
		ohlcv.High[i] = r.High.Float64
		ohlcv.Low[i] = r.Low.Float64
		ohlcv.Close[i] = r.Close
		ohlcv.Volume[i] = float64(r.Volume.Int64)
	}
	return ohlcv
}

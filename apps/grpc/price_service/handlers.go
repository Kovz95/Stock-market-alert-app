package main

import (
	"context"
	"time"

	"github.com/jackc/pgx/v5/pgtype"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

	"stockalert/calendar"
	db "stockalert/database/generated"
	pricev1 "stockalert/gen/go/price/v1"
)

func (s *Server) GetStockMetadataMap(ctx context.Context, req *pricev1.GetStockMetadataMapRequest) (*pricev1.GetStockMetadataMapResponse, error) {
	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	rows, err := db.New(conn).ListStockMetadataForPriceDb(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to list stock metadata: %v", err)
	}

	items := make([]*pricev1.StockMetadata, 0, len(rows))
	for _, r := range rows {
		items = append(items, &pricev1.StockMetadata{
			Symbol:   r.Symbol,
			Name:     pgTextString(r.Name),
			Exchange: pgTextString(r.Exchange),
			Isin:     pgTextString(r.Isin),
		})
	}
	return &pricev1.GetStockMetadataMapResponse{Items: items}, nil
}

func (s *Server) GetFullStockMetadata(ctx context.Context, req *pricev1.GetFullStockMetadataRequest) (*pricev1.GetFullStockMetadataResponse, error) {
	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	rows, err := db.New(conn).ListFullStockMetadata(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to list full stock metadata: %v", err)
	}

	items := make([]*pricev1.FullStockMetadata, 0, len(rows))
	for _, r := range rows {
		item := &pricev1.FullStockMetadata{
			Symbol:             r.Symbol,
			Name:               pgTextString(r.Name),
			Exchange:           pgTextString(r.Exchange),
			Country:            pgTextString(r.Country),
			Isin:               pgTextString(r.Isin),
			AssetType:          pgTextString(r.AssetType),
			RbicsEconomy:       pgTextString(r.RbicsEconomy),
			RbicsSector:        pgTextString(r.RbicsSector),
			RbicsSubsector:     pgTextString(r.RbicsSubsector),
			RbicsIndustryGroup: pgTextString(r.RbicsIndustryGroup),
			RbicsIndustry:      pgTextString(r.RbicsIndustry),
			RbicsSubindustry:   pgTextString(r.RbicsSubindustry),
			DataSource:         pgTextString(r.DataSource),
			EtfIssuer:          interfaceToString(r.EtfIssuer),
			EtfAssetClass:      interfaceToString(r.EtfAssetClass),
			EtfFocus:           interfaceToString(r.EtfFocus),
			EtfNiche:           interfaceToString(r.EtfNiche),
		}
		if r.ClosingPrice.Valid {
			item.ClosingPrice = &r.ClosingPrice.Float64
		}
		if r.MarketValue.Valid {
			item.MarketValue = &r.MarketValue.Float64
		}
		if r.Sales.Valid {
			item.Sales = &r.Sales.Float64
		}
		if r.AvgDailyVolume.Valid {
			item.AvgDailyVolume = &r.AvgDailyVolume.Float64
		}
		if r.LastUpdated.Valid {
			item.LastUpdated = timestamppb.New(r.LastUpdated.Time)
		}
		// ExpenseRatio from sqlc is interface{}; set optional when it's a non-zero float64
		if expenseRatio, ok := r.ExpenseRatio.(float64); ok && expenseRatio != 0 {
			item.ExpenseRatio = &expenseRatio
		}

		if aum, ok := r.Aum.(float64); ok && aum != 0 {
			item.Aum = &aum
		}
		items = append(items, item)
	}
	return &pricev1.GetFullStockMetadataResponse{Items: items}, nil
}

func interfaceToString(v interface{}) string {
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

func (s *Server) GetDatabaseStats(ctx context.Context, req *pricev1.GetDatabaseStatsRequest) (*pricev1.GetDatabaseStatsResponse, error) {
	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	q := db.New(conn)
	stats := &pricev1.DatabaseStats{}

	if row, err := q.GetDailyStats(ctx); err == nil {
		stats.DailyRecords = row.RecordCount
		stats.DailyTickers = row.TickerCount
		if row.MinDate.Valid {
			stats.DailyMin = timestamppb.New(dateToTime(row.MinDate))
		}
		if row.MaxDate.Valid {
			stats.DailyMax = timestamppb.New(dateToTime(row.MaxDate))
		}
	}

	if row, err := q.GetHourlyStats(ctx); err == nil {
		stats.HourlyRecords = row.RecordCount
		stats.HourlyTickers = row.TickerCount
		if t, ok := row.MinDatetime.(time.Time); ok {
			stats.HourlyMin = timestamppb.New(t)
		}
		if t, ok := row.MaxDatetime.(time.Time); ok {
			stats.HourlyMax = timestamppb.New(t)
		}
	}

	if row, err := q.GetWeeklyStats(ctx); err == nil {
		stats.WeeklyRecords = row.RecordCount
		stats.WeeklyTickers = row.TickerCount
		if row.MinDate.Valid {
			stats.WeeklyMin = timestamppb.New(dateToTime(row.MinDate))
		}
		if row.MaxDate.Valid {
			stats.WeeklyMax = timestamppb.New(dateToTime(row.MaxDate))
		}
	}

	return &pricev1.GetDatabaseStatsResponse{Stats: stats}, nil
}

func (s *Server) LoadPriceData(ctx context.Context, req *pricev1.LoadPriceDataRequest) (*pricev1.LoadPriceDataResponse, error) {
	if req.MaxRows <= 0 {
		req.MaxRows = 5000
	}
	if req.MaxRows > 50000 {
		req.MaxRows = 50000
	}

	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	q := db.New(conn)

	dayFilter := dayFilterToInt(req.DayFilter)

	switch req.Timeframe {
	case pricev1.Timeframe_TIMEFRAME_HOURLY:
		startTS, endTS := timestampsToTimestamptz(req.StartDate, req.EndDate)
		rows, err := q.ListHourlyPrices(ctx, db.ListHourlyPricesParams{
			Tickers:   req.Tickers,
			StartTs:   startTS,
			EndTs:     endTS,
			DayFilter: dayFilter,
			LimitRows: req.MaxRows,
		})
		if err != nil {
			return nil, status.Errorf(codes.Internal, "failed to list hourly prices: %v", err)
		}
		return &pricev1.LoadPriceDataResponse{Rows: hourlyRowsToProto(rows)}, nil

	case pricev1.Timeframe_TIMEFRAME_DAILY:
		startDate, endDate := timestampsToDates(req.StartDate, req.EndDate)
		rows, err := q.ListDailyPrices(ctx, db.ListDailyPricesParams{
			Tickers:   req.Tickers,
			StartDate: startDate,
			EndDate:   endDate,
			DayFilter: dayFilter,
			LimitRows: req.MaxRows,
		})
		if err != nil {
			return nil, status.Errorf(codes.Internal, "failed to list daily prices: %v", err)
		}
		return &pricev1.LoadPriceDataResponse{Rows: dailyRowsToProto(rows)}, nil

	case pricev1.Timeframe_TIMEFRAME_WEEKLY:
		startDate, endDate := timestampsToDates(req.StartDate, req.EndDate)
		rows, err := q.ListWeeklyPrices(ctx, db.ListWeeklyPricesParams{
			Tickers:   req.Tickers,
			StartDate: startDate,
			EndDate:   endDate,
			DayFilter: dayFilter,
			LimitRows: req.MaxRows,
		})
		if err != nil {
			return nil, status.Errorf(codes.Internal, "failed to list weekly prices: %v", err)
		}
		return &pricev1.LoadPriceDataResponse{Rows: weeklyRowsToProto(rows)}, nil

	default:
		return nil, status.Errorf(codes.InvalidArgument, "unsupported timeframe: %v", req.Timeframe)
	}
}

func pgTextString(t pgtype.Text) string {
	if !t.Valid {
		return ""
	}
	return t.String
}

func dateToTime(d pgtype.Date) time.Time {
	if !d.Valid {
		return time.Time{}
	}
	return time.Date(d.Time.Year(), d.Time.Month(), d.Time.Day(), 0, 0, 0, 0, time.UTC)
}

func dayFilterToInt(df pricev1.DayFilter) int32 {
	switch df {
	case pricev1.DayFilter_DAY_FILTER_WEEKDAYS:
		return 1
	case pricev1.DayFilter_DAY_FILTER_WEEKENDS:
		return 2
	default:
		return 0
	}
}

func timestampsToDates(start, end *timestamppb.Timestamp) (pgtype.Date, pgtype.Date) {
	var s, e pgtype.Date
	if start != nil {
		t := start.AsTime()
		s = pgtype.Date{Time: time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, time.UTC), Valid: true}
	}
	if end != nil {
		t := end.AsTime()
		e = pgtype.Date{Time: time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, time.UTC), Valid: true}
	}
	return s, e
}

func timestampsToTimestamptz(start, end *timestamppb.Timestamp) (pgtype.Timestamptz, pgtype.Timestamptz) {
	var s, e pgtype.Timestamptz
	if start != nil {
		s = pgtype.Timestamptz{Time: start.AsTime(), Valid: true}
	}
	if end != nil {
		e = pgtype.Timestamptz{Time: end.AsTime(), Valid: true}
	}
	return s, e
}

func dailyRowsToProto(rows []db.ListDailyPricesRow) []*pricev1.PriceRow {
	out := make([]*pricev1.PriceRow, 0, len(rows))
	for _, r := range rows {
		var t *timestamppb.Timestamp
		if r.Date.Valid {
			t = timestamppb.New(dateToTime(r.Date))
		}
		out = append(out, &pricev1.PriceRow{
			Ticker: r.Ticker,
			Time:   t,
			Open:   pgFloat64(r.Open),
			High:   pgFloat64(r.High),
			Low:    pgFloat64(r.Low),
			Close:  r.Close,
			Volume: pgInt64(r.Volume),
		})
	}
	return out
}

func hourlyRowsToProto(rows []db.ListHourlyPricesRow) []*pricev1.PriceRow {
	out := make([]*pricev1.PriceRow, 0, len(rows))
	for _, r := range rows {
		var t *timestamppb.Timestamp
		if r.Datetime.Valid {
			t = timestamppb.New(r.Datetime.Time)
		}
		out = append(out, &pricev1.PriceRow{
			Ticker: r.Ticker,
			Time:   t,
			Open:   pgFloat64(r.Open),
			High:   pgFloat64(r.High),
			Low:    pgFloat64(r.Low),
			Close:  r.Close,
			Volume: pgInt64(r.Volume),
		})
	}
	return out
}

func weeklyRowsToProto(rows []db.ListWeeklyPricesRow) []*pricev1.PriceRow {
	out := make([]*pricev1.PriceRow, 0, len(rows))
	for _, r := range rows {
		var t *timestamppb.Timestamp
		if r.WeekEnding.Valid {
			t = timestamppb.New(dateToTime(r.WeekEnding))
		}
		out = append(out, &pricev1.PriceRow{
			Ticker: r.Ticker,
			Time:   t,
			Open:   pgFloat64(r.Open),
			High:   pgFloat64(r.High),
			Low:    pgFloat64(r.Low),
			Close:  r.Close,
			Volume: pgInt64(r.Volume),
		})
	}
	return out
}

func pgFloat64(f pgtype.Float8) float64 {
	if !f.Valid {
		return 0
	}
	return f.Float64
}

func pgInt64(i pgtype.Int8) int64 {
	if !i.Valid {
		return 0
	}
	return i.Int64
}

// --- Stale scan handlers ---

func (s *Server) ScanStaleDaily(ctx context.Context, req *pricev1.ScanStaleDailyRequest) (*pricev1.ScanStaleDailyResponse, error) {
	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	rows, err := db.New(conn).LastDailyPerTicker(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to list last daily per ticker: %v", err)
	}

	expected := calendar.ExpectedLastTradingDate(calendar.ExchangeNYSE, time.Now().UTC())
	expectedDate := time.Date(expected.Year(), expected.Month(), expected.Day(), 0, 0, 0, 0, time.UTC)
	var out []*pricev1.StaleTickerRow
	limit := req.Limit
	if limit <= 0 {
		limit = 10000
	}
	for _, r := range rows {
		if !r.LastDate.Valid {
			continue
		}
		lastDate := time.Date(r.LastDate.Time.Year(), r.LastDate.Time.Month(), r.LastDate.Time.Day(), 0, 0, 0, 0, time.UTC)
		if !lastDate.Before(expectedDate) {
			continue
		}
		daysOld := int32(expectedDate.Sub(lastDate).Hours() / 24)
		out = append(out, &pricev1.StaleTickerRow{
			Ticker:      r.Ticker,
			LastUpdate:  timestamppb.New(lastDate),
			DaysOld:     daysOld,
			CompanyName: pgTextString(r.Name),
			Exchange:    pgTextString(r.Exchange),
		})
		if int32(len(out)) >= limit {
			break
		}
	}
	return &pricev1.ScanStaleDailyResponse{Rows: out}, nil
}

func (s *Server) ScanStaleWeekly(ctx context.Context, req *pricev1.ScanStaleWeeklyRequest) (*pricev1.ScanStaleWeeklyResponse, error) {
	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	expected := calendar.ExpectedLastWeekEnding(calendar.ExchangeNYSE, time.Now().UTC())
	limit := req.Limit
	if limit <= 0 {
		limit = 10000
	}
	rows, err := db.New(conn).StaleWeeklyTickers(ctx, db.StaleWeeklyTickersParams{
		ExpectedWeekEnding: pgtype.Date{Time: expected, Valid: true},
		LimitRows:          limit,
	})
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to list stale weekly tickers: %v", err)
	}

	out := make([]*pricev1.StaleTickerRow, 0, len(rows))
	for _, r := range rows {
		var lastUpdate *timestamppb.Timestamp
		if r.LastDate.Valid {
			lastUpdate = timestamppb.New(dateToTime(r.LastDate))
		}
		out = append(out, &pricev1.StaleTickerRow{
			Ticker:      r.Ticker,
			LastUpdate:  lastUpdate,
			DaysOld:     r.DaysOld,
			CompanyName: pgTextString(r.Name),
			Exchange:    pgTextString(r.Exchange),
		})
	}
	return &pricev1.ScanStaleWeeklyResponse{Rows: out}, nil
}

func (s *Server) ScanStaleHourly(ctx context.Context, req *pricev1.ScanStaleHourlyRequest) (*pricev1.ScanStaleHourlyResponse, error) {
	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	rows, err := db.New(conn).LastHourlyPerTicker(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to list last hourly per ticker: %v", err)
	}

	refHour := referenceHourUTC(time.Now().UTC())
	limit := req.Limit
	if limit <= 0 {
		limit = 10000
	}
	var staleRows []*pricev1.StaleHourlyRow
	upToDate := int32(0)
	for _, r := range rows {
		lastDt, ok := r.LastDt.(time.Time)
		if !ok {
			continue
		}
		lastHour := lastDt.Truncate(time.Hour)
		if !lastHour.Before(refHour) {
			upToDate++
			continue
		}
		hoursBehind := refHour.Sub(lastHour).Hours()
		staleRows = append(staleRows, &pricev1.StaleHourlyRow{
			Ticker:      r.Ticker,
			LastHour:    timestamppb.New(lastDt),
			HoursBehind: hoursBehind,
		})
		if int32(len(staleRows)) >= limit {
			break
		}
	}
	return &pricev1.ScanStaleHourlyResponse{
		Rows:          staleRows,
		LatestHour:    timestamppb.New(refHour),
		TotalTickers:  int32(len(rows)),
		UpToDateCount: upToDate,
	}, nil
}

func (s *Server) GetHourlyDataQuality(ctx context.Context, req *pricev1.GetHourlyDataQualityRequest) (*pricev1.GetHourlyDataQualityResponse, error) {
	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	q := db.New(conn)
	resp := &pricev1.GetHourlyDataQualityResponse{}

	staleRow, err := q.HourlyQualityStale(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get hourly quality stale: %v", err)
	}
	resp.TotalTickers = staleRow.TotalTickers
	resp.StaleTickers = staleRow.StaleTickers
	if t, ok := staleRow.OldestStale.(time.Time); ok {
		resp.OldestStale = timestamppb.New(t)
	}

	gapsRow, err := q.HourlyQualityGaps(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to get hourly quality gaps: %v", err)
	}
	resp.GapTickers = gapsRow.GapTickers
	resp.WorstGapHours = gapsRow.WorstGapHours
	resp.WorstCalendarGapHours = gapsRow.WorstCalendarGapHours

	return resp, nil
}

// streamWithContext is used to get the request context from the streaming RPC.
type streamWithContext interface {
	Send(*pricev1.UpdatePricesProgress) error
	Context() context.Context
}

func (s *Server) UpdatePrices(req *pricev1.UpdatePricesRequest, stream grpc.ServerStreamingServer[pricev1.UpdatePricesProgress]) error {
	ctx := context.Background()
	if sc, ok := stream.(streamWithContext); ok {
		ctx = sc.Context()
	}
	log := s.logger.With("rpc", "UpdatePrices", "timeframe", timeframeToString(req.Timeframe), "exchanges", req.Exchanges)

	if len(req.Exchanges) == 0 {
		log.Warn("UpdatePrices rejected: no exchanges")
		_ = stream.Send(&pricev1.UpdatePricesProgress{
			Done:         true,
			ErrorMessage: "at least one exchange is required",
		})
		return nil
	}
	tf := timeframeToString(req.Timeframe)
	if tf == "" {
		log.Warn("UpdatePrices rejected: invalid timeframe", "timeframe", req.Timeframe)
		_ = stream.Send(&pricev1.UpdatePricesProgress{
			Done:         true,
			ErrorMessage: "invalid timeframe: must be HOURLY, DAILY, or WEEKLY",
		})
		return nil
	}
	tickerFilter := len(req.Tickers) > 0
	log.Info("UpdatePrices started", "ticker_filter", tickerFilter, "requested_tickers", len(req.Tickers))

	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		log.Error("UpdatePrices failed to acquire connection", "error", err)
		_ = stream.Send(&pricev1.UpdatePricesProgress{Done: true, ErrorMessage: "failed to acquire connection"})
		return nil
	}
	defer conn.Release()

	q := db.New(conn)
	metaRows, err := q.ListStockMetadataForPriceDb(ctx)
	if err != nil {
		log.Error("UpdatePrices failed to list metadata", "error", err)
		_ = stream.Send(&pricev1.UpdatePricesProgress{Done: true, ErrorMessage: "failed to list metadata"})
		return nil
	}

	exchangeToTickers := make(map[string][]string)
	for _, r := range metaRows {
		ex := pgTextString(r.Exchange)
		if ex == "" {
			continue
		}
		exchangeToTickers[ex] = append(exchangeToTickers[ex], r.Symbol)
	}

	requestTickersSet := make(map[string]struct{})
	for _, t := range req.Tickers {
		requestTickersSet[t] = struct{}{}
	}
	if len(requestTickersSet) > 0 {
		for ex := range exchangeToTickers {
			filtered := exchangeToTickers[ex][:0]
			for _, t := range exchangeToTickers[ex] {
				if _, ok := requestTickersSet[t]; ok {
					filtered = append(filtered, t)
				}
			}
			exchangeToTickers[ex] = filtered
		}
	}

	for _, ex := range req.Exchanges {
		log.Info("UpdatePrices exchange ticker count", "exchange", ex, "tickers", len(exchangeToTickers[ex]))
	}

	total := int32(len(req.Exchanges))
	for i, exchange := range req.Exchanges {
		tickers := exchangeToTickers[exchange]
		excLog := log.With("exchange", exchange, "exchange_index", i+1, "exchange_total", total, "ticker_count", len(tickers))
		excLog.Info("UpdatePrices processing exchange")
		updated, failed, errUpdate := s.updater.updateForTickers(ctx, q, exchange, tf, tickers)
		if errUpdate != nil {
			excLog.Error("UpdatePrices exchange failed", "error", errUpdate)
			_ = stream.Send(&pricev1.UpdatePricesProgress{
				Exchange:      exchange,
				ExchangeIndex:  int32(i + 1),
				ExchangeTotal:  total,
				Done:          true,
				ErrorMessage:  errUpdate.Error(),
			})
			return nil
		}
		excLog.Info("UpdatePrices exchange complete", "tickers_updated", updated, "tickers_failed", failed)
		if err := stream.Send(&pricev1.UpdatePricesProgress{
			Exchange:       exchange,
			ExchangeIndex:  int32(i + 1),
			ExchangeTotal:  total,
			TickersUpdated: int32(updated),
			TickersFailed:  int32(failed),
		}); err != nil {
			return err
		}
	}
	log.Info("UpdatePrices completed successfully")
	return stream.Send(&pricev1.UpdatePricesProgress{Done: true})
}

func timeframeToString(tf pricev1.Timeframe) string {
	switch tf {
	case pricev1.Timeframe_TIMEFRAME_HOURLY:
		return "hourly"
	case pricev1.Timeframe_TIMEFRAME_DAILY:
		return "daily"
	case pricev1.Timeframe_TIMEFRAME_WEEKLY:
		return "weekly"
	default:
		return ""
	}
}

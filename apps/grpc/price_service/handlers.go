package main

import (
	"context"
	"time"

	"github.com/jackc/pgx/v5/pgtype"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/timestamppb"

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
			StartTs:  startTS,
			EndTs:    endTS,
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

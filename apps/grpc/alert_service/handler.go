package main

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/structpb"
	"google.golang.org/protobuf/types/known/timestamppb"

	alertv1 "stockalert/gen/go/alert/v1"
	db "stockalert/database/generated"
)

func (s *Server) ListAlerts(ctx context.Context, req *alertv1.ListAlertsRequest) (*alertv1.ListAlertsResponse, error) {
	pageSize := req.GetPageSize()
	if pageSize <= 0 {
		pageSize = 20
	}
	if pageSize > 100 {
		pageSize = 100
	}
	page := req.GetPage()
	if page < 1 {
		page = 1
	}
	offset := (page - 1) * pageSize

	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	q := db.New(conn)

	hasFilters := req.GetSearch() != "" ||
		len(req.GetExchanges()) > 0 ||
		len(req.GetTimeframes()) > 0 ||
		len(req.GetCountries()) > 0 ||
		req.GetTriggeredFilter() != "" ||
		req.GetConditionSearch() != ""

	var alerts []db.Alert
	var totalCount int64

	if hasFilters {
		// Use non-nil slices so pgx sends PostgreSQL '{}' instead of NULL;
		// NULL would make cardinality(NULL)=NULL and exclude all rows.
		exchanges := req.GetExchanges()
		if exchanges == nil {
			exchanges = []string{}
		}
		timeframes := req.GetTimeframes()
		if timeframes == nil {
			timeframes = []string{}
		}
		countries := req.GetCountries()
		if countries == nil {
			countries = []string{}
		}
		alerts, err = q.SearchAlertsPaginated(ctx, db.SearchAlertsPaginatedParams{
			Search:           req.GetSearch(),
			FilterExchanges:  exchanges,
			FilterTimeframes: timeframes,
			FilterCountries:  countries,
			TriggeredFilter:  req.GetTriggeredFilter(),
			ConditionSearch:  req.GetConditionSearch(),
			Lim:              pageSize,
			Off:              offset,
		})
		if err != nil {
			return nil, status.Errorf(codes.Internal, "failed to search alerts: %v", err)
		}
		totalCount, err = q.CountSearchAlerts(ctx, db.CountSearchAlertsParams{
			Search:           req.GetSearch(),
			FilterExchanges:  exchanges,
			FilterTimeframes: timeframes,
			FilterCountries:  countries,
			TriggeredFilter:  req.GetTriggeredFilter(),
			ConditionSearch:  req.GetConditionSearch(),
		})
		if err != nil {
			return nil, status.Errorf(codes.Internal, "failed to count search alerts: %v", err)
		}
	} else {
		alerts, err = q.ListAlertsPaginated(ctx, db.ListAlertsPaginatedParams{
			Limit:  pageSize,
			Offset: offset,
		})
		if err != nil {
			return nil, status.Errorf(codes.Internal, "failed to list alerts: %v", err)
		}
		totalCount, err = q.CountAlerts(ctx)
		if err != nil {
			return nil, status.Errorf(codes.Internal, "failed to count alerts: %v", err)
		}
	}

	protoAlerts := make([]*alertv1.Alert, len(alerts))
	for i, a := range alerts {
		protoAlerts[i] = dbAlertToProto(a)
	}

	hasNextPage := offset+int32(len(alerts)) < int32(totalCount)

	return &alertv1.ListAlertsResponse{
		Alerts:      protoAlerts,
		HasNextPage: hasNextPage,
		TotalCount:  int32(totalCount),
	}, nil
}

const searchAlertsStreamChunkSize = 100

func (s *Server) SearchAlertsStream(req *alertv1.SearchAlertsStreamRequest, stream grpc.ServerStreamingServer[alertv1.SearchAlertsStreamChunk]) error {
	ctx := stream.Context()
	exchanges := req.GetExchanges()
	if exchanges == nil {
		exchanges = []string{}
	}
	timeframes := req.GetTimeframes()
	if timeframes == nil {
		timeframes = []string{}
	}
	countries := req.GetCountries()
	if countries == nil {
		countries = []string{}
	}

	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	q := db.New(conn)
	offset := int32(0)
	for {
		alerts, err := q.SearchAlertsPaginated(ctx, db.SearchAlertsPaginatedParams{
			Search:           req.GetSearch(),
			FilterExchanges:  exchanges,
			FilterTimeframes: timeframes,
			FilterCountries:  countries,
			TriggeredFilter:  req.GetTriggeredFilter(),
			ConditionSearch:  req.GetConditionSearch(),
			Lim:              searchAlertsStreamChunkSize,
			Off:              offset,
		})
		if err != nil {
			return status.Errorf(codes.Internal, "failed to search alerts: %v", err)
		}
		protoAlerts := make([]*alertv1.Alert, len(alerts))
		for i, a := range alerts {
			protoAlerts[i] = dbAlertToProto(a)
		}
		done := len(alerts) < searchAlertsStreamChunkSize
		if err := stream.Send(&alertv1.SearchAlertsStreamChunk{Alerts: protoAlerts, Done: done}); err != nil {
			return err
		}
		if done {
			break
		}
		offset += searchAlertsStreamChunkSize
	}
	return nil
}

func (s *Server) GetAlert(ctx context.Context, req *alertv1.GetAlertRequest) (*alertv1.GetAlertResponse, error) {
	if req.GetAlertId() == "" {
		return nil, status.Error(codes.InvalidArgument, "alert_id is required")
	}

	alertID, err := parseUUID(req.GetAlertId())
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid alert_id: %v", err)
	}

	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	alert, err := db.New(conn).GetAlert(ctx, alertID)
	if err != nil {
		if err == pgx.ErrNoRows {
			return nil, status.Error(codes.NotFound, "alert not found")
		}
		return nil, status.Errorf(codes.Internal, "failed to get alert: %v", err)
	}

	return &alertv1.GetAlertResponse{Alert: dbAlertToProto(alert)}, nil
}

func (s *Server) CreateAlert(ctx context.Context, req *alertv1.CreateAlertRequest) (*alertv1.CreateAlertResponse, error) {
	if req.GetName() == "" {
		return nil, status.Error(codes.InvalidArgument, "name is required")
	}

	alertID := pgtype.UUID{}
	// Generate UUID via Postgres
	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	err = conn.QueryRow(ctx, "SELECT uuid_generate_v4()").Scan(&alertID)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to generate UUID: %v", err)
	}

	params := db.CreateAlertParams{
		AlertID:              alertID,
		Name:                 req.GetName(),
		StockName:            textFromString(req.GetStockName()),
		Ticker:               textFromString(req.GetTicker()),
		Ticker1:              textFromString(req.GetTicker1()),
		Ticker2:              textFromString(req.GetTicker2()),
		Conditions:           structToJSON(req.GetConditions()),
		CombinationLogic:     textFromString(req.GetCombinationLogic()),
		LastTriggered:        pgtype.Timestamptz{},
		Action:               textFromString(req.GetAction()),
		Timeframe:            textFromString(req.GetTimeframe()),
		Exchange:             textFromString(req.GetExchange()),
		Country:              textFromString(req.GetCountry()),
		Ratio:                textFromString(req.GetRatio()),
		IsRatio:              pgtype.Bool{Bool: req.GetIsRatio(), Valid: true},
		AdjustmentMethod:     textFromString(req.GetAdjustmentMethod()),
		DtpParams:            structToJSON(req.GetDtpParams()),
		MultiTimeframeParams: structToJSON(req.GetMultiTimeframeParams()),
		MixedTimeframeParams: structToJSON(req.GetMixedTimeframeParams()),
		RawPayload:           structToJSON(req.GetRawPayload()),
	}

	alert, err := db.New(conn).CreateAlert(ctx, params)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to create alert: %v", err)
	}

	return &alertv1.CreateAlertResponse{Alert: dbAlertToProto(alert)}, nil
}

func (s *Server) UpdateAlert(ctx context.Context, req *alertv1.UpdateAlertRequest) (*alertv1.UpdateAlertResponse, error) {
	if req.GetAlertId() == "" {
		return nil, status.Error(codes.InvalidArgument, "alert_id is required")
	}

	alertID, err := parseUUID(req.GetAlertId())
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid alert_id: %v", err)
	}

	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	params := db.UpdateAlertParams{
		AlertID:              alertID,
		Name:                 req.GetName(),
		StockName:            textFromString(req.GetStockName()),
		Ticker:               textFromString(req.GetTicker()),
		Ticker1:              textFromString(req.GetTicker1()),
		Ticker2:              textFromString(req.GetTicker2()),
		Conditions:           structToJSON(req.GetConditions()),
		CombinationLogic:     textFromString(req.GetCombinationLogic()),
		Action:               textFromString(req.GetAction()),
		Timeframe:            textFromString(req.GetTimeframe()),
		Exchange:             textFromString(req.GetExchange()),
		Country:              textFromString(req.GetCountry()),
		Ratio:                textFromString(req.GetRatio()),
		IsRatio:              pgtype.Bool{Bool: req.GetIsRatio(), Valid: true},
		AdjustmentMethod:     textFromString(req.GetAdjustmentMethod()),
		DtpParams:            structToJSON(req.GetDtpParams()),
		MultiTimeframeParams: structToJSON(req.GetMultiTimeframeParams()),
		MixedTimeframeParams: structToJSON(req.GetMixedTimeframeParams()),
		RawPayload:           structToJSON(req.GetRawPayload()),
	}

	alert, err := db.New(conn).UpdateAlert(ctx, params)
	if err != nil {
		if err == pgx.ErrNoRows {
			return nil, status.Error(codes.NotFound, "alert not found")
		}
		return nil, status.Errorf(codes.Internal, "failed to update alert: %v", err)
	}

	return &alertv1.UpdateAlertResponse{Alert: dbAlertToProto(alert)}, nil
}

func (s *Server) DeleteAlert(ctx context.Context, req *alertv1.DeleteAlertRequest) (*alertv1.DeleteAlertResponse, error) {
	if req.GetAlertId() == "" {
		return nil, status.Error(codes.InvalidArgument, "alert_id is required")
	}

	alertID, err := parseUUID(req.GetAlertId())
	if err != nil {
		return nil, status.Errorf(codes.InvalidArgument, "invalid alert_id: %v", err)
	}

	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	err = db.New(conn).DeleteAlert(ctx, alertID)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to delete alert: %v", err)
	}

	return &alertv1.DeleteAlertResponse{}, nil
}

func (s *Server) BulkDeleteAlerts(ctx context.Context, req *alertv1.BulkDeleteAlertsRequest) (*alertv1.BulkDeleteAlertsResponse, error) {
	if len(req.GetAlertIds()) == 0 {
		return &alertv1.BulkDeleteAlertsResponse{DeletedCount: 0}, nil
	}

	uuids := make([]pgtype.UUID, 0, len(req.GetAlertIds()))
	for _, id := range req.GetAlertIds() {
		u, err := parseUUID(id)
		if err != nil {
			return nil, status.Errorf(codes.InvalidArgument, "invalid alert_id %q: %v", id, err)
		}
		uuids = append(uuids, u)
	}

	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	deleted, err := db.New(conn).BulkDeleteAlerts(ctx, uuids)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to bulk delete alerts: %v", err)
	}

	return &alertv1.BulkDeleteAlertsResponse{DeletedCount: int32(deleted)}, nil
}

func (s *Server) BulkUpdateLastTriggered(ctx context.Context, req *alertv1.BulkUpdateLastTriggeredRequest) (*alertv1.BulkUpdateLastTriggeredResponse, error) {
	conn, err := s.pool.Acquire(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to acquire connection: %v", err)
	}
	defer conn.Release()

	tx, err := conn.Begin(ctx)
	if err != nil {
		return nil, status.Errorf(codes.Internal, "failed to begin transaction: %v", err)
	}
	defer tx.Rollback(ctx)

	q := db.New(tx)
	var updatedCount int32
	for _, trigger := range req.GetTriggers() {
		alertID, err := parseUUID(trigger.GetAlertId())
		if err != nil {
			return nil, status.Errorf(codes.InvalidArgument, "invalid alert_id %q: %v", trigger.GetAlertId(), err)
		}

		ts := pgtype.Timestamptz{}
		if trigger.GetLastTriggered() != nil {
			ts = pgtype.Timestamptz{
				Time:  trigger.GetLastTriggered().AsTime(),
				Valid: true,
			}
		}

		err = q.BulkUpdateLastTriggered(ctx, db.BulkUpdateLastTriggeredParams{
			AlertID:       alertID,
			LastTriggered: ts,
		})
		if err != nil {
			return nil, status.Errorf(codes.Internal, "failed to update alert %q: %v", trigger.GetAlertId(), err)
		}
		updatedCount++
	}

	if err := tx.Commit(ctx); err != nil {
		return nil, status.Errorf(codes.Internal, "failed to commit transaction: %v", err)
	}

	return &alertv1.BulkUpdateLastTriggeredResponse{UpdatedCount: updatedCount}, nil
}

// --- Conversion helpers ---

func dbAlertToProto(a db.Alert) *alertv1.Alert {
	alert := &alertv1.Alert{
		AlertId:          uuidToString(a.AlertID),
		Name:             a.Name,
		StockName:        a.StockName.String,
		Ticker:           a.Ticker.String,
		Ticker1:          a.Ticker1.String,
		Ticker2:          a.Ticker2.String,
		CombinationLogic: a.CombinationLogic.String,
		Action:           a.Action.String,
		Timeframe:        a.Timeframe.String,
		Exchange:         a.Exchange.String,
		Country:          a.Country.String,
		Ratio:            a.Ratio.String,
		IsRatio:          a.IsRatio.Bool,
		AdjustmentMethod: a.AdjustmentMethod.String,
	}

	if a.LastTriggered.Valid {
		alert.LastTriggered = timestamppb.New(a.LastTriggered.Time)
	}
	if a.CreatedAt.Valid {
		alert.CreatedAt = timestamppb.New(a.CreatedAt.Time)
	}
	if a.UpdatedAt.Valid {
		alert.UpdatedAt = timestamppb.New(a.UpdatedAt.Time)
	}

	alert.Conditions = jsonToStruct(a.Conditions)
	alert.DtpParams = jsonToStruct(a.DtpParams)
	alert.MultiTimeframeParams = jsonToStruct(a.MultiTimeframeParams)
	alert.MixedTimeframeParams = jsonToStruct(a.MixedTimeframeParams)
	alert.RawPayload = jsonToStruct(a.RawPayload)

	return alert
}

func parseUUID(s string) (pgtype.UUID, error) {
	var u pgtype.UUID
	err := u.Scan(s)
	if err != nil {
		return u, fmt.Errorf("invalid UUID %q: %w", s, err)
	}
	return u, nil
}

func uuidToString(u pgtype.UUID) string {
	if !u.Valid {
		return ""
	}
	b := u.Bytes
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x",
		b[0:4], b[4:6], b[6:8], b[8:10], b[10:16])
}

func textFromString(s string) pgtype.Text {
	if s == "" {
		return pgtype.Text{}
	}
	return pgtype.Text{String: s, Valid: true}
}

func structToJSON(s *structpb.Struct) []byte {
	if s == nil {
		return nil
	}
	b, err := json.Marshal(s.AsMap())
	if err != nil {
		return nil
	}
	return b
}

func jsonToStruct(b []byte) *structpb.Struct {
	if len(b) == 0 {
		return nil
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		// JSON may be stored as an array (e.g. legacy or alternate format); wrap so we still return a Struct.
		var arr []any
		if arrErr := json.Unmarshal(b, &arr); arrErr != nil {
			return nil
		}
		m = map[string]any{"conditions": arr}
	}
	s, err := structpb.NewStruct(m)
	if err != nil {
		return nil
	}
	return s
}

package status

import (
	"context"
	"encoding/json"
	"log/slog"
	"time"

	"github.com/jackc/pgx/v5/pgtype"

	"stockalert/discord"
	db "stockalert/database/generated"
)

const documentKey = "scheduler_status"

// Manager updates scheduler status in app_documents for monitoring.
type Manager struct {
	queries *db.Queries
	logger  *slog.Logger
}

// NewManager creates a status manager that writes to app_documents.
func NewManager(queries *db.Queries, logger *slog.Logger) *Manager {
	if logger == nil {
		logger = slog.Default()
	}
	return &Manager{queries: queries, logger: logger.With("component", "status_manager")}
}

// UpdateRunning sets status to "running" and records the current job.
func (m *Manager) UpdateRunning(ctx context.Context, exchange, timeframe string) error {
	m.logger.Debug("updating status",
		"type", "running",
		"exchange", exchange,
		"timeframe", timeframe,
	)
	payload := map[string]interface{}{
		"status":      "running",
		"heartbeat":   time.Now().UTC().Format(time.RFC3339),
		"current_job": map[string]string{
			"exchange":  exchange,
			"timeframe": timeframe,
			"started":   time.Now().UTC().Format(time.RFC3339),
		},
	}
	return m.upsert(ctx, payload)
}

// UpdateSuccess clears current_job and sets last_run and last_result.
func (m *Manager) UpdateSuccess(ctx context.Context, exchange, timeframe string, priceStats *discord.PriceStats, alertStats *discord.AlertStats) error {
	m.logger.Debug("updating status",
		"type", "success",
		"exchange", exchange,
		"timeframe", timeframe,
	)
	payload := map[string]interface{}{
		"status":    "running",
		"heartbeat": time.Now().UTC().Format(time.RFC3339),
		"last_run": map[string]interface{}{
			"exchange":     exchange,
			"timeframe":    timeframe,
			"completed_at": time.Now().UTC().Format(time.RFC3339),
		},
		"last_result": map[string]interface{}{
			"price_stats": priceStats,
			"alert_stats": alertStats,
		},
	}
	return m.upsert(ctx, payload)
}

// UpdateError sets status to "error" and last_error.
func (m *Manager) UpdateError(ctx context.Context, exchange, timeframe, errMsg string) error {
	m.logger.Warn("updating status",
		"type", "error",
		"exchange", exchange,
		"timeframe", timeframe,
		"error_message", errMsg,
	)
	payload := map[string]interface{}{
		"status":    "error",
		"heartbeat": time.Now().UTC().Format(time.RFC3339),
		"last_error": map[string]string{
			"exchange":  exchange,
			"timeframe": timeframe,
			"message":   errMsg,
			"time":      time.Now().UTC().Format(time.RFC3339),
		},
	}
	return m.upsert(ctx, payload)
}

func (m *Manager) upsert(ctx context.Context, payload map[string]interface{}) error {
	raw, err := json.Marshal(payload)
	if err != nil {
		return err
	}
	err = m.queries.UpsertAppDocument(ctx, db.UpsertAppDocumentParams{
		DocumentKey: documentKey,
		Payload:     raw,
		SourcePath:  pgtype.Text{},
	})
	if err != nil {
		m.logger.Error("upsert failed", "error", err)
	}
	return err
}

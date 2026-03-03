package handler

import (
	"context"
	"encoding/json"

	"github.com/hibiken/asynq"

	"stockalert/apps/scheduler/internal/tasks"
)

// HourlyHandler processes task:hourly payloads (one exchange per task).
type HourlyHandler struct {
	*Common
}

// ProcessTask implements asynq.Handler.
func (h *HourlyHandler) ProcessTask(ctx context.Context, t *asynq.Task) error {
	var p tasks.Payload
	if err := json.Unmarshal(t.Payload(), &p); err != nil {
		h.Logger.Error("hourly task unmarshal error", "error", err, "raw_payload", string(t.Payload()))
		return err
	}
	if p.Exchange == "" {
		p.Exchange = "NYSE"
	}
	if p.Timeframe == "" {
		p.Timeframe = "hourly"
	}
	h.Logger.Info("hourly task received", "exchange", p.Exchange, "timeframe", p.Timeframe)
	statusNotifier := h.statusNotifierFor("hourly", "1h")
	_, _, err := h.Execute(ctx, p.Exchange, "hourly", statusNotifier)
	return err
}

// NewHourlyHandler returns a handler for hourly tasks.
func NewHourlyHandler(c *Common) *HourlyHandler {
	return &HourlyHandler{Common: c}
}

package handler

import (
	"context"
	"encoding/json"
	"log"

	"github.com/hibiken/asynq"

	"stockalert/apps/scheduler/internal/tasks"
)

// DailyHandler processes task:daily payloads (one exchange per task).
type DailyHandler struct {
	*Common
}

// ProcessTask implements asynq.Handler.
func (h *DailyHandler) ProcessTask(ctx context.Context, t *asynq.Task) error {
	var p tasks.Payload
	if err := json.Unmarshal(t.Payload(), &p); err != nil {
		return err
	}
	if p.Exchange == "" {
		p.Exchange = "NYSE"
	}
	if p.Timeframe == "" {
		p.Timeframe = "daily"
	}
	log.Printf("daily task: exchange=%s", p.Exchange)
	statusNotifier := h.statusNotifierFor("daily", "1d")
	_, _, err := h.Execute(ctx, p.Exchange, "daily", statusNotifier)
	return err
}

// NewDailyHandler returns a handler for daily tasks.
func NewDailyHandler(c *Common) *DailyHandler {
	return &DailyHandler{Common: c}
}

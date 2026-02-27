package handler

import (
	"context"
	"encoding/json"
	"log"

	"github.com/hibiken/asynq"

	"stockalert/apps/scheduler/internal/tasks"
)

// WeeklyHandler processes task:weekly payloads (one exchange per task).
type WeeklyHandler struct {
	*Common
}

// ProcessTask implements asynq.Handler.
func (h *WeeklyHandler) ProcessTask(ctx context.Context, t *asynq.Task) error {
	var p tasks.Payload
	if err := json.Unmarshal(t.Payload(), &p); err != nil {
		return err
	}
	if p.Exchange == "" {
		p.Exchange = "NYSE"
	}
	if p.Timeframe == "" {
		p.Timeframe = "weekly"
	}
	log.Printf("weekly task: exchange=%s", p.Exchange)
	statusNotifier := h.statusNotifierFor("weekly", "1wk")
	_, _, err := h.Execute(ctx, p.Exchange, "weekly", statusNotifier)
	return err
}

// NewWeeklyHandler returns a handler for weekly tasks.
func NewWeeklyHandler(c *Common) *WeeklyHandler {
	return &WeeklyHandler{Common: c}
}

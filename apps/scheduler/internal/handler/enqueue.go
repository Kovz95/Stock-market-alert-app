package handler

import (
	"context"
	"encoding/json"
	"log"
	"time"

	"github.com/hibiken/asynq"

	"stockalert/calendar"

	"stockalert/apps/scheduler/internal/tasks"
)

// EnqueueHandler enqueues task:daily, task:weekly, task:hourly for exchanges that are due.
type EnqueueHandler struct {
	client *asynq.Client
	window  time.Duration // enqueue if next run is within this window
}

// NewEnqueueHandler creates a handler that enqueues due exchange tasks.
func NewEnqueueHandler(client *asynq.Client) *EnqueueHandler {
	return &EnqueueHandler{
		client: client,
		window: 15 * time.Minute,
	}
}

// ProcessTask implements asynq.Handler. It enqueues task:daily, task:weekly, task:hourly
// for each exchange when due (daily/weekly from GetNextDailyRunTime, hourly when exchange is open).
func (h *EnqueueHandler) ProcessTask(ctx context.Context, _ *asynq.Task) error {
	now := time.Now().UTC()
	deadline := now.Add(h.window)

	for exchange := range calendar.ExchangeSchedules {
		// Daily: enqueue if next run (close+40min) is within window
		nextDaily := calendar.GetNextDailyRunTime(exchange, now)
		if !nextDaily.After(deadline) {
			h.enqueueOne(ctx, exchange, "daily")
		}
		// Weekly: same run time but only on Friday
		if nextDaily.Weekday() == time.Friday && !nextDaily.After(deadline) {
			h.enqueueOne(ctx, exchange, "weekly")
		}
		// Hourly: enqueue when exchange is open (dedup 30min avoids duplicate runs)
		if calendar.IsExchangeOpen(exchange, now) {
			h.enqueueOne(ctx, exchange, "hourly")
		}
	}
	return nil
}

func (h *EnqueueHandler) enqueueOne(ctx context.Context, exchange, timeframe string) {
	payload, _ := json.Marshal(tasks.Payload{Exchange: exchange, Timeframe: timeframe})
	task := asynq.NewTask(tasks.TaskName(timeframe), payload, asynq.Unique(30*time.Minute))
	info, err := h.client.EnqueueContext(ctx, task)
	if err != nil {
		log.Printf("enqueue %s %s: %v", exchange, timeframe, err)
		return
	}
	log.Printf("enqueued %s %s id=%s", exchange, timeframe, info.ID)
}

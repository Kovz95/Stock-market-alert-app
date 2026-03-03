// Package schedule provides a background scheduler that enqueues per-exchange
// daily, weekly, and hourly tasks using calendar.GetNextDailyRunTime and
// ProcessAt for DST-correct timing.
package schedule

import (
	"context"
	"encoding/json"
	"log/slog"
	"time"

	"github.com/hibiken/asynq"

	"stockalert/calendar"

	"stockalert/apps/scheduler/internal/tasks"
)

const (
	cycleInterval = 15 * time.Minute
	dailyUnique   = 12 * time.Hour
	hourlyUnique  = 30 * time.Minute
)

// Scheduler runs scheduleAll on startup and every 15 minutes, enqueueing
// task:daily, task:weekly (when Friday), and task:hourly (when exchange open)
// with ProcessAt/Unique so the same worker processes them at the right time.
type Scheduler struct {
	client *asynq.Client
	logger *slog.Logger
	stop   chan struct{}
	done   chan struct{}
}

// New returns a Scheduler that uses the given client and logger.
func New(client *asynq.Client, logger *slog.Logger) *Scheduler {
	return &Scheduler{
		client: client,
		logger: logger.With("component", "schedule"),
		stop:   make(chan struct{}),
		done:   make(chan struct{}),
	}
}

// Start starts the background goroutine that runs scheduleAll on startup and every 15 min.
func (s *Scheduler) Start(ctx context.Context) {
	go s.run(ctx)
}

// run loops: scheduleAll once, then every 15 min until stop or ctx done.
func (s *Scheduler) run(ctx context.Context) {
	defer close(s.done)
	ticker := time.NewTicker(cycleInterval)
	defer ticker.Stop()

	s.scheduleAll(ctx)
	for {
		select {
		case <-s.stop:
			return
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.scheduleAll(ctx)
		}
	}
}

// Stop signals the scheduler to stop and waits for the run goroutine to exit.
func (s *Scheduler) Stop() {
	close(s.stop)
	<-s.done
}

// scheduleAll iterates all exchanges, computes next run times, and enqueues
// daily (and weekly when Friday) with ProcessAt + Unique(12h), and hourly
// when exchange is open with Unique(30min).
func (s *Scheduler) scheduleAll(ctx context.Context) {
	now := time.Now().UTC()
	exchangeCount := len(calendar.ExchangeSchedules)
	s.logger.Info("schedule cycle starting",
		"exchange_count", exchangeCount,
		"at", now.Format(time.RFC3339),
	)
	start := time.Now()
	scheduled := 0

	for exchange := range calendar.ExchangeSchedules {
		nextDaily := calendar.GetNextDailyRunTime(exchange, now)
		payload, _ := json.Marshal(tasks.Payload{Exchange: exchange, Timeframe: "daily"})
		task := asynq.NewTask(tasks.TypeDaily, payload)
		_, err := s.client.EnqueueContext(ctx, task, asynq.ProcessAt(nextDaily), asynq.Unique(dailyUnique))
		if err != nil {
			s.logger.Warn("schedule daily failed", "exchange", exchange, "error", err)
			continue
		}
		scheduled++
		s.logger.Debug("task scheduled",
			"exchange", exchange,
			"timeframe", "daily",
			"process_at", nextDaily.Format(time.RFC3339),
		)
		if nextDaily.Weekday() == time.Friday {
			payloadWeekly, _ := json.Marshal(tasks.Payload{Exchange: exchange, Timeframe: "weekly"})
			taskWeekly := asynq.NewTask(tasks.TypeWeekly, payloadWeekly)
			_, err = s.client.EnqueueContext(ctx, taskWeekly, asynq.ProcessAt(nextDaily), asynq.Unique(dailyUnique))
			if err != nil {
				s.logger.Warn("schedule weekly failed", "exchange", exchange, "error", err)
				continue
			}
			scheduled++
			s.logger.Debug("task scheduled",
				"exchange", exchange,
				"timeframe", "weekly",
				"process_at", nextDaily.Format(time.RFC3339),
			)
		}
		if calendar.IsExchangeOpen(exchange, now) {
			payloadHourly, _ := json.Marshal(tasks.Payload{Exchange: exchange, Timeframe: "hourly"})
			taskHourly := asynq.NewTask(tasks.TypeHourly, payloadHourly)
			_, err = s.client.EnqueueContext(ctx, taskHourly, asynq.Unique(hourlyUnique))
			if err != nil {
				s.logger.Warn("schedule hourly failed", "exchange", exchange, "error", err)
				continue
			}
			scheduled++
			s.logger.Debug("task scheduled", "exchange", exchange, "timeframe", "hourly")
		}
	}

	s.logger.Info("schedule cycle complete",
		"scheduled", scheduled,
		"duration_ms", time.Since(start).Milliseconds(),
	)
}

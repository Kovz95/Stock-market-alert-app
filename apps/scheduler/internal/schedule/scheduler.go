// Package schedule provides a background scheduler that enqueues per-exchange
// daily, weekly, and hourly tasks using calendar.GetNextDailyRunTime and
// ProcessAt for DST-correct timing.
package schedule

import (
	"context"
	"encoding/json"
	"log/slog"
	"sort"
	"strings"
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
// FMPAPIKey (from env FMP_API_KEY) is used to fetch market hours for open/closed; if empty, fallback calendar is used.
type Scheduler struct {
	client    *asynq.Client
	logger    *slog.Logger
	fmpAPIKey string
	stop      chan struct{}
	done      chan struct{}
}

// New returns a Scheduler that uses the given client, logger, and optional FMP API key for market hours.
func New(client *asynq.Client, logger *slog.Logger, fmpAPIKey string) *Scheduler {
	return &Scheduler{
		client:    client,
		logger:    logger.With("component", "schedule"),
		fmpAPIKey: fmpAPIKey,
		stop:      make(chan struct{}),
		done:      make(chan struct{}),
	}
}

// Start starts the background goroutine that runs scheduleAll on startup and every 15 min.
func (s *Scheduler) Start(ctx context.Context) {
	go s.run(ctx)
}

// run loops: scheduleAll once after a short delay (so Redis/server are ready), then every 15 min until stop or ctx done.
func (s *Scheduler) run(ctx context.Context) {
	defer close(s.done)
	ticker := time.NewTicker(cycleInterval)
	defer ticker.Stop()

	// Give Redis and asynq server a moment to be ready before first enqueue.
	const startupDelay = 3 * time.Second
	select {
	case <-time.After(startupDelay):
		s.logger.Info("running initial schedule cycle")
		s.scheduleAll(ctx)
	case <-s.stop:
		return
	case <-ctx.Done():
		return
	}
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

// isUniqueConflict returns true if err is asynq's "task already exists" from
// the Unique option. That is expected when we re-run every 15 min and the
// same task was already enqueued within the uniqueness window (12h daily, 30m hourly).
func isUniqueConflict(err error) bool {
	return err != nil && strings.Contains(err.Error(), "task already exists")
}

// scheduleAll iterates all exchanges (in deterministic order), and for each
// exchange attempts to enqueue: (1) daily, (2) weekly if next daily run is
// Friday, (3) hourly if the exchange is currently open. Each task type uses
// Unique so duplicates within the window are skipped; failures for one type
// do not skip the others. Tasks use ProcessAt and appear in Redis as
// "scheduled" until run time.
//
// Hourly: scheduled_hourly can be 0 even when markets are open because we use
// Unique(30min)—only one task per exchange per 30 min. Check open_exchanges in
// the log: if > 0, hourly was considered and either newly enqueued or already enqueued.
func (s *Scheduler) scheduleAll(ctx context.Context) {
	now := time.Now().UTC()
	exchanges := make([]string, 0, len(calendar.ExchangeSchedules))
	for exchange := range calendar.ExchangeSchedules {
		exchanges = append(exchanges, exchange)
	}
	sort.Strings(exchanges)
	exchangeCount := len(exchanges)

	// Single FMP call per cycle: use for "is exchange open" when available; fallback to local calendar on error or missing key.
	var fmpSnapshot *calendar.MarketHoursSnapshot
	if s.fmpAPIKey != "" {
		snap, err := calendar.FetchAllExchangeMarketHours(ctx, s.fmpAPIKey, nil)
		if err != nil {
			s.logger.Warn("FMP market hours fetch failed, using calendar fallback", "error", err)
		} else {
			fmpSnapshot = snap
		}
	}

	s.logger.Info("schedule cycle starting",
		"exchange_count", exchangeCount,
		"now_utc", now.Format(time.RFC3339),
		"fmp_snapshot", fmpSnapshot != nil,
	)
	start := time.Now()
	var scheduledDaily, scheduledWeekly, scheduledHourly, openExchangeCount int

	for _, exchange := range exchanges {
		nextDaily := calendar.GetNextDailyRunTime(exchange, now)
		payload, _ := json.Marshal(tasks.Payload{Exchange: exchange, Timeframe: "daily"})
		task := asynq.NewTask(tasks.TypeDaily, payload)
		_, err := s.client.EnqueueContext(ctx, task, asynq.ProcessAt(nextDaily), asynq.Unique(dailyUnique))
		if err != nil {
			if isUniqueConflict(err) {
				s.logger.Debug("schedule daily skipped (already enqueued)", "exchange", exchange)
			} else {
				s.logger.Warn("schedule daily failed", "exchange", exchange, "error", err)
			}
		} else {
			scheduledDaily++
			s.logger.Debug("task scheduled",
				"exchange", exchange,
				"timeframe", "daily",
				"process_at", nextDaily.Format(time.RFC3339),
			)
		}

		if nextDaily.Weekday() == time.Friday {
			payloadWeekly, _ := json.Marshal(tasks.Payload{Exchange: exchange, Timeframe: "weekly"})
			taskWeekly := asynq.NewTask(tasks.TypeWeekly, payloadWeekly)
			_, err = s.client.EnqueueContext(ctx, taskWeekly, asynq.ProcessAt(nextDaily), asynq.Unique(dailyUnique))
			if err != nil {
				if isUniqueConflict(err) {
					s.logger.Debug("schedule weekly skipped (already enqueued)", "exchange", exchange)
				} else {
					s.logger.Warn("schedule weekly failed", "exchange", exchange, "error", err)
				}
			} else {
				scheduledWeekly++
				s.logger.Debug("task scheduled",
					"exchange", exchange,
					"timeframe", "weekly",
					"process_at", nextDaily.Format(time.RFC3339),
				)
			}
		}

		open := false
		if fmpSnapshot != nil {
			open = calendar.IsExchangeOpenFromSnapshot(exchange, fmpSnapshot)
		} else {
			open = calendar.IsExchangeOpen(exchange, now)
		}
		if open {
			openExchangeCount++
			payloadHourly, _ := json.Marshal(tasks.Payload{Exchange: exchange, Timeframe: "hourly"})
			taskHourly := asynq.NewTask(tasks.TypeHourly, payloadHourly)
			// Schedule for 30 min from now so the task stays visible in "scheduled" until run time.
			nextHourly := now.Add(hourlyUnique)
			_, err = s.client.EnqueueContext(ctx, taskHourly, asynq.ProcessAt(nextHourly), asynq.Unique(hourlyUnique))
			if err != nil {
				if isUniqueConflict(err) {
					s.logger.Debug("schedule hourly skipped (already enqueued)", "exchange", exchange)
				} else {
					s.logger.Warn("schedule hourly failed", "exchange", exchange, "error", err)
				}
				continue
			}
			scheduledHourly++
			s.logger.Debug("task scheduled",
				"exchange", exchange,
				"timeframe", "hourly",
				"process_at", nextHourly.Format(time.RFC3339),
			)
		}
	}

	s.logger.Info("schedule cycle complete",
		"scheduled_daily", scheduledDaily,
		"scheduled_weekly", scheduledWeekly,
		"scheduled_hourly", scheduledHourly,
		"open_exchanges", openExchangeCount,
		"duration_ms", time.Since(start).Milliseconds(),
	)
	// When no exchanges are considered open, log why for NYSE to debug UTC vs local timing.
	if openExchangeCount == 0 && exchangeCount > 0 {
		s.logger.Info("hourly none open; NYSE diagnostic",
			"status", calendar.ExchangeOpenStatus(calendar.ExchangeNYSE, now),
		)
	}
}

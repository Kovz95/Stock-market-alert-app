package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"time"

	"github.com/hibiken/asynq"

	"stockalert/calendar"

	schedulerv1 "stockalert/gen/go/scheduler/v1"
)

const defaultQueue = "default"

// taskPayload matches apps/scheduler/internal/tasks.Payload.
type taskPayload struct {
	Exchange  string `json:"exchange"`
	Timeframe string `json:"timeframe"`
}

func taskName(timeframe string) string {
	switch timeframe {
	case "weekly":
		return "task:weekly"
	case "hourly":
		return "task:hourly"
	default:
		return "task:daily"
	}
}

func (s *Server) StartScheduler(ctx context.Context, _ *schedulerv1.StartSchedulerRequest) (*schedulerv1.StartSchedulerResponse, error) {
	s.logger.Info("StartScheduler: request received", "queue", defaultQueue)

	if s.inspector == nil {
		s.logger.Warn("StartScheduler: inspector not available")
		return &schedulerv1.StartSchedulerResponse{
			Success: false,
			Message: "inspector not available",
		}, nil
	}

	s.logger.Debug("StartScheduler: unpausing queue", "queue", defaultQueue)
	if err := s.inspector.UnpauseQueue(defaultQueue); err != nil {
		s.logger.Error("StartScheduler: failed to unpause queue",
			"queue", defaultQueue,
			"error", err,
		)
		return &schedulerv1.StartSchedulerResponse{
			Success: false,
			Message: fmt.Sprintf("failed to unpause queue: %v", err),
		}, nil
	}

	// Enqueue all scheduled tasks so the queue is populated immediately (worker also runs schedule every 15 min).
	if s.client != nil {
		n := enqueueScheduleAll(ctx, s.client, s.logger)
		s.logger.Info("StartScheduler: enqueued tasks on start", "count", n)
	}

	s.logger.Info("StartScheduler: queue successfully unpaused", "queue", defaultQueue)
	return &schedulerv1.StartSchedulerResponse{
		Success: true,
		Message: "Scheduler queue resumed",
	}, nil
}

func (s *Server) StopScheduler(ctx context.Context, _ *schedulerv1.StopSchedulerRequest) (*schedulerv1.StopSchedulerResponse, error) {
	s.logger.Info("StopScheduler: request received", "queue", defaultQueue)

	if s.inspector == nil {
		s.logger.Warn("StopScheduler: inspector not available")
		return &schedulerv1.StopSchedulerResponse{
			Success: false,
			Message: "inspector not available",
		}, nil
	}

	s.logger.Debug("StopScheduler: pausing queue", "queue", defaultQueue)
	if err := s.inspector.PauseQueue(defaultQueue); err != nil {
		s.logger.Error("StopScheduler: failed to pause queue",
			"queue", defaultQueue,
			"error", err,
		)
		return &schedulerv1.StopSchedulerResponse{
			Success: false,
			Message: fmt.Sprintf("failed to pause queue: %v", err),
		}, nil
	}

	total, err := purgeQueue(s.inspector, defaultQueue, s.logger)
	if err != nil {
		s.logger.Warn("StopScheduler: purge had errors (queue already paused)", "error", err, "deleted", total)
		// Queue is already paused; report success and that we cleared what we could
	}
	if total > 0 {
		s.logger.Info("StopScheduler: queue purged", "queue", defaultQueue, "tasks_removed", total)
	}

	return &schedulerv1.StopSchedulerResponse{
		Success: true,
		Message: "Scheduler queue paused and cleared",
	}, nil
}

// purgeQueue deletes all tasks in the queue (scheduled, pending, retry, archived, completed, active).
// Active tasks are canceled via CancelProcessing first (required by asynq); then bulk deletes run
// so any canceled tasks that land in archived are removed. Returns total number of tasks removed
// and any error from the last failing operation.
func purgeQueue(inspector *asynq.Inspector, queue string, logger *slog.Logger) (int, error) {
	var total int

	// Active first: asynq does not allow DeleteTask on active; must use CancelProcessing.
	// Canceled tasks may move to archived, which we delete in the next step.
	infos, err := inspector.ListActiveTasks(queue, asynq.PageSize(500))
	if err != nil {
		return total, fmt.Errorf("list active: %w", err)
	}
	for _, info := range infos {
		if err := inspector.CancelProcessing(info.ID); err != nil {
			logger.Debug("purgeQueue: cancel active task failed", "task_id", info.ID, "error", err)
			continue
		}
		total++
		logger.Debug("purgeQueue: canceled active task", "task_id", info.ID)
	}
	if len(infos) > 0 {
		logger.Info("purgeQueue: canceled active tasks", "count", len(infos))
	}

	// Bulk-deletable states (same order as apps/scheduler/cmd/resetqueue).
	// Run after canceling active so any tasks that moved to archived are removed.
	type bulkOp struct {
		name string
		del  func(string) (int, error)
	}
	for _, op := range []bulkOp{
		{"scheduled", inspector.DeleteAllScheduledTasks},
		{"pending", inspector.DeleteAllPendingTasks},
		{"retry", inspector.DeleteAllRetryTasks},
		{"archived", inspector.DeleteAllArchivedTasks},
		{"completed", inspector.DeleteAllCompletedTasks},
	} {
		n, err := op.del(queue)
		if err != nil {
			return total, fmt.Errorf("delete all %s: %w", op.name, err)
		}
		if n > 0 {
			logger.Debug("purgeQueue: deleted", "state", op.name, "count", n)
			total += n
		}
	}

	return total, nil
}

const (
	enqueueDailyUnique  = 12 * time.Hour
	enqueueHourlyUnique = 30 * time.Minute
)

func isUniqueConflict(err error) bool {
	return err != nil && strings.Contains(err.Error(), "task already exists")
}

// enqueueScheduleAll enqueues daily, weekly (if Friday), and hourly (if exchange open) tasks for all exchanges.
// Matches the worker's schedule logic so Start immediately repopulates the queue. Returns number enqueued.
func enqueueScheduleAll(ctx context.Context, client *asynq.Client, logger *slog.Logger) int {
	now := time.Now().UTC()
	exchanges := make([]string, 0, len(calendar.ExchangeSchedules))
	for exchange := range calendar.ExchangeSchedules {
		exchanges = append(exchanges, exchange)
	}
	sort.Strings(exchanges)
	var enqueued int
	for _, exchange := range exchanges {
		nextDaily := calendar.GetNextDailyRunTime(exchange, now)
		payload, _ := json.Marshal(taskPayload{Exchange: exchange, Timeframe: "daily"})
		task := asynq.NewTask(taskName("daily"), payload)
		_, err := client.EnqueueContext(ctx, task, asynq.ProcessAt(nextDaily), asynq.Unique(enqueueDailyUnique))
		if err != nil {
			if isUniqueConflict(err) {
				logger.Debug("enqueueScheduleAll: daily skipped (already enqueued)", "exchange", exchange)
			} else {
				logger.Warn("enqueueScheduleAll: daily failed", "exchange", exchange, "error", err)
			}
		} else {
			enqueued++
		}

		if nextDaily.Weekday() == time.Friday {
			payloadWeekly, _ := json.Marshal(taskPayload{Exchange: exchange, Timeframe: "weekly"})
			taskWeekly := asynq.NewTask(taskName("weekly"), payloadWeekly)
			_, err = client.EnqueueContext(ctx, taskWeekly, asynq.ProcessAt(nextDaily), asynq.Unique(enqueueDailyUnique))
			if err != nil {
				if isUniqueConflict(err) {
					logger.Debug("enqueueScheduleAll: weekly skipped (already enqueued)", "exchange", exchange)
				} else {
					logger.Warn("enqueueScheduleAll: weekly failed", "exchange", exchange, "error", err)
				}
			} else {
				enqueued++
			}
		}

		if calendar.IsExchangeOpen(exchange, now) {
			payloadHourly, _ := json.Marshal(taskPayload{Exchange: exchange, Timeframe: "hourly"})
			taskHourly := asynq.NewTask(taskName("hourly"), payloadHourly)
			nextHourly := now.Add(enqueueHourlyUnique)
			_, err = client.EnqueueContext(ctx, taskHourly, asynq.ProcessAt(nextHourly), asynq.Unique(enqueueHourlyUnique))
			if err != nil {
				if isUniqueConflict(err) {
					logger.Debug("enqueueScheduleAll: hourly skipped (already enqueued)", "exchange", exchange)
				} else {
					logger.Warn("enqueueScheduleAll: hourly failed", "exchange", exchange, "error", err)
				}
			} else {
				enqueued++
			}
		}
	}
	return enqueued
}

func (s *Server) RunExchangeJob(ctx context.Context, req *schedulerv1.RunExchangeJobRequest) (*schedulerv1.RunExchangeJobResponse, error) {
	timeframe := req.Timeframe
	if timeframe == "" {
		timeframe = "daily"
		s.logger.Debug("RunExchangeJob: timeframe defaulted", "timeframe", timeframe)
	}

	s.logger.Info("RunExchangeJob: request received",
		"exchange", req.Exchange,
		"timeframe", timeframe,
		"task_type", taskName(timeframe),
	)

	if s.client == nil {
		s.logger.Warn("RunExchangeJob: asynq client not available",
			"exchange", req.Exchange,
			"timeframe", timeframe,
		)
		return &schedulerv1.RunExchangeJobResponse{
			Success: false,
			Message: "asynq client not available",
		}, nil
	}

	payload, err := json.Marshal(taskPayload{
		Exchange:  req.Exchange,
		Timeframe: timeframe,
	})
	if err != nil {
		s.logger.Error("RunExchangeJob: failed to marshal task payload",
			"exchange", req.Exchange,
			"timeframe", timeframe,
			"error", err,
		)
		return nil, fmt.Errorf("marshal payload: %w", err)
	}

	task := asynq.NewTask(taskName(timeframe), payload)
	s.logger.Debug("RunExchangeJob: enqueueing task",
		"task_type", taskName(timeframe),
		"exchange", req.Exchange,
		"timeframe", timeframe,
		"uniqueness_ttl_minutes", 30,
	)

	enqueueStart := time.Now()
	info, err := s.client.Enqueue(task, asynq.Unique(30*time.Minute))
	if err != nil {
		s.logger.Error("RunExchangeJob: failed to enqueue task",
			"task_type", taskName(timeframe),
			"exchange", req.Exchange,
			"timeframe", timeframe,
			"error", err,
			"duration_ms", time.Since(enqueueStart).Milliseconds(),
		)
		return &schedulerv1.RunExchangeJobResponse{
			Success: false,
			Message: fmt.Sprintf("failed to enqueue: %v", err),
		}, nil
	}

	s.logger.Info("RunExchangeJob: task enqueued successfully",
		"task_id", info.ID,
		"task_type", info.Type,
		"exchange", req.Exchange,
		"timeframe", timeframe,
		"queue", info.Queue,
		"state", info.State.String(),
		"duration_ms", time.Since(enqueueStart).Milliseconds(),
	)

	return &schedulerv1.RunExchangeJobResponse{
		Success: true,
		Message: fmt.Sprintf("Enqueued %s/%s (task_id=%s)", req.Exchange, timeframe, info.ID),
	}, nil
}

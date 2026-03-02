package main

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/hibiken/asynq"

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

	s.logger.Info("StopScheduler: queue successfully paused", "queue", defaultQueue)
	return &schedulerv1.StopSchedulerResponse{
		Success: true,
		Message: "Scheduler queue paused",
	}, nil
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

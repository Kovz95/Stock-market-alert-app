package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
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
	if s.inspector == nil {
		return &schedulerv1.StartSchedulerResponse{
			Success: false,
			Message: "inspector not available",
		}, nil
	}

	if err := s.inspector.UnpauseQueue(defaultQueue); err != nil {
		log.Printf("start scheduler: unpause queue: %v", err)
		return &schedulerv1.StartSchedulerResponse{
			Success: false,
			Message: fmt.Sprintf("failed to unpause queue: %v", err),
		}, nil
	}

	return &schedulerv1.StartSchedulerResponse{
		Success: true,
		Message: "Scheduler queue resumed",
	}, nil
}

func (s *Server) StopScheduler(ctx context.Context, _ *schedulerv1.StopSchedulerRequest) (*schedulerv1.StopSchedulerResponse, error) {
	if s.inspector == nil {
		return &schedulerv1.StopSchedulerResponse{
			Success: false,
			Message: "inspector not available",
		}, nil
	}

	if err := s.inspector.PauseQueue(defaultQueue); err != nil {
		log.Printf("stop scheduler: pause queue: %v", err)
		return &schedulerv1.StopSchedulerResponse{
			Success: false,
			Message: fmt.Sprintf("failed to pause queue: %v", err),
		}, nil
	}

	return &schedulerv1.StopSchedulerResponse{
		Success: true,
		Message: "Scheduler queue paused",
	}, nil
}

func (s *Server) RunExchangeJob(ctx context.Context, req *schedulerv1.RunExchangeJobRequest) (*schedulerv1.RunExchangeJobResponse, error) {
	if s.client == nil {
		return &schedulerv1.RunExchangeJobResponse{
			Success: false,
			Message: "asynq client not available",
		}, nil
	}

	timeframe := req.Timeframe
	if timeframe == "" {
		timeframe = "daily"
	}

	payload, err := json.Marshal(taskPayload{
		Exchange:  req.Exchange,
		Timeframe: timeframe,
	})
	if err != nil {
		return nil, fmt.Errorf("marshal payload: %w", err)
	}

	task := asynq.NewTask(taskName(timeframe), payload)
	info, err := s.client.Enqueue(task, asynq.Unique(30*time.Minute))
	if err != nil {
		log.Printf("run exchange job: enqueue: %v", err)
		return &schedulerv1.RunExchangeJobResponse{
			Success: false,
			Message: fmt.Sprintf("failed to enqueue: %v", err),
		}, nil
	}

	return &schedulerv1.RunExchangeJobResponse{
		Success: true,
		Message: fmt.Sprintf("Enqueued %s/%s (task_id=%s)", req.Exchange, timeframe, info.ID),
	}, nil
}

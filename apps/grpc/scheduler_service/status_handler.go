package main

import (
	"context"
	"encoding/json"
	"time"

	"google.golang.org/protobuf/types/known/timestamppb"

	db "stockalert/database/generated"
	schedulerv1 "stockalert/gen/go/scheduler/v1"
)

// statusPayload mirrors the JSON stored in app_documents by the scheduler status manager.
type statusPayload struct {
	Status     string          `json:"status"`
	Heartbeat  string          `json:"heartbeat"`
	CurrentJob *statusJob      `json:"current_job,omitempty"`
	LastRun    *statusLastRun  `json:"last_run,omitempty"`
	LastError  *statusError    `json:"last_error,omitempty"`
}

type statusJob struct {
	Exchange  string `json:"exchange"`
	Timeframe string `json:"timeframe"`
	Started   string `json:"started"`
}

type statusLastRun struct {
	Exchange    string `json:"exchange"`
	Timeframe   string `json:"timeframe"`
	CompletedAt string `json:"completed_at"`
}

type statusError struct {
	Exchange  string `json:"exchange"`
	Timeframe string `json:"timeframe"`
	Message   string `json:"message"`
	Time      string `json:"time"`
}

func (s *Server) GetSchedulerStatus(ctx context.Context, _ *schedulerv1.GetSchedulerStatusRequest) (*schedulerv1.GetSchedulerStatusResponse, error) {
	queries := db.New(s.pool)

	doc, err := queries.GetAppDocument(ctx, "scheduler_status")
	if err != nil {
		// No document means scheduler has never run.
		return &schedulerv1.GetSchedulerStatusResponse{
			Status: "unknown",
		}, nil
	}

	var payload statusPayload
	if err := json.Unmarshal(doc.Payload, &payload); err != nil {
		return nil, err
	}

	resp := &schedulerv1.GetSchedulerStatusResponse{
		Status: payload.Status,
	}

	if t, err := time.Parse(time.RFC3339, payload.Heartbeat); err == nil {
		resp.Heartbeat = timestamppb.New(t)
	}

	if j := payload.CurrentJob; j != nil {
		resp.CurrentJob = &schedulerv1.CurrentJob{
			Exchange:  j.Exchange,
			Timeframe: j.Timeframe,
		}
		if t, err := time.Parse(time.RFC3339, j.Started); err == nil {
			resp.CurrentJob.Started = timestamppb.New(t)
		}
	}

	if lr := payload.LastRun; lr != nil {
		resp.LastRun = &schedulerv1.LastRun{
			Exchange:  lr.Exchange,
			Timeframe: lr.Timeframe,
		}
		if t, err := time.Parse(time.RFC3339, lr.CompletedAt); err == nil {
			resp.LastRun.CompletedAt = timestamppb.New(t)
		}
	}

	if le := payload.LastError; le != nil {
		resp.LastError = &schedulerv1.LastError{
			Exchange:  le.Exchange,
			Timeframe: le.Timeframe,
			Message:   le.Message,
		}
		if t, err := time.Parse(time.RFC3339, le.Time); err == nil {
			resp.LastError.Time = timestamppb.New(t)
		}
	}

	// Asynq queue info
	if s.inspector != nil {
		if info, err := s.inspector.GetQueueInfo("default"); err == nil {
			resp.QueueSize = int32(info.Size)
			resp.ActiveWorkers = int32(info.Active)
		}
		// Count live worker processes connected to Redis across all containers.
		if servers, err := s.inspector.Servers(); err == nil {
			resp.WorkerProcesses = int32(len(servers))
		}
	}

	return resp, nil
}

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
	Status     string         `json:"status"`
	Heartbeat  string         `json:"heartbeat"`
	CurrentJob *statusJob     `json:"current_job,omitempty"`
	LastRun    *statusLastRun `json:"last_run,omitempty"`
	LastError  *statusError   `json:"last_error,omitempty"`
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
	s.logger.Debug("GetSchedulerStatus: request received")

	queries := db.New(s.pool)

	s.logger.Debug("GetSchedulerStatus: fetching app_document", "key", "scheduler_status")
	doc, err := queries.GetAppDocument(ctx, "scheduler_status")
	if err != nil {
		s.logger.Warn("GetSchedulerStatus: scheduler_status document not found, returning unknown",
			"error", err,
		)
		return &schedulerv1.GetSchedulerStatusResponse{
			Status: "unknown",
		}, nil
	}
	s.logger.Debug("GetSchedulerStatus: document fetched", "payload_bytes", len(doc.Payload))

	var payload statusPayload
	if err := json.Unmarshal(doc.Payload, &payload); err != nil {
		s.logger.Error("GetSchedulerStatus: failed to unmarshal status payload", "error", err)
		return nil, err
	}
	s.logger.Info("GetSchedulerStatus: status parsed",
		"status", payload.Status,
		"heartbeat", payload.Heartbeat,
		"has_current_job", payload.CurrentJob != nil,
		"has_last_run", payload.LastRun != nil,
		"has_last_error", payload.LastError != nil,
	)

	resp := &schedulerv1.GetSchedulerStatusResponse{
		Status: payload.Status,
	}

	if t, err := time.Parse(time.RFC3339, payload.Heartbeat); err == nil {
		resp.Heartbeat = timestamppb.New(t)
		heartbeatAge := time.Since(t)
		s.logger.Debug("GetSchedulerStatus: heartbeat parsed",
			"heartbeat", t.UTC().Format(time.RFC3339),
			"age_seconds", int(heartbeatAge.Seconds()),
		)
	} else if payload.Heartbeat != "" {
		s.logger.Warn("GetSchedulerStatus: failed to parse heartbeat", "raw", payload.Heartbeat, "error", err)
	}

	if j := payload.CurrentJob; j != nil {
		s.logger.Info("GetSchedulerStatus: active job detected",
			"exchange", j.Exchange,
			"timeframe", j.Timeframe,
			"started", j.Started,
		)
		resp.CurrentJob = &schedulerv1.CurrentJob{
			Exchange:  j.Exchange,
			Timeframe: j.Timeframe,
		}
		if t, err := time.Parse(time.RFC3339, j.Started); err == nil {
			resp.CurrentJob.Started = timestamppb.New(t)
			s.logger.Debug("GetSchedulerStatus: current job running duration",
				"duration_seconds", int(time.Since(t).Seconds()),
			)
		} else {
			s.logger.Warn("GetSchedulerStatus: failed to parse current job start time",
				"raw", j.Started, "error", err,
			)
		}
	}

	if lr := payload.LastRun; lr != nil {
		s.logger.Debug("GetSchedulerStatus: last run info",
			"exchange", lr.Exchange,
			"timeframe", lr.Timeframe,
			"completed_at", lr.CompletedAt,
		)
		resp.LastRun = &schedulerv1.LastRun{
			Exchange:  lr.Exchange,
			Timeframe: lr.Timeframe,
		}
		if t, err := time.Parse(time.RFC3339, lr.CompletedAt); err == nil {
			resp.LastRun.CompletedAt = timestamppb.New(t)
		} else {
			s.logger.Warn("GetSchedulerStatus: failed to parse last run completed_at",
				"raw", lr.CompletedAt, "error", err,
			)
		}
	}

	if le := payload.LastError; le != nil {
		s.logger.Warn("GetSchedulerStatus: last error present",
			"exchange", le.Exchange,
			"timeframe", le.Timeframe,
			"message", le.Message,
			"time", le.Time,
		)
		resp.LastError = &schedulerv1.LastError{
			Exchange:  le.Exchange,
			Timeframe: le.Timeframe,
			Message:   le.Message,
		}
		if t, err := time.Parse(time.RFC3339, le.Time); err == nil {
			resp.LastError.Time = timestamppb.New(t)
		} else {
			s.logger.Warn("GetSchedulerStatus: failed to parse last error time",
				"raw", le.Time, "error", err,
			)
		}
	}

	if s.inspector != nil {
		s.logger.Debug("GetSchedulerStatus: fetching asynq queue info", "queue", "default")
		if info, err := s.inspector.GetQueueInfo("default"); err == nil {
			resp.QueueSize = int32(info.Size)
			resp.ActiveWorkers = int32(info.Active)
			s.logger.Info("GetSchedulerStatus: queue info fetched",
				"queue", "default",
				"size", info.Size,
				"active", info.Active,
				"pending", info.Pending,
				"retry", info.Retry,
				"archived", info.Archived,
				"completed", info.Completed,
				"paused", info.Paused,
			)
		} else {
			s.logger.Warn("GetSchedulerStatus: failed to get queue info", "queue", "default", "error", err)
		}

		s.logger.Debug("GetSchedulerStatus: fetching asynq server list")
		if servers, err := s.inspector.Servers(); err == nil {
			resp.WorkerProcesses = int32(len(servers))
			s.logger.Info("GetSchedulerStatus: worker processes counted",
				"worker_processes", len(servers),
			)
			for i, srv := range servers {
				s.logger.Debug("GetSchedulerStatus: worker server detail",
					"index", i,
					"host", srv.Host,
					"pid", srv.PID,
					"status", srv.Status,
					"concurrency", srv.Concurrency,
					"active_workers", srv.ActiveWorkers,
				)
			}
		} else {
			s.logger.Warn("GetSchedulerStatus: failed to list asynq servers", "error", err)
		}
	} else {
		s.logger.Debug("GetSchedulerStatus: inspector not available, skipping queue info")
	}

	s.logger.Debug("GetSchedulerStatus: response ready",
		"status", resp.Status,
		"queue_size", resp.QueueSize,
		"active_workers", resp.ActiveWorkers,
		"worker_processes", resp.WorkerProcesses,
	)

	return resp, nil
}

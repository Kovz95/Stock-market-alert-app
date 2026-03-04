package main

import (
	"context"
	"encoding/json"

	"github.com/hibiken/asynq"
	"google.golang.org/protobuf/types/known/timestamppb"

	schedulerv1 "stockalert/gen/go/scheduler/v1"
)

// payloadFields matches the JSON payload enqueued by the Go scheduler (exchange + timeframe).
type payloadFields struct {
	Exchange  string `json:"exchange"`
	Timeframe string `json:"timeframe"`
}

func (s *Server) ListQueueTasks(ctx context.Context, req *schedulerv1.ListQueueTasksRequest) (*schedulerv1.ListQueueTasksResponse, error) {
	queue := req.Queue
	if queue == "" {
		queue = defaultQueue
	}
	s.logger.Debug("ListQueueTasks: request received", "queue", queue)

	var out []*schedulerv1.QueueTask

	opts := []asynq.ListOption{}
	opts = append(opts, asynq.PageSize(100))

	for _, state := range []string{"scheduled", "pending", "active"} {
		var infos []*asynq.TaskInfo
		var err error
		switch state {
		case "scheduled":
			infos, err = s.inspector.ListScheduledTasks(queue, opts...)
		case "pending":
			infos, err = s.inspector.ListPendingTasks(queue, opts...)
		case "active":
			infos, err = s.inspector.ListActiveTasks(queue, opts...)
		default:
			continue
		}
		if err != nil {
			s.logger.Warn("ListQueueTasks: list failed", "queue", queue, "state", state, "error", err)
			continue
		}
		for _, info := range infos {
			exchange, timeframe := decodePayload(info.Payload)
			t := &schedulerv1.QueueTask{
				State:     state,
				Type:      info.Type,
				Exchange:  exchange,
				Timeframe: timeframe,
				Id:        info.ID,
			}
			if !info.NextProcessAt.IsZero() {
				t.NextProcessAt = timestamppb.New(info.NextProcessAt)
			}
			out = append(out, t)
		}
	}

	s.logger.Info("ListQueueTasks: returning tasks", "queue", queue, "count", len(out))
	return &schedulerv1.ListQueueTasksResponse{Tasks: out}, nil
}

func decodePayload(payload []byte) (exchange, timeframe string) {
	var p payloadFields
	if err := json.Unmarshal(payload, &p); err != nil {
		return "-", "-"
	}
	return p.Exchange, p.Timeframe
}

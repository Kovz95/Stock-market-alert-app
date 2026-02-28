package main

import (
	"github.com/hibiken/asynq"
	"github.com/jackc/pgx/v5/pgxpool"

	schedulerv1 "stockalert/gen/go/scheduler/v1"
)

// Server implements the SchedulerService gRPC server.
type Server struct {
	schedulerv1.UnimplementedSchedulerServiceServer
	pool      *pgxpool.Pool
	client    *asynq.Client
	inspector *asynq.Inspector
}

// NewServer creates a new SchedulerService server.
func NewServer(pool *pgxpool.Pool, client *asynq.Client, inspector *asynq.Inspector) *Server {
	return &Server{
		pool:      pool,
		client:    client,
		inspector: inspector,
	}
}

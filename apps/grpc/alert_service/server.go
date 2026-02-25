package main

import (
	"github.com/jackc/pgx/v5/pgxpool"

	alertv1 "stockalert/gen/go/alert/v1"
)

// Server implements the AlertService gRPC server.
type Server struct {
	alertv1.UnimplementedAlertServiceServer
	pool *pgxpool.Pool
}

// NewServer creates a new AlertService server.
func NewServer(pool *pgxpool.Pool) *Server {
	return &Server{pool: pool}
}

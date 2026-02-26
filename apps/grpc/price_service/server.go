package main

import (
	"github.com/jackc/pgx/v5/pgxpool"

	pricev1 "stockalert/gen/go/price/v1"
)

// Server implements the PriceService gRPC server.
type Server struct {
	pricev1.UnimplementedPriceServiceServer
	pool *pgxpool.Pool
}

// NewServer creates a new PriceService server.
func NewServer(pool *pgxpool.Pool) *Server {
	return &Server{pool: pool}
}

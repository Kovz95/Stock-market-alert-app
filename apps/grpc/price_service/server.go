package main

import (
	"log/slog"

	"github.com/jackc/pgx/v5/pgxpool"

	pricev1 "stockalert/gen/go/price/v1"
)

// Server implements the PriceService gRPC server.
type Server struct {
	pricev1.UnimplementedPriceServiceServer
	pool    *pgxpool.Pool
	updater *priceUpdater
	logger  *slog.Logger
}

// NewServer creates a new PriceService server.
func NewServer(pool *pgxpool.Pool, updater *priceUpdater, logger *slog.Logger) *Server {
	if logger == nil {
		logger = slog.Default()
	}
	return &Server{pool: pool, updater: updater, logger: logger}
}

package main

import (
	"github.com/jackc/pgx/v5/pgxpool"

	"stockalert/alert"
	"stockalert/discord"
	alertv1 "stockalert/gen/go/alert/v1"
)

// Server implements the AlertService gRPC server.
type Server struct {
	alertv1.UnimplementedAlertServiceServer
	pool    *pgxpool.Pool
	checker *alert.Checker
	router  *discord.Router
	accum   *discord.Accumulator
	updater *priceUpdater
}

// NewServer creates a new AlertService server.
func NewServer(
	pool *pgxpool.Pool,
	checker *alert.Checker,
	router *discord.Router,
	accum *discord.Accumulator,
	updater *priceUpdater,
) *Server {
	return &Server{
		pool:    pool,
		checker: checker,
		router:  router,
		accum:   accum,
		updater: updater,
	}
}

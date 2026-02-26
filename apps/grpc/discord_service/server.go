package main

import (
	"github.com/jackc/pgx/v5/pgxpool"
	discordv1 "stockalert/gen/go/discord/v1"
)

// Server implements DiscordConfigService.
type Server struct {
	discordv1.UnimplementedDiscordConfigServiceServer
	pool *pgxpool.Pool
}

// NewServer creates a new DiscordConfigService server.
func NewServer(pool *pgxpool.Pool) *Server {
	return &Server{pool: pool}
}

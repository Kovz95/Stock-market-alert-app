package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"

	"github.com/jackc/pgx/v5/pgxpool"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"

	"stockalert/alert"
	db "stockalert/database/generated"
	"stockalert/discord"
	alertv1 "stockalert/gen/go/alert/v1"
	"stockalert/indicator"
)

func main() {
	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		log.Fatal("DATABASE_URL is required")
	}

	port := os.Getenv("PORT")
	if port == "" {
		port = "50051"
	}

	fmpAPIKey := os.Getenv("FMP_API_KEY")
	if fmpAPIKey == "" {
		log.Println("warning: FMP_API_KEY not set — EvaluateExchange price updates will fail")
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	pool, err := pgxpool.New(ctx, dbURL)
	if err != nil {
		log.Fatalf("failed to create connection pool: %v", err)
	}
	defer pool.Close()

	if err := pool.Ping(ctx); err != nil {
		log.Fatalf("failed to ping database: %v", err)
	}
	log.Println("connected to database")

	queries := db.New(pool)

	// Alert evaluation dependencies
	registry := indicator.NewDefaultRegistry()
	checker := alert.NewChecker(queries, registry)

	router, err := discord.NewRouter(ctx, queries)
	if err != nil {
		log.Printf("discord router (using defaults): %v", err)
	}
	notifier := discord.NewNotifier()
	accum := discord.NewAccumulator(notifier)

	// Price updater (nil-safe: EvaluateExchange returns an error if apiKey is missing)
	var updater *priceUpdater
	if fmpAPIKey != "" {
		fmp := newFMPClient(fmpAPIKey)
		updater = newPriceUpdater(pool, queries, fmp)
	}

	grpcServer := grpc.NewServer()
	alertv1.RegisterAlertServiceServer(grpcServer, NewServer(pool, checker, router, accum, updater))

	healthServer := health.NewServer()
	healthpb.RegisterHealthServer(grpcServer, healthServer)
	healthServer.SetServingStatus("", healthpb.HealthCheckResponse_SERVING)

	reflection.Register(grpcServer)

	lis, err := net.Listen("tcp", fmt.Sprintf(":%s", port))
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		sig := <-sigCh
		log.Printf("received signal %v, shutting down", sig)
		healthServer.SetServingStatus("", healthpb.HealthCheckResponse_NOT_SERVING)
		grpcServer.GracefulStop()
	}()

	log.Printf("alert_service listening on :%s", port)
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}

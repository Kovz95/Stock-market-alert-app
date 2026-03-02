package main

import (
	"context"
	"fmt"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"
	_ "time/tzdata" // embed IANA timezone database for distroless/scratch containers

	"github.com/hibiken/asynq"
	"github.com/jackc/pgx/v5/pgxpool"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"
	"google.golang.org/grpc/status"

	schedulerv1 "stockalert/gen/go/scheduler/v1"
)

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelDebug,
	}))
	slog.SetDefault(logger)

	logger.Info("scheduler_service starting")

	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		logger.Error("DATABASE_URL is required")
		os.Exit(1)
	}

	redisAddr := os.Getenv("REDIS_ADDR")
	if redisAddr == "" {
		redisAddr = "localhost:6379"
	}
	logger.Debug("redis address resolved", "addr", redisAddr)

	port := os.Getenv("PORT")
	if port == "" {
		port = "50051"
	}
	logger.Debug("gRPC port resolved", "port", port)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	logger.Info("connecting to database")
	pool, err := pgxpool.New(ctx, dbURL)
	if err != nil {
		logger.Error("failed to create connection pool", "error", err)
		os.Exit(1)
	}
	defer pool.Close()

	if err := pool.Ping(ctx); err != nil {
		logger.Error("failed to ping database", "error", err)
		os.Exit(1)
	}
	logger.Info("database connection established",
		"max_conns", pool.Config().MaxConns,
		"min_conns", pool.Config().MinConns,
	)

	logger.Info("connecting to Redis", "addr", redisAddr)
	redisOpt := asynq.RedisClientOpt{Addr: redisAddr}
	asynqClient := asynq.NewClient(redisOpt)
	defer asynqClient.Close()
	logger.Info("asynq client created")

	inspector := asynq.NewInspector(redisOpt)
	defer inspector.Close()
	logger.Info("asynq inspector created")

	grpcServer := grpc.NewServer(
		grpc.UnaryInterceptor(unaryLoggingInterceptor(logger)),
	)
	schedulerv1.RegisterSchedulerServiceServer(grpcServer, NewServer(pool, asynqClient, inspector, logger))
	logger.Info("SchedulerService registered")

	healthServer := health.NewServer()
	healthpb.RegisterHealthServer(grpcServer, healthServer)
	healthServer.SetServingStatus("", healthpb.HealthCheckResponse_SERVING)
	logger.Info("health server registered", "status", "SERVING")

	reflection.Register(grpcServer)
	logger.Debug("gRPC server reflection enabled")

	lis, err := net.Listen("tcp", fmt.Sprintf(":%s", port))
	if err != nil {
		logger.Error("failed to listen", "port", port, "error", err)
		os.Exit(1)
	}

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		sig := <-sigCh
		logger.Info("shutdown signal received, stopping gracefully", "signal", sig.String())
		healthServer.SetServingStatus("", healthpb.HealthCheckResponse_NOT_SERVING)
		logger.Debug("health status set to NOT_SERVING")
		grpcServer.GracefulStop()
		logger.Info("gRPC server stopped")
	}()

	logger.Info("scheduler_service ready", "port", port)
	if err := grpcServer.Serve(lis); err != nil {
		logger.Error("gRPC serve error", "error", err)
		os.Exit(1)
	}
}

// unaryLoggingInterceptor logs the method, duration, and result code for every unary RPC.
func unaryLoggingInterceptor(logger *slog.Logger) grpc.UnaryServerInterceptor {
	return func(
		ctx context.Context,
		req any,
		info *grpc.UnaryServerInfo,
		handler grpc.UnaryHandler,
	) (any, error) {
		start := time.Now()
		logger.Debug("gRPC request received", "method", info.FullMethod)

		resp, err := handler(ctx, req)

		dur := time.Since(start)
		code := codes.OK
		if err != nil {
			code = status.Code(err)
		}

		logFn := logger.Info
		if err != nil {
			logFn = logger.Warn
		}
		logFn("gRPC request completed",
			"method", info.FullMethod,
			"code", code.String(),
			"duration_ms", dur.Milliseconds(),
			"error", err,
		)
		return resp, err
	}
}

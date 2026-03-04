// Package main implements the Asynq worker that processes scheduled
// price update and alert check tasks.
package main

import (
	"context"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"
	_ "time/tzdata" // embed IANA timezone database for distroless/scratch containers

	"github.com/hibiken/asynq"
	"github.com/jackc/pgx/v5/pgxpool"

	"stockalert/alert"
	"stockalert/discord"
	"stockalert/indicator"
	db "stockalert/database/generated"

	"stockalert/apps/scheduler/internal/config"
	"stockalert/apps/scheduler/internal/handler"
	"stockalert/apps/scheduler/internal/price"
	"stockalert/apps/scheduler/internal/schedule"
	"stockalert/apps/scheduler/internal/status"
	"stockalert/apps/scheduler/internal/tasks"
)

const (
	dbConnectRetries = 30
	dbConnectBackoff = 2 * time.Second
)

// maskSecret masks a secret string, showing only the first 4 chars.
func maskSecret(s string) string {
	if len(s) <= 4 {
		return "****"
	}
	return s[:4] + "****"
}

func main() {
	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelDebug}))
	slog.SetDefault(logger)

	cfg := config.Load()
	if cfg.DatabaseURL == "" {
		logger.Error("DATABASE_URL is required")
		os.Exit(1)
	}

	logger.Info("worker initializing",
		"redis_addr", cfg.RedisAddr,
		"database_url", maskSecret(cfg.DatabaseURL),
		"fmp_api_key", maskSecret(cfg.FMPAPIKey),
		"job_timeout", cfg.JobTimeout().String(),
		"shadow_mode", cfg.ShadowMode,
		"fmp_daily_concurrency", cfg.FMPDailyConcurrency,
		"fmp_weekly_concurrency", cfg.FMPWeeklyConcurrency,
		"fmp_hourly_concurrency", cfg.FMPHourlyConcurrency,
		"discord_webhook_daily_set", cfg.DiscordWebhookDaily != "",
		"discord_webhook_weekly_set", cfg.DiscordWebhookWeekly != "",
		"discord_webhook_hourly_set", cfg.DiscordWebhookHourly != "",
	)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var pool *pgxpool.Pool
	for attempt := 1; attempt <= dbConnectRetries; attempt++ {
		var err error
		pool, err = pgxpool.New(ctx, cfg.DatabaseURL)
		if err != nil {
			logger.Warn("database connect failed",
				"attempt", attempt,
				"max_attempts", dbConnectRetries,
				"error", err,
			)
			if attempt == dbConnectRetries {
				logger.Error("database connection exhausted retries", "error", err)
				os.Exit(1)
			}
			time.Sleep(dbConnectBackoff)
			continue
		}
		if err := pool.Ping(ctx); err != nil {
			pool.Close()
			logger.Warn("database ping failed",
				"attempt", attempt,
				"max_attempts", dbConnectRetries,
				"error", err,
			)
			if attempt == dbConnectRetries {
				logger.Error("database ping exhausted retries", "error", err)
				os.Exit(1)
			}
			time.Sleep(dbConnectBackoff)
			continue
		}
		break
	}
	defer pool.Close()

	poolStats := pool.Stat()
	logger.Info("database connected",
		"max_conns", poolStats.MaxConns(),
		"total_conns", poolStats.TotalConns(),
	)

	queries := db.New(pool)

	// Router loads config from DB; create with background load
	router, err := discord.NewRouter(ctx, queries)
	if err != nil {
		logger.Warn("discord router init error, using defaults", "error", err)
	}

	registry := indicator.NewDefaultRegistry()
	checker := alert.NewChecker(queries, registry)
	notifier := discord.NewNotifier()
	accum := discord.NewAccumulator(notifier)
	fmpClient := price.NewFMPClient(cfg.FMPAPIKey)
	updater := price.NewUpdater(queries, fmpClient, logger, cfg.FMPDailyConcurrency, cfg.FMPWeeklyConcurrency, cfg.FMPHourlyConcurrency)
	statusMgr := status.NewManager(queries, logger)

	common := &handler.Common{
		Queries:       queries,
		Checker:       checker,
		Router:        router,
		Accum:         accum,
		Notifier:      notifier,
		Updater:       updater,
		Status:        statusMgr,
		Logger:        logger,
		JobTimeout:    cfg.JobTimeout(),
		WebhookDaily:  cfg.DiscordWebhookDaily,
		WebhookWeekly: cfg.DiscordWebhookWeekly,
		WebhookHourly: cfg.DiscordWebhookHourly,
		ShadowDir:     "",
	}
	if cfg.ShadowMode {
		common.ShadowDir = cfg.ShadowOutputDir
		logger.Info("shadow mode enabled", "output_dir", common.ShadowDir)
	}

	asynqClient := asynq.NewClient(asynq.RedisClientOpt{Addr: cfg.RedisAddr})
	defer asynqClient.Close()

	concurrency := cfg.Concurrency
	if concurrency < 1 {
		concurrency = 1
	}
	logger.Info("worker concurrency", "concurrency", concurrency)
	srv := asynq.NewServer(
		asynq.RedisClientOpt{Addr: cfg.RedisAddr},
		asynq.Config{
			Concurrency: concurrency,
			LogLevel:    asynq.InfoLevel,
		},
	)

	mux := asynq.NewServeMux()
	mux.Handle(tasks.TypeDaily, handler.NewDailyHandler(common))
	mux.Handle(tasks.TypeWeekly, handler.NewWeeklyHandler(common))
	mux.Handle(tasks.TypeHourly, handler.NewHourlyHandler(common))
	logger.Info("registered handlers", "task_types", []string{"daily", "weekly", "hourly"})

	go func() {
		if err := srv.Run(mux); err != nil {
			logger.Error("asynq server error", "error", err)
		}
	}()

	sched := schedule.New(asynqClient, logger, cfg.FMPAPIKey)
	sched.Start(ctx)

	jobTypes := []string{"daily", "weekly", "hourly"}
	for _, jt := range jobTypes {
		logger.Info("worker starting", "job_type", jt)
		common.NotifyWorkerLifecycle(jt, "start")
	}

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	received := <-sig

	logger.Info("shutting down worker", "signal", received.String())
	sched.Stop()
	for _, jt := range jobTypes {
		common.NotifyWorkerLifecycle(jt, "stop")
	}
	srv.Shutdown()
}

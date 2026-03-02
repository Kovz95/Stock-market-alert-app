// Package main implements the Asynq worker that processes scheduled
// price update and alert check tasks.
package main

import (
	"context"
	"log"
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
	"stockalert/apps/scheduler/internal/status"
	"stockalert/apps/scheduler/internal/tasks"
)

const (
	dbConnectRetries = 30
	dbConnectBackoff = 2 * time.Second
)

func main() {
	cfg := config.Load()
	if cfg.DatabaseURL == "" {
		log.Fatal("DATABASE_URL is required")
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	var pool *pgxpool.Pool
	for attempt := 1; attempt <= dbConnectRetries; attempt++ {
		var err error
		pool, err = pgxpool.New(ctx, cfg.DatabaseURL)
		if err != nil {
			log.Printf("database connect (attempt %d/%d): %v", attempt, dbConnectRetries, err)
			if attempt == dbConnectRetries {
				log.Fatalf("database: %v", err)
			}
			time.Sleep(dbConnectBackoff)
			continue
		}
		if err := pool.Ping(ctx); err != nil {
			pool.Close()
			log.Printf("database ping (attempt %d/%d): %v", attempt, dbConnectRetries, err)
			if attempt == dbConnectRetries {
				log.Fatalf("database ping: %v", err)
			}
			time.Sleep(dbConnectBackoff)
			continue
		}
		break
	}
	defer pool.Close()

	queries := db.New(pool)

	// Router loads config from DB; create with background load
	router, err := discord.NewRouter(ctx, queries)
	if err != nil {
		log.Printf("discord router (using defaults): %v", err)
	}

	registry := indicator.NewDefaultRegistry()
	checker := alert.NewChecker(queries, registry)
	notifier := discord.NewNotifier()
	accum := discord.NewAccumulator(notifier)
	fmpClient := price.NewFMPClient(cfg.FMPAPIKey)
	updater := price.NewUpdater(queries, fmpClient)
	statusMgr := status.NewManager(queries)

	common := &handler.Common{
		Queries:       queries,
		Checker:       checker,
		Router:        router,
		Accum:         accum,
		Notifier:      notifier,
		Updater:       updater,
		Status:        statusMgr,
		JobTimeout:    cfg.JobTimeout(),
		WebhookDaily:  cfg.DiscordWebhookDaily,
		WebhookWeekly: cfg.DiscordWebhookWeekly,
		WebhookHourly: cfg.DiscordWebhookHourly,
		ShadowDir:     "",
	}
	if cfg.ShadowMode {
		common.ShadowDir = cfg.ShadowOutputDir
		log.Printf("shadow mode enabled: writing results to %s", common.ShadowDir)
	}

	asynqClient := asynq.NewClient(asynq.RedisClientOpt{Addr: cfg.RedisAddr})
	defer asynqClient.Close()

	srv := asynq.NewServer(
		asynq.RedisClientOpt{Addr: cfg.RedisAddr},
		asynq.Config{
			Concurrency: 4,
			LogLevel:    asynq.InfoLevel,
		},
	)

	mux := asynq.NewServeMux()
	var handlers int
	var jobTypes []string
	if cfg.HandlesTaskType("daily") {
		mux.Handle(tasks.TypeDaily, handler.NewDailyHandler(common))
		jobTypes = append(jobTypes, "daily")
		handlers++
	}
	if cfg.HandlesTaskType("weekly") {
		mux.Handle(tasks.TypeWeekly, handler.NewWeeklyHandler(common))
		jobTypes = append(jobTypes, "weekly")
		handlers++
	}
	if cfg.HandlesTaskType("hourly") {
		mux.Handle(tasks.TypeHourly, handler.NewHourlyHandler(common))
		jobTypes = append(jobTypes, "hourly")
		handlers++
	}
	if cfg.HandlesTaskType("enqueue") {
		mux.Handle(tasks.EnqueueTaskType, handler.NewEnqueueHandler(asynqClient))
		handlers++
	}
	if handlers == 0 {
		log.Fatal("WORKER_TASK_TYPES did not register any handlers; use daily, weekly, hourly, and/or enqueue")
	}

	go func() {
		if err := srv.Run(mux); err != nil {
			log.Printf("asynq server: %v", err)
		}
	}()

	for _, jt := range jobTypes {
		log.Printf("%s worker starting", jt)
		common.NotifyWorkerLifecycle(jt, "start")
	}

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig

	for _, jt := range jobTypes {
		log.Printf("%s worker stopping", jt)
		common.NotifyWorkerLifecycle(jt, "stop")
	}

	log.Println("shutting down worker")
	srv.Shutdown()
}

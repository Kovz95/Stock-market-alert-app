// Package main implements the Asynq cron scheduler that registers
// periodic enqueue tasks; the worker then enqueues per-exchange jobs.
package main

import (
	"log"
	"os"
	"os/signal"
	"syscall"

	"github.com/hibiken/asynq"

	"stockalert/apps/scheduler/internal/config"
	"stockalert/apps/scheduler/internal/tasks"
)

func main() {
	cfg := config.Load()
	if cfg.RedisAddr == "" {
		cfg.RedisAddr = "localhost:6379"
	}

	scheduler := asynq.NewScheduler(
		asynq.RedisClientOpt{Addr: cfg.RedisAddr},
		&asynq.SchedulerOpts{
			LogLevel: asynq.InfoLevel,
		},
	)

	// Run enqueue task every 15 minutes; it enqueues task:daily, task:weekly, task:hourly
	// for exchanges that are due (using calendar.GetNextDailyRunTime and IsExchangeOpen).
	_, err := scheduler.Register("*/15 * * * *", asynq.NewTask(tasks.EnqueueTaskType, nil))
	if err != nil {
		log.Fatalf("register cron: %v", err)
	}

	go func() {
		log.Println("scheduler starting")
		if err := scheduler.Run(); err != nil {
			log.Printf("scheduler: %v", err)
		}
	}()

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)
	<-sig
	log.Println("shutting down scheduler")
	scheduler.Shutdown()
}

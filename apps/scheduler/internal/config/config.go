package config

import (
	"os"
	"strconv"
	"time"
)

// Config holds env-based configuration for the worker.
type Config struct {
	DatabaseURL string
	RedisAddr   string
	FMPAPIKey   string

	// Discord webhooks (optional). If set, status notifications are sent.
	DiscordWebhookDaily  string
	DiscordWebhookWeekly string
	DiscordWebhookHourly string

	JobTimeoutSec int

	// Concurrency is the number of tasks this worker process runs in parallel (Asynq concurrency).
	// Default 1 = one job at a time per process; increase for parallel exchange jobs.
	Concurrency int

	// FMPHourlyConcurrency is how many FMP API calls run in parallel during hourly price update.
	// Higher values speed up large exchanges (e.g. NASDAQ) but must stay within FMP rate limits.
	FMPHourlyConcurrency int

	// ShadowMode: when true, write alert trigger results to ShadowOutputDir for comparison with Python.
	ShadowMode      bool
	ShadowOutputDir string
}

// Load reads configuration from environment variables.
func Load() *Config {
	c := &Config{
		DatabaseURL: os.Getenv("DATABASE_URL"),
		RedisAddr:   os.Getenv("REDIS_ADDR"),
		FMPAPIKey:   os.Getenv("FMP_API_KEY"),

		DiscordWebhookDaily:  os.Getenv("DISCORD_WEBHOOK_DAILY"),
		DiscordWebhookWeekly: os.Getenv("DISCORD_WEBHOOK_WEEKLY"),
		DiscordWebhookHourly: os.Getenv("DISCORD_WEBHOOK_HOURLY"),

		JobTimeoutSec:         900,
		Concurrency:            1,  // one job at a time per process by default
		FMPHourlyConcurrency:   25, // parallel FMP fetches for hourly (stay under API rate limit)

		ShadowMode:      os.Getenv("SCHEDULER_SHADOW_MODE") == "true" || os.Getenv("SCHEDULER_SHADOW_MODE") == "1",
		ShadowOutputDir: os.Getenv("SCHEDULER_SHADOW_OUTPUT_DIR"),
	}
	if c.ShadowOutputDir == "" {
		c.ShadowOutputDir = "shadow_results"
	}
	if v := os.Getenv("SCHEDULER_JOB_TIMEOUT"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 60 {
			c.JobTimeoutSec = n
		}
	}
	if v := os.Getenv("SCHEDULER_CONCURRENCY"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 {
			c.Concurrency = n
		}
	}
	if v := os.Getenv("SCHEDULER_FMP_HOURLY_CONCURRENCY"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n >= 1 {
			c.FMPHourlyConcurrency = n
		}
	}
	if c.JobTimeoutSec == 0 {
		c.JobTimeoutSec = 900
	}
	if c.RedisAddr == "" {
		c.RedisAddr = "localhost:6379"
	}
	return c
}

// JobTimeout returns the job timeout as a duration.
func (c *Config) JobTimeout() time.Duration {
	return time.Duration(c.JobTimeoutSec) * time.Second
}

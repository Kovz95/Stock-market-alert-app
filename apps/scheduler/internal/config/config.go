package config

import (
	"os"
	"strconv"
	"strings"
	"time"
)

// Config holds env-based configuration for the scheduler and worker.
type Config struct {
	DatabaseURL string
	RedisAddr   string
	FMPAPIKey   string

	// WorkerTaskTypes restricts which task types this worker handles (comma-separated:
	// enqueue, daily, weekly, hourly). Empty = all types.
	WorkerTaskTypes []string

	// Discord webhooks (optional). If set, status notifications are sent.
	DiscordWebhookDaily  string
	DiscordWebhookWeekly string
	DiscordWebhookHourly string

	JobTimeoutSec int

	// ShadowMode: when true, write alert trigger results to ShadowOutputDir for comparison with Python.
	ShadowMode     bool
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

		JobTimeoutSec: 900,

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
	if c.RedisAddr == "" {
		c.RedisAddr = "localhost:6379"
	}
	if v := os.Getenv("WORKER_TASK_TYPES"); v != "" {
		for _, s := range strings.Split(v, ",") {
			if t := strings.TrimSpace(strings.ToLower(s)); t != "" {
				c.WorkerTaskTypes = append(c.WorkerTaskTypes, t)
			}
		}
	}
	return c
}

// HandlesTaskType returns true if this worker should handle the given task type
// (e.g. "daily", "weekly", "hourly", "enqueue"). Empty WorkerTaskTypes = handle all.
func (c *Config) HandlesTaskType(taskType string) bool {
	if len(c.WorkerTaskTypes) == 0 {
		return true
	}
	t := strings.ToLower(strings.TrimSpace(taskType))
	for _, allowed := range c.WorkerTaskTypes {
		if allowed == t {
			return true
		}
	}
	return false
}

// JobTimeout returns the job timeout as a duration.
func (c *Config) JobTimeout() time.Duration {
	return time.Duration(c.JobTimeoutSec) * time.Second
}

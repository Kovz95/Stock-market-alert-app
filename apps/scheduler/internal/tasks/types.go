package tasks

// Task type constants for Asynq.
const (
	TypeDaily  = "task:daily"
	TypeWeekly = "task:weekly"
	TypeHourly = "task:hourly"

	// EnqueueTaskType is the cron-driven task that enqueues exchange jobs.
	EnqueueTaskType = "scheduler:enqueue"
)

// Payload holds exchange and timeframe for a scheduled job.
type Payload struct {
	Exchange  string `json:"exchange"`
	Timeframe string `json:"timeframe"` // "daily", "weekly", "hourly"
}

// TaskName returns the Asynq task type for the given timeframe.
func TaskName(timeframe string) string {
	switch timeframe {
	case "weekly":
		return TypeWeekly
	case "hourly":
		return TypeHourly
	default:
		return TypeDaily
	}
}

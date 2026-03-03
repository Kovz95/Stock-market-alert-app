// Package main provides a CLI to reset (purge) the Asynq queue by deleting all
// scheduled, pending, retry, archived, and completed tasks. Use this when Redis
// has bad or stale task data.
//
// Usage:
//
//	# Dry run (show what would be deleted):
//	REDIS_ADDR=localhost:6379 go run ./cmd/resetqueue/
//
//	# Actually delete all tasks (requires --yes):
//	REDIS_ADDR=localhost:6379 go run ./cmd/resetqueue/ --yes
//
// For a full Redis reset (if this Redis DB is only used by Asynq), stop the
// worker and run: redis-cli -u <REDIS_ADDR> FLUSHDB
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/hibiken/asynq"

	"stockalert/apps/scheduler/internal/config"
)

const defaultQueue = "default"

func main() {
	confirm := flag.Bool("yes", false, "confirm reset (required to actually delete tasks)")
	flag.Parse()

	cfg := config.Load()
	inspector := asynq.NewInspector(asynq.RedisClientOpt{Addr: cfg.RedisAddr})
	defer inspector.Close()

	fmt.Fprintf(os.Stderr, "Redis: %s  Queue: %s\n", cfg.RedisAddr, defaultQueue)
	if !*confirm {
		fmt.Fprintf(os.Stderr, "Dry run (use --yes to delete).\n\n")
	}

	// Dry run: list and count only. With --yes: delete and count.
	type taskType struct {
		name string
		list func(string, ...asynq.ListOption) ([]*asynq.TaskInfo, error)
		del  func(string) (int, error)
	}
	types := []taskType{
		{"scheduled", inspector.ListScheduledTasks, inspector.DeleteAllScheduledTasks},
		{"pending", inspector.ListPendingTasks, inspector.DeleteAllPendingTasks},
		{"retry", inspector.ListRetryTasks, inspector.DeleteAllRetryTasks},
		{"archived", inspector.ListArchivedTasks, inspector.DeleteAllArchivedTasks},
		{"completed", inspector.ListCompletedTasks, inspector.DeleteAllCompletedTasks},
	}

	var total int
	for _, t := range types {
		var n int
		var err error
		if *confirm {
			n, err = t.del(defaultQueue)
		} else {
			var infos []*asynq.TaskInfo
			infos, err = t.list(defaultQueue)
			n = len(infos)
		}
		if err != nil {
			op := "List"
			if *confirm {
				op = "Delete"
			}
			fmt.Fprintf(os.Stderr, "%s %s: %v\n", op, t.name, err)
			os.Exit(1)
		}
		if n > 0 {
			verb := "Would delete"
			if *confirm {
				verb = "Deleted"
			}
			fmt.Fprintf(os.Stderr, "%s %d %s task(s).\n", verb, n, t.name)
			total += n
		}
	}

	if total == 0 {
		fmt.Fprintf(os.Stderr, "Queue is already empty.\n")
		return
	}
	if *confirm {
		fmt.Fprintf(os.Stderr, "Reset complete. Total tasks removed: %d\n", total)
	} else {
		fmt.Fprintf(os.Stderr, "Run with --yes to remove %d task(s).\n", total)
	}
}

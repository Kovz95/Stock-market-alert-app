// Package main provides a CLI to list scheduled, pending, and active tasks in the asynq queue.
//
// Usage:
//
//	REDIS_ADDR=localhost:6379 go run ./
//	# or build and run:
//	go build -o listqueue . && ./listqueue
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"text/tabwriter"
	"time"

	"github.com/hibiken/asynq"

	"stockalert/apps/scheduler/internal/config"
	"stockalert/apps/scheduler/internal/tasks"
)

const defaultQueue = "default"

func main() {
	cfg := config.Load()
	inspector := asynq.NewInspector(asynq.RedisClientOpt{Addr: cfg.RedisAddr})
	defer inspector.Close()

	fmt.Fprintf(os.Stderr, "Redis: %s  Queue: %s\n\n", cfg.RedisAddr, defaultQueue)

	tw := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	defer tw.Flush()

	// Header
	fmt.Fprintln(tw, "State\tType\tExchange\tTimeframe\tNext process at\tID")
	fmt.Fprintln(tw, "-----\t----\t--------\t---------\t---------------\t---")

	printTasks(tw, inspector, "scheduled", defaultQueue, inspector.ListScheduledTasks)
	printTasks(tw, inspector, "pending", defaultQueue, inspector.ListPendingTasks)
	printTasks(tw, inspector, "active", defaultQueue, inspector.ListActiveTasks)

	fmt.Fprintf(os.Stderr, "\nUse asynq dash (go install github.com/hibiken/asynq/tools/asynq@latest) for a live UI.\n")
}

type listFunc func(queue string, opts ...asynq.ListOption) ([]*asynq.TaskInfo, error)

func printTasks(tw *tabwriter.Writer, inspector *asynq.Inspector, state, queue string, list listFunc) {
	infos, err := list(queue)
	if err != nil {
		fmt.Fprintf(os.Stderr, "List %s: %v\n", state, err)
		return
	}
	if len(infos) == 0 {
		return
	}
	for _, info := range infos {
		exchange, timeframe := decodePayload(info.Payload)
		nextAt := ""
		if !info.NextProcessAt.IsZero() {
			nextAt = info.NextProcessAt.UTC().Format(time.RFC3339)
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%s\t%s\n",
			state,
			info.Type,
			exchange,
			timeframe,
			nextAt,
			info.ID,
		)
	}
}

func decodePayload(payload []byte) (exchange, timeframe string) {
	var p tasks.Payload
	if err := json.Unmarshal(payload, &p); err != nil {
		return "-", "-"
	}
	return p.Exchange, p.Timeframe
}

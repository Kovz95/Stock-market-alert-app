# Scheduler Stop/Start Reset Plan

## Problem: "Weird state" after Stop/Start

**Current behavior:**

1. **Stop** (UI) → gRPC `StopScheduler` → only **pauses** the queue (`inspector.PauseQueue("default")`). All tasks remain in Redis (scheduled, pending, active, retry, archived).
2. **Start** → `UnpauseQueue("default")`. The same tasks are still there; no cleanup.
3. The **worker** is a separate process (`apps/scheduler/cmd/worker`). The UI does not start or stop it. It runs a **schedule loop** every 15 minutes that enqueues daily/weekly/hourly tasks with `ProcessAt` and `Unique`.

So you can get:

- Queue paused but worker still running (or vice versa).
- Stale tasks with old `ProcessAt` times mixed with new ones after multiple stop/start cycles.
- Tasks stuck in **active** if the worker died without graceful shutdown.
- Duplicate or confusing queue state when you expect a "clean" restart.

## Desired behavior (your approach)

- **Stop**: Pause the queue **and** remove all current tasks from Redis (purge). When the worker is stopped, the queue is empty.
- **Start**: Unpause the queue. When the worker runs again, its scheduler runs `scheduleAll` on startup and every 15 min, so tasks are **re-added** with fresh `ProcessAt`. System is in a known, reset state.

## Implementation plan

### 1. Backend: Purge queue on Stop (gRPC scheduler service)

**File:** `apps/grpc/scheduler_service/control_handler.go`

- In **StopScheduler**:
  1. Call `inspector.PauseQueue(defaultQueue)` (keep current behavior).
  2. Then run a **purge** of all task states so Redis is cleared.

**Purge logic** (reuse same states as `apps/scheduler/cmd/resetqueue/main.go`):

| State      | List                     | Delete                         |
|-----------|--------------------------|--------------------------------|
| scheduled | ListScheduledTasks       | DeleteAllScheduledTasks        |
| pending   | ListPendingTasks         | DeleteAllPendingTasks         |
| retry     | ListRetryTasks           | DeleteAllRetryTasks            |
| archived  | ListArchivedTasks        | DeleteAllArchivedTasks         |
| completed | ListCompletedTasks       | DeleteAllCompletedTasks        |
| active    | ListActiveTasks          | per-task: DeleteTask(queue, id)|

- **Active tasks**: Asynq Inspector has no `DeleteAllActiveTasks`. Use `ListActiveTasks(queue)` then for each task call `inspector.DeleteTask(queue, info.ID)`. That way, when the worker is still running, active tasks are removed from Redis (the worker may see an error when it tries to complete them; acceptable for a "reset" flow). If the worker is stopped first, there are no active tasks.
- Add a small helper in the same package, e.g. `purgeQueue(ctx, inspector, queue string) (int, error)`, that runs the above and returns total deleted (for logging). Call it from `StopScheduler` after `PauseQueue`.

**StartScheduler:** No change. Keep only `UnpauseQueue(defaultQueue)`. The worker’s scheduler is responsible for re-adding tasks when it runs.

### 2. Optional: Make purge configurable (later)

If you want to support "Stop (pause only)" vs "Stop and clear queue", you could:

- Add a proto field to `StopSchedulerRequest`, e.g. `clear_queue bool`.
- If `clear_queue == true` (or default), do pause + purge. If false, only pause.

For the first version, implementing "always purge on stop" is enough and matches your description.

### 3. Worker lifecycle (clarification)

- The **UI Stop** does not shut down the worker process; it only pauses the queue and (with this change) purges tasks.
- To fully "reset":
  - User clicks **Stop** → queue paused + Redis purged.
  - User stops the worker process (e.g. Ctrl+C, or your deployment’s stop).
  - User starts the worker again (e.g. run the binary or restart the container).
  - User clicks **Start** → queue unpaused. Worker’s `scheduleAll` runs on startup and every 15 min, so tasks are re-enqueued with correct `ProcessAt`.

So "when you hit stop and it shuts down the worker" can be interpreted as: **Stop** clears the queue so that when the worker is shut down (by whatever means), there’s nothing left in Redis; when the worker is started again and **Start** is clicked, the scheduler repopulates the queue. No code change is required for worker start/stop unless you want the UI to trigger worker process lifecycle (e.g. separate gRPC or signal), which is a larger change.

### 4. UI (apps/ui/web)

- **No change required** for the reset behavior. The scheduler page already calls `stopScheduler()` and `startScheduler()`. Once the backend purges on stop and the worker repopulates on start, the UI will just show the new state (e.g. empty queue after stop, then tasks reappearing after start when the worker runs).
- Optional: After **Stop**, you could show a short message like "Queue paused and cleared. Start the scheduler again to re-enqueue tasks (worker will repopulate within 15 minutes)." if you want to set expectations.

### 5. Order of operations on Stop

Recommended order:

1. **Pause** the queue first (so no new tasks are picked from pending/scheduled while we purge).
2. **Purge** all states (scheduled, pending, retry, archived, completed, then active).
3. Log counts and return success.

This avoids a race where the worker grabs a task between delete and pause.

### 6. Files to touch (summary)

| File | Change |
|------|--------|
| `apps/grpc/scheduler_service/control_handler.go` | Add `purgeQueue` helper; call it from `StopScheduler` after `PauseQueue`. Handle active via `ListActiveTasks` + `DeleteTask` per task. |
| `apps/ui/web/app/scheduler/page.tsx` | No change (or optional copy/message as above). |

### 7. Testing

- **Manual:** Start worker, wait for some scheduled/pending tasks, click Stop in UI → queue should show 0 tasks and paused. Click Start, wait for worker’s next schedule cycle (or restart worker) → tasks reappear with fresh times.
- **Optional:** Add an integration test that calls StopScheduler, then lists queue tasks and asserts count 0 (or use inspector GetQueueInfo and assert sizes 0).

---

## Summary

- **Root cause of weird state:** Stop only paused the queue; tasks stayed in Redis. Start only unpaused; no re-enqueue from a single source of truth.
- **Fix:** On Stop, **pause then purge** all tasks (all states including active). On Start, **unpause** only. Worker’s existing schedule loop re-adds tasks on startup and every 15 minutes, giving a clean reset.

Implementing the purge in `control_handler.go` as above is sufficient; no proto or UI changes are strictly required for the reset behavior.

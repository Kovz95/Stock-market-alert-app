# Scheduler (Go worker + asynq)

This app runs the Asynq worker that processes daily, weekly, and hourly price/alert tasks. It also runs a **schedule loop** every 15 minutes that enqueues those tasks with `ProcessAt` and `Unique` options.

## Concurrency (why multiple jobs run with one worker)

The worker uses **Asynq’s per-process concurrency**: one process can run several tasks in parallel (multiple goroutines). That is controlled by **`SCHEDULER_CONCURRENCY`** (default **1**).

- **Default 1**: One job at a time per process. With a single worker process you’ll see one “job starting” / “hourly task received” at a time; others wait in the queue.
- **Set to 4** (or higher): Up to that many tasks run in parallel in the same process (e.g. COLOMBIA, LONDON, BUENOS AIRES at once).

Example: to allow 4 concurrent jobs in one process:

```bash
SCHEDULER_CONCURRENCY=4
```

## FMP rate limiting and 429s

The FMP client **rate-limits** all requests (daily, weekly, hourly) with a minimum interval between calls. It also **retries on 429** (up to 3 times with backoff). This keeps you under FMP’s per-minute limits and avoids failed jobs when you occasionally hit 429.

- **`FMP_MIN_INTERVAL_MS`** – Minimum milliseconds between any two FMP API requests (default **200** ≈ 5 req/s, ~300/min). Increase (e.g. `300` or `500`) if you still see 429s.
- **`SCHEDULER_FMP_DAILY_CONCURRENCY`**, **`SCHEDULER_FMP_WEEKLY_CONCURRENCY`**, **`SCHEDULER_FMP_HOURLY_CONCURRENCY`** – Max parallel FMP fetches per job (default **10**). Throughput is still capped by `FMP_MIN_INTERVAL_MS`. Lower concurrency reduces burst; higher concurrency just means more goroutines waiting on the same rate limit.

The schedule loop’s single **market-hours** FMP call also retries on 429.

## Hourly price update performance

For large exchanges (e.g. NASDAQ with 1800+ tickers), the hourly job fetches prices from FMP in **parallel**. That is controlled by **`SCHEDULER_FMP_HOURLY_CONCURRENCY`** (default **10**).

- Higher values (e.g. 15–20) allow more parallel fetches; actual throughput is still capped by **`FMP_MIN_INTERVAL_MS`**.
- Lower values (e.g. 5) reduce burst and can help if you still see 429s.

Prices are then written in bulk (batch upsert), so total time is dominated by FMP fetch rate and parallelism.

## Why you might not see hourly jobs enqueued

1. **Market hours**  
   Hourly tasks are only enqueued when `calendar.IsExchangeOpen(exchange, now)` is true at the time of the 15‑minute cycle. If it’s outside market hours for all exchanges (e.g. weekend, or outside 9:30–16:00 local for each exchange), `scheduled_hourly` and `open_exchanges` will be 0.

2. **Where to look**  
   Hourly tasks are enqueued with **ProcessAt(now + 30 min)** and **Unique(30 min)**. They appear under **Scheduled** in the queue (e.g. listqueue or Asynqmon), not in the immediate “Enqueued”/pending list until their `ProcessAt` time is reached.

3. **Why `scheduled_hourly` is often 0 in logs**  
   We only enqueue one hourly task per exchange per 30 minutes. On the next 15‑minute cycle the same exchange is still open, but Asynq returns “task already exists” and we skip. So `scheduled_hourly` is only non‑zero in the cycle where we *newly* enqueue. Check **open_exchanges** in the same log line: if it’s > 0, markets were open and hourly was either newly enqueued or already enqueued (skip).

## Why you see "task already exists"

The scheduler enqueues each exchange’s tasks with **Unique** (12h for daily/weekly, 30m for hourly). Every 15 minutes it tries to enqueue again. If a task for that exchange was already enqueued within the uniqueness window, Asynq returns **"task already exists"** and does not add a duplicate.

That is **expected**. Those messages are now logged at **Debug** (e.g. `schedule daily skipped (already enqueued)`). Only real enqueue failures are logged as Warn.

## Why jobs don’t appear to be "running"

- **Daily and weekly** tasks are enqueued with **ProcessAt(next run time)** (e.g. next market close). They stay in Redis as **scheduled** until that time. They are not "running" until the worker picks them up at `ProcessAt`.
- **Hourly** tasks are enqueued when the exchange is open, with **ProcessAt(now + 30 min)** and Unique(30 min). They appear in **scheduled** until that time, then move to pending/active.

So you will only see tasks **actively running** when:

1. Their `ProcessAt` time has been reached (daily/weekly), or they were enqueued for immediate run (hourly), and  
2. The **worker process** is running (same process that runs the schedule loop and the Asynq server).

To verify:

- Ensure the **worker** is running (the binary that starts the Asynq server and the schedule loop).
- Use the **listqueue** CLI or [Asynqmon](https://github.com/hibiken/asynqmon) / **asynq** CLI against the same Redis to inspect **scheduled** vs **pending** vs **active** tasks.

## Check what jobs are scheduled in the queue

**Option 1: listqueue (in-repo CLI)**  
From `apps/scheduler`:

```bash
# Use same REDIS_ADDR as the worker (default: localhost:6379)
set REDIS_ADDR=localhost:6379   # Windows
export REDIS_ADDR=localhost:6379   # Linux/macOS

go run ./cmd/listqueue/
# or build once and run:
go build -o listqueue ./cmd/listqueue/ && ./listqueue
```

This prints a table of **scheduled**, **pending**, and **active** tasks (exchange, timeframe, next process time, task ID).

**Option 2: Asynq CLI**  
Install and run the official dashboard (same Redis as worker):

```bash
go install github.com/hibiken/asynq/tools/asynq@latest
asynq dash --redis-addr=localhost:6379
```

**Option 3: Asynqmon**  
Web UI for monitoring; point it at your Redis URL.

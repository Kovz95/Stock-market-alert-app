# `apps/scheduler/` — Go Scheduler Worker (Asynq)

## What This Service Is

The scheduler is an **Asynq-based Go worker** that processes stock market price updates and alert evaluations across 42+ global exchanges. It runs three task types (`task:daily`, `task:weekly`, `task:hourly`) and a **schedule loop** that enqueues those tasks every 15 minutes with DST-correct timing.

This is the Go port of the original Python scheduler, using Asynq for task queuing and go-talib for indicator computation.

---

## How It Fits Into The App

```
Schedule Loop (every 15 min)
  → Checks which exchanges are open (FMP API + calendar)
  → Enqueues tasks with ProcessAt (future run time) and Unique (dedup)
      → Asynq worker picks up tasks at ProcessAt time
          → Fetches prices from FMP API
          → Upserts prices to PostgreSQL
          → Loads alerts from DB
          → Pre-warms OHLCV cache
          → Evaluates alert conditions via expr/indicator packages
          → Sends triggered alerts to Discord (economy-based routing)
          → Writes audit records (bulk COPY)
          → Updates scheduler status in app_documents table
```

**External dependencies:**
- **PostgreSQL** — price data, alerts, audit records, status, Discord routing config
- **Redis** — Asynq task queue
- **FMP API** — price data fetching (daily, weekly, hourly candles)
- **Discord Webhooks** — alert notifications and job status updates

---

## Directory Structure

```
apps/scheduler/
├── cmd/
│   ├── worker/          # Main worker entry point (starts Asynq server + schedule loop)
│   ├── listqueue/       # CLI: inspect Asynq queue (scheduled, pending, active tasks)
│   └── resetqueue/      # CLI: purge all tasks from Asynq (requires --yes flag)
├── internal/
│   ├── config/          # Environment-based configuration (env vars → Config struct)
│   ├── handler/         # Task handlers (daily, weekly, hourly) + shared Execute() logic
│   ├── price/           # FMP API client, price updater, candle timing detection
│   ├── schedule/        # Schedule loop (enqueues tasks every 15 min)
│   ├── status/          # Status manager (writes heartbeat to app_documents table)
│   └── tasks/           # Task type constants and Payload struct
├── Dockerfile           # Multi-stage build (golang:1.24-alpine → alpine:3.21)
├── go.mod               # Module: stockalert/apps/scheduler
└── README.md            # Operational guide (concurrency, rate limiting, troubleshooting)
```

---

## File-by-File Breakdown

### `cmd/worker/main.go` — Entry Point

The main binary that starts everything:

1. Connects to PostgreSQL with retry logic (30 attempts, 2s backoff)
2. Creates all dependencies: indicator registry, alert checker, Discord router/notifier, FMP client, price updater
3. Starts Asynq server with configurable concurrency
4. Registers three handlers: `task:daily`, `task:weekly`, `task:hourly`
5. Starts schedule loop in a background goroutine
6. Sends Discord "worker online/offline" notifications on lifecycle events
7. Uses structured JSON logging (slog)

### `cmd/listqueue/main.go` — Queue Inspection

Prints a table of scheduled, pending, and active tasks. Useful for verifying task state without Asynqmon. Uses page size 500 to show all ~132 tasks (44 exchanges x 3 timeframes).

### `cmd/resetqueue/main.go` — Queue Reset

Purges all tasks from Asynq (scheduled, pending, retry, archived, completed). **Requires `--yes` flag** for safety (dry-run by default). Use when Redis has stale or corrupt task data.

### `internal/config/config.go` — Configuration

Loads all settings from environment variables. Provides typed accessors and defaults. See [Environment Variables](#environment-variables) below.

### `internal/tasks/types.go` — Task Types

Defines constants and the shared `Payload` struct:

```go
const (
    TypeDaily  = "task:daily"
    TypeWeekly = "task:weekly"
    TypeHourly = "task:hourly"
)

type Payload struct {
    Exchange  string
    Timeframe string
}
```

### `internal/handler/common.go` — Shared Handler Logic

The `Common` struct holds all dependencies. The `Execute()` method is the main job lifecycle, called by all three handlers:

1. Check if hourly job should run (`calendar.IsExchangeOpen`)
2. Notify Discord job start
3. Update prices via `Updater.UpdateForExchange()`
4. Load alerts from DB (with timeframe variant matching: `"daily"/"1d"`, `"weekly"/"1wk"`, `"hourly"/"1h"`)
5. Pre-warm price cache via `Checker.PreWarmCache()`
6. Evaluate alerts via `Checker.CheckAlerts()` with trigger callback
7. Trigger callback accumulates Discord embeds
8. Flush accumulated embeds
9. Report success/error to status manager and Discord

**Shadow mode:** When `SCHEDULER_SHADOW_MODE=true`, writes triggered alert IDs to JSON files for Go-vs-Python comparison.

### `internal/handler/daily.go`, `weekly.go`, `hourly.go` — Task Handlers

Thin wrappers that implement `asynq.Handler`. Each unmarshals the `Payload` and delegates to `Common.Execute()`. Hourly handler checks `calendar.IsExchangeOpen()` and returns `SkipRetry` if market is closed.

### `internal/schedule/scheduler.go` — Schedule Loop

Runs every 15 minutes. For each cycle:

- **Daily:** Enqueues with `ProcessAt(market close + 40min)`, `Unique(12h)`. Only on trading days.
- **Weekly:** Same as daily, but only on Fridays.
- **Hourly:** Enqueues with `ProcessAt(now + 20-30min)`, `Unique(1h)`. Only when exchange is open.

Uses FMP API to fetch real market hours (with fallback to calendar). `nextHourlyRunTime()` probes FMP to detect actual candle duration (handles lunch breaks, 30min/15min candles for non-US exchanges).

"Task already exists" errors are expected (due to Unique dedup) and logged at Debug level.

### `internal/price/updater.go` — Price Updater

Orchestrates FMP API calls and bulk price upserts:

- **Daily:** Parallel fetch (configurable concurrency), individual `UpsertDailyPrice` per row
- **Weekly:** Fetch daily data, resample to weekly bars (Friday week-ending), `UpsertWeeklyPrice`
- **Hourly:** Parallel fetch (configurable concurrency), bulk upsert in 10k-row chunks via `BulkUpsertHourlyPrices`

Returns `PriceStats{Total, Updated, Failed, Skipped}`.

### `internal/price/fmp_client.go` — FMP API Client

HTTP client with global rate limiting and 429 retry:

- Rate limit via `FMP_MIN_INTERVAL_MS` (default 200ms = ~5 req/s)
- Retries 429 responses up to 3 times with backoff from Retry-After header
- `FetchDaily(ticker, limit)` — `/historical-price-full/{ticker}?timeseries={limit}`
- `FetchHourly(ticker)` — `/historical-chart/1hour/{ticker}`

### `internal/price/candle_timing.go` — Candle Timing Detection

Determines true candle duration for each exchange by fetching recent candles from FMP:

- Some exchanges use 30min bars (Hong Kong, Paris), some have lunch breaks (90+ min gaps)
- Fetches last 5 candles, finds minimum gap >= 15min (filters out lunch breaks)
- `ExchangeRepresentativeTicker` maps each exchange to a liquid ticker for probing (e.g. AAPL for NYSE, SHEL.L for London)
- Falls back to arithmetic timing based on `HourlyAlignment` map

### `internal/status/manager.go` — Status Manager

Writes scheduler status to `app_documents` table (key: `"scheduler_status"`):

- `UpdateRunning()` — sets status=running, records current job + heartbeat
- `UpdateSuccess()` — clears current_job, sets last_run + last_result
- `UpdateError()` — sets status=error, records last_error

Used by the Streamlit UI for "worker status" display.

### `internal/price/cache.go` — Run Cache

Simple in-memory cache tracking which tickers were updated in the current run. Not shared across jobs. Distinct from the alert checker's OHLCV cache.

---

## Shared Packages

The scheduler depends on these sibling modules via `replace` directives in `go.mod`:

| Package | Purpose | Key Exports |
|---|---|---|
| `indicator/` | 60+ technical indicators (go-talib + custom) | `Registry`, `NewDefaultRegistry()`, `OHLCV`, `IndicatorFunc` |
| `expr/` | Expression language parser + evaluator | `ParseCondition()`, `ParseConditionList()`, `NewEvaluator()`, `EvalConditionList()` |
| `alert/` | Alert evaluation orchestrator | `Checker`, `CheckAlerts()`, `PreWarmCache()` |
| `calendar/` | Market hours, DST handling, exchange schedules | `IsExchangeOpen()`, `GetNextDailyRunTime()`, `ExchangeSchedules` |
| `discord/` | Webhook notifications + routing | `Notifier`, `Router`, `Accumulator`, `StatusNotifier` |
| `database/` | sqlc-generated PostgreSQL client | `Queries`, all query methods |

See `expr/CLAUDE.md` for detailed documentation of the expression language.

---

## Environment Variables

### Required

| Variable | Description |
|---|---|
| `DATABASE_URL` | PostgreSQL connection string |
| `REDIS_ADDR` | Redis address for Asynq (default: `localhost:6379`) |
| `FMP_API_KEY` | Financial Modeling Prep API key |

### Scheduler Tuning

| Variable | Default | Description |
|---|---|---|
| `SCHEDULER_CONCURRENCY` | `2` | Parallel Asynq jobs per worker process |
| `SCHEDULER_JOB_TIMEOUT` | `900` | Job timeout in seconds (15 min) |
| `SCHEDULER_FMP_DAILY_CONCURRENCY` | `10` | Parallel FMP fetches for daily jobs |
| `SCHEDULER_FMP_WEEKLY_CONCURRENCY` | `10` | Parallel FMP fetches for weekly jobs |
| `SCHEDULER_FMP_HOURLY_CONCURRENCY` | `10` | Parallel FMP fetches for hourly jobs |
| `FMP_MIN_INTERVAL_MS` | `200` | Min ms between FMP requests (~5 req/s) |

### Discord

| Variable | Default | Description |
|---|---|---|
| `DISCORD_SEND_ENABLED` | `false` | Enable Discord sending (`true`/`false`, `1`/`0`) |
| `DISCORD_ENVIRONMENT` | — | `PROD` or `DEV` (prefixed to messages) |
| `DISCORD_WEBHOOK_INTERVAL_SEC` | `2` | Min seconds between sends to same webhook |
| `DISCORD_WEBHOOK_DAILY` | — | Webhook URL for daily job status |
| `DISCORD_WEBHOOK_WEEKLY` | — | Webhook URL for weekly job status |
| `DISCORD_WEBHOOK_HOURLY` | — | Webhook URL for hourly job status |

### Shadow Mode

| Variable | Default | Description |
|---|---|---|
| `SCHEDULER_SHADOW_MODE` | `false` | Write triggered alert IDs to JSON for Go-vs-Python comparison |
| `SCHEDULER_SHADOW_OUTPUT_DIR` | `shadow_results` | Output directory for shadow JSON files |

---

## Key Architectural Decisions

### Task Scheduling (Not Cron)

The schedule loop runs every 15 minutes and enqueues tasks with **`ProcessAt`** (future execution time) and **`Unique`** (dedup window). This is intentional — it allows DST-correct scheduling and handles the fact that different exchanges close at different times.

- Daily/weekly: `ProcessAt(market close + 40 min)`, `Unique(12h)`
- Hourly: `ProcessAt(candle close + 20 min)`, `Unique(1h)`

### Deferred Bulk Writes

Alert evaluation collects audit records and trigger updates in memory, then flushes them in bulk after all alerts finish:

- Audit records: bulk COPY protocol
- Trigger updates: bulk `UPDATE ... WHERE id IN (...)`

This avoids per-alert round trips to the database.

### Price Cache Strategy

Two distinct caches:

1. **Run cache** (price updater): tracks which tickers were updated this run to avoid duplicate FMP calls
2. **OHLCV cache** (alert checker): pre-warmed via batch query before evaluation. One query loads all tickers for an exchange+timeframe. Not shared across jobs.

### Panic Recovery

go-talib panics when data length < period. The evaluator wraps indicator calls in `recover()` and returns an error instead of crashing the worker.

### NaN Safety

If either side of a comparison resolves to NaN (insufficient data), the comparison returns `false` — never panics, never triggers.

### Timeframe Variant Matching

Alerts in the DB may use `"daily"` or `"1d"`, `"weekly"` or `"1wk"`, `"hourly"` or `"1h"`. The handler queries for both variants when loading alerts.

### Economy-Based Discord Routing

Triggered alerts are routed to different Discord channels based on the ticker's economy (loaded from `discord_routing` table). The `Accumulator` batches embeds (max 10 per message) and the `Notifier` rate-limits sends (2s per webhook).

---

## Building and Running

### Local Development

```bash
# From project root (go.work handles module resolution)
cd apps/scheduler

# Run the worker
go run ./cmd/worker/

# Inspect the queue
go run ./cmd/listqueue/

# Reset the queue (dry run)
go run ./cmd/resetqueue/

# Reset the queue (for real)
go run ./cmd/resetqueue/ --yes
```

### Docker Build

The Dockerfile uses a multi-stage build. Context **must be the repo root** (not `apps/scheduler/`):

```bash
# From project root
docker build -f apps/scheduler/Dockerfile -t scheduler .
docker run --env-file .env scheduler
```

The Dockerfile copies only needed modules (alert, calendar, database, discord, expr, indicator, apps/scheduler) and does NOT copy `go.work` to avoid pulling in gRPC/gen/go modules.

### Build constraints

- `CGO_ENABLED=0` — static binary, no CGo dependencies
- `GOOS=linux` — targets Linux for container deployment
- Base image: `alpine:3.21`

---

## Running Tests

```bash
# All scheduler tests
cd apps/scheduler && go test ./...

# Specific packages
cd apps/scheduler && go test ./internal/handler/ -v
cd apps/scheduler && go test ./internal/price/ -v

# Shared package tests (from project root)
go test ./indicator/...
go test ./expr/...
go test ./alert/...
go test ./calendar/...
```

### Test Patterns

- Integration tests use mock FMP client (`internal/price/mock_fmp.go`) and in-memory caches
- Unit tests verify indicator calculations against known values
- Expression tests verify parsing and evaluation logic
- All tests use standard Go `testing` package

---

## Operational Notes

### Inspecting the Queue

Use `cmd/listqueue` or the Asynq CLI:

```bash
# In-repo CLI
go run ./cmd/listqueue/

# Official Asynq CLI
go install github.com/hibiken/asynq/tools/asynq@latest
asynq dash --redis-addr=localhost:6379
```

### Common Log Messages

| Message | Level | Meaning |
|---|---|---|
| `schedule daily skipped (already enqueued)` | Debug | Expected — Unique dedup working correctly |
| `scheduled_hourly=0, open_exchanges=0` | Info | No exchanges open right now (weekend, off-hours) |
| `indicator "X" panicked` | Warn | go-talib panic caught — data too short for period |
| `FMP 429, retrying` | Warn | Rate limited by FMP — auto-retries with backoff |

### Troubleshooting

1. **No hourly jobs running:** Check if exchanges are open (`calendar.IsExchangeOpen`). Hourly tasks only enqueue when market is open.
2. **Tasks stuck in "scheduled":** They haven't reached their `ProcessAt` time yet. Daily/weekly tasks schedule for market close + 40 min.
3. **429 errors from FMP:** Increase `FMP_MIN_INTERVAL_MS` (e.g. 300 or 500) or decrease `SCHEDULER_FMP_*_CONCURRENCY`.
4. **Worker won't start (DB connection):** Retries 30 times with 2s backoff. Check `DATABASE_URL` is correct and PostgreSQL is reachable.

---

## Key Metrics and Limits

| Metric | Value |
|---|---|
| Exchanges supported | 42 |
| Indicators available | 60+ |
| Tasks per 15-min cycle | ~132 (44 daily + 44 weekly + 44 hourly) |
| FMP rate limit | ~5 req/s (~300/min) at default 200ms interval |
| Discord rate limit | 2s min interval per webhook (~30 req/min) |
| Hourly upsert chunk size | 10,000 rows per bulk upsert |
| Daily FMP fetch limit | 750 days per request |
| Max Discord embeds per message | 10 |
| DB connection retries | 30 attempts, 2s backoff |

---

## How To Extend

### Add a New Task Type

1. Add the constant in `internal/tasks/types.go`
2. Create a handler in `internal/handler/` (implement `asynq.Handler`)
3. Register it in `cmd/worker/main.go`
4. Add scheduling logic in `internal/schedule/scheduler.go`

### Add a New Exchange

1. Add the exchange schedule in `calendar/exchanges.go` (`ExchangeSchedules` map)
2. Add a representative ticker in `internal/price/candle_timing.go` (`ExchangeRepresentativeTicker`)
3. Add candle timezone mapping in `internal/price/candle_timing.go` (`fmpCandleTimezone`)
4. Add hourly alignment if non-standard in `calendar/exchanges.go` (`HourlyAlignment`)

### Add a New Indicator

See `expr/CLAUDE.md` for the full process. In short:

1. Implement the function in `indicator/` (return `[]float64` same length as input, NaN-fill leading values)
2. Register in `indicator/registry.go` → `NewDefaultRegistry()`
3. Optionally add positional param remapping in `expr/parser.go`
4. Optionally add an alias in `expr/evaluator.go`
5. Write tests

---

## Module Dependencies

```
apps/scheduler
├── github.com/hibiken/asynq    (task queue)
├── github.com/jackc/pgx/v5     (PostgreSQL driver)
├── stockalert/alert             (alert evaluation orchestrator)
│   ├── stockalert/expr          (expression parser/evaluator)
│   │   └── stockalert/indicator (technical indicators + go-talib)
│   └── stockalert/database      (sqlc-generated queries)
├── stockalert/calendar          (market hours, DST, exchange schedules)
├── stockalert/discord           (webhook notifications + routing)
│   └── stockalert/database
└── stockalert/indicator
```

All local dependencies use `replace` directives in `go.mod` pointing to sibling directories (e.g. `../../alert`).

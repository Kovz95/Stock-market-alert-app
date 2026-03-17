# Stock Market Alert App

A production-grade stock market alert system built as a microservices monorepo. Users define technical-indicator conditions on stocks/ETFs/futures across global exchanges, and the system automatically evaluates those alerts on a market-aware schedule, routing triggered notifications to Discord channels segmented by industry/economy.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Repository Structure](#repository-structure)
3. [Services Overview](#services-overview)
   - [Next.js Web App](#nextjs-web-app)
   - [Alert Service (gRPC)](#alert-service-grpc)
   - [Price Service (gRPC)](#price-service-grpc)
   - [Discord Service (gRPC)](#discord-service-grpc)
   - [Scheduler Service (gRPC)](#scheduler-service-grpc)
   - [Worker / Scheduler Process](#worker--scheduler-process)
   - [Envoy Proxy](#envoy-proxy)
4. [Database Schema](#database-schema)
5. [Alert Evaluation Flow](#alert-evaluation-flow)
6. [Scheduler & Task Queue Flow](#scheduler--task-queue-flow)
7. [Technical Indicator Library](#technical-indicator-library)
8. [Proto / Code Generation](#proto--code-generation)
9. [Environment Variables](#environment-variables)
10. [Local Development Setup](#local-development-setup)
11. [Running with Docker Compose](#running-with-docker-compose)
12. [Go Workspace](#go-workspace)
13. [UI Pages Reference](#ui-pages-reference)
14. [Discord Notification Routing](#discord-notification-routing)
15. [Adding a New Indicator](#adding-a-new-indicator)
16. [Adding a New gRPC Method](#adding-a-new-grpc-method)
17. [Scripts Reference](#scripts-reference)
18. [Testing](#testing)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Browser                                                                │
│  Next.js 16 (React 19) — port 8501                                     │
│  Server Actions → nice-grpc → Envoy (HTTP/1.1 gRPC-Web) → port 8081  │
└─────────────────────────┬───────────────────────────────────────────────┘
                          │ gRPC-Web (HTTP/1.1)
                ┌─────────▼──────────┐
                │   Envoy Proxy      │  port 8081
                │  (path-based       │  Routes /stockalert.*.Service/*
                │   routing)         │  to correct backend
                └──┬──────┬──────┬──┬┘
                   │      │      │  │
          ┌────────▼┐ ┌───▼──┐ ┌▼──┴──┐ ┌──────────────────┐
          │ alert_  │ │price_│ │discord│ │ scheduler_service │
          │ service │ │serv. │ │ serv. │ │  (gRPC + Asynq   │
          │ :50051  │ │:50051│ │:50051 │ │   inspector)     │
          └────┬────┘ └──┬───┘ └───┬───┘ └──────┬───────────┘
               │         │         │              │
               └─────────┴────┬────┴──────────────┘
                              │ PostgreSQL (port 5433 on host)
                        ┌─────▼──────┐
                        │  Postgres  │
                        │  (port 5432│
                        │  internal) │
                        └────────────┘
                              ▲
                ┌─────────────┤  Redis (port 6378 on host)
                │       ┌─────┴──────┐
                │       │   Redis    │
                │       │  :6379     │
                │       └─────▲──────┘
                │             │ Asynq task queue
                │    ┌────────┴──────────────────────────┐
                │    │  Worker Process (apps/scheduler)   │
                │    │  ┌──────────────────────────────┐  │
                │    │  │  Background Scheduler goroutine│  │
                │    │  │  (scheduleAll every 15 min)   │  │
                │    │  └──────────────────────────────┘  │
                │    │  ┌──────────────────────────────┐  │
                │    │  │  Asynq Worker                 │  │
                │    │  │  DailyHandler                 │  │
                │    │  │  WeeklyHandler                │  │
                │    │  │  HourlyHandler                │  │
                │    │  └──────────────────────────────┘  │
                │    └───────────────────────────────────-┘
                │
                └── FMP API (Financial Modeling Prep)
                    Price data fetched during scheduled jobs
```

### Key Design Decisions

- **gRPC-Web via Envoy**: Browsers can't speak HTTP/2 gRPC natively. Next.js Server Actions call `nice-grpc` (Node.js gRPC client), which hits the Envoy proxy. Envoy translates the gRPC-Web requests into native gRPC/HTTP2 and routes to the correct backend by URL prefix.
- **Single Worker Binary**: The `apps/scheduler` binary contains both the Asynq worker (task consumer) and the background scheduler goroutine (task producer). They share the same process, database pool, and dependencies.
- **No gRPC between Worker and Alert Service**: The worker calls the alert evaluation code directly as Go library calls (not over the network), since it shares the same Go module workspace.
- **sqlc for DB layer**: All SQL queries are defined in `database/sql/queries/` and the Go structs/methods are code-generated into `database/generated/`. Do not hand-edit generated files.
- **Proto-first API contract**: All inter-service communication is defined in `proto/`. TypeScript and Go client/server code are both generated from these definitions.

---

## Repository Structure

```
Stock-market-alert-app/
├── apps/
│   ├── grpc/
│   │   ├── alert_service/      # Go gRPC: alert CRUD, evaluation, audit, portfolios
│   │   ├── price_service/      # Go gRPC: price data, scanner, on-demand updates
│   │   ├── discord_service/    # Go gRPC: Discord webhook config management
│   │   └── scheduler_service/  # Go gRPC: scheduler monitoring & control (Asynq inspector)
│   ├── scheduler/
│   │   ├── cmd/
│   │   │   ├── worker/         # Main: Asynq worker + background scheduler
│   │   │   ├── listqueue/      # CLI: inspect Asynq queue
│   │   │   └── resetqueue/     # CLI: clear Asynq queue
│   │   └── internal/
│   │       ├── config/         # Env-based config loading
│   │       ├── handler/        # DailyHandler, WeeklyHandler, HourlyHandler
│   │       ├── price/          # FMP client, price updater, candle timing
│   │       ├── schedule/       # Background scheduler (scheduleAll loop)
│   │       ├── status/         # Status manager (writes to DB)
│   │       └── tasks/          # Task type constants & payload struct
│   └── ui/
│       └── web/                # Next.js 16 (React 19) frontend
│           ├── app/            # Next.js App Router pages
│           │   ├── alerts/     # List, add, delete, history, audit pages
│           │   ├── database/   # Stock database page
│           │   ├── discord/    # Discord config (daily/hourly/weekly)
│           │   ├── portfolios/ # Portfolio management
│           │   ├── price-database/ # Price data viewer
│           │   ├── scanner/    # Stock scanner
│           │   └── scheduler/  # Scheduler status & control
│           ├── actions/        # Next.js Server Actions (gRPC calls)
│           ├── components/     # Shared UI components (shadcn/ui)
│           ├── lib/
│           │   ├── grpc/       # gRPC channel & client setup
│           │   ├── hooks/      # TanStack Query hooks
│           │   └── store/      # Jotai atoms
│           └── Dockerfile
├── proto/                      # Protobuf definitions (source of truth)
│   ├── alert/v1/alert.proto
│   ├── price/v1/price.proto
│   ├── discord/v1/discord.proto
│   └── scheduler/v1/scheduler.proto
├── gen/
│   ├── go/                     # Generated Go proto stubs
│   └── ts/                     # Generated TypeScript proto stubs
├── alert/                      # Go lib: alert checker & evaluator
├── calendar/                   # Go lib: exchange schedule & market hours
├── database/                   # sqlc schema, queries, and generated code
├── discord/                    # Go lib: Discord notification (formatter, router, accumulator)
├── expr/                       # Go lib: condition expression parser/evaluator
├── indicator/                  # Go lib: technical indicator registry (50+ indicators)
├── pages/                      # Legacy Streamlit pages (Python, being migrated)
├── src/                        # Legacy Python services (backend, scheduler, etc.)
├── scripts/
│   ├── analysis/               # Read-only diagnostic and monitoring scripts
│   ├── maintenance/            # Data maintenance scripts
│   └── migration/              # One-time data migration scripts
├── db/postgres_schema.sql      # Bootstrap schema (used by Docker)
├── database/sql/schema.sql     # Canonical schema (used by sqlc)
├── docker-compose.yml
├── envoy.yaml                  # Envoy proxy routing config
├── buf.gen.yaml                # Buf codegen config
├── buf.yaml                    # Buf workspace config
├── go.work                     # Go workspace (ties all Go modules together)
└── .env.example                # Environment variable template
```

---

## Services Overview

### Next.js Web App

**Location**: `apps/ui/web`  
**Port**: 3000 (internal) / 8501 (host)  
**Stack**: Next.js 16, React 19, TypeScript, TailwindCSS 4, shadcn/ui, Jotai, TanStack Query, nice-grpc, Recharts, lightweight-charts

The frontend is a Next.js App Router application. All data fetching happens in **Server Actions** (`actions/`) using `nice-grpc`, which communicates directly with the Envoy proxy. No REST API layer exists — the frontend talks gRPC via Envoy.

**State management**:
- **Jotai atoms** (`lib/store/`) hold client-side UI state (filters, selections, form state)
- **`jotai-tanstack-query`** atoms bridge Jotai with TanStack Query for server state
- **TanStack Query hooks** (`lib/hooks/`) wrap Server Actions with caching and loading state

**gRPC client setup** (`lib/grpc/channel.ts`):
```ts
const channel = createChannel(process.env.GRPC_ENDPOINT || "127.0.0.1:8081");
export const alertClient = clientFactory.create(AlertServiceDefinition, channel);
export const priceClient  = clientFactory.create(PriceServiceDefinition, channel);
// ... discord, scheduler clients
```

The `GRPC_ENDPOINT` env var points to the Envoy proxy. In Docker, this is `envoy:8081`. Locally, it's `127.0.0.1:8081`.

---

### Alert Service (gRPC)

**Location**: `apps/grpc/alert_service`  
**Port**: 50051  
**Default port env**: `PORT`

Responsibilities:
- Full CRUD for alerts (`alerts` table)
- Alert audit log queries (`alert_audits` table)
- Portfolio CRUD (`portfolios`, `portfolio_stocks` tables)
- Stock search (from `stock_metadata`)
- **`EvaluateExchange` RPC**: synchronous alert evaluation for a given exchange + timeframe — fetches prices, runs indicator math, fires Discord

**Key handlers**:
| File | Purpose |
|------|---------|
| `handler.go` | Core alert CRUD (List, Get, Create, Update, Delete, BulkDelete) |
| `evaluate_handler.go` | `EvaluateExchange`: load alerts → pre-warm cache → evaluate → flush Discord |
| `audit_handler.go` | Audit log queries (summary, performance metrics, log pagination) |
| `alert_history_handler.go` | Trigger history by ticker, stock search |
| `portfolio_handler.go` | Portfolio CRUD + add/remove stocks |
| `price_updater.go` | FMP price update logic used by EvaluateExchange |

**Required env**:
```
DATABASE_URL=postgresql://...
FMP_API_KEY=...        # Optional; EvaluateExchange price updates fail without it
PORT=50051             # Optional; defaults to 50051
```

---

### Price Service (gRPC)

**Location**: `apps/grpc/price_service`  
**Port**: 50051

Responsibilities:
- Read/query the price database (daily, hourly, weekly OHLCV)
- Stale ticker scanning (which tickers are behind on data)
- **Scanner**: run condition-based scans across the full stock universe
- On-demand price updates (server-streaming with progress events)
- Stock metadata access (for filters and the Stock Database UI)

**Key handlers** (`handlers.go`, `scan.go`, `updater.go`, `fmp.go`):
| RPC | Description |
|-----|-------------|
| `GetDatabaseStats` | Record counts and date ranges per timeframe |
| `LoadPriceData` | Paginated OHLCV data for UI charting |
| `ScanStaleDaily/Weekly/Hourly` | Find tickers missing recent bars |
| `GetHourlyDataQuality` | Freshness + gap metrics for hourly data |
| `RunScan` | Evaluate indicator conditions across the universe |
| `UpdatePrices` | Server-streaming FMP price update with progress events |
| `GetFullStockMetadata` | Full stock_metadata rows including ETF fields |

**Required env**:
```
DATABASE_URL=postgresql://...
FMP_API_KEY=...        # Optional; UpdatePrices fails without it
PORT=50051
```

---

### Discord Service (gRPC)

**Location**: `apps/grpc/discord_service`  
**Port**: 50051

Manages Discord webhook configuration stored in the database (`app_documents` table as JSON blobs). Supports three independent channel sets: **hourly**, **daily**, and **weekly**.

Each channel set maps RBICS economy categories (Technology, Energy, Financials, etc.) plus special keys (ETFs, Futures, Default) to Discord webhook URLs.

**Key RPCs**:
| RPC | Description |
|-----|-------------|
| `GetHourlyDiscordConfig` | Returns all channel → webhook mappings for hourly |
| `UpdateHourlyChannelWebhook` | Set/update a webhook URL for a channel |
| `CopyDailyToHourly` | Copy daily webhook config to hourly config |
| `ResolveHourlyChannelForTicker` | Look up which channel a ticker routes to |
| `SendHourlyTestMessage` | Send a test Discord message to verify config |

Daily and Weekly have the same RPC shape (same pattern, different document key).

**Required env**:
```
DATABASE_URL=postgresql://...
PORT=50051
```

---

### Scheduler Service (gRPC)

**Location**: `apps/grpc/scheduler_service`  
**Port**: 50051

Provides a monitoring and control interface over the Asynq task queue via the Asynq Inspector API. Does **not** enqueue tasks itself — that is done by the Worker process.

**Key RPCs**:
| RPC | Description |
|-----|-------------|
| `GetSchedulerStatus` | Heartbeat, current job, queue stats, worker counts |
| `GetExchangeSchedule` | Per-exchange scheduled run times, time remaining |
| `StartScheduler` | Unpause the Asynq queue (resumes task processing) |
| `StopScheduler` | Pause the Asynq queue (workers stop picking up tasks) |
| `RunExchangeJob` | Manually enqueue a job for a specific exchange/timeframe |
| `ListQueueTasks` | List pending/scheduled/active tasks in the queue |

**Required env**:
```
DATABASE_URL=postgresql://...
REDIS_ADDR=redis:6379
PORT=50051
```

---

### Worker / Scheduler Process

**Location**: `apps/scheduler/cmd/worker/main.go`  
**Docker service**: `worker`

This is the core automated processing binary. It serves two responsibilities in one process:

#### 1. Background Scheduler (goroutine)

`apps/scheduler/internal/schedule/scheduler.go`

- Runs `scheduleAll()` on startup and then every **15 minutes**
- For each exchange in `calendar.ExchangeSchedules`:
  - Enqueues a `task:daily` task via Asynq using `ProcessAt(nextDailyRunTime)` + `Unique(12h)`
  - On Fridays, also enqueues a `task:weekly` at the same time
  - If the exchange is currently open (checked via FMP or local calendar), enqueues a `task:hourly` at 20 minutes after the next candle close
- `Unique` deduplication means re-running `scheduleAll` won't create duplicate tasks within the uniqueness window

**Candle timing for hourly**:
1. If `FMP_API_KEY` is set, probes the last real candle timestamp from FMP to determine when the next candle closes
2. Falls back to arithmetic using `calendar.HourlyAlignment` (hourly/half-hourly/quarter-hourly per exchange) and the exchange's open minute (e.g. NYSE opens at :30 → candles at :30 past)

#### 2. Asynq Worker

- Registers handlers for `task:daily`, `task:weekly`, `task:hourly`
- Picks up tasks from Redis queue and executes `handler.Common.Execute(exchange, timeframe)`

**`Execute` flow** (in `handler/common.go`):
1. **Update prices**: Call FMP API to fetch recent OHLCV bars and upsert into `daily_prices`/`weekly_prices`/`hourly_prices`
2. **Load alerts**: Query `alerts` table for this exchange + timeframe
3. **Pre-warm cache**: Batch-load price data into memory for all tickers in scope
4. **Evaluate alerts**: For each alert, run the condition expression through `expr.Evaluator` with `indicator.Registry`
5. **Discord notifications**: Triggered alerts are formatted as Discord embeds and batched via `discord.Accumulator`, then flushed via `discord.Notifier`
6. **Audit logging**: Every alert evaluation writes a row to `alert_audits`
7. **Status update**: Write job start/end to `app_documents` for scheduler status dashboard

**Worker config env**:
```
DATABASE_URL=postgresql://...
REDIS_ADDR=redis:6379
FMP_API_KEY=...
DISCORD_WEBHOOK_DAILY=https://discord.com/api/webhooks/...
DISCORD_WEBHOOK_WEEKLY=https://discord.com/api/webhooks/...
DISCORD_WEBHOOK_HOURLY=https://discord.com/api/webhooks/...
SCHEDULER_JOB_TIMEOUT=900           # seconds, default 900
SCHEDULER_CONCURRENCY=2             # parallel jobs per process, default 2
SCHEDULER_FMP_DAILY_CONCURRENCY=10  # parallel FMP API calls during daily update
SCHEDULER_FMP_WEEKLY_CONCURRENCY=10
SCHEDULER_FMP_HOURLY_CONCURRENCY=10
SCHEDULER_SHADOW_MODE=false         # write trigger results to file for comparison
```

---

### Envoy Proxy

**Location**: `envoy.yaml`  
**Port**: 8081 (only port exposed to the web service and host)

Routes gRPC-Web requests from the Next.js frontend to the appropriate backend gRPC service based on URL path prefix:

| Path prefix | Backend service | Timeout |
|-------------|-----------------|---------|
| `/stockalert.alert.v1.AlertService/` | `alert_service:50051` | 30s |
| `/stockalert.discord.v1.DiscordConfigService/` | `discord_service:50051` | 30s |
| `/stockalert.price.v1.PriceService/` | `price_service:50051` | 600s |
| `/stockalert.scheduler.v1.SchedulerService/` | `scheduler_service:50051` | 30s |

The longer timeout for `PriceService` accommodates the streaming `UpdatePrices` RPC that can take several minutes for a full universe update.

---

## Database Schema

**PostgreSQL 16** — bootstrapped from `db/postgres_schema.sql` on first Docker start.

| Table | Description |
|-------|-------------|
| `daily_prices` | Daily OHLCV bars. PK: `(ticker, date)` |
| `hourly_prices` | Hourly OHLCV bars. PK: `(ticker, datetime)` |
| `weekly_prices` | Weekly OHLCV bars (Friday close). PK: `(ticker, week_ending)` |
| `ticker_metadata` | Per-ticker first/last date and exchange info |
| `daily_move_stats` | Z-score and sigma-level daily move stats |
| `alert_audits` | Every alert evaluation is logged here. PK: `id` (bigserial) |
| `continuous_prices` | Futures continuous contract data |
| `futures_metadata` | Futures contract specs |
| `stock_metadata` | Equity/ETF metadata with RBICS classification and ETF fields |
| `alerts` | User-defined alert configurations. PK: `alert_id` (UUID) |
| `portfolios` | Portfolio groups. PK: `id` |
| `portfolio_stocks` | Portfolio → ticker membership. PK: `(portfolio_id, ticker)` |
| `app_documents` | Generic JSON document store (Discord configs, scheduler status). PK: `document_key` |

**Important indexes**:
- `idx_alert_audits_ticker_ts` — critical for alert history queries
- `idx_alerts_ticker`, `idx_alerts_exchange`, `idx_alerts_ratio` — alert lookup during evaluation
- `idx_stock_metadata_exchange`, `idx_stock_metadata_country` — stock database filters

**sqlc** generates the Go query layer. Schema source of truth is `database/sql/schema.sql`. Queries are in `database/sql/queries/`. Re-generate after any schema or query change:
```bash
cd database
sqlc generate
```

---

## Alert Evaluation Flow

```
User defines alert in UI
        │
        ▼
CreateAlert RPC → INSERT INTO alerts
        │
        ▼
Worker picks up task:daily/weekly/hourly
(scheduled at exchange market close)
        │
        ▼
handler.Common.Execute(exchange, timeframe)
        │
        ├── 1. FMP API → bulk fetch OHLCV → UPSERT daily/weekly/hourly_prices
        │
        ├── 2. SELECT * FROM alerts WHERE exchange = ? AND timeframe = ?
        │
        ├── 3. PreWarmCache: bulk SELECT price history for all tickers
        │
        └── 4. For each alert:
                │
                ├── Parse conditions JSON → []Condition{indicator, op, value, ...}
                │
                ├── For each condition:
                │   expr.Evaluator.Eval(condition, ohlcv)
                │   └── indicator.Registry.Get(name)(ohlcv, params)
                │       └── Returns []float64 series (same length as price data)
                │
                ├── Apply combination_logic ("AND" / "OR" / "1 AND (2 OR 3)")
                │
                ├── If triggered:
                │   ├── discord.Router.ResolveWebhookURL(ticker, timeframe, exchange)
                │   ├── discord.FormatAlertEmbed(...)
                │   └── discord.Accumulator.Add(webhookURL, embed)
                │
                └── INSERT INTO alert_audits (result, timing, error, ...)

        └── 5. discord.Accumulator.FlushAll() — send batched Discord messages
```

### Condition Expression Format

Conditions are stored as JSONB in the `alerts.conditions` column. Each condition references an indicator by name with optional parameters:

```json
{
  "conditions": {
    "1": { "indicator": "rsi", "period": 14, "operator": "<", "value": 30 },
    "2": { "indicator": "ema", "period": 20, "operator": "crossover", "value": "ema(50)" }
  },
  "combination_logic": "1 AND 2"
}
```

The `expr` package parses these into an AST and the `evaluator.go` resolves indicator calls recursively (nested indicators like `sma(input=rsi(14))` are supported).

---

## Scheduler & Task Queue Flow

```
Scheduler goroutine (every 15 min)
        │
        ├── FMP call: is each exchange currently open?
        │   (fallback: calendar.IsExchangeOpen)
        │
        ├── For each exchange:
        │   ├── asynq.Enqueue(task:daily, ProcessAt=nextDailyRunTime, Unique=12h)
        │   ├── If nextDailyRunTime is Friday:
        │   │   asynq.Enqueue(task:weekly, same ProcessAt)
        │   └── If exchange is open:
        │       nextHourly = lastCandleEnd + 20min (FMP) or arithmetic
        │       asynq.Enqueue(task:hourly, ProcessAt=nextHourly, Unique=1h)
        │
        └── Log cycle summary (scheduled_daily, scheduled_weekly, scheduled_hourly)

Redis (Asynq queue) — tasks sit in "scheduled" state until ProcessAt time
        │
        ▼ (at ProcessAt time)
Asynq Server (worker)
        ├── DailyHandler.ProcessTask  → Common.Execute(exchange, "daily")
        ├── WeeklyHandler.ProcessTask → Common.Execute(exchange, "weekly")
        └── HourlyHandler.ProcessTask → Common.Execute(exchange, "hourly")
```

**Queue control** (via Scheduler UI page or SchedulerService gRPC):
- **Stop**: Pauses the Asynq queue — workers stop picking up tasks; tasks remain in queue
- **Start**: Unpauses the queue — workers resume
- **RunExchangeJob**: Manually enqueue any exchange/timeframe for immediate processing

**Utility CLIs** (in `apps/scheduler/cmd/`):
```bash
go run ./apps/scheduler/cmd/listqueue   # Show all tasks in queue
go run ./apps/scheduler/cmd/resetqueue  # Clear all tasks from queue
```

---

## Technical Indicator Library

**Location**: `indicator/`

The indicator library is a pure Go module with 50+ technical indicators. All indicators implement the same `IndicatorFunc` signature:

```go
type IndicatorFunc func(data *OHLCV, params map[string]interface{}) ([]float64, error)
```

All indicators are registered in `indicator/registry.go` via `NewDefaultRegistry()`. The registry is case-insensitive and is instantiated once in both the `alert_service` and the worker.

**Indicator categories**:

| Category | Indicators |
|----------|-----------|
| Basic (TA-Lib wrappers) | `sma`, `ema`, `rsi`, `volume_ratio`, `roc`, `atr`, `cci`, `willr` |
| Trend / Momentum | `adx`, `plus_di`, `minus_di`, `aroon_osc`, `cmo`, `mom`, `macd`, `macd_line`, `macd_signal`, `macd_histogram` |
| Volatility | `bbands`, `natr`, `stddev`, `atr` |
| Volume | `obv`, `mfi`, `ad`, `obv_macd`, `obv_macd_signal` |
| Oscillators | `stoch_k`, `stoch_d`, `stoch_rsi_k`, `stoch_rsi_d` |
| Regression | `linear_reg`, `linear_reg_slope` |
| Adaptive MAs | `hma`, `frama`, `kama` |
| Composite | `ewo`, `ma_spread_zscore`, `zscore`, `harsi_flip` |
| Slope + Curvature | `slope_sma`, `slope_ema`, `slope_hma`, `ma_slope_curve_*` (9 variants) |
| Supertrend | `supertrend`, `supertrend_upper`, `supertrend_lower` |
| Ichimoku | `ichimoku_conversion`, `ichimoku_base`, `ichimoku_span_a/b`, `ichimoku_lagging`, `ichimoku_cloud_top/bottom/signal` |
| Donchian | `donchian_upper`, `donchian_lower`, `donchian_basis`, `donchian_width`, `donchian_position` |
| Trend Magic | `trend_magic`, `trend_magic_signal` |
| Kalman | `kalman_roc_stoch`, `kalman_roc_stoch_signal`, `kalman_roc_stoch_crossover` |
| Pivot S/R | `pivot_sr`, `pivot_sr_crossover`, `pivot_sr_proximity` |
| SAR | `sar` |
| Custom | `my_smoothed_rsi` |

The `indicatorList.ts` in the UI must be kept in sync with `indicator/registry.go` when adding new indicators.

---

## Proto / Code Generation

**Tool**: [Buf](https://buf.build/)  
**Config**: `buf.gen.yaml`, `buf.yaml`

Proto source files are in `proto/`. Generated code goes to `gen/go/` (Go) and `gen/ts/` (TypeScript). Never edit generated files.

**To regenerate after proto changes**:
```bash
buf generate
```

**Plugins used**:
- `protoc-gen-go` — Go message structs
- `buf.build/grpc/go` — Go gRPC server/client stubs
- `protoc-gen-ts_proto` — TypeScript with `nice-grpc` output

**Proto packages**:
| Proto file | Go package | TypeScript |
|-----------|-----------|-----------|
| `proto/alert/v1/alert.proto` | `stockalert/gen/go/alert/v1` | `gen/ts/alert/v1/alert.ts` |
| `proto/price/v1/price.proto` | `stockalert/gen/go/price/v1` | `gen/ts/price/v1/price.ts` |
| `proto/discord/v1/discord.proto` | `stockalert/gen/go/discord/v1` | `gen/ts/discord/v1/discord.ts` |
| `proto/scheduler/v1/scheduler.proto` | `stockalert/gen/go/scheduler/v1` | `gen/ts/scheduler/v1/scheduler.ts` |

---

## Environment Variables

Copy `.env.example` to `.env` and fill in values. The `.env` file is loaded by Docker Compose for all services.

| Variable | Used by | Description |
|----------|---------|-------------|
| `POSTGRES_USER` | Docker Compose | PostgreSQL username |
| `POSTGRES_PASSWORD` | Docker Compose | PostgreSQL password (**required**) |
| `POSTGRES_DB` | Docker Compose | PostgreSQL database name |
| `DATABASE_URL` | All Go services | Full PostgreSQL connection string |
| `REDIS_ADDR` | Worker, scheduler_service | Redis address (default: `localhost:6379`) |
| `FMP_API_KEY` | alert_service, price_service, worker | Financial Modeling Prep API key |
| `GRPC_ENDPOINT` | Next.js web | Envoy proxy address (default: `127.0.0.1:8081`) |
| `DISCORD_WEBHOOK_DAILY` | Worker | Discord webhook for daily job status |
| `DISCORD_WEBHOOK_WEEKLY` | Worker | Discord webhook for weekly job status |
| `DISCORD_WEBHOOK_HOURLY` | Worker | Discord webhook for hourly job status |
| `SCHEDULER_JOB_TIMEOUT` | Worker | Per-job timeout in seconds (default: 900) |
| `SCHEDULER_CONCURRENCY` | Worker | Parallel jobs per worker process (default: 2) |
| `SCHEDULER_FMP_DAILY_CONCURRENCY` | Worker | Parallel FMP API calls during daily update (default: 10) |
| `SCHEDULER_FMP_WEEKLY_CONCURRENCY` | Worker | Parallel FMP API calls during weekly update (default: 10) |
| `SCHEDULER_FMP_HOURLY_CONCURRENCY` | Worker | Parallel FMP API calls during hourly update (default: 10) |
| `SCHEDULER_SHADOW_MODE` | Worker | Write results to file instead of Discord (default: false) |

---

## Local Development Setup

### Prerequisites

Install every tool below before running anything in this repo. All commands are for **PowerShell on Windows**.

---

#### Go 1.24+

The workspace `go.work` requires Go 1.24.

1. Download the installer from https://go.dev/dl/ and run it.
2. Verify:
   ```powershell
   go version
   # go version go1.24.x windows/amd64
   ```

After installing Go, add the Go bin directory to your PATH so `go install`-ed tools are available:
```powershell
# Add to your PowerShell profile ($PROFILE) or System Environment Variables
$env:PATH += ";$env:USERPROFILE\go\bin"
```

---

#### Node.js 20+

1. Download and install from https://nodejs.org/en/download (LTS recommended).
2. Verify:
   ```powershell
   node --version
   # v20.x.x or higher
   ```

---

#### pnpm

The frontend workspace uses pnpm (not npm or yarn).

```powershell
corepack enable
corepack prepare pnpm@latest --activate
```

Or via npm:
```powershell
npm install -g pnpm
```

Verify:
```powershell
pnpm --version
```

---

#### Docker Desktop

Required to run PostgreSQL, Redis, Envoy, and the full stack via Docker Compose.

1. Download from https://www.docker.com/products/docker-desktop/
2. Install and start Docker Desktop.
3. Verify:
   ```powershell
   docker --version
   docker compose version
   ```

---

#### Python 3.13 + uv

Python is used for the legacy Streamlit app and all scripts in `scripts/`. The project requires Python 3.13 exactly (see `.python-version`).

Install **uv** (the package manager used by this project):
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then install Python 3.13 and create the virtual environment:
```powershell
uv python install 3.13
uv sync
```

Verify:
```powershell
uv run python --version
# Python 3.13.x
```

**TA-Lib C library** (required by `ta-lib` Python package):

The `ta-lib` Python package wraps a C library that must be installed separately on Windows.

1. Download the pre-built Windows wheel from https://github.com/cgohlke/talib-build/releases — pick the `.whl` matching your Python version and architecture (e.g. `TA_Lib-0.6.x-cp313-cp313-win_amd64.whl`).
2. Install it directly:
   ```powershell
   uv pip install TA_Lib-0.6.x-cp313-cp313-win_amd64.whl
   ```

---

#### Buf CLI

Used to regenerate Go and TypeScript code from `.proto` files.

```powershell
go install github.com/bufbuild/buf/cmd/buf@latest
```

Verify:
```powershell
buf --version
```

---

#### protoc-gen-go (Go protobuf plugin)

Required by `buf generate` to produce Go message structs.

```powershell
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
```

Verify:
```powershell
protoc-gen-go --version
```

---

#### protoc-gen-ts_proto (TypeScript protobuf plugin)

Required by `buf generate` to produce TypeScript stubs. It is installed as a local dev dependency of the frontend and must be available on `PATH` for `buf` to find it.

```powershell
cd apps\ui\web
pnpm install
# Add the local bin directory to PATH for this session:
$env:PATH = "$(Resolve-Path node_modules\.bin);$env:PATH"
cd ..\..\..
```

To make this permanent, add the absolute path to `apps/ui/web/node_modules/.bin` to your system `PATH`, or run the `$env:PATH` line before any `buf generate` invocation.

---

#### sqlc

Used to regenerate the Go database query layer from SQL files.

```powershell
go install github.com/sqlc-dev/sqlc/cmd/sqlc@latest
```

Verify:
```powershell
sqlc version
```

---

#### grpc_health_probe (optional — Docker only)

The Docker health checks use `grpc_health_probe` inside the containers. It is downloaded automatically during Docker builds via the Dockerfiles — you do **not** need to install it on your machine.

---

### Verify everything is installed

Run this to confirm all required tools are on your PATH:

```powershell
go version          # go1.24+
node --version      # v20+
pnpm --version      # 9+
docker --version
buf --version
sqlc version
protoc-gen-go --version
```

---

### 1. Start infrastructure

```bash
# Start PostgreSQL and Redis only
docker compose up postgres redis -d
```

PostgreSQL is available on `localhost:5433` (host port mapped from container 5432).  
Redis is available on `localhost:6378` (host port mapped from container 6379).

### 2. Set up environment

```bash
cp .env.example .env
# Edit .env — at minimum set POSTGRES_PASSWORD and FMP_API_KEY
```

### 3. Start Go services

Each service is a standalone Go binary. Open separate terminals:

```bash
# Alert service
cd apps/grpc/alert_service
go run . 

# Price service
cd apps/grpc/price_service
go run .

# Discord service
cd apps/grpc/discord_service
go run .

# Scheduler service (needs Redis)
cd apps/grpc/scheduler_service
DATABASE_URL=... REDIS_ADDR=localhost:6378 go run .

# Worker (needs Redis; runs scheduler + worker)
cd apps/scheduler
DATABASE_URL=... REDIS_ADDR=localhost:6378 FMP_API_KEY=... go run ./cmd/worker
```

> All services read `DATABASE_URL` from env. For local dev use port **5433** (the Docker host mapping):
> `DATABASE_URL=postgresql://stockalertapp:password@localhost:5433/stockalertapp`

### 4. Start Envoy proxy

```bash
docker compose up envoy -d
```

### 5. Start the Next.js frontend

```bash
cd apps/ui/web
pnpm install
pnpm dev
```

Open http://localhost:3000

---

## Running with Docker Compose

The `docker-compose.yml` defines the full production stack.

```bash
# First run — copy and configure env
cp .env.example .env
# Edit .env (set POSTGRES_PASSWORD, FMP_API_KEY, Discord webhooks)

# Build and start everything
docker compose up --build -d

# View logs
docker compose logs -f worker
docker compose logs -f alert_service

# Stop
docker compose down
```

**Service ports exposed to host**:
| Service | Host port | Container port |
|---------|-----------|----------------|
| `web` (Next.js) | 8501 | 3000 |
| `envoy` | 8081 | 8081 |
| `postgres` | 5433 | 5432 |
| `redis` | 6378 | 6379 |

**Health checks**: All gRPC services use `grpc_health_probe`. The `autoheal` sidecar automatically restarts unhealthy containers.

**Startup order**: `postgres` → `alert_service`, `price_service`, `discord_service` → `envoy` → `web`; `redis` → `worker`, `scheduler_service`

---

## Go Workspace

The root `go.work` ties all Go modules together so they can reference each other with local paths. This is critical — shared libraries (`alert`, `calendar`, `database`, `discord`, `expr`, `indicator`) are referenced by all service `go.mod` files as `stockalert/<module>`.

```
go.work
├── alert/go.mod          (module stockalert/alert)
├── calendar/go.mod       (module stockalert/calendar)
├── database/go.mod       (module stockalert/database)
├── discord/go.mod        (module stockalert/discord)
├── expr/go.mod           (module stockalert/expr)
├── gen/go/go.mod         (module stockalert/gen/go)
├── indicator/go.mod      (module stockalert/indicator)
├── apps/grpc/alert_service/go.mod
├── apps/grpc/price_service/go.mod
├── apps/grpc/discord_service/go.mod
├── apps/grpc/scheduler_service/go.mod
└── apps/scheduler/go.mod
```

When adding a new Go dependency to any module, run `go get <package>` from within that module's directory. Then run `go work sync` from the root if needed.

---

## UI Pages Reference

| Route | Page | gRPC calls |
|-------|------|-----------|
| `/` | Dashboard | `GetDashboardStats`, `GetTriggerCountByDay`, `GetTopTriggeredAlerts` |
| `/alerts` | Alert list | `ListAlerts`, `SearchAlertsStream` |
| `/alerts/add` | Add alert | `CreateAlert`, `SearchStocks` |
| `/alerts/delete` | Bulk delete | `ListAlerts`, `SearchAlertsStream`, `BulkDeleteAlerts` |
| `/alerts/history` | Trigger history | `GetTriggerHistoryByTicker`, `SearchStocks`, `ListPortfolios` |
| `/alerts/audit` | Audit logs | `GetAuditSummary`, `GetPerformanceMetrics`, `GetAuditLog`, `GetFailedPriceData`, `ClearAuditData` |
| `/scanner` | Stock scanner | `RunScan`, `GetFullStockMetadata` |
| `/price-database` | Price data viewer | `GetDatabaseStats`, `LoadPriceData`, `ScanStaleDaily/Weekly/Hourly`, `UpdatePrices` |
| `/database/stock` | Stock database | `GetFullStockMetadata` |
| `/portfolios` | Portfolio manager | `ListPortfolios`, `CreatePortfolio`, `UpdatePortfolio`, `DeletePortfolio`, `AddStocksToPortfolio`, `RemoveStocksFromPortfolio`, `SearchStocks` |
| `/discord/daily` | Daily Discord config | `GetDailyDiscordConfig`, `UpdateDailyChannelWebhook`, `CopyBaseToDaily`, `SendDailyTestMessage` |
| `/discord/hourly` | Hourly Discord config | `GetHourlyDiscordConfig`, `UpdateHourlyChannelWebhook`, `CopyDailyToHourly`, `SendHourlyTestMessage` |
| `/discord/weekly` | Weekly Discord config | `GetWeeklyDiscordConfig`, `UpdateWeeklyChannelWebhook`, `CopyBaseToWeekly`, `SendWeeklyTestMessage` |
| `/scheduler` | Scheduler control | `GetSchedulerStatus`, `GetExchangeSchedule`, `StartScheduler`, `StopScheduler`, `RunExchangeJob`, `ListQueueTasks` |

---

## Discord Notification Routing

Discord notifications are routed by RBICS economy category. When an alert triggers:

1. `discord.Router.ResolveWebhookURL(ticker, timeframe, exchange, isRatio)` is called
2. The router looks up the ticker's `rbics_economy` from `stock_metadata`
3. The economy name maps to a Discord channel webhook URL from `app_documents` (stored as JSON)
4. Special routing rules: ETFs go to "ETFs" channel, ratios/futures go to "Futures" channel, unknown → "Default"

Each timeframe (daily/hourly/weekly) has its own independent channel config. The Discord Config pages in the UI allow managing these per-channel webhook URLs.

**`discord.Accumulator`**: Alert embeds are batched per webhook URL to avoid sending dozens of individual Discord messages per evaluation cycle. All embeds for one webhook are collected then sent as a single request.

---

## Adding a New Indicator

1. **Implement the function** in `indicator/` (add to an existing file or create a new one):
   ```go
   func MyIndicator(data *OHLCV, params map[string]interface{}) ([]float64, error) {
       period := paramInt(params, "period", 14)
       // ... compute and return []float64 same length as data.Close
   }
   ```

2. **Register it** in `indicator/registry.go` inside `NewDefaultRegistry()`:
   ```go
   r.Register("my_indicator", MyIndicator)
   ```

3. **Add it to the UI list** in `apps/ui/web/app/alerts/add/_components/indicatorList.ts`:
   ```ts
   export const INDICATOR_NAMES: string[] = [
     // ... existing entries (keep alphabetical)
     "my_indicator",
   ]
   ```

4. Optionally add documentation to `apps/ui/web/app/alerts/add/_components/IndicatorGuide.tsx`

---

## Adding a New gRPC Method

1. **Define the RPC** in the appropriate proto file under `proto/`:
   ```protobuf
   message MyNewRequest { string param = 1; }
   message MyNewResponse { string result = 1; }
   
   service AlertService {
     // ... existing RPCs
     rpc MyNew(MyNewRequest) returns (MyNewResponse);
   }
   ```

2. **Regenerate code**:
   ```bash
   buf generate
   ```

3. **Implement the handler** in the Go service (e.g. `apps/grpc/alert_service/`). The generated interface will require your new method.

4. **Add a Server Action** in `apps/ui/web/actions/`:
   ```ts
   "use server";
   export async function myNewAction(param: string): Promise<string> {
     const response = await alertClient.myNew({ param });
     return response.result;
   }
   ```

5. **Use in UI** via a hook (`lib/hooks/`) or direct call from a Server Component/Action.

---

## Scripts Reference

All scripts are in `scripts/` organized by category. They require the Python virtualenv to be active.

### Analysis (`scripts/analysis/`)

| Script | Purpose |
|--------|---------|
| `analyze_alerts.py` | Alert performance analysis and reporting |
| `check_alert_routing.py` | Verify Discord routing decisions for tickers |
| `check_scheduler_status.py` | Check Go scheduler status from Redis |
| `check_hourly_data.py` | Inspect hourly price data coverage |
| `check_hourly_failures.py` | Find exchanges with high hourly failure rates |
| `test_alert_trigger.py` | Manually trigger alert evaluation for debugging |
| `test_condition_logic.py` | Test condition expression parsing |
| `view_scheduler_logs.py` | Print formatted scheduler logs |

### Maintenance (`scripts/maintenance/`)

| Script | Purpose |
|--------|---------|
| `daily_full_update.py` | Trigger a full daily price update |
| `manage_alert_status.py` | Bulk enable/disable alerts |
| `run_scheduled_price_update.py` | Manual scheduled price update |
| `run_scheduler_watchdog.py` | Start the Python scheduler watchdog |

### Migration (`scripts/migration/`)

| Script | Purpose |
|--------|---------|
| `import_sqlite_to_postgres.py` | One-time SQLite → PostgreSQL migration |
| `load_json_to_postgres.py` | Load legacy JSON data into PostgreSQL |

---

## Testing

### Go tests

Run from the workspace root or any module directory:

```bash
# Unit tests for the indicator library
cd indicator && go test ./...

# Expression evaluator tests
cd expr && go test ./...

# Alert checker tests
cd alert && go test ./...

# Scheduler integration tests (requires DATABASE_URL and REDIS_ADDR)
cd apps/scheduler && go test ./internal/handler/... -tags integration
```

### Python tests (legacy)

```bash
# From project root with .venv active
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

### Shadow mode

The worker supports a shadow mode (`SCHEDULER_SHADOW_MODE=true`) that writes alert evaluation results to files instead of (or in addition to) Discord. This is used to compare Go evaluation results against the legacy Python evaluator:

```bash
SCHEDULER_SHADOW_MODE=true SCHEDULER_SHADOW_OUTPUT_DIR=./shadow_results go run ./apps/scheduler/cmd/worker
# Then compare:
python scripts/analysis/compare_shadow_results.py
```

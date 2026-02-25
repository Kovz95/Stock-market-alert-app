# Migration Plan: Streamlit → Next.js + gRPC Microservices + sqlc

This document outlines a phased plan to migrate the Stock Market Alert app from a Python Streamlit monolith to a **Next.js** frontend with **Shadcn UI**, a **gRPC backend** composed of microservices behind an **Envoy gateway**, **Protocol Buffers** for API contracts, and **sqlc** for type-safe database access in Go services.

The target layout follows the **DCW reference** (see `GRPC_MICROSERVICE_EXAMPLE.md`): **pnpm workspaces** at the repo root, a **Go workspace** (`go.work`) at the root, **docker-compose** for the local stack, and a single shared **database** layer with sqlc used by all Go services.

---

## Quick Reference (Target Setup)

```bash
# Start infrastructure (Postgres, Envoy, Redis, optional Jaeger)
docker-compose up -d

# Install frontend deps and run Next.js
pnpm install && pnpm dev

# Generate protobuf code (Go + TypeScript)
buf generate

# Generate SQL code (shared database package)
sqlc generate -f database/sqlc.yaml

# View traces (if Jaeger enabled)
open http://localhost:16686
```

---

## 1. Current State Summary

### 1.1 Application Structure

| Layer | Technology | Key Locations |
|-------|------------|---------------|
| UI | Streamlit | `Home.py`, `pages/*.py` |
| Data access | Raw SQL + pandas, SQLAlchemy | `src/data_access/*.py` |
| Business logic | Python services | `src/services/*.py` |
| Database | PostgreSQL | `db/postgres_schema.sql`, `DATABASE_URL` |
| Cache | Redis | `src/data_access/redis_support.py` |

### 1.2 Pages (→ Next.js Routes)

| Streamlit Page | Purpose | Primary Data |
|----------------|---------|--------------|
| Home | Dashboard, scheduler status, quick links | Alerts, document store, Redis |
| Add_Alert | Create/edit alerts | Alerts, metadata, FMP (prices) |
| Delete_Alert | Remove alerts | Alerts |
| Alert_History | Alert list, last triggered | Alerts |
| Scanner | Scan symbols against conditions | Alerts, daily/hourly prices, metadata |
| Price_Database | Daily/weekly/hourly price data | daily_prices, weekly_prices, hourly_prices |
| Stock_Database | Stock metadata | stock_metadata |
| My_Portfolio | Portfolios and stocks | portfolios, portfolio_stocks |
| Discord_Management (Daily/Weekly/Hourly) | Discord webhook config per schedule | Document store, portfolios |
| Daily_Weekly_Scheduler_Status | Scheduler status | Document store, Redis |
| Hourly_Scheduler_Status | Hourly scheduler status | Document store, Redis |
| Market_Hours | Exchange calendar | Config / external |
| Daily_Move_Tracker | Move stats (sigma, zscore) | daily_move_stats, daily_prices |
| Alert_Audit_Logs | Audit trail for alert runs | alert_audits |

### 1.3 Data Access Surface (→ sqlc + gRPC)

- **alerts** – CRUD, list, cache invalidation (Redis)
- **portfolios** + **portfolio_stocks** – CRUD, list
- **daily_prices**, **weekly_prices**, **hourly_prices** – read/upsert, stats
- **ticker_metadata** – read/upsert
- **stock_metadata** – read (and possibly sync from FMP)
- **app_documents** – key/value docs (scheduler status, config)
- **alert_audits** – insert, query by time/ticker
- **daily_move_stats** – read/write for tracker
- **continuous_prices**, **futures_metadata** – futures feature

### 1.4 External Integrations

- **FMP API** – price and metadata; **owned by price-svc** in the new architecture (currently used by backend_fmp, smart_price_fetcher, etc.)
- **Discord** – webhooks for alerts and portfolios (discord_routing, discord_logger)
- **Redis** – caching (alerts list, portfolios, document cache)
- **APScheduler** – daily/weekly/hourly jobs (scheduler_discord, hourly_scheduler_discord, etc.)

---

## 2. Target Architecture

### 2.1 High-Level Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Frontend (Next.js + Shadcn)                   │
│                      http://localhost:3000                       │
└───────────────────────────────┬─────────────────────────────────┘
                                │ gRPC-Web (e.g. nice-grpc)
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Envoy Proxy (Port 80)                       │
│              Frontend → Backend ingress only                     │
└───┬─────────────┬─────────────┬─────────────┬─────────────┬──────┘
    │             │             │             │             │
    ▼             ▼             ▼             ▼             ▼
┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
│ Alert   │ │ Price   │ │Portfolio │ │Metadata │ │Document │
│ Service │ │ Service │ │ Service  │ │ Service │ │ Service │
│ :50051  │ │ :50051  │ │ :50051   │ │ :50051  │ │ :50051  │
└────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
     │  ◄──── direct gRPC between services ────►  │
     │           │           │           │           │
     └───────────┴───────────┴───────────┴───────────┘
                              │
                    ┌─────────┴─────────┐       ┌───────────┐
                    │    PostgreSQL     │       │   FMP API  │
                    │  (shared schema   │       │ (owned by  │
                    │   database/)      │       │ price-svc) │
                    └───────────────────┘       └───────────┘
```

### 2.2 Root-Level Tooling (pnpm + Go workspace + Docker)

- **pnpm workspaces** at the repo root (`pnpm-workspace.yaml`). All Node/Next.js apps live under `apps/` (e.g. `apps/ui/web`). Use `pnpm install` at root and `pnpm --filter=web dev` or `pnpm dev` from the web app.
- **Go workspace** at the repo root (`go.work`). All Go microservices live under `apps/grpc/<service_name>/` (e.g. `apps/grpc/alert_service/`). Shared Go code lives in `core/` (mappers, tracing). Each service has its own `go.mod`; the root `go.work` includes the app and core modules.
- **docker-compose** at the root for local development: PostgreSQL, Envoy, Redis, and optionally Jaeger. Run `docker-compose up -d` before starting frontend or gRPC services.

### 2.3 Design Decisions

- **Next.js** as the only UI; **server actions** call the backend via **gRPC-Web** (e.g. **nice-grpc**) to Envoy on port 80. Generated TypeScript client from the same `.proto` files as Go.
- **Envoy** as single entry point for **frontend → backend** traffic: gRPC-Web/HTTP to the browser or Next.js server; routes by **service prefix** (e.g. `/stockalert.alert.v1.AlertService/`) to the corresponding Go service.
- **Microservices in Go** under `apps/grpc/`; they share one **database** package generated by **sqlc** from `database/sql/` (single schema + query files). Each service only uses the queries it needs. All services listen on **`:50051`** — each runs in its own Docker container so there are no port collisions.
- **Inter-service communication**: Services call each other **directly via gRPC** (not through Envoy). In Docker, services are reachable by container name (e.g. `price_service:50051`). Each service that needs another creates a gRPC client at startup. For example, alert-svc may call price-svc to fetch current prices during alert evaluation. Envoy is only for external (frontend) ingress.
- **FMP API ownership**: **price-svc** owns all FMP integration — fetching daily/weekly/hourly prices and ticker metadata. Other services that need price data call price-svc via gRPC rather than calling FMP directly.
- **Health checks**: All Go services implement the **gRPC Health Checking Protocol** (`grpc.health.v1.Health`). Docker-compose uses `grpc_health_probe` or equivalent for container health checks. Envoy uses health checks on each cluster to route only to healthy backends.
- **Redis**: remains for caching; services that need it (e.g. alert_service) use Redis in the Go service layer and invalidate on write.
- **Testing**: Manual testing during initial build-out. Automated tests (Go unit tests, integration tests, frontend e2e) will be added once core functionality is working.
- **CI/CD**: Will be designed and implemented after the migration reaches functional parity.

### 2.4 Project Structure (Reference Layout)

Matches the pattern from `GRPC_MICROSERVICE_EXAMPLE.md`: pnpm workspaces, go.work, shared database, and Envoy at root.

```
stock-market-alert-app/
├── apps/
│   ├── grpc/                        # Go microservices
│   │   ├── alert_service/            # Alerts + alert_audits
│   │   ├── price_service/            # daily/weekly/hourly_prices, ticker_metadata
│   │   ├── portfolio_service/        # portfolios, portfolio_stocks
│   │   ├── metadata_service/         # stock_metadata
│   │   └── document_service/         # app_documents
│   └── ui/
│       └── web/                      # Next.js + Shadcn frontend
├── core/                             # Shared Go packages
│   ├── shared/                       # Proto ↔ DB mappers (e.g. enums)
│   └── tracing/                      # OpenTelemetry setup
├── database/
│   ├── sql/
│   │   ├── schema.sql                # Full PostgreSQL schema (from db/postgres_schema.sql)
│   │   └── queries/                  # sqlc query files per domain
│   │       ├── alert_queries.sql
│   │       ├── alert_audit_queries.sql
│   │       ├── price_queries.sql
│   │       ├── portfolio_queries.sql
│   │       ├── metadata_queries.sql
│   │       └── document_queries.sql
│   ├── generated/                   # Generated Go from sqlc (one package)
│   └── sqlc.yaml                     # sqlc config
├── gen/
│   ├── go/                           # Generated Go from protobuf (buf)
│   └── ts/                           # Generated TypeScript from protobuf (buf)
├── proto/                            # Protocol buffer definitions
│   ├── alert/v1/
│   ├── price/v1/
│   ├── portfolio/v1/
│   ├── metadata/v1/
│   ├── document/v1/
│   └── shared/v1/                    # Shared enums / types
├── docker-compose.yml                # Postgres, Envoy, Redis, optional Jaeger
├── envoy.yaml                        # API gateway (route by service prefix)
├── go.work                           # Go workspace (apps/grpc/*, core)
├── pnpm-workspace.yaml               # pnpm monorepo (apps/ui/web)
├── buf.yaml                          # Buf config
└── buf.gen.yaml                     # Buf codegen (go + ts)
```

---

## 3. Microservice Boundaries

### 3.1 Recommended Services

| Service | Responsibility | DB Tables (sqlc) | Proto Package |
|---------|----------------|------------------|---------------|
| **alert-svc** | Alerts CRUD, list, alert evaluation trigger (optional), audit log write | `alerts`, `alert_audits` | `stockalert.alert.v1` |
| **price-svc** | Daily/weekly/hourly prices, ticker_metadata, stats, batch reads, **FMP API integration** (price/metadata fetch) | `daily_prices`, `weekly_prices`, `hourly_prices`, `ticker_metadata` | `stockalert.price.v1` |
| **portfolio-svc** | Portfolios and portfolio_stocks | `portfolios`, `portfolio_stocks` | `stockalert.portfolio.v1` |
| **metadata-svc** | Stock metadata (read/sync from FMP or internal) | `stock_metadata` | `stockalert.metadata.v1` |
| **document-svc** | Key-value docs (scheduler status, config) | `app_documents` | `stockalert.document.v1` |
| **stats-svc** (optional) | Daily move stats (zscore, sigma) | `daily_move_stats` | `stockalert.stats.v1` |
| **scheduler-svc** (optional) | Job orchestration; can stay Python initially and call other svcs via gRPC | — | `stockalert.scheduler.v1` (status only) |

Start with **alert-svc**, **price-svc**, **portfolio-svc**, **document-svc**, and **metadata-svc**. Add **stats-svc** when you migrate Daily_Move_Tracker; keep **scheduler-svc** thin (e.g. status read from document-svc).

### 3.2 Shared Database (Single sqlc Package)

Following the reference layout: **one PostgreSQL instance** and **one shared sqlc project** at `database/`. The `database/sql/schema.sql` holds the full schema (migrated from `db/postgres_schema.sql`). Query files live in `database/sql/queries/` per domain (e.g. `alert_queries.sql`, `price_queries.sql`). A single `sqlc generate -f database/sqlc.yaml` produces one Go package in `database/generated/`. Each Go service in `apps/grpc/` imports this package and only calls the queries it needs (e.g. alert_service uses alert + alert_audit queries only). No per-service sqlc projects; one schema, one generated package, multiple services.

---

## 4. Protocol Buffers and gRPC

### 4.1 Repo Layout

Aligned with the reference: `proto/` by domain with `v1` packages; **buf** generates into `gen/go/` and `gen/ts/`. Root `buf.yaml` and `buf.gen.yaml` define lint and codegen.

- **proto/alert/v1/alert.proto** – `AlertService`, Alert messages, ListAlertsReq/Resp, CreateAlert, GetAlert, UpdateAlert, DeleteAlert.
- **proto/price/v1/price.proto** – `PriceService`, DailyPrice, GetDailyPrices, GetWeeklyPrices, GetHourlyPrices, GetStatistics, Upsert*.
- **proto/portfolio/v1/portfolio.proto** – `PortfolioService`, Portfolio, ListPortfolios, GetPortfolio, SavePortfolio, DeletePortfolio.
- **proto/metadata/v1/metadata.proto** – `MetadataService`, StockMetadata, ListStockMetadata, GetStockMetadata.
- **proto/document/v1/document.proto** – `DocumentService`, GetDocument, SaveDocument (key/value).
- **proto/shared/v1/** – shared enums or types if needed.

Run **`buf generate`** at root to produce `gen/go/` (for Go services) and `gen/ts/` (for Next.js). Frontend uses the TypeScript client (e.g. **nice-grpc** with generated types) to call Envoy.

### 4.2 Naming Conventions

- **Services**: `AlertService`, `PriceService`, `PortfolioService`, `MetadataService`, `DocumentService` (package names e.g. `alert.v1`, `price.v1`).
- **RPCs**: `ListAlerts`, `GetAlert`, `CreateAlert`, `UpdateAlert`, `DeleteAlert`; `GetDailyPrices`, `UpsertDailyPrices`; etc.
- **Messages**: `XxxRequest` / `XxxResponse`; use `google.protobuf.Timestamp` for times; avoid free-form JSON in proto where possible.

### 4.3 Next.js ↔ Backend Communication

- **gRPC-Web** from Next.js to Envoy (port 80). Use **nice-grpc** (or similar) with the generated TypeScript definitions so server actions call the backend with type-safe requests/responses.
- Flow: **Browser → Next.js (Server Action / fetch) → Next.js server → gRPC-Web client (nice-grpc) → Envoy → backend Go service**.
- **Server actions** (`"use server"`) perform the gRPC calls; **React Query** hooks call those server actions and cache results. **Jotai** can hold UI state (modals, form data) as in the reference. Keep the gRPC client usage server-only so it never ships to the client bundle.

---

## 5. sqlc Setup (Shared Database Package)

### 5.1 Single sqlc Project at Root

- **One sqlc project** under `database/`: `database/sql/schema.sql` (full schema), `database/sql/queries/*.sql` (one file per domain), and `database/sqlc.yaml` pointing at them. Run **`sqlc generate -f database/sqlc.yaml`** to produce a single Go package in **`database/generated/`**.
- All Go services in `apps/grpc/*` import this generated package (e.g. `import "stock-market-alert-app/database/generated"` or the module path you use). Each service only calls the queries it needs; no per-service sqlc configs.

### 5.2 Layout

- **database/sql/schema.sql** – Full PostgreSQL schema (copy/adapt from `db/postgres_schema.sql`). Single source of truth for tables and types.
- **database/sql/queries/alert_queries.sql** – `ListAlerts`, `GetAlert`, `CreateAlert`, `UpdateAlert`, `DeleteAlert`.
- **database/sql/queries/alert_audit_queries.sql** – `InsertAlertAudit`, `ListAlertAudits`, etc.
- **database/sql/queries/price_queries.sql** – `GetDailyPrices`, `GetWeeklyPrices`, `GetHourlyPrices`, `UpsertDailyPrices`, `GetStatistics`, etc.
- **database/sql/queries/portfolio_queries.sql** – Portfolio and portfolio_stocks CRUD.
- **database/sql/queries/metadata_queries.sql** – Stock metadata read (and optional upsert).
- **database/sql/queries/document_queries.sql** – `GetDocument`, `UpsertDocument`.

Use sqlc named parameters (e.g. `sqlc.arg(alert_id)`) and `:one` / `:many` / `:exec` annotations. **database/sqlc.yaml** sets `engine: "postgresql"`, `sql` dirs, and `gen.go` output to `database/generated`.

### 5.3 Mapping Current Repositories to sqlc Queries

| Current (Python) | sqlc queries (example) |
|------------------|------------------------|
| `alert_repository.list_alerts` | `ListAlerts` (with optional limit) |
| `alert_repository.get_alert` | `GetAlert` |
| `alert_repository.create_alert` | `CreateAlert` |
| `alert_repository.update_alert` | `UpdateAlert` |
| `alert_repository.delete_alert` | `DeleteAlert` |
| `portfolio_repository.list_portfolios` | Join portfolios + portfolio_stocks → `ListPortfoliosWithStocks` or separate queries |
| `daily_price_repository.get_daily_prices` | `GetDailyPrices` (ticker, date range) |
| `document_store.load_document` | `GetDocument` |
| `document_store.save_document` | `UpsertDocument` |
| `metadata_repository.fetch_stock_metadata_map` | `ListStockMetadata` or `GetStockMetadataBySymbol` |

Implement Redis caching in the **service layer** (Go): e.g. alert-svc checks Redis before calling sqlc, and invalidates on create/update/delete.

---

## 6. Envoy Gateway

### 6.1 Role

- **Listen on port 80** for HTTP/gRPC-Web from the Next.js app (and optionally browser). Terminate gRPC-Web and forward gRPC to backend services.
- **Route by service prefix**: each gRPC service has a path like `/{package}.{ServiceName}/` (e.g. `/stockalert.alert.v1.AlertService/`). Envoy matches the prefix and forwards to the corresponding cluster.

### 6.2 Config Pattern (from reference)

- **docker-compose**: Envoy container uses `envoy.yaml` from the repo root. Backend services (alert_service, price_service, etc.) each listen on their own port (e.g. 50051) and are reachable by service name (e.g. `alert_service:50051`).
- **Routes**: One route per service, match prefix, route to cluster.
- **Clusters**: One cluster per Go service (`alert_service`, `price_service`, `portfolio_service`, `metadata_service`, `document_service`), type `STRICT_DNS` or `STATIC`, host/port as in docker-compose (e.g. `alert_service:50051`).

Example route block:

```yaml
routes:
  - match: { prefix: "/stockalert.alert.v1.AlertService/" }
    route: { cluster: alert_service }
  - match: { prefix: "/stockalert.price.v1.PriceService/" }
    route: { cluster: price_service }
  # ... portfolio, metadata, document
```

When adding a new service, add one route and one cluster in **envoy.yaml** and ensure the service is defined in **docker-compose.yml**.

### 6.3 Alignment with reference

The layout and tooling follow **GRPC_MICROSERVICE_EXAMPLE.md**: **docker-compose** for Postgres, Envoy, Redis (and optional Jaeger); **envoy.yaml** at root; routing by **prefix** per gRPC service. Same pattern for adding a new service: new route + cluster in Envoy, new service in docker-compose, new app under **apps/grpc/**.

---

## 7. Next.js + Shadcn UI

### 7.1 App Structure (under pnpm workspace)

The Next.js app lives at **apps/ui/web/** and is part of the root **pnpm workspace** (`pnpm-workspace.yaml` lists `apps/ui/web`). Run from root with `pnpm --filter=web dev` or from `apps/ui/web` with `pnpm dev`.

```
apps/ui/web/
  app/
    layout.tsx
    page.tsx                    # Home
    add-alert/page.tsx
    alerts/page.tsx
    scanner/page.tsx
    price-database/page.tsx
    stock-database/page.tsx
    portfolio/page.tsx
    discord/page.tsx
    scheduler-status/page.tsx
    market-hours/page.tsx
    daily-move-tracker/page.tsx
    alert-audit-logs/page.tsx
  components/
    ui/                         # Shadcn
    alerts/
    portfolio/
    ...
  lib/
    grpc/                       # Channel + client creation (server-only), using gen/ts
    atoms/                      # Jotai atoms (UI state, form state)
    hooks/                      # React Query hooks that call server actions
  actions/                      # Server actions ("use server") that call gRPC
    alert-actions.ts
    portfolio-actions.ts
    ...
  package.json
```

### 7.2 State and gRPC (Reference Pattern)

- **Server actions** (`"use server"`): Create gRPC channel (to Envoy, e.g. `GRPC_ENDPOINT=localhost:80`), create client from generated TS definition, call RPC (e.g. `client.listAlerts({})`), return typed data. Keep these in `actions/*.ts` and import only on the server.
- **React Query**: Hooks (e.g. `useAlerts()`, `useCreateAlert()`) call server actions as `queryFn` / `mutationFn`. Invalidate cache on mutations so lists stay fresh.
- **Jotai**: Atoms for UI state (modal open, selected alert, form fields). Use focusAtom for nested form state if needed. Keeps components simple and testable.
- **Types**: Use the generated TypeScript types from `gen/ts` so request/response shapes match the Go backend. Map gRPC errors to user-facing messages and use toast (e.g. sonner) or form state for feedback.

### 7.3 Page-to-Page Mapping

| Streamlit | Next.js route | Main server actions / data |
|-----------|----------------|----------------------------|
| Home | `/` | List alerts summary, scheduler status (document_service), quick links |
| Add_Alert | `/add-alert` | CreateAlert, UpdateAlert, GetAlert; metadata (metadata_service); optional price fetch (price_service) |
| Delete_Alert | `/alerts` or `/delete-alert` | DeleteAlert, ListAlerts |
| Alert_History | `/alerts` | ListAlerts |
| Scanner | `/scanner` | ListAlerts, GetDailyPrices/GetHourlyPrices (price_service), metadata |
| Price_Database | `/price-database` | price_service: GetDailyPrices, GetWeeklyPrices, GetHourlyPrices, GetStatistics |
| Stock_Database | `/stock-database` | metadata_service: ListStockMetadata / GetStockMetadata |
| My_Portfolio | `/portfolio` | portfolio_service: ListPortfolios, GetPortfolio, SavePortfolio, DeletePortfolio |
| Discord_Management | `/discord` | GetDocument, SaveDocument (scheduler config), ListPortfolios |
| Scheduler_Status | `/scheduler-status` | GetDocument (scheduler_status_*) via document_service |
| Market_Hours | `/market-hours` | Static or small config API |
| Daily_Move_Tracker | `/daily-move-tracker` | stats queries or price_service + stats |
| Alert_Audit_Logs | `/alert-audit-logs` | alert_service ListAlertAudits |

### 7.4 Conventions (from GRPC_MICROSERVICE_EXAMPLE.md)

- **Go**: Package names lowercase; use gRPC status codes for errors; record errors in spans for tracing; validate at service boundary. Handlers: acquire DB connection from pool, use `database/generated` queries, convert DB types to proto via **core/shared** mappers. All services register the **gRPC Health Checking Protocol**. Services that need other services create gRPC clients at startup (e.g. alert-svc creates a price-svc client via `price_service:50051`).
- **TypeScript**: Server actions with `"use server"`; client components with `"use client"` when using hooks; React Query for server state; Jotai for UI/form state. Use generated TS types from **gen/ts** for type-safe gRPC calls.
- **Adding a new entity**: Define proto → buf generate → add schema + queries → sqlc generate → implement handler in Go → add server action + hook + UI in **apps/ui/web**.

---

## 8. Phased Migration Plan

### Phase 1 – Foundation (weeks 1–2)

1. **Monorepo layout**
   - Add **pnpm-workspace.yaml** at root (e.g. `packages: ["apps/*"]` or `["apps/ui/*"]` so `apps/ui/web` is the web app).
   - Add **go.work** at root; include `apps/grpc/alert_service`, and later other services, plus `core` when you add it.
   - Add **docker-compose.yml** with PostgreSQL, Envoy, Redis (and optionally Jaeger). Use the same `DATABASE_URL` as today so schema can be applied from `db/postgres_schema.sql` (or a copy under `database/sql/schema.sql`).

2. **Proto and codegen**
   - Add **proto/** with `alert/v1`, and optionally stubs for `price/v1`, `document/v1`. Define **AlertService** with ListAlerts, GetAlert (and optionally CreateAlert).
   - Add **buf.yaml** and **buf.gen.yaml** at root; generate **Go** to **gen/go/** and **TypeScript** to **gen/ts/** (e.g. via buf plugins for go and ts). Run **`buf generate`**.

3. **Shared database and sqlc**
   - Add **database/sql/schema.sql** (copy from `db/postgres_schema.sql`; keep only tables needed for alerts to start: `alerts`, `alert_audits` if desired).
   - Add **database/sql/queries/alert_queries.sql** with ListAlerts, GetAlert, CreateAlert, UpdateAlert, DeleteAlert. Add **database/sqlc.yaml** and run **`sqlc generate -f database/sqlc.yaml`**; output to **database/generated/**.

4. **First Go service**
   - Create **apps/grpc/alert_service/** with `go.mod`, `main.go`, server setup (listener, pool from `DATABASE_URL`, optional OpenTelemetry), and handler that uses `database/generated` and implements **AlertService**. Register the **gRPC Health Checking Protocol** (`grpc.health.v1.Health`) so Envoy and Docker can probe liveness. Register in **docker-compose** as `alert_service` on port 50051 with a health check using `grpc_health_probe`.

5. **Envoy**
   - Add **envoy.yaml** at root: listener on port 80, route prefix `/stockalert.alert.v1.AlertService/` → cluster `alert_service` at `alert_service:50051`. Add Envoy service to docker-compose. Run `docker-compose up -d` and verify with grpcurl (through Envoy or directly to alert_service).

6. **Next.js shell**
   - Create **apps/ui/web/** (Next.js + Shadcn) under the pnpm workspace. Add **actions/alert-actions.ts** with server actions that create a gRPC-Web channel (e.g. nice-grpc) to `GRPC_ENDPOINT=localhost:80` and call ListAlerts / GetAlert. Add a page (e.g. **app/alerts/page.tsx**) that uses a React Query hook calling those actions and displays the list. Ensure `GRPC_ENDPOINT` is set in env (e.g. `.env`).

### Phase 2 – Core services and pages (weeks 3–5)

7. **price_service**
   - Add **database/sql/queries/price_queries.sql** (daily_prices, weekly_prices, hourly_prices, ticker_metadata); run sqlc generate. Create **apps/grpc/price_service/** using shared `database/generated`; implement PriceService (GetDailyPrices, GetWeeklyPrices, GetHourlyPrices, GetStatistics, Upsert*). **price-svc owns FMP integration** — implement FMP API client within the service for fetching prices and ticker metadata (RPCs like `FetchLatestPrices`, `SyncTickerMetadata`). Add cluster and route in **envoy.yaml** and service in **docker-compose**.

8. **portfolio_service** and **document_service**
   - Add **portfolio_queries.sql** and **document_queries.sql** under **database/sql/queries/**; sqlc generate. Create **apps/grpc/portfolio_service/** and **apps/grpc/document_service/**; add to **go.work**, **envoy.yaml**, and **docker-compose**.

9. **metadata_service**
   - Add **metadata_queries.sql**; create **apps/grpc/metadata_service/** (read-only to start); optional FMP sync job later.

10. **Next.js pages**
    - Add Alert (create/update), Delete Alert, Scanner, Price Database, Stock Database, My Portfolio, Scheduler Status, Discord Management. Use server actions in **apps/ui/web/actions/** that call gRPC via Envoy; React Query hooks and Jotai as in the reference.

### Phase 3 – Schedulers and background jobs (weeks 6–7)

11. **Keep schedulers in Python initially**
    - Scheduler processes (daily/weekly/hourly) continue to run; replace direct DB/Redis access with **gRPC calls** to alert_service, price_service, document_service. Use the same proto; implement a small Python gRPC client for each job.

12. **Optional: scheduler in Go**
    - If desired, add a thin Go service or cron that triggers “run check” RPCs on alert_service and writes status via document_service.

### Phase 4 – Parity and cutover (weeks 8–9)

13. **stats (optional)**
    - Add **database/sql/queries/stats_queries.sql** for daily_move_stats; either extend price_service or add a small stats service. Add Daily Move Tracker page in **apps/ui/web**.

14. **Alert audit**
    - Ensure alert_audits are written by alert_service (or by the scheduler calling alert_service). Add Alert Audit Logs page.

15. **Redis**
    - Replicate current cache behavior in alert_service and portfolio_service (cache list/get, invalidate on write).

16. **Decommission Streamlit**
    - Run both apps in parallel; switch traffic to Next.js; retire Streamlit and direct Python DB access.

---

## 9. File and Code Mapping Cheat Sheet

| Current | Target |
|---------|--------|
| `src/data_access/alert_repository.py` | **database/sql/queries/alert_queries.sql** + **apps/grpc/alert_service/** (Go handlers using database/generated) |
| `src/data_access/portfolio_repository.py` | **database/sql/queries/portfolio_queries.sql** + **apps/grpc/portfolio_service/** |
| `src/data_access/daily_price_repository.py` | **database/sql/queries/price_queries.sql** + **apps/grpc/price_service/** |
| `src/data_access/document_store.py` | **database/sql/queries/document_queries.sql** + **apps/grpc/document_service/** |
| `src/data_access/metadata_repository.py` | **database/sql/queries/metadata_queries.sql** + **apps/grpc/metadata_service/** |
| `src/data_access/redis_support.py` | Redis usage inside Go services (e.g. alert_service, portfolio_service) |
| `pages/*.py` | **apps/ui/web/app/** routes + **apps/ui/web/actions/** server actions + hooks/atoms |
| `db/postgres_schema.sql` | **database/sql/schema.sql** (single schema for sqlc) |
| FMP logic (`backend_fmp.py`, `smart_price_fetcher.py`) | **apps/grpc/price_service/** (price-svc owns all FMP integration) |
| Discord / Scheduler logic | Inside a service or separate worker calling gRPC; Envoy + **docker-compose** at root |

---

## 10. Risks and Mitigations

| Risk | Mitigation |
|------|-------------|
| Proto churn | Freeze core messages early; use optional fields for new data. |
| Dual-write during migration | Run Streamlit and Next.js in parallel; both write to same DB until cutover, or Streamlit reads via gRPC adapters. |
| Scheduler dependency on DB | Replace DB calls in schedulers with gRPC to the new services; keep one Python env for cron. |
| Redis key format | Keep same key scheme (e.g. `stockalert:alerts:list`) in Go so cache stays valid. |

---

## 11. Next Steps

1. Add **pnpm-workspace.yaml**, **go.work**, and **docker-compose.yml** at repo root; create **apps/ui/web** (Next.js + Shadcn) and **apps/grpc/alert_service** directories.
2. Create **proto/alert/v1** and **database/sql/** (schema + alert_queries); run **buf generate** and **sqlc generate -f database/sqlc.yaml**.
3. Implement **AlertService** in **apps/grpc/alert_service** using **database/generated**; add **envoy.yaml** and Envoy + Postgres (and Redis) to docker-compose.
4. Add server actions and one **apps/ui/web** page (e.g. `/alerts`) that lists alerts via gRPC-Web to Envoy.
5. Expand to remaining services and pages following the phases above.

For implementation details (handler pattern with tracing, Jotai/React Query patterns, Envoy route format, sqlc query conventions), see **GRPC_MICROSERVICE_EXAMPLE.md** in the repo root.

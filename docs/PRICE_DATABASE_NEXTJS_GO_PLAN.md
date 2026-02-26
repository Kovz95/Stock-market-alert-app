# Price Database: Next.js + Go Microservice Plan

This document plans how to rebuild the Streamlit **Price Database** page (`pages/Price_Database.py`) as a **Next.js** front end backed by a **Golang microservice**, reusing the existing PostgreSQL schema and aligning with your current gRPC/Next.js patterns.

---

## 1. What the Current Page Does (Summary)

The Streamlit page is ~1,250 lines and provides:

| Area | Features |
|------|----------|
| **Filters (sidebar)** | Timeframe (hourly/daily/weekly), exchanges multiselect, search by ticker or company name, ticker multiselect, max rows (100–50k), day filter (all/weekdays/weekends), date range (from DB stats) |
| **Load data** | Single “Load Data” action: runs filtered query against `daily_prices` / `hourly_prices` / `weekly_prices`, stores result in session state |
| **Database statistics** | Counts and ticker counts for hourly/daily/weekly tables; date ranges per table |
| **Data table tab** | Paginated table, column picker, sort, optional company/exchange from metadata |
| **Charts tab** | Ticker selector, candlestick + volume (Plotly) or line chart, latest OHLCV metrics |
| **Analysis tab** | Per-ticker summary stats (mean/std/min/max, volume, date range); recent price changes table (latest vs previous close, % change), styled by sign |
| **Export tab** | CSV and Excel download with summary info |
| **Stale data – daily** | “Scan Daily Data” → list of tickers where last daily bar &lt; expected trading day; filter by closed markets; multiselect + “Refresh selected” / “Refresh All” (calls FMP + writes DB + resamples weekly on Friday); retry failed; “Why might a ticker not update?” expander |
| **Stale data – weekly** | “Scan Weekly Data” → list of tickers where last week_ending &lt; expected Friday; “Resample selected” / “Resample All” (from existing daily data, no API); retry failed |
| **Hourly section** | “Days to fetch”, “Skip existing”; “Update All Hourly Data”; “Check for stale hourly data” (exchange-aware expected hour); “Update stale hourly data”; hourly stats (tickers, records, earliest/latest hour); “Hourly Data Quality” (stale &gt;48h, gaps &gt;72 trading hours in last 60d) |

**Backend dependencies (Python):**

- `db_config` (Postgres/SQLite), `metadata_repository` (stock metadata map), `read_sql_with_engine` (pandas over SQLAlchemy)
- `OptimizedDailyPriceCollector` (FMP daily + weekly resample), `HourlyPriceCollector` (FMP hourly)
- `calendar_adapter`: `get_session_bounds`, `get_hourly_alignment`, `get_calendar_timezone` (exchange-aware trading hours)
- `stale_data`: `is_data_stale`, `get_last_trading_day`
- Index creation on price tables for performance

---

## 2. Target Architecture

- **Next.js app** (`apps/ui/web`): new route(s) under e.g. `/price-database` (or `/data/price-database`). Uses **React Query** + **Jotai** (or existing store) for server state and filter state. Calls the new Go service via **gRPC** (preferred, to match alerts/discord) or **REST** (e.g. BFF in Next.js that calls Go).
- **Golang microservice**: new service (e.g. `apps/grpc/price_service` or `prices_service`) that:
  - Uses the **existing Postgres DB** (same schema: `daily_prices`, `hourly_prices`, `weekly_prices`, `stock_metadata`).
  - Exposes gRPC (and optionally REST) for: metadata, stats, price query, stale scans, and **orchestration** of refresh/resample. Actual **FMP API calls and heavy compute** can stay in Python initially (see §6) or be reimplemented in Go later.
- **Shared**: Proto definitions under `proto/`, generated TS (Next.js) and Go (service); existing `database/generated` models (`DailyPrice`, `HourlyPrice`, `WeeklyPrice`, `StockMetadatum`) and new **sqlc** queries for price tables.

---

## 3. Golang Microservice Plan

### 3.1 Service layout

- **Location**: e.g. `apps/grpc/price_service/` (or `prices_service/`), mirroring `alert_service` and `discord_service`.
- **Structure**:
  - `main.go` – entry, config (DB URL, port, optional FMP key if Go does fetch later), start gRPC server.
  - `server.go` – implement `PriceServiceServer` (see proto below).
  - `handler_*.go` – handlers: metadata, stats, load prices, stale daily/weekly/hourly, refresh/resample (orchestration).
  - `config.go` – env/config for DB and optional FMP.
- **Database**: Use existing `pgxpool` + **sqlc**. Add new query file(s) for price tables (see §3.3).

### 3.2 Proto (new)

- **File**: `proto/price/v1/price.proto` (or under existing package if you prefer).
- **Service**: `PriceService` with RPCs aligned to Streamlit actions:

| RPC | Request | Response | Notes |
|-----|---------|----------|--------|
| `GetStockMetadataMap` | (optional filter) | `map<string, StockMetadata>` or list | Exchanges, tickers, names for filters and display. Can reuse/copy from alert if you expose metadata there. |
| `GetDatabaseStats` | - | Hourly/daily/weekly record counts, ticker counts, date ranges | Replaces `get_database_stats()`. |
| `LoadPriceData` | Timeframe, tickers[], start_date, end_date, max_rows, day_filter | Stream or single response of `PriceRow` (ticker, date/datetime, O,H,L,C, volume) | Replaces `load_price_data()`. For large payloads, consider server-streaming. |
| `ScanStaleDaily` | (optional limit) | List of stale daily tickers (ticker, last_date, expected_date, days_old, company, exchange) | Replaces `fetch_stale_ticker_dataframe("daily", ...)`. |
| `ScanStaleWeekly` | (optional limit) | List of stale weekly tickers | Replaces `fetch_stale_ticker_dataframe("weekly", ...)`. |
| `ScanStaleHourly` | - | List of stale hourly tickers + metadata (latest_hour, total_tickers, up_to_date count) | Replaces `fetch_stale_hourly_data()`. |
| `RefreshStaleDaily` | List of tickers | Success/failed/skipped lists | Orchestrates “refresh daily + weekly resample”. Can call out to Python worker or later FMP from Go. |
| `ResampleWeekly` | List of tickers | Success/failed lists | Resample from existing daily data only (no FMP). |
| `UpdateHourlyBatch` | Tickers[], days_back, skip_existing | Success/failed lists | Orchestrates hourly update; can call Python or later FMP from Go. |
| `GetHourlyDataQuality` | - | Stale ticker count, gap ticker count, worst gap hours, etc. | Replaces `_compute_hourly_data_quality()`. |

- **Messages**: Define `StockMetadata` (symbol, name, exchange, isin, …), `PriceRow` (ticker, date/datetime, open, high, low, close, volume), `DatabaseStats`, `StaleTickerRow`, `StaleHourlyMeta`, and request/response wrappers for each RPC. Use `google.protobuf.Timestamp` for dates/times where appropriate.

### 3.3 SQL / sqlc

- **New query file**: e.g. `database/sql/queries/price_queries.sql` (and sqlc config to include it).
- **Queries to add** (conceptually; exact names up to you):
  - **Stats**: `GetDailyStats`, `GetHourlyStats`, `GetWeeklyStats` (COUNT(*), COUNT(DISTINCT ticker), MIN/MAX date/datetime).
  - **Load prices**: `ListDailyPrices`, `ListHourlyPrices`, `ListWeeklyPrices` with params: tickers (array), start_date, end_date, day_filter (e.g. 0=all, 1=weekdays, 2=weekends), limit. Use existing indexes (ticker, date DESC) etc.
  - **Stale daily**: Per-ticker MAX(date), join or subquery with expected date (logic in Go or raw SQL with passed-in expected date); return ticker, last_date, days_old.
  - **Stale weekly**: Same idea for `weekly_prices` and expected week_ending.
  - **Stale hourly**: Per-ticker MAX(datetime); expected hour logic can be in Go (exchange-aware) or approximated in SQL.
  - **Hourly quality**: Stale count (last_dt &lt; NOW() - 48h), gap metrics (trading-hour gaps in last 60 days) – mirror the Streamlit CTEs.
- **Indexes**: Already in schema (`idx_daily_prices_ticker_date`, `idx_hourly_prices_ticker_datetime`, `idx_weekly_prices_ticker_week`). Add `idx_daily_prices_ticker_date` if you have (ticker, date DESC) for list performance.

### 3.4 Business logic in Go

- **Calendar / trading hours**: Port or call the Python `calendar_adapter` logic (exchange → timezone, session bounds, hourly alignment) so Go can compute “expected” last trading day and “expected” last hourly bar. Alternatively, keep a small Python helper and call it from Go for these values until you fully port.
- **Stale daily/weekly**: Implement “expected date” per exchange (weekday, time-of-day for “today vs yesterday”) and “expected week_ending” (e.g. last Friday after US close). Filter out tickers whose exchange is “closed” (holiday) so they don’t show as stale — either in SQL (if you have exchange on price table or join metadata) or in Go after fetch.
- **Refresh / resample / hourly update**: Initially these can be **orchestration only**: validate input, maybe enqueue jobs to a **Python worker** (existing collectors) that reads/writes the same DB. Later you can move FMP calls and resample logic into Go.

---

## 4. Next.js App Plan

### 4.1 Route and layout

- **Route**: e.g. `app/price-database/page.tsx` (or `app/data/price-database/page.tsx`).
- **Layout**: Reuse existing app layout and sidebar; add a “Price Database” (or “Data → Price database”) entry in `app-sidebar.tsx` pointing to this route.

### 4.2 State and data flow

- **Filters**: Keep in React state (e.g. `useState`) or Jotai atoms: timeframe, selectedExchanges, searchType, searchInput, selectedTickers, maxRows, dayFilter, startDate, endDate. Persist to URL searchParams if you want shareable links.
- **Metadata**: One React Query key for “stock metadata” (list or map) – used for exchange list, ticker search, and enriching tables. Call `GetStockMetadataMap` (or equivalent) on load; cache 5–30 min.
- **Database stats**: One query for `GetDatabaseStats`; refetch on focus or after refresh actions so “Database Statistics” and date-range defaults stay correct.
- **Price data**: Query key depends on filters + “load” action. Either:
  - “Load Data” sets a “request” (e.g. filters hash) in state and a query fetches when that request exists; or
  - Single mutation “load price data” that returns rows and stores them in React state or a dedicated atom. Prefer one source of truth (server state) so table/charts/export/analysis all read the same dataset.
- **Stale sections**: Each section (daily, weekly, hourly) has “scan” (query or mutation returning list + meta) and “refresh/resample/update” (mutation). After mutation, invalidate stats and optionally stale list.

### 4.3 UI components (by section)

- **Sidebar**: Form with timeframe radio, exchange multiselect, search type + input, ticker multiselect (filtered by search + exchange), max rows, day filter, start/end date (from stats when available), “Load Data” button. Mirror Streamlit’s “Apply Filters” + “Load Data” flow.
- **Main**:
  - **Database statistics**: Row of metric cards (hourly/daily/weekly records and tickers) – data from `GetDatabaseStats`.
  - **Content when data loaded**: Tabs: “Data Table”, “Charts”, “Analysis”, “Export”.
    - **Data table**: Client-side pagination (or server-side if you add pagination to `LoadPriceData`), column visibility, sort. Use your existing `DataTable` or similar; add company/exchange from metadata.
    - **Charts**: Ticker select, candlestick + volume (e.g. **Lightweight Charts** or **Recharts** + custom candlestick, or **TradingView** widget). Show latest OHLCV as small metrics row.
    - **Analysis**: Summary stats by ticker (table); “Recent price changes” table (latest vs previous close, % change, color by sign). Compute from loaded rows in the client or add a small “analysis” RPC that returns aggregates.
    - **Export**: Buttons to download CSV/Excel (generate in client from loaded data or call a small export RPC). Show record count, ticker count, date range.
  - **Stale Data Monitor**: Two sub-tabs: “Daily Stale”, “Weekly Stale”.
    - **Daily**: “Scan Daily Data” button → show table of stale tickers (with company, exchange, last update, days old); multiselect; “Refresh selected” / “Refresh All”; “Last run summary” (success/failed/skipped) + “Retry failed”; expander “Why might a ticker not update?”.
    - **Weekly**: Same pattern with “Scan Weekly Data”, “Resample selected” / “Resample All”, retry failed.
  - **Hourly Data Management**: “Days to fetch”, “Skip existing” checkbox; “Update All Hourly Data” and “Update stale hourly data” buttons; “Check for stale hourly data” → table of stale tickers + metadata (latest hour, coverage %); “Hourly Data Statistics” and “Hourly Data Quality” (stale >48h, gaps >72h) from `GetHourlyDataQuality` or stats.

### 4.4 gRPC client and actions

- **Channel**: Add a **PriceService** client next to `alertClient` and `discordClient` in `lib/grpc/channel.ts` (new proto + generated TS in `gen/ts/price/v1/`).
- **Server actions**: e.g. `actions/price-database-actions.ts`: thin wrappers that call the price service RPCs (GetStockMetadataMap, GetDatabaseStats, LoadPriceData, ScanStaleDaily, ScanStaleWeekly, ScanStaleHourly, RefreshStaleDaily, ResampleWeekly, UpdateHourlyBatch, GetHourlyDataQuality). Return plain objects so React Query can cache them.
- **Hooks**: e.g. `lib/hooks/usePriceDatabase.ts`: `useStockMetadata()`, `useDatabaseStats()`, `usePriceData(filters, enabled)`, `useStaleDaily()`, `useStaleWeekly()`, `useStaleHourly()`, and mutations for load, refresh, resample, update hourly. Invalidate relevant queries on mutation success.

### 4.5 Charting and export

- **Charts**: Use a library that supports candlestick + volume (e.g. `lightweight-charts` or Recharts with a candlestick series). Data = current loaded price rows filtered by selected ticker, sorted by date.
- **Export**: CSV can be done in the browser (e.g. `papaparse` or simple join). Excel: use `xlsx` or `exceljs` in the client, or add a small “Export to Excel” RPC that returns a blob.

---

## 5. Implementation Phases (Suggested)

| Phase | Scope | Deliverables |
|------|--------|--------------|
| **1 – Go read path** | DB access + read-only API | sqlc price queries (stats, list daily/hourly/weekly); `PriceService` with GetDatabaseStats, LoadPriceData, GetStockMetadataMap (or reuse alert service for metadata); proto + Go server + TS client |
| **2 – Next.js viewer** | Filters + load + table + export | Route, sidebar filters, “Load Data”, table with pagination/sort/columns, CSV/Excel export, database stats cards |
| **3 – Charts and analysis** | Charts + analysis tab | Candlestick+volume chart, analysis tab (summary stats + recent changes) |
| **4 – Stale scan (read-only)** | Stale detection in Go | ScanStaleDaily, ScanStaleWeekly, ScanStaleHourly, GetHourlyDataQuality; port or reuse calendar/expected-date logic |
| **5 – Next.js stale UI** | Stale tabs + scan only | Daily/Weekly/Hourly stale tabs, scan buttons, tables, quality metrics; no refresh yet |
| **6 – Refresh orchestration** | Go calls Python or FMP | RefreshStaleDaily, ResampleWeekly, UpdateHourlyBatch in Go: either call existing Python scripts/HTTP or implement FMP + resample in Go; progress can be streaming or polling |
| **7 – Next.js refresh UI** | Buttons and feedback | Refresh/Resample/Update buttons, progress (streaming or polling), last-run summary, retry failed |

---

## 6. Decoupling FMP and Python (Optional)

- **Option A – Orchestration in Go, workers in Python**: Go service enqueues “refresh daily for [tickers]” / “update hourly for [tickers]” to a queue (e.g. Redis, SQS). Existing Python workers (or a single “price worker” script) consume jobs, call FMP, write to Postgres. Go “Refresh” RPC returns immediately with a job_id; front end polls a “JobStatus” RPC or uses server-streaming for progress. Keeps FMP and resample logic in Python.
- **Option B – Port to Go**: Implement FMP client and daily/weekly/hourly collectors in Go; resample weekly from daily in Go. Single deployment, no Python for this flow.
- **Option C – Hybrid**: Resample weekly and “read” paths in Go; daily/hourly **fetch** still in Python via a small HTTP API that the Go service calls. Reduces Python surface while reusing battle-tested collectors.

---

## 7. Files to Add/Change (Checklist)

**Proto & codegen**

- [ ] `proto/price/v1/price.proto` – service + messages
- [ ] Generate Go: `gen/go/price/v1/`
- [ ] Generate TS: `gen/ts/price/v1/`

**Go**

- [ ] `database/sql/queries/price_queries.sql` – stats, list prices, stale queries, hourly quality
- [ ] sqlc config: include `price_queries.sql`; regenerate `database/generated/`
- [ ] `apps/grpc/price_service/` – main, server, config, handlers (metadata, stats, load, stale, refresh/resample/hourly)

**Next.js**

- [ ] `apps/ui/web/lib/grpc/channel.ts` – add price service client
- [ ] `apps/ui/web/actions/price-database-actions.ts`
- [ ] `apps/ui/web/lib/hooks/usePriceDatabase.ts`
- [ ] `apps/ui/web/app/price-database/page.tsx` (and optional sub-routes)
- [ ] `apps/ui/web/app/price-database/_components/` – filters, table, charts, analysis, export, stale daily/weekly/hourly sections
- [ ] Sidebar: link to Price Database

**Infra**

- [ ] Wire new service in docker-compose / k8s / dev script (same DB, new port if needed)
- [ ] Env: DB URL, optional FMP key, optional queue URL for Option A

---

## 8. Summary

- **Backend**: New **Golang gRPC service** with sqlc queries on `daily_prices`, `hourly_prices`, `weekly_prices`, and `stock_metadata`; implement read path first (stats, load, metadata, stale scans, hourly quality), then refresh/resample/hourly as orchestration (calling Python or FMP from Go as you prefer).
- **Front end**: **Next.js** page with filters, load, table, charts, analysis, export, and stale monitor (daily/weekly/hourly) using React Query and the new gRPC client.
- **Alignment**: Same DB schema, same proto/codegen patterns as alert and discord services; optional REST BFF in Next.js if you prefer HTTP for large payloads (e.g. streamed price rows).

This plan should be enough to implement the Price Database feature in your Next.js app with a Go microservice while keeping the door open to keep or migrate FMP/calendar logic.

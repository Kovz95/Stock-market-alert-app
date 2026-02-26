# Alert Audit Logs – Next.js Implementation Plan

This plan maps the Streamlit **Alert Audit Logs** page (`pages/Alert_Audit_Logs.py`) to the Next.js app in `apps/ui/web`, including backend API needs, data types, and UI structure.

---

## 1. Summary of Current Streamlit Page

The Streamlit page provides:

| Section | Purpose |
|--------|---------|
| **Performance Overview** | 4 metrics: Total Checks, Success Rate, Cache Hit Rate, Avg Execution (ms); error-rate banner |
| **Quick Actions** | Refresh Data, Clear All Audit Data, Export Summary CSV |
| **Alert Evaluation Summary** | Paginated/filtered table of audit summary rows (alert_id, ticker, stock_name, exchange, timeframe, action, type, checks, triggers, times) |
| **Detailed Alert History** | When an Alert ID is selected: raw history table + execution time trend line chart |
| **Analytics & Insights** | Success rate by timeframe (bar), Cache hit rate vs check frequency (scatter) |
| **Failed Price Data Analysis** | Failed-alert metrics, asset-type breakdown (pie), failures by exchange (bar), top 20 problematic tickers table, export CSV/report, insights |

**Data sources (Python):**

- `get_audit_summary(days)` → DataFrame (summary per alert/ticker/timeframe/type)
- `get_performance_metrics(days)` → dict (total_checks, success_rate, cache_hit_rate, avg_execution_time_ms, error_rate, etc.)
- `get_alert_history(alert_id, limit)` → DataFrame (raw audit rows)
- Direct SQL on `alert_audits` for failed price data + `load_market_data()` for asset type

---

## 2. Architecture Overview

- **Next.js** today uses **gRPC** for alerts (`alertClient` in `apps/ui/web/actions/alert-actions.ts`). Audit data is **not** in the current gRPC proto; it lives in the Python codebase and PostgreSQL/SQLite `alert_audits` table.
- **Recommended approach:** Expose audit data via a **backend API** (REST or additional gRPC service) that reads from the same DB as `alert_audit_logger`, then have Next.js call it from **server actions** (or API routes that proxy to the backend). Keep the same patterns as alerts: server actions + TanStack Query + hooks.

---

## 3. Backend API (New)

You need an API that the Next.js app can call. Two options:

### Option A – Extend gRPC (same server as alerts)

- Add an **Audit** service (or extend Alert service) in the same gRPC server that already has DB access (or can connect to the same PostgreSQL as `alert_audits`).
- Define RPCs and messages for: audit summary, performance metrics, alert history, failed price data.

### Option B – REST API (e.g. FastAPI/Flask in Python)

- Small REST service that uses existing `src.services.alert_audit_logger` and `src.data_access.db_config` (and `load_market_data` if you keep asset-type logic in Python).
- Next.js server actions call this REST API with `fetch()`.

**Required endpoints (REST-style; map to gRPC if using Option A):**

| Endpoint | Method | Query/body | Returns |
|----------|--------|------------|--------|
| `/audit/summary` | GET | `days=7` | `{ rows: AuditSummaryRow[] }` |
| `/audit/metrics` | GET | `days=7` | `PerformanceMetrics` |
| `/audit/history` | GET | `alertId=...&limit=100` | `{ rows: AuditHistoryRow[] }` |
| `/audit/failed` | GET | `days=7` | `FailedPriceDataResponse` |
| `/audit/clear` | POST | (optional body) | `{ deletedCount: number }` |

Implement these by calling the same logic as in the Streamlit page (e.g. `get_audit_summary`, `get_performance_metrics`, `get_alert_history`, and the failed-price SQL + asset-type logic). Return JSON with the shapes below.

---

## 4. Data Types (TypeScript)

Define these in the Next.js app (e.g. `apps/ui/web/lib/types/audit.ts` or next to actions).

### 4.1 Performance metrics (from `get_performance_metrics`)

```ts
export type PerformanceMetrics = {
  total_checks: number;
  successful_price_pulls: number;
  success_rate: number;
  cache_hit_rate: number;
  avg_execution_time_ms: number;
  total_errors: number;
  error_rate: number;
  analysis_period_days: number;
};
```

### 4.2 Audit summary row (from `get_audit_summary`)

```ts
export type AuditSummaryRow = {
  alert_id: string;
  ticker: string;
  stock_name: string | null;
  exchange: string | null;
  timeframe: string | null;
  action: string | null;
  evaluation_type: string;
  total_checks: number;
  successful_price_pulls: number;
  successful_evaluations: number;
  total_triggers: number;
  avg_execution_time_ms: number | null;
  last_check: string;   // ISO
  first_check: string; // ISO
};
```

### 4.3 Alert history row (from `get_alert_history`)

```ts
export type AuditHistoryRow = {
  id?: number;
  timestamp: string;
  alert_id: string;
  ticker: string;
  stock_name: string | null;
  exchange: string | null;
  timeframe: string | null;
  action: string | null;
  evaluation_type: string;
  price_data_pulled: boolean;
  price_data_source: string | null;
  conditions_evaluated: boolean;
  alert_triggered: boolean;
  trigger_reason: string | null;
  execution_time_ms: number | null;
  cache_hit: boolean;
  error_message: string | null;
};
```

### 4.4 Failed price data (from failed-price SQL + asset type)

```ts
export type FailedAlertRow = {
  alert_id: string;
  ticker: string;
  stock_name: string | null;
  exchange: string | null;
  timeframe: string | null;
  asset_type: string;   // "Stock" | "ETF" | "Unknown"
  failure_count: number;
  last_failure: string;
  first_failure: string;
  avg_execution_time: number | null;
};

export type FailedPriceDataResponse = {
  rows: FailedAlertRow[];
  totalFailedAlerts: number;
  totalFailures: number;
  failureRate: number;
  assetTypeBreakdown: { asset_type: string; failed_alerts: number; failure_count: number }[];
  exchangeBreakdown: { exchange: string; failed_alerts: number; failure_count: number }[];
};
```

Use these types in server actions and UI.

---

## 5. Next.js App Structure

### 5.1 Routing and sidebar

- **Route:** `app/alerts/audit/page.tsx` (or `app/audit/page.tsx` if you prefer a top-level “Audit” section).
- **Sidebar:** In `app-sidebar.tsx`, set the “Alert Audit” item to point to this route, e.g. `url: "/alerts/audit"`.

### 5.2 Server actions

- **File:** `apps/ui/web/actions/audit-actions.ts` (or `audit-actions.ts` next to `alert-actions.ts`).
- **Implement:**  
  - `getAuditSummary(days: number)`  
  - `getPerformanceMetrics(days: number)`  
  - `getAlertHistory(alertId: string, limit?: number)`  
  - `getFailedPriceData(days: number)`  
  - `clearAllAuditData()` (optional; call backend POST `/audit/clear` or equivalent).  

Each action should call your backend (REST or gRPC), parse JSON (or proto) into the TypeScript types above, and return them. Handle errors and return `null` or throw as appropriate.

### 5.3 Data fetching and state

- **Hooks:** e.g. `apps/ui/web/lib/hooks/useAudit.ts` (or `useAuditLogs.ts`).
  - `useAuditSummary(days)` – `useQuery` on `getAuditSummary(days)`.
  - `usePerformanceMetrics(days)` – `useQuery` on `getPerformanceMetrics(days)`.
  - `useAlertHistory(alertId, limit)` – `useQuery` on `getAlertHistory(alertId, limit)`, `enabled: !!alertId`.
  - `useFailedPriceData(days)` – `useQuery` on `getFailedPriceData(days)`.
  - Optional: `useClearAuditData()` – `useMutation` that invalidates the audit-related query keys.
- **Filter state:** Keep `days`, `alertIdFilter`, `tickerFilter`, `evaluationType`, `statusFilter`, `maxRows` in component state (or in a small store, e.g. Jotai) so the page and tables react to filter changes.

---

## 6. Page Layout and Components

Mirror the Streamlit layout with a single scrollable page.

### 6.1 Layout

- **Top:** Page title and short description (e.g. “Alert Audit Logs & Analytics”).
- **Filters:** Sidebar or top bar: days (1–90), Alert ID, Ticker, Evaluation type (All / scheduled / manual / test), Status (All / Success / Error / Triggered / Not Triggered), Max rows (e.g. 100–5000).
- **Main content (order below):**

1. **Performance Overview + Quick Actions**  
   - Left: 4 metric cards (Total Checks, Success Rate, Cache Hit Rate, Avg Execution) + error-rate banner (warning/info/success by threshold).  
   - Right: Buttons – Refresh (invalidate queries), Clear All Audit Data (call mutation, then invalidate), Export Summary CSV (client-side from summary data).

2. **Alert Evaluation Summary**  
   - Subheading: “Alert Evaluation Summary” and “Showing data for the last X days”.  
   - Table: columns aligned with `AuditSummaryRow` (Alert ID, Ticker, Stock Name, Exchange, Timeframe, Action, Type, Checks, Price Pulls, Evaluations, Triggers, Avg Time (ms), Last Check, First Check). Apply filters (alert_id, ticker, evaluation_type, status) and max rows in the client (or ask the backend for server-side filtering if you prefer).  
   - Optional: client-side “Search in results” (filter rows by string in key columns).

3. **Detailed Alert History**  
   - Shown when “Alert ID” filter is set.  
   - Subheading: “Detailed Alert History for Alert ID: …”.  
   - Table: history rows (Timestamp, Ticker, Type, Price Data, Source, Cache Hit, Evaluated, Triggered, Trigger Reason, Time (ms), Error).  
   - **Performance trend chart:** Execution time (ms) over time – line chart (e.g. Recharts `LineChart` with `ChartContainer`), only when `history.length > 1`.

4. **Analytics & Insights**  
   - Two charts in a row:  
     - Success rate by timeframe (bar chart).  
     - Cache hit rate vs check frequency (e.g. scatter or bar for “top N tickers”).  
   - Data derived from the audit summary response (same as Streamlit).

5. **Failed Price Data Analysis**  
   - Subheading and “Refresh Failed Data” button.  
   - Metrics: Failed Alerts, Total Failures, Avg Failures/Alert, Failure Rate.  
   - Asset type breakdown: list + pie chart (Recharts).  
   - Failures by exchange: list (top 10) + bar chart (top 15).  
   - Table: top 20 failed alerts (`FailedAlertRow`).  
   - Export: Download Failed Alerts CSV, Download Summary Report (text).  
   - Short insights block (high/moderate/acceptable failure rate + worst exchange, optional ticker-pattern notes).

6. **Footer**  
   - Short tips (filters, export, what the system tracks).

Use your existing UI primitives: `Card`, `Table`, `Button`, `Select`, `Input`, `ChartContainer` + Recharts (Bar, Line, Pie, Scatter as needed).

---

## 7. Implementation Checklist

- [ ] **Backend**
  - [ ] Add REST or gRPC endpoints for: audit summary, metrics, history, failed price data, (optional) clear.
  - [ ] Implement using existing `alert_audit_logger` and DB; reuse failed-price SQL and asset-type logic.
  - [ ] Return JSON (or proto) matching the TypeScript types above.

- [ ] **Next.js – core**
  - [ ] Add types in `lib/types/audit.ts` (or equivalent).
  - [ ] Add `actions/audit-actions.ts` and call backend from each action.
  - [ ] Add `lib/hooks/useAudit.ts` with `useAuditSummary`, `usePerformanceMetrics`, `useAlertHistory`, `useFailedPriceData`, (optional) `useClearAuditData`.
  - [ ] Create route `app/alerts/audit/page.tsx` and wire sidebar “Alert Audit” to `/alerts/audit`.

- [ ] **Next.js – UI**
  - [ ] Performance overview: 4 metric cards + error-rate banner.
  - [ ] Quick actions: Refresh, Clear audit data, Export summary CSV.
  - [ ] Filters: days, alert ID, ticker, evaluation type, status, max rows.
  - [ ] Summary table with column mapping and client-side filtering/pagination.
  - [ ] Detailed history: table + execution time line chart when alert ID is set.
  - [ ] Analytics: success rate by timeframe (bar), cache hit vs checks (scatter/bar).
  - [ ] Failed price data: metrics, asset-type list + pie, exchange list + bar, top-20 table, CSV + report export, insights.
  - [ ] Loading and error states (reuse patterns from `AlertsLoading` / `AlertsError`).
  - [ ] Date/time display in user’s locale or fixed timezone (e.g. America/New_York) to match Streamlit.

- [ ] **Polish**
  - [ ] Empty states when no audit data or no failed data.
  - [ ] Tooltips or help text for metrics and filters where useful.
  - [ ] Optional: server-side filtering/pagination for summary table if row count is large.

---

## 8. Notes

- **Timezone:** Streamlit converts to Eastern; in Next.js you can format timestamps with `Intl` or a library (e.g. `date-fns-tz`) using `America/New_York` for consistency.
- **Export:** CSV and report can be generated in the browser from the data already fetched (no extra backend call needed), using blob download and a filename with current date.
- **Clear audit data:** Expose only if appropriate for your environment (e.g. admin-only); consider a confirmation dialog.
- **Summary table “cache_hit”:** The Streamlit summary aggregates by alert/ticker/timeframe/action/type and does not expose a `cache_hit` column in the summary table; the “Cache hit rate vs check frequency” chart in the Python code uses a groupby on `ticker` and a `cache_hit` sum. If your backend summary does not include cache_hit per row, you may need an extra endpoint or aggregated field for that chart (e.g. “cache stats by ticker”); the plan above assumes the failed-price and analytics endpoints (or summary) provide enough to replicate those charts.

Once the backend returns the shapes above and the actions/hooks are wired, the page is a matter of layout and using your existing table and chart components with the audit data.

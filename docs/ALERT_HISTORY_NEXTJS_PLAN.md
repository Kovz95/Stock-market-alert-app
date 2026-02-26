# Alert History Lookup – Next.js Migration Plan

This plan describes how to migrate the Streamlit **Alert History** page (`pages/Alert_History.py`) to the Next.js app in `apps/ui/web`. The page lets users search for a company by ticker/name and view alert trigger history, or browse alerts with filters and jump to full history for a ticker.

---

## 1. Summary of Current Streamlit Page

The Streamlit page has **two tabs**:

| Tab | Purpose |
|-----|--------|
| **Search by Ticker/Company** | Search companies by ticker or name → select one → view company info, active alerts for that ticker, and trigger history from the audit log (with options: triggers only vs all evaluations, days of history). |
| **Browse with Filters** | Filter alerts by exchange, economy, action, portfolio membership, conditions, date range, and trigger status (All / Triggered Only / Not Triggered). Show filtered alerts with trigger count in range and “View Full History” to switch to the search tab with that ticker pre-filled. |

### Data sources (Python)

| Data | Source | Used for |
|------|--------|----------|
| Stock metadata | `fetch_stock_metadata_map()` → `{ ticker: { name, exchange, type, rbics_economy, ... } }` | Search companies by name/ticker; enrich alerts with economy |
| Alerts list | `fetch_alerts_list()` (Redis-backed list) | Active alerts for a ticker; filter options; browse results |
| Portfolios | `fetch_portfolios()` → `{ portfolio_id: { name, stocks: [{ symbol }] } }` | “In Any Portfolio” and “Specific Portfolios” filters |
| Trigger history by ticker | Direct SQL on `alert_audits` WHERE ticker = %s (triggered only or all, limit 50/100) | Tab 1: trigger history table and stats |
| Triggered alerts in date range | Direct SQL on `alert_audits` (triggered, timestamp BETWEEN start/end) | Tab 2: which alerts triggered in range; trigger count per alert |

---

## 2. Gap Analysis

| Capability | Next.js / gRPC today | Needed for Alert History |
|------------|----------------------|---------------------------|
| List alerts | ✅ `ListAlerts` (gRPC) | Use for “alerts for ticker” and browse; may need to fetch all pages or support filter by ticker |
| Alert by ID | ✅ `GetAlert` | Not required for this page |
| Audit summary | ✅ `GetAuditSummary(days)` | Use for browse tab; currently “last N days” only (no custom start/end) |
| Audit history | ✅ `GetAlertHistory(alertId, limit)` | **By alert ID only.** Need **history by ticker** for Tab 1 |
| Stock search | ❌ | Need **search companies by ticker/name** (stock_metadata) |
| Portfolios | ❌ | Need **list portfolios** (and ticker→portfolios map) for filters |
| Trigger counts in date range | Partial | `GetAuditSummary(days)` gives last N days; Tab 2 uses explicit start/end. Either add date range to summary or accept “last N days” and derive range in UI |

---

## 3. Backend API Additions

The gRPC Alert service (or a separate service with DB access to `alert_audits` and `stock_metadata`) should expose the following.

### 3.1 Trigger history by ticker (required for Tab 1)

**RPC:** `GetTriggerHistoryByTicker`

- **Request:** `ticker` (string), `include_all_evaluations` (bool, default false), `limit` (int32, default 50), optional `days_back` (int32, filter to last N days).
- **Response:** List of rows similar to `AuditHistoryRow`: `timestamp`, `alert_id`, `ticker`, `alert_triggered`, `trigger_reason`, `execution_time_ms`, `evaluation_type`; optionally `alert_name` (from alerts list) for display.
- **Implementation:** Query `alert_audits WHERE ticker = $1` (and optionally `alert_triggered = true` when `include_all_evaluations` is false), ORDER BY timestamp DESC, LIMIT. Optionally filter by `timestamp >= now() - days_back`. Map `alert_id` to alert name using in-memory alert list or a join/subquery if alerts are in the same DB.

### 3.2 Search stocks (required for Tab 1)

**RPC:** `SearchStocks` (or REST equivalent)

- **Request:** `query` (string), optional `limit` (int32, default 20).
- **Response:** List of `{ ticker, name, exchange, type }` (and optionally `rbics_economy`).
- **Implementation:** Query `stock_metadata` (or equivalent) WHERE symbol ILIKE %query% OR name ILIKE %query%, ordered by relevance (exact ticker match first), limit 20. The Go service would need access to the same DB that holds `stock_metadata` (today only Python has this).

### 3.3 List portfolios (required for Tab 2 filters)

**RPC:** `ListPortfolios`

- **Request:** (none or optional include_stocks).
- **Response:** List of `{ portfolio_id, name, tickers?: string[] }` (tickers optional for “In Any Portfolio” and “Specific Portfolios” filters).
- **Implementation:** If portfolios live in PostgreSQL/Redis, add a small handler that returns them; otherwise a thin REST proxy to existing Python/Redis is an option.

### 3.4 Audit summary by date range (optional; improves Tab 2)

- **Option A:** Add optional `start_date` / `end_date` (or `start_time`/`end_time`) to `GetAuditSummary`; when set, filter `alert_audits` by `timestamp BETWEEN start AND end` instead of “last N days”.
- **Option B:** Keep “last N days” only; in the UI compute `days` from the user’s date range (e.g. end_date - start_date) and show a note that the summary is for “last N days” rather than the exact range. Tab 2 can still work with a single “days” selector for consistency with the rest of the app.

Recommendation: Start with Option B; add date range to the backend later if needed.

---

## 4. Data Types (TypeScript)

Define in `apps/ui/web/lib/types/alert-history.ts` (or next to actions).

### 4.1 Stock search result

```ts
export type StockSearchResult = {
  ticker: string;
  name: string;
  exchange: string;
  type: string;
  rbicsEconomy?: string;
};
```

### 4.2 Trigger history row (by ticker)

Reuse or extend existing `AuditHistoryRow` from audit-actions; ensure fields used by the Streamlit page are present: `timestamp`, `alertId`, `ticker`, `alertTriggered`, `triggerReason`, `executionTimeMs`, `evaluationType`. Add `alertName?: string` if the backend returns it.

### 4.3 Portfolio (for filters)

```ts
export type Portfolio = {
  portfolioId: string;
  name: string;
  tickers?: string[];  // symbols in this portfolio
};
```

### 4.4 Alert with optional trigger stats (for browse tab)

Use existing Alert type from gRPC. For “trigger count in range” and “last triggered”, either:

- Derive from `GetAuditSummary(days)` (each row has `totalTriggers`, `lastCheck`, `firstCheck`), and match summary rows to alerts by `alertId` + ticker + timeframe + action, or
- Add a small backend type that returns alert + trigger count + last triggered for the selected period.

---

## 5. Next.js App Structure

### 5.1 Route and sidebar

- **Route:** `app/alerts/history/page.tsx` (or `app/alerts/alert-history/page.tsx`).
- **Sidebar:** In `app-sidebar.tsx`, set the existing “Alert History” item to `url: "/alerts/history"` (or the chosen path).

### 5.2 Server actions

- **File:** `apps/ui/web/actions/alert-history-actions.ts` (or extend `audit-actions.ts`).
- **Actions:**
  - `searchStocks(query: string, limit?: number): Promise<StockSearchResult[]>` — call new gRPC (or REST) SearchStocks.
  - `getTriggerHistoryByTicker(ticker: string, options?: { includeAllEvaluations?: boolean; limit?: number; daysBack?: number }): Promise<AuditHistoryRow[]>` — call new gRPC GetTriggerHistoryByTicker.
  - `listPortfolios(): Promise<Portfolio[]>` — call new gRPC ListPortfolios.

Existing:

- `listAlertsPaginated` / list all alerts — use for “alerts for this ticker” (filter client-side by ticker) and for browse (fetch all or paginated and filter).
- `getAuditSummary(days)` — use for browse tab to get trigger counts per alert in the last N days; match to alerts client-side.

### 5.3 Hooks

- **File:** `apps/ui/web/lib/hooks/useAlertHistory.ts` (or add to `useAudit.ts`).
  - `useStockSearch(query: string)` — `useQuery` on `searchStocks(query)`, `enabled: query.length >= 2`.
  - `useTriggerHistoryByTicker(ticker: string, options)` — `useQuery` on `getTriggerHistoryByTicker(ticker, options)`, `enabled: !!ticker`.
  - `usePortfolios()` — `useQuery` on `listPortfolios()`.

Continue using:

- `useAlertsPaginated()` or a “list all alerts” helper for Tab 1 (alerts for ticker) and Tab 2 (filtered list).
- `useAuditSummary(days)` for Tab 2 trigger stats when date range is expressed as “last N days”.

### 5.4 URL state for “View Full History”

- When user clicks “View Full History” in the Browse tab, navigate to the same page with query param: `/alerts/history?tab=search&ticker=AAPL`.
- On load, if `ticker` is present, switch to the Search tab and run the search for that ticker (or pre-fill the input and run search once).

---

## 6. Page Layout and Components

Single page with two tabs (e.g. using `Tabs` from `@/components/ui/tabs`).

### 6.1 Tab 1: Search by Ticker/Company

1. **Search bar:** Text input + “Search” button; optional popular tickers (e.g. AAPL, MSFT) as quick buttons.
2. **When no search / no results:** Short “How to use” and “Popular tickers”.
3. **When results:**  
   - If multiple matches: select dropdown (e.g. “Ticker - Name (Exchange)”); if single, auto-select.  
   - Company summary: 4 metric-style cards (Ticker, Company, Exchange, Type).  
   - **Active alerts:** List of expanders/cards per alert (name, timeframe, action, exchange, economy, conditions). Reuse patterns from alerts list.  
   - **Trigger history:**  
     - Toggle: “Show all evaluations (not just triggers)”.  
     - “Days of history” number input (1–365).  
     - Metrics row: Total Triggers, Total Evaluations, Unique Alerts, Days Since Last Trigger.  
     - List of recent triggers (or evaluations) in expanders: alert name, date/time, triggered yes/no, source, execution time, condition details (formatted).  
     - Optional: small “Alert statistics” table (by alert name: total checks, triggers, trigger rate %) from the history data.

### 6.2 Tab 2: Browse with Filters

1. **Filters row:**  
   - Exchange (multiselect, include “All”).  
   - Economy (multiselect).  
   - Action (multiselect).  
   - “In Any Portfolio” (checkbox).  
   - Specific Portfolios (multiselect).  
   - Conditions (multiselect, from unique condition strings in alerts).  
   - Trigger Status: All / Triggered Only / Not Triggered.  
   - Date range: Start date, End date (or single “Days” selector that sets range relative to today).  
   - “Apply Filters” button.
2. **Results:** “Results: N alerts” with caption explaining the date range and trigger filter.
3. **Alert cards:** For each filtered alert, an expander/card with:  
   - Stock (name, ticker), Exchange, Economy, Portfolio(s).  
   - Action, Timeframe, Triggers in range, Last triggered.  
   - Conditions list.  
   - **“View Full History”** button → navigate to `/alerts/history?tab=search&ticker=<ticker>`.

Filter logic (client-side):

- Start from all alerts (from ListAlerts, possibly paginated or “load all” for this page).
- Enrich with economy from stock search/metadata if available, or from alert payload if present.
- Filter by exchange, economy, action, conditions, portfolio (using `listPortfolios()` to get ticker→portfolios map).
- For “Triggered Only” / “Not Triggered”: use `GetAuditSummary(days)`; for “Triggered Only” keep alerts that have at least one row in the summary with `totalTriggers > 0` in that period; for “Not Triggered” keep alerts with no such row. Use `days` = (end_date - start_date).days (or similar).
- Apply date range to the summary (if backend supports date range, use it; otherwise use `days` as above).

### 6.3 Shared UI

- Reuse `Card`, `Table`, `Button`, `Input`, `Select`, `Tabs`, expanders/collapsibles.  
- Format condition details (JSON or string) with a small helper similar to Streamlit’s `format_condition`.  
- Format dates with existing `formatAuditDate` / `formatAuditDateTime` or equivalent.  
- Loading and empty states for search, history, and browse results.

---

## 7. Implementation Checklist

### Backend

- [ ] **GetTriggerHistoryByTicker**  
  - Add RPC (and proto) to Alert service (or audit service).  
  - Implement in Go: query `alert_audits` by ticker, optional filter by `alert_triggered`, limit, optional `days_back`.  
  - Return list of history rows; optionally join with alerts to include alert name.
- [ ] **SearchStocks**  
  - Add RPC or REST endpoint that queries `stock_metadata` (or equivalent).  
  - Ensure the service has DB (or API) access to stock metadata; implement in Go or keep a small Python proxy and call from Next.js.
- [ ] **ListPortfolios**  
  - Add RPC (or REST) that returns portfolio list and optionally tickers per portfolio.  
  - Implement using existing portfolio data source (DB/Redis).
- [ ] **(Optional)** Extend **GetAuditSummary** with optional `start_date` / `end_date` for exact Tab 2 date range.

### Next.js – core

- [ ] Add types: `StockSearchResult`, `Portfolio`, and ensure trigger history row type matches backend.
- [ ] Add server actions: `searchStocks`, `getTriggerHistoryByTicker`, `listPortfolios`.
- [ ] Add hooks: `useStockSearch`, `useTriggerHistoryByTicker`, `usePortfolios`.
- [ ] Create route `app/alerts/history/page.tsx`.
- [ ] Update sidebar: set “Alert History” to `/alerts/history`.

### Next.js – Tab 1 (Search by Ticker/Company)

- [ ] Search input + button; read `?ticker=` from URL and pre-fill / auto-search.
- [ ] Company results (dropdown if multiple) and company summary cards.
- [ ] Section “Active alerts for this ticker” using alerts list filtered by ticker.
- [ ] Section “Alert trigger history”: toggle (all evaluations vs triggers only), days input, metrics row, list of events in expanders, optional stats table.
- [ ] Popular tickers and “How to use” when no search.
- [ ] Format condition details (JSON → readable bullets).

### Next.js – Tab 2 (Browse with Filters)

- [ ] Filters: exchange, economy, action, in-any-portfolio, specific portfolios, conditions, trigger status, date range (or days).
- [ ] “Apply Filters” and result count/caption.
- [ ] Alert cards/expanders with stock info, alert details, conditions, “View Full History” linking to `?tab=search&ticker=...`.
- [ ] Wire “Triggered Only” / “Not Triggered” using audit summary data (and optional date range if backend supports it).

### Polish

- [ ] Loading and error states for all async data.
- [ ] Empty states (no search results, no alerts for ticker, no triggers, no alerts matching filters).
- [ ] Optional: persist filter state in URL so “View Full History” and back keeps filters.

---

## 8. Notes

- **Economy (rbics_economy):** Streamlit enriches alerts from `stock_metadata`. If the Alert proto or ListAlerts response does not include economy, either add it in the backend when returning alerts or fetch stock metadata (e.g. SearchStocks or a batch “get metadata for these tickers”) and merge client-side.
- **Conditions filter:** Streamlit builds `all_conditions` from unique `cond.conditions` across alerts. In Next.js, derive the same list from the alerts array and use it in the multiselect.
- **Alert name in history:** Backend can return `alert_name` in each trigger history row by resolving `alert_id` to the name (from alerts list or DB); otherwise the client can maintain a small map from alertId→name from the alerts list when rendering.
- **Timezone:** Use the same date/time formatting as the rest of the app (e.g. `formatAuditDateTime`) for consistency.

Once the new RPCs and actions are in place, the page is primarily layout and wiring existing and new hooks to the two-tab UI described above.

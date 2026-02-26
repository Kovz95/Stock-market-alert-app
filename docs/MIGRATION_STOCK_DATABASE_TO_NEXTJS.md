# Migration Plan: Stock Database Page (Streamlit → Next.js)

This document outlines a plan to migrate the Streamlit **Stock Database** page (`pages/Stock_Database.py`) to the existing Next.js app at `apps/ui/web/`. The sidebar already links to `/database/stock`; this route does not exist yet.

---

## 1. Current Streamlit Page Summary

### 1.1 Data sources

| Source | Purpose |
|--------|---------|
| `fetch_stock_metadata_df()` | Full `stock_metadata` table (Postgres). Python uses `SELECT * FROM stock_metadata`; ETF fields may be in `raw_payload` and need flattening (see `src/utils/utils.py` and `fetch_stock_metadata_map`). |
| `database_filters.json` | Optional filter defaults/summaries (via `json_bridge` → document store). Used for total_symbols, total_stocks, total_etfs, sectors, asset_classes. |
| `get_country_display_name`, `get_exchange_display_name` | From `src/utils/reference_data` — human-readable labels for country/exchange codes. |

### 1.2 UI sections (in order)

1. **Title** – "Stock Database" + total symbol count.
2. **Database overview statistics** – 5 metrics: Total Symbols, Exchanges, Countries, Stocks, ETFs.
3. **RBICS industry breakdown** – For stocks: economy counts in two columns + 3 summary metrics (total categories, largest, smallest).
4. **ETF analysis** – Asset class, focus, niche counts (top 8 each) in 3 columns.
5. **Top exchanges / top countries** – Top 10 each with counts and percentages.
6. **Sidebar filters** – Country, Exchange, Asset Type (All/Stocks/ETFs), then either:
   - **Stock filters**: Economy → Sector → Subsector → Industry Group → Industry → Subindustry (cascading multiselects).
   - **ETF filters**: Issuer, Asset Class, Focus, Niche (multiselects).
   - "Clear All Filters" button.
7. **Summary metrics row** – Total Symbols, Stocks, ETFs, and either Sectors / Asset Classes / Categories.
8. **Symbol list** – Search box + data table with:
   - Columns: Symbol, name, asset_type, exchange, country, isin; plus RBICS columns for stocks and ETF columns (etf_issuer, asset_class, etf_focus, etf_niche, expense_ratio, aum) when relevant.
   - Download CSV.
   - Tall dataframe (~100 rows visible).
9. **Statistics** – Asset type distribution, top sectors/asset classes, top economies/issuers (3 columns).
10. **Exchange breakdown** – All exchanges with counts in 3 columns + total.

### 1.3 Key behaviors

- **Cascading filters**: e.g. sectors depend on selected economies; subsectors on sectors; exchanges can be limited by selected countries.
- **Asset-type split**: Stocks = `rbics_economy` present and `etf_issuer` absent; ETFs = `etf_issuer` present. Filtering uses `asset_type` (Stock/ETF) and RBICS/ETF fields.
- **Search**: Client-side text search across all columns of the filtered dataset.
- **Display columns**: Change by asset type (stocks vs ETFs get different RBICS/ETF columns).

---

## 2. Backend / API Gaps

The Next.js app currently gets stock metadata only via the **price** gRPC service:

- **Proto**: `proto/price/v1/price.proto` – `StockMetadata` has only `symbol`, `name`, `exchange`, `isin`.
- **Use**: Price Database page uses this for ticker/exchange filters and display.

The **Stock Database** page needs the **full** metadata used by the Streamlit app:

- **From table**: symbol, isin, name, exchange, country, rbics_* (6 columns), closing_price, market_value, sales, avg_daily_volume, data_source, last_updated, asset_type, raw_payload.
- **From raw_payload** (flattened): etf_issuer, etf_asset_class (or asset_class), etf_focus, etf_niche, expense_ratio, aum (see `src/utils/utils.py` and how `fetch_stock_metadata_map` merges `raw_payload`).

So the migration depends on exposing **full stock metadata** (table + flattened ETF fields) to the Next.js app.

---

## 3. Recommended Backend Approach

**Option A – New gRPC RPC (recommended if you want one stack)**  
- Add e.g. `GetFullStockMetadata` to the price (or a new “metadata”) service.
- Return a list of messages that include all table columns plus flattened ETF fields from `raw_payload`.
- Implement in Go using existing DB access; keep parity with Python’s `fetch_stock_metadata_df` / map (including flattening of `raw_payload`).

**Option B – REST API**  
- Add a Next.js API route (e.g. `app/api/stock-metadata/route.ts`) that calls the Python backend (e.g. FastAPI) or reads from Postgres via a shared client.
- Python/FastAPI can reuse `metadata_repository.fetch_stock_metadata_df()` and a small helper to flatten `raw_payload` into columns, then return JSON.

**Option C – Direct DB from Next.js**  
- Next.js server actions or API routes connect to Postgres (e.g. via Prisma/Drizzle or `pg`) and run a query that returns all columns + flattened JSON fields (e.g. `raw_payload->>'etf_issuer'`, etc.). No Python in the loop.

For **reference data** (country/exchange display names), either:

- Replicate the mapping in TypeScript (from `src/utils/reference_data`), or  
- Expose a small REST/gRPC endpoint that returns display names for country/exchange codes.

---

## 4. Next.js Implementation Plan

### 4.1 Route and layout

- **Route**: `apps/ui/web/app/database/stock/page.tsx` (matches sidebar link `/database/stock`).
- **Layout**: Reuse existing app layout and sidebar; no new layout needed.

### 4.2 Data layer

1. **Full stock metadata**
   - Add a server action or API call that returns **full** stock metadata (table + flattened ETF fields).
   - Define a TypeScript type (e.g. `FullStockMetadataRow`) matching the Streamlit dataframe columns (symbol, name, exchange, country, isin, asset_type, rbics_*, etf_issuer, asset_class, etf_focus, etf_niche, expense_ratio, aum, etc.).
   - Use React Query (e.g. `useStockDatabaseMetadata()` or `useFullStockMetadata()`) with a 5–10 minute stale time; cache key e.g. `["stock-database", "metadata"]`.

2. **Database filters (optional)**
   - `database_filters.json` is a legacy document-store file (see `src/data_access/json_bridge.py`) that the Streamlit page used for optional precomputed totals/sectors/asset_classes. It is **not** exposed by the new `GetFullStockMetadata` RPC. You can ignore it: compute totals and category counts from the full metadata response on the client instead.

3. **Reference data**
   - Add a small module or API that maps country/exchange codes to display names (port from `reference_data` or call a tiny backend).
   - Use it wherever you show country or exchange labels.

### 4.3 Page structure (mirroring Streamlit)

Use a **single page** with sections. Suggested structure:

| Section | Component / location | Notes |
|--------|------------------------|--------|
| Title + description | Top of page | "Stock Database" + total symbol count. |
| Overview stats | `StockDatabaseStatsCards` | 5 metrics: Total Symbols, Exchanges, Countries, Stocks, ETFs. |
| RBICS breakdown | `RbicsBreakdownSection` | Two-column list + 3 summary metrics; only when stocks exist. |
| ETF analysis | `EtfAnalysisSection` | 3 columns: asset class, focus, niche (top 8 each). |
| Top exchanges / countries | `TopExchangesSection` | Top 10 exchanges and top 10 countries with counts and %. |
| Sidebar (or collapsible panel) | `StockDatabaseFilters` | Country, Exchange, Asset Type, then Stock or ETF filters; Clear button. |
| Summary metrics | Same area as filters or above table | Total Symbols, Stocks, ETFs, Sectors/Asset Classes/Categories. |
| Symbol list | `StockDatabaseTable` | Search + table + CSV download. |
| Statistics | `StockDatabaseStatistics` | Asset type distribution, top sectors/issuers/economies. |
| Exchange breakdown | `ExchangeBreakdownSection` | All exchanges in 3 columns + total. |

Layout: **grid with sidebar** (e.g. `grid-cols-[280px_1fr]` like Price Database) so filters stay on the left and main content on the right.

### 4.4 Components to build

1. **`StockDatabaseFilters`**
   - State: selected countries, exchanges, asset type, and (for stocks) economy, sector, subsector, industry group, industry, subindustry; (for ETFs) issuer, asset class, focus, niche.
   - Cascading options: e.g. exchange options depend on selected countries; sector options on economies; etc. Compute from the full metadata array in memory (or from a pre-aggregated API).
   - "Clear all" resets state.

2. **`StockDatabaseTable`**
   - Input: filtered array of `FullStockMetadataRow`.
   - Search: client-side filter by search term across visible string fields.
   - Columns: dynamic by asset type (RBICS vs ETF columns); use TanStack Table (like `PriceDataTable`) with column visibility.
   - CSV download: serialize filtered data to CSV and trigger download (same pattern as Price Database export).

3. **Stats and breakdown components**
   - Pure presentational: receive aggregated data or the full list and compute counts/percentages in the component or in a small helper.
   - Use existing UI primitives (cards, badges, etc.) to match the rest of the app.

### 4.5 Filtering and aggregation

- **Filtering**: Apply all sidebar selections (country, exchange, asset type, RBICS hierarchy, ETF filters) in memory on the client, or via a single server action that accepts filter params and returns filtered rows. Client-side is simpler and matches Streamlit; server-side is better if the dataset is huge.
- **Aggregations**: Compute overview stats, RBICS counts, ETF breakdowns, top exchanges/countries, and exchange breakdown from the same full dataset (or from a dedicated “stats” response). Prefer doing this once when metadata is loaded and then updating when filters change (still client-side) unless you introduce a dedicated stats API.

### 4.6 Reference data and CSV

- **Country/Exchange labels**: Use the reference module or API wherever you render country or exchange (table, filters, breakdowns).
- **CSV export**: Include the same columns as the visible table (and optionally all columns); format numbers/dates consistently (e.g. expense_ratio, aum).

---

## 5. Implementation Order

1. **Backend**
   - Implement full stock metadata API (gRPC or REST) with flattened `raw_payload` and define the response shape.
   - (Optional) API or TS module for country/exchange display names.
   - (Optional) Endpoint or logic for `database_filters.json` if you still want it.

2. **Next.js data layer**
   - Add server action or fetch helper + TypeScript types for full metadata.
   - Add React Query hook(s) for metadata (and optionally filters).
   - Add reference-data helper or hook for display names.

3. **Layout and shell**
   - Create `app/database/stock/page.tsx` with title, description, and layout (sidebar + main).
   - Wire in the metadata hook and show a loading/error state.

4. **Filters**
   - Implement `StockDatabaseFilters` with full state and cascading options; connect to page state and pass filtered list to the rest of the page.

5. **Stats and breakdowns**
   - Implement overview stats, RBICS section, ETF section, top exchanges/countries, and exchange breakdown; feed them from the full (or filtered) metadata.

6. **Table and export**
   - Implement `StockDatabaseTable` with search, columns by asset type, and CSV download.
   - Add the small “Statistics” block and exchange breakdown below.

7. **Polish**
   - Loading skeletons, empty states, and error messages.
   - Ensure sidebar link `/database/stock` is correct and the page is reachable.

---

## 6. Files to Create / Modify

### New files (suggested)

- `apps/ui/web/app/database/stock/page.tsx` – main page.
- `apps/ui/web/app/database/stock/_components/StockDatabaseFilters.tsx`
- `apps/ui/web/app/database/stock/_components/StockDatabaseTable.tsx`
- `apps/ui/web/app/database/stock/_components/StockDatabaseStatsCards.tsx`
- `apps/ui/web/app/database/stock/_components/RbicsBreakdownSection.tsx`
- `apps/ui/web/app/database/stock/_components/EtfAnalysisSection.tsx`
- `apps/ui/web/app/database/stock/_components/TopExchangesSection.tsx`
- `apps/ui/web/app/database/stock/_components/ExchangeBreakdownSection.tsx`
- `apps/ui/web/app/database/stock/_components/StockDatabaseStatistics.tsx`
- `apps/ui/web/app/database/stock/_components/index.ts`
- `apps/ui/web/actions/stock-database-actions.ts` (or extend price-database-actions) – server actions for full metadata and optional filters.
- `apps/ui/web/lib/hooks/useStockDatabase.ts` – React Query hook(s) for metadata.
- `apps/ui/web/lib/reference-data.ts` (or similar) – country/exchange display names.

### Existing files to touch

- `apps/ui/web/components/app-sidebar.tsx` – ensure "Stock" points to `/database/stock` (already does).
- Backend: price service (or new service) – add full metadata RPC and, if needed, flattening of `raw_payload` in Go; or add REST endpoint + Python/TS implementation as chosen.

---

## 7. Data Shape Reference (Full metadata row)

Use this as the target TypeScript interface so the backend and frontend stay aligned. Align with `stock_metadata` table + flattened `raw_payload` from Python.

```ts
interface FullStockMetadataRow {
  symbol: string;
  name: string | null;
  exchange: string | null;
  country: string | null;
  isin: string | null;
  asset_type: string | null;
  rbics_economy: string | null;
  rbics_sector: string | null;
  rbics_subsector: string | null;
  rbics_industry_group: string | null;
  rbics_industry: string | null;
  rbics_subindustry: string | null;
  etf_issuer: string | null;
  asset_class: string | null;   // ETF asset class (from raw_payload)
  etf_focus: string | null;
  etf_niche: string | null;
  expense_ratio: number | null;
  aum: number | null;
  closing_price?: number | null;
  market_value?: number | null;
  last_updated?: string | null;
  // ... other table columns if needed for display
}
```

---

## 8. Summary

- **Streamlit page**: Rich filters (country, exchange, asset type, RBICS, ETF), overview stats, RBICS/ETF breakdowns, searchable table, CSV export.
- **Gap**: Next.js currently only has minimal stock metadata (symbol, name, exchange, isin) from the price gRPC service. Full metadata (table + flattened ETF from `raw_payload`) must be exposed by the backend.
- **Plan**: Add full metadata API, then implement the Stock Database page in Next.js under `/database/stock` with filters, stats, breakdowns, table, and export, reusing existing UI patterns (e.g. Price Database) and adding reference data for country/exchange labels.

Once the backend exposes full stock metadata and optional reference/filters data, the migration can proceed in the order above with minimal dependency on the existing Streamlit app.

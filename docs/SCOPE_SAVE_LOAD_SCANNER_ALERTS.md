# Implementation Scope & Time Estimate

**Prepared for:** Software contractor scoping  
**Date:** February 26, 2026  
**App:** Stock Market Alert (Streamlit scanner, alerts, DB)

---

## Summary Table

| # | Item | Scope | Est. Hours |
|---|------|--------|------------|
| 1 | Scanner: Save/Load configuration (one-click load, conditions + stock list) | Medium | 3–5 |
| 2 | Indicators: Add conditions in Scanner + Add Alert (per videos) | Medium–High | 6–10 |
| 3 | Scanner: Fix filter count (country + exchange “matches”) | Small | 1–2 |
| 4 | Alerts page: Remove buy/sell labels | Small | 0.5–1 |
| 5 | Delete remaining futures data from DB | Small | 1 |
| 6 | Scanner: Scan over X-day period (e.g. last 100 days, per day) | High | 8–14 |
| 7 | New indicator: MA slope + convexity | Medium (spec TBD) | 4–8 |

**Total (range):** **23.5–41 hours** (about **3–5 days** at 8 h/day, or **1–2 sprints** with buffer).

---

## 1. Scanner: Save/Load configuration

**Goal:** Load scans in one click: restore conditions **and** the effective list of stocks (from filters), with minimal steps.

**Current behavior (from codebase):**
- Save already persists: `conditions`, `logic`, `timeframe`, `scan_mode`, and `filters` (portfolio, asset_types, countries, exchanges, economies, sectors, subsectors, industry_groups, industries, subindustries).
- Load restores those into session state and reruns; filters then drive which symbols are scanned (no separate “symbol list” is saved).
- Persistence uses `saved_scans.json` (via `json_bridge` → document store key `saved_scans`).
- Pair Trading mode also saves/loads `pair_symbols`.

**Planned work:**
- **One-click load:** Add a “Quick load” area: e.g. list of saved scans as buttons (or a single dropdown + “Load” that applies and reruns immediately). Ensure one click applies the chosen scan and runs the UI with that config (no extra step).
- **Conditions + “list of stocks”:** The “list of stocks” is currently implied by filters (portfolio + asset type + country + exchange + RBICS). Ensure load restores **all** of these so that after load, “Symbols matching filters” reflects the same universe as when the scan was saved. No separate “custom symbol list” exists today; if a custom-symbol-list mode is added later, save/load of that list can be included in the same config.
- **Edge cases:** Handle missing keys in saved config (defaults), and ensure Load works when `saved_scans` is read from the document store (not only from a local file).

**Estimate:** 3–5 hours (UI for one-click load, verification of all filter keys, document-store path, light testing).

---

## 2. Indicators: Add conditions in Scanner and Add Alert page

**Goal:** Add condition types/options for some indicators in both the **Scanner** and **Add Alert** pages, per the two videos (videos not attached; spec to be confirmed with Ryan).

**Current behavior:**
- **Scanner** (`pages/Scanner.py`): Condition builder uses dropdowns by category (Price Data, Moving Averages, RSI, MACD, Bollinger Bands, Volume, ATR, CCI, Williams %R, ROC, EWO, MA Z-Score, HARSI, OBV MACD, SAR, SuperTrend, Trend Magic, Ichimoku Cloud, Kalman ROC Stoch, Pivot S/R, Donchian Channels, Custom).
- **Add Alert** (`pages/Add_Alert.py`): Similar indicator/condition UI and syntax; backend uses `supported_indicators` in `src/utils/utils.py` and evaluation in `src/services/backend.py`.
- **Doc:** `docs/add_alert_indicator_catalog.md` lists condition types and presets.

**Planned work:**
- Identify which indicators need **new or extended conditions** (from videos/spec).
- Implement those condition types in:
  - Scanner: condition builder dropdowns and the string expressions that get appended to `scanner_conditions`.
  - Add Alert: same condition builder and expression format.
- Ensure backend `evaluate_expression_list` / `indicator_calculation` and any parser support the new forms (or extend them).
- Keep `add_alert_indicator_catalog.md` (and any Scanner help) in sync.

**Estimate:** 6–10 hours (depends on how many indicators and how many new condition types; clarify with Ryan/videos).

---

## 3. Scanner: Filters not adding up (e.g. country + exchange → “18 matches”)

**Goal:** Fix the “Symbols matching filters” count so it correctly reflects the selected country and exchange filters.

**Current behavior:**
- `_count_filtered_symbols()` in `pages/Scanner.py` applies: portfolio, asset_type, **country**, **exchange**, then RBICS filters. Logic is AND: symbol must match all selected filter sets.
- Same logic is used to build `symbols_to_scan` when running the scan.
- Metadata comes from `fetch_stock_metadata_map()` → `stock_metadata` (columns include `country`, `exchange`).

**Possible causes of wrong count:**
- **Normalization:** Mismatch between displayed filter options and stored values (e.g. “United States” vs “USA”, or “NYSE American” vs “NYSE AMERICAN”). Fix by normalizing when comparing (e.g. strip, case-fold) or by canonicalizing values when building filter options and when saving/loading.
- **Column names:** If metadata dict keys are not `country`/`exchange` (e.g. `Country`/`Exchange` from DB), the count would be wrong. Code uses `info.get('country')` and `info.get('exchange')`; ensure these match the actual keys from `fetch_stock_metadata_map()`.
- **Data:** Only 18 rows in `stock_metadata` might actually have that exact country + exchange combination; then the count would be “correct” but surprising. Optional: add a small debug or tooltip (e.g. “Count uses: country in […] and exchange in […]”) so users can verify.

**Planned work:**
- Verify metadata keys and value samples for `country` and `exchange`.
- Normalize filter comparison (e.g. strip, case-insensitive) for country and exchange so “United States” and “United States ” (or different casing) are treated the same.
- Optionally add a short debug/explainer near the count.
- Re-test with Country = United States, Exchange = NYSE, NASDAQ, NYSE American and confirm count is plausible (or fix data/display).

**Estimate:** 1–2 hours.

---

## 4. Alerts page: Remove buy/sell labels

**Goal:** Remove buy/sell (action) from the Alerts (Alert History) page: no Action filter and no Action in card/expander text.

**Current behavior (from `pages/Alert_History.py`):**
- Filter: “Action” multiselect with options from `all_actions` (buy/sell), label “Filter by buy/sell action”.
- Card/expander: “**Action:** {action}” and expander title includes `{action}`.
- Alerts still have an `action` field (set on Add Alert as “Buy” or “Sell”).

**Planned work:**
- Remove the “Action” filter (multiselect and its use in `filter_alerts`).
- Remove “Action” from the alert card and from the expander title/body (no “Buy”/“Sell” labels on the page).
- Optionally keep storing `action` in the DB for backward compatibility; only the UI is changed.

**Estimate:** 0.5–1 hour.

---

## 5. Delete remaining futures data from DB

**Goal:** Remove remaining futures-related data from the application database.

**Current behavior:**
- Futures are persisted via the document store (`src/data_access/document_store.py`) and the JSON bridge (`src/data_access/json_bridge.py`).
- Document keys used for futures: `futures_alerts`, `futures_database`, `futures_scheduler_config`, `futures_scheduler_status`, `ib_futures_config`.
- These map to `app_documents` in PostgreSQL (and optionally Redis cache).

**Planned work:**
- Delete documents for: `futures_alerts`, `futures_database`, `futures_scheduler_config`, `futures_scheduler_status`, `ib_futures_config` (and any other futures-related keys if discovered).
- Use the existing document store API (e.g. `delete_document` if present, or overwrite with empty/default and then remove).
- Optionally provide a small script under `scripts/maintenance/` (e.g. `delete_futures_documents.py`) so the operation is repeatable and documented.
- Do **not** remove futures code paths from the repo unless requested; only clear stored data.

**Estimate:** 1 hour.

---

## 6. Scanner: Scan over X-day period (e.g. last 100 days, each day)

**Goal:** Let the user enter a number of days (e.g. 100) and run the same scan **as of the last bar of each day** over that period, then see which symbols matched on which days (or aggregated).

**Current behavior:**
- Scan runs once: for each symbol, get price data, compute indicators, evaluate conditions on the **latest** bar (e.g. `Close[-1]`, `…[-1]`).
- No historical “as-of” date or multi-day scan.

**Planned work:**
- **UI:** Input for “Scan over last N days” (e.g. number input, default off or 1 for “today only”).
- **Logic:** For each day in the range (e.g. last 100 trading days), for each symbol:
  - Get price data up to that day’s close (or use a cached/sampled series).
  - Evaluate the same conditions **as of that day** (e.g. `Close[-1]` = that day’s close).
  - Record symbol × date matches.
- **Output:** Table or export: symbol, list of dates when it matched (and optionally condition values). Optionally: “matched on N of last M days”, or a matrix symbol × date.
- **Performance:** 100 days × thousands of symbols can be heavy; consider batching, caching, and progress feedback. May need to limit max days or symbols in a single run.

**Estimate:** 8–14 hours (design of as-of evaluation, data access for historical bars, UI, aggregation, performance and limits).

---

## 7. New indicator: Slope and convexity of moving average line

**Goal:** Add one new indicator that uses the **slope** and **convexity** of a moving average line. Exact formula and inputs to be provided by Ryan.

**Current behavior:**
- `src/utils/utils.py` has `supported_indicators` including `slope_sma`, `slope_ema`, `slope_hma` (slope-only).
- Indicators are used in conditions in Scanner and Add Alert; backend evaluates via `indicator_calculation` and expression parsing.

**Planned work:**
- Implement the new indicator (e.g. “MA slope and convexity” or separate “MA convexity”) in the indicators layer (e.g. `src/utils/indicators.py` or equivalent).
- Register it in `supported_indicators` and wire it into the backend evaluator.
- Add condition types in Scanner and Add Alert (e.g. “MA slope & convexity” with operator vs threshold, or crossover-style, per spec).
- Add syntax to the indicator catalog/help.

**Estimate:** 4–8 hours (depends on exact math and how many condition variants; spec TBD from Ryan).

---

## Assumptions and risks

- **#2 (Indicators):** Scope depends on the two videos; if many indicators and many new condition types, estimate can move toward the high end.
- **#3 (Filter count):** If the issue is incomplete or inconsistent data in `stock_metadata`, fixing may require a small data cleanup or migration (not included in 1–2 h).
- **#6 (X-day scan):** Assumes price data is available historically (e.g. from DB or API) for each “as-of” date; if not, backfill or data pipeline work would be extra.
- **#7 (MA slope/convexity):** Pending exact definition from Ryan; estimate is placeholder until then.

---

## Suggested order of implementation

1. **#4** – Remove buy/sell labels (quick win).  
2. **#5** – Delete futures data (cleanup).  
3. **#3** – Fix scanner filter count (unblocks trust in “matches”).  
4. **#1** – Save/load one-click and conditions + filters (high value).  
5. **#2** – Indicator conditions (after videos/spec clarified).  
6. **#7** – New MA slope/convexity indicator (after spec from Ryan).  
7. **#6** – X-day period scan (largest feature).

---

*Document generated from codebase review. Adjust estimates after reviewing the two videos and Ryan’s spec for the new indicator.*

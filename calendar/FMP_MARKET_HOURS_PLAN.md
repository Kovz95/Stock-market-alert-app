# Plan: Use FMP API for Exchange Market Hours and Open/Closed Status

This document outlines how to replace the current calendar logic (hardcoded schedules in Go, `exchange_calendars` in Python) with the Financial Modeling Prep (FMP) **Market Hours** and **Holidays** APIs, and how to use a **single call** to the All Exchange Market Hours API before each scheduler run.

---

## 1. FMP APIs to Use

| API | Endpoint | Purpose |
|-----|----------|---------|
| **All Exchange Market Hours** | `GET https://financialmodelingprep.com/stable/all-exchange-market-hours` | One call per scheduler cycle: get `isMarketOpen`, `timezone`, `openingHour`, `closingHour` for every exchange. |
| **Exchange Market Hours** | `GET https://financialmodelingprep.com/stable/exchange-market-hours?exchange={exchange}` | Per-exchange details when needed (e.g. UI, diagnostics, or fallback if all-exchange omits an exchange). |
| **Holidays by Exchange** | `GET https://financialmodelingprep.com/stable/holidays-by-exchange?exchange={exchange}&from=YYYY-MM-DD&to=YYYY-MM-DD` | Exchange holidays for next daily run time, early closes, and UI (holiday calendar). |

**Authentication:** Use the same `FMP_API_KEY` (query param `apikey=`) as existing FMP usage in the repo. Confirm that the `/stable/` base path accepts the same key; if not, document the alternative.

---

## 2. Single-Call Strategy: All Exchange Market Hours Before Each Scheduler Run

### Goal

Before each 15-minute scheduler cycle (Go: `schedule.scheduleAll`), make **one** HTTP request to:

```
GET https://financialmodelingprep.com/stable/all-exchange-market-hours?apikey=<FMP_API_KEY>
```

Use the response to:

1. **Determine if an exchange is open** — Use FMP’s `isMarketOpen` for each exchange instead of computing it locally. No need to convert open/close to UTC for this specific check.
2. **Optionally derive UTC open/close** — For “next daily run” and UI, use FMP’s `timezone`, `openingHour`, and `closingHour` to build today’s open/close in that timezone, then convert to UTC for comparison with “current time UTC”.

### Response Shape (per exchange)

```json
{
  "exchange": "NASDAQ",
  "name": "NASDAQ Global Market",
  "openingHour": "09:30 AM -04:00",
  "closingHour": "04:00 PM -04:00",
  "timezone": "America/New_York",
  "isMarketOpen": true
}
```

- **`isMarketOpen`** — Use as-is for “is this exchange open right now?” in the scheduler and anywhere else that only needs a boolean.
- **`timezone`** — IANA name (e.g. `America/New_York`). Use it to interpret `openingHour`/`closingHour` and convert to UTC when needed.
- **`openingHour` / `closingHour`** — Time strings with offset (e.g. `"09:30 AM -04:00"`). For “next close” or “next daily run” we need “today’s” (or next trading day’s) open/close in UTC; that requires parsing these and applying the exchange’s `timezone` for the relevant date (and optionally cross-checking with Holidays API for early close / closed days).

### Using FMP Timezone to Convert to UTC

- **Current time:** We already have `now` in UTC (e.g. `time.Now().UTC()` in Go, `datetime.now(timezone.utc)` in Python).
- **Interpret open/close in exchange local time:**  
  - In the exchange’s `timezone`, take “today” (or next trading day) and the parsed open/close time to form two timestamps (open and close) in that timezone.  
  - Convert those timestamps to UTC.  
  - Compare `now_utc` with `open_utc` and `close_utc` to implement “is open” locally if we ever need to (e.g. offline fallback); for the primary path we use FMP’s `isMarketOpen`.
- **Parsing `openingHour` / `closingHour`:**  
  - Format is like `"09:30 AM -04:00"` or `"04:00 PM +10:00"`. Parse time part and offset; the offset can be used to validate or build the local time in the given IANA `timezone` for the correct date (accounting for DST via the timezone).

### Caching and Failure Handling

- **Cache:** Cache the “all exchange market hours” response for the duration of the scheduler cycle (e.g. up to 15 minutes). That way:
  - One call per cycle.
  - All `IsExchangeOpen(exchange, now)`-style checks in that cycle use the same snapshot.
- **Failure:** If the FMP call fails (timeout, 4xx/5xx, invalid JSON):
  - **Fallback:** Use existing logic (Go: current `calendar.IsExchangeOpen` and schedules; Python: `exchange_calendars` or existing fallback in `calendar_adapter`).
  - Log clearly so we can monitor FMP availability.
- **Missing exchange:** If our list of exchanges includes a code that FMP doesn’t return (e.g. different naming), treat as “closed” or fall back to local schedule for that exchange only.

---

## 3. Exchange Code Mapping (FMP ↔ Internal)

FMP uses short symbols (e.g. `NASDAQ`, `NYSE`, `ASX`). Our codebase uses names like `"NYSE"`, `"NASDAQ"`, `"LONDON"`, `"OMX NORDIC STOCKHOLM"`, etc.

- **Action:** Build a mapping **FMP exchange code → internal exchange key** (and optionally the reverse). Use it when reading the all-exchange response so that `calendar.IsExchangeOpen("LONDON", now)` looks up the FMP entry that corresponds to London (e.g. `LSE` or whatever FMP returns).
- **Discovery:** On first implementation, call `all-exchange-market-hours` once and record the exact `exchange` values FMP returns; then define the mapping for all exchanges we care about. Some may be 1:1 (e.g. `NASDAQ` → `"NASDAQ"`), others may need a table (e.g. `XETR` → `"XETRA"`).

---

## 4. Where “Is Open” and Calendar Are Used

### Go (scheduler and gRPC)

- **`apps/scheduler/internal/schedule/scheduler.go`**  
  - Every 15 min: `scheduleAll` calls `calendar.IsExchangeOpen(exchange, now)` to decide whether to enqueue hourly tasks.  
  - **Change:** Pass in a “snapshot” from the single all-exchange call (e.g. `map[exchange]bool` or a small struct with `IsOpen` and optional UTC bounds). Prefer `calendar.IsExchangeOpenFromSnapshot(exchange, snapshot)` or inject a provider that returns open/closed from the cached FMP response.
- **`calendar.GetNextDailyRunTime`**  
  - Used for `ProcessAt(nextDaily)` for daily/weekly tasks.  
  - **Options:** (A) Keep computing “next close + 40 min” locally using FMP’s timezone and parsed open/close for the next trading day (and optionally Holidays API for early close / closed days). (B) Or keep current logic as fallback and only switch to FMP-derived “next run” in a later phase.
- **`apps/scheduler/internal/handler/common.go`**  
  - `calendar.IsExchangeOpen(exchange, runTime)` to decide if an hourly job should run.  
  - **Change:** Use same snapshot or FMP-backed API so that “is open” is consistent with the scheduler.
- **`apps/grpc/scheduler_service/schedule_handler.go`**  
  - Uses `calendar.IsExchangeOpen`, `GetCalendarTimezone`, `GetExchangeCloseTime`, `GetNextDailyRunTime`, `GetHourlyAlignment` for status and schedule tables.  
  - **Change:** Is-open and timezone can come from FMP cache or per-exchange API; close time / next run can stay as today or use FMP + holidays.
- **`apps/grpc/alert_service/evaluate_handler.go`**  
  - `calendar.IsExchangeOpen(exchange, time.Now())` for hourly alerts.  
  - **Change:** Use FMP-backed is-open (from cache or a small service).

### Python (Streamlit and jobs)

- **`src/services/calendar_adapter.py`**  
  - `is_exchange_open`, `get_session_bounds`, `get_calendar_timezone`, `get_next_daily_run_time`, `get_market_open_close_times`.  
  - **Change:** Add FMP as primary source: call all-exchange (or per-exchange) and use `isMarketOpen` and timezone/open/close; keep `exchange_calendars` and existing fallbacks when FMP is unavailable or exchange not in FMP.
- **`src/services/hourly_data_scheduler.py`**  
  - Uses `calendar_adapter.is_exchange_open` and `any_market_open()`.  
  - **Change:** No API change if `calendar_adapter` speaks FMP under the hood.
- **`src/services/scheduler_job_handler.py`**  
  - Uses `is_exchange_open` before running hourly work.  
  - **Change:** Same as above.
- **Pages:** `Market_Hours.py`, `Daily_Weekly_Scheduler_Status.py`, `Hourly_Scheduler_Status.py`, `Price_Database.py` use calendar_adapter; they benefit from FMP once the adapter uses it.

---

## 5. Holidays by Exchange

- **Use:**  
  - Next daily run: skip closed days and respect early-close days (use `adjOpenTime`/`adjCloseTime` when present).  
  - UI: show “Exchange holidays” and “early close” for the next N days.
- **Caching:** Cache per exchange and date range (e.g. “from today to today+1 year”). Refresh daily or when building next-run schedule.
- **Parameters:** `exchange`, `from`, `to` (required for range). Use same exchange code mapping as market hours.

---

## 6. Implementation Phases

### Phase 1: FMP client and “all exchange” snapshot (Go + Python)

1. **Go**
   - Add a small FMP “calendar” client that calls `all-exchange-market-hours` (and optionally `exchange-market-hours` and `holidays-by-exchange`) with existing `FMP_API_KEY`.
   - Define response structs and exchange-code mapping (FMP symbol → internal key).
   - In the scheduler, at the start of `scheduleAll`:
     - Call FMP all-exchange once (or use a 15-minute cache).
     - Build `map[internalExchange]FMPMarketHours` (or similar) including `isMarketOpen`.
   - Add `calendar.IsExchangeOpenFromSnapshot(exchange, snapshot)` (or equivalent) and use it in `scheduleAll` and in the handler that checks “should run” for hourly. Keep existing `IsExchangeOpen` as fallback when snapshot is missing or FMP fails.

2. **Python**
   - Add FMP market-hours helpers (e.g. in `calendar_adapter` or a small `fmp_market_hours.py`): `fetch_all_exchange_market_hours()`, `is_exchange_open_from_fmp(snapshot, exchange)`.
   - In `calendar_adapter.is_exchange_open`, try FMP first (using a short-lived cache of the all-exchange response), then fall back to `exchange_calendars` / existing logic.
   - Use FMP’s `timezone` where we currently use config timezone for that exchange.

### Phase 2: Use FMP timezone and open/close for UTC logic

1. Parse `openingHour` / `closingHour` and, using FMP’s `timezone`, compute “today’s” (or next trading day’s) open and close in UTC.
2. Use these UTC bounds where we currently use local session open/close (e.g. for diagnostics, `ExchangeOpenStatus`, or an optional “next close” time). Keep existing `GetNextDailyRunTime` logic as fallback.
3. Optionally integrate **Holidays by Exchange**: when computing “next run”, skip closed days and use `adjCloseTime` when provided.

### Phase 3: Next daily run and holidays

1. **Next daily run:** Use FMP holidays + (optionally) FMP open/close to compute “next session close + 40 min” and replace or augment `GetNextDailyRunTime`.
2. **UI and APIs:** Expose FMP holidays and market hours in scheduler status and Market Hours page.

### Phase 4: Cleanup and fallback policy

1. Document when we use FMP vs fallback (e.g. “FMP primary; if FMP fails or exchange not in response, use hardcoded schedule / exchange_calendars”).
2. Remove or reduce duplicate logic (e.g. trim hardcoded open/close tables in Go if FMP is the source of truth for the exchanges we support).

---

## 7. Summary Table

| Concern | Current | After FMP |
|--------|--------|-----------|
| Is exchange open? | Go: weekday + local open/close. Python: exchange_calendars / fallback. | One call to all-exchange per cycle; use `isMarketOpen` (with fallback). |
| Exchange timezone | Go: `ExchangeTimezones` / `GetCalendarTimezone`. Python: config / exchange_calendars. | FMP `timezone` from all-exchange or per-exchange API. |
| Market hours (open/close) | Go: `localSessionOpen`/`Close`, `ExchangeSchedules`. Python: exchange_calendars / fallback. | FMP `openingHour` / `closingHour` + `timezone` → convert to UTC when needed. |
| Holidays | Not in Go; Python via exchange_calendars. | FMP Holidays by Exchange; use for next-run and UI. |
| Scheduler “single call” | N/A | One `all-exchange-market-hours` per 15-min cycle; cache and reuse for all exchanges in that cycle. |

---

## 8. Technical Notes

- **Base URL:** FMP “stable” APIs: `https://financialmodelingprep.com/stable/`. Existing price APIs use `https://financialmodelingprep.com/api/v3/`. Use the same `apikey` query parameter unless FMP docs state otherwise.
- **Rate limits:** One all-exchange call every 15 minutes is minimal; still respect FMP rate limits for holidays and per-exchange calls (e.g. cache holidays per exchange per day).
- **Testing:** Add unit tests with saved JSON for all-exchange and holidays; integration test with real API (optional, env-gated). Keep existing calendar tests and add “FMP snapshot” tests (e.g. when snapshot says NYSE open, `IsExchangeOpenFromSnapshot("NYSE", snapshot)` is true).

This plan keeps the “single call per scheduler run” approach, uses FMP’s timezone for UTC conversion where needed, and preserves fallbacks so behavior remains correct when FMP is down or an exchange is missing from FMP.

---
# Spec: Scanner Weekly Fixes (07)
---

## Goal

The market scanner's weekly timeframe silently returns false negatives. Three root causes: (1) `scanLookbackDays = 400` in `apps/grpc/price_service/scan.go` is applied as a calendar-day window to every timeframe, which on weekly yields only ~57 bars — insufficient for SMA(50)/SMA(200) and any long-period indicator, so conditions that involve them evaluate NaN-to-`false` with no user-visible error. (2) The scanner's "Lookback days" input is actually a *bar count* on the backend, so weekly users who enter `10` get 10 weeks instead of 10 days. (3) The weekly updater writes the current in-progress week with `week_ending` set to the upcoming Friday, so on any day except Friday the scanner evaluates `[-1]` against a partial bar. This spec fixes all three, plus the parallel 365-day weekly bug in the scheduler's `PreWarmCache`, and removes a stray Go file that shouldn't be in the Next.js app.

## Scope

### In scope

- Replace the single `scanLookbackDays` constant in `apps/grpc/price_service/scan.go` with per-timeframe constants (daily 400, weekly 2000, hourly 90 calendar days) and wire each `load*OHLCV` function to its matching constant.
- Filter partial current-week rows (`week_ending > today`) out of the series returned by `loadWeeklyOHLCV` so `[-1]` is always a closed weekly bar.
- Update `sinceDateForTimeframe` in `apps/scheduler/internal/handler/common.go` so weekly returns `-2000` days (aligned with scanner weekly lookback).
- Rename the scanner UI input label from "Lookback days" to "Lookback bars" and update the accompanying help text to describe the bar-count semantics. Input max stays at 250.
- Delete `apps/ui/web/app/scanner/price_updater.go` (dead Go code sitting inside the Next.js app).

### Out of scope

- Changing `scanMinBars` (stays at 50).
- Renaming the proto field `lookback_days`, the Jotai atoms `scannerLookbackDaysAtom` / `scannerLookbackInputAtom`, or the preset schema.
- Changing how `weekly_prices` rows are *written* — the partial-week row keeps being upserted; the fix is read-side only.
- Applying the partial-week filter to the scheduler's alert checker (scheduler enqueues weekly tasks only on Fridays, so the problem doesn't arise there).
- Any change to hourly or daily partial-candle handling.
- New scanner condition types or UI features.

## Source excerpts

- `apps/grpc/price_service/scan.go` — `scanLookbackDays` constant (line 19); `loadDailyOHLCV` (line 297), `loadWeeklyOHLCV` (line 321), `loadHourlyOHLCV` (line 309); `scanOneTicker` bar-range loop (lines 243–293).
- `apps/scheduler/internal/handler/common.go` — `sinceDateForTimeframe` (line 328, weekly branch at line 333–334).
- `apps/scheduler/internal/price/updater.go` — `resampleDailyToWeekly` (line 277) for how `week_ending = upcoming Friday` is assigned to in-progress weeks.
- `apps/ui/web/app/scanner/page.tsx` — `ScannerTimeframeAndLookback` component (lines 105–149) with current "Lookback days" label.
- `apps/ui/web/app/scanner/price_updater.go` — entire file to delete.

## Acceptance criteria

1. **Per-timeframe scan lookback window**
   - `apps/grpc/price_service/scan.go` no longer has a single `scanLookbackDays` constant used for all three timeframes. It has three named constants (or equivalent map lookup) with values `400` for daily, `2000` for weekly, `90` for hourly.
   - `loadDailyOHLCV` uses the daily constant; `loadWeeklyOHLCV` uses the weekly constant; `loadHourlyOHLCV` uses the hourly constant.
   - `go build ./apps/grpc/price_service/...` exits 0.

2. **Partial weekly candle filtered out of scanner**
   - `loadWeeklyOHLCV` drops any DB row whose `week_ending` is strictly greater than today (UTC) before building the `OHLCV` struct.
   - When the scanner is run on a non-Friday, the last element of `ohlcv.Dates` is the most recent *closed* Friday's date, not a future-Friday date.

3. **Scheduler weekly PreWarmCache lookback aligned**
   - `sinceDateForTimeframe("weekly")` in `apps/scheduler/internal/handler/common.go` returns `time.Now().UTC().AddDate(0, 0, -2000)`.
   - `go build ./apps/scheduler/...` exits 0.

4. **UI label rename**
   - The `<Label>` above the lookback input in `apps/ui/web/app/scanner/page.tsx` reads `Lookback bars` (not `Lookback days`).
   - Help text (inline hint, tooltip, or sibling `<p>` — placement is at implementer's discretion) explains that the value is the number of bars of the selected timeframe (e.g. "10 weekly bars = 10 weeks").
   - The input's `max={250}` constraint is unchanged.
   - The Jotai atoms `scannerLookbackDaysAtom` and `scannerLookbackInputAtom` retain their current names and shapes — no rename, no schema migration for saved presets.

5. **Stray Go file removed**
   - `apps/ui/web/app/scanner/price_updater.go` no longer exists.
   - `grep -r "price_updater" apps/ui/web/` returns no results in `app/scanner/` (other matches in other directories, if any, are left alone).

6. **Build and type checks**
   - From `apps/ui/web/`: `pnpm typecheck` exits 0 and `pnpm lint` exits 0.
   - From repo root: `go build ./...` exits 0.

7. **Behavioral validation (manual)**
   - Running the scanner on a weekday (Mon–Thu) with timeframe = Weekly and an SMA(200)-based condition now produces non-zero matches on tickers that have at least 200 weeks of weekly history, where previously zero matches were returned.
   - Running the scanner on a weekday with timeframe = Weekly and the condition `Close[-1] > 0` confirms the `match_date` in the returned rows (when lookback > 0) never includes a future-Friday date.

## Conventions

- Constant naming in `scan.go`: prefer `scanLookbackDaysDaily`, `scanLookbackDaysWeekly`, `scanLookbackDaysHourly` (or a `scanLookbackDays map[pricev1.Timeframe]int` literal) to keep grep-ability of the old name. Pick one idiom and apply consistently.
- The weekly filter must use `time.Now().UTC()` as "today" — matching the timezone semantics the DB queries already use.
- The partial-week filter is applied after the DB query returns rows (cheaper than changing SQL; `scanLookbackDays` already overshoots by ~1 week so filtering one tail row is fine).
- UI copy: match the tone of other help text in the scanner (e.g. the `ScannerFilters` component) — terse, muted-foreground color, no emojis.
- No proto changes, no new RPCs, no new fields — this spec is backend-logic and UI-copy only.
- Constants committed here must be kept in sync between `scan.go` and `common.go` for weekly: if someone later tunes one, a comment in each file should cross-reference the other. Add a short comment like `// keep aligned with apps/grpc/price_service/scan.go scanLookbackDaysWeekly`.

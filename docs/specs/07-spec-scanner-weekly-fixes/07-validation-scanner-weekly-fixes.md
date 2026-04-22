---
# Validation: Scanner Weekly Fixes (07)
---

## Automated verification

From repo root:

```bash
# Go build — must be clean across scanner gRPC service and scheduler
go build ./apps/grpc/price_service/...
go build ./apps/scheduler/...
go build ./...
```

**Expected:** All three commands exit 0 with no compilation errors.

From `apps/ui/web/`:

```bash
pnpm typecheck
pnpm lint
```

**Expected:** Both exit 0.

Structural grep checks:

```bash
# New constants exist in scan.go
grep -n "scanLookbackDaysDaily\|scanLookbackDaysWeekly\|scanLookbackDaysHourly" apps/grpc/price_service/scan.go
```

**Expected:** Three matching lines, one per constant, with values `400`, `2000`, `90`.

```bash
# Old single constant is gone
grep -n "^\s*scanLookbackDays\s*=" apps/grpc/price_service/scan.go
```

**Expected:** No matches (the bare `scanLookbackDays =` definition has been removed or renamed).

```bash
# Scheduler weekly lookback matches scanner
grep -n "AddDate(0, 0, -2000)" apps/scheduler/internal/handler/common.go
```

**Expected:** At least one match inside `sinceDateForTimeframe`, under the `"weekly"` case.

```bash
# Partial-week filter is in place
grep -n "WeekEnding.Time.After" apps/grpc/price_service/scan.go
```

**Expected:** One match inside `loadWeeklyOHLCV`.

```bash
# UI label is renamed
grep -n "Lookback bars" apps/ui/web/app/scanner/page.tsx
grep -n "Lookback days" apps/ui/web/app/scanner/page.tsx
```

**Expected:** First command returns one line (the `<Label>` text). Second returns no matches.

```bash
# Stray Go file is gone
test ! -f apps/ui/web/app/scanner/price_updater.go && echo OK
grep -rn "price_updater" apps/ui/web/
```

**Expected:** First prints `OK`. Second returns no matches within `apps/ui/web/`.

## Manual checks

1. **Weekly scanner now evaluates long-period indicators.**
   Open `/scanner`, set Timeframe = Weekly, Lookback bars = 0, Portfolio = All (or a large liquid subset). Add a condition such as `Close[-1] > sma(200)[-1]`. Run the scan. Previously this returned zero matches for any ticker due to NaN from insufficient data. Now, tickers with ≥ 200 weeks of history should produce matches.

2. **Partial current-week bar is excluded.**
   On a Mon/Tue/Wed/Thu, in `/scanner` set Timeframe = Weekly, Lookback bars = 5, add a trivially-true condition like `Close[-1] > 0`. Run the scan. Inspect the `Match Date` column in the results table — every date listed must be a Friday less than or equal to today. No future Friday should appear. Re-running on a Friday after market close should include that Friday as a legitimate match date.

3. **UI label and help text.**
   Visual check of the `ScannerTimeframeAndLookback` component on `/scanner`: the label reads "Lookback bars" and a muted-foreground hint beneath the input explains bar-count semantics.

4. **Saved presets still load.**
   Load any preset saved before this change. The `lookbackDays` numeric value should populate the input correctly (semantics unchanged; only the label changed).

5. **Scheduler weekly alerts benefit from the aligned lookback.**
   On the next Friday weekly run, inspect the scheduler logs for evaluation of an alert using a weekly SMA(200)-based condition. Previously, that alert evaluated against ~52 bars and silently failed. After the fix, the pre-warmed cache should hold ~285 weekly bars and the indicator computes real values.

## Traceability

- Feature spec: `07-spec-scanner-weekly-fixes.md`
- Task breakdown: `07-tasks-scanner-weekly-fixes.md`
- Questions and decisions: `07-questions-1-scanner-weekly-fixes.md`
- Per-task evidence: `07-proofs/07-task-NN-proofs.md`
- Upstream specs: spec 02 (`02-spec-scanner-results-datatable`) — context for the scanner UI this spec touches.
- Parent epic: none.

## Definition of done

- [ ] AC 1: Per-timeframe scan lookback constants exist (daily=400, weekly=2000, hourly=90); each `load*OHLCV` uses the matching constant; `go build` of the gRPC service exits 0.
- [ ] AC 2: `loadWeeklyOHLCV` filters rows with `week_ending > today`; scanner never evaluates `[-1]` against a partial week.
- [ ] AC 3: `sinceDateForTimeframe("weekly")` returns `-2000` days; `go build` of the scheduler exits 0.
- [ ] AC 4: Scanner UI label is "Lookback bars" with help text; atoms and preset schema untouched; `max={250}` unchanged.
- [ ] AC 5: `apps/ui/web/app/scanner/price_updater.go` deleted; no remaining references within `apps/ui/web/`.
- [ ] AC 6: `pnpm typecheck`, `pnpm lint`, and `go build ./...` all exit 0.
- [ ] AC 7: Manual verification confirms weekly scanner produces matches for SMA(200)-based conditions and never returns future-Friday `match_date` values.
- [ ] Proof artifacts contain real command output, not placeholders.

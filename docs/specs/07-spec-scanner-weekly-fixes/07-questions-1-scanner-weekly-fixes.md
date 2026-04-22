---
# Questions: Scanner Weekly Fixes (07) — Round 1
---

## Resolved

| # | Question | Resolution |
|---|----------|------------|
| Q1 | What should the per-timeframe scan lookback window be? | Daily: **400 calendar days** (unchanged — ~280 trading days, supports SMA(200) + buffer). Weekly: **2000 calendar days** (~285 weekly bars, supports SMA(200) weekly + buffer). Hourly: **90 calendar days** (~60 trading days × 6.5h ≈ 390 hourly bars, supports most hourly indicators). These are the target values for the three `load*OHLCV` functions in `apps/grpc/price_service/scan.go`. |
| Q2 | Should `scanMinBars = 50` change? | No. 50 is a sensible floor that applies uniformly across timeframes — any ticker with fewer than 50 bars on the requested timeframe is not scannable regardless. Leaving at 50. |
| Q3 | Should we also fix the scheduler's weekly `PreWarmCache` lookback? | Yes. `sinceDateForTimeframe` in `apps/scheduler/internal/handler/common.go` currently returns `-365` days for weekly, yielding ~52 weekly bars — the same NaN-trigger bug that affects the scanner. Set weekly to **-2000 days** to match Q1. Daily stays at -365 (~252 trading days, still OK for SMA(200)). Hourly stays at -7. |
| Q4 | How should the "Lookback days" input be relabeled? | Rename the label to **"Lookback bars"** and update the helper/hint text to read "Scan the last N bars of the selected timeframe (e.g. 10 weekly bars = 10 weeks)". Semantics are unchanged — the backend already treats this value as a bar count. Do **not** rename the Jotai atoms (`scannerLookbackDaysAtom` / `scannerLookbackInputAtom`) or the proto field `lookback_days`; those are internal wire names and a rename would break saved presets. Only the user-facing label and help text change. |
| Q5 | Should the max value on the input change per timeframe? | Keep a single max of **250** across timeframes (unchanged). 250 daily bars ≈ 1 year, 250 weekly bars ≈ 5 years, 250 hourly bars ≈ 1.5 months — all sensible ceilings. |
| Q6 | How should the partial / in-progress weekly candle be handled in the scanner? | Filter rows where `week_ending > today` out of the OHLCV returned by `loadWeeklyOHLCV`. The weekly updater writes a row with `week_ending` set to the upcoming Friday as soon as any daily bar for that week lands, so on Mon/Tue/Wed/Thu the last row represents an unfinished week. Excluding it means the scanner always evaluates against closed weekly bars. Applied at load time in `scan.go`, not at query time, so the underlying DB query is untouched. |
| Q7 | Does the same partial-week fix need to be applied to the scheduler's alert checker? | No. The scheduler only enqueues the weekly task on Fridays (see `apps/scheduler/internal/schedule/scheduler.go:163` — `if nextDaily.Weekday() == time.Friday`), and the daily task that writes Friday's closing bar runs at market-close + 40 min. The partial-week scenario is scanner-specific (users can run the scanner any weekday). Out of scope for the scheduler. |
| Q8 | What should happen to the stray `apps/ui/web/app/scanner/price_updater.go`? | Delete it. The file is Go code inside a Next.js (TypeScript) app — it is not compiled, referenced, or executed. The canonical price updater lives at `apps/scheduler/internal/price/updater.go` and `apps/grpc/alert_service/price_updater.go`. Leaving the stray file in the scanner directory is confusing and risks someone editing it under the assumption it is live. |
| Q9 | Do we need to add tests for the new lookback constants? | No new unit tests for the constants themselves (they are trivial). The validation plan covers: (a) `go build ./...` exits 0; (b) scanner manual test of SMA(200) weekly producing matches where it previously returned no results; (c) scanner manual test run on a Wednesday confirming `[-1]` bar's `week_ending` is the prior Friday. |
| Q10 | Should any scanner preset schema change? | No. Presets store `lookbackDays: number`. Semantics unchanged; only the UI label changes. Loaded presets continue to work. |

## Open

_None._

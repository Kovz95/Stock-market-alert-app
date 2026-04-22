---
# Tasks: Scanner Weekly Fixes (07)
---

## Task 01 — Per-timeframe scan lookback in scan.go

- In `apps/grpc/price_service/scan.go`, replace the single `scanLookbackDays = 400` line in the `const (...)` block with three named constants:
  ```go
  scanLookbackDaysDaily  = 400  // ~280 trading days; supports SMA(200) + buffer
  scanLookbackDaysWeekly = 2000 // ~285 weekly bars; supports SMA(200) weekly + buffer
  scanLookbackDaysHourly = 90   // ~60 trading days × 6.5h ≈ 390 hourly bars
  ```
- Update `loadDailyOHLCV` to compute `since` with `scanLookbackDaysDaily`.
- Update `loadWeeklyOHLCV` to compute `since` with `scanLookbackDaysWeekly`.
- Update `loadHourlyOHLCV` to compute `since` with `scanLookbackDaysHourly`.
- Add a one-line comment above `scanLookbackDaysWeekly` noting it must stay aligned with `sinceDateForTimeframe("weekly")` in `apps/scheduler/internal/handler/common.go`.
- No other code in `scan.go` changes in this task.

**Proof:** 07-proofs/07-task-01-proofs.md

---

## Task 02 — Filter partial current-week bar in scanner

- In `apps/grpc/price_service/scan.go`, modify `loadWeeklyOHLCV` to drop rows whose `week_ending` is strictly after today's date (UTC) before the slice is handed to `rowsToOHLCVWeekly`.
- Implementation sketch: after `rows, err := q.GetWeeklyPricesBatch(...)`, add:
  ```go
  today := time.Now().UTC().Truncate(24 * time.Hour)
  filtered := rows[:0]
  for _, r := range rows {
      if r.WeekEnding.Valid && r.WeekEnding.Time.After(today) {
          continue
      }
      filtered = append(filtered, r)
  }
  rows = filtered
  ```
- `rowsToOHLCVWeekly` stays as-is.
- Do **not** apply a similar filter to `loadDailyOHLCV` or `loadHourlyOHLCV`.

**Proof:** 07-proofs/07-task-02-proofs.md

---

## Task 03 — Align scheduler weekly PreWarmCache lookback

- In `apps/scheduler/internal/handler/common.go`, update `sinceDateForTimeframe`:
  ```go
  case "weekly":
      return now.AddDate(0, 0, -2000)
  ```
- Add a one-line comment referencing `scanLookbackDaysWeekly` in `apps/grpc/price_service/scan.go` so the two stay aligned.
- Daily and hourly branches are unchanged.

**Proof:** 07-proofs/07-task-03-proofs.md

---

## Task 04 — Rename scanner lookback label and add help text

- In `apps/ui/web/app/scanner/page.tsx`, inside `ScannerTimeframeAndLookback`:
  - Change the `<Label>` text from `Lookback days` to `Lookback bars`.
  - Add a sibling paragraph below the input with `text-xs text-muted-foreground` styling:
    ```
    Number of bars of the selected timeframe (e.g. 10 weekly bars = 10 weeks).
    ```
  - Leave all other behavior of the component untouched: `min={0}`, `max={250}`, the existing `onChange`/`onBlur` logic, and the two atoms (`scannerLookbackDaysAtom`, `scannerLookbackInputAtom`).
- Do not rename the atoms or touch any preset serialization.

**Proof:** 07-proofs/07-task-04-proofs.md

---

## Task 05 — Delete stray Go file from scanner directory

- Delete `apps/ui/web/app/scanner/price_updater.go`.
- Verify nothing else in `apps/ui/web/` references it:
  ```bash
  grep -rn "price_updater" apps/ui/web/
  ```
  should now return zero matches in that tree.
- No replacement — the canonical price updaters live in `apps/scheduler/internal/price/updater.go` and `apps/grpc/alert_service/price_updater.go`.

**Proof:** 07-proofs/07-task-05-proofs.md

---

## Task 06 — Validate and capture proof artifacts

- From repo root:
  ```bash
  go build ./apps/grpc/price_service/...
  go build ./apps/scheduler/...
  go build ./...
  ```
  Confirm all exit 0.
- From `apps/ui/web/`:
  ```bash
  pnpm typecheck
  pnpm lint
  ```
  Confirm both exit 0.
- Start the dev stack (scanner frontend + gRPC backend) and manually verify:
  - The scanner page shows `Lookback bars` (not `Lookback days`) with the new help text.
  - Switching timeframe to Weekly, adding a condition like `sma(200)_weekly[-1] > sma(50)_weekly[-1]` or `Close[-1] > sma(200)[-1]` on a well-seeded ticker produces matches (or at least runs without NaN-all-false). Previously, this would have produced zero matches due to insufficient weekly history.
  - Running on a Mon/Tue/Wed/Thu with `lookbackDays = 0`, the single returned match row's implicit `[-1]` bar is a closed Friday (observable by enabling a small `lookbackDays` and checking that `match_date` values are all Fridays ≤ today, never a future Friday).
- Capture command output in the relevant proof files.

**Proof:** 07-proofs/07-task-06-proofs.md

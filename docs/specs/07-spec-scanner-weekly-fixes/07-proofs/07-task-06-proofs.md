# Proofs: Task 06 — Validate and capture proof artifacts

## Planned evidence

- Captured output from:
  ```bash
  go build ./apps/grpc/price_service/...
  go build ./apps/scheduler/...
  go build ./...
  ```
  all exit 0.
- Captured output from (run in `apps/ui/web/`):
  ```bash
  pnpm typecheck
  pnpm lint
  ```
  both exit 0.
- Manual-test notes:
  - Before/after comparison: scanner with Timeframe=Weekly and condition `Close[-1] > sma(200)[-1]` on a liquid ticker set. Record match count before this spec landed (expected 0) and after (expected > 0 on tickers with enough history).
  - Weekday (non-Friday) scanner run with `lookbackDays > 0` and a trivially-true weekly condition: confirm no row in the results has a future-Friday `match_date`.
  - Screenshot of the scanner page with the new "Lookback bars" label and help text.
- Any anomalies or deviations from the acceptance criteria.

## Completion notes

(Fill in after implementation)

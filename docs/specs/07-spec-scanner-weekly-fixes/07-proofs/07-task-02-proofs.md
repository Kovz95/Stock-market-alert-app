# Proofs: Task 02 — Filter partial current-week bar in scanner

## Planned evidence

- Diff of `apps/grpc/price_service/scan.go` showing the new filter loop in `loadWeeklyOHLCV` that drops rows with `week_ending > today` before constructing the `OHLCV` struct.
- Output of:
  ```bash
  grep -n "WeekEnding.Time.After" apps/grpc/price_service/scan.go
  go build ./apps/grpc/price_service/...
  ```
- Optional: a small integration log / screenshot of a scanner run on a non-Friday showing that the `match_date` values in the returned rows are all Fridays ≤ today.

## Completion notes

(Fill in after implementation)

# Proofs: Task 01 — Per-timeframe scan lookback in scan.go

## Planned evidence

- Diff of `apps/grpc/price_service/scan.go` showing the old single `scanLookbackDays = 400` replaced by three new constants (`scanLookbackDaysDaily`, `scanLookbackDaysWeekly`, `scanLookbackDaysHourly`) and each `load*OHLCV` function referencing its matching constant.
- Output of:
  ```bash
  grep -n "scanLookbackDays" apps/grpc/price_service/scan.go
  go build ./apps/grpc/price_service/...
  ```

## Completion notes

(Fill in after implementation)

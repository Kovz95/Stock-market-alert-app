# Proofs: Task 03 — Align scheduler weekly PreWarmCache lookback

## Planned evidence

- Diff of `apps/scheduler/internal/handler/common.go` showing the `"weekly"` branch of `sinceDateForTimeframe` updated from `-365` to `-2000`, with a cross-reference comment.
- Output of:
  ```bash
  grep -n "AddDate(0, 0, -2000)" apps/scheduler/internal/handler/common.go
  go build ./apps/scheduler/...
  ```

## Completion notes

(Fill in after implementation)

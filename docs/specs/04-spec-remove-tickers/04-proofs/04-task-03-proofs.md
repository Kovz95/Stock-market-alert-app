# Proofs: Task 03 - Implement Go `PreviewDeleteTicker` handler

## Planned evidence

- `git diff apps/grpc/price_service/ticker_deletion.go` (or wherever the handler lives) showing the new method.
- Output of `go build ./apps/grpc/price_service/...` — exit code 0.
- A manual gRPC call (via `grpcurl` or a small Go test harness) showing a valid response for an existing ticker and a response with `exists=false`, all-zero counts for a bogus ticker.
- A manual call with empty `ticker` shows `InvalidArgument`.

## Completion notes

(Fill in after implementation)

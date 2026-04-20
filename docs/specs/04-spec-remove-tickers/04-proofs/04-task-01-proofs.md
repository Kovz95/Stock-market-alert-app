# Proofs: Task 01 - Extend the price proto with ticker-deletion RPCs

## Planned evidence

- `git diff proto/price/v1/price.proto` showing the two new RPCs and `TickerDeletionCounts` message.
- Output of `buf generate` (or project equivalent) with no errors.
- `grep -n "PreviewDeleteTicker\|DeleteTicker\|TickerDeletionCounts" gen/ts/price/v1/price.ts gen/go/price/v1/price.pb.go gen/go/price/v1/price_grpc.pb.go` showing generated symbols.

## Completion notes

(Fill in after implementation)

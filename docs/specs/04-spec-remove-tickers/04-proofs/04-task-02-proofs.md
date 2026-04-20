# Proofs: Task 02 - Add sqlc queries for counts and cascade deletes

## Planned evidence

- Listing of new/modified file(s) under `database/sql/queries/`.
- Output of `sqlc generate` with no errors.
- `grep -n "CountTickerReferences\|TickerExists\|DeleteTickerFrom" database/sql/queries/*.sql` showing all 14 new queries.
- `grep -n "DeleteTickerFromStockMetadata\|DeleteTickerFromAlertsRatio\|CountTickerReferences" gen/go/db/*.go` confirming generated methods.

## Completion notes

(Fill in after implementation)

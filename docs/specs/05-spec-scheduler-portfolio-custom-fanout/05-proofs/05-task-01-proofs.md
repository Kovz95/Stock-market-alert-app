# Proofs: Task 01 - Add sqlc queries for portfolio fan-out

## Planned evidence

- `git diff database/sql/queries/` showing the two new queries (`ListPortfoliosForFanout`, `ListPortfolioStocks`).
- Output of `sqlc generate` with no errors.
- `grep -n "ListPortfoliosForFanout\|ListPortfolioStocks" database/db/*.go` (or whatever path holds generated code) showing generated method signatures.

## Completion notes

(Fill in after implementation)

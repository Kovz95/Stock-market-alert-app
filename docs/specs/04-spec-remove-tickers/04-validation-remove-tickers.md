# Validation: Remove Tickers (04)

## Automated verification

From repository root:

```bash
# Regenerate protos (must succeed after the new RPCs are added)
buf generate

# Regenerate sqlc queries (must succeed after the new .sql file is added)
sqlc generate

# Go build (verifies the new handlers compile and all call sites are wired)
go build ./...

# Go unit + integration tests for the new handler
go test ./apps/grpc/price_service/... -run "TestDeleteTicker|TestPreviewDeleteTicker" -v

# Full Go test suite (verifies no regression in alert_service, scheduler, etc.)
go test ./...

# Next.js type checking (verifies server actions and hooks compile against new proto types)
pnpm --filter web typecheck

# Next.js lint
pnpm --filter web lint
```

**Expected:**

- `buf generate` — no errors, `gen/ts/price/v1/price.ts` and `gen/go/price/v1/price*.pb.go` now contain `PreviewDeleteTicker`, `DeleteTicker`, `TickerDeletionCounts`, and their request/response messages.
- `sqlc generate` — no errors, generated Go code includes `CountTickerReferences`, `TickerExists`, and all twelve `DeleteTickerFrom*` methods.
- `go build ./...` — exit code 0.
- `go test ./apps/grpc/price_service/... -run "TestDeleteTicker|TestPreviewDeleteTicker"` — all five new tests pass: `TestDeleteTicker_FullCascade`, `TestDeleteTicker_NonExistent`, `TestDeleteTicker_InvalidArgument`, `TestDeleteTicker_RollbackOnError`, `TestPreviewDeleteTicker_Counts`.
- `go test ./...` — exit code 0 (no regressions).
- `pnpm --filter web typecheck` — exit code 0.
- `pnpm --filter web lint` — exit code 0 or unchanged baseline.

## Structural checks

Run from repo root to prove all of the spec's required files exist and reference the right symbols:

```bash
# Proto has both new RPCs
grep -n "rpc PreviewDeleteTicker" proto/price/v1/price.proto
grep -n "rpc DeleteTicker"        proto/price/v1/price.proto
grep -n "TickerDeletionCounts"    proto/price/v1/price.proto

# sqlc query names present
grep -n "CountTickerReferences"             database/sql/queries/*.sql
grep -n "DeleteTickerFromStockMetadata"     database/sql/queries/*.sql
grep -n "DeleteTickerFromAlertsDirect"      database/sql/queries/*.sql
grep -n "DeleteTickerFromAlertsRatio"       database/sql/queries/*.sql

# Go handler methods
grep -n "func (s \*Server) PreviewDeleteTicker" apps/grpc/price_service/
grep -n "func (s \*Server) DeleteTicker"        apps/grpc/price_service/

# Server actions exported
grep -n "export async function previewDeleteTicker" apps/ui/web/actions/stock-database-actions.ts
grep -n "export async function deleteTicker"        apps/ui/web/actions/stock-database-actions.ts

# Hooks exported
grep -n "usePreviewDeleteTicker" apps/ui/web/lib/hooks/useStockDatabase.ts
grep -n "useDeleteTicker"        apps/ui/web/lib/hooks/useStockDatabase.ts

# Dialog component exists
ls apps/ui/web/app/database/stock/_components/DeleteTickerDialog.tsx
```

**Expected:** every grep returns at least one match; `ls` succeeds.

## Manual checks

End-to-end smoke test (cannot be automated — requires dev server + real DB):

1. Start backend services (`price_service`) and `pnpm --filter web dev`.
2. Navigate to `http://localhost:3000/database/stock`.
3. Pick a real ticker (say `AAPL` in a staging DB — **do not test on production data**). Before clicking Delete, record via psql:
   ```sql
   SELECT
     (SELECT COUNT(*) FROM stock_metadata     WHERE symbol = 'AAPL') AS stock_metadata,
     (SELECT COUNT(*) FROM ticker_metadata    WHERE ticker = 'AAPL') AS ticker_metadata,
     (SELECT COUNT(*) FROM daily_prices       WHERE ticker = 'AAPL') AS daily_prices,
     (SELECT COUNT(*) FROM hourly_prices      WHERE ticker = 'AAPL') AS hourly_prices,
     (SELECT COUNT(*) FROM weekly_prices      WHERE ticker = 'AAPL') AS weekly_prices,
     (SELECT COUNT(*) FROM daily_move_stats   WHERE ticker = 'AAPL') AS daily_move_stats,
     (SELECT COUNT(*) FROM alerts             WHERE ticker = 'AAPL') AS alerts_direct,
     (SELECT COUNT(*) FROM alerts             WHERE ticker1 = 'AAPL' OR ticker2 = 'AAPL') AS alerts_ratio,
     (SELECT COUNT(*) FROM alert_audits       WHERE ticker = 'AAPL') AS alert_audits,
     (SELECT COUNT(*) FROM portfolio_stocks   WHERE ticker = 'AAPL') AS portfolio_stocks;
   ```
4. Click the Delete icon on the AAPL row.
5. **Confirm** the dialog counts match the psql output exactly. Take a screenshot.
6. Click **Cancel**. Re-run the psql query — numbers unchanged.
7. Click the Delete icon again, then **Delete permanently**.
8. Confirm:
   - Sonner success toast appears with a counts summary.
   - The AAPL row disappears from the Stock Database table.
   - Running the psql query again returns all zeroes.
9. Navigate to `/alerts` — any alert that referenced AAPL is gone.
10. Navigate to `/portfolios` — any portfolio that had AAPL as a holding no longer shows it.
11. Trigger a scheduler run (`/scheduler` → Run Exchange Job) for an exchange that AAPL belonged to. Inspect the `price_service` logs — no reference to AAPL in the evaluation loop.

## Traceability

- Feature spec: `04-spec-remove-tickers.md`
- Task breakdown: `04-tasks-remove-tickers.md`
- Questions and decisions: `04-questions-1-remove-tickers.md`
- Per-task evidence: `04-proofs/04-task-NN-proofs.md` (NN = 01..10)
- Upstream specs: none
- Parent epic: none (this is a single-spec feature)

## Definition of done

- [ ] Proto contract: `TickerDeletionCounts`, two request/response pairs, two RPCs on `PriceService` — regenerated code compiles.
- [ ] sqlc queries: one count query, one exists query, twelve delete queries — generated Go code compiles.
- [ ] Go `PreviewDeleteTicker` handler implemented and returns correct counts.
- [ ] Go `DeleteTicker` handler runs all eleven deletes in one transaction with rollback on error.
- [ ] Five Go tests pass: full cascade, non-existent, invalid argument, rollback on error, preview counts.
- [ ] Next.js server actions `previewDeleteTicker` and `deleteTicker` exported with correct types.
- [ ] Hooks `usePreviewDeleteTicker` and `useDeleteTicker` exported; deletion invalidates stock-database, alerts, portfolios query keys.
- [ ] `DeleteTickerDialog` component renders loading / counts / empty / error states and handles mutation success/failure per spec.
- [ ] Stock Database table shows Delete icon in a new Actions column; clicking opens the dialog.
- [ ] `pnpm --filter web typecheck` exits 0.
- [ ] `pnpm --filter web lint` exits 0 or unchanged baseline.
- [ ] `go test ./...` exits 0.
- [ ] Manual end-to-end smoke passes on a staging DB.
- [ ] All proof files contain real command output, not placeholders.

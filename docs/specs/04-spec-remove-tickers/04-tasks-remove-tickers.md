# Tasks: Remove Tickers (04)

## Task 01 - Extend the price proto with ticker-deletion RPCs

- Edit `proto/price/v1/price.proto`:
  - Add `TickerDeletionCounts` message with every field listed in the spec Contract section.
  - Add `PreviewDeleteTickerRequest`, `PreviewDeleteTickerResponse`, `DeleteTickerRequest`, `DeleteTickerResponse` exactly as specified.
  - Register `PreviewDeleteTicker` and `DeleteTicker` on the existing `PriceService`.
- Regenerate protobuf code (`buf generate` or the project's codegen script referenced in `buf.gen.yaml`).
- Confirm generated files contain the new symbols:
  - `gen/ts/price/v1/price.ts` — new message types and `PreviewDeleteTicker` / `DeleteTicker` on the service definition.
  - `gen/go/price/v1/price.pb.go` and `price_grpc.pb.go` — new server interface stubs.

**Proof:** 04-proofs/04-task-01-proofs.md

## Task 02 - Add sqlc queries for counts and cascade deletes

- Add a new file `database/sql/queries/ticker_deletion_queries.sql` (or append to `stock_metadata_queries.sql`) with the following sqlc-named queries:
  - `CountTickerReferences :one` — one SELECT that returns all twelve counts as named columns (use scalar subqueries). This must cover: `stock_metadata`, `ticker_metadata`, `daily_prices`, `hourly_prices`, `weekly_prices`, `continuous_prices` (by `symbol`), `daily_move_stats`, `futures_metadata` (by `symbol`), `alerts` direct (`ticker = $1`), `alerts` ratio (`ticker1 = $1 OR ticker2 = $1`), `alert_audits`, `portfolio_stocks`.
  - `TickerExists :one` — returns boolean `EXISTS(SELECT 1 FROM stock_metadata WHERE symbol = $1 UNION ALL SELECT 1 FROM ticker_metadata WHERE ticker = $1)`.
  - One `:execrows` DELETE per table, in this exact order:
    1. `DeleteTickerFromPortfolioStocks`
    2. `DeleteTickerFromAlertAudits`
    3. `DeleteTickerFromAlertsDirect` — `DELETE FROM alerts WHERE ticker = $1`
    4. `DeleteTickerFromAlertsRatio` — `DELETE FROM alerts WHERE ticker1 = $1 OR ticker2 = $1`
    5. `DeleteTickerFromDailyMoveStats`
    6. `DeleteTickerFromDailyPrices`
    7. `DeleteTickerFromHourlyPrices`
    8. `DeleteTickerFromWeeklyPrices`
    9. `DeleteTickerFromContinuousPrices` — WHERE `symbol = $1`
    10. `DeleteTickerFromFuturesMetadata` — WHERE `symbol = $1`
    11. `DeleteTickerFromTickerMetadata`
    12. `DeleteTickerFromStockMetadata` — WHERE `symbol = $1`
- Run `sqlc generate` (invoke via the existing project script — check `go.work` or `Makefile`/package.json scripts).
- Confirm generated Go code in `gen/go/db/` (or the project's sqlc output path) exposes typed methods for each new query.

**Proof:** 04-proofs/04-task-02-proofs.md

## Task 03 - Implement Go `PreviewDeleteTicker` handler

- Create `apps/grpc/price_service/ticker_deletion.go` (or add to an existing handlers file).
- Implement:
  ```go
  func (s *Server) PreviewDeleteTicker(ctx context.Context, req *pricev1.PreviewDeleteTickerRequest) (*pricev1.PreviewDeleteTickerResponse, error)
  ```
- Validate: empty `ticker` → `codes.InvalidArgument` with message `"ticker is required"`.
- Acquire a connection from `s.pool`, call the generated `CountTickerReferences` and `TickerExists` queries.
- Map the row to `TickerDeletionCounts`, populate `exists`, return.
- All errors → `codes.Internal` with the pg error formatted into the status message.

**Proof:** 04-proofs/04-task-03-proofs.md

## Task 04 - Implement Go `DeleteTicker` handler with transaction

- In the same file, implement:
  ```go
  func (s *Server) DeleteTicker(ctx context.Context, req *pricev1.DeleteTickerRequest) (*pricev1.DeleteTickerResponse, error)
  ```
- Validate empty ticker (same as Task 03).
- Begin a transaction: `tx, err := s.pool.BeginTx(ctx, pgx.TxOptions{})`. `defer tx.Rollback(ctx)`.
- Call each `DeleteTickerFrom*` generated method in the exact order listed in Task 02, passing `tx` as the DBTX. Capture each `int64` result (sqlc returns `(int64, error)` for `:execrows`). Accumulate into a `TickerDeletionCounts` struct.
- On any per-step error: return a response with `success=false, error_message="<step>: <err>"`. The deferred `tx.Rollback` handles rollback.
- On success: `tx.Commit(ctx)`. Return `success=true`, populated counts, and the original ticker string.
- Edge case: if the ticker has zero rows everywhere, the transaction still runs (all DELETEs are no-ops) and commits cleanly with all-zero counts and `success=true`.

**Proof:** 04-proofs/04-task-04-proofs.md

## Task 05 - Write Go tests for the deletion handlers

- Create `apps/grpc/price_service/ticker_deletion_test.go` (follow the existing test-fixture pattern in `apps/grpc/alert_service/*_test.go` or similar). Use a transactional test DB (spin up a `pgxpool` against a test database, wrap each test in a sub-transaction, or use the project's existing integration-test helper — whichever the existing tests use).
- Tests:
  1. `TestDeleteTicker_FullCascade` — seed a ticker `ZZDEAD` into **every** reference table (at minimum one row in each). Call `DeleteTicker("ZZDEAD")`. Assert:
     - `success == true`
     - Each count field matches what was seeded
     - Post-call `COUNT(*)` for `ZZDEAD` in every table returns 0.
  2. `TestDeleteTicker_NonExistent` — call with a ticker that has no rows anywhere. Assert `success == true`, all counts zero, no error.
  3. `TestDeleteTicker_InvalidArgument` — empty string → gRPC `codes.InvalidArgument`.
  4. `TestDeleteTicker_RollbackOnError` — induce an error on the 5th DELETE step (e.g. temporarily break the connection, or use a test hook to inject an error from one of the generated query methods). Assert:
     - `success == false`, `error_message` contains the failing step name
     - Every seeded row is still present (rollback worked).
  5. `TestPreviewDeleteTicker_Counts` — seed known row counts across tables, call preview, assert counts match. Verify `exists=true` when metadata is present and `exists=false` when only non-metadata tables have rows.
- Run `go test ./apps/grpc/price_service/...` — all pass.

**Proof:** 04-proofs/04-task-05-proofs.md

## Task 06 - Add Next.js server actions

- Edit `apps/ui/web/actions/stock-database-actions.ts`:
  - Ensure `"use server"` is still present at the top of the file.
  - Import the new request/response types from `../../../../gen/ts/price/v1/price`.
  - Export `TickerDeletionCounts`, `PreviewDeleteTickerResult`, `DeleteTickerResult` types exactly as listed in spec acceptance criterion 5.
  - Implement a `toDeletionCounts(proto)` mapper that converts each `int64` field via `Number(val)`.
  - Implement:
    ```typescript
    export async function previewDeleteTicker(ticker: string): Promise<PreviewDeleteTickerResult>;
    export async function deleteTicker(ticker: string): Promise<DeleteTickerResult>;
    ```
  - Both call `priceClient.<rpc>(...)` and map the response.
- Run `pnpm --filter web typecheck` → exits 0.

**Proof:** 04-proofs/04-task-06-proofs.md

## Task 07 - Add TanStack hooks for preview and delete

- Edit `apps/ui/web/lib/hooks/useStockDatabase.ts`:
  - Add `usePreviewDeleteTicker(ticker: string | null)` — `useQuery` with queryKey `["stock-database", "preview-delete", ticker]`, `enabled: ticker !== null`, `queryFn: () => previewDeleteTicker(ticker!)`. Default `staleTime: 0` so the preview is fresh every time a new dialog opens.
  - Add `useDeleteTicker()` — `useMutation` calling `deleteTicker`. `onSuccess` must call `queryClient.invalidateQueries` for **all three** of: the stock-database list key (`STOCK_DATABASE_KEY`), the alerts list key (`ALERTS_KEY` from `lib/store/alerts.ts`), and the portfolios key if one exists (`PORTFOLIOS_KEY`; if the portfolios hook uses a different key, import it). This ensures every page that lists data keyed on the deleted ticker refreshes.
- Do **not** alter existing hooks' behavior.

**Proof:** 04-proofs/04-task-07-proofs.md

## Task 08 - Build the `DeleteTickerDialog` component

- Create `apps/ui/web/app/database/stock/_components/DeleteTickerDialog.tsx`:
  - `"use client"` header.
  - Props: `{ ticker: string | null; onClose: () => void; }`. Open when `ticker !== null`.
  - Uses shadcn `AlertDialog`, `AlertDialogContent`, `AlertDialogHeader`, `AlertDialogTitle`, `AlertDialogDescription`, `AlertDialogFooter`, `AlertDialogCancel`, `AlertDialogAction`.
  - Calls `usePreviewDeleteTicker(ticker)` on open and `useDeleteTicker()` for the mutation.
  - Render states:
    - **Loading** (preview pending): `Skeleton` rows where the count list will go; Cancel button enabled, Delete button disabled.
    - **Loaded with counts**: bullet list of non-zero counts, formatted with `Intl.NumberFormat("en-US").format(n)`. Include human labels per spec acceptance criterion 7 (e.g. "1,247 daily price bars", "N direct alerts", "N ratio alerts (this ticker is ticker1 or ticker2)"). Omit rows where count === 0.
    - **Empty state** (`exists === false` and all counts zero): "Nothing to delete — this ticker isn't in the database." Delete button disabled.
    - **Error during preview**: small destructive-styled message and a disabled Delete button.
  - Destructive button text: `Delete permanently`, `variant="destructive"`. While the mutation is pending: show `Loader2Icon` with `animate-spin`, disable both footer buttons.
  - On mutation success: `toast.success(\`Removed ${ticker}. Deleted ${totalAlerts} alerts, ${totalBars} price bars, ${counts.alertAudits} audit rows.\`)` where `totalAlerts = alertsDirect + alertsRatio` and `totalBars = dailyPrices + hourlyPrices + weeklyPrices + continuousPrices`. Then call `onClose()`.
  - On mutation failure: `toast.error(result.errorMessage || "Failed to delete ticker")`. Keep dialog open.
- Use only semantic tokens and shadcn primitives; no hardcoded colors.

**Proof:** 04-proofs/04-task-08-proofs.md

## Task 09 - Wire the delete button into the Stock Database table

- Locate the table component that renders rows for `/database/stock` (inside `apps/ui/web/app/database/stock/_components/`).
- Add an **Actions** column at the trailing edge of the table.
- Each row renders a `DeleteTickerButton` (either inline or imported) — a `Button` with `variant="ghost"`, `size="icon"`, `className="text-muted-foreground hover:text-destructive"`, and a `<Trash2Icon className="size-4" />` child. The button's `aria-label` is `` `Delete ${ticker}` ``.
- Button `onClick` sets a local `selectedTickerToDelete` state (or a Jotai atom in `lib/store/stock-database.ts`) to the row's ticker; this opens `DeleteTickerDialog`.
- On dialog close, clear `selectedTickerToDelete` back to `null`.
- Table header cell reads `Actions` (or similar short label). The column does not sort.

**Proof:** 04-proofs/04-task-09-proofs.md

## Task 10 - Validate and capture proof artifacts

- Run `pnpm --filter web typecheck`; capture output — must exit 0.
- Run `pnpm --filter web lint`; capture output.
- Run `go test ./apps/grpc/price_service/...`; capture output — all new tests pass.
- Run `go build ./...` from repo root; must exit 0.
- Start the Go services (price_service) and the Next.js dev server. Manually:
  - Navigate to `/database/stock`. Pick a real ticker known to be present. Click its Delete icon.
  - Confirm the preview dialog shows plausible non-zero counts matching `psql` output for that ticker across all twelve tables.
  - Click **Cancel** — confirm nothing was deleted (`psql` counts unchanged) and the dialog closes.
  - Click Delete icon again, then **Delete permanently** — confirm:
    - Success toast appears with counts.
    - The row disappears from the table.
    - `psql` reports zero rows for that ticker across every referenced table.
  - Navigate to `/alerts` and `/portfolios` — confirm alerts referencing the deleted ticker and portfolio memberships are gone.
- Pick a ticker that doesn't exist (type into a test form or hit the server action directly via a dev script). Confirm `success=true`, all-zero counts, no error.
- Induce a rollback scenario (optional manual check — covered automatically by Task 05 test).
- Fill in every proof file with real command output, psql snapshots before/after, and screenshots of the dialog + success toast.
- Confirm every acceptance criterion in `04-spec-remove-tickers.md` is met; tick the Definition-of-done checklist in the validation file.

**Proof:** 04-proofs/04-task-10-proofs.md

# Spec: Remove Tickers (04)

## Goal

Allow a user to permanently remove a ticker from the system in a single confirmed action. Exchange-delisted or otherwise abandoned tickers still have rows in `stock_metadata`, `ticker_metadata`, `daily_prices`, `hourly_prices`, `weekly_prices`, `daily_move_stats`, and sometimes `continuous_prices`/`futures_metadata`. They also still appear in `alerts` (directly or as `ticker1`/`ticker2` of a ratio), `alert_audits`, and `portfolio_stocks`. Since these tickers never receive new data, they pollute the Stock Database table, scanner filters, portfolio holdings, and stale-data warnings in the UI, and evaluations against them either fail silently or waste cycles. This spec adds a `DeleteTicker` flow — a preview RPC, a cascading-delete RPC, and a Next.js confirmation dialog wired into the existing Stock Database page — so the user can purge a dead ticker from every reference table in one atomic transaction.

## Scope

### In scope

- **Proto** (`proto/price/v1/price.proto`): add two RPCs to `PriceService`:
  - `PreviewDeleteTicker` → returns per-table row counts for a ticker without mutating anything.
  - `DeleteTicker` → deletes the ticker from every reference table, then from `ticker_metadata` and `stock_metadata`, in a single Postgres transaction. Returns the actual deleted counts per table.
- **SQL queries** (`database/sql/queries/stock_metadata_queries.sql`, new named queries; or a new `ticker_deletion_queries.sql`): sqlc-generated `CountTickerReferences*` and `DeleteTicker*` queries covering all tables in the cascade list below.
- **Go service** (`apps/grpc/price_service/`): implement both RPCs. The cascade runs inside a single `pgx.Tx` (BEGIN … COMMIT). Any error rolls back the entire cascade.
- **Next.js server actions** (`apps/ui/web/actions/stock-database-actions.ts`): add `previewDeleteTicker(ticker)` and `deleteTicker(ticker)` server actions. Return plain serializable objects.
- **Jotai store + TanStack hooks**: extend `apps/ui/web/lib/store/stock-database.ts` and `apps/ui/web/lib/hooks/useStockDatabase.ts` with `usePreviewDeleteTicker` (query) and `useDeleteTicker` (mutation that invalidates the stock-database query key).
- **UI** (`apps/ui/web/app/database/stock/_components/`): add a row-level **Delete** icon button to the existing Stock Database table. Clicking it opens a shadcn `AlertDialog` that:
  - Shows a brief loading state, then renders the preview counts ("1 stock_metadata, 3 alerts (2 direct, 1 ratio), 1,247 daily bars, …").
  - Has a destructive **Delete permanently** button that runs the mutation, closes the dialog on success, surfaces a Sonner success toast with the deletion summary, and invalidates the stock-database list so the row disappears.
  - Has a **Cancel** button that dismisses the dialog with no side effects.
- **No change** to: Python Streamlit pages, alert evaluation logic, scheduler code, scanner logic, Discord routing. These read from the same tables and will naturally stop seeing the deleted ticker on the next iteration.

### Out of scope

- **Bulk delete** (selecting N tickers and purging all at once). Start with single-ticker delete; revisit if the single-ticker flow proves insufficient.
- **Soft delete** (moving rows to an archive table, tombstoning `stock_metadata`, or adding a `deleted_at` column). Deletion is hard.
- **Undo / restore.** There is no undo. Users must confirm.
- **Cleanup of `app_documents`** (legacy JSON snapshots of `alerts.json`, `portfolios.json`, etc.). Those are stale regardless and not authoritative.
- **Flagging or surfacing stale tickers automatically** (highlighting rows with old `ticker_metadata.last_update`, adding a "Stale" tab). This spec only adds the action; discovery is left to the user.
- **Python Streamlit parity.** No Streamlit "remove ticker" page is added.
- **Changes to `DeleteAlert` / `BulkDeleteAlerts` behavior** around dangling audit rows. If an alert is deleted directly (not via ticker removal), its audit rows still stay orphaned (current behavior). That is a separate concern.
- **Permission / auth gating.** The project has no auth layer today; this spec doesn't introduce one.
- **Re-typing the ticker as a confirmation challenge.** A single click on the destructive button in the preview dialog is sufficient — same bar as the existing Delete Alerts flow.

## Source excerpts

- `database/sql/schema.sql:8–165` — schema for every cascade target: `daily_prices`, `hourly_prices`, `weekly_prices`, `ticker_metadata`, `daily_move_stats`, `alert_audits`, `continuous_prices`, `futures_metadata`, `stock_metadata`.
- `database/sql/schema.sql:177–210` — `alerts` schema showing `ticker`, `ticker1`, `ticker2` columns and their indexes. Ratio alerts are the ones with non-null `ticker1`/`ticker2`.
- `database/sql/schema.sql:225–232` — `portfolio_stocks` schema (no FK to a ticker table; join is by the `ticker` string).
- `database/sql/queries/alert_queries.sql` — existing sqlc patterns used by the `alert_service` (reference for query naming and parameter style).
- `proto/price/v1/price.proto:250–262` — current `PriceService` definition. The two new RPCs append here.
- `apps/grpc/price_service/handlers.go` — reference for existing handler layout and `pgxpool` usage; the new handlers sit alongside existing ones.
- `apps/ui/web/actions/stock-database-actions.ts` — existing server actions for the Stock Database page; the two new actions go here.
- `apps/ui/web/app/database/stock/_components/` — existing table/filter components; the delete button attaches to the existing row renderer.
- `apps/ui/web/CLAUDE.md §11, §12, §13, §16` — How to Add a New Page / RPC / Component, plus hard rules for Timestamps, "use server"/"use client" discipline, and semantic tokens.
- `apps/ui/web/app/alerts/delete/_components/` — existing destructive-action UI pattern for selecting and deleting alerts; reuse the same confirmation/toast idiom.

## Contract

Append to `proto/price/v1/price.proto`, under `PriceService`. Package stays `stockalert.price.v1`.

```proto
message TickerDeletionCounts {
  // Per-table row counts. Each field is >= 0.
  int64 stock_metadata      = 1;  // 0 or 1
  int64 ticker_metadata     = 2;  // 0 or 1
  int64 daily_prices        = 3;
  int64 hourly_prices       = 4;
  int64 weekly_prices       = 5;
  int64 continuous_prices   = 6;
  int64 daily_move_stats    = 7;
  int64 futures_metadata    = 8;  // 0 or 1
  int64 alerts_direct       = 9;  // alerts where ticker = $1
  int64 alerts_ratio        = 10; // alerts where ticker1 = $1 OR ticker2 = $1
  int64 alert_audits        = 11;
  int64 portfolio_stocks    = 12;
}

message PreviewDeleteTickerRequest {
  string ticker = 1;
}
message PreviewDeleteTickerResponse {
  string ticker                    = 1;
  bool   exists                    = 2;   // true iff stock_metadata OR ticker_metadata has a row
  TickerDeletionCounts counts      = 3;   // what *would* be deleted
}

message DeleteTickerRequest {
  string ticker = 1;
}
message DeleteTickerResponse {
  bool   success                   = 1;
  string error_message             = 2;
  string ticker                    = 3;
  TickerDeletionCounts counts      = 4;   // what *was* deleted (all zeroes on failure)
}

service PriceService {
  // ... existing RPCs unchanged ...
  rpc PreviewDeleteTicker(PreviewDeleteTickerRequest) returns (PreviewDeleteTickerResponse);
  rpc DeleteTicker(DeleteTickerRequest) returns (DeleteTickerResponse);
}
```

**Transaction contract (must be preserved):** `DeleteTicker` runs all eleven `DELETE` statements inside one `BEGIN … COMMIT` on a single connection. If any step errors, the transaction rolls back and `success=false`. On a successful commit, the RPC returns the counts observed from each `DELETE … ; GET DIAGNOSTICS` (or sqlc `:execrows`).

**Preview consistency:** `PreviewDeleteTicker` runs the eleven `COUNT(*)`/`SELECT COUNT` queries on the same pool connection in a read-only snapshot. The counts reflect the DB state at the time of the call; they are not re-validated inside the subsequent `DeleteTicker` call, so the counts shown in the dialog may drift if another process writes in the interim. This is acceptable — the final response reports actual deleted counts.

## Acceptance criteria

1. **Proto contract**
   - `proto/price/v1/price.proto` contains `TickerDeletionCounts`, the two request/response message pairs, and both RPCs registered on `PriceService`.
   - Protobuf code regeneration produces `gen/ts/price/v1/price.ts` and `gen/go/price/v1/price*.pb.go` with the new symbols.
   - `pnpm --filter web typecheck` exits 0 after regen.
   - `go build ./...` exits 0 from the repo root.

2. **SQL queries**
   - `database/sql/queries/` contains sqlc-named queries for each cascade step:
     - `CountTickerReferences` (single query returning all twelve counts in one row — `COUNT(*)` per table joined via `UNION ALL` or a single `SELECT` with subselects), OR twelve individual `:one` queries. Either is acceptable.
     - `DeleteTickerFromPortfolioStocks`, `DeleteTickerFromAlertAudits`, `DeleteTickerFromAlerts` (covers `ticker`, `ticker1`, `ticker2` in one statement), `DeleteTickerFromDailyMoveStats`, `DeleteTickerFromDailyPrices`, `DeleteTickerFromHourlyPrices`, `DeleteTickerFromWeeklyPrices`, `DeleteTickerFromContinuousPrices`, `DeleteTickerFromFuturesMetadata`, `DeleteTickerFromTickerMetadata`, `DeleteTickerFromStockMetadata` — each `:execrows` so the Go handler can read the affected row count.
   - `sqlc generate` exits 0 and updates `gen/go/db/` (or equivalent, per existing layout).

3. **Go `PreviewDeleteTicker`**
   - Implemented in `apps/grpc/price_service/` (new file `ticker_deletion.go` or added to `handlers.go`).
   - Returns `exists=true` iff `stock_metadata` OR `ticker_metadata` has a row for the ticker.
   - Fills every field of `TickerDeletionCounts` from its corresponding `COUNT(*)` result.
   - Reads are on a single acquired pool connection; no transaction is required (read-only).
   - On DB error: returns gRPC `codes.Internal` with the pg error in the status message (follow existing handler error patterns).

4. **Go `DeleteTicker`**
   - Implemented alongside `PreviewDeleteTicker`.
   - Begins a transaction (`pool.BeginTx`), runs the eleven `DELETE`s in the order listed in R04 of the questions file, captures each `CommandTag.RowsAffected()`, then commits.
   - On any error mid-cascade: rolls back and returns `success=false, error_message="<step_name>: <pg error>"`.
   - On success: populates `TickerDeletionCounts` with the actual deleted row counts per step. `alerts_direct` and `alerts_ratio` come from two separate `DELETE`s or one combined `DELETE … RETURNING` — implementer's choice, but both fields must be populated accurately.
   - Empty ticker string → gRPC `codes.InvalidArgument` with message `"ticker is required"`.
   - Non-existent ticker (no rows in any table) → `success=true`, all counts zero. No error. The UI can decide whether to show a "nothing to delete" message.

5. **Next.js server actions**
   - `apps/ui/web/actions/stock-database-actions.ts` exports:
     ```typescript
     export type TickerDeletionCounts = {
       stockMetadata: number;
       tickerMetadata: number;
       dailyPrices: number;
       hourlyPrices: number;
       weeklyPrices: number;
       continuousPrices: number;
       dailyMoveStats: number;
       futuresMetadata: number;
       alertsDirect: number;
       alertsRatio: number;
       alertAudits: number;
       portfolioStocks: number;
     };
     export type PreviewDeleteTickerResult = {
       ticker: string;
       exists: boolean;
       counts: TickerDeletionCounts;
     };
     export type DeleteTickerResult = {
       success: boolean;
       errorMessage?: string;
       ticker: string;
       counts: TickerDeletionCounts;
     };
     export async function previewDeleteTicker(ticker: string): Promise<PreviewDeleteTickerResult>;
     export async function deleteTicker(ticker: string): Promise<DeleteTickerResult>;
     ```
   - Both actions begin with `"use server"` (file-level directive already present).
   - int64 fields map to `number` via `Number(val)` per `apps/ui/web/CLAUDE.md §16.6`.
   - `pnpm --filter web typecheck` exits 0.

6. **Store + hooks**
   - `apps/ui/web/lib/hooks/useStockDatabase.ts` exports `usePreviewDeleteTicker(ticker: string | null)` — a TanStack `useQuery` keyed on `["stock-database", "preview-delete", ticker]` that is disabled when `ticker` is null and runs when a ticker is set (i.e., when the dialog opens).
   - Exports `useDeleteTicker()` — a `useMutation` that calls `deleteTicker(ticker)` and on success calls `queryClient.invalidateQueries({ queryKey: STOCK_DATABASE_KEY })` (the existing key for the full metadata list) so the deleted row disappears from the table.
   - Existing hooks and atoms remain backward-compatible.

7. **UI — delete button & confirmation dialog**
   - `apps/ui/web/app/database/stock/_components/` gains:
     - A `DeleteTickerButton.tsx` client component rendered as an icon button (lucide `Trash2Icon`, `size-4`, variant `ghost`, `text-muted-foreground hover:text-destructive`) in a new trailing "Actions" column of the Stock Database table.
     - A `DeleteTickerDialog.tsx` client component using shadcn `AlertDialog`. On open it calls `usePreviewDeleteTicker(ticker)`. It renders:
       - Title: `Remove {ticker}?`
       - Description: one paragraph explaining this is permanent and lists what will be removed.
       - A `Skeleton` during load.
       - A bulleted list of non-zero preview counts (hide rows that are zero) formatted with `Intl.NumberFormat`. The ratio-alert line reads `"N ratio alerts (this ticker appears as ticker1 or ticker2)"` whenever `alertsRatio > 0`.
       - If `exists === false` AND every count is zero: the dialog shows "Nothing to delete — this ticker isn't in the database." and the destructive button is disabled.
       - A Cancel button and a destructive **Delete permanently** button (variant `destructive`).
   - On destructive click: run `useDeleteTicker().mutate(ticker)`. While pending, disable both buttons and show `Loader2Icon` spinner on the destructive button. On success: `toast.success` with a summary (`Removed {ticker}. Deleted {alerts} alerts, {bars} price bars, {audits} audit rows.`), close dialog, list refreshes. On failure: `toast.error(errorMessage)`, keep the dialog open so the user sees the error.
   - All UI uses semantic Tailwind tokens only (no hardcoded colors); follows `apps/ui/web/CLAUDE.md §13` and §16.10.

8. **Transactional correctness**
   - Integration test seeds a fake ticker `ZZDEAD` into every reference table, calls `DeleteTicker("ZZDEAD")`, and asserts every target table has zero rows for `ZZDEAD` after the call. The returned counts match the seeded counts exactly.
   - A rollback test induces a SQL error partway through the cascade (e.g. by mocking one query to return an error) and asserts that **no** rows were deleted — all seeded rows still exist.
   - These tests live in `apps/grpc/price_service/ticker_deletion_test.go` and run via `go test ./apps/grpc/price_service/...`.

9. **End-to-end smoke**
   - Start the Go `price_service` and Next.js dev server. Load `/database/stock`. Click the Delete icon on a real ticker already present in the DB but known-dead. Confirm:
     - The preview counts match what `psql` reports for that ticker across all tables.
     - Clicking **Delete permanently** closes the dialog, shows the success toast, the row disappears from the table, and a subsequent `psql` query returns zero rows across every referenced table.
     - Clicking **Cancel** on a fresh dialog mutates nothing (psql counts unchanged).
   - Trigger an alert evaluation for an exchange the deleted ticker belonged to. Confirm the scheduler/evaluator no longer attempts the deleted ticker (log inspection is enough).

10. **No regression**
    - `pnpm --filter web typecheck` exits 0.
    - `pnpm --filter web lint` exits 0 (or existing lint baseline preserved).
    - `go test ./...` from repo root passes.
    - Existing Stock Database filters, sorting, export, and row rendering are unchanged for tickers that are not deleted.
    - Existing `/alerts`, `/portfolios`, `/scanner`, `/price-database` pages render and function unchanged.

## Conventions

- **RPC placement**: `PriceService` owns all metadata and price-data RPCs; both new RPCs live there. Do not create a new service.
- **RPC naming**: follow the `<Verb><Resource>` pattern (`DeleteTicker`, `PreviewDeleteTicker`). The preview verb is `Preview` rather than `Count` because the intent is "show me what a delete would do" — not a generic counting operation.
- **sqlc query naming**: one `DeleteTickerFrom<TableCamel>` query per table (`:execrows`), plus a preview set. Match the existing `database/sql/queries/` file layout.
- **Transaction pattern**: use `pgxpool.Pool.BeginTx(ctx, pgx.TxOptions{})` with `defer tx.Rollback(ctx)` and explicit `tx.Commit(ctx)` at the end. Mirror any existing transaction pattern already present in `apps/grpc/` (check the alert_service portfolio handler for the closest precedent).
- **Error messages**: prefix the failing step so debugging is easy. Example: `"delete_from_alerts: ERROR: syntax error at or near …"`.
- **No cascading via `ON DELETE CASCADE`**: do not add schema-level foreign keys as part of this spec. The cascade is application-level, expressed explicitly in the transaction, because the existing tables have no FKs on the ticker string (a retrofit would be a larger migration and is out of scope).
- **Server-action types**: plain JS primitives only. `int64` → `number` via `Number(val)`. No `Timestamp` fields in these messages, so nothing to convert.
- **Invalidation key**: reuse the existing `STOCK_DATABASE_KEY` from `lib/store/stock-database.ts`. Also invalidate any other keys that list alerts or portfolios (`ALERTS_KEY`, `PORTFOLIOS_KEY`) so those views refresh after a ticker deletion. Fetch these from their existing store files — do not re-declare them.
- **Dark mode / styling**: shadcn primitives only; semantic tokens only; no new UI dependencies. See `apps/ui/web/CLAUDE.md §13`.
- **Sidebar**: no new sidebar entry is added — the feature is inline on the existing `/database/stock` page. If later we add a dedicated "Manage Tickers" page, that is a separate spec.
- **Ticker casing**: the ticker string passed to both RPCs is used verbatim in `WHERE ticker = $1` / `WHERE symbol = $1` / `WHERE ticker1 = $1 OR ticker2 = $1`. Do not upper-case or lower-case server-side. The UI always passes the exact string from the table row.

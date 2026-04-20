# Questions: Remove Tickers (04) — Round 1

## Resolved

**R01. Why now?**
Old exchange-delisted tickers linger in the database. They never receive new price data, yet they keep appearing in the Stock Database table, in ticker dropdowns, in scanner filters, and in stale-data warnings. Users want a single action that removes a dead ticker from the system entirely — price bars, metadata, alerts keyed on it, and audit rows — rather than manually scrubbing each table.

**R02. Which service owns the new RPC?**
`PriceService` in `proto/price/v1/price.proto`. The service already owns all stock-metadata and price-data reads (`GetFullStockMetadata`, `LoadPriceData`, `GetDatabaseStats`, `ScanStale*`). Ticker deletion is a natural extension of the metadata/price surface. `AlertService` is alert-centric and shouldn't grow into metadata ops.

**R03. One RPC or two (preview + delete)?**
**Two RPCs.** `PreviewDeleteTicker` returns per-table row counts without mutating anything. `DeleteTicker` performs the cascade in a single DB transaction and returns the actual deletion counts. Two RPCs keep the preview safely side-effect-free and make the confirmation dialog straightforward — the UI calls preview on open, delete on confirm.

**R04. Tables to cascade through (in order)**
All deletions run inside one Postgres transaction. Order is chosen to minimize lock contention and reflect dependency direction (reference tables first, then primary tables):

1. `portfolio_stocks` — WHERE `ticker = $1`
2. `alert_audits` — WHERE `ticker = $1`
3. `alerts` — WHERE `ticker = $1 OR ticker1 = $1 OR ticker2 = $1` (ratio alerts are also deleted)
4. `daily_move_stats` — WHERE `ticker = $1`
5. `daily_prices` — WHERE `ticker = $1`
6. `hourly_prices` — WHERE `ticker = $1`
7. `weekly_prices` — WHERE `ticker = $1`
8. `continuous_prices` — WHERE `symbol = $1`
9. `futures_metadata` — WHERE `symbol = $1`
10. `ticker_metadata` — WHERE `ticker = $1`
11. `stock_metadata` — WHERE `symbol = $1` (last, since other tables key off the symbol string)

**R05. Audit trail — keep or delete?**
**Delete.** Once a ticker's metadata is gone, audit rows referencing that ticker string become dangling — they can no longer be joined back to a live stock for display (`GetTriggerHistoryByTicker` would return orphans with no current metadata). Deleting audit rows also keeps `ClearAuditData`-style integrity: the audit table should only contain rows for tickers currently tracked by the system. Users who want to keep audit history should not delete the ticker.

**R06. Ratio alerts (ticker1/ticker2)**
Delete the full alert row if either `ticker1` or `ticker2` matches the ticker being removed. A ratio alert with one dead leg is broken; half-deleting it would be worse than full removal. The preview surfaces these separately so the user sees "3 ratio alerts will be deleted (this ticker is ticker1 in 2 and ticker2 in 1)".

**R07. `app_documents` (legacy JSON payloads)**
**Out of scope.** `app_documents` holds legacy JSON snapshots (alerts.json, portfolios.json) that the Postgres tables have superseded. New deletions only affect the live tables; any stale JSON snapshot is already stale and not the authoritative source. If a future audit finds the documents still drive behavior somewhere, that is a separate cleanup spec.

**R08. Where does the UI live?**
Inline on the existing Stock Database page (`/database/stock`). That page already renders every ticker with exchange, RBICS, last-updated, and stock name — it is the natural place to identify and remove dead tickers. Add a row-level "Delete" icon button; clicking it opens a confirmation dialog that shows the preview counts. No new top-level route is added.

**R09. Bulk delete?**
**Out of scope for this spec.** Start with single-ticker delete. Bulk delete multiplies the blast radius and requires its own UX (which tickers? stream progress? partial failure?). If users end up with dozens of dead tickers to purge, a follow-up spec can add a bulk flow once the single-ticker cascade is battle-tested.

**R10. Exchange/asset-type filter (are these really dead tickers)?**
**Out of scope for this spec.** The UI presents the delete button on every row; the user decides whether the ticker is dead. A future enhancement could highlight stale tickers (e.g. `ticker_metadata.last_update` older than N days) in the table or add a "Stale tickers" tab, but that is separate UX work and not required for the core deletion flow.

**R11. Permissions / confirmation**
Single-step confirmation via a modal `AlertDialog` showing the preview counts and requiring a click on a red destructive button. No re-typing of the ticker name, no soft-delete trash. This matches the project's existing destructive-action pattern (see the Delete Alerts page and `BulkDeleteAlerts`).

**R12. Python Streamlit parity?**
**No.** The Streamlit UI has no existing "remove ticker" feature. Nothing to port. The new feature lives in the Next.js UI only. Python services that read the deleted tables (scheduler, alert checker) will naturally stop evaluating the removed ticker on the next run — no Python code change is required.

**R13. Transactionality and error behavior**
The whole cascade runs in one `BEGIN ... COMMIT` block. If any step errors, the transaction rolls back and no rows are deleted. The RPC returns `success=false, error_message="<step>: <pg error>"`. Partial state is not possible.

**R14. Return-value shape**
`DeleteTickerResponse` returns a `DeletionCounts` struct with one int64 per affected table, plus a top-level `success` and `error_message`. The UI can show "Removed AAPL: 1 stock_metadata, 3 alerts, 1,247 daily bars, …" in a success toast.

**R15. Tests**
- Go unit tests for the handler using a transactional fixture DB (same pattern used by existing `alert_service` tests).
- An integration test that seeds rows in every affected table for a fake ticker, calls `DeleteTicker`, and asserts zero rows remain in every table.
- A rollback test that induces a SQL error mid-cascade and asserts the full cascade is rolled back.

**R16. Ticker casing**
Tickers are stored as-is (usually uppercase, e.g. `AAPL`, `SPY`). Match exactly — do not uppercase/lowercase in the handler. If the caller passes `aapl`, the RPC returns zero counts because no row matches. The UI always passes the canonical ticker string from the table row.

**R17. Scanner / evaluation impact after delete**
After a delete:
- The scheduler loops iterate over `ticker_metadata` or `stock_metadata`, so the ticker vanishes from the next run automatically.
- `LoadPriceData` called with the deleted ticker returns zero rows (same behavior as an unknown ticker).
- No Python code changes needed.

## Open

(none — all design questions for round 1 are resolved above)

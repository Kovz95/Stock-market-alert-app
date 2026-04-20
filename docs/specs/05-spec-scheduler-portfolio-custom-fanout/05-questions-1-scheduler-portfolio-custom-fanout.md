# Questions: Scheduler Portfolio + Custom Channel Fan-out (05) — Round 1

## Resolved

**R01. Why now?**
The Go scheduler (`apps/scheduler/`) is the production runtime going forward; the Python `StockAlertChecker` is legacy and being retired. But two alert-dispatch paths that exist in Python were never ported:

1. **Portfolio fan-out** (`src/services/stock_alert_checker.py:452-476` → `src/services/portfolio_discord.py`). When an alert triggers for a ticker, Python iterates `portfolios` looking for any enabled portfolio that contains the ticker in its stocks list, and sends the embed to each portfolio's `discord_webhook`.
2. **Custom-condition-channel fan-out** (`src/services/stock_alert_checker.py:437-443` → `src/services/discord_routing.py:221-323`). When an alert triggers, Python also checks the `custom_discord_channels` document for any enabled channel whose condition matches the alert's condition (exact-match after normalization, or the literal `price_level` keyword), and delivers the embed to each matching webhook.

The Go scheduler's `onTriggered` callback (`apps/scheduler/internal/handler/common.go:196-232`) resolves a single economy-routed webhook and accumulates the embed to it — nothing else. Result: the CBRE portfolio (69 stocks, `discord_webhook` set) has received no alerts since the Python scheduler was retired, and custom channels created via the spec-03 UI never fire.

**R02. Which package owns the fan-out logic?**
The shared `discord/` package, sitting next to `routing.go`. Add two new source files:
- `discord/portfolios.go` — loads portfolios from Postgres and resolves matching webhooks for a ticker.
- `discord/custom.go` — loads the `custom_discord_channels` document and resolves matching webhooks for a triggered alert's conditions.

Both new types (`PortfolioResolver`, `CustomChannelResolver`) follow the same shape as the existing `Router`: loaded at worker startup, cached in memory, refreshed via a TTL identical to the Python router (5 minutes).

The scheduler's `Common` struct gains two fields of those types, and `onTriggered` calls both resolvers in addition to the existing `Router.ResolveWebhookURL`.

**R03. How is portfolio membership matched?**
By uppercase-insensitive string match against `portfolio_stocks.ticker`. Python uses `stock.get('symbol', '').upper() == ticker.upper()` (`src/services/portfolio_discord.py:53`). The Go resolver normalizes to uppercase once at load time and looks up tickers uppercase. Tickers with exchange suffixes (e.g. `SHEL.L`, `BHP-AU`) are matched verbatim against the stored string — the resolver does not strip or mutate the suffix.

**R04. How is a "custom channel condition match" defined?**
Exactly as Python's `check_custom_channel_condition` (`src/services/discord_routing.py:221-323`):

1. Skip channels where `enabled != true`.
2. Skip channels with the legacy `condition_type`/`condition` split schema (detected by presence of `condition_type` without `condition`).
3. Normalize both the channel's stored condition string and each of the alert's condition strings:
   - Strip spaces around `<`, `>`, `=`, `!`, `(`, `)`, `,`, `[`, `]`.
   - Collapse all remaining whitespace runs to a single space.
   - `strings.TrimSpace` + `strings.ToLower`.
4. **Special case:** if the normalized channel condition equals the literal `"price_level"`, the channel matches when **any** of the alert's condition strings matches the regex `(close|open|high|low)(\[-?\d+\])?\s*([<>=!]+)\s*\$?\d+\.?\d*` (case-insensitive).
5. Otherwise, the channel matches when the normalized channel condition is byte-equal to any normalized alert condition.

This Go implementation is a line-for-line port; no semantic drift is permitted. A unit test suite (Task 06) pins every behavior in the Python logic including the `price_level` keyword and the normalization edge cases.

**R05. Where do the alert's condition strings come from?**
From `alert.ExtractConditions(a.Conditions)` (`alert/evaluator.go:59`), which already exists and is used by the scheduler elsewhere. The scheduler's `onTriggered` at `common.go:210` calls this to build the embed description — the same extracted `[]string` is reused for custom-channel matching.

**R06. Cache TTL and reload policy**
Match the Python router (`src/services/discord_routing.py:24-26` — `_CONFIG_CACHE_TTL = 300` seconds). Both resolvers reload lazily when a call lands after the TTL. Loads go through a `sync.Mutex` so concurrent triggers don't cause dogpile reloads.

**R07. Dedup rule when the economy channel equals a portfolio/custom webhook**
Dedup at the `Accumulator.Add()` call site, not in the resolvers. Before adding to the accumulator for a given webhook URL, check a `seen map[string]bool` local to the `onTriggered` closure. If the URL has already been added for this alert, skip. This matches the Python behavior where the same webhook URL receiving two copies of the same embed is considered a bug.

**R08. New sqlc queries**
Two new queries in a new file `database/sql/queries/portfolio_queries.sql`:

```sql
-- name: ListPortfoliosForFanout :many
SELECT id, name, discord_webhook, enabled
FROM portfolios
WHERE enabled = TRUE AND discord_webhook IS NOT NULL AND discord_webhook <> '';

-- name: ListPortfolioStocks :many
SELECT portfolio_id, ticker
FROM portfolio_stocks;
```

Custom channels already have a read path via `GetAppDocument("custom_discord_channels")` (see `apps/grpc/discord_service/custom_channels.go:42`) — no new sqlc query needed. Reuse the existing generated `GetAppDocument`.

**R09. Should the resolvers go in a new package, or live inside `discord/`?**
Inside `discord/`. Both resolvers send alerts to Discord webhooks through the same `Accumulator`; they're the same concern. A new package would force the scheduler's `Common` struct to import two more packages, plus force the `discord/` package to export internal plumbing. Keep everything in `discord/`, matching the existing `Router`, `Notifier`, `StatusNotifier`, `Accumulator` pattern.

**R10. Should this also change the Python scheduler?**
**No.** Python is frozen. Do not modify `src/services/stock_alert_checker.py` or `src/services/discord_routing.py` or `src/services/portfolio_discord.py`. The custom-channel storage (`custom_discord_channels` document) and the portfolio storage (`portfolios` + `portfolio_stocks` tables) are **shared** between Go and Python — the Go resolvers read the same rows Python wrote. No schema changes.

**R11. Condition matching: support `conditions` object shapes beyond `string`**
Python walks multiple shapes in `check_custom_channel_condition` (lines 268-289): dict with `conditions` key, list of lists, list of strings. `alert.ExtractConditions` already collapses all those shapes into a flat `[]string` (see `alert/evaluator.go:53-58` docstring and its tests). The Go resolver works purely against that flat list, so all shape handling is already covered.

**R12. Rate limiting**
Already handled. The existing `Accumulator` and `Notifier` enforce per-webhook rate limits (`DISCORD_WEBHOOK_INTERVAL_SEC`, default 2s). Nothing to add at the fan-out layer.

**R13. Alert embed reuse**
The embed built by `discord.FormatAlertEmbed` (`common.go:216`) is identical for every destination. Build it once per trigger and hand the same `discord.Embed` value to every `Accumulator.Add` call. Do not re-build for each portfolio or custom channel.

**R14. Failure isolation**
A failure resolving one portfolio's webhook (e.g. stale row, malformed URL) must not prevent fan-out to other portfolios or to the economy channel. Resolvers return `([]string, error)` — on error they log at Warn level and return whatever partial list they built. The scheduler's `onTriggered` treats any returned URL list as authoritative regardless of accompanying error.

**R15. Scheduler status reporting**
Triggered-alert metrics (`alertStats.Triggered`) count **alerts**, not Discord sends. Do not increment Triggered per fan-out destination. Do not add a "fan-out sent" stat to `AlertStats` in this spec — it's out of scope. (Observability for fan-out volume is a follow-up.)

**R16. Tests**
- Unit tests for `discord/custom.go` condition matching (port Python's `check_custom_channel_condition` test cases including `price_level`, normalization, enabled/disabled).
- Unit tests for `discord/portfolios.go` membership lookup including empty ticker, multiple portfolios containing the same ticker, disabled portfolios, portfolios with blank webhook.
- Integration test for `common.go`'s `onTriggered` verifying that given a seed portfolio + custom channel + triggered alert, all three webhooks receive the embed and the dedup rule collapses a duplicate URL.
- No end-to-end test against a live Discord webhook (existing tests don't send, they mock `Notifier`).

**R17. Backward compatibility**
If the `portfolios` table is empty or the `custom_discord_channels` document doesn't exist, both resolvers return an empty slice with no error. The scheduler behaves exactly as today: only the economy channel receives the embed.

**R18. Shadow mode interaction**
Shadow mode (`SCHEDULER_SHADOW_MODE=true`) records only the triggered alert IDs — see `ShadowRecord` (`common.go:24-28`). Fan-out destinations aren't part of the shadow record. No change to shadow output in this spec.

## Open

(none — all design questions for round 1 are resolved above)

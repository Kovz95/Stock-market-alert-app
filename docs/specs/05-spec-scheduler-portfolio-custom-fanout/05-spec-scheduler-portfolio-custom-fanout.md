# Spec: Scheduler Portfolio + Custom Channel Fan-out (05)

## Goal

Restore two alert-dispatch paths that exist in the legacy Python scheduler but were never ported to the Go scheduler (`apps/scheduler/`), which is the production runtime going forward. When a triggered alert fires in Go, the current implementation (`apps/scheduler/internal/handler/common.go:196-232`) delivers the embed only to the economy-routed webhook resolved by `discord.Router`. Two other destinations that Python used to hit are silently dropped:

1. **Portfolios**: every enabled row in the `portfolios` table whose membership (via `portfolio_stocks`) contains the alert's ticker must also receive the embed on its `discord_webhook`. The user's `CBRE` portfolio (69 stocks, webhook set) has received zero alerts since Python was retired last fall.
2. **Custom condition channels**: every enabled entry in the `custom_discord_channels` app document whose condition string matches the alert's condition (after the Python-equivalent normalization, or the special `price_level` keyword) must receive the embed on its stored webhook. Channels created via the spec-03 UI never fire today.

This spec ports both fan-out paths into the shared `discord/` package as two new resolvers, adds sqlc queries for portfolio membership, and threads the resolver calls through `onTriggered` with per-alert webhook-URL deduplication — so when an alert fires, the embed goes to **every** destination Python used to reach, once each.

## Scope

### In scope

- **`discord/portfolios.go`** (new): `PortfolioResolver` type that loads `portfolios` + `portfolio_stocks` from Postgres at startup, caches them in memory with a 5-minute TTL (matching Python's `_CONFIG_CACHE_TTL`), exposes `ResolveWebhooks(ticker string) []string`. Uppercase-insensitive ticker match; exchange suffixes preserved verbatim. Disabled portfolios and portfolios with blank `discord_webhook` are excluded at load time.
- **`discord/custom.go`** (new): `CustomChannelResolver` type that loads the `custom_discord_channels` document via the existing `GetAppDocument` sqlc query, caches it with the same 5-minute TTL, exposes `ResolveWebhooks(conditions []string) []string`. Condition matching is a line-for-line port of Python's `check_custom_channel_condition` (`src/services/discord_routing.py:221-323`) including:
  - Enabled/disabled filtering.
  - Skip for legacy `condition_type`-split schema entries.
  - Whitespace normalization: strip spaces around `< > = ! ( ) , [ ]`, collapse runs of whitespace to one space, `TrimSpace` + `ToLower`.
  - Special case: channel condition equal to the literal `price_level` matches any alert condition matching the regex `(close|open|high|low)(\[-?\d+\])?\s*([<>=!]+)\s*\$?\d+\.?\d*` (case-insensitive).
  - Otherwise: byte-equal match between the normalized channel condition and any normalized alert condition.
- **sqlc queries** (`database/sql/queries/portfolio_queries.sql`, new): `ListPortfoliosForFanout :many` (enabled portfolios with non-blank webhooks) and `ListPortfolioStocks :many` (all `(portfolio_id, ticker)` rows). The custom-channel side reuses the existing `GetAppDocument` query for key `custom_discord_channels`.
- **Scheduler wiring** (`apps/scheduler/internal/handler/common.go`, `apps/scheduler/cmd/scheduler/main.go` or equivalent startup wiring): add `PortfolioResolver` and `CustomChannelResolver` fields to the `Common` struct. Construct them at worker startup. In `onTriggered`, after building the embed once, call both resolvers, merge the returned URLs with the existing economy webhook, deduplicate against a `seen map[string]bool` local to the closure, and call `Accum.Add(url, embed)` per unique URL.
- **Unit tests**: `discord/portfolios_test.go`, `discord/custom_test.go` covering every behavior listed in R04 and R16 of the questions file, plus an integration test for the combined `onTriggered` fan-out + dedup.
- **No Python changes.** Python (`src/services/stock_alert_checker.py`, `src/services/discord_routing.py`, `src/services/portfolio_discord.py`) is frozen. The storage (`portfolios` + `portfolio_stocks` tables, `custom_discord_channels` app document) is shared; Go reads the same rows Python wrote.

### Out of scope

- **Any change to `custom_discord_channels` or `portfolios` storage schema.** Shared with Python. No migration.
- **Any change to the Python scheduler.** Python is frozen.
- **Per-destination observability / fan-out send counters.** `alertStats.Triggered` still counts alerts, not Discord sends. Surfacing per-webhook volume is a follow-up.
- **Changes to shadow mode.** `ShadowRecord` records triggered alert IDs only; fan-out destinations are not recorded and this spec does not change that.
- **Rate limiting changes.** `Accumulator` + `Notifier` already enforce per-webhook rate limits (`DISCORD_WEBHOOK_INTERVAL_SEC`). No change.
- **Embed format changes.** The embed produced by `discord.FormatAlertEmbed` is sent verbatim to every destination — portfolio and custom channels receive the exact same embed the economy channel receives.
- **New admin UI for managing portfolios.** `portfolios` is already editable through existing UI paths; this spec only reads from that storage.
- **Back-filling alerts that were missed while the Go scheduler was running without fan-out.** Only forward-going alerts reach the new destinations.
- **End-to-end tests against a live Discord webhook.** Existing test harness mocks `Notifier`; this spec follows that pattern.

## Source excerpts

- `src/services/stock_alert_checker.py:437-476` — Python reference for both fan-out paths: the block at 437-443 handles custom channels; 452-476 handles portfolios. Both run after the primary economy dispatch.
- `src/services/discord_routing.py:24-26` — `_CONFIG_CACHE_TTL = 300` seconds — the cache TTL the Go resolvers match.
- `src/services/discord_routing.py:221-323` — `check_custom_channel_condition` and `is_price_level_condition`. This is the function being ported line-for-line into `discord/custom.go`.
- `src/services/portfolio_discord.py` — Python portfolio membership logic. Line 53: `stock.get('symbol', '').upper() == ticker.upper()` — the uppercase-insensitive match rule.
- `apps/scheduler/internal/handler/common.go:196-232` — current `onTriggered`, where resolver calls are added.
- `apps/scheduler/internal/handler/common.go:24-28` — `ShadowRecord` (unchanged by this spec).
- `discord/routing.go` — existing `Router` struct; reference pattern for the two new resolvers (in-memory cache + TTL reload under a mutex).
- `discord/accumulator.go` — `Accumulator.Add(url, embed)` signature; the dedup loop in `onTriggered` wraps calls to this.
- `alert/evaluator.go:53-58` — `ExtractConditions(raw []byte) ([]string, error)`. Already collapses every condition-shape variant (nested dicts, list-of-lists, list-of-strings) to a flat `[]string`. Both resolvers work against that flat slice.
- `apps/grpc/discord_service/custom_channels.go:42` — existing `GetAppDocument("custom_discord_channels")` read path; the new resolver reuses this sqlc-generated query.
- `database/sql/schema.sql:215-232` — `portfolios` + `portfolio_stocks` schema (referenced for the new sqlc query file).

## Contract

No protobuf changes. The new code is entirely internal to the Go scheduler and the shared `discord/` package.

The two new resolvers expose the following Go APIs:

```go
// discord/portfolios.go
type PortfolioResolver struct {
    // unexported fields: db pool, cache, mutex, TTL, last-load timestamp, logger
}

func NewPortfolioResolver(pool *pgxpool.Pool, ttl time.Duration, logger *slog.Logger) *PortfolioResolver

// ResolveWebhooks returns the set of discord_webhook URLs for every enabled portfolio
// containing `ticker` (uppercase-insensitive match). Returns nil on cache miss with
// fatal error; returns a (possibly empty) slice with logged warnings on partial failure.
func (r *PortfolioResolver) ResolveWebhooks(ctx context.Context, ticker string) ([]string, error)
```

```go
// discord/custom.go
type CustomChannelResolver struct {
    // unexported fields: db pool, cache, mutex, TTL, last-load timestamp, logger
}

func NewCustomChannelResolver(pool *pgxpool.Pool, ttl time.Duration, logger *slog.Logger) *CustomChannelResolver

// ResolveWebhooks returns the set of webhook URLs for every enabled custom channel whose
// condition matches any string in `conditions` (after normalization, or via the
// `price_level` special keyword).
func (r *CustomChannelResolver) ResolveWebhooks(ctx context.Context, conditions []string) ([]string, error)
```

**Caching contract:** Both resolvers load lazily on first `ResolveWebhooks` call and reload when `time.Since(lastLoad) >= ttl`. A `sync.Mutex` gates the reload path so concurrent alert triggers don't dogpile. Default TTL is 5 minutes (`5 * time.Minute`), matching Python.

**Failure-isolation contract:** A load error (DB unreachable, malformed JSON in the custom-channels document, etc.) is logged at Warn level; the resolver returns whatever partial list it could build plus the error. The scheduler's `onTriggered` treats the returned URL list as authoritative regardless of whether `err != nil` — a Warn log and a shorter fan-out list is preferable to dropping the entire alert.

**Scheduler dispatch contract:** `onTriggered` accumulates the destination set in this order, deduplicating by URL into a local `seen map[string]bool`:

1. The economy webhook from `Router.ResolveWebhookURL(ticker, timeframe, exchange, isRatio)` (existing behavior).
2. All URLs from `PortfolioResolver.ResolveWebhooks(ctx, ticker)`.
3. All URLs from `CustomChannelResolver.ResolveWebhooks(ctx, conditions)` where `conditions = alert.ExtractConditions(a.Conditions)` (already computed for the embed).

For each unique URL, `Accum.Add(url, embed)` is called **once** with the single `discord.Embed` already built. The embed is not re-built per destination.

## Acceptance criteria

1. **Portfolio resolver (`discord/portfolios.go`)**
   - `NewPortfolioResolver(pool, ttl, logger)` exists and returns a pointer to a new resolver.
   - First call to `ResolveWebhooks(ctx, "AAPL")` triggers a single Postgres roundtrip running the new `ListPortfoliosForFanout` and `ListPortfolioStocks` sqlc queries.
   - Membership match is uppercase-insensitive: a portfolio with `portfolio_stocks.ticker = 'aapl'` or `'AAPL'` matches a call for `"aapl"`, `"AAPL"`, or `"Aapl"`.
   - Exchange-suffixed tickers match verbatim: `"SHEL.L"` matches only the stored row `'SHEL.L'`, never `'SHEL'`.
   - Disabled portfolios (`enabled = FALSE`) and portfolios with blank `discord_webhook` are excluded at load (the sqlc query's WHERE clause filters them out).
   - Multiple portfolios containing the same ticker all return their webhooks; duplicates within the returned slice are permitted (dedup happens at the call site in `onTriggered`).
   - Empty `portfolios` table → `ResolveWebhooks` returns `(nil, nil)` or `([]string{}, nil)`.
   - Cache reload after TTL: a second call within 5 minutes of a successful load issues **no** DB queries. A call after `time.Since(lastLoad) > ttl` issues the two queries again.
   - Concurrent callers during reload do not both issue queries; a `sync.Mutex` serializes the reload path.

2. **Custom-channel resolver (`discord/custom.go`)**
   - `NewCustomChannelResolver(pool, ttl, logger)` exists and returns a pointer to a new resolver.
   - First call to `ResolveWebhooks(ctx, conditions)` issues a single `GetAppDocument("custom_discord_channels")` query.
   - Enabled/disabled filtering: channels with `enabled != true` never match.
   - Legacy-schema filter: channels with `condition_type` populated but no `condition` field are skipped (per Python `check_custom_channel_condition` path at lines 236-244 of `discord_routing.py`).
   - Normalization: both the channel's stored condition and each alert condition are run through: strip spaces around `< > = ! ( ) , [ ]`, collapse whitespace runs to single space, `TrimSpace`, `ToLower`. Match is byte-equal on the resulting strings.
   - `price_level` special keyword: when the channel's normalized condition is literally `"price_level"`, it matches iff **any** alert condition matches `(?i)(close|open|high|low)(\[-?\d+\])?\s*([<>=!]+)\s*\$?\d+\.?\d*`. This is case-insensitive and anchored with `MatchString`, not `FindString`.
   - Empty `custom_discord_channels` document or missing document → `ResolveWebhooks` returns `(nil, nil)` or `([]string{}, nil)`.
   - TTL and concurrency guarantees are identical to `PortfolioResolver`.

3. **sqlc queries**
   - `database/sql/queries/portfolio_queries.sql` exists and contains:
     - `ListPortfoliosForFanout :many` — returns `(id, name, discord_webhook, enabled)` filtered by `enabled = TRUE AND discord_webhook IS NOT NULL AND discord_webhook <> ''`.
     - `ListPortfolioStocks :many` — returns `(portfolio_id, ticker)` for every row in `portfolio_stocks`.
   - `sqlc generate` exits 0 and regenerates the `database/db/` bindings.
   - No new query is added for `custom_discord_channels`; the existing `GetAppDocument` query is reused.

4. **Scheduler wiring**
   - `Common` struct (`apps/scheduler/internal/handler/common.go`) gains fields `Portfolios *discord.PortfolioResolver` and `CustomChannels *discord.CustomChannelResolver`.
   - The scheduler's main/startup path (wherever `Common` is constructed — typically `apps/scheduler/cmd/scheduler/main.go` or `apps/scheduler/internal/handler/handler.go`) wires both resolvers with the shared `pgxpool.Pool` and `5 * time.Minute` TTL.
   - `onTriggered` builds the embed once (existing `FormatAlertEmbed` call), then fans out to the merged, deduplicated URL set as specified in the Contract section.

5. **Dedup**
   - Given: portfolio CBRE has webhook `W1`; a custom channel for condition `X` has webhook `W2`; the economy routing for the triggered ticker resolves to `W1` (same as CBRE).
   - Result: `Accum.Add` is called exactly **twice** — once for `W1`, once for `W2`. Never three times.
   - A `seen map[string]bool` local to the `onTriggered` closure enforces this. Not a resolver-level concern.

6. **Failure isolation**
   - If the portfolio resolver returns `(partialURLs, err)` with `err != nil`, the scheduler still dispatches to every URL in `partialURLs` AND still dispatches to the economy webhook AND still calls the custom-channel resolver. A single portfolio's malformed webhook does not block any other destination.
   - A Warn-level log entry is emitted containing the error text; the alert trigger itself succeeds.

7. **Backward compatibility**
   - With an empty `portfolios` table and no `custom_discord_channels` document: the scheduler behaves **exactly** as it does today — only the economy webhook receives the embed.
   - No environment variable gates this feature. The new code paths are always on.

8. **Tests**
   - `discord/portfolios_test.go` exists and covers: empty table, single portfolio match, multi-portfolio match for same ticker, uppercase-insensitive match, exchange-suffixed tickers, disabled-portfolio exclusion, blank-webhook exclusion, cache TTL hit (no extra queries), cache TTL miss (reload). Uses a mock or a real transactional test DB — match whatever pattern already exists in `discord/` for unit tests.
   - `discord/custom_test.go` exists and covers: enabled vs disabled, legacy schema skip, normalization of each operator/bracket character, `price_level` keyword with positive and negative regex matches, byte-equal non-match for near-miss strings, empty document, cache TTL hit/miss.
   - `apps/scheduler/internal/handler/common_test.go` (either new file or addition) has an integration test for `onTriggered` that seeds a portfolio + custom channel + triggered alert with `W1` shared by the economy channel and a portfolio, and asserts `Accum.Add` is called exactly twice with the correct URLs — verifying dedup and fan-out together.
   - `go test ./discord/... ./apps/scheduler/...` passes.

9. **No regression**
   - `go build ./...` exits 0 from repo root.
   - `go test ./...` exits 0.
   - Existing Python fan-out code is unmodified.
   - Existing `discord.Router` behavior is unmodified.
   - Existing `Accumulator` and `Notifier` behavior is unmodified.
   - The scheduler's shadow mode records the same `ShadowRecord` data it recorded before this spec.

## Conventions

- **Package placement**: both resolvers live in `discord/`, same package as `Router`, `Notifier`, `Accumulator`. Do not introduce new packages; do not force new imports on the scheduler's `Common` beyond the existing `discord` import.
- **Normalization port**: the regex and character list used in `discord/custom.go` normalization must be a line-for-line port of `normalize_condition_string` in `src/services/discord_routing.py`. No semantic drift. The `price_level` regex is reproduced verbatim from `is_price_level_condition`.
- **Ticker casing**: the portfolio resolver normalizes tickers to uppercase **once at load time**. Lookups uppercase the incoming ticker string once per `ResolveWebhooks` call. Neither the stored string nor the incoming string is mutated beyond that. Do not strip exchange suffixes (`.L`, `-AU`, etc.).
- **Dedup placement**: dedup lives at the `Accumulator.Add` call site inside `onTriggered`, not in resolvers. Resolvers are free to return overlapping slices; the closure consolidates.
- **Error style**: resolvers return `([]string, error)`. Callers in `onTriggered` log the error at Warn and proceed with the returned slice. No panics from fan-out paths.
- **Cache TTL constant**: define as a package-level default in `discord/` (e.g. `DefaultResolverTTL = 5 * time.Minute`). Do not hardcode `300 * time.Second` or similar at the call site.
- **Condition extraction**: do **not** call `alert.ExtractConditions` twice. `onTriggered` already computes it at line 210. Reuse that `[]string` for both the embed description and the custom-channel resolver call.
- **No protobuf changes**: this is an internal-dispatch feature. No RPC, no server action, no UI change.
- **No new sqlc queries for custom channels**: reuse `GetAppDocument`. If the Go bindings already expose a typed `CustomChannels` helper (see `apps/grpc/discord_service/custom_channels.go`), decide at implementation time whether the resolver wraps that helper or calls `GetAppDocument` directly; either is acceptable as long as it does not duplicate parsing logic.
- **Logging**: use the `*slog.Logger` already passed through `Common`. Log resolver load events at Debug, load errors at Warn, and include the resolver name, duration, and row count.
- **Testing pattern**: follow the existing pattern in the `discord/` package. If existing tests use a mock `pgxpool`, mock. If they use a test DB via `testcontainers` or a `TEST_DATABASE_URL`, use that. Do not introduce a third pattern.
- **Shared storage invariant**: the code path that reads `portfolios` / `portfolio_stocks` / `custom_discord_channels` must never write. Python owns writes today; Go reads only. Any write path is a future spec.

# Tasks: Scheduler Portfolio + Custom Channel Fan-out (05)

## Task 01 - Add sqlc queries for portfolio fan-out

- Create `database/sql/queries/portfolio_queries.sql` (or append to an existing `*_queries.sql` file that already contains portfolio reads — inspect the directory first and choose the most-specific existing file).
- Add two sqlc-named queries:
  ```sql
  -- name: ListPortfoliosForFanout :many
  SELECT id, name, discord_webhook, enabled
  FROM portfolios
  WHERE enabled = TRUE
    AND discord_webhook IS NOT NULL
    AND discord_webhook <> '';

  -- name: ListPortfolioStocks :many
  SELECT portfolio_id, ticker
  FROM portfolio_stocks;
  ```
- Run `sqlc generate` (use whatever command the project already uses — check `sqlc.yaml` / `Makefile` / `go.work` scripts).
- Confirm generated Go in `database/db/` (or the project's sqlc output path) exposes `ListPortfoliosForFanout` and `ListPortfolioStocks` typed methods.

**Proof:** 05-proofs/05-task-01-proofs.md

## Task 02 - Implement `PortfolioResolver` in `discord/`

- Create `discord/portfolios.go`:
  - Define `type PortfolioResolver struct` with unexported fields: `pool *pgxpool.Pool`, `ttl time.Duration`, `logger *slog.Logger`, `mu sync.Mutex`, `loadedAt time.Time`, and an in-memory cache (e.g. `byTicker map[string][]string` where key is uppercase ticker and value is the list of webhook URLs for portfolios containing it).
  - `NewPortfolioResolver(pool *pgxpool.Pool, ttl time.Duration, logger *slog.Logger) *PortfolioResolver`.
  - `ResolveWebhooks(ctx context.Context, ticker string) ([]string, error)`:
    - Uppercase the incoming ticker.
    - Acquire `mu`; if `loadedAt` is zero or `time.Since(loadedAt) >= ttl`, call `loadLocked(ctx)` to rebuild the `byTicker` map; release `mu`.
    - Return `byTicker[upper(ticker)]` (copy the slice — callers may retain it).
  - `loadLocked`: calls `ListPortfoliosForFanout` and `ListPortfolioStocks`, builds `portfolioID -> webhook` first, iterates `portfolio_stocks`, normalizes ticker to uppercase, appends webhook to `byTicker[upper]`. On per-query error returns error but updates `loadedAt` so retry still respects TTL (avoid tight retry loops).
- Export a package constant `DefaultResolverTTL = 5 * time.Minute`.
- No Discord send calls live in this file. It only resolves URLs.

**Proof:** 05-proofs/05-task-02-proofs.md

## Task 03 - Implement `CustomChannelResolver` in `discord/`

- Create `discord/custom.go`:
  - Define `type CustomChannelResolver struct` with same field pattern as `PortfolioResolver` but holding the parsed custom-channel entries (a slice of an internal struct with `Enabled bool`, `Webhook string`, `NormalizedCondition string`, `IsPriceLevel bool`).
  - `NewCustomChannelResolver(pool *pgxpool.Pool, ttl time.Duration, logger *slog.Logger) *CustomChannelResolver`.
  - `loadLocked`: call the existing `GetAppDocument("custom_discord_channels")` sqlc method. Parse the JSON (reusing the existing `customChannelEntry` struct from `apps/grpc/discord_service/custom_channels.go` if it is exported, otherwise define a local matching struct — **do not** copy parsing logic by hand; prefer to share a type). For each entry:
    - Skip if `enabled != true`.
    - Skip if `condition_type` is set and `condition` is not (legacy schema per R04).
    - Normalize `condition` via `normalizeConditionString(raw string) string`:
      - Regex replace `\s*([<>=!(),\[\]])\s*` → `$1` (strip spaces around operators/brackets/commas).
      - Replace runs of whitespace (`\s+`) with a single space.
      - `strings.TrimSpace`, `strings.ToLower`.
    - Flag `IsPriceLevel = (normalized == "price_level")`.
  - `ResolveWebhooks(ctx context.Context, conditions []string) ([]string, error)`:
    - Trigger reload if TTL expired (same pattern as `PortfolioResolver`).
    - Normalize every alert condition once into a `[]string`.
    - Precompile the `price_level` regex: `regexp.MustCompile(`(?i)(close|open|high|low)(\[-?\d+\])?\s*([<>=!]+)\s*\$?\d+\.?\d*`)` — **package-level `var`** so it compiles once.
    - For each cached entry: if `IsPriceLevel`, match iff any normalized alert condition matches the regex (`re.MatchString`). Else: match iff any normalized alert condition equals the cached `NormalizedCondition`. Append `Webhook` on match.
    - Return the resulting slice (duplicates OK; dedup at call site).
- No Discord send calls live in this file.

**Proof:** 05-proofs/05-task-03-proofs.md

## Task 04 - Wire resolvers into the scheduler's `Common` struct

- Edit `apps/scheduler/internal/handler/common.go`:
  - Add fields to `Common`:
    ```go
    Portfolios     *discord.PortfolioResolver
    CustomChannels *discord.CustomChannelResolver
    ```
  - Update the `onTriggered` callback (currently at lines 196-232):
    - After the existing `embed := discord.FormatAlertEmbed(...)` and the economy-webhook resolution:
      ```go
      conditions := alert.ExtractConditions(a.Conditions) // reuse the slice already computed for the embed description
      ```
      (Do not compute it twice — if the existing code already binds it to a variable, use that.)
    - Build a `seen := make(map[string]bool)` and an `addURL := func(url string) { if url == "" || seen[url] { return }; seen[url] = true; c.Accum.Add(url, embed) }` closure.
    - Call `addURL(webhookURL)` for the economy URL.
    - Call `portfolioURLs, err := c.Portfolios.ResolveWebhooks(ctx, ticker)`; if `err != nil` log at Warn with the error; iterate `portfolioURLs` and call `addURL`.
    - Call `customURLs, err := c.CustomChannels.ResolveWebhooks(ctx, conditions)`; same warn-and-continue on error; iterate `customURLs` and call `addURL`.
- Locate the scheduler's main wiring (`apps/scheduler/cmd/scheduler/main.go` or wherever `Common` is constructed) and initialize both resolvers with the existing `pool` and `discord.DefaultResolverTTL`, passing the existing slog logger.
- `go build ./apps/scheduler/...` exits 0.

**Proof:** 05-proofs/05-task-04-proofs.md

## Task 05 - Unit tests for `PortfolioResolver`

- Create `discord/portfolios_test.go`. Follow whatever DB-mock or real-DB pattern the `discord/` package already uses in its tests (check `discord/routing_test.go` — if it exists, match it).
- Tests:
  1. `TestPortfolioResolver_Empty` — no rows in `portfolios` → `ResolveWebhooks` returns empty slice, no error.
  2. `TestPortfolioResolver_SingleMatch` — one enabled portfolio with webhook `W1` containing ticker `AAPL` → returns `["W1"]`.
  3. `TestPortfolioResolver_MultiMatch` — two enabled portfolios with webhooks `W1`, `W2` both containing `AAPL` → returns both.
  4. `TestPortfolioResolver_CaseInsensitive` — stored ticker `'aapl'`, call with `"AAPL"` → match. And vice versa.
  5. `TestPortfolioResolver_ExchangeSuffix` — stored ticker `'SHEL.L'`, call with `"SHEL"` → **no match**. Call with `"SHEL.L"` → match.
  6. `TestPortfolioResolver_DisabledExcluded` — disabled portfolio excluded (validates the SQL `WHERE enabled = TRUE`).
  7. `TestPortfolioResolver_BlankWebhookExcluded` — webhook `''` or NULL excluded.
  8. `TestPortfolioResolver_CacheHit` — second call within TTL issues zero DB queries (use a counting mock or `pgmock`).
  9. `TestPortfolioResolver_CacheReload` — advance clock past TTL → reload issues queries again.

**Proof:** 05-proofs/05-task-05-proofs.md

## Task 06 - Unit tests for `CustomChannelResolver`

- Create `discord/custom_test.go` with the same DB-pattern choice as Task 05.
- Tests:
  1. `TestCustomChannelResolver_Empty` — empty document → empty slice.
  2. `TestCustomChannelResolver_Enabled` — one enabled entry with condition `close > 100` matches alert condition `close > 100` → returns its webhook.
  3. `TestCustomChannelResolver_Disabled` — same entry with `enabled: false` → no match.
  4. `TestCustomChannelResolver_LegacySchemaSkipped` — entry with `condition_type: "price"` and no `condition` → no match (regardless of alert).
  5. `TestCustomChannelResolver_NormalizationWhitespace` — channel stored as `"close  >  100"`, alert condition `"close > 100"` → match.
  6. `TestCustomChannelResolver_NormalizationOperators` — channel `"close>100"`, alert `"close > 100"` → match. Channel `"a ( b )"`, alert `"a(b)"` → match.
  7. `TestCustomChannelResolver_NormalizationCase` — channel `"CLOSE > 100"`, alert `"close > 100"` → match.
  8. `TestCustomChannelResolver_PriceLevelMatch` — channel condition `"price_level"`, alert condition `"close > $100"` → match. Alert condition `"close[-1] >= 50.5"` → match. Alert condition `"rsi > 30"` → no match.
  9. `TestCustomChannelResolver_PriceLevelCaseInsensitive` — channel `"PRICE_LEVEL"` (after normalization `"price_level"`), alert `"CLOSE > $100"` → match.
  10. `TestCustomChannelResolver_ExactMatchNotPartial` — channel `"close > 100"`, alert `"close > 1000"` → **no** match (byte-equal on normalized strings).
  11. `TestCustomChannelResolver_CacheHit` and `TestCustomChannelResolver_CacheReload` — TTL behavior mirrors Task 05.
- Cross-reference Python `tests/test_discord_routing.py` (if any exists) when picking edge cases.

**Proof:** 05-proofs/05-task-06-proofs.md

## Task 07 - Integration test for `onTriggered` fan-out + dedup

- Create `apps/scheduler/internal/handler/common_fanout_test.go` (new file so existing tests are not disturbed).
- Build a `Common` with:
  - A fake `discord.Router` returning webhook `W1` for the test ticker.
  - A `PortfolioResolver` seeded with one portfolio containing the test ticker, webhook `W1` (shared with economy — exercises dedup).
  - A `PortfolioResolver` also seeded with a second portfolio containing the test ticker, webhook `W2` (non-dedup case).
  - A `CustomChannelResolver` seeded with one channel matching the test alert condition, webhook `W3`.
  - A mock `Accumulator` that records each `Add(url, embed)` call into a slice.
- Build a triggered alert with one condition string, one ticker, and call the `onTriggered` callback directly (or via the full task handler path if the test pattern already covers that).
- Assert:
  - `Accumulator.Add` was called exactly three times.
  - The URLs were `W1`, `W2`, `W3` in any order.
  - The embed passed to each call is the same `*discord.Embed` pointer (built once, shared across destinations).
- Add a second test case where the custom channel also resolves to `W1`:
  - Assert `Accum.Add` called **twice** total — W1 (once) and W2 (once). Never three times.

**Proof:** 05-proofs/05-task-07-proofs.md

## Task 08 - Validate and capture proof artifacts

- Run `sqlc generate` — exits 0; the two new query methods are present.
- Run `go build ./...` from repo root — exits 0.
- Run `go vet ./...` from repo root — exits 0.
- Run `go test ./discord/... -v -run "Portfolio|Custom"` — all Task 05 and Task 06 tests pass.
- Run `go test ./apps/scheduler/... -v -run "Fanout|OnTriggered"` — all Task 07 tests pass.
- Run `go test ./...` from repo root — full suite exits 0 (no regressions in `alert/`, `indicator/`, `expr/`, `calendar/`, `discord/`, or other `apps/`).
- Spot-check in a dev environment:
  - Seed one portfolio row (matching a ticker that will fire an alert soon) and one custom-channel entry (matching a condition that will fire). Point both webhooks at a throwaway Discord channel.
  - Start the Go scheduler, let an alert trigger for that ticker/condition, confirm the embed arrives on both portfolio and custom channels in addition to the economy channel. Confirm only one message arrives per destination per trigger.
- Fill in every `05-proofs/05-task-NN-proofs.md` with actual command output.
- Walk through every acceptance criterion in `05-spec-scheduler-portfolio-custom-fanout.md`; check off the Definition-of-done list in the validation file.

**Proof:** 05-proofs/05-task-08-proofs.md

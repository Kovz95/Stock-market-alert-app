# Validation: Scheduler Portfolio + Custom Channel Fan-out (05)

## Automated verification

From repository root:

```bash
# Regenerate sqlc bindings (required after adding portfolio_queries.sql)
sqlc generate

# Go build — verifies the new resolvers and the updated Common struct compile
go build ./...

# Go vet — catches misuses of the new APIs
go vet ./...

# Targeted resolver tests
go test ./discord/... -v -run "Portfolio|Custom"

# Targeted scheduler fan-out tests
go test ./apps/scheduler/... -v -run "Fanout|OnTriggered"

# Full Go test suite — no regressions anywhere
go test ./...
```

**Expected:**

- `sqlc generate` — no errors. `database/db/` (or whatever path the project uses for generated code) contains typed `ListPortfoliosForFanout` and `ListPortfolioStocks` methods.
- `go build ./...` — exit 0.
- `go vet ./...` — exit 0.
- `go test ./discord/... -run "Portfolio|Custom"` — all nine `PortfolioResolver` tests and all eleven `CustomChannelResolver` tests pass.
- `go test ./apps/scheduler/... -run "Fanout|OnTriggered"` — both Task 07 integration tests pass (standard 3-destination case, dedup case).
- `go test ./...` — exit 0. No regression in `alert/`, `indicator/`, `expr/`, `calendar/`, `apps/grpc/*`, or other modules.

## Structural checks

Run from repo root to prove every required file exists and references the right symbols:

```bash
# sqlc queries present
grep -n "ListPortfoliosForFanout"  database/sql/queries/*.sql
grep -n "ListPortfolioStocks"      database/sql/queries/*.sql

# Resolver source files exist and expose the expected types
ls discord/portfolios.go
ls discord/custom.go
grep -n "type PortfolioResolver"        discord/portfolios.go
grep -n "type CustomChannelResolver"    discord/custom.go
grep -n "func NewPortfolioResolver"     discord/portfolios.go
grep -n "func NewCustomChannelResolver" discord/custom.go
grep -n "DefaultResolverTTL"            discord/

# Scheduler wiring
grep -n "Portfolios"     apps/scheduler/internal/handler/common.go
grep -n "CustomChannels" apps/scheduler/internal/handler/common.go
grep -n "PortfolioResolver"     apps/scheduler/cmd/
grep -n "CustomChannelResolver" apps/scheduler/cmd/

# Dedup closure present in onTriggered
grep -n "seen\s*:=" apps/scheduler/internal/handler/common.go

# Tests present
ls discord/portfolios_test.go
ls discord/custom_test.go
ls apps/scheduler/internal/handler/common_fanout_test.go

# No Python changes
git diff --stat src/services/stock_alert_checker.py src/services/discord_routing.py src/services/portfolio_discord.py
```

**Expected:** every `grep` returns at least one match; every `ls` succeeds. The final `git diff --stat` reports no changes to any of the three Python files.

## Manual checks

End-to-end smoke test (cannot be automated — requires a live scheduler + throwaway Discord channels):

1. Pick (or create) two Discord webhooks: `WEBHOOK_PORTFOLIO` and `WEBHOOK_CUSTOM`. Send a sanity `curl -X POST` to each to confirm they work.
2. Seed one row in `portfolios` with `discord_webhook = WEBHOOK_PORTFOLIO`, `enabled = TRUE`, `name = 'spec05-smoke'`. Seed one row in `portfolio_stocks` for that portfolio with a `ticker` that is known to fire an alert soon on the scheduler (e.g. a liquid ticker that has an active alert configured).
3. Open the spec-03 custom-channel UI and create a custom channel with:
   - Webhook: `WEBHOOK_CUSTOM`.
   - Condition: a simple condition string that matches one of the live alerts (e.g. `close > 100` if you know of an alert using that).
   - Enabled: true.
4. Start the Go scheduler locally (or on the staging host) pointing at the same Postgres.
5. Wait for (or force via the `/scheduler` → Run Exchange Job button) a scheduler evaluation that should trigger the alert.
6. Confirm:
   - The original economy-routed channel receives the alert (existing behavior unchanged).
   - `WEBHOOK_PORTFOLIO` receives the same alert embed.
   - `WEBHOOK_CUSTOM` receives the same alert embed.
   - Each webhook receives **exactly one** copy per trigger, even if any two of the three routes happen to resolve to the same webhook URL.
7. Repeat with a ticker not in any portfolio and a condition that matches no custom channel — only the economy channel should receive the alert (backward compatibility check).
8. Disable the portfolio (`UPDATE portfolios SET enabled = FALSE`) and wait 6 minutes (TTL + buffer). Trigger again. Confirm the portfolio webhook stops receiving alerts.

## Traceability

- Feature spec: `05-spec-scheduler-portfolio-custom-fanout.md`
- Task breakdown: `05-tasks-scheduler-portfolio-custom-fanout.md`
- Questions and decisions: `05-questions-1-scheduler-portfolio-custom-fanout.md`
- Per-task evidence: `05-proofs/05-task-NN-proofs.md` (NN = 01..08)
- Upstream specs: `03-spec-custom-discord-condition-channels/` (defines the `custom_discord_channels` document schema this spec reads)
- Python reference (not modified): `src/services/stock_alert_checker.py:437-476`, `src/services/discord_routing.py:221-323`, `src/services/portfolio_discord.py`
- Parent epic: none

## Definition of done

- [ ] `database/sql/queries/portfolio_queries.sql` contains `ListPortfoliosForFanout` and `ListPortfolioStocks`; `sqlc generate` succeeds.
- [ ] `discord/portfolios.go` defines `PortfolioResolver` with constructor, TTL cache, and uppercase-insensitive ticker matching.
- [ ] `discord/custom.go` defines `CustomChannelResolver` with constructor, TTL cache, the normalization port, and the `price_level` regex.
- [ ] `discord.DefaultResolverTTL` is defined and set to 5 minutes.
- [ ] `Common` struct has `Portfolios` and `CustomChannels` fields; scheduler startup constructs both resolvers with the shared pool and TTL.
- [ ] `onTriggered` builds the embed once, fans out to economy + portfolio + custom-channel URLs, deduplicates via a local `seen map[string]bool`, and calls `Accum.Add` once per unique URL.
- [ ] `PortfolioResolver` unit tests (9) pass.
- [ ] `CustomChannelResolver` unit tests (11) pass.
- [ ] `common_fanout_test.go` integration tests (2) pass.
- [ ] `go build ./...`, `go vet ./...`, and `go test ./...` all exit 0 from repo root.
- [ ] Manual smoke test confirms all three destinations receive exactly one embed per trigger, and backward compatibility holds for tickers/conditions with no fan-out match.
- [ ] Every proof file contains real command output, not placeholders.
- [ ] No changes to any file in `src/services/` (Python is frozen).

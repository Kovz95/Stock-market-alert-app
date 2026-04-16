# Validation: Slow Stochastic Level-Cross Conditions (01)

## Automated verification

From repository root (PowerShell):

```powershell
# 1. Go unit tests — expr package (catalog expansion + evaluator round-trip)
go test ./expr/... -run TestSlowStochLevelCross -v

# 2. Full Go test suite — no regressions in expr or indicator
go test ./expr/... ./indicator/...

# 3. TypeScript type check
cd apps/ui/web
pnpm typecheck

# 4. Next.js production build
pnpm build

# Return to root
cd ../../..
```

**Expected results:**

| Command | Expected |
|---|---|
| `go test ./expr/... -run TestSlowStochLevelCross -v` | `PASS`, all sub-tests listed and green |
| `go test ./expr/... ./indicator/...` | `ok` for every package, exit 0 |
| `pnpm typecheck` | No type errors, exit 0 |
| `pnpm build` | Compiled successfully, exit 0 |

## Structural spot-checks

```powershell
# Confirm 8 new type literals appear in types.ts
Select-String -Path apps/ui/web/app/alerts/add/_components/types.ts `
  -Pattern "slow_stoch_(k|d)_cross_(above|below)_(oversold|overbought)"

# Confirm 8 new entries in SLOW_STOCH_TYPES in ConditionBuilder.tsx
Select-String -Path apps/ui/web/app/alerts/add/_components/ConditionBuilder.tsx `
  -Pattern "slow_stoch_(k|d)_cross_(above|below)_(oversold|overbought)"

# Confirm 8 new case blocks in catalog.go
Select-String -Path expr/catalog.go `
  -Pattern "slow_stoch_(k|d)_cross_(above|below)_(oversold|overbought)"
```

**Expected:** Each command returns exactly 8 matching lines.

## Traceability

- Feature spec: `01-spec-slow-stoch-level-crosses.md`
- Task breakdown: `01-tasks-slow-stoch-level-crosses.md`
- Questions and decisions: `01-questions-1-slow-stoch-level-crosses.md`
- Per-task evidence:
  - `01-proofs/01-task-01-proofs.md` — types.ts changes
  - `01-proofs/01-task-02-proofs.md` — ConditionBuilder.tsx changes
  - `01-proofs/01-task-03-proofs.md` — catalog.go changes
  - `01-proofs/01-task-04-proofs.md` — test output
  - `01-proofs/01-task-05-proofs.md` — full validation run

## Manual checks

1. Open the Add Alert page (`/alerts/add`), set ticker, select category "Slow Stochastic".
2. Confirm the condition type dropdown contains a "Level Crosses" group with 8 new entries.
3. Select "%K crosses above oversold" — verify "Oversold level" and "Overbought level" number inputs appear.
4. Change the oversold level to 25 and confirm the generated condition string in the preview shows `25` in both positions of the compound expression.
5. Switch to "%K oversold (< 20)" — confirm the level inputs are **not** shown.

## Definition of done

- [ ] AC1: `SlowStochConditionType` contains all 8 new literals; `pnpm typecheck` exits 0
- [ ] AC2: `buildConditionExpression` returns correct compound expression strings for all 8 types
- [ ] AC3: All 8 catalog keys expand correctly; `go test ./expr/... -run TestSlowStochLevelCross` exits 0
- [ ] AC4: Evaluator round-trip test passes (cross above oversold → true; cross below oversold → false)
- [ ] AC5: `SLOW_STOCH_TYPES` contains 8 new entries with `group: "Level Crosses"`
- [ ] AC6: Level inputs render conditionally for new cross types only
- [ ] AC7: `go test ./expr/... ./indicator/...` exits 0; `pnpm build` exits 0; existing slow stoch expressions unchanged
- [ ] Proof artifacts contain real command output, not placeholders

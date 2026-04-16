# Tasks: Slow Stochastic Level-Cross Conditions (01)

## Task 01 — Extend `SlowStochConditionType` and expression generation in `types.ts`

- In `apps/ui/web/app/alerts/add/_components/types.ts`, add 8 new string literals to the `SlowStochConditionType` union:
  - `"slow_stoch_k_cross_above_oversold"`
  - `"slow_stoch_k_cross_below_oversold"`
  - `"slow_stoch_k_cross_above_overbought"`
  - `"slow_stoch_k_cross_below_overbought"`
  - `"slow_stoch_d_cross_above_oversold"`
  - `"slow_stoch_d_cross_below_oversold"`
  - `"slow_stoch_d_cross_above_overbought"`
  - `"slow_stoch_d_cross_below_overbought"`
- In `buildConditionExpression`, inside `case "slow_stoch":`, add a case for each new type. Resolve `slowStochOversoldLevel` (default 20) and `slowStochOverboughtLevel` (default 80) from `params`. Example:
  - `slow_stoch_k_cross_above_oversold` → `` `(${kFn}[-1] > ${oversold}) and (${kFn}[-2] <= ${oversold})` ``
  - `slow_stoch_k_cross_below_oversold` → `` `(${kFn}[-1] < ${oversold}) and (${kFn}[-2] >= ${oversold})` ``
  - `slow_stoch_k_cross_above_overbought` → `` `(${kFn}[-1] > ${overbought}) and (${kFn}[-2] <= ${overbought})` ``
  - `slow_stoch_k_cross_below_overbought` → `` `(${kFn}[-1] < ${overbought}) and (${kFn}[-2] >= ${overbought})` ``
  - Mirror for %D variants using `dFn`.

**Proof:** 01-proofs/01-task-01-proofs.md

---

## Task 02 — Add 8 new entries to `SLOW_STOCH_TYPES` and level inputs in `ConditionBuilder.tsx`

- In `apps/ui/web/app/alerts/add/_components/ConditionBuilder.tsx`, append 8 entries to `SLOW_STOCH_TYPES` under `group: "Level Crosses"`:
  ```ts
  { value: "slow_stoch_k_cross_above_oversold",  label: "%K crosses above oversold",  group: "Level Crosses" },
  { value: "slow_stoch_k_cross_below_oversold",  label: "%K crosses below oversold",  group: "Level Crosses" },
  { value: "slow_stoch_k_cross_above_overbought", label: "%K crosses above overbought", group: "Level Crosses" },
  { value: "slow_stoch_k_cross_below_overbought", label: "%K crosses below overbought", group: "Level Crosses" },
  { value: "slow_stoch_d_cross_above_oversold",  label: "%D crosses above oversold",  group: "Level Crosses" },
  { value: "slow_stoch_d_cross_below_oversold",  label: "%D crosses below oversold",  group: "Level Crosses" },
  { value: "slow_stoch_d_cross_above_overbought", label: "%D crosses above overbought", group: "Level Crosses" },
  { value: "slow_stoch_d_cross_below_overbought", label: "%D crosses below overbought", group: "Level Crosses" },
  ```
- Inside the `{category === "slow_stoch" && ...}` render block, add a conditional section that renders two number inputs when the selected type is any new level-cross type:
  - "Oversold level (0–100)" → `params.slowStochOversoldLevel ?? 20`
  - "Overbought level (0–100)" → `params.slowStochOverboughtLevel ?? 80`
- Gate condition: `type.includes("cross_above_oversold") || type.includes("cross_below_oversold") || type.includes("cross_above_overbought") || type.includes("cross_below_overbought")`.

**Proof:** 01-proofs/01-task-02-proofs.md

---

## Task 03 — Add 8 catalog entries to `expr/catalog.go`

- In `expr/catalog.go` `ExpandCatalogCondition`, add 8 new `case` blocks after the existing volume section. Each parses an optional integer level from `valueStr` (defaulting to 20 for oversold keys and 80 for overbought keys). All use `smooth_k=14, smooth_d=3`.
- Example for `slow_stoch_k_cross_above_oversold`:
  ```go
  case "slow_stoch_k_cross_above_oversold":
      level := 20
      if v, err := strconv.Atoi(strings.TrimSpace(valueStr)); err == nil && v >= 0 && v <= 100 {
          level = v
      }
      return fmt.Sprintf(
          "(slow_stoch_k(smooth_k=14, smooth_d=3)[-1] > %d) and (slow_stoch_k(smooth_k=14, smooth_d=3)[-2] <= %d)",
          level, level,
      )
  ```
- Remaining 7 cases follow the same pattern, swapping `k`/`d`, `>/<`, and `20`/`80` defaults.

**Proof:** 01-proofs/01-task-03-proofs.md

---

## Task 04 — Write tests for the new catalog entries and evaluator round-trip

- In `expr/` (new file `catalog_slow_stoch_test.go` or appended to `expr_test.go`), add a test function `TestSlowStochLevelCross`:
  - Table-driven test covering all 8 catalog keys with explicit level values and with empty value strings (defaults).
  - Assert the exact expanded expression string for at least 4 of the 8 variants.
  - Construct a synthetic `indicator.OHLCV` dataset (≥ 30 bars of monotonically increasing prices) where `slow_stoch_k` is guaranteed to have crossed from ≤ 20 to > 20 at the last bar. Call `EvalCondition` with `"slow_stoch_k_cross_above_oversold: 20"` and assert `true`.
  - Call `EvalCondition` with `"slow_stoch_k_cross_below_oversold: 20"` on the same dataset and assert `false`.
- `go test ./expr/... -run TestSlowStochLevelCross` must exit 0.

**Proof:** 01-proofs/01-task-04-proofs.md

---

## Task 05 — Validate and capture proof artifacts

- Run `go test ./expr/... ./indicator/...` and capture output.
- Run `pnpm typecheck` in `apps/ui/web` and capture output.
- Run `pnpm build` in `apps/ui/web` and capture output.
- Confirm all acceptance criteria in the spec pass.
- Fill in all proof files with real command output.

**Proof:** 01-proofs/01-task-05-proofs.md

# Spec: Slow Stochastic Level-Cross Conditions (01)

## Goal

The slow stochastic condition builder currently supports static level checks (%K oversold, %K overbought, custom comparison) and a K/D line crossover. Users want to detect the *moment* the %K or %D line crosses above or below the oversold (20) or overbought (80) thresholds — a one-bar event rather than a persistent level check. This is a standard stochastic trading signal (e.g. "exit oversold") missing from the current builder.

## Scope

### In scope

- 8 new `SlowStochConditionType` string literals in `types.ts`
- Expression generation for all 8 types in `buildConditionExpression` (`types.ts`)
- 8 new entries in `SLOW_STOCH_TYPES` array (`ConditionBuilder.tsx`), in a new **"Level Crosses"** group
- Two optional UI level params rendered when any new cross type is selected:
  - `slowStochOversoldLevel` (default 20, range 0–100)
  - `slowStochOverboughtLevel` (default 80, range 0–100)
- 8 new catalog keys in `expr/catalog.go` `ExpandCatalogCondition`, using default smooth_k=14, smooth_d=3; value field accepts the threshold level as an integer
- Tests for the new catalog expansions in `expr/expr_test.go` (or a dedicated `catalog_test.go`)

### Out of scope

- New indicator functions in `indicator/` (not required; crossovers are compound expressions)
- Changes to the `alert.proto`, gRPC services, or database schema
- Multi-bar or percentage-of-range crossing variants
- Applying level-cross patterns to `stoch_k`/`stoch_d` (fast stochastic) or `stoch_rsi_k`/`stoch_rsi_d` — those are separate features

## Acceptance criteria

1. **Type system**
   - `types.ts` `SlowStochConditionType` includes exactly these 8 new literals, and TypeScript compilation (`pnpm typecheck` in `apps/ui/web`) exits 0:
     - `slow_stoch_k_cross_above_oversold`
     - `slow_stoch_k_cross_below_oversold`
     - `slow_stoch_k_cross_above_overbought`
     - `slow_stoch_k_cross_below_overbought`
     - `slow_stoch_d_cross_above_oversold`
     - `slow_stoch_d_cross_below_oversold`
     - `slow_stoch_d_cross_above_overbought`
     - `slow_stoch_d_cross_below_overbought`

2. **Expression generation**
   - For each new type, `buildConditionExpression` returns the correct compound expression string. Example with smooth_k=14, smooth_d=3 and oversold=20:
     - `slow_stoch_k_cross_above_oversold` → `(slow_stoch_k(smooth_k=14, smooth_d=3)[-1] > 20) and (slow_stoch_k(smooth_k=14, smooth_d=3)[-2] <= 20)`
     - `slow_stoch_k_cross_below_oversold` → `(slow_stoch_k(smooth_k=14, smooth_d=3)[-1] < 20) and (slow_stoch_k(smooth_k=14, smooth_d=3)[-2] >= 20)`
     - `slow_stoch_k_cross_above_overbought` → `(slow_stoch_k(smooth_k=14, smooth_d=3)[-1] > 80) and (slow_stoch_k(smooth_k=14, smooth_d=3)[-2] <= 80)`
     - `slow_stoch_k_cross_below_overbought` → `(slow_stoch_k(smooth_k=14, smooth_d=3)[-1] < 80) and (slow_stoch_k(smooth_k=14, smooth_d=3)[-2] >= 80)`
     - Symmetric %D variants use `slow_stoch_d(...)`.

3. **Catalog expansion**
   - `ExpandCatalogCondition("slow_stoch_k_cross_above_oversold: 20")` returns `(slow_stoch_k(smooth_k=14, smooth_d=3)[-1] > 20) and (slow_stoch_k(smooth_k=14, smooth_d=3)[-2] <= 20)`.
   - `ExpandCatalogCondition("slow_stoch_k_cross_above_oversold")` (no value) returns the same string using the default level of 20.
   - All 8 catalog keys expand correctly, verified by `go test ./expr/... -run TestSlowStochLevelCross` exiting 0.

4. **Evaluator round-trip**
   - A synthesised OHLCV dataset where slow_stoch_k crosses from below 20 to above 20 at the last bar evaluates `EvalCondition(data, "slow_stoch_k_cross_above_oversold: 20", nil)` → `true`.
   - The same dataset with the condition inverted (`slow_stoch_k_cross_below_oversold: 20`) evaluates → `false`.
   - Verified by `go test ./expr/... -run TestSlowStochLevelCross` exiting 0.

5. **UI — condition type list**
   - `SLOW_STOCH_TYPES` in `ConditionBuilder.tsx` contains exactly 8 new entries, all with `group: "Level Crosses"`.
   - Labels follow the pattern: `"%K crosses above oversold"`, `"%K crosses below oversold"`, `"%K crosses above overbought"`, `"%K crosses below overbought"` (and equivalent %D variants).

6. **UI — level parameters**
   - When any of the 8 new types is selected, two number inputs are rendered: "Oversold level (0–100)" and "Overbought level (0–100)".
   - Defaults are 20 and 80 respectively.
   - When a non-cross type is selected (e.g. `slow_stoch_k_oversold`), these inputs are **not** rendered. Verified by visual inspection.

7. **No regressions**
   - `go test ./expr/... ./indicator/...` exits 0.
   - `pnpm typecheck` in `apps/ui/web` exits 0.
   - `pnpm build` in `apps/ui/web` exits 0.
   - Existing slow stoch condition types (`slow_stoch_k_oversold`, `slow_stoch_k_cross_above_d`, etc.) produce unchanged expression strings.

## Conventions

- Cross direction definition: "cross above level L" means `line[-1] > L AND line[-2] <= L`. "Cross below level L" means `line[-1] < L AND line[-2] >= L`. This is identical to the existing `price_cross_above_ma` pattern in `catalog.go` and `slow_stoch_k_cross_above_d` in `types.ts`.
- The catalog entries use hardcoded smooth_k=14 and smooth_d=3 (the same defaults as all other slow stoch catalog calls). Period customisation is reserved for the UI expression builder path via `slowStochSmoothK` / `slowStochSmoothD` params.
- New UI params `slowStochOversoldLevel` and `slowStochOverboughtLevel` are stored in `ConditionParams` only; they are not surfaced as separate catalog value fields.
- TypeScript type union must remain in `types.ts`; no duplication into separate files.
- `ConditionBuilder.tsx` section guarded by `category === "slow_stoch"` must not re-render the level inputs for non-cross types — gate on `type.includes("cross_above_oversold") || type.includes("cross_below_oversold") || type.includes("cross_above_overbought") || type.includes("cross_below_overbought")`.

# Questions & Decisions: Slow Stochastic Level-Cross Conditions (01)

## Resolved

| # | Question | Decision |
|---|---|---|
| 1 | Apply to %K only, or both %K and %D? | Both. The existing oversold/overbought level types already have %K and %D variants. Four cross types × two lines = **8 new condition types** total. |
| 2 | Are the oversold (20) and overbought (80) levels configurable per-condition or hardcoded? | Configurable via UI params `slowStochOversoldLevel` (default 20, 0–100) and `slowStochOverboughtLevel` (default 80, 0–100), matching the style of `slowStochLevel` used by the existing custom comparison types. |
| 3 | Cross definition — one-bar or multi-bar? | One-bar strict cross: `line[-1]` has moved past the level, `line[-2]` has not. Identical to the existing `slow_stoch_k_cross_above_d` / `slow_stoch_k_cross_below_d` implementation. |
| 4 | Cross above oversold (20) — is this entry into oversold or exit? | **Exit** oversold (bullish). K/D was ≤ 20 last bar and is now > 20. Cross *below* oversold is *entry* (bearish/deepening). |
| 5 | Do `expr/catalog.go` entries need to be added? | Yes. Catalog entries are required so these conditions work when stored and evaluated as string-based conditions (not just from the UI expression builder). Defaults: smooth_k=14, smooth_d=3. Value field accepts the level integer. |
| 6 | Does `indicator/` need new functions? | No. `slow_stoch_k` and `slow_stoch_d` are already registered. The level-cross logic is compound expressions composed in `expr/catalog.go` and `types.ts`, exactly as `price_cross_above_ma` works (no new indicator function). |
| 7 | Do the existing `slow_stoch_k_compare` / `slow_stoch_d_compare` types cover the new use cases? | Partially (static level threshold), but they do not detect a *crossing* event. The new types are distinct. |
| 8 | Where does the UI render the new level inputs? | Inside the existing `{category === "slow_stoch" && ...}` block in `ConditionBuilder.tsx`, gated on `type` being one of the new cross types. Two new number inputs: "Oversold level" and "Overbought level". |

## Open

None.

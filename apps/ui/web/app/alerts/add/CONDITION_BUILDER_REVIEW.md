# Condition Builder Review: Web vs Streamlit vs Backend

This document compares the web Add Alert condition builder with Streamlit (`Add_Alert.py`) and the backend evaluators (Go `expr` + `alert` packages).

---

## Summary

**The web condition builder works the same as Streamlit** for the conditions it supports. The Go alert evaluator supports both **catalog form** (e.g. `price_above: 150`) and **expression form** (e.g. `Close[-1] > 150`, `rsi(14)[-1] < 30`). The web emits the same catalog/expression strings that Streamlit uses, and the Go `expr` package expands catalog forms before evaluation.

One **storage-format fix** was applied: the Go `ExtractConditions` function now accepts the **condition_1 object format** (`{"condition_1": {"conditions": [...], "combination_logic": "AND"}}`) that the web (and Streamlit when saving via API) send, so alerts created from the web are evaluated correctly.

---

## 1. Condition string format

### Streamlit (`Add_Alert.py`)

- Builds condition strings and appends them to `st.session_state.all_conditions`.
- Examples:
  - Price: `price_above: 100.0`, `price_below: 200`, `price_equals: 125.5`
  - MA: `price_above_ma: 20 (SMA)`, `price_below_ma: 50 (EMA)`, `ma_crossover: 10 > 20 (SMA)`
  - RSI: `rsi(14)[-1] < 30`, `rsi(14)[-1] > 70`
  - MACD: built as expression or named (e.g. `macd_bullish_crossover`)
  - Bollinger: `price_above_upper_band`, `price_below_lower_band`
  - Volume: `volume_above_average: 1.5x`, `volume_spike: 2x`
- On save, conditions are wrapped as:  
  `{"condition_1": {"conditions": entry_conditions_list, "combination_logic": "AND"}}`  
  and normalized to a list of `{"index": i, "conditions": string}` for storage (Streamlit `_conditions_to_storage_format`).

### Web (`types.ts` → `conditionEntryToExpression`)

| Category        | Web output example                          | Streamlit equivalent     | Backend (Go) |
|----------------|----------------------------------------------|---------------------------|--------------|
| Price          | `price_above: 150`, `price_below: 100`, `price_equals: 125.5` | Same                     | Catalog expanded to `close[-1] > 150` etc. |
| Moving average | `price_above_ma: 20 (SMA)`, `price_below_ma: 50 (EMA)`, `ma_crossover: 10 > 20` | Same (MA type optional)  | Catalog expanded; `ma_crossover` defaults to SMA |
| RSI            | `rsi(14)[-1] < 30`, `rsi(14)[-1] > 70`, `rsi(14)[-1] > 50` | Same expression form     | Evaluated as expression (no catalog) |
| MACD           | `macd_bullish_crossover`, `macd_bearish_crossover`, `macd_histogram_positive` | Same named form          | Catalog expanded (12,26,9) |
| Bollinger      | `price_above_upper_band`, `price_below_lower_band` | Same                     | Catalog expanded (20, 2) |
| Volume         | `volume_above_average: 1.5x`, `volume_spike: 1.5x` | Same                     | Catalog expanded (period 20) |
| Custom         | User expression passed through              | Same                     | Must be valid expression form |

So the web produces the **same** catalog and expression strings that Streamlit uses and that the backend supports.

---

## 2. Backend evaluation

### Go (`expr` package)

- **Catalog form**: `ExpandCatalogCondition()` turns UI-style strings into expressions, e.g.  
  `price_above: 150` → `close[-1] > 150`,  
  `price_above_ma: 20 (SMA)` → `close[-1] > sma(20)[-1]`,  
  `volume_above_average: 1.5x` → `volume_ratio(20)[-1] > 1.5`.
- **Expression form**: Strings that are not catalog form (e.g. `rsi(14)[-1] < 30`) are evaluated as-is.
- Supported catalog keys: `price_above`, `price_below`, `price_equals`, `price_above_ma`, `price_below_ma`, `ma_crossover`, `ma_crossunder`, `rsi_oversold`, `rsi_overbought`, `macd_bullish_crossover`, `macd_bearish_crossover`, `macd_histogram_positive`, `price_above_upper_band`, `price_below_lower_band`, `volume_above_average`, `volume_spike`, `volume_below_average`.

### Python (`src/services/backend.py`)

- Evaluates only **expression form** (e.g. `Close[-1] > 150`, `rsi(14)[-1] < 30`) via `simplify_conditions` and `apply_function`.
- It does **not** expand catalog form. Alerts created in the web/Streamlit with catalog strings are evaluated by the **Go** checker (scheduler / EvaluateExchange), not by the Python stock_alert_checker in the path that uses this backend.

So for alerts created in the web and run through the normal Go-based scheduler/evaluator, behavior is correct.

---

## 3. Conditions payload shape and storage

### Web send

- `buildConditionsStruct()` returns:  
  `{ condition_1: { conditions: string[], combination_logic: "AND" | "OR" } }`
- This object is sent as the `conditions` field of the Create Alert request and stored in the DB as JSON.

### Go `ExtractConditions` (before fix)

- Only accepted:
  - JSON array of strings: `["..."]`
  - JSON array of objects: `[{"conditions": "..."}]`
  - Nested array: `[["..."]]`
  - Single string: `"..."`
- It did **not** accept an object like `{"condition_1": {"conditions": [...], "combination_logic": "AND"}}`, so alerts created from the web could not be evaluated.

### Fix applied

- In `alert/evaluator.go`, `ExtractConditions` now also handles a **top-level JSON object** (e.g. `{"condition_1": { "conditions": ["..."], "combination_logic": "AND" }}`).
- It iterates over each key’s value, and for each value that is an object with a `conditions` key, it flattens the condition list (supporting both `conditions` as array of strings and as single string).
- So the web (and any client that sends the condition_1 shape) now works with the Go checker. A test was added in `alert/evaluator_test.go` for this format.

---

## 4. Combination logic

- **Streamlit**: `st.session_state.condition_logic` ("AND" or "OR") → stored with the alert.
- **Web**: `combinationLogic` "AND" | "OR" in form state and in `condition_1.combination_logic`; also sent and stored in the `combination_logic` column.
- **Go**: `EvaluateAlert` uses the alert’s `combination_logic` and `EvalConditionList` to combine results (AND/OR and complex expressions like "1 AND (2 OR 3)").

So combination logic is aligned between web and backend.

---

## 5. Gaps and differences (optional follow-ups)

| Item | Streamlit | Web | Note |
|------|-----------|-----|------|
| MA types for crossover | SMA, EMA, HMA, FRAMA, KAMA | Only SMA, EMA in builder | Web could add HMA (and optionally FRAMA/KAMA) to match. |
| RSI “rsi_level” | RSI Value / custom level | Supported as `rsi(period)[-1] <level>` etc. | Same behavior via expression. |
| Custom expression | Free text | Single custom expression field | Same idea; both require valid expression form for the evaluator. |
| Multi-timeframe / mixed | Checkboxes and docs | Not in web yet | Documented as optional in PARITY_PLAN.md. |
| More indicators (ATR, CCI, etc.) | Many in Streamlit | Not in web builder | Could add later; backend supports many via expression form. |

None of these affect the conditions that the web builder currently supports; they work the same as in Streamlit for that subset.

---

## 6. Conclusion

- The **web condition builder matches Streamlit** for the condition types it implements (price, MA, RSI, MACD, Bollinger, volume, custom).
- The **Go evaluator** supports both catalog and expression forms used by the web and Streamlit.
- **Storage format** is now supported: Go `ExtractConditions` accepts the `condition_1` object format, so web-created alerts are evaluated correctly by the scheduler/EvaluateExchange path.

No changes to the web condition builder were required for parity; the only change was in the Go alert package to accept the condition_1 conditions format.

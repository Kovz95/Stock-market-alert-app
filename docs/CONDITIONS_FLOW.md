# How conditions are stored, retrieved, parsed, and evaluated

This document describes the full pipeline for alert conditions: from the Next.js UI through storage to evaluation in Go.

---

## 1. Adding a condition in the UI (Next.js)

- **Where:** `apps/ui/web/app/alerts/add/` — Add Alert flow.
- **Components:**
  - **ConditionBuilder** — User picks **Category** (e.g. "MA Slope + Curvature") and **Condition type** (e.g. "Slope turn up"), then fills **params** (MA length, slope lookback, etc.).
  - **types.ts** — Defines `ConditionCategory`, condition types, `ConditionParams`, and **`conditionEntryToExpression(entry)`**.
- **Result:** Each added condition is a **ConditionEntry** `{ id, category, type, params }`. When the user saves the alert, entries are converted to **condition strings** (e.g. `ma_slope_curve_turn_up(ma_len=200, slope_lookback=3, ma_type='HMA')[-1] == 1`) and sent to the API.

---

## 2. Sending conditions to the API (Next.js → gRPC)

- **Where:** `apps/ui/web/app/alerts/add/_components/AddAlertForm.tsx`, `apps/ui/web/actions/alert-actions.ts`.
- **Flow:**
  1. Form state holds `conditions: ConditionEntry[]` and `combinationLogic: "AND" | "OR"`.
  2. On submit, **`buildConditionsStruct(entries, combinationLogic)`** (in `types.ts`) is called:
     - Each entry is turned into a string via **`conditionEntryToExpression(entry)`**.
     - Result shape: `{ condition_1: { conditions: string[], combination_logic: "AND"|"OR" } }`.
  3. That object is sent as the **conditions** field of **CreateAlertRequest** (gRPC) to the backend.
- **Proto:** `proto/alert/v1/alert.proto` — `CreateAlertRequest.conditions` is `google.protobuf.Struct` (JSON-like key/value).

---

## 3. Storage (database)

- **Where:** Alert service (Go gRPC server) persists the alert; PostgreSQL stores the alert row.
- **Schema:** The **conditions** column holds the **conditions struct** as JSON. Typical stored value:
  ```json
  {
    "condition_1": {
      "conditions": [
        "ma_slope_curve_turn_up(ma_len=200, slope_lookback=3, ma_type='HMA')[-1] == 1",
        "rsi(14)[-1] < 30"
      ],
      "combination_logic": "AND"
    }
  }
  ```
- **combination_logic** is also stored on the alert row (e.g. `combination_logic` column) so the evaluator knows how to combine the condition results (AND / OR / "1 AND (2 OR 3)").

---

## 4. Retrieval (when evaluating alerts)

- **Where:** Alert checker / EvaluateExchange flow loads alerts from the database.
- **Flow:**
  1. Alerts for the exchange/timeframe are queried; each row includes **conditions** (JSON bytes) and **combination_logic**.
  2. For each alert, the evaluator receives:
     - **conditions** — raw JSON (the struct above).
     - **combination_logic** — string (e.g. `"AND"`, `"OR"`).
     - **OHLCV** — price data for the alert’s ticker/timeframe.
     - **ticker** — for context (e.g. pivot_sr).

---

## 5. Parsing conditions (Go)

- **Where:** `alert/evaluator.go`, `expr/parser.go`.
- **Flow:**
  1. **`alert.ExtractConditions(conditions []byte)`** turns the stored JSON into a **flat list of condition strings**:
     - If the JSON is `{ "condition_1": { "conditions": ["expr1", "expr2"], "combination_logic": "AND" } }`, it extracts `["expr1", "expr2"]`.
     - Also supports array of strings, array of `{ "conditions": "..." }`, etc.
  2. Each condition string (e.g. `ma_slope_curve_turn_up(ma_len=200)[-1] == 1`) is parsed by **`expr.ParseCondition(cond)`**:
     - **splitOnOperator** finds the comparison operator (`==`, `>`, etc.) and splits into left and right.
     - **ParseOperand** on each side:
       - **Left:** `ma_slope_curve_turn_up(ma_len=200)[-1]` → indicator name `ma_slope_curve_turn_up`, params `{ "ma_len": 200 }`, specifier `-1`.
       - **Right:** `1` → numeric literal.
     - **remapPositionalParams** (in `parser.go`) maps generic names to indicator-specific ones (e.g. `period` → `ma_len` for `ma_slope_curve_*`).
  3. Result is a **Comparison** `{ Left, Op, Right }` with operands that can be numeric or indicator references (name + params + specifier).

---

## 6. Evaluation (Go)

- **Where:** `expr/evaluator.go`, `indicator/registry.go`.
- **Flow:**
  1. **`expr.Evaluator.EvalConditionList(data, conditions, combinationLogic, ctx)`** evaluates each condition string:
     - **EvalCondition** for each string: optionally **ExpandCatalogCondition** (for catalog forms like `price_above: 150`), then **ParseCondition** and **evalComparison**.
  2. For each **evalComparison**:
     - **resolveOperand** for left and right. For an indicator operand:
       - **computeSeries**: `e.registry.Get(name)` → get the indicator function (e.g. `MaSlopeCurveTurnUp`).
       - Call `fn(data, params)` → get the full series (e.g. 0/1 for each bar).
       - Apply **specifier** (e.g. `-1`) to get the scalar at the last bar.
     - **compareValues** (e.g. `val1 == val2` for `==`).
  3. Combine per-condition boolean results using **combination_logic** (AND / OR / "1 AND (2 OR 3)").
  4. Return **true** if the combined result is true → alert triggers.

---

## 7. MA Slope + Curvature in this pipeline

| Step | How MA Slope + Curvature is handled |
|------|-------------------------------------|
| **UI** | Category "MA Slope + Curvature" with types (slope_positive, slope_turn_up, bend_up, etc.) and params (ma_len, ma_type, slope_lookback, …). |
| **Expression** | `conditionEntryToExpression` emits strings like `ma_slope_curve_slope(ma_len=200, ...)[-1] > 0` or `ma_slope_curve_turn_up(ma_len=200)[-1] == 1`. |
| **Storage** | Same as any condition: stored in the **conditions** JSON struct. |
| **Retrieval** | Same: **ExtractConditions** returns the flat list including these strings. |
| **Parse** | **ParseCondition** / **ParseOperand** recognize indicator names like `ma_slope_curve_turn_up`, `ma_slope_curve_slope`, and parse named params (`ma_len`, `ma_type`, etc.). **remapPositionalParams** maps `period` → `ma_len` for `ma_slope_curve*`. |
| **Evaluate** | **registry.Get("ma_slope_curve_turn_up")** etc. returns the Go indicator function; it is called with OHLCV and params; the returned series is indexed by specifier `-1` and compared (e.g. to 1). |

---

## 8. Summary

- **Stored:** Conditions are stored as a **JSON struct** in the alert row (`condition_1: { conditions: string[], combination_logic }`).
- **Retrieved:** The evaluator loads the alert and passes the **conditions** bytes and **combination_logic** to the expression evaluator.
- **Parsed:** **ExtractConditions** → list of strings; **ParseCondition** → list of **Comparison** (left/op/right with indicator name + params + specifier).
- **Evaluated:** For each comparison, indicator **name** is looked up in the **indicator Registry**; the function is run on OHLCV with **params**; the **specifier** is applied to get a scalar; left and right are compared; results are combined with **combination_logic**.

Adding a new condition type (like MA Slope + Curvature) requires:
1. **Go:** Implement and register the indicator(s) in `indicator/` and, if needed, param remapping in `expr/parser.go`.
2. **Next.js:** Add category and types in `types.ts`, implement **conditionEntryToExpression**, and add the UI in **ConditionBuilder**.

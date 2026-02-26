# Add Alert Page – TypeScript Component Build Plan

This document outlines how to build out the Next.js Add Alert page to parity with the Streamlit `pages/Add_Alert.py` flow, using TypeScript components and (optionally) a bulk-create server action.

---

## 1. Current State

- **Route**: `/alerts/add` (`apps/ui/web/app/alerts/add/page.tsx`)
- **Server action**: `createAlert` in `apps/ui/web/actions/alert-actions.ts` (single alert)
- **Hook**: `useCreateAlert()` in `apps/ui/web/lib/hooks/useAlerts.ts`
- **Proto**: `CreateAlertRequest` in `gen/ts/alert/v1/alert.ts` (name, stockName, ticker, ticker1, ticker2, conditions, combinationLogic, action, timeframe, exchange, country, ratio, isRatio, adjustmentMethod, plus optional structs: dtpParams, multiTimeframeParams, mixedTimeframeParams, rawPayload)

The current add page is a minimal form: name (optional), stock name, ticker, action, timeframe, exchange, country. It does not yet support ratio alerts, conditions, multi-timeframe, or bulk create.

---

## 2. Streamlit Add_Alert.py – Features to Port

| Feature | Description | Priority |
|--------|-------------|----------|
| **Alert name** | Optional; auto-generated from stock + conditions if blank | Done |
| **Action** | Buy / Sell | Done |
| **Ratio toggle** | No / Yes – single ticker vs ratio of two assets | High |
| **Single-stock flow** | Ticker selection (with filters, “Apply to all”) | High |
| **Ratio flow** | First exchange + stock, second exchange + stock, adjustment method | High |
| **Timeframe** | 1D, 1W, 1M | Done |
| **Multi-timeframe** | Optional; primary + comparison timeframe | Medium |
| **Mixed timeframe** | Optional; combined daily/weekly conditions | Medium |
| **Conditions** | Category → type → params (price, MA, RSI, MACD, BB, volume, etc.) | High |
| **Condition combination** | AND / OR for multiple conditions | High |
| **Bulk create** | Same conditions applied to multiple selected stocks | High |

---

## 3. Recommended Component Structure

Use a single add-alert page that composes smaller components. Suggested layout:

```
app/alerts/add/
  page.tsx                    # Route; composes form and handles submit
  _components/
    AddAlertForm.tsx          # Form wrapper, state, submit handler
    AlertBasicFields.tsx      # Name, action, timeframe, exchange, country
    AlertTickerSection.tsx    # Ratio toggle + single ticker OR ratio pair
    AlertConditionsSection.tsx # Condition list + combination logic (AND/OR)
    ConditionBuilder.tsx      # One condition: category → type → params
    ConditionRow.tsx          # Display one condition + remove
    IndicatorGuide.tsx        # Collapsible guide (optional)
```

### 3.1 Page (`page.tsx`)

- Renders heading and `AddAlertForm`.
- Can pass `searchParams` for prefilling (e.g. `?ticker=AAPL`) later.

### 3.2 AddAlertForm

- Holds form state (name, action, ratio mode, tickers, conditions array, combinationLogic, timeframe, exchange, country, etc.).
- Validates before submit (e.g. at least one condition for full parity; for MVP you may allow no conditions if backend supports it).
- On submit:
  - **Single alert**: call existing `createAlert` with one payload.
  - **Bulk**: call new `createAlertsBulk` (or loop `createAlert`) with an array of tickers + shared condition payload.
- Uses `useCreateAlert()` for single, and a new `useCreateAlertsBulk()` (or a wrapper that calls `createAlert` in a loop) for bulk.
- Renders: `AlertBasicFields`, `AlertTickerSection`, `AlertConditionsSection`, and Submit/Cancel.

### 3.3 AlertBasicFields

- Props: `name`, `onNameChange`, `action`, `onActionChange`, `timeframe`, `onTimeframeChange`, `exchange`, `onExchangeChange`, `country`, `onCountryChange`.
- Uses existing UI: `Field`, `Input`, `Select` (action, timeframe, exchange, country).

### 3.4 AlertTickerSection

- **Ratio mode**: `isRatio: boolean` (No/Yes).
- **When single**: one ticker input (and optionally a future stock picker with filters / “Apply to all”).
- **When ratio**: two ticker/name rows (ticker1 + exchange, ticker2 + exchange), plus optional `adjustmentMethod` select.
- Output: either `{ single: { ticker, stockName?, exchange, country? } }` or `{ ratio: { ticker1, ticker2, stockName1?, stockName2?, exchange?, country?, adjustmentMethod? } }`.
- Can be split later into `SingleTickerField.tsx` and `RatioTickerFields.tsx` if it grows.

### 3.5 AlertConditionsSection

- State: `conditions: ConditionEntry[]`, `combinationLogic: 'AND' | 'OR'`.
- Renders:
  - Select for combination logic (AND/OR).
  - List of `ConditionRow` (each shows one condition and a remove button).
  - “Add condition” button that opens/adds a `ConditionBuilder`.
- Each `ConditionEntry` could be: `{ id: string, category: string, type: string, params: Record<string, unknown> }` or a backend-ready shape (e.g. condition string or small struct).

### 3.6 ConditionBuilder

- Category select (e.g. Price, Moving Averages, RSI, MACD, Bollinger, Volume, etc.).
- Type/condition select and param inputs depend on category (number inputs, selects).
- On confirm, add one entry to `conditions` in parent state.
- Can mirror Streamlit’s structure: one component with internal category → type → params layout, or one component per category for clarity.

### 3.7 ConditionRow

- Displays a human-readable summary of one condition and a remove button; keyed by `id`.

### 3.8 IndicatorGuide (optional)

- Collapsible section (e.g. `Collapsible` or accordion) with the same indicator/condition syntax guide as in Streamlit, for reference.

---

## 4. Server Actions

### 4.1 Keep existing `createAlert`

- Signature remains: one `CreateAlertRequest`-like object (minus optional structs if you still omit them).
- Use for single-alert create and for each item in bulk when no dedicated bulk RPC exists.

### 4.2 Add bulk create (if desired)

**Option A – Front-end loop (no backend change)**  
- New server action, e.g. `createAlertsBulk(data: { alerts: CreateAlertInput[] })`, that loops and calls the existing gRPC `createAlert` for each item.
- Return type: `{ created: number, failed: number, errors: string[] }` (and optionally `skippedDuplicates` if you add duplicate detection).
- Pros: no proto/backend change. Cons: N round-trips (or N from Next to gRPC).

**Option B – New gRPC BulkCreateAlerts**  
- Add `BulkCreateAlertsRequest` / `BulkCreateAlertsResponse` in proto; backend implements a batch insert.
- Single round-trip; better for “Apply to all” with many tickers.
- Implement in `alert-actions.ts`: e.g. `createAlertsBulk(request)` calling the new RPC.

Recommendation: start with **Option A** in the server action; add **Option B** later if you need performance.

---

## 5. Data Flow and Types

### 5.1 Form state type (example)

```ts
type AddAlertFormState = {
  name: string;
  action: "Buy" | "Sell";
  isRatio: boolean;
  // Single
  ticker: string;
  stockName: string;
  exchange: string;
  country: string;
  // Ratio
  ticker1: string;
  ticker2: string;
  stockName1?: string;
  stockName2?: string;
  adjustmentMethod: string;
  // Conditions
  conditions: ConditionEntry[];
  combinationLogic: "AND" | "OR";
  // Time
  timeframe: string;
  // Optional advanced
  multiTimeframe?: { primary: string; comparison: string };
  mixedTimeframe?: boolean;
};
```

### 5.2 Mapping to CreateAlertRequest

- `name`, `stockName`, `ticker`, `ticker1`, `ticker2`, `action`, `timeframe`, `exchange`, `country`, `ratio` (e.g. `isRatio ? "Yes" : "No"`), `isRatio`, `adjustmentMethod`, `combinationLogic` map straight.
- `conditions`: build the `conditions` struct (or backend-defined shape) from `conditions[]` and `combinationLogic` before calling `createAlert` / bulk.

---

## 6. Implementation Order

1. **Phase 1 (current)**  
   - Add-alert page and sidebar link.  
   - Basic fields only (name, ticker, stock name, action, timeframe, exchange, country).  
   - Single `createAlert` only.

2. **Phase 2 – Ratio and ticker**  
   - Add `AlertTickerSection` with ratio toggle.  
   - Single ticker input and ratio pair (ticker1/ticker2 + optional adjustment method).  
   - Submit single alert for either mode.

3. **Phase 3 – Conditions**  
   - Add `AlertConditionsSection`, `ConditionBuilder`, `ConditionRow`.  
   - At least one category (e.g. price above/below) to prove the pipeline.  
   - Map conditions + combinationLogic to `CreateAlertRequest.conditions`.

4. **Phase 4 – More condition types**  
   - Add more categories/types (MA, RSI, MACD, etc.) in `ConditionBuilder`.  
   - Optionally add multi-timeframe and mixed-timeframe fields.

5. **Phase 5 – Bulk create**  
   - Add “Apply to multiple tickers” (e.g. multi-select or “Apply to all” from a list).  
   - Implement `createAlertsBulk` server action (loop over `createAlert` or new RPC).  
   - Progress/feedback (toast or inline) for created/skipped/failed.

6. **Phase 6 – Polish**  
   - Stock picker with filters (if you have a symbol list or API).  
   - Indicator guide component.  
   - Validation messages and accessibility.

---

## 7. Files to Touch

| Area | Files |
|------|--------|
| Page | `apps/ui/web/app/alerts/add/page.tsx` |
| Form components | `apps/ui/web/app/alerts/add/_components/*.tsx` |
| Server actions | `apps/ui/web/actions/alert-actions.ts` (add `createAlertsBulk` when needed) |
| Hooks | `apps/ui/web/lib/hooks/useAlerts.ts` (add `useCreateAlertsBulk` when needed) |
| Types | Shared types in `apps/ui/web/app/alerts/add/_components/types.ts` or in a shared `types/` dir |

---

## 8. Summary

- **Add-alert page**: Implemented at `/alerts/add` with basic fields and single `createAlert`.
- **Plan**: Build out with `AddAlertForm`, `AlertBasicFields`, `AlertTickerSection`, `AlertConditionsSection`, `ConditionBuilder`, and `ConditionRow`; then add bulk create via a new server action (and optionally a gRPC bulk RPC later).
- **Bulk create**: Start by adding a `createAlertsBulk` server action that loops `createAlert`; extend with a dedicated bulk RPC if required for scale.

# MA Slope + Curvature Indicator – Implementation Plan

This plan covers adding the **MA Slope + Curvature (MTF, HMA/EMA/SMA)** logic from your Pine Script v6 indicator into (1) the **Go `indicator/` package** and (2) the **Streamlit UI condition builder** (Scanner), so you can build conditions like “slope turns up”, “curvature above threshold”, etc.

---

## 1. Overview of the Pine Logic

The Pine script:

- **MA**: HMA, EMA, SMA, WMA, or RMA of `source` (default Close) with configurable length and optional higher timeframe.
- **Slope**: `(MA - MA[slope_lookback]) / slope_lookback`, then optionally smoothed (None / EMA / SMA / RMA).
- **Curvature**: change of smoothed slope (slope - slope[1]), then optionally smoothed.
- **Normalization**: None, ATR-based, or percent-of-MA.
- **Signals** (one-bar pulses): early bend up/down, bend up/down (curvature cross), turn up/down (slope cross).

For your app we **ignore multi-timeframe (MTF)** in the first version: everything runs on the chart/eval timeframe. MTF can be added later if you add timeframe resolution to your data pipeline.

---

## 2. Go `indicator/` Package

### 2.1 File and naming

- Add **`indicator/ma_slope_curve.go`**.
- Use a single internal helper that computes MA, slope, curvature, and (optionally) signal booleans, then expose **multiple registry entries** that return one series each (same pattern as MACD with line/signal/histogram).

### 2.2 Parameters (from Pine, single-TF)

Map Pine inputs to one shared param set used by all MA Slope + Curvature outputs:

| Pine input       | Go param name   | Type   | Default | Notes |
|------------------|-----------------|--------|---------|--------|
| Source           | `input`         | string | "Close" | OHLCV column |
| MA Type          | `ma_type`       | string | "HMA"   | HMA, EMA, SMA, WMA, RMA |
| MA Length        | `ma_len` / `timeperiod` | int | 200 | |
| Slope lookback   | `slope_lookback`| int    | 3       | Bars for slope |
| Smoothing        | `smooth_type`   | string | "EMA"   | None, EMA, SMA, RMA |
| Smooth length    | `smooth_len`    | int    | 2       | |
| Normalize by     | `norm_mode`     | string | "ATR"   | None, ATR, Percent |
| ATR length       | `atr_len`       | int    | 14      | Used if norm_mode == ATR |
| Slope threshold  | `slope_thr`     | float64| 0        | For signal logic |
| Curvature threshold | `curve_thr`  | float64| 0        | For signal logic |

Use `paramInt`, `paramFloat`, `paramString`, `resolveInput` from the existing indicator package.

### 2.3 Helper: MA, slope, curvature, signals

- **f_ma(src, maType, maLen)**  
  Switch on `ma_type`: SMA, EMA, HMA (use existing `computeHMA`), WMA, RMA. Reuse `talib` or existing helpers where possible; add **RMA** (Wilders) if missing: RMA = EMA with alpha = 1/length.

- **f_smooth(x, smoothType, smoothLen)**  
  Switch: None → identity; EMA/SMA/RMA → existing EWM/RollingMean/RMA.

- **Single-pass computation** (no `request.security`; one TF):
  - `m = f_ma(src)`
  - `slopeRaw = (m - m[slopeLB]) / slopeLB`, then `slopeSm = f_smooth(slopeRaw)`
  - `curveRaw = slopeSm - slopeSm[1]`, then `curveSm = f_smooth(curveRaw)`
  - Normalize: if ATR, divide by ATR(high, low, close, atrLen); if Percent, divide by m and multiply by 100 (guard m != 0).
  - Optionally compute booleans: crossover/crossunder of slope vs ±slopeThr and curve vs ±curveThr; then one-bar pulse (current true and previous false).

Return from the helper:

- `ma` ([]float64)
- `slope` ([]float64) normalized
- `curve` ([]float64) normalized
- Optional: `turnUp`, `turnDn`, `bendUp`, `bendDn`, `earlyUp`, `earlyDn` ([]float64 0/1) for alerts

### 2.4 IndicatorFuncs to register

Each returns `([]float64, error)` and uses the same params so the backend can call one set of options:

- **ma_slope_curve_ma** → MA line (for overlay/value conditions).
- **ma_slope_curve_slope** → normalized slope.
- **ma_slope_curve_curve** → normalized curvature.
- **ma_slope_curve_turn_up** → 1.0 on turn-up pulse, else 0.0.
- **ma_slope_curve_turn_dn**
- **ma_slope_curve_bend_up**
- **ma_slope_curve_bend_dn**
- **ma_slope_curve_early_up**
- **ma_slope_curve_early_dn**

Register all in **`registry.go`** in `NewDefaultRegistry()`.

### 2.5 RMA

Pine’s `ta.rma(src, length)` is Wilders smoothing (RMA). If not already in the Go codebase, add a small helper (e.g. in `types.go` or `ma_slope_curve.go`): RMA = EMA with alpha = 1/length, and use it in both `f_ma` and `f_smooth`.

### 2.6 Tests

- **indicator/ma_slope_curve_test.go**: test with fixed OHLCV; check that slope/curvature lengths match input, and that for a known trend up, slope > 0 and curvature has the expected sign. Optionally test one pulse (e.g. turn_up == 1 on one bar).

---

## 3. Python Side (So the UI and Alerts Actually Use It)

Your **condition evaluation** path uses **Python** (`src/utils/indicators.py` + `src/services/backend.py`), not the Go service. So you have two ways to use the new logic:

- **Option A (recommended for consistency and no new infra):** Implement the same MA Slope + Curvature logic in **Python** in `src/utils/indicators.py`, and wire it in `backend._calculate_indicator`. Then the Scanner and alerts work with the same expression format as today.
- **Option B:** Implement only in Go and add a way for the backend to call the Go indicator service for these indicator names; then implement a thin Python wrapper that calls Go and returns series. More infra work.

Recommended: **Option A** for the condition builder and alerts; keep Go implementation for parity, future use, or other consumers.

### 3.1 Python implementation (`src/utils/indicators.py`)

- Add:
  - **MA_SLOPE_CURVATURE_MA(df, ...)**  
    Returns the MA series (same params as below).
  - **MA_SLOPE_CURVATURE_SLOPE(df, ...)**  
    Returns normalized slope.
  - **MA_SLOPE_CURVATURE_CURVE(df, ...)**  
    Returns normalized curvature.
  - **MA_SLOPE_CURVATURE_TURN_UP / TURN_DN / BEND_UP / BEND_DN / EARLY_UP / EARLY_DN**  
    Return 0/1 series (pulse).

- Use one internal helper that computes MA, slope, curvature, and signals (mirror Go), then each public function returns the right series.
- Reuse existing `HMA`, `EMA`, `SMA`, and add WMA/RMA if missing (e.g. `talib` or simple alpha=1/length for RMA).
- Normalization: ATR via `talib.ATR`, percent = (slope/curve) / MA * 100 with guard for zero MA.

### 3.2 Backend wiring (`src/services/backend.py`)

- In **`_calculate_indicator`**, add branches for:
  - `ma_slope_curve_ma`, `ma_slope_curve_slope`, `ma_slope_curve_curve`
  - `ma_slope_curve_turn_up`, `ma_slope_curve_turn_dn`, `ma_slope_curve_bend_up`, `ma_slope_curve_bend_dn`, `ma_slope_curve_early_up`, `ma_slope_curve_early_dn`
- Map param names from condition strings: e.g. `ma_len`/`timeperiod`, `ma_type`, `input`, `slope_lookback`, `smooth_type`, `smooth_len`, `norm_mode`, `atr_len`, `slope_thr`, `curve_thr`.
- Ensure **`ind_to_dict`** can parse calls like:
  - `ma_slope_curve_slope(ma_len=200, slope_lookback=3)[-1]`
  - `ma_slope_curve_turn_up(ma_len=200)[-1] == 1`
  so that `ind` and params are filled correctly.

---

## 4. Scanner UI – Building Conditions

### 4.1 New category

In **`pages/Scanner.py`**, in the “Select Indicator Category” dropdown, add:

- **"MA Slope + Curvature"**

### 4.2 Sub-options and generated condition strings

Under **"MA Slope + Curvature"**, offer:

1. **Condition type** (selectbox):
   - Slope value (e.g. slope &gt; 0, slope &lt; 0, slope vs threshold)
   - Curvature value (e.g. curvature &gt; 0, curvature &lt; threshold)
   - Slope turn up / turn down (pulse)
   - Curvature bend up / bend down (pulse)
   - Early bend up / early bend down (pulse)

2. **Common parameters** (same for all):
   - MA type: HMA, EMA, SMA, WMA, RMA (default HMA)
   - MA length (default 200)
   - Source: Close, Open, High, Low
   - Slope lookback (default 3)
   - Smoothing: None, EMA, SMA, RMA (default EMA)
   - Smooth length (default 2)
   - Normalize by: None, ATR, Percent (default ATR)
   - ATR length (default 14) if ATR
   - Slope threshold / Curvature threshold (for signal logic; can be 0)

3. **Generated expressions** (examples to emit into the condition list):
   - Slope &gt; 0:  
     `ma_slope_curve_slope(ma_len=200, slope_lookback=3, ma_type='HMA', ...)[-1] > 0`
   - Slope turn up (pulse):  
     `ma_slope_curve_turn_up(ma_len=200, slope_lookback=3, ...)[-1] == 1`
   - Curvature &gt; threshold:  
     `ma_slope_curve_curve(ma_len=200, curve_thr=0.001, ...)[-1] > 0.001`
   - Early bend up:  
     `ma_slope_curve_early_up(ma_len=200, ...)[-1] == 1`

Use the same param dict for all so the backend receives consistent keys (`ma_len`, `ma_type`, `slope_lookback`, etc.).

### 4.3 UI layout suggestion

- First row: **Condition type** (Slope value / Curvature value / Turn up / Turn down / Bend up / Bend down / Early bend up / Early bend down).
- Next: **MA type**, **MA length**, **Source**.
- Then: **Slope lookback**, **Smoothing**, **Smooth length**, **Normalize by**, **ATR length** (conditional), **Slope threshold**, **Curvature threshold**.
- Build one condition string from the selected type and params, then append to the condition list as you do for MACD/SuperTrend/etc.

---

## 5. Order of Implementation

1. **Go**
   - Add RMA if missing.
   - Implement `ma_slope_curve.go` (helper + all 9 IndicatorFuncs).
   - Register in `registry.go`.
   - Add tests in `ma_slope_curve_test.go`.

2. **Python**
   - Implement MA Slope + Curvature helpers and public functions in `indicators.py`.
   - In `backend._calculate_indicator`, add branches for all 9 names and map params.
   - Verify `ind_to_dict` parses `ma_slope_curve_*(...)` (add tests if needed).

3. **UI**
   - Add “MA Slope + Curvature” to Scanner category dropdown.
   - Add condition-type submenu and parameter inputs.
   - Build condition strings and add to list.

4. **Optional**
   - Document in README or docs that MTF is not supported in the first version; slope/curvature are on the evaluation timeframe only.

---

## 6. Condition String Examples (for copy/paste or UI)

- Slope positive:  
  `ma_slope_curve_slope(ma_len=200, slope_lookback=3, ma_type='HMA', input='Close')[-1] > 0`
- Slope turn up (alert):  
  `ma_slope_curve_turn_up(ma_len=200, slope_lookback=3)[-1] == 1`
- Curvature above threshold:  
  `ma_slope_curve_curve(ma_len=200, curve_thr=0.001)[-1] > 0.001`
- Early bend up:  
  `ma_slope_curve_early_up(ma_len=200)[-1] == 1`

These assume param names match what you use in Go and Python (e.g. `ma_len`, `ma_type`, `slope_lookback`, `norm_mode`, `atr_len`, `slope_thr`, `curve_thr`).

---

## 7. Summary

| Layer        | Action |
|-------------|--------|
| **Go indicator/** | New `ma_slope_curve.go` with shared helper and 9 registered outputs (MA, slope, curve, 6 signal pulses); add RMA if needed; tests. |
| **Python indicators.py** | Same logic: helper + 9 functions; reuse HMA/EMA/SMA, add WMA/RMA if needed. |
| **backend.py**        | Map 9 indicator names in `_calculate_indicator`; ensure `ind_to_dict` supports the new param names. |
| **Scanner.py**        | New category “MA Slope + Curvature” with condition types and params, emitting the condition strings above. |

This gives you the indicator in `indicator/` and the ability to build MA Slope + Curvature conditions in the UI and use them in alerts/scans.

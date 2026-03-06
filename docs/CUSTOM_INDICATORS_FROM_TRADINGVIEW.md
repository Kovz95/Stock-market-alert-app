# Using a TradingView (Pine Script) Indicator as a Custom Indicator

This guide explains how to take an indicator you built in TradingView (Pine Script) and use it in this app so you can write conditions like `my_indicator(14)[-1] > 0` in alerts and the scanner.

---

## Overview

1. **Translate** your Pine Script logic into a Go function that matches the app’s indicator contract.
2. **Register** that function under a name (e.g. `my_indicator`) in the indicator registry.
3. **Optionally** add parameter remapping in the expression parser so `my_indicator(14)` maps to your param names.

The app does **not** run Pine Script directly; you reimplement the logic in Go. For most indicators you can lean on the existing `indicator` package and **go-talib** so the translation is straightforward.

---

## 1. The indicator contract

Every indicator in this app is a function with this signature:

```go
func MyIndicator(data *indicator.OHLCV, params map[string]interface{}) ([]float64, error)
```

- **`data`**: OHLCV bars (Open, High, Low, Close, Volume). Index 0 = oldest bar; `data.Len()` = number of bars.
- **`params`**: Named arguments from the expression, e.g. `my_indicator(14, input="High")` → `params["period"] = 14`, `params["input"] = "High"`. Use `paramInt`, `paramFloat`, `paramString` and `resolveInput` from the indicator package.
- **Return**: A slice of `float64` with **the same length as** `data.Close`. Use `math.NaN()` for bars where the value cannot be computed (e.g. warmup).

---

## 2. Where to put your code

- **New indicator logic**: add a new file under `indicator/`, e.g. `indicator/custom.go`.
- **Registration**: add one or more `r.Register("name", MyIndicator)` calls in `indicator/registry.go` inside `NewDefaultRegistry()` (after the existing blocks).

If you prefer not to touch `registry.go`, you can instead create a **custom registry** that starts from the default and adds your indicators (see section 5). Then you’d wire that registry where the app builds the evaluator (e.g. alert service, scanner).

---

## 3. Example: Pine Script → Go

**Pine Script (TradingView)** — “double smoothed RSI” style:

```pinescript
// @version=5
indicator("My RSI", overlay = false)
len = input.int(14, "Length")
src = input.source(close, "Source")
r = ta.rsi(src, len)
smoothed = ta.ema(r, 7)
plot(smoothed, "Smoothed RSI")
```

**Go equivalent** in `indicator/custom.go`:

```go
package indicator

import (
	"github.com/markcheno/go-talib"
)

// MySmoothedRSI computes EMA(RSI(source, length), smooth_len).
// Params: timeperiod (int, default 14), smooth (int, default 7), input (string, default "Close").
func MySmoothedRSI(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "timeperiod", 14)
	smooth := paramInt(params, "smooth", 7)
	input := resolveInput(data, params)

	rsi := talib.Rsi(input, period)
	smoothed := talib.Ema(rsi, smooth)
	return smoothed, nil
}
```

Then in **`indicator/registry.go`** inside `NewDefaultRegistry()` add:

```go
// Custom (from TradingView / Pine)
r.Register("my_smoothed_rsi", MySmoothedRSI)
```

Users can then write conditions like:

- `my_smoothed_rsi(14)[-1] < 30`
- `my_smoothed_rsi(14, smooth=7)[-1] > 70`
- `my_smoothed_rsi(14, input="High")[-1] > 50`

---

## 4. Parameter names and the expression parser

Expressions like `my_smoothed_rsi(14)` pass a single positional argument. The parser stores it as `params["period"]`. Your Go code can use `paramInt(params, "period", 14)` or you can remap `period` to something else (e.g. `timeperiod`) in the parser so it matches your indicator’s param names.

To have the first positional argument map to **`timeperiod`** (like SMA/RSI), add your indicator to the remap in **`expr/parser.go`** in `remapPositionalParams()`:

```go
if period, ok := op.Params["period"]; ok {
	switch ind {
	// ... existing cases ...
	case "my_smoothed_rsi":
		op.Params["timeperiod"] = period
		delete(op.Params, "period")
	}
}
```

If you use **keyword-only** params in expressions (e.g. `my_smoothed_rsi(timeperiod=14, smooth=7)`), no remap is needed; the parser already passes through named params.

---

## 5. Optional: custom registry without editing `registry.go`

If you don’t want to modify `indicator/registry.go`, you can build a registry that includes both the default indicators and your custom ones:

```go
// In your app's main or config package
func NewRegistryWithCustom() *indicator.Registry {
	r := indicator.NewDefaultRegistry()
	r.Register("my_smoothed_rsi", indicator.MySmoothedRSI)
	// r.Register("another_custom", indicator.AnotherCustom)
	return r
}
```

Then use `NewRegistryWithCustom()` wherever you currently use `indicator.NewDefaultRegistry()` (e.g. in the alert service main, scanner, or worker). That way custom indicators live in your code and the core indicator package stays unchanged.

---

## 6. Pine → Go translation tips

| Pine | Go (this app) |
|------|----------------|
| `close`, `high`, `low`, `open`, `volume` | `data.Close`, `data.High`, `data.Low`, `data.Open`, `data.Volume` |
| `ta.sma(src, len)` | `talib.Sma(src, len)` |
| `ta.ema(src, len)` | `talib.Ema(src, len)` |
| `ta.rsi(src, len)` | `talib.Rsi(src, len)` |
| `ta.bb(close, 20, 2)` | Use indicator `bbands` with `timeperiod`, `std_dev`, `type` |
| `ta.macd(...)` | Use indicator `macd` with `fast_period`, `slow_period`, `signal_period`, `type` |
| `ta.atr(14)` | `talib.Atr(data.High, data.Low, data.Close, 14)` |
| `src[1]` (previous bar) | `src[i-1]` in a loop; or build a shifted slice with `indicator.Shift(src, 1)` |
| `math.max(a, b)` | `math.Max(a, b)` |
| Array / series length | `len(data.Close)` or `data.Len()` |

Use existing helpers in the `indicator` package so your custom code stays short and consistent: `paramInt`, `paramFloat`, `paramString`, `resolveInput`, `Shift`, `NaN`, and go-talib for standard studies.

---

## 7. Summary

1. Implement a function `func(data *OHLCV, params map[string]interface{}) ([]float64, error)` in a new file under `indicator/` (e.g. `custom.go`).
2. Register it in `indicator/registry.go` with `r.Register("my_indicator", MyIndicator)` inside `NewDefaultRegistry()`, or use a `NewRegistryWithCustom()` that calls `NewDefaultRegistry()` and then registers your indicators.
3. If you use positional args like `my_indicator(14)`, add a remap in `expr/parser.go` so `period` (or similar) maps to your param names if needed.
4. Use your indicator in conditions: `my_indicator(14)[-1] > 30` or in the scanner like any other indicator.

There is no automatic Pine→Go converter; you reimplement the logic in Go. For most indicators built from `ta.*` and simple math, the translation is a small amount of code.

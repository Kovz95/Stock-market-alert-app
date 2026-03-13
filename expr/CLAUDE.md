# `expr/` — Expression Language for Stock Alert Conditions

## What This Package Is

`expr` is a **standalone Go module** (`stockalert/expr`) that implements a mini expression language for evaluating stock market alert conditions.

It takes a human-readable string like `"RSI(14)[-1] < 30"` and determines whether that condition is true or false for a given stock's price history. It is the **brain of the alert and scanner systems** — whenever the app needs to decide "did this alert trigger?", `expr` is doing the work.

---

## How It Fits Into The App

```
User creates an alert in the UI
  → Conditions are stored as strings in the database
  → alert/evaluator.go reads those strings from the database
      → Calls expr.Evaluator to evaluate each condition
          → expr fetches the computed indicator value from indicator/
          → expr compares the value and returns true or false
  → Alert fires (or doesn't)
```

There are two places in the codebase that use `expr` directly:

| File | What It Does |
|---|---|
| `alert/evaluator.go` | Wraps `expr.Evaluator` for the alert-triggering pipeline |
| `apps/grpc/price_service/scan.go` | Uses `expr` for the real-time market scanner feature |

The `expr` package **never touches the database** and **never fetches prices**. It receives already-loaded OHLCV price data and indicator functions from `indicator/`, then evaluates expressions. This makes it easy to test in isolation.

---

## File-by-File Breakdown

### `ast.go` — Data Structures

Defines the three core data types that represent a parsed expression:

```
Operand         → one side of a comparison
                  e.g. "RSI(14)[-1]" or the number "30"

Comparison      → a full binary comparison
                  e.g. "RSI(14)[-1] < 30"

CombinedCondition → a list of Comparisons with AND/OR logic
                  e.g. ["RSI(14)[-1] < 30", "close[-1] > 100"] combined with "AND"
```

**`Operand` fields:**

| Field | Type | Meaning |
|---|---|---|
| `IsNum` | bool | True if this is a plain number (like `30`) |
| `Number` | float64 | The number value when `IsNum` is true |
| `Indicator` | string | The indicator/column name, always lowercase (e.g. `"rsi"`, `"close"`) |
| `Params` | map[string]interface{} | Keyword arguments parsed from the expression |
| `Specifier` | int | The bracket index (e.g. `-1` from `[-1]`). Default is `-1` (last bar) |

---

### `tokenizer.go` — Lexer

Converts a raw expression string into a list of tokens. This is the first step in parsing.

**Token types:**
- `TokenIdent` — identifier like `Close`, `RSI`, `sma`
- `TokenNumber` — a number like `30`, `-2`, `14.5`
- `TokenLParen` / `TokenRParen` — `(` and `)`
- `TokenLBracket` / `TokenRBracket` — `[` and `]`
- `TokenComma` — `,`
- `TokenEquals` — `=` (used for keyword arguments)
- `TokenOp` — comparison operators: `>`, `<`, `>=`, `<=`, `==`, `!=`
- `TokenString` — quoted string: `'upper'`, `"Close"`
- `TokenEOF` — end of input

You typically never call the tokenizer directly. The parser calls it internally.

---

### `parser.go` — Parser

Converts tokens into `Operand` and `Comparison` structs.

**Public functions you call:**

```go
// Parse a single operand (one side of a comparison)
op, err := expr.ParseOperand("RSI(14)[-1]")

// Parse a full condition like "RSI(14)[-1] < 30"
comp, err := expr.ParseCondition("RSI(14)[-1] < 30")

// Parse a list of conditions with combination logic
combined, err := expr.ParseConditionList(
    []string{"RSI(14)[-1] < 30", "close[-1] > 100"},
    "AND",
)
```

**Positional parameter remapping:** The parser automatically converts positional arguments to named arguments for each indicator. For example:
- `rsi(14)` → `{timeperiod: 14}` (the `14` maps to `timeperiod`)
- `macd(12, 26, 9)` → `{fast_period: 12, slow_period: 26, signal_period: 9}`
- `bbands(20, 2.0)` → `{timeperiod: 20, std_dev: 2.0}`

This remapping table lives in `remapPositionalParams()` inside `parser.go`.

---

### `catalog.go` — Catalog Condition Expander

The UI lets users pick "catalog" conditions from a dropdown instead of writing raw expressions. These are simple named shortcuts with optional values.

`ExpandCatalogCondition()` converts these into real expression strings:

| Catalog input | Expands to |
|---|---|
| `price_above: 150` | `close[-1] > 150` |
| `price_below: 100` | `close[-1] < 100` |
| `price_equals: 50` | `close[-1] == 50` |
| `price_above_ma: 20` | `close[-1] > sma(20)[-1]` |
| `price_above_ma: 20 (EMA)` | `close[-1] > ema(20)[-1]` |
| `price_below_ma: 50 (HMA)` | `close[-1] < hma(50)[-1]` |
| `ma_crossover: 10 > 20` | `sma(10)[-1] > sma(20)[-1]` |
| `ma_crossover: 10 > 20 (EMA)` | `ema(10)[-1] > ema(20)[-1]` |
| `ma_crossunder: 10 > 20` | `sma(10)[-1] < sma(20)[-1]` |
| `rsi_oversold: 30` | `rsi(14)[-1] < 30` |
| `rsi_oversold: 25 (21)` | `rsi(21)[-1] < 25` (custom level + period) |
| `rsi_overbought: 70` | `rsi(14)[-1] > 70` |
| `macd_bullish_crossover` | `macd(12, 26, 9, type=line)[-1] > macd(12, 26, 9, type=signal)[-1]` |
| `macd_bearish_crossover` | `macd(12, 26, 9, type=line)[-1] < macd(12, 26, 9, type=signal)[-1]` |
| `macd_histogram_positive` | `macd(12, 26, 9, type=histogram)[-1] > 0` |
| `price_above_upper_band` | `close[-1] > bbands(20, 2.0, type='upper')[-1]` |
| `price_below_lower_band` | `close[-1] < bbands(20, 2.0, type='lower')[-1]` |
| `volume_above_average: 1.5x` | `volume_ratio(20)[-1] > 1.5` |
| `volume_spike: 2x` | `volume_ratio(20)[-1] > 2` |
| `volume_below_average: 0.5x` | `volume_ratio(20)[-1] < 0.5` |

If the input string is NOT a catalog condition, `ExpandCatalogCondition` returns `""` and the evaluator treats it as a raw expression.

---

### `evaluator.go` — Evaluator

This is the entry point for actually running an expression against real price data.

**Creating an evaluator:**
```go
registry := indicator.NewDefaultRegistry()
eval := expr.NewEvaluator(registry)
```

**Evaluating a single condition:**
```go
triggered, err := eval.EvalCondition(data, "RSI(14)[-1] < 30", ctx)
```

**Evaluating multiple conditions with combination logic:**
```go
triggered, err := eval.EvalConditionList(
    data,
    []string{"RSI(14)[-1] < 30", "close[-1] > 100"},
    "AND",     // or "OR" or "1 AND (2 OR 3)"
    ctx,
)
```

The `ctx` parameter is a `map[string]interface{}` that carries runtime context. Currently only `ctx["ticker"]` is forwarded to indicators (used by a small number of indicators that need the ticker symbol).

**What happens inside `EvalCondition`:**
1. If the condition is a catalog shortcut, expand it first
2. Parse the condition string into a `Comparison`
3. Resolve each operand:
   - If it's a number, use the number directly
   - If it's a price column (`close`, `open`, `high`, `low`, `volume`), get that column from the OHLCV data
   - If it's an indicator, call the registered indicator function
4. Apply the `[-n]` bracket specifier to get a single scalar value from the series (default `-1` = last bar, `-2` = second to last, etc.)
5. Compare the two scalars with the operator

**NaN safety:** If either side of a comparison resolves to NaN (e.g., not enough data to compute the indicator), the comparison returns `false` rather than panicking.

**Panic safety:** If a go-talib indicator panics (e.g., period larger than data length), the panic is caught and returned as an error.

---

## Expression Syntax Reference

### Price columns (no parameters needed)

```
Close[-1]       last closing price
Open[-1]        last opening price
High[-1]        last high
Low[-1]         last low
Volume[-1]      last volume
```

Column names are case-insensitive. `Close`, `close`, `CLOSE` all work.

### Indicator with positional parameter

```
rsi(14)[-1]               RSI with period 14, last bar
sma(20)[-1]               SMA with period 20, last bar
ema(50)[-2]               EMA with period 50, second-to-last bar
atr(14)[-1]               ATR with period 14
```

### Indicator with keyword parameters

```
EWO(sma1_length=5, sma2_length=35)[-1]
bbands(20, 2.0, type='upper')[-1]
macd(12, 26, 9, type=line)[-1]
macd(12, 26, 9, type=signal)[-1]
macd(12, 26, 9, type=histogram)[-1]
supertrend(10, multiplier=3.0)[-1]
```

### Nested indicators (indicator-as-input)

```
zscore(rsi(14), lookback=20)[-1]          z-score of RSI
sma(period=20, input=rsi(14))[-1]         SMA applied to RSI values
```

### Comparison operators

```
>    greater than
<    less than
>=   greater than or equal
<=   less than or equal
==   equal
!=   not equal
```

### Full condition examples

```
RSI(14)[-1] < 30
close[-1] > sma(20)[-1]
ema(10)[-1] > ema(50)[-1]
macd(12, 26, 9, type=line)[-1] > macd(12, 26, 9, type=signal)[-1]
close[-1] > bbands(20, 2.0, type='upper')[-1]
volume_ratio(20)[-1] > 1.5
zscore(rsi(14), lookback=20)[-1] > 2
```

### Combination logic

```
""              default, same as AND
"AND"           all conditions must be true
"OR"            any condition must be true
"1 AND 2"       conditions 1 and 2 must be true
"1 AND (2 OR 3)"  condition 1 AND (condition 2 OR condition 3)
"NOT 1 AND 2"   NOT condition 1, AND condition 2
```

Conditions are referenced by their 1-based index in the list.

---

## Indicator Aliases

These names in an expression are automatically mapped to their canonical registry name:

| Alias | Canonical |
|---|---|
| `bb` | `bbands` |
| `psar` | `sar` |
| `harsi` | `harsi_flip` |

---

## All Available Indicators

These are registered in `indicator/registry.go`. Use any of these names in an expression:

**Price / Volume:**
`close`, `open`, `high`, `low`, `volume`

**Moving Averages:**
`sma`, `ema`, `hma`, `frama`, `kama`

**Oscillators:**
`rsi`, `stoch_k`, `stoch_d`, `stoch_rsi_k`, `stoch_rsi_d`, `cci`, `willr`, `roc`, `mom`, `cmo`, `mfi`, `adx`

**Trend / Volatility:**
`atr`, `natr`, `atr`, `bbands` (alias: `bb`), `sar` (alias: `psar`), `supertrend`, `supertrend_upper`, `supertrend_lower`

**MACD family:**
`macd`, `macd_line`, `macd_signal`, `macd_histogram`

**Statistical:**
`stddev`, `linear_reg`, `linear_reg_slope`, `aroon_osc`, `zscore`, `ma_spread_zscore`

**Volume:**
`volume_ratio`, `obv`, `obv_macd`, `obv_macd_signal`

**Advanced / Custom:**
`ewo`, `harsi_flip` (alias: `harsi`), `slope_sma`, `slope_ema`, `slope_hma`

**MA Slope + Curvature:**
`ma_slope_curve_ma`, `ma_slope_curve_slope`, `ma_slope_curve_curve`, `ma_slope_curve_turn_up`, `ma_slope_curve_turn_dn`, `ma_slope_curve_bend_up`, `ma_slope_curve_bend_dn`, `ma_slope_curve_early_up`, `ma_slope_curve_early_dn`

**Ichimoku:**
`ichimoku_conversion`, `ichimoku_base`, `ichimoku_span_a`, `ichimoku_span_b`, `ichimoku_lagging`, `ichimoku_cloud_top`, `ichimoku_cloud_bottom`, `ichimoku_cloud_signal`

**Donchian:**
`donchian_upper`, `donchian_lower`, `donchian_basis`, `donchian_width`, `donchian_position`

**Pivot Support/Resistance:**
`pivot_sr`, `pivot_sr_crossover`, `pivot_sr_proximity`

**Trend Magic:**
`trend_magic`, `trend_magic_signal`

**Kalman:**
`kalman_roc_stoch`, `kalman_roc_stoch_signal`, `kalman_roc_stoch_crossover`

**Custom:**
`my_smoothed_rsi`

---

## Running Tests

From within the `expr/` directory:

```bash
go test ./...
```

Or from the project root:

```bash
cd expr && go test ./...
```

The test files are:
- `expr_test.go` — tokenizer, parser, evaluator, and catalog tests
- `nested_indicator_test.go` — tests for nested indicator expressions (zscore, MA-of-indicator, FRAMA/KAMA, MACD components)

---

## How To Extend

### Add a new catalog shortcut

Edit `catalog.go` and add a new `case` inside the `switch key` block in `ExpandCatalogCondition()`:

```go
case "my_new_shortcut":
    // valueStr is everything after the colon in "my_new_shortcut: <value>"
    if v, err := strconv.ParseFloat(strings.TrimSpace(valueStr), 64); err == nil {
        return fmt.Sprintf("my_indicator(14)[-1] > %g", v)
    }
    return ""
```

Then add a test in `expr_test.go` in the `TestExpandCatalogCondition` function.

### Add a new indicator

1. **Implement the function** in `indicator/` (add it to an appropriate file like `indicator/basic.go`, `indicator/advanced.go`, or create a new file):

```go
// MyIndicator computes my custom indicator.
func MyIndicator(data *OHLCV, params map[string]interface{}) ([]float64, error) {
    period := paramInt(params, "timeperiod", 14)
    // ... compute the series ...
    // IMPORTANT: return a slice with the same length as data.Close
    // Fill leading values (where calculation is impossible) with math.NaN()
    result := make([]float64, len(data.Close))
    for i := range result {
        result[i] = math.NaN()
    }
    // ... fill in real values starting at index `period-1` ...
    return result, nil
}
```

2. **Register it** in `indicator/registry.go` inside `NewDefaultRegistry()`:

```go
r.Register("my_indicator", MyIndicator)
```

3. **Add positional parameter remapping** (optional) in `parser.go` inside `remapPositionalParams()`, if your indicator should accept a shorthand positional argument:

```go
case "my_indicator":
    // Map the first positional arg to "timeperiod"
    // (Most indicators already get this via the default sma/ema/rsi case above)
```

4. **Add an alias** (optional) in `evaluator.go` in the `indicatorAliases` map:

```go
var indicatorAliases = map[string]string{
    "bb":           "bbands",
    "psar":         "sar",
    "harsi":        "harsi_flip",
    "myind":        "my_indicator",  // new alias
}
```

5. **Write tests** in `expr/expr_test.go` or `expr/nested_indicator_test.go`.

### Add a new operator

Edit `ast.go` and add the operator to `ComparisonOps`. Then handle it in `compareValues()` in `evaluator.go`.

---

## Common Mistakes and Edge Cases

**"Unknown indicator" error:**
The indicator name in the expression does not match any registered name. Check spelling and case — all indicator names are compared lowercase. Make sure the indicator is registered in `indicator/registry.go`.

**NaN result (condition always false):**
This usually means there isn't enough price history to compute the indicator. For example, RSI(14) needs at least 14 bars. The evaluator silently returns `false` for NaN comparisons — it does NOT return an error.

**Panic caught as error:**
Some go-talib functions panic when the data is shorter than the period. The evaluator wraps the call in a recover, so you get an error like `indicator "rsi" panicked (data may be too short)` instead of a crash.

**Positional param going to wrong key:**
If you add a new indicator with positional args and it's not in `remapPositionalParams()`, the first positional arg lands in `"period"` and the rest in `"_pos_1"`, `"_pos_2"`, etc. These `_pos_N` keys are deleted before the indicator function is called, so those params are silently lost. Always add a remapping entry for multi-param indicators.

**`[-1]` is the default:**
You don't need to write `Close[-1]` — `Close` alone defaults to specifier `-1`. But being explicit is clearer and always valid.

**Nested operands resolve at evaluation time:**
When you write `zscore(rsi(14), lookback=20)`, the inner `rsi(14)` is stored as a nested `*Operand` during parsing. At evaluation time, `computeSeries` recursively computes the inner series first, then passes it to the outer indicator under the key `_computed_input`. The outer indicator function must look for its input under `params["_computed_input"]` (not `params["input"]`).

---

## Module Structure

```
expr/
├── ast.go           Data types: Operand, Comparison, CombinedCondition
├── tokenizer.go     Lexer: string → []Token
├── parser.go        Parser: []Token → Operand / Comparison / CombinedCondition
├── catalog.go       Catalog expander: "rsi_oversold: 30" → "rsi(14)[-1] < 30"
├── evaluator.go     Evaluator: runs comparisons against real OHLCV data
├── expr_test.go     Tests for all of the above
├── nested_indicator_test.go  Tests for nested indicator expressions
├── go.mod           Module: stockalert/expr, depends on stockalert/indicator
└── go.sum           Dependency checksums
```

The `expr` module has one dependency outside the standard library: `stockalert/indicator` (the sibling `indicator/` directory, referenced via `replace` in `go.mod`).

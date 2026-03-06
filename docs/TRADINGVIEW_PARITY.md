# TradingView vs This App: Indicators and Conditions

This doc compares how **TradingView** (Pine Script + Alerts) structures indicators and conditions with how **this app** (Go `indicator/` + `expr/`) does it, so you can align naming, parameters, and UX where desired.

---

## 1. Indicator naming and invocation

| Aspect | TradingView (Pine Script) | This app |
|--------|---------------------------|----------|
| **Namespace** | Built-in functions live under `ta.*`: `ta.sma()`, `ta.rsi()`, `ta.bb()`, `ta.macd()`, etc. | No namespace: `sma()`, `rsi()`, `bbands()`, `macd()`. Aliases: `bb` → `bbands`, `psar` → `sar`, `harsi` → `harsi_flip`. |
| **Arguments** | Usually (source, length): `ta.sma(close, 20)`, `ta.rsi(close, 14)`. Source defaults to `close` when omitted in some functions. | Params by name: `sma(20)` → `timeperiod: 20`, default source `Close`; `rsi(14)` → `timeperiod: 14`. Parser remaps positional to named (e.g. first arg → `timeperiod` or `length`). |
| **Multi-output** | One function returns multiple series: `[upper, middle, lower] = ta.bb(close, 20, 2)`; access by index or named. | Either one function + `type` param (e.g. `bbands(20, 2, type='upper')`) or separate registry names (e.g. `donchian_upper`, `donchian_lower`). |

So: **conceptually the same** (indicator name + params → one or more series). Differences are surface: no `ta.` prefix here, and multi-output is either a `type` param or separate indicator names.

---

## 2. Price / series references

| Aspect | TradingView | This app |
|--------|-------------|----------|
| **OHLCV** | `open`, `high`, `low`, `close`, `volume`. Also `hl2`, `hlc3`, `ohlc4` etc. | Same columns: `close`, `open`, `high`, `low`, `volume`; plus `HL2`, `HLC3`, `OHLC4` in `resolveInput` / `Column()`. |
| **Indexing** | `close[1]` = previous bar, `close` = current bar. Index 0 is oldest in history. | `close[-1]` = last (most recent) bar; `[-2]` = second-to-last. Same “0 = oldest” series, Python-style negative index from end. |

So: **same data**, different index convention (Pine positive offset vs this app’s negative-from-end).

---

## 3. Conditions and alerts

| Aspect | TradingView | This app |
|--------|-------------|----------|
| **Condition form** | You write a Pine expression; alert fires when that expression is true (e.g. `ta.crossover(shortMa, longMa)`, `rsi < 30`). | **Expression form**: e.g. `rsi(14)[-1] < 30`, `close[-1] > sma(20)[-1]`. **Catalog form**: short keys that expand to expressions, e.g. `rsi_oversold: 30` → `rsi(14)[-1] < 30`, `price_above_ma: 20 (SMA)` → `close[-1] > sma(20)[-1]`. |
| **Combination** | You combine in Pine with `and` / `or` (e.g. `rsi < 30 and close > ta.sma(close, 50)`). | List of condition strings + combination: `AND`, `OR`, or `1 AND (2 OR 3)` (numeric refs to condition indices). |
| **Catalog / UI** | No fixed “catalog”; user writes script. Alert dialog picks “when script condition is true” or “when order fill”, etc. | Fixed catalog for the UI: price_above, price_below_ma, rsi_oversold, rsi_overbought, macd_bullish_crossover, price_above_upper_band, volume_above_average, etc. Same catalog drives Add Alert and Scanner. |

So: **TradingView = free-form Pine; this app = expression language + catalog shortcuts** that compile to the same expression format the backend evaluates.

---

## 4. Parameter naming alignment

Many indicators use the same idea (period/length) with different names. This app’s parser already remaps so that **positional and common names** line up with what the Go indicators expect:

| Indicator(s) | TradingView-style | This app registry param |
|--------------|-------------------|-------------------------|
| SMA, EMA, RSI, ROC, ATR, CCI, WILLR, volume_ratio | period / length | `timeperiod` |
| MACD | fast, slow, signal | `fast_period`, `slow_period`, `signal_period` |
| BBANDS | period, mult | `timeperiod`, `std_dev`, `type` (upper/middle/lower) |
| SAR | start, inc, max | `acceleration`, `max_acceleration` |
| Supertrend | length, mult | `period`, `multiplier` |
| Donchian | length | `length` |
| MA slope/curve | ma_len, slope_lookback, etc. | `ma_len`, `slope_lookback`, … (see indicator package) |

So: **structure matches**; we use our own param names and remap in `expr` so user-facing expressions (and catalog) stay simple.

---

## 5. Summary: does this match TradingView?

- **Indicators**: Same *concepts* (SMA, RSI, MACD, BB, etc.) and same typical inputs (source, period/length). Differences: no `ta.` prefix, multi-output via `type` or separate names, and explicit param names in the backend.
- **Data**: Same OHLCV + HL2/HLC3/OHLC4; indexing is negative-from-end here instead of Pine’s positive offset.
- **Conditions**: Same idea (boolean expression over series), but this app adds a **catalog layer** (e.g. `rsi_oversold: 30`) that expands to the same expression form the evaluator runs. Combination is AND/OR plus numeric refs instead of inline Pine.

So it **does match TradingView in structure and intent**; the app is a constrained, catalog-friendly version of “indicator + condition” that can be evaluated server-side without running Pine. To make the app feel even closer to TradingView you could:

- Expose a `ta_` or `ta.` prefix in the expression parser (optional, cosmetic).
- Add more catalog entries that mirror common Pine patterns (e.g. `ta.crossover(short, long)` → a “MA crossover” condition).
- Document the mapping (e.g. in the Indicator Guide) so users used to TradingView know the equivalent expressions.

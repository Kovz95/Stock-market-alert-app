# TA-Lib (go-talib) Indicators Roadmap

Your app already exposes many go-talib functions (SMA, EMA, RSI, ROC, ATR, CCI, WILLR, MACD, BBANDS, SAR, KAMA) plus custom/composite indicators. Below is what’s **not yet** exposed and is worth adding, grouped by use case.

---

## Already exposed (go-talib)

| App indicator   | go-talib  | Notes                    |
|-----------------|-----------|--------------------------|
| sma, ema        | Sma, Ema  | basic.go                 |
| rsi             | Rsi       | basic.go                 |
| roc             | Roc       | basic.go                 |
| atr             | Atr       | basic.go                 |
| cci             | Cci       | basic.go                 |
| willr           | WillR     | basic.go                 |
| macd            | Macd      | macd.go (line/signal/hist) |
| bbands          | BBands    | bbands.go (upper/mid/lower) |
| sar             | Sar       | sar.go                   |
| kama            | Kama      | advanced.go              |
| volume_ratio    | Sma(vol)  | basic.go                 |
| donchian_*      | Max, Min  | donchian.go              |
| ichimoku_*      | Max, Min  | ichimoku.go              |
| supertrend      | Atr       | supertrend.go            |
| obv_macd        | (uses Obv, Ema) | obv_macd.go      |
| kalman_roc_stoch| Min, Max, Sma, Ema | kalman.go   |

---

## High priority (commonly used, thin wrappers)

| go-talib    | Suggested name   | Params              | Description                    |
|-------------|------------------|----------------------|--------------------------------|
| **Adx**     | adx              | timeperiod (14)      | Trend strength (0–100).       |
| **Stoch**   | stoch_k, stoch_d | fast_k(5), slow_k(3), slow_d(3) | Stochastic %K and %D. |
| **StochRsi**| stoch_rsi_k, stoch_rsi_d | timeperiod(14), fast_k(5), fast_d(3) | Stochastic RSI. |
| **Obv**     | obv              | —                    | On Balance Volume (raw).      |
| **Mfi**     | mfi              | timeperiod (14)      | Money Flow Index (volume).    |
| **Mom**     | mom              | timeperiod (10)      | Momentum (close - close[n]).  |
| **Ad**      | ad               | —                    | Accumulation/Distribution.    |

These are single-series or two-series outputs; your registry can expose `adx`, `stoch_k`, `stoch_d`, `stoch_rsi_k`, `stoch_rsi_d`, `obv`, `mfi`, `mom`, `ad` with `IndicatorFunc` wrappers that call go-talib and return one series each (or two names for Stoch/StochRsi).

---

## Medium priority (trend / volatility / slope) — **implemented**

| go-talib         | Registry name    | Params              | Description                    |
|------------------|------------------|----------------------|--------------------------------|
| **PlusDI**       | plus_di          | timeperiod (14)      | +DI (part of ADX system).     |
| **MinusDI**      | minus_di         | timeperiod (14)      | -DI.                          |
| **Natr**         | natr             | timeperiod (14)      | Normalized ATR (%).            |
| **LinearRegSlope** | linear_reg_slope | timeperiod (14), input | Slope of linear regression. |
| **LinearReg**    | linear_reg       | timeperiod (14), input | Linear regression value.   |
| **StdDev**       | stddev           | timeperiod (20), nb_dev (1), input | Rolling std dev. |
| **AroonOsc**     | aroon_osc        | timeperiod (14)      | Aroon Up - Aroon Down.        |
| **Cmo**          | cmo              | timeperiod (14), input | Chande Momentum Oscillator.  |

All implemented in `indicator/talib_ext.go`, registered in `registry.go`, with parser remaps and UI list updated.

---

## Lower priority (nice to have)

| go-talib    | Suggested name | Notes                          |
|-------------|----------------|--------------------------------|
| AdxR        | adxr            | ADX smoothed (rating).         |
| AdOsc       | ad_osc          | Chaikin A/D Oscillator.        |
| Aroon       | aroon_up, aroon_dn | Two series.                |
| Dema        | dema            | Double EMA (you have in kalman). |
| Tema        | tema            | Triple EMA.                    |
| Ppo         | ppo             | Percentage Price Oscillator.    |
| Trix        | trix            | Triple EMA ROC.                |
| UltOsc      | ult_osc         | Ultimate Oscillator (3 periods). |
| Trima       | trima           | Triangular MA.                  |
| MidPoint    | mid_point       | Midpoint of period range.       |
| MidPrice    | mid_price       | (High+Low)/2 over period.       |
| TypPrice    | typ_price       | (H+L+C)/3.                      |
| Sum         | sum             | Sum over period.                |
| Var         | var             | Variance over period.           |
| Beta        | beta            | Beta of two series.             |
| Correl      | correl          | Correlation of two series.      |

---

## Implementation pattern

For each new indicator:

1. **Add a function** in `indicator/basic.go` (or a new file, e.g. `indicator/talib_ext.go`) that matches `IndicatorFunc`:
   - `func ADX(data *OHLCV, params map[string]interface{}) ([]float64, error)`
   - Use `paramInt(params, "timeperiod", 14)`, `resolveInput(data, params)` where applicable.
   - Call the go-talib function and return a single `[]float64` (same length as `data.Close`; leading NaNs where needed).
2. **Register** in `indicator/registry.go`: `r.Register("adx", ADX)`.
3. **Optional:** add param remapping in `expr/parser.go` so `adx(14)` maps to `timeperiod: 14` (if not already under a generic “period” remap).
4. **Optional:** add the name to the UI `indicatorList.ts` so it appears in “Any indicator”.

Multi-output indicators (e.g. Stoch returns K and D):

- Either register two names: `stoch_k` and `stoch_d`, each returning one series, or
- One name with a `type` param (e.g. `stoch(5,3,3, type='k')` / `type='d'`) like MACD/BBANDS.

---

## Summary

- **Implement next (high value, low effort):** ADX, Stoch (K/D), StochRsi (K/D), OBV, MFI, Mom, Ad.
- **Then:** PlusDI, MinusDI, Natr, LinearRegSlope, StdDev, AroonOsc, Cmo.
- **Later:** Dema, Tema, Ppo, Trix, UltOsc, Trima, and the rest as needed.

All of these are thin wrappers around existing go-talib calls; no new algorithms, just exposure in your registry and optional expr/UI wiring.

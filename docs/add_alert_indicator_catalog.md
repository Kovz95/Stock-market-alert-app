# Add Alert Indicator Catalog (Streamlit Add Alert page)

For each indicator category, this lists the condition types and the available preset conditions/outputs.

## Price Data
- Price Comparison: `price_above`, `price_below`, `price_equals` (numeric value).
- Price Data Points: `Close[-1]`, `Open[-1]`, `High[-1]`, `Low[-1]`, `Close[-2]`, `Open[-2]`, `High[-2]`, `Low[-2]`, `Close[0]`, `Open[0]`, `High[0]`, `Low[0]`.

## Moving Averages
- Price vs MA: `price_above_ma`, `price_below_ma` on SMA/EMA/HMA/FRAMA/KAMA with custom period.
- MA Crossover: `ma_crossover`, `ma_crossunder` (fast > slow) for SMA/EMA/HMA/FRAMA/KAMA.
- MA Value: Raw MA value for FRAMA, KAMA, or SMA/EMA/HMA on inputs Close/Open/High/Low/EWO/RSI/MACD (line/signal/histogram).

## RSI
- RSI Levels: Oversold (`< level`), Overbought (`> level`), Neutral (`between 30 and 70`), all with custom period/levels.
- RSI Value: Raw `rsi(period)[-1]`.

## MACD
- MACD Crossovers: Bullish crossover, Bearish crossover.
- MACD Values: MACD Line/Signal/Histogram with operator `>`, `<`, `>=`, `<=`, `==` and numeric threshold.

## Bollinger Bands
- Band Value: `bb_upper`, `bb_middle`, `bb_lower`, or `bb_width` with custom period and std dev.

## Volume
- Volume Conditions: `volume_above_average` (multiplier), `volume_spike` (multiplier), `volume_below_average` (fraction).
- Volume Data: `volume[-1]`, `volume[0]`, `volume_avg(20)[-1]`, `volume[-1] / volume_avg(20)[-1]`.

## ATR
- Value: `atr(period)[-1]`.

## CCI
- Value: `cci(period)[-1]`.

## Williams %R
- Value: `willr(period)[-1]`.

## ROC
- Value: `roc(period)[-1]`.

## EWO
- Condition Type: EWO Levels – Above Zero, Below Zero, Crossover Above Zero, Crossover Below Zero.
- Condition Type: EWO Value – operator `>`, `<`, `>=`, `<=`, `==` against numeric threshold.

## MA Z-Score (price vs MA spread)
- Z-Score Condition: operator `>`, `>=`, `<`, `<=` vs threshold on `MA_SPREAD_ZSCORE` (percent or absolute spread) using SMA/EMA/HMA, configurable MA length and mean/std windows.

## HARSI
- HARSI_FLIP: Equals 1 (green to red), Equals 2 (red to green), `> 0` (any flip).
- HARSI: Raw `HARSI(period, smoothing)[-1]`.

## OBV MACD
- OBV_MACD (Value): `OBV_MACD(...)[-1]`.
- OBV_MACD_SIGNAL (Direction): `OBV_MACD_SIGNAL(...)[-1]` (1 bullish, -1 bearish) with channel sensitivity `p`.

## SAR
- Value: `sar(acceleration, max_acceleration)[-1]`.

## SuperTrend
- Trend Direction: Uptrend, Downtrend.
- Price vs SuperTrend: Price above upper band, Price below lower band.
- Trend Change: Changed to Uptrend, Changed to Downtrend, Any change.

## Trend Magic
- Trend Direction: Bullish (CCI >= 0), Bearish (CCI < 0).
- Price vs Trend Magic: Price above, Price below, Price crossed Trend Magic.
- Trend Crossover: Buy (Low crosses above), Sell (High crosses below), Any cross.

## Ichimoku Cloud
- Price vs Cloud: Price above/below/in cloud; entered cloud from above/below/any; crossed above/below cloud.
- Line Crossovers: Conversion crosses above/below Base; Price crosses above/below Conversion; Price crosses above/below Base.
- Cloud Color: Bullish cloud, Bearish cloud, Color changed to bullish, Color changed to bearish.
- Individual Lines: Price above/below Conversion; Price above/below Base; Conversion above/below Base.
- Lagging Span: Above/below price (displaced); crossed above/below price.

## Kalman ROC Stoch
- Direction: Uptrend (white), Downtrend (blue).
- Crossovers: Bullish crossover, Bearish crossunder, Any cross.
- Levels: Above 60, Below 10, Above 50, Below 50, Between 10 and 60.
- Value: Raw `KALMAN_ROC_STOCH(...)[-1]`.

## Pivot S/R
- Alert When: Any Signal (proximity or crossover), Near Support, Near Resistance, Near Any Level, Bullish Crossover, Bearish Crossover, Any Crossover, Broke Strong Support (3+ touches), Broke Strong Resistance (3+ touches), Custom Signal Value (-3..3).

## Donchian Channels
- Channel Lines: Upper Band, Lower Band, Basis (middle), Price vs Upper, Price vs Lower, Price vs Basis.
- Channel Breakout: Upper Band Breakout, Lower Band Breakout, Basis Cross Up, Basis Cross Down.
- Channel Position: Position Value, Near Upper Band, Near Lower Band, Near Middle.
- Channel Width: Width Value, Width Expanding, Width Contracting.

## Custom
- Free-form: Any custom condition text entry.

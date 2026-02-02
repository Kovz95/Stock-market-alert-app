import talib
import numpy as np
import pandas as pd

# PIVOT_SR, PIVOT_SR_CROSSOVER, PIVOT_SR_PROXIMITY live in src.services.pivot_support_resistance
# and are imported by backend.py directly to avoid circular import (services -> backend -> indicators -> services).

# CLASSIFICATION ON BASIS OF NUMBER OF INPUTS

# INPUT FRIENDLY
def SMA(df, timeperiod, input):
    if type(input) == str and input in ["Close", "Open", "High", "Low", "Volume"]:
        return talib.SMA(df[input], timeperiod = timeperiod)
    return talib.SMA(input, timeperiod = timeperiod)

def EMA(df, timeperiod, input):
    if type(input) == str and input in ["Close", "Open", "High", "Low", "Volume"]:
        return talib.EMA(df[input], timeperiod = timeperiod)
    return talib.EMA(input, timeperiod = timeperiod)

def HMA(df, timeperiod, input):
    if type(input) == str and input in ["Close", "Open", "High", "Low", "Volume"]:
        prices = df[input]
    else:
        prices = input

    half_period = timeperiod // 2
    sqrt_period = int(np.sqrt(timeperiod))

    wma_half_period = talib.WMA(prices, timeperiod=half_period)
    wma_full_period = talib.WMA(prices, timeperiod=timeperiod)

    wma_delta = 2 * wma_half_period - wma_full_period
    hma_values = talib.WMA(wma_delta, timeperiod=sqrt_period)

    return hma_values

def FRAMA(df, length=16, FC=1, SC=198, price_type='HL2'):
    """
    Fractal Adaptive Moving Average (FRAMA)

    Args:
        df: DataFrame with OHLC data
        length: Period for fractal calculation (default 16)
        FC: Fast constant (default 1)
        SC: Slow constant (default 198)
        price_type: Price to use - 'HL2' (default), 'Close', 'Open', 'High', 'Low'

    Returns:
        Series with FRAMA values
    """
    # Get price data
    if price_type == 'HL2':
        price = (df['High'] + df['Low']) / 2
    elif price_type in ['Close', 'Open', 'High', 'Low']:
        price = df[price_type]
    else:
        price = (df['High'] + df['Low']) / 2

    # Initialize output array
    out = pd.Series(index=df.index, dtype=float)

    # Calculate constants
    len1 = length // 2
    w = np.log(2 / (SC + 1))

    # Rolling calculations
    for i in range(length, len(df)):
        # First half period
        H1 = df['High'].iloc[i-len1:i].max()
        L1 = df['Low'].iloc[i-len1:i].min()
        N1 = (H1 - L1) / len1 if len1 > 0 else 0

        # Second half period
        H2 = df['High'].iloc[i-length:i-len1].max()
        L2 = df['Low'].iloc[i-length:i-len1].min()
        N2 = (H2 - L2) / len1 if len1 > 0 else 0

        # Full period
        H3 = df['High'].iloc[i-length:i].max()
        L3 = df['Low'].iloc[i-length:i].min()
        N3 = (H3 - L3) / length if length > 0 else 0

        # Calculate fractal dimension
        if N1 > 0 and N2 > 0 and N3 > 0:
            dimen = (np.log(N1 + N2) - np.log(N3)) / np.log(2)
        else:
            # Use previous dimension if current calculation invalid
            dimen = out.iloc[i-1] if i > length and not pd.isna(out.iloc[i-1]) else 1

        # Calculate alpha
        alpha1 = np.exp(w * (dimen - 1))
        oldalpha = 1 if alpha1 > 1 else (0.01 if alpha1 < 0.01 else alpha1)

        # Calculate N
        oldN = (2 - oldalpha) / oldalpha
        N = (((SC - FC) * (oldN - 1)) / (SC - 1)) + FC

        # Calculate final alpha
        alpha_ = 2 / (N + 1)
        alpha = 2 / (SC + 1) if alpha_ < 2 / (SC + 1) else (1 if alpha_ > 1 else alpha_)

        # Calculate FRAMA
        if i == length:
            out.iloc[i] = price.iloc[i]
        else:
            out.iloc[i] = (1 - alpha) * out.iloc[i-1] + alpha * price.iloc[i]

    return out

def KAMA(df, length=21, price_type='Close', fast_end=0.666, slow_end=0.0645):
    """
    Kaufman Adaptive Moving Average (KAMA)

    Args:
        df: DataFrame with OHLC data
        length: Period for efficiency ratio calculation (default 21)
        price_type: Price to use - 'Close' (default), 'Open', 'High', 'Low', 'HL2'
        fast_end: Fast endpoint for smoothing (default 0.666)
        slow_end: Slow endpoint for smoothing (default 0.0645)

    Returns:
        Series with KAMA values
    """
    # Get price data
    if price_type == 'HL2':
        price = (df['High'] + df['Low']) / 2
    elif price_type in ['Close', 'Open', 'High', 'Low']:
        price = df[price_type]
    else:
        price = df['Close']

    # Initialize output array with NaN
    kama = pd.Series(index=df.index, dtype=float)

    # Calculate for each bar starting from length
    for i in range(length, len(df)):
        # Calculate signal (price change over period)
        signal = abs(price.iloc[i] - price.iloc[i-length])

        # Calculate noise (sum of absolute price changes)
        noise = 0
        for j in range(i-length+1, i+1):
            noise += abs(price.iloc[j] - price.iloc[j-1])

        # Calculate efficiency ratio
        if noise != 0:
            efficiency_ratio = signal / noise
        else:
            efficiency_ratio = 0

        # Calculate smoothing constant
        smooth = (efficiency_ratio * (fast_end - slow_end) + slow_end) ** 2

        # Calculate KAMA
        if i == length:
            # Initialize with simple moving average
            kama.iloc[i] = price.iloc[i-length+1:i+1].mean()
        else:
            # KAMA formula: previous KAMA + smooth * (price - previous KAMA)
            kama.iloc[i] = kama.iloc[i-1] + smooth * (price.iloc[i] - kama.iloc[i-1])

    return kama

def SLOPE_SMA(df, timeperiod, input):
    sma = SMA(df, timeperiod, input)
    return pd.Series(np.gradient(sma))

def SLOPE_EMA(df, timeperiod, input):
    ema = EMA(df, timeperiod,input)
    return pd.Series(np.gradient(ema,input))

def SLOPE_HMA(df, timeperiod, input):
    hma = HMA(df, timeperiod, input)
    return pd.Series(np.gradient(hma))

def RSI(df, timeperiod, input):
    if type(input) == str and input in ["Close", "Open", "High", "Low", "Volume"]:
        return talib.RSI(df[input], timeperiod=timeperiod)
    return talib.RSI(input, timeperiod=timeperiod)

def ROC(df, timeperiod, input):
    if type(input) == str and input in ["Close", "Open", "High", "Low", "Volume"]:
        return talib.ROC(df[input], timeperiod=timeperiod)
    return talib.ROC(input, timeperiod=timeperiod)

def EWO(df, sma1_length=5, sma2_length=35, source='Close', use_percent=True):
    """
    Elliott Wave Oscillator (EWO)

    Calculates the difference between two SMAs.
    Can show as absolute difference or as percentage of current price.

    Args:
        df: DataFrame with OHLC data
        sma1_length: Fast SMA period (default 5)
        sma2_length: Slow SMA period (default 35)
        source: Price source - 'Close', 'Open', 'High', 'Low' (default 'Close')
        use_percent: If True, show difference as percent of current price (default True)

    Returns:
        Series with EWO values
    """
    # Get source price
    if type(source) == str and source in ["Close", "Open", "High", "Low", "Volume"]:
        src = df[source]
    else:
        src = source

    # Calculate SMAs
    sma1 = talib.SMA(src, timeperiod=sma1_length)
    sma2 = talib.SMA(src, timeperiod=sma2_length)

    # Calculate difference
    sma_diff = sma1 - sma2

    # Return as percentage or absolute value
    if use_percent:
        return (sma_diff / src) * 100
    else:
        return sma_diff

def MA_SPREAD_ZSCORE(
    df,
    *,
    price_col: str = "Close",
    ma_length: int = 20,
    spread_mean_window: int | None = None,
    spread_std_window: int | None = None,
    ma_type: str = "SMA",
    spread_window: int | None = None,
    use_percent: bool = False,
) -> pd.DataFrame:
    """
    Compute the spread between price and a moving average plus rolling statistics.

    Args:
        df: DataFrame containing price data.
        price_col: Column to use for the price series (default ``Close``).
        ma_length: Lookback period for the moving average (default 20).
        spread_mean_window: Rolling window for the spread mean. Defaults to ``ma_length``.
        spread_std_window: Rolling window for the spread standard deviation.
            If ``None`` it falls back to ``spread_mean_window``.
        ma_type: Type of moving average to use (``SMA``, ``EMA`` or ``HMA``).
        spread_window: Deprecated alias that sets both ``spread_mean_window`` and
            ``spread_std_window`` when provided.
        use_percent: If ``True``, compute spread as percentage difference
            ``(price - moving_average) / moving_average * 100``. Otherwise
            compute the absolute spread (default ``False``).

    Returns:
        DataFrame with columns ``price``, ``moving_average``, ``spread``,
        ``spread_mean``, ``spread_std``, ``zscore``, ``upper_band`` and ``lower_band``.
    """

    if price_col not in df.columns:
        raise ValueError(f"Column '{price_col}' not found in dataframe")

    if spread_window is not None:
        if spread_mean_window is None:
            spread_mean_window = spread_window
        if spread_std_window is None:
            spread_std_window = spread_window

    if spread_mean_window is None:
        spread_mean_window = ma_length
    if spread_std_window is None:
        spread_std_window = spread_mean_window

    price = df[price_col].astype(float)

    ma_type_norm = ma_type.upper()
    if ma_type_norm == "EMA":
        moving_average = talib.EMA(price, timeperiod=ma_length)
    elif ma_type_norm == "HMA":
        moving_average = HMA(df, ma_length, price_col)
    else:
        moving_average = talib.SMA(price, timeperiod=ma_length)
    moving_average = pd.Series(moving_average, index=price.index, dtype=float)

    if use_percent:
        ma_safe = moving_average.replace(0, np.nan)
        spread = (price - moving_average) / ma_safe * 100
    else:
        spread = price - moving_average
    spread_mean = spread.rolling(window=spread_mean_window, min_periods=spread_mean_window).mean()
    spread_std = spread.rolling(window=spread_std_window, min_periods=spread_std_window).std(ddof=0)

    zscore = (spread - spread_mean) / spread_std

    result = pd.DataFrame(
        {
            "price": price,
            "moving_average": moving_average,
            "spread": spread,
            "spread_mean": spread_mean,
            "spread_std": spread_std,
            "zscore": zscore,
        }
    )

    result["upper_band"] = result["spread_mean"] + 2 * result["spread_std"]
    result["lower_band"] = result["spread_mean"] - 2 * result["spread_std"]

    return result

# INPUT UNFRIENDLY

def ATR(df, timeperiod):
    close = df['Close']
    high = df['High']
    low = df['Low']

    return talib.ATR(high, low, close, timeperiod = timeperiod)

def CCI(df, timeperiod):
    close = df['Close']
    high = df['High']
    low = df['Low']

    return talib.CCI(high, low, close, timeperiod = timeperiod)

def WILLR(df, timeperiod):
    return talib.WILLR(df["High"], df['Low'], df['Close'], timeperiod=timeperiod)


def HARSI_Flip(df, timeperiod, smoothing):
    """
    Heikin-Ashi RSI Flip indicator matching TradingView exactly.
    Returns: 0 = no change, 1 = green to red (sell), 2 = red to green (buy)

    Based on JayRogers HARSI Oscillator Pine Script v6
    """
    def calculate_harsi_base(df, timeperiod, smoothing):
        # Calculate zero-centered RSI (RSI - 50) for Close, High, Low
        cRSI = pd.Series(RSI(df, timeperiod, "Close"), index=df.index) - 50
        hRSIr = pd.Series(RSI(df, timeperiod, "High"), index=df.index) - 50
        lRSIr = pd.Series(RSI(df, timeperiod, "Low"), index=df.index) - 50

        # oRSI = previous close RSI (or current if first bar)
        oRSI = cRSI.shift(1).fillna(cRSI)

        # hRSI = max of high/low RSI, lRSI = min of high/low RSI
        hRSI = np.maximum(hRSIr, lRSIr)
        lRSI = np.minimum(hRSIr, lRSIr)

        # HA Close = (oRSI + hRSI + lRSI + cRSI) / 4
        close_ha = (oRSI + hRSI + lRSI + cRSI) / 4.0

        # Initialize HA open
        open_ha = pd.Series(index=df.index, dtype=float)

        # Find first valid index
        first = close_ha.first_valid_index()
        if first is None:
            return open_ha, pd.Series(np.nan, df.index), pd.Series(np.nan, df.index), close_ha

        # First bar: open = (oRSI + cRSI) / 2
        open_ha.loc[first] = (oRSI.loc[first] + cRSI.loc[first]) / 2.0

        # Subsequent bars: open = (previous_open * smoothing + previous_close) / (smoothing + 1)
        idxs = list(df.index)
        start = idxs.index(first)
        for i in range(start + 1, len(idxs)):
            prev_idx = idxs[i - 1]
            cur_idx = idxs[i]
            open_ha.loc[cur_idx] = (open_ha.loc[prev_idx] * smoothing + close_ha.loc[prev_idx]) / (smoothing + 1.0)

        # HA High = max(hRSI, open, close)
        high_ha = pd.Series(np.maximum.reduce([hRSI, open_ha, close_ha]), index=df.index)
        # HA Low = min(lRSI, open, close)
        low_ha = pd.Series(np.minimum.reduce([lRSI, open_ha, close_ha]), index=df.index)

        return open_ha, high_ha, low_ha, close_ha

    def har_si_colors(df, timeperiod, smoothing):
        o, h, l, c = calculate_harsi_base(df, timeperiod, smoothing)
        # Candle color: green if close > open, else red
        return pd.Series(np.where(c > o, "green", "red"), index=df.index)

    def color_transitions(colors):
        prev = colors.shift(1)
        result = pd.Series(0, index=colors.index)
        # 1 = green to red (bearish flip)
        result[(prev == "green") & (colors == "red")] = 1
        # 2 = red to green (bullish flip)
        result[(prev == "red") & (colors == "green")] = 2
        return result

    return color_transitions(har_si_colors(df, timeperiod, smoothing))


# MULTI INPUTTED SINGLE OUTPUT

def SAR(df, acceleration, max_acceleration):
    return talib.SAR(df['High'], df['Low'], acceleration= acceleration, maximum=max_acceleration)

# MULTI INPUT AND MULTI OUTPUT

def BBANDS(df, timeperiod, std_dev, type):
    upper, middle, lower = talib.BBANDS(df['Close'], timeperiod = timeperiod, nbdevdn= std_dev, nbdevup= std_dev, matype=0)
    if type == "upper":
        return upper
    elif type == "middle":
        return middle
    elif type == "lower":
        return lower

def MACD(df, fast_period, slow_period, signal_period, type):
    macd, macdsignal, macdhistory  = talib.MACD(df['Close'], fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)
    if type == "line":
        return macd
    elif type == "signal":
        return macdsignal




def SUPERTREND(df, period=10, multiplier=3.0, use_hl2=True, use_builtin_atr=True):
    """
    Supertrend indicator - trend following indicator based on ATR

    Parameters:
    - df: DataFrame with OHLC data
    - period: ATR period (default: 10)
    - multiplier: ATR multiplier (default: 3.0)
    - use_hl2: Use (High+Low)/2 as source, otherwise use Close (default: True)
    - use_builtin_atr: Use built-in ATR calculation (default: True)

    Returns:
    - Series with Supertrend values (trend direction: 1 for uptrend, -1 for downtrend)
    """
    high = df['High']
    low = df['Low']
    close = df['Close']

    # Source price
    src = (high + low) / 2 if use_hl2 else close

    # Calculate ATR
    if use_builtin_atr:
        # True Range
        hl = high - low
        hc = abs(high - close.shift())
        lc = abs(low - close.shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
    else:
        # Alternative ATR using SMA of True Range
        hl = high - low
        hc = abs(high - close.shift())
        lc = abs(low - close.shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

    # Calculate basic upper and lower bands
    up = src - (multiplier * atr)
    dn = src + (multiplier * atr)

    # Initialize trend
    trend = pd.Series(1, index=df.index)

    # Calculate final bands and trend
    for i in range(1, len(df)):
        # Update upper band
        if close.iloc[i-1] > up.iloc[i-1]:
            up.iloc[i] = max(up.iloc[i], up.iloc[i-1]) if not pd.isna(up.iloc[i-1]) else up.iloc[i]

        # Update lower band
        if close.iloc[i-1] < dn.iloc[i-1]:
            dn.iloc[i] = min(dn.iloc[i], dn.iloc[i-1]) if not pd.isna(dn.iloc[i-1]) else dn.iloc[i]

        # Update trend
        if trend.iloc[i-1] == -1 and close.iloc[i] > dn.iloc[i-1]:
            trend.iloc[i] = 1
        elif trend.iloc[i-1] == 1 and close.iloc[i] < up.iloc[i-1]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]

    return trend

def SUPERTREND_UPPER(df, period=10, multiplier=3.0, use_hl2=True, use_builtin_atr=True):
    """
    Supertrend upper band (support in uptrend)
    """
    high = df['High']
    low = df['Low']
    close = df['Close']

    src = (high + low) / 2 if use_hl2 else close

    if use_builtin_atr:
        hl = high - low
        hc = abs(high - close.shift())
        lc = abs(low - close.shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
    else:
        hl = high - low
        hc = abs(high - close.shift())
        lc = abs(low - close.shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

    up = src - (multiplier * atr)

    for i in range(1, len(df)):
        if close.iloc[i-1] > up.iloc[i-1]:
            up.iloc[i] = max(up.iloc[i], up.iloc[i-1]) if not pd.isna(up.iloc[i-1]) else up.iloc[i]

    # Only return upper band when in uptrend
    trend = SUPERTREND(df, period, multiplier, use_hl2, use_builtin_atr)
    return up.where(trend == 1)

def SUPERTREND_LOWER(df, period=10, multiplier=3.0, use_hl2=True, use_builtin_atr=True):
    """
    Supertrend lower band (resistance in downtrend)
    """
    high = df['High']
    low = df['Low']
    close = df['Close']

    src = (high + low) / 2 if use_hl2 else close

    if use_builtin_atr:
        hl = high - low
        hc = abs(high - close.shift())
        lc = abs(low - close.shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
    else:
        hl = high - low
        hc = abs(high - close.shift())
        lc = abs(low - close.shift())
        tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

    dn = src + (multiplier * atr)

    for i in range(1, len(df)):
        if close.iloc[i-1] < dn.iloc[i-1]:
            dn.iloc[i] = min(dn.iloc[i], dn.iloc[i-1]) if not pd.isna(dn.iloc[i-1]) else dn.iloc[i]

    # Only return lower band when in downtrend
    trend = SUPERTREND(df, period, multiplier, use_hl2, use_builtin_atr)
    return dn.where(trend == -1)

def ICHIMOKU(df, conversion_periods=9, base_periods=26, span_b_periods=52, displacement=26, visual=False):
    """
    Ichimoku Cloud indicator - comprehensive trend following system

    Parameters:
    - df: DataFrame with OHLC data
    - conversion_periods: Period for Conversion Line (Tenkan-sen) (default: 9)
    - base_periods: Period for Base Line (Kijun-sen) (default: 26)
    - span_b_periods: Period for Leading Span B (Senkou Span B) (default: 52)
    - displacement: Forward displacement for cloud (default: 26)

    Returns:
    - Dictionary with all Ichimoku components
    """
    high = df['High']
    low = df['Low']
    close = df['Close']

    # Donchian channel calculation (average of highest high and lowest low)
    def donchian(high_series, low_series, periods):
        highest = high_series.rolling(window=periods).max()
        lowest = low_series.rolling(window=periods).min()
        return (highest + lowest) / 2

    # Calculate all lines
    conversion_line = donchian(high, low, conversion_periods)  # Tenkan-sen
    base_line = donchian(high, low, base_periods)  # Kijun-sen

    # Leading Span A (Senkou Span A) - average of conversion and base
    leading_span_a = (conversion_line + base_line) / 2

    # Leading Span B (Senkou Span B) - donchian of longer period
    leading_span_b = donchian(high, low, span_b_periods)

    # Lagging Span (Chikou Span)
    lagging_span = close

    if visual:
        # Traditional plotting shifts spans forward and lagging backward.
        leading_span_a = leading_span_a.shift(displacement)
        leading_span_b = leading_span_b.shift(displacement)
        lagging_span = lagging_span.shift(-displacement)

    return {
        'conversion': conversion_line,
        'base': base_line,
        'span_a': leading_span_a,
        'span_b': leading_span_b,
        'lagging': lagging_span
    }

def ICHIMOKU_CONVERSION(df, periods=9):
    """Ichimoku Conversion Line (Tenkan-sen)"""
    high = df['High']
    low = df['Low']
    highest = high.rolling(window=periods).max()
    lowest = low.rolling(window=periods).min()
    return (highest + lowest) / 2

def ICHIMOKU_BASE(df, periods=26):
    """Ichimoku Base Line (Kijun-sen)"""
    high = df['High']
    low = df['Low']
    highest = high.rolling(window=periods).max()
    lowest = low.rolling(window=periods).min()
    return (highest + lowest) / 2

def ICHIMOKU_SPAN_A(df, conversion_periods=9, base_periods=26, displacement=26, visual=False):
    """Ichimoku Leading Span A (Senkou Span A).

    For scanning/backtesting we return the unshifted series so the latest bar is usable.
    Set visual=True to apply the traditional forward displacement for plotting.
    """
    conversion = ICHIMOKU_CONVERSION(df, conversion_periods)
    base = ICHIMOKU_BASE(df, base_periods)
    span_a = (conversion + base) / 2
    return span_a.shift(displacement) if visual else span_a

def ICHIMOKU_SPAN_B(df, periods=52, displacement=26, visual=False):
    """Ichimoku Leading Span B (Senkou Span B)"""
    high = df['High']
    low = df['Low']
    highest = high.rolling(window=periods).max()
    lowest = low.rolling(window=periods).min()
    span_b = (highest + lowest) / 2
    return span_b.shift(displacement) if visual else span_b

def ICHIMOKU_LAGGING(df, displacement=26, visual=False):
    """Ichimoku Lagging Span (Chikou Span)"""
    return df['Close'] if not visual else df['Close'].shift(-displacement)

def ICHIMOKU_CLOUD_TOP(df, conversion_periods=9, base_periods=26, span_b_periods=52, displacement=26, visual=False):
    """Returns the top of the Ichimoku cloud (max of Span A and Span B)"""
    span_a = ICHIMOKU_SPAN_A(df, conversion_periods, base_periods, displacement, visual)
    span_b = ICHIMOKU_SPAN_B(df, span_b_periods, displacement, visual)
    return pd.concat([span_a, span_b], axis=1).max(axis=1)

def ICHIMOKU_CLOUD_BOTTOM(df, conversion_periods=9, base_periods=26, span_b_periods=52, displacement=26, visual=False):
    """Returns the bottom of the Ichimoku cloud (min of Span A and Span B)"""
    span_a = ICHIMOKU_SPAN_A(df, conversion_periods, base_periods, displacement, visual)
    span_b = ICHIMOKU_SPAN_B(df, span_b_periods, displacement, visual)
    return pd.concat([span_a, span_b], axis=1).min(axis=1)

def ICHIMOKU_CLOUD_SIGNAL(df, conversion_periods=9, base_periods=26, span_b_periods=52, displacement=26, visual=False):
    """
    Returns cloud signal: 1 for bullish (green) cloud, -1 for bearish (red) cloud
    """
    span_a = ICHIMOKU_SPAN_A(df, conversion_periods, base_periods, displacement, visual)
    span_b = ICHIMOKU_SPAN_B(df, span_b_periods, displacement, visual)
    signal = pd.Series(index=df.index, dtype=int)
    signal[span_a > span_b] = 1  # Bullish cloud
    signal[span_a < span_b] = -1  # Bearish cloud
    signal[span_a == span_b] = 0  # Neutral
    return signal

def KALMAN_ROC_STOCH(df, ma_type='TEMA', lsma_off=0, smooth_len=12, kal_src='Close',
                     sharp=25.0, k_period=1.0, roc_len=9, stoch_len=14,
                     smooth_k=1, smooth_d=3):
    """
    Kalman Smoothed ROC & Stochastic indicator

    Parameters:
    - df: DataFrame with OHLC data
    - ma_type: Moving average type for smoothing ('EMA','DEMA','TEMA','WMA','VWMA','SMA','SMMA','HMA','LSMA','PEMA')
    - lsma_off: LSMA offset (default: 0)
    - smooth_len: Smoothing length (default: 12)
    - kal_src: Kalman source column (default: 'Close')
    - sharp: Sharpness for Kalman filter (default: 25.0)
    - k_period: Filter period for Kalman (default: 1.0)
    - roc_len: ROC length (default: 9)
    - stoch_len: Stochastic %K length (default: 14)
    - smooth_k: %K smoothing (default: 1)
    - smooth_d: %D smoothing (default: 3)

    Returns:
    - Series with blended indicator values
    """
    high = df['High']
    low = df['Low']
    close = df['Close']
    volume = df['Volume'] if 'Volume' in df.columns else None

    # Get source price for Kalman
    if kal_src == 'Close':
        src = close
    elif kal_src == 'Open':
        src = df['Open']
    elif kal_src == 'High':
        src = high
    elif kal_src == 'Low':
        src = low
    elif kal_src == 'HL2':
        src = (high + low) / 2
    elif kal_src == 'HLC3':
        src = (high + low + close) / 3
    elif kal_src == 'OHLC4':
        src = (df['Open'] + high + low + close) / 4
    else:
        src = close

    # Kalman Filter implementation
    kfilt = pd.Series(index=df.index, dtype=float)
    vel = pd.Series(0.0, index=df.index)

    # Initialize first value
    kfilt.iloc[0] = src.iloc[0]
    vel.iloc[0] = 0.0

    # Calculate Kalman filter
    for i in range(1, len(df)):
        dist = src.iloc[i] - kfilt.iloc[i-1]
        err = kfilt.iloc[i-1] + dist * np.sqrt(sharp * k_period / 100)
        vel.iloc[i] = vel.iloc[i-1] + dist * (k_period / 100)
        kfilt.iloc[i] = err + vel.iloc[i]

    # Calculate ROC on Kalman filtered values
    roc = 100 * (kfilt - kfilt.shift(roc_len)) / kfilt.shift(roc_len)

    # Calculate Stochastic
    lowest_low = low.rolling(window=stoch_len).min()
    highest_high = high.rolling(window=stoch_len).max()
    k_raw = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k_sma = k_raw.rolling(window=smooth_k).mean()
    d_sma = k_sma.rolling(window=smooth_d).mean()

    # Blend ROC and Stochastic D
    blend_raw = (roc + d_sma) / 2

    # Apply MA smoothing based on type
    ma_type = ma_type.upper()
    if ma_type == 'SMA':
        blend = blend_raw.rolling(window=smooth_len).mean()
    elif ma_type == 'EMA':
        blend = blend_raw.ewm(span=smooth_len, adjust=False).mean()
    elif ma_type == 'DEMA':
        ema1 = blend_raw.ewm(span=smooth_len, adjust=False).mean()
        ema2 = ema1.ewm(span=smooth_len, adjust=False).mean()
        blend = 2 * ema1 - ema2
    elif ma_type == 'TEMA':
        ema1 = blend_raw.ewm(span=smooth_len, adjust=False).mean()
        ema2 = ema1.ewm(span=smooth_len, adjust=False).mean()
        ema3 = ema2.ewm(span=smooth_len, adjust=False).mean()
        blend = 3 * (ema1 - ema2) + ema3
    elif ma_type == 'WMA':
        weights = np.arange(1, smooth_len + 1)
        blend = blend_raw.rolling(window=smooth_len).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)
    elif ma_type == 'VWMA' and volume is not None:
        blend = (blend_raw * volume).rolling(window=smooth_len).sum() / volume.rolling(window=smooth_len).sum()
    elif ma_type == 'SMMA':
        # Smoothed Moving Average (SMMA)
        blend = pd.Series(index=df.index, dtype=float)
        blend.iloc[:smooth_len] = blend_raw.iloc[:smooth_len].mean()
        for i in range(smooth_len, len(df)):
            blend.iloc[i] = (blend.iloc[i-1] * (smooth_len - 1) + blend_raw.iloc[i]) / smooth_len
    elif ma_type == 'HMA':
        # Hull Moving Average
        half_len = int(smooth_len / 2)
        sqrt_len = int(np.sqrt(smooth_len))
        wma_half = blend_raw.rolling(window=half_len).apply(lambda x: np.dot(x, np.arange(1, half_len + 1)) / np.arange(1, half_len + 1).sum(), raw=True)
        wma_full = blend_raw.rolling(window=smooth_len).apply(lambda x: np.dot(x, np.arange(1, smooth_len + 1)) / np.arange(1, smooth_len + 1).sum(), raw=True)
        raw_hma = 2 * wma_half - wma_full
        blend = raw_hma.rolling(window=sqrt_len).apply(lambda x: np.dot(x, np.arange(1, sqrt_len + 1)) / np.arange(1, sqrt_len + 1).sum(), raw=True)
    elif ma_type == 'LSMA':
        # Linear Regression (Least Squares Moving Average)
        blend = blend_raw.rolling(window=smooth_len).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] * (len(x) - 1 - lsma_off) + np.polyfit(np.arange(len(x)), x, 1)[1],
            raw=True
        )
    elif ma_type == 'PEMA':
        # Pentuple Exponential Moving Average
        ema1 = blend_raw.ewm(span=smooth_len, adjust=False).mean()
        ema2 = ema1.ewm(span=smooth_len, adjust=False).mean()
        ema3 = ema2.ewm(span=smooth_len, adjust=False).mean()
        ema4 = ema3.ewm(span=smooth_len, adjust=False).mean()
        ema5 = ema4.ewm(span=smooth_len, adjust=False).mean()
        ema6 = ema5.ewm(span=smooth_len, adjust=False).mean()
        ema7 = ema6.ewm(span=smooth_len, adjust=False).mean()
        ema8 = ema7.ewm(span=smooth_len, adjust=False).mean()
        blend = 8*ema1 - 28*ema2 + 56*ema3 - 70*ema4 + 56*ema5 - 28*ema6 + 8*ema7 - ema8
    else:
        # Default to SMA if unknown type
        blend = blend_raw.rolling(window=smooth_len).mean()

    return blend

def KALMAN_ROC_STOCH_SIGNAL(df, ma_type='TEMA', lsma_off=0, smooth_len=12, kal_src='Close',
                            sharp=25.0, k_period=1.0, roc_len=9, stoch_len=14,
                            smooth_k=1, smooth_d=3):
    """
    Returns signal based on Kalman ROC Stochastic direction
    1 = uptrend (white), -1 = downtrend (blue), 0 = neutral
    """
    blend = KALMAN_ROC_STOCH(df, ma_type, lsma_off, smooth_len, kal_src,
                             sharp, k_period, roc_len, stoch_len, smooth_k, smooth_d)

    signal = pd.Series(0, index=df.index, dtype=int)
    signal[blend > blend.shift(1)] = 1   # Uptrend (white)
    signal[blend < blend.shift(1)] = -1  # Downtrend (blue)
    signal[blend == blend.shift(1)] = 0  # No change

    return signal

def KALMAN_ROC_STOCH_CROSSOVER(df, ma_type='TEMA', lsma_off=0, smooth_len=12, kal_src='Close',
                               sharp=25.0, k_period=1.0, roc_len=9, stoch_len=14,
                               smooth_k=1, smooth_d=3):
    """
    Returns crossover signals
    1 = bullish crossover (buy), -1 = bearish crossunder (sell), 0 = no signal
    """
    blend = KALMAN_ROC_STOCH(df, ma_type, lsma_off, smooth_len, kal_src,
                             sharp, k_period, roc_len, stoch_len, smooth_k, smooth_d)

    signal = pd.Series(0, index=df.index, dtype=int)

    # Bullish crossover
    signal[(blend > blend.shift(1)) & (blend.shift(1) <= blend.shift(2))] = 1

    # Bearish crossunder
    signal[(blend < blend.shift(1)) & (blend.shift(1) >= blend.shift(2))] = -1

    return signal

def CCI(df, period=20, price_type='Close'):
    """
    Commodity Channel Index (CCI)

    Parameters:
    - df: DataFrame with OHLC data
    - period: Period for CCI calculation (default: 20)
    - price_type: Price to use (default: 'Close')

    Returns:
    - Series with CCI values
    """
    if price_type == 'Close':
        price = df['Close']
    elif price_type == 'HLC':
        price = (df['High'] + df['Low'] + df['Close']) / 3
    else:
        price = df[price_type]

    sma = price.rolling(window=period).mean()
    mean_deviation = (price - sma).abs().rolling(window=period).mean()
    cci = (price - sma) / (0.015 * mean_deviation)

    return cci

def TREND_MAGIC(df, cci_period=20, atr_multiplier=1.0, atr_period=5, price_type='Close'):
    """
    Trend Magic indicator - combines CCI and ATR for trend detection

    Parameters:
    - df: DataFrame with OHLC data
    - cci_period: Period for CCI calculation (default: 20)
    - atr_multiplier: Multiplier for ATR bands (default: 1.0)
    - atr_period: Period for ATR calculation (default: 5)
    - price_type: Price type for CCI (default: 'Close')

    Returns:
    - Series with Trend Magic line values
    """
    high = df['High']
    low = df['Low']
    close = df['Close']

    # Calculate ATR using SMA
    hl = high - low
    hc = abs(high - close.shift())
    lc = abs(low - close.shift())
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_period).mean()

    # Calculate CCI
    cci = CCI(df, cci_period, price_type)

    # Calculate upper and lower bands
    up_t = low - atr * atr_multiplier
    down_t = high + atr * atr_multiplier

    # Initialize MagicTrend
    magic_trend = pd.Series(index=df.index, dtype=float)

    # Calculate MagicTrend
    for i in range(len(df)):
        if i == 0:
            magic_trend.iloc[i] = up_t.iloc[i] if cci.iloc[i] >= 0 else down_t.iloc[i]
        else:
            if cci.iloc[i] >= 0:
                # Bullish - use upper band (support)
                if pd.notna(magic_trend.iloc[i-1]) and up_t.iloc[i] < magic_trend.iloc[i-1]:
                    magic_trend.iloc[i] = magic_trend.iloc[i-1]
                else:
                    magic_trend.iloc[i] = up_t.iloc[i]
            else:
                # Bearish - use lower band (resistance)
                if pd.notna(magic_trend.iloc[i-1]) and down_t.iloc[i] > magic_trend.iloc[i-1]:
                    magic_trend.iloc[i] = magic_trend.iloc[i-1]
                else:
                    magic_trend.iloc[i] = down_t.iloc[i]

    return magic_trend

def TREND_MAGIC_SIGNAL(df, cci_period=20, atr_multiplier=1.0, atr_period=5, price_type='Close'):
    """
    Trend Magic signal - returns 1 for bullish, -1 for bearish

    Parameters:
    - df: DataFrame with OHLC data
    - cci_period: Period for CCI calculation (default: 20)
    - atr_multiplier: Multiplier for ATR bands (default: 1.0)
    - atr_period: Period for ATR calculation (default: 5)
    - price_type: Price type for CCI (default: 'Close')

    Returns:
    - Series with signal values (1 for bullish, -1 for bearish)
    """
    cci = CCI(df, cci_period, price_type)
    signal = pd.Series(index=df.index, dtype=int)
    signal[cci >= 0] = 1
    signal[cci < 0] = -1
    return signal

def DONCHIAN_UPPER(df, length=20, offset=0):
    """
    Donchian Channels Upper Band - Highest high over the period

    Parameters:
    - df: DataFrame with OHLC data
    - length: Period for highest high calculation (default: 20)
    - offset: Offset for shifting the channel (default: 0)

    Returns:
    - Series with upper band values
    """
    upper = df['High'].rolling(window=length).max()
    if offset != 0:
        upper = upper.shift(offset)
    return upper

def DONCHIAN_LOWER(df, length=20, offset=0):
    """
    Donchian Channels Lower Band - Lowest low over the period

    Parameters:
    - df: DataFrame with OHLC data
    - length: Period for lowest low calculation (default: 20)
    - offset: Offset for shifting the channel (default: 0)

    Returns:
    - Series with lower band values
    """
    lower = df['Low'].rolling(window=length).min()
    if offset != 0:
        lower = lower.shift(offset)
    return lower

def DONCHIAN_BASIS(df, length=20, offset=0):
    """
    Donchian Channels Basis (Middle Line) - Average of upper and lower bands

    Parameters:
    - df: DataFrame with OHLC data
    - length: Period for channel calculation (default: 20)
    - offset: Offset for shifting the channel (default: 0)

    Returns:
    - Series with basis (middle) line values
    """
    upper = DONCHIAN_UPPER(df, length, offset=0)  # Don't apply offset yet
    lower = DONCHIAN_LOWER(df, length, offset=0)  # Don't apply offset yet
    basis = (upper + lower) / 2
    if offset != 0:
        basis = basis.shift(offset)
    return basis

def DONCHIAN_WIDTH(df, length=20):
    """
    Donchian Channels Width - Distance between upper and lower bands

    Parameters:
    - df: DataFrame with OHLC data
    - length: Period for channel calculation (default: 20)

    Returns:
    - Series with channel width values
    """
    upper = DONCHIAN_UPPER(df, length, offset=0)
    lower = DONCHIAN_LOWER(df, length, offset=0)
    return upper - lower

def DONCHIAN_POSITION(df, length=20):
    """
    Price position within Donchian Channels (0 to 1, where 0 is at lower band, 1 is at upper band)

    Parameters:
    - df: DataFrame with OHLC data
    - length: Period for channel calculation (default: 20)

    Returns:
    - Series with position values (0 to 1)
    """
    upper = DONCHIAN_UPPER(df, length, offset=0)
    lower = DONCHIAN_LOWER(df, length, offset=0)
    width = upper - lower

    # Avoid division by zero
    position = pd.Series(index=df.index, dtype=float)
    mask = width != 0
    position[mask] = (df['Close'][mask] - lower[mask]) / width[mask]
    position[~mask] = 0.5  # Default to middle if width is 0

    return position

def OBV_MACD(df, window_len=28, v_len=14, obv_len=1, ma_type='DEMA', ma_len=9, slow_len=26, slope_len=2):
    """
    OBV MACD Indicator - MACD based on OBV with various MA types

    Parameters:
    - df: DataFrame with OHLC and Volume data
    - window_len: Window length for price/volume spread (default: 28)
    - v_len: Volume smoothing length (default: 14)
    - obv_len: OBV EMA length (default: 1)
    - ma_type: Moving average type (default: 'DEMA')
                Options: 'TDEMA', 'TTEMA', 'TEMA', 'DEMA', 'EMA', 'AVG', 'THMA', 'ZLEMA', 'ZLDEMA', 'ZLTEMA', 'DZLEMA', 'TZLEMA', 'LLEMA', 'NMA'
    - ma_len: MA length (default: 9)
    - slow_len: MACD slow length (default: 26)
    - slope_len: Slope calculation length (default: 2)

    Returns:
    - Series with OBV MACD values
    """

    # Helper functions for various MA types
    def dema(src, length):
        ma1 = talib.EMA(src, timeperiod=length)
        ma2 = talib.EMA(ma1, timeperiod=length)
        return 2 * ma1 - ma2

    def tema(src, length):
        ma1 = talib.EMA(src, timeperiod=length)
        ma2 = talib.EMA(ma1, timeperiod=length)
        ma3 = talib.EMA(ma2, timeperiod=length)
        return 3 * (ma1 - ma2) + ma3

    def tdema(src, length):
        ma1 = dema(src, length)
        ma2 = dema(ma1, length)
        ma3 = dema(ma2, length)
        return 3 * (ma1 - ma2) + ma3

    def ttema(src, length):
        ma1 = tema(src, length)
        ma2 = tema(ma1, length)
        ma3 = tema(ma2, length)
        return 3 * (ma1 - ma2) + ma3

    def thma(src, length):
        hma1 = HMA(df, length, src)
        hma2 = HMA(df, length, hma1)
        hma3 = HMA(df, length, hma2)
        return 3 * (hma1 - hma2) + hma3

    def zlema(src, length):
        lag = int((length - 1) / 2)
        zlsrc = src + (src - src.shift(lag))
        return talib.EMA(zlsrc, timeperiod=length)

    def zldema(src, length):
        lag = int((length - 1) / 2)
        zlsrc = src + (src - src.shift(lag))
        return dema(zlsrc, length)

    def zltema(src, length):
        lag = int((length - 1) / 2)
        zlsrc = src + (src - src.shift(lag))
        return tema(zlsrc, length)

    def dzlema(src, length):
        ma1 = zlema(src, length)
        ma2 = zlema(ma1, length)
        return 2 * ma1 - ma2

    def tzlema(src, length):
        ma1 = zlema(src, length)
        ma2 = zlema(ma1, length)
        ma3 = zlema(ma2, length)
        return 3 * (ma1 - ma2) + ma3

    def llema(src, length):
        srcnew = 0.25 * src + 0.5 * src.shift(1) + 0.25 * src.shift(2)
        return talib.EMA(srcnew, timeperiod=length)

    def nma(src, length1, length2):
        lambda_val = length1 / length2
        alpha = lambda_val * (length1 - 1) / (length1 - lambda_val)
        ma1 = talib.EMA(src, timeperiod=length1)
        ma2 = talib.EMA(ma1, timeperiod=length2)
        return (1 + alpha) * ma1 - alpha * ma2

    # Calculate OBV-based indicator
    price_spread = df['High'].sub(df['Low']).rolling(window=window_len).std()

    # Calculate OBV
    close_change = df['Close'].diff()
    sign = np.sign(close_change)
    v = (sign * df['Volume']).cumsum()

    # Smooth OBV
    smooth = talib.SMA(v.values, timeperiod=v_len)
    v_spread = (v - smooth).rolling(window=window_len).std()

    # Calculate shadow
    shadow = pd.Series(index=df.index, dtype=float)
    mask = v_spread != 0
    shadow[mask] = (v[mask] - smooth[mask]) / v_spread[mask] * price_spread[mask]
    shadow[~mask] = 0

    # Calculate out
    out = pd.Series(index=df.index, dtype=float)
    out = np.where(shadow > 0, df['High'] + shadow, df['Low'] + shadow)
    out = pd.Series(out, index=df.index)

    # Apply OBV EMA
    obvema = talib.EMA(out.values, timeperiod=obv_len)
    obvema = pd.Series(obvema, index=df.index)

    # Apply selected MA type
    if ma_type == 'EMA':
        ma = talib.EMA(obvema.values, timeperiod=ma_len)
    elif ma_type == 'DEMA':
        ma = dema(obvema, ma_len)
    elif ma_type == 'TEMA':
        ma = tema(obvema, ma_len)
    elif ma_type == 'TDEMA':
        ma = tdema(obvema, ma_len)
    elif ma_type == 'TTEMA':
        ma = ttema(obvema, ma_len)
    elif ma_type == 'THMA':
        ma = thma(obvema, ma_len)
    elif ma_type == 'ZLEMA':
        ma = zlema(obvema, ma_len)
    elif ma_type == 'ZLDEMA':
        ma = zldema(obvema, ma_len)
    elif ma_type == 'ZLTEMA':
        ma = zltema(obvema, ma_len)
    elif ma_type == 'DZLEMA':
        ma = dzlema(obvema, ma_len)
    elif ma_type == 'TZLEMA':
        ma = tzlema(obvema, ma_len)
    elif ma_type == 'LLEMA':
        ma = llema(obvema, ma_len)
    elif ma_type == 'NMA':
        ma = nma(obvema, ma_len, 26)
    else:  # AVG
        ma = (ttema(obvema, ma_len) + tdema(obvema, ma_len)) / 2

    ma = pd.Series(ma, index=df.index) if isinstance(ma, np.ndarray) else ma

    # Calculate MACD
    slow_ma = talib.EMA(df['Close'].values, timeperiod=slow_len)
    macd = ma - slow_ma

    # Calculate slope
    def calc_slope(src, length):
        slope = pd.Series(index=src.index, dtype=float)
        for i in range(length, len(src)):
            sumX = sum(range(1, length + 1))
            sumY = sum(src.iloc[i - length + 1:i + 1])
            sumXSqr = sum([(j + 1) ** 2 for j in range(length)])
            sumXY = sum([(j + 1) * src.iloc[i - length + j + 1] for j in range(length)])

            slope_val = (length * sumXY - sumX * sumY) / (length * sumXSqr - sumX * sumX)
            average = sumY / length
            intercept = average - slope_val * sumX / length + slope_val

            slope.iloc[i] = intercept + slope_val * (length - 0)  # offset=0

        return slope

    trend_line = calc_slope(macd, slope_len)

    return trend_line

def OBV_MACD_SIGNAL(df, window_len=28, v_len=14, obv_len=1, ma_type='DEMA', ma_len=9, slow_len=26, slope_len=2, p=1):
    """
    OBV MACD Signal - Returns trend channel signal (1 for bullish, -1 for bearish)

    Parameters: Same as OBV_MACD plus:
    - p: Channel sensitivity parameter (default: 1)

    Returns:
    - Series with signals: 1 (bullish), -1 (bearish), 0 (neutral)
    """

    # Get OBV MACD trend line
    src = OBV_MACD(df, window_len, v_len, obv_len, ma_type, ma_len, slow_len, slope_len)

    # T-Channels calculation (from the Pine Script)
    b = pd.Series(0.0, index=df.index)
    dev = pd.Series(0.0, index=df.index)
    oc = pd.Series(0, index=df.index)

    n = np.arange(len(df))

    for i in range(1, len(df)):
        # Calculate adaptive threshold
        a = abs(src.iloc[:i+1] - b.iloc[i-1]).sum() / (i) * p if i > 0 else 0

        # Update trend line
        if src.iloc[i] > b.iloc[i-1] + a:
            b.iloc[i] = src.iloc[i]
        elif src.iloc[i] < b.iloc[i-1] - a:
            b.iloc[i] = src.iloc[i]
        else:
            b.iloc[i] = b.iloc[i-1]

        # Update deviation
        if b.iloc[i] != b.iloc[i-1]:
            dev.iloc[i] = a
        else:
            dev.iloc[i] = dev.iloc[i-1]

        # Update trend direction
        if b.iloc[i] > b.iloc[i-1]:
            oc.iloc[i] = 1
        elif b.iloc[i] < b.iloc[i-1]:
            oc.iloc[i] = -1
        else:
            oc.iloc[i] = oc.iloc[i-1]

    return oc

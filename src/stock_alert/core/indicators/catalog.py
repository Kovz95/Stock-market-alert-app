"""Technical indicator catalog and configurations."""

import operator

from indicators_lib import (
    ATR,
    BBANDS,
    CCI,
    EMA,
    EWO,
    FRAMA,
    HMA,
    KAMA,
    MA_SPREAD_ZSCORE,
    MACD,
    ROC,
    RSI,
    SAR,
    SLOPE_EMA,
    SLOPE_HMA,
    SLOPE_SMA,
    SMA,
    WILLR,
    HARSI_Flip,
)

# Predefined suggestions for technical indicators (Single timeframe mode)
predefined_suggestions = [
    "sma(period = )[-1]",
    "hma(period = )[-1]",
    "rsi(period = )[-1]",
    "ema(period = )[-1]",
    "slope_sma(period = )[-1]",
    "slope_ema(period = )[-1]",
    "slope_hma(period = )[-1]",
    "bb(period = , std_dev = , type = )[-1]",
    "macd(fast_period = , slow_period = , signal_period = , type = )[-1]",
    "breakout",
    "atr(period = )[-1]",
    "cci(period = )[-1]",
    "roc(period = )[-1]",
    "williamsr(period = )[-1]",
    "Close[-1]",
    "Open[-1]",
    "Low[-1]",
    "High[-1]",
    "HARSI_Flip(period = , smoothing = )[-1]",
    "SROCST(ma_type = EMA, lsma_offset = 0, smoothing_length = 12, kalman_src = Close, sharpness = 25, filter_period = 1, roc_length = 9, k_length = 14, k_smoothing = 1, d_smoothing = 3)[-1]",
]

# Predefined suggestions for multiple timeframes mode
predefined_suggestions_alt = [
    "sma(period = ,timeframe = )[-1]",
    "hma(period = ,timeframe = )[-1]",
    "rsi(period = ,timeframe = )[-1]",
    "ema(period = ,timeframe = )[-1]",
    "slope_sma(period = ,timeframe = )[-1]",
    "slope_ema(period = ,timeframe = )[-1]",
    "slope_hma(period = ,timeframe = )[-1]",
    "bb(period = , std_dev = , type = ,timeframe = )[-1]",
    "macd(fast_period = , slow_period = , signal_period = , type = ,timeframe = )[-1]",
    "Breakout",
    "atr(period = ,timeframe = )[-1]",
    "cci(period = ,timeframe = )[-1]",
    "roc(period = ,timeframe = )[-1]",
    "WilliamSR(period = ,timeframe = )[-1]",
    "psar(acceleration = , max_acceleration = ,timeframe = )[-1]",
    "Close(timeframe = )[-1]",
    "Open(timeframe = )[-1]",
    "Low(timeframe = )[-1]",
    "High(timeframe = )[-1]",
]

# Inverse operator mapping
inverse_map = {">": "<=", "<": ">=", "==": "!=", "!=": "==", ">=": "<", "<=": ">"}

# Operator functions
ops = {
    ">": operator.gt,
    "<": operator.lt,
    "==": operator.eq,
    ">=": operator.ge,
    "<=": operator.le,
    "!=": operator.ne,
}

# Supported indicator functions
supported_indicators = {
    "sma": SMA,
    "ema": EMA,
    "hma": HMA,
    "frama": FRAMA,
    "kama": KAMA,
    "ewo": EWO,
    "ma_spread_zscore": MA_SPREAD_ZSCORE,
    "slope_sma": SLOPE_SMA,
    "slope_ema": SLOPE_EMA,
    "slope_hma": SLOPE_HMA,
    "rsi": RSI,
    "atr": ATR,
    "cci": CCI,
    "bb": BBANDS,
    "roc": ROC,
    "williamsr": WILLR,
    "macd": MACD,
    "psar": SAR,
    "HARSI_Flip": HARSI_Flip,
}

# USED IN APPLY FUNCTION ONLY
period_and_input = ["sma", "ema", "rsi", "hma", "slope_sma", "slope_ema", "slope_hma", "roc"]

period_only = [
    "sma",
    "ema",
    "rsi",
    "hma",
    "slope_sma",
    "slope_ema",
    "slope_hma",
    "roc",
    "atr",
    "cci",
    "willr",
    "bbands",
]

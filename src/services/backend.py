"""
Backend module for evaluating technical indicator expressions.

This module provides the core functionality for:
- Parsing indicator expressions (ind_to_dict)
- Calculating indicator values (apply_function, indicator_calculation)
- Evaluating conditional expressions (evaluate_expression, evaluate_expression_list)
- Parsing condition strings (simplify_conditions)
"""

from __future__ import annotations

import re
import ast
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import talib

from src.utils.indicators import (
    SMA,
    EMA,
    HMA,
    FRAMA,
    KAMA,
    RSI,
    ROC,
    ATR,
    CCI,
    WILLR,
    EWO,
    MA_SPREAD_ZSCORE,
    MACD,
    BBANDS,
    SAR,
    HARSI_Flip,
    SUPERTREND,
    SUPERTREND_UPPER,
    SUPERTREND_LOWER,
    ICHIMOKU_CLOUD_TOP,
    ICHIMOKU_CLOUD_BOTTOM,
    ICHIMOKU_CLOUD_SIGNAL,
    ICHIMOKU_CONVERSION,
    ICHIMOKU_BASE,
    ICHIMOKU_LAGGING,
    SLOPE_SMA,
    SLOPE_EMA,
    SLOPE_HMA,
    DONCHIAN_UPPER,
    DONCHIAN_LOWER,
    DONCHIAN_BASIS,
    DONCHIAN_WIDTH,
    DONCHIAN_POSITION,
    TREND_MAGIC,
    TREND_MAGIC_SIGNAL,
    KALMAN_ROC_STOCH,
    KALMAN_ROC_STOCH_SIGNAL,
    OBV_MACD,
    OBV_MACD_SIGNAL,
    PIVOT_SR,
    PIVOT_SR_CROSSOVER,
    PIVOT_SR_PROXIMITY,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Classes
# =============================================================================

class _SeriesIndexer:
    """Safe Series wrapper that supports Python-style negative indexing."""

    def __init__(self, series: pd.Series):
        self.series = series

    def __getitem__(self, idx):
        try:
            if isinstance(idx, int):
                return self._get_by_position(idx)
            return self.series[idx]
        except Exception:
            return np.nan

    def _get_by_position(self, pos: int):
        try:
            return self.series.iloc[pos]
        except Exception:
            return np.nan

    def __getattr__(self, item):
        return getattr(self.series, item)

    def __array__(self, *args, **kwargs):
        return np.array(self.series, *args, **kwargs)

    def __len__(self):
        return len(self.series)


# =============================================================================
# Parsing Functions
# =============================================================================

def ind_to_dict(ind: str, debug_mode: bool = False) -> Optional[Dict[str, Any]]:
    """
    Parse an indicator expression string into a dictionary format.

    Handles:
    - Numeric values: "150", "-2.0", "3.14"
    - Indicators with brackets: "Close[-1]", "rsi(14)[-1]"
    - Simple indicator names: "Close", "Open"
    - Indicators with kwargs: "EWO(sma1_length=5, sma2_length=35)"

    Args:
        ind: The indicator expression string
        debug_mode: Enable debug output

    Returns:
        dict: Parsed indicator structure with keys like:
            - isNum: bool (True if numeric value)
            - number: value (if numeric)
            - ind: indicator name
            - specifier: index (like -1 from [-1])
            - period: period value (like 14 from rsi(14))
            - operable: bool (True if can be used in calculations)
            - Additional indicator-specific parameters
    """
    if not ind or not isinstance(ind, str):
        return None

    ind = ind.strip()

    # Check if it's a numeric value
    try:
        num_val = float(ind)
        if num_val.is_integer():
            num_val = int(num_val)
        return {
            "isNum": True,
            "number": num_val,
            "operable": True,
            "specifier": -1,
        }
    except ValueError:
        pass

    result = {
        "isNum": False,
        "operable": True,
        "specifier": -1,
    }

    # Extract bracket specifier like [-1] at the end
    bracket_match = re.search(r'\[(-?\d+)\]\s*$', ind)
    if bracket_match:
        result["specifier"] = int(bracket_match.group(1))
        ind = re.sub(r'\[(-?\d+)\]\s*$', '', ind).strip()

    # Extract function parameters like rsi(14) or EWO(sma1_length=5, ...)
    func_match = re.match(r'(\w+)\s*\((.*)\)\s*$', ind, re.DOTALL)
    if func_match:
        func_name = func_match.group(1)
        params_str = func_match.group(2).strip()
        result["ind"] = func_name

        # Parse parameters
        if params_str:
            params = _parse_params(params_str)
            result.update(params)
    else:
        # Simple column name like "Close", "Open", etc.
        result["ind"] = ind

    if debug_mode:
        logger.debug(f"ind_to_dict: '{ind}' -> {result}")

    return result


def _parse_params(params_str: str) -> Dict[str, Any]:
    """Parse function parameters from a string."""
    params = {}

    # Try simple single numeric parameter first
    params_str = params_str.strip()
    if params_str and '=' not in params_str and ',' not in params_str:
        try:
            if '.' in params_str:
                params["period"] = float(params_str)
            else:
                params["period"] = int(params_str)
            return params
        except ValueError:
            pass

    # Parse key=value pairs
    # Handle nested parentheses and quoted strings
    current_param = ""
    paren_depth = 0
    in_quotes = False
    quote_char = None

    for char in params_str:
        if char in ('"', "'") and (not in_quotes or quote_char == char):
            in_quotes = not in_quotes
            quote_char = char if in_quotes else None
            current_param += char
        elif char == '(' and not in_quotes:
            paren_depth += 1
            current_param += char
        elif char == ')' and not in_quotes:
            paren_depth -= 1
            current_param += char
        elif char == ',' and paren_depth == 0 and not in_quotes:
            _process_param(current_param.strip(), params)
            current_param = ""
        else:
            current_param += char

    # Process last parameter
    if current_param.strip():
        _process_param(current_param.strip(), params)

    return params


def _process_param(param: str, params: Dict[str, Any]) -> None:
    """Process a single parameter and add to params dict."""
    if not param:
        return

    if '=' in param:
        key, val = param.split('=', 1)
        key = key.strip()
        val = val.strip()

        # Try to evaluate the value
        try:
            params[key] = ast.literal_eval(val)
        except Exception:
            # Remove quotes if present
            if (val.startswith('"') and val.endswith('"')) or \
               (val.startswith("'") and val.endswith("'")):
                val = val[1:-1]
            params[key] = val
    else:
        # Positional parameter - treat as period
        try:
            if '.' in param:
                params["period"] = float(param)
            else:
                params["period"] = int(param)
        except ValueError:
            params["input"] = param.strip("'\"")


def simplify_conditions(cond: str) -> Optional[Dict[str, Any]]:
    """
    Parse a condition string into a structured dictionary.

    Handles simple binary comparisons like:
    - "Close[-1] > sma(20)[-1]"
    - "rsi(14)[-1] < 30"
    - "Close[-1] > 150"

    Args:
        cond: The condition string

    Returns:
        dict: {
            "ind1": parsed left side (indicator dict),
            "ind2": parsed right side (indicator dict),
            "comparison": operator string (">", "<", ">=", "<=", "==", "!=")
        }
    """
    if not cond or not isinstance(cond, str):
        return None

    cond = cond.strip()

    # Find comparison operators (check >= and <= before > and <)
    operators = [">=", "<=", "==", "!=", ">", "<"]
    operator = None
    left_part = None
    right_part = None

    for op in operators:
        # Find the operator position, being careful of nested parentheses
        idx = _find_operator_position(cond, op)
        if idx >= 0:
            operator = op
            left_part = cond[:idx].strip()
            right_part = cond[idx + len(op):].strip()
            break

    if not operator or left_part is None or right_part is None:
        return None

    # Parse left and right sides
    try:
        ind1 = ind_to_dict(left_part, debug_mode=False)
    except Exception:
        ind1 = None

    try:
        ind2 = ind_to_dict(right_part, debug_mode=False)
    except Exception:
        ind2 = None

    return {
        "ind1": ind1,
        "ind2": ind2,
        "comparison": operator
    }


def _find_operator_position(expr: str, op: str) -> int:
    """Find the position of an operator outside of parentheses."""
    paren_depth = 0
    bracket_depth = 0

    for i in range(len(expr) - len(op) + 1):
        char = expr[i]
        if char == '(':
            paren_depth += 1
        elif char == ')':
            paren_depth -= 1
        elif char == '[':
            bracket_depth += 1
        elif char == ']':
            bracket_depth -= 1
        elif paren_depth == 0 and bracket_depth == 0:
            if expr[i:i+len(op)] == op:
                return i

    return -1


# =============================================================================
# Indicator Calculation Functions
# =============================================================================

def apply_function(
    df: pd.DataFrame,
    ind: Dict[str, Any],
    vals: Optional[Dict[str, Any]] = None,
    debug_mode: bool = False
) -> Any:
    """
    Apply an indicator function to a DataFrame.

    Args:
        df: DataFrame with OHLCV data
        ind: Indicator dictionary from ind_to_dict
        vals: Optional values dictionary (for variable substitution)
        debug_mode: Enable debug output

    Returns:
        Scalar value or Series depending on whether specifier is provided
    """
    if not ind or not isinstance(ind, dict):
        return None

    # Handle numeric values
    if ind.get("isNum"):
        return ind.get("number")

    func = ind.get("ind", "")
    if not func:
        return None

    func_lower = func.lower()
    specifier = ind.get("specifier")

    # Calculate the indicator series
    series = _calculate_indicator(df, ind, func_lower, debug_mode)

    if series is None:
        return None

    # Apply specifier if provided
    if specifier is not None and specifier != -1:
        return _apply_specifier(series, specifier)

    return series


def _calculate_indicator(
    df: pd.DataFrame,
    ind: Dict[str, Any],
    func_lower: str,
    debug_mode: bool = False
) -> Optional[Union[pd.Series, pd.DataFrame, float]]:
    """Calculate an indicator series from the indicator dictionary."""

    # Get common parameters
    period = _as_int(ind.get("period"), 20)
    input_col = ind.get("input", "Close")
    if isinstance(input_col, str):
        input_col = input_col.strip("'\"")

    # Price columns
    if func_lower in ("close", "open", "high", "low", "volume"):
        col_name = func_lower.capitalize()
        if col_name not in df.columns:
            return None
        return df[col_name]

    # Simple moving averages
    if func_lower == "sma":
        return SMA(df, period, input_col)
    if func_lower == "ema":
        return EMA(df, period, input_col)
    if func_lower == "hma":
        return HMA(df, period, input_col)
    if func_lower == "frama":
        length = _as_int(ind.get("length", ind.get("period")), 16)
        fc = _as_int(ind.get("FC", ind.get("fc")), 1)
        sc = _as_int(ind.get("SC", ind.get("sc")), 198)
        price_type = ind.get("price_type", "HL2")
        return FRAMA(df, length, fc, sc, price_type)
    if func_lower == "kama":
        length = _as_int(ind.get("length", ind.get("period")), 21)
        price_type = ind.get("price_type", "Close")
        return KAMA(df, length, price_type)

    # Slope indicators
    if func_lower == "slope_sma":
        return SLOPE_SMA(df, period, input_col)
    if func_lower == "slope_ema":
        return SLOPE_EMA(df, period, input_col)
    if func_lower == "slope_hma":
        return SLOPE_HMA(df, period, input_col)

    # Momentum indicators
    if func_lower == "rsi":
        return RSI(df, period, input_col)
    if func_lower == "roc":
        return ROC(df, period, input_col)
    if func_lower in ("willr", "williamsr"):
        return WILLR(df, period)
    if func_lower == "cci":
        return CCI(df, period)

    # Volatility indicators
    if func_lower == "atr":
        return ATR(df, period)

    # EWO (Elliott Wave Oscillator)
    if func_lower == "ewo":
        sma1 = _as_int(ind.get("sma1_length", 5), 5)
        sma2 = _as_int(ind.get("sma2_length", 35), 35)
        source = ind.get("source", ind.get("input", "Close"))
        if isinstance(source, str):
            source = source.strip("'\"")
        use_percent = _as_bool(ind.get("use_percent", True), True)
        return EWO(df, sma1_length=sma1, sma2_length=sma2, source=source, use_percent=use_percent)

    # MA Spread Z-Score
    if func_lower == "ma_spread_zscore":
        ma_length = _as_int(ind.get("ma_length", ind.get("period", 20)), 20)
        mean_window = _as_int(ind.get("spread_mean_window", ind.get("spread_window", ma_length)), ma_length)
        std_window = _as_int(ind.get("spread_std_window", mean_window), mean_window)
        price_col = ind.get("price_col", "Close")
        ma_type = ind.get("ma_type", "SMA")
        use_percent = _as_bool(ind.get("use_percent", False), False)
        output = ind.get("output", "zscore")

        result = MA_SPREAD_ZSCORE(
            df,
            price_col=price_col,
            ma_length=ma_length,
            spread_mean_window=mean_window,
            spread_std_window=std_window,
            ma_type=ma_type,
            use_percent=use_percent,
        )
        if result is None or result.empty:
            return None

        # Extract requested output column
        target_col = _find_column(result, output, "zscore")
        return result[target_col]

    # MACD
    if func_lower == "macd":
        fast = _as_int(ind.get("fast_period", ind.get("fast", 12)), 12)
        slow = _as_int(ind.get("slow_period", ind.get("slow", 26)), 26)
        signal = _as_int(ind.get("signal_period", ind.get("signal", 9)), 9)
        output = ind.get("output", ind.get("type", "line")).lower()
        return MACD(df, fast, slow, signal, output)

    # Bollinger Bands
    if func_lower in ("bbands", "bb"):
        std_dev = _as_float(ind.get("std_dev", ind.get("std", 2)), 2)
        output = ind.get("output", ind.get("type", "middle")).lower()
        return BBANDS(df, period, std_dev, output)
    if func_lower == "bb_upper":
        std_dev = _as_float(ind.get("std_dev", ind.get("std", 2)), 2)
        return BBANDS(df, period, std_dev, "upper")
    if func_lower == "bb_middle":
        std_dev = _as_float(ind.get("std_dev", ind.get("std", 2)), 2)
        return BBANDS(df, period, std_dev, "middle")
    if func_lower == "bb_lower":
        std_dev = _as_float(ind.get("std_dev", ind.get("std", 2)), 2)
        return BBANDS(df, period, std_dev, "lower")
    if func_lower == "bb_width":
        std_dev = _as_float(ind.get("std_dev", ind.get("std", 2)), 2)
        upper = BBANDS(df, period, std_dev, "upper")
        lower = BBANDS(df, period, std_dev, "lower")
        return upper - lower

    # Parabolic SAR
    if func_lower in ("psar", "sar"):
        accel = _as_float(ind.get("acceleration", ind.get("accel", 0.02)), 0.02)
        max_accel = _as_float(ind.get("max_acceleration", ind.get("max_accel", 0.2)), 0.2)
        return SAR(df, accel, max_accel)

    # HARSI Flip
    if func_lower in ("harsi_flip", "harsi"):
        timeperiod = _as_int(ind.get("period", ind.get("timeperiod", 14)), 14)
        smoothing = _as_int(ind.get("smoothing", 3), 3)
        return HARSI_Flip(df, timeperiod, smoothing)

    # Supertrend
    if func_lower == "supertrend":
        period_st = _as_int(ind.get("period", 10), 10)
        multiplier = _as_float(ind.get("multiplier", 3.0), 3.0)
        return SUPERTREND(df, period_st, multiplier)
    if func_lower == "supertrend_upper":
        period_st = _as_int(ind.get("period", 10), 10)
        multiplier = _as_float(ind.get("multiplier", 3.0), 3.0)
        return SUPERTREND_UPPER(df, period_st, multiplier)
    if func_lower == "supertrend_lower":
        period_st = _as_int(ind.get("period", 10), 10)
        multiplier = _as_float(ind.get("multiplier", 3.0), 3.0)
        return SUPERTREND_LOWER(df, period_st, multiplier)

    # Ichimoku indicators
    if func_lower == "ichimoku_cloud_top":
        return _calculate_ichimoku(df, ind, ICHIMOKU_CLOUD_TOP)
    if func_lower == "ichimoku_cloud_bottom":
        return _calculate_ichimoku(df, ind, ICHIMOKU_CLOUD_BOTTOM)
    if func_lower == "ichimoku_cloud_signal":
        return _calculate_ichimoku(df, ind, ICHIMOKU_CLOUD_SIGNAL)
    if func_lower == "ichimoku_conversion":
        periods = _as_int(ind.get("conversion_periods", ind.get("conversion", 9)), 9)
        return ICHIMOKU_CONVERSION(df, periods)
    if func_lower == "ichimoku_base":
        periods = _as_int(ind.get("base_periods", ind.get("base", 26)), 26)
        return ICHIMOKU_BASE(df, periods)
    if func_lower == "ichimoku_lagging":
        displacement = _as_int(ind.get("displacement", 26), 26)
        visual = _as_bool(ind.get("visual", False), False)
        return ICHIMOKU_LAGGING(df, displacement, visual)

    # Donchian Channel
    if func_lower == "donchian_upper":
        length = _as_int(ind.get("length", ind.get("period", 20)), 20)
        return DONCHIAN_UPPER(df, length)
    if func_lower == "donchian_lower":
        length = _as_int(ind.get("length", ind.get("period", 20)), 20)
        return DONCHIAN_LOWER(df, length)
    if func_lower == "donchian_basis":
        length = _as_int(ind.get("length", ind.get("period", 20)), 20)
        return DONCHIAN_BASIS(df, length)
    if func_lower == "donchian_width":
        length = _as_int(ind.get("length", ind.get("period", 20)), 20)
        return DONCHIAN_WIDTH(df, length)
    if func_lower == "donchian_position":
        length = _as_int(ind.get("length", ind.get("period", 20)), 20)
        return DONCHIAN_POSITION(df, length)

    # Trend Magic
    if func_lower == "trend_magic":
        cci_period = _as_int(ind.get("cci_period", 20), 20)
        atr_mult = _as_float(ind.get("atr_multiplier", 1.0), 1.0)
        atr_period = _as_int(ind.get("atr_period", 5), 5)
        return TREND_MAGIC(df, cci_period, atr_mult, atr_period)
    if func_lower == "trend_magic_signal":
        cci_period = _as_int(ind.get("cci_period", 20), 20)
        atr_mult = _as_float(ind.get("atr_multiplier", 1.0), 1.0)
        atr_period = _as_int(ind.get("atr_period", 5), 5)
        return TREND_MAGIC_SIGNAL(df, cci_period, atr_mult, atr_period)

    # Kalman ROC Stochastic
    if func_lower == "kalman_roc_stoch":
        return KALMAN_ROC_STOCH(df)
    if func_lower == "kalman_roc_stoch_signal":
        return KALMAN_ROC_STOCH_SIGNAL(df)

    # OBV MACD
    if func_lower == "obv_macd":
        return OBV_MACD(df)
    if func_lower == "obv_macd_signal":
        return OBV_MACD_SIGNAL(df)

    # Pivot Support/Resistance
    if func_lower == "pivot_sr":
        return PIVOT_SR(df)
    if func_lower == "pivot_sr_crossover":
        return PIVOT_SR_CROSSOVER(df)
    if func_lower == "pivot_sr_proximity":
        return PIVOT_SR_PROXIMITY(df)

    # If no match found, return None
    if debug_mode:
        logger.warning(f"Unknown indicator: {func_lower}")
    return None


def _calculate_ichimoku(df: pd.DataFrame, ind: Dict[str, Any], func) -> pd.Series:
    """Calculate Ichimoku indicator with parameters."""
    conv = _as_int(ind.get("conversion_periods", ind.get("conversion", 9)), 9)
    base = _as_int(ind.get("base_periods", ind.get("base", 26)), 26)
    span_b = _as_int(ind.get("span_b_periods", ind.get("span_b", 52)), 52)
    displacement = _as_int(ind.get("displacement", 26), 26)
    visual = _as_bool(ind.get("visual", False), False)
    return func(df, conv, base, span_b, displacement, visual=visual)


def _apply_specifier(series: Any, specifier: int) -> Any:
    """Apply index specifier to get a specific value from a series."""
    if series is None:
        return None

    # Handle DataFrame - get the last column
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, -1]

    # Handle Series
    if hasattr(series, "iloc"):
        try:
            if specifier < -len(series) or specifier >= len(series):
                return None
            value = series.iloc[specifier]
            return None if pd.isna(value) else value
        except Exception:
            return None

    # Return scalar as-is
    return series


def indicator_calculation(
    df: pd.DataFrame,
    ind: Dict[str, Any],
    vals: Optional[Dict[str, Any]] = None,
    debug_mode: bool = False
) -> Any:
    """
    Calculate an indicator value from a parsed indicator dictionary.

    This is essentially a wrapper around apply_function.

    Args:
        df: DataFrame with price data
        ind: Indicator dictionary from ind_to_dict or simplify_conditions
        vals: Optional values dictionary (legacy parameter)
        debug_mode: Enable debug output

    Returns:
        Indicator value (scalar or Series)
    """
    return apply_function(df, ind, vals, debug_mode)


# =============================================================================
# Expression Evaluation Functions
# =============================================================================

def evaluate_expression(
    df: pd.DataFrame,
    exp: str,
    debug_mode: bool = False
) -> bool:
    """
    Evaluate a conditional expression against a DataFrame.

    Handles expressions like:
    - "Close[-1] > sma(20)[-1]"
    - "rsi(14)[-1] < 30"
    - "(EWO(...)[-1] > 0) & (EWO(...)[-2] <= 0)"
    - Complex Ichimoku expressions

    Args:
        df: DataFrame with OHLCV data
        exp: The expression string to evaluate
        debug_mode: Enable debug output

    Returns:
        bool: True if condition is met, False otherwise
    """
    if not exp or not isinstance(exp, str):
        return False

    exp = exp.strip()
    if not exp:
        return False

    # Try specialized handlers first
    result = _try_evaluate_ewo_cross(df, exp)
    if result is not None:
        return result

    result = _try_evaluate_ichimoku(df, exp)
    if result is not None:
        return result

    # Try simple comparison
    result = _evaluate_simple_comparison(df, exp, debug_mode)
    if result is not None:
        return result

    # Try Python eval fallback
    result = _try_evaluate_python(df, exp)
    if result is not None:
        return result

    if debug_mode:
        logger.warning(f"Could not evaluate expression: {exp}")

    return False


def _evaluate_simple_comparison(
    df: pd.DataFrame,
    exp: str,
    debug_mode: bool = False
) -> Optional[bool]:
    """
    Evaluate a simple binary comparison expression.

    Handles: "indicator1 operator indicator2"
    where operator is one of: >, <, >=, <=, ==, !=
    """
    try:
        parsed = simplify_conditions(exp)
        if not parsed:
            return None

        ind1 = parsed.get("ind1")
        ind2 = parsed.get("ind2")
        op = parsed.get("comparison")

        if not op or ind1 is None or ind2 is None:
            return None

        # Get values
        val1 = apply_function(df, ind1, None, debug_mode)
        val2 = apply_function(df, ind2, None, debug_mode)

        # Extract scalar values if needed
        if hasattr(val1, "iloc"):
            val1 = val1.iloc[-1]
        if hasattr(val2, "iloc"):
            val2 = val2.iloc[-1]

        # Handle NaN values
        if pd.isna(val1) or pd.isna(val2):
            return False

        # Perform comparison
        if op == ">":
            return bool(val1 > val2)
        elif op == "<":
            return bool(val1 < val2)
        elif op == ">=":
            return bool(val1 >= val2)
        elif op == "<=":
            return bool(val1 <= val2)
        elif op == "==":
            return bool(val1 == val2)
        elif op == "!=":
            return bool(val1 != val2)

        return None
    except Exception as e:
        if debug_mode:
            logger.debug(f"Simple comparison failed: {e}")
        return None


def _try_evaluate_ewo_cross(df: pd.DataFrame, exp: str) -> Optional[bool]:
    """Handle expressions like (EWO(...)[-1] > 0) & (EWO(...)[-2] <= 0)."""
    if "EWO(" not in exp or "&" not in exp:
        return None

    match = re.search(r"EWO\(([^)]*)\)", exp)
    if not match:
        return None

    kwargs = _parse_ewo_kwargs(match.group(1))

    sma1 = int(kwargs.get("sma1_length", 5))
    sma2 = int(kwargs.get("sma2_length", 35))
    source = kwargs.get("source", kwargs.get("input", "Close"))
    if isinstance(source, str):
        source = source.strip("'\"")
    use_percent = kwargs.get("use_percent", True)
    if isinstance(use_percent, str):
        use_percent = use_percent.lower() in ("true", "1", "yes", "y")

    series = EWO(df, sma1_length=sma1, sma2_length=sma2, source=source, use_percent=use_percent)
    if series is None or series.empty or len(series) < 2:
        return None

    last = series.iloc[-1]
    prev = series.iloc[-2]
    if pd.isna(last) or pd.isna(prev):
        return None

    exp_compact = exp.replace(" ", "").lower()
    up_cross = "[-1]>0" in exp_compact and "[-2]<=" in exp_compact
    down_cross = "[-1]<0" in exp_compact and "[-2]>=" in exp_compact

    if up_cross:
        return bool(last > 0 and prev <= 0)
    if down_cross:
        return bool(last < 0 and prev >= 0)
    return None


def _parse_ewo_kwargs(param_str: str) -> Dict[str, Any]:
    """Parse EWO parameters from string."""
    params = {}
    for part in param_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.strip()
        val = val.strip()
        try:
            params[key] = ast.literal_eval(val)
        except Exception:
            params[key] = val.strip("'\"")
    return params


def _try_evaluate_ichimoku(df: pd.DataFrame, exp: str) -> Optional[bool]:
    """
    Evaluate Ichimoku expressions with Python's evaluator.
    """
    if "ICHIMOKU" not in exp:
        return None

    try:
        ctx = _build_eval_context(df)

        # Add Ichimoku functions
        ctx["ICHIMOKU_CLOUD_TOP"] = lambda *args, **kwargs: _SeriesIndexer(ICHIMOKU_CLOUD_TOP(df, *args, **kwargs))
        ctx["ICHIMOKU_CLOUD_BOTTOM"] = lambda *args, **kwargs: _SeriesIndexer(ICHIMOKU_CLOUD_BOTTOM(df, *args, **kwargs))
        ctx["ICHIMOKU_CLOUD_SIGNAL"] = lambda *args, **kwargs: _SeriesIndexer(ICHIMOKU_CLOUD_SIGNAL(df, *args, **kwargs))
        ctx["ICHIMOKU_CONVERSION"] = lambda *args, **kwargs: _SeriesIndexer(ICHIMOKU_CONVERSION(df, *args, **kwargs))
        ctx["ICHIMOKU_BASE"] = lambda *args, **kwargs: _SeriesIndexer(ICHIMOKU_BASE(df, *args, **kwargs))
        ctx["ICHIMOKU_LAGGING"] = lambda *args, **kwargs: _SeriesIndexer(ICHIMOKU_LAGGING(df, *args, **kwargs))

        result = eval(exp, {"__builtins__": {}}, ctx)
        return bool(result)
    except Exception:
        return None


def _try_evaluate_python(df: pd.DataFrame, exp: str) -> Optional[bool]:
    """
    Generic Python eval fallback for expressions the simple parser rejects.
    """
    try:
        ctx = _build_eval_context(df)
        result = eval(exp, {"__builtins__": {}}, ctx)
        return bool(result)
    except Exception:
        return None


def _build_eval_context(df: pd.DataFrame) -> Dict[str, Any]:
    """Build evaluation context with price columns and indicator functions."""
    ctx = {
        "Close": _SeriesIndexer(df["Close"]),
        "Open": _SeriesIndexer(df["Open"]) if "Open" in df.columns else None,
        "High": _SeriesIndexer(df["High"]) if "High" in df.columns else None,
        "Low": _SeriesIndexer(df["Low"]) if "Low" in df.columns else None,
        "Volume": _SeriesIndexer(df["Volume"]) if "Volume" in df.columns else None,
    }

    # Add indicator functions
    indicator_names = [
        "sma", "ema", "hma", "frama", "kama",
        "rsi", "atr", "cci", "roc", "williamsr", "willr",
        "macd", "bb_upper", "bb_middle", "bb_lower", "bb_width",
        "psar", "sar",
        "ewo", "ma_spread_zscore",
        "harsi_flip",
        "supertrend", "supertrend_upper", "supertrend_lower",
        "donchian_upper", "donchian_lower", "donchian_basis",
        "trend_magic", "trend_magic_signal",
    ]
    for name in indicator_names:
        ctx[name] = _make_indicator_func(df, name)

    ctx["zscore"] = lambda val, lookback=20: _compute_zscore(val, lookback, df)

    return ctx


def _make_indicator_func(df: pd.DataFrame, name: str):
    """Create an indicator function for use in eval context."""
    lname = name.lower()

    def _wrapper(*args, **kwargs):
        ind = {"ind": lname}

        # Map positional args to common parameters
        if args:
            if lname in {"sma", "ema", "hma", "rsi", "roc", "atr", "cci", "willr", "williamsr"}:
                ind["period"] = args[0]
            elif lname == "macd":
                if len(args) > 0:
                    ind["fast_period"] = args[0]
                if len(args) > 1:
                    ind["slow_period"] = args[1]
                if len(args) > 2:
                    ind["signal_period"] = args[2]
            elif lname in ("psar", "sar"):
                if len(args) > 0:
                    ind["acceleration"] = args[0]
                if len(args) > 1:
                    ind["max_acceleration"] = args[1]
            elif lname.startswith("bb_"):
                if len(args) > 0:
                    ind["period"] = args[0]
                if len(args) > 1:
                    ind["std_dev"] = args[1]

        ind.update(kwargs)

        try:
            series = apply_function(df, ind)
        except Exception:
            series = None

        if series is None:
            series = pd.Series(np.nan, index=df.index)

        if hasattr(series, "__len__"):
            return _SeriesIndexer(series)
        return series

    return _wrapper


def _compute_zscore(val: Any, lookback: int, df: pd.DataFrame) -> Optional[_SeriesIndexer]:
    """Compute rolling z-score of a series-like value."""
    try:
        win = int(lookback)
    except Exception:
        win = 20

    series = val
    if isinstance(series, _SeriesIndexer):
        series = series.series
    if isinstance(series, str) and series in df.columns:
        series = df[series]
    if not hasattr(series, "rolling"):
        return None

    try:
        mean = series.rolling(win).mean()
        std = series.rolling(win).std()
        z = (series - mean) / std
        return _SeriesIndexer(z)
    except Exception:
        return None


def evaluate_expression_list(
    df: pd.DataFrame,
    exps: List[str],
    combination: str = "1"
) -> bool:
    """
    Evaluate a list of expressions with combination logic.

    Args:
        df: DataFrame with price data
        exps: List of condition strings
        combination: How to combine conditions:
            - "AND" or "1" - all conditions must be True
            - "OR" - any condition must be True
            - "1 AND 2" - conditions 1 and 2 must be True
            - "1 OR 2" - condition 1 or 2 must be True
            - "(1 AND 2) OR 3" - complex logic expressions
            - Custom expression like "1 AND (2 OR 3)"

    Returns:
        bool: True if the combination logic evaluates to True, False otherwise
    """
    if not exps:
        return False

    # Evaluate each expression
    results = []
    for exp in exps:
        try:
            result = evaluate_expression(df, exp, debug_mode=False)
            results.append(bool(result))
        except Exception:
            results.append(False)

    # Handle single condition
    if len(results) == 1:
        return results[0]

    # Normalize combination logic
    if not combination or combination == "1":
        combination = "AND"

    combination_upper = combination.upper().strip()

    # Handle simple AND/OR
    if combination_upper == "AND":
        return all(results)
    elif combination_upper == "OR":
        return any(results)

    # Handle complex expressions like "1 AND 2", "(1 OR 2) AND 3", etc.
    try:
        eval_expr = combination
        for i, result in enumerate(results, 1):
            # Replace condition number with its boolean value
            # Use word boundaries to avoid replacing "10" when looking for "1"
            eval_expr = re.sub(r'\b' + str(i) + r'\b', str(result), eval_expr)

        # Replace logical operators with Python syntax
        eval_expr = eval_expr.upper()
        eval_expr = eval_expr.replace("AND", "and")
        eval_expr = eval_expr.replace("OR", "or")
        eval_expr = eval_expr.replace("NOT", "not")

        # Evaluate the expression
        result = eval(eval_expr, {"__builtins__": {}}, {})
        return bool(result)
    except Exception:
        # If complex expression fails, default to AND
        return all(results)


# =============================================================================
# Helper Functions
# =============================================================================

def _as_int(value: Any, default: int) -> int:
    """Convert value to int with default fallback."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _as_float(value: Any, default: float) -> float:
    """Convert value to float with default fallback."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _as_bool(value: Any, default: bool) -> bool:
    """Convert value to bool with default fallback."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        val = value.strip().lower()
        if val in ("true", "1", "yes", "y"):
            return True
        if val in ("false", "0", "no", "n"):
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return default


def _find_column(df: pd.DataFrame, target: str, fallback: str) -> str:
    """Find a column name in a DataFrame, case-insensitive."""
    if target:
        for col in df.columns:
            if col.lower() == target.lower():
                return col
    return fallback if fallback in df.columns else df.columns[-1]

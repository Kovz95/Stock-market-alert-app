#!/usr/bin/env python3
"""
Condition Logic Testing Tool

A CLI tool to test and debug alert condition evaluation logic.
Helps you:
1. See exactly what values indicators return for a given ticker
2. Verify comparison logic is working correctly
3. Understand why a specific condition triggers or doesn't trigger

Usage:
    # Test a single condition with a ticker
    python test_condition_logic.py --ticker AAPL --condition "HARSI_Flip(period=14, smoothing=3)[-1] > 0"

    # Test an existing alert by ID
    python test_condition_logic.py --alert-id c312abe3-cf01-4a64-bcb6-649831403aa9

    # Show indicator values over recent bars
    python test_condition_logic.py --ticker AAPL --indicator "HARSI_Flip(period=14, smoothing=3)" --bars 10

    # Debug mode with full trace
    python test_condition_logic.py --ticker AAPL --condition "Close[-1] > sma(20)[-1]" --debug
"""

import argparse
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from backend import (
    evaluate_expression,
    simplify_conditions,
    indicator_calculation,
    ind_to_dict,
    apply_function,
)
from backend_fmp import FMPDataFetcher
from src.data_access.alert_repository import get_alert, list_alerts

# Import indicator functions directly for special handling
from src.utils.indicators import (
    HARSI_Flip,
    SMA,
    EMA,
    HMA,
    RSI,
    ATR,
    CCI,
    WILLR,
    ROC,
    EWO,
    MA_SPREAD_ZSCORE,
    MACD,
    BBANDS,
    SAR,
    SUPERTREND,
    ICHIMOKU_CLOUD_TOP,
    ICHIMOKU_CLOUD_BOTTOM,
    ICHIMOKU_CLOUD_SIGNAL,
    ICHIMOKU_CONVERSION,
    ICHIMOKU_BASE,
    ICHIMOKU_LAGGING,
)


# HARSI flip value explanations
HARSI_FLIP_EXPLANATIONS = {
    0: "no flip",
    1: "bearish flip (green to red)",
    2: "bullish flip (red to green)",
}


def parse_indicator_params(indicator_str: str) -> Dict[str, Any]:
    """
    Parse an indicator string and extract function name and parameters.

    Examples:
        "HARSI_Flip(period=14, smoothing=3)" -> {"func": "harsi_flip", "period": 14, "smoothing": 3}
        "RSI(14)" -> {"func": "rsi", "period": 14}
        "Close" -> {"func": "close"}
    """
    result = {"func": None, "specifier": None}

    # Extract specifier like [-1]
    specifier_match = re.search(r'\[(-?\d+)\]', indicator_str)
    if specifier_match:
        result["specifier"] = int(specifier_match.group(1))
        indicator_str = re.sub(r'\[-?\d+\]', '', indicator_str).strip()

    # Check for function call pattern: func(params)
    func_match = re.match(r'(\w+)\s*\(([^)]*)\)', indicator_str)
    if func_match:
        result["func"] = func_match.group(1).lower()
        params_str = func_match.group(2).strip()

        if params_str:
            # Try to parse as single number first
            try:
                result["period"] = int(params_str)
            except ValueError:
                # Parse named parameters: key=value, key=value
                for param in params_str.split(','):
                    param = param.strip()
                    if '=' in param:
                        key, val = param.split('=', 1)
                        key = key.strip().lower()
                        val = val.strip().strip('"\'')

                        # Convert to appropriate type
                        try:
                            if '.' in val:
                                result[key] = float(val)
                            else:
                                result[key] = int(val)
                        except ValueError:
                            # Boolean check
                            if val.lower() in ('true', 'yes', '1'):
                                result[key] = True
                            elif val.lower() in ('false', 'no', '0'):
                                result[key] = False
                            else:
                                result[key] = val
    else:
        # Simple name like "Close", "Open"
        result["func"] = indicator_str.strip().lower()

    return result


def calculate_indicator_direct(df: pd.DataFrame, indicator_str: str) -> Optional[pd.Series]:
    """
    Calculate an indicator by directly calling indicator functions.

    This bypasses backend.py's parsing for better control.
    """
    params = parse_indicator_params(indicator_str)
    func_name = params.get("func", "")

    if not func_name:
        return None

    series = None

    try:
        # Price columns
        if func_name in ("close", "open", "high", "low", "volume"):
            col_name = func_name.capitalize()
            if col_name in df.columns:
                series = df[col_name]

        # HARSI_Flip - special handling
        elif func_name == "harsi_flip":
            period = params.get("period", params.get("timeperiod", 14))
            smoothing = params.get("smoothing", 3)
            series = HARSI_Flip(df, period, smoothing)

        # Moving averages
        elif func_name == "sma":
            period = params.get("period", 20)
            input_col = params.get("input", "Close")
            series = SMA(df, period, input_col)

        elif func_name == "ema":
            period = params.get("period", 20)
            input_col = params.get("input", "Close")
            series = EMA(df, period, input_col)

        elif func_name == "hma":
            period = params.get("period", 20)
            input_col = params.get("input", "Close")
            series = HMA(df, period, input_col)

        # Oscillators
        elif func_name == "rsi":
            period = params.get("period", 14)
            input_col = params.get("input", "Close")
            series = RSI(df, period, input_col)

        elif func_name == "cci":
            period = params.get("period", 20)
            series = CCI(df, period)

        elif func_name in ("willr", "williamsr"):
            period = params.get("period", 14)
            series = WILLR(df, period)

        elif func_name == "roc":
            period = params.get("period", 10)
            input_col = params.get("input", "Close")
            series = ROC(df, period, input_col)

        # ATR
        elif func_name == "atr":
            period = params.get("period", 14)
            series = ATR(df, period)

        # EWO
        elif func_name == "ewo":
            sma1 = params.get("sma1_length", 5)
            sma2 = params.get("sma2_length", 35)
            source = params.get("source", "Close")
            use_percent = params.get("use_percent", True)
            series = EWO(df, sma1, sma2, source, use_percent)

        # MA Spread Z-Score
        elif func_name == "ma_spread_zscore":
            ma_length = params.get("ma_length", params.get("period", 20))
            spread_mean_window = params.get("spread_mean_window", ma_length)
            spread_std_window = params.get("spread_std_window", spread_mean_window)
            price_col = params.get("price_col", "Close")
            ma_type = params.get("ma_type", "SMA")
            use_percent = params.get("use_percent", False)
            output = params.get("output", "zscore")

            result_df = MA_SPREAD_ZSCORE(
                df,
                price_col=price_col,
                ma_length=ma_length,
                spread_mean_window=spread_mean_window,
                spread_std_window=spread_std_window,
                ma_type=ma_type,
                use_percent=use_percent,
            )
            if result_df is not None and not result_df.empty:
                if output in result_df.columns:
                    series = result_df[output]
                else:
                    series = result_df["zscore"] if "zscore" in result_df.columns else result_df.iloc[:, -1]

        # MACD
        elif func_name == "macd":
            fast = params.get("fast_period", params.get("fast", 12))
            slow = params.get("slow_period", params.get("slow", 26))
            signal = params.get("signal_period", params.get("signal", 9))
            output = params.get("output", "line").lower()

            if output == "signal":
                series = MACD(df, fast, slow, signal, "signal")
            else:
                series = MACD(df, fast, slow, signal, "line")

        # Bollinger Bands
        elif func_name in ("bb_upper", "bb_middle", "bb_lower", "bbands"):
            period = params.get("period", params.get("timeperiod", 20))
            std_dev = params.get("std_dev", params.get("std", 2))
            band_type = params.get("type", "upper")

            if func_name == "bb_upper":
                band_type = "upper"
            elif func_name == "bb_middle":
                band_type = "middle"
            elif func_name == "bb_lower":
                band_type = "lower"

            series = BBANDS(df, period, std_dev, band_type)

        # Parabolic SAR
        elif func_name in ("sar", "psar"):
            accel = params.get("acceleration", 0.02)
            max_accel = params.get("max_acceleration", 0.2)
            series = SAR(df, accel, max_accel)

        # Supertrend
        elif func_name == "supertrend":
            period = params.get("period", 10)
            multiplier = params.get("multiplier", 3.0)
            series = SUPERTREND(df, period, multiplier)

        # Ichimoku components
        elif func_name == "ichimoku_cloud_top":
            conv = params.get("conversion_periods", 9)
            base = params.get("base_periods", 26)
            span_b = params.get("span_b_periods", 52)
            disp = params.get("displacement", 26)
            series = ICHIMOKU_CLOUD_TOP(df, conv, base, span_b, disp)

        elif func_name == "ichimoku_cloud_bottom":
            conv = params.get("conversion_periods", 9)
            base = params.get("base_periods", 26)
            span_b = params.get("span_b_periods", 52)
            disp = params.get("displacement", 26)
            series = ICHIMOKU_CLOUD_BOTTOM(df, conv, base, span_b, disp)

        elif func_name == "ichimoku_cloud_signal":
            conv = params.get("conversion_periods", 9)
            base = params.get("base_periods", 26)
            span_b = params.get("span_b_periods", 52)
            disp = params.get("displacement", 26)
            series = ICHIMOKU_CLOUD_SIGNAL(df, conv, base, span_b, disp)

        elif func_name == "ichimoku_conversion":
            periods = params.get("periods", 9)
            series = ICHIMOKU_CONVERSION(df, periods)

        elif func_name == "ichimoku_base":
            periods = params.get("periods", 26)
            series = ICHIMOKU_BASE(df, periods)

        elif func_name == "ichimoku_lagging":
            disp = params.get("displacement", 26)
            series = ICHIMOKU_LAGGING(df, disp)

        # If we got a series, return it
        if series is not None:
            return pd.Series(series) if not isinstance(series, pd.Series) else series

    except Exception as e:
        print(f"  [DEBUG] Direct calculation error for {func_name}: {e}")

    return None


def get_price_data(ticker: str, timeframe: str = "1d", days: int = 200) -> Optional[pd.DataFrame]:
    """Fetch price data for a ticker."""
    fmp = FMPDataFetcher()
    try:
        if timeframe in ("1wk", "weekly"):
            df = fmp.get_historical_data(ticker, period="1day", timeframe="1wk")
        elif timeframe in ("1h", "hourly"):
            df = fmp.get_historical_data(ticker, period="1hour")
        else:
            df = fmp.get_historical_data(ticker, period="1day")

        if df is not None and not df.empty:
            if len(df) > days:
                df = df.tail(days)
            return df
        return None
    except Exception as e:
        print(f"Error fetching price data for {ticker}: {e}")
        return None


def parse_condition(condition_str: str) -> Dict[str, Any]:
    """
    Parse a condition string and return structured information.

    Returns:
        dict with keys:
            - original: Original condition string
            - left_str: Left side of the comparison
            - operator: Comparison operator
            - right_str: Right side of the comparison
            - left_parsed: Parsed left indicator dict
            - right_parsed: Parsed right indicator dict (or numeric value)
    """
    result = {
        "original": condition_str,
        "left_str": None,
        "operator": None,
        "right_str": None,
        "left_parsed": None,
        "right_parsed": None,
    }

    parsed = simplify_conditions(condition_str)
    if not parsed:
        return result

    # Find operator position to split the string
    operators = [">=", "<=", "==", "!=", ">", "<"]
    for op in operators:
        if op in condition_str:
            parts = condition_str.split(op, 1)
            if len(parts) == 2:
                result["left_str"] = parts[0].strip()
                result["operator"] = op
                result["right_str"] = parts[1].strip()
                break

    result["left_parsed"] = parsed.get("ind1")
    result["right_parsed"] = parsed.get("ind2")

    return result


def calculate_indicator_value(
    df: pd.DataFrame,
    indicator_str: str,
    debug: bool = False
) -> Tuple[Any, str]:
    """
    Calculate an indicator value and return it with an explanation.

    Returns:
        Tuple of (value, explanation)
    """
    try:
        # Parse the indicator string
        params = parse_indicator_params(indicator_str)
        specifier = params.get("specifier", -1)

        # Check for numeric literal
        try:
            num_val = float(indicator_str.strip())
            return num_val, "numeric literal"
        except ValueError:
            pass

        # Try direct calculation first (handles HARSI_Flip and other special indicators)
        series = calculate_indicator_direct(df, indicator_str)

        # Fall back to backend.apply_function if direct calc didn't work
        if series is None:
            parsed = ind_to_dict(indicator_str, debug_mode=debug)
            if not parsed:
                return None, "Failed to parse indicator"

            if parsed.get("isNum"):
                value = parsed.get("number")
                return value, "numeric literal"

            series = apply_function(df, parsed, debug_mode=debug)

        if series is None:
            return None, "Indicator returned no data"

        # Extract value based on specifier
        if not hasattr(series, "__len__") or isinstance(series, (int, float, np.integer, np.floating)):
            # Scalar result
            value = series
        else:
            # Series result - apply specifier
            if specifier is not None and len(series) > 0:
                try:
                    value = series.iloc[specifier]
                except (IndexError, KeyError):
                    value = None
            else:
                value = series.iloc[-1] if len(series) > 0 else None

        # Generate explanation
        explanation = generate_value_explanation_from_params(params, value)

        return value, explanation
    except Exception as e:
        if debug:
            import traceback
            traceback.print_exc()
        return None, f"Error: {str(e)}"


def generate_value_explanation(parsed: Dict[str, Any], value: Any) -> str:
    """Generate a human-readable explanation for an indicator value (legacy format)."""
    if parsed.get("isNum"):
        return "numeric literal"

    ind_name = (parsed.get("ind") or "").lower()

    # Special explanations for known indicators
    if ind_name == "harsi_flip":
        if value in HARSI_FLIP_EXPLANATIONS:
            return HARSI_FLIP_EXPLANATIONS[value]
        return f"HARSI flip value"

    if ind_name in ("close", "open", "high", "low", "volume"):
        return f"price column"

    if ind_name in ("sma", "ema", "hma"):
        period = parsed.get("period", "?")
        return f"{ind_name.upper()}({period}) value"

    if ind_name == "rsi":
        period = parsed.get("period", 14)
        if value is not None:
            if value > 70:
                return f"RSI({period}) - overbought zone"
            elif value < 30:
                return f"RSI({period}) - oversold zone"
            else:
                return f"RSI({period}) - neutral zone"
        return f"RSI({period}) value"

    if ind_name == "ewo":
        if value is not None:
            if value > 0:
                return "EWO positive (bullish momentum)"
            else:
                return "EWO negative (bearish momentum)"
        return "Elliott Wave Oscillator value"

    if ind_name == "ma_spread_zscore":
        if value is not None:
            if value > 2:
                return "z-score > 2 (price far above MA)"
            elif value < -2:
                return "z-score < -2 (price far below MA)"
            else:
                return "z-score within normal range"
        return "MA spread z-score"

    if "ichimoku" in ind_name:
        return f"Ichimoku component value"

    return "calculated value"


def generate_value_explanation_from_params(params: Dict[str, Any], value: Any) -> str:
    """Generate a human-readable explanation for an indicator value (new format)."""
    func_name = (params.get("func") or "").lower()

    # HARSI Flip
    if func_name == "harsi_flip":
        if value is not None:
            int_value = int(value) if not pd.isna(value) else None
            if int_value in HARSI_FLIP_EXPLANATIONS:
                return HARSI_FLIP_EXPLANATIONS[int_value]
        return "HARSI flip value"

    # Price columns
    if func_name in ("close", "open", "high", "low", "volume"):
        return "price column"

    # Moving averages
    if func_name in ("sma", "ema", "hma"):
        period = params.get("period", "?")
        return f"{func_name.upper()}({period}) value"

    # RSI
    if func_name == "rsi":
        period = params.get("period", 14)
        if value is not None and not pd.isna(value):
            if value > 70:
                return f"RSI({period}) - overbought zone"
            elif value < 30:
                return f"RSI({period}) - oversold zone"
            else:
                return f"RSI({period}) - neutral zone"
        return f"RSI({period}) value"

    # EWO
    if func_name == "ewo":
        if value is not None and not pd.isna(value):
            if value > 0:
                return "EWO positive (bullish momentum)"
            else:
                return "EWO negative (bearish momentum)"
        return "Elliott Wave Oscillator value"

    # MA Spread Z-Score
    if func_name == "ma_spread_zscore":
        if value is not None and not pd.isna(value):
            if value > 2:
                return "z-score > 2 (price far above MA)"
            elif value < -2:
                return "z-score < -2 (price far below MA)"
            else:
                return "z-score within normal range"
        return "MA spread z-score"

    # CCI
    if func_name == "cci":
        period = params.get("period", 20)
        if value is not None and not pd.isna(value):
            if value > 100:
                return f"CCI({period}) - overbought"
            elif value < -100:
                return f"CCI({period}) - oversold"
            else:
                return f"CCI({period}) - neutral"
        return f"CCI({period}) value"

    # Williams %R
    if func_name in ("willr", "williamsr"):
        if value is not None and not pd.isna(value):
            if value > -20:
                return "Williams %R - overbought"
            elif value < -80:
                return "Williams %R - oversold"
            else:
                return "Williams %R - neutral"
        return "Williams %R value"

    # Ichimoku
    if "ichimoku" in func_name:
        return "Ichimoku component value"

    # Supertrend
    if func_name == "supertrend":
        if value is not None and not pd.isna(value):
            if value == 1:
                return "uptrend"
            elif value == -1:
                return "downtrend"
        return "Supertrend signal"

    # MACD
    if func_name == "macd":
        if value is not None and not pd.isna(value):
            if value > 0:
                return "MACD positive (bullish)"
            else:
                return "MACD negative (bearish)"
        return "MACD value"

    # ATR
    if func_name == "atr":
        return "ATR volatility measure"

    # ROC
    if func_name == "roc":
        if value is not None and not pd.isna(value):
            if value > 0:
                return "ROC positive (upward momentum)"
            else:
                return "ROC negative (downward momentum)"
        return "Rate of Change"

    return "calculated value"


def test_condition(
    ticker: str,
    condition: str,
    timeframe: str = "1d",
    debug: bool = False
) -> Dict[str, Any]:
    """
    Test a condition against a ticker and return detailed results.

    Returns:
        dict with test results including values, comparison, and final result
    """
    result = {
        "ticker": ticker,
        "condition": condition,
        "timeframe": timeframe,
        "success": False,
        "error": None,
        "parsed": None,
        "left_value": None,
        "left_explanation": None,
        "right_value": None,
        "right_explanation": None,
        "comparison_result": None,
    }

    # Fetch price data
    df = get_price_data(ticker, timeframe)
    if df is None or df.empty:
        result["error"] = f"No price data available for {ticker}"
        return result

    result["data_points"] = len(df)
    result["date_range"] = f"{df.index[0]} to {df.index[-1]}" if hasattr(df.index[0], 'strftime') else f"index 0 to {len(df)-1}"

    # Parse the condition
    parsed = parse_condition(condition)
    result["parsed"] = {
        "left": parsed["left_str"],
        "operator": parsed["operator"],
        "right": parsed["right_str"],
    }

    if not parsed["operator"]:
        result["error"] = "Failed to parse condition - no comparison operator found"
        return result

    # Calculate left side value
    if parsed["left_str"]:
        left_val, left_expl = calculate_indicator_value(df, parsed["left_str"], debug)
        result["left_value"] = left_val
        result["left_explanation"] = left_expl

    # Calculate right side value
    if parsed["right_str"]:
        right_val, right_expl = calculate_indicator_value(df, parsed["right_str"], debug)
        result["right_value"] = right_val
        result["right_explanation"] = right_expl

    # Perform comparison directly if we have both values
    # This handles indicators that evaluate_expression doesn't support well
    left_val = result.get("left_value")
    right_val = result.get("right_value")
    op = parsed.get("operator")

    if left_val is not None and right_val is not None and op and not pd.isna(left_val) and not pd.isna(right_val):
        try:
            if op == ">":
                result["comparison_result"] = float(left_val) > float(right_val)
            elif op == "<":
                result["comparison_result"] = float(left_val) < float(right_val)
            elif op == ">=":
                result["comparison_result"] = float(left_val) >= float(right_val)
            elif op == "<=":
                result["comparison_result"] = float(left_val) <= float(right_val)
            elif op == "==":
                result["comparison_result"] = float(left_val) == float(right_val)
            elif op == "!=":
                result["comparison_result"] = float(left_val) != float(right_val)
            result["success"] = True
        except (ValueError, TypeError) as e:
            # Fall back to evaluate_expression if direct comparison fails
            try:
                comparison_result = evaluate_expression(df, condition, debug_mode=debug)
                result["comparison_result"] = bool(comparison_result)
                result["success"] = True
            except Exception as e2:
                result["error"] = f"Evaluation error: {str(e2)}"
    else:
        # Fall back to evaluate_expression
        try:
            comparison_result = evaluate_expression(df, condition, debug_mode=debug)
            result["comparison_result"] = bool(comparison_result)
            result["success"] = True
        except Exception as e:
            result["error"] = f"Evaluation error: {str(e)}"

    return result


def test_alert(alert_id: str, debug: bool = False) -> Dict[str, Any]:
    """
    Test all conditions for an existing alert.

    Returns:
        dict with alert info and results for each condition
    """
    result = {
        "alert_id": alert_id,
        "success": False,
        "error": None,
        "alert": None,
        "conditions_results": [],
        "overall_result": None,
    }

    # Load the alert
    alert = get_alert(alert_id)
    if not alert:
        result["error"] = f"Alert not found: {alert_id}"
        return result

    result["alert"] = {
        "name": alert.get("name", "Unnamed"),
        "ticker": alert.get("ticker", alert.get("ticker1", "Unknown")),
        "timeframe": alert.get("timeframe", "1d"),
        "combination_logic": alert.get("combination_logic", "AND"),
    }

    # Get ticker
    ticker = alert.get("ticker", alert.get("ticker1", ""))
    if not ticker:
        result["error"] = "No ticker specified in alert"
        return result

    timeframe = alert.get("timeframe", "1d")

    # Extract conditions
    conditions = []
    raw_conditions = alert.get("conditions", [])
    if isinstance(raw_conditions, list):
        for cond in raw_conditions:
            if isinstance(cond, dict):
                cond_str = cond.get("conditions", "")
                if cond_str:
                    conditions.append(cond_str)
            elif isinstance(cond, str):
                conditions.append(cond)
            elif isinstance(cond, list) and len(cond) > 0:
                if isinstance(cond[0], str):
                    conditions.append(cond[0])

    if not conditions:
        result["error"] = "No conditions found in alert"
        return result

    # Test each condition
    condition_results = []
    for i, cond in enumerate(conditions):
        cond_result = test_condition(ticker, cond, timeframe, debug)
        cond_result["condition_index"] = i + 1
        condition_results.append(cond_result)

    result["conditions_results"] = condition_results

    # Calculate overall result based on combination logic
    combination = alert.get("combination_logic", "AND").upper()
    individual_results = [r.get("comparison_result", False) for r in condition_results]

    if combination == "AND":
        result["overall_result"] = all(individual_results)
    elif combination == "OR":
        result["overall_result"] = any(individual_results)
    else:
        # Complex combination logic - try to evaluate
        try:
            from backend import evaluate_expression_list
            df = get_price_data(ticker, timeframe)
            if df is not None:
                result["overall_result"] = evaluate_expression_list(df, conditions, combination)
        except Exception:
            result["overall_result"] = all(individual_results)

    result["success"] = True
    return result


def show_indicator_history(
    ticker: str,
    indicator: str,
    bars: int = 10,
    timeframe: str = "1d",
    debug: bool = False
) -> Dict[str, Any]:
    """
    Show indicator values over recent bars.

    Returns:
        dict with indicator history
    """
    result = {
        "ticker": ticker,
        "indicator": indicator,
        "timeframe": timeframe,
        "success": False,
        "error": None,
        "values": [],
    }

    # Fetch price data
    df = get_price_data(ticker, timeframe)
    if df is None or df.empty:
        result["error"] = f"No price data available for {ticker}"
        return result

    # Remove any specifier like [-1] for full series
    indicator_base = re.sub(r'\[-?\d+\]$', '', indicator.strip())

    try:
        # First try direct calculation (handles HARSI_Flip and other special indicators)
        series = calculate_indicator_direct(df, indicator_base)

        # Fall back to backend.apply_function if direct calc didn't work
        if series is None:
            parsed = ind_to_dict(indicator_base, debug_mode=debug)
            if parsed:
                parsed.pop("specifier", None)  # Don't use specifier - get full series
                series = apply_function(df, parsed, debug_mode=debug)

        if series is None:
            result["error"] = "Indicator returned no data (not supported or calculation error)"
            return result

        if not hasattr(series, "__len__"):
            result["error"] = "Indicator returned a scalar, not a series"
            return result

        # Get last N bars
        series = pd.Series(series) if not isinstance(series, pd.Series) else series
        history = []

        # Parse for explanation generation
        params = parse_indicator_params(indicator_base)

        start_idx = max(0, len(series) - bars)
        for i in range(start_idx, len(series)):
            bar_offset = i - len(series)  # Will be negative (e.g., -10, -9, ..., -1)
            value = series.iloc[i]

            # Generate explanation
            explanation = generate_value_explanation_from_params(params, value)

            is_current = (i == len(series) - 1)

            history.append({
                "bar": bar_offset,
                "value": value if not pd.isna(value) else None,
                "explanation": explanation,
                "is_current": is_current,
            })

        result["values"] = history
        result["success"] = True

    except Exception as e:
        result["error"] = f"Error calculating indicator: {str(e)}"

    return result


def format_value(value: Any) -> str:
    """Format a value for display."""
    if value is None:
        return "None"
    if pd.isna(value):
        return "NaN"
    if isinstance(value, float):
        if abs(value) < 0.01 or abs(value) > 10000:
            return f"{value:.6g}"
        return f"{value:.4f}"
    return str(value)


def print_condition_test(result: Dict[str, Any]) -> None:
    """Pretty print condition test results."""
    print()
    print("=" * 60)
    print("CONDITION TEST RESULTS")
    print("=" * 60)
    print()

    print(f"Condition: {result['condition']}")
    print(f"Ticker:    {result['ticker']}")
    print(f"Timeframe: {result['timeframe']}")

    if result.get("error"):
        print()
        print(f"ERROR: {result['error']}")
        return

    print(f"Data:      {result.get('data_points', '?')} bars")
    print()

    print("Parsed:")
    parsed = result.get("parsed", {})
    print(f"  Left side:  {parsed.get('left', 'N/A')}")
    print(f"  Operator:   {parsed.get('operator', 'N/A')}")
    print(f"  Right side: {parsed.get('right', 'N/A')}")
    print()

    print("Values:")
    left_val = format_value(result.get("left_value"))
    right_val = format_value(result.get("right_value"))
    left_expl = result.get("left_explanation", "")
    right_expl = result.get("right_explanation", "")

    print(f"  Left value:  {left_val}  ({left_expl})")
    print(f"  Right value: {right_val}  ({right_expl})")
    print()

    comp_result = result.get("comparison_result")
    op = parsed.get("operator", "?")
    result_str = "TRUE" if comp_result else "FALSE"
    result_mark = "+" if comp_result else "x"

    print(f"Result: {left_val} {op} {right_val} -> {result_str} {result_mark}")
    print()


def print_alert_test(result: Dict[str, Any]) -> None:
    """Pretty print alert test results."""
    print()
    print("=" * 60)
    print("ALERT TEST RESULTS")
    print("=" * 60)
    print()

    if result.get("error"):
        print(f"ERROR: {result['error']}")
        return

    alert = result.get("alert", {})
    print(f"Alert ID:          {result['alert_id']}")
    print(f"Name:              {alert.get('name', 'N/A')}")
    print(f"Ticker:            {alert.get('ticker', 'N/A')}")
    print(f"Timeframe:         {alert.get('timeframe', 'N/A')}")
    print(f"Combination Logic: {alert.get('combination_logic', 'N/A')}")
    print()

    conditions_results = result.get("conditions_results", [])
    print(f"Conditions ({len(conditions_results)} total):")
    print("-" * 60)

    for cond_result in conditions_results:
        idx = cond_result.get("condition_index", "?")
        cond = cond_result.get("condition", "N/A")

        print(f"\n[Condition {idx}] {cond}")

        if cond_result.get("error"):
            print(f"  ERROR: {cond_result['error']}")
            continue

        left_val = format_value(cond_result.get("left_value"))
        right_val = format_value(cond_result.get("right_value"))
        left_expl = cond_result.get("left_explanation", "")
        right_expl = cond_result.get("right_explanation", "")

        print(f"  Left:   {left_val} ({left_expl})")
        print(f"  Right:  {right_val} ({right_expl})")

        comp_result = cond_result.get("comparison_result")
        result_str = "TRUE" if comp_result else "FALSE"
        result_mark = "+" if comp_result else "x"
        print(f"  Result: {result_str} {result_mark}")

    print()
    print("-" * 60)
    overall = result.get("overall_result")
    overall_str = "TRIGGERED" if overall else "NOT TRIGGERED"
    overall_mark = "+" if overall else "x"
    print(f"Overall Result: {overall_str} {overall_mark}")
    print()


def print_indicator_history(result: Dict[str, Any]) -> None:
    """Pretty print indicator history."""
    print()
    print("=" * 60)
    print("INDICATOR HISTORY")
    print("=" * 60)
    print()

    if result.get("error"):
        print(f"ERROR: {result['error']}")
        return

    print(f"{result['indicator']} for {result['ticker']} (timeframe: {result['timeframe']})")
    print()

    values = result.get("values", [])
    if not values:
        print("No values to display")
        return

    print(f"Last {len(values)} bars:")
    print("-" * 40)

    for item in values:
        bar = item["bar"]
        value = format_value(item["value"])
        explanation = item.get("explanation", "")
        current_marker = " <- Current" if item.get("is_current") else ""

        print(f"  Bar {bar:>3}: {value:>12}  ({explanation}){current_marker}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Test and debug alert condition evaluation logic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test a single condition
  python test_condition_logic.py --ticker AAPL --condition "Close[-1] > sma(20)[-1]"

  # Test an existing alert
  python test_condition_logic.py --alert-id c312abe3-cf01-4a64-bcb6-649831403aa9

  # Show indicator history
  python test_condition_logic.py --ticker AAPL --indicator "RSI(14)" --bars 10

  # Debug mode
  python test_condition_logic.py --ticker V --condition "HARSI_Flip(period=14, smoothing=3)[-1] > 0" --debug
        """
    )

    parser.add_argument("--ticker", "-t", help="Stock ticker symbol")
    parser.add_argument("--condition", "-c", help="Condition string to test")
    parser.add_argument("--alert-id", "-a", help="Alert ID to test")
    parser.add_argument("--indicator", "-i", help="Indicator to show history for")
    parser.add_argument("--bars", "-b", type=int, default=10, help="Number of bars for history (default: 10)")
    parser.add_argument("--timeframe", "-tf", default="1d", help="Timeframe: 1d, 1wk, 1h (default: 1d)")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode with full trace")
    parser.add_argument("--list-alerts", "-l", action="store_true", help="List available alerts")

    args = parser.parse_args()

    # List alerts mode
    if args.list_alerts:
        print("\nAvailable Alerts:")
        print("-" * 80)
        alerts = list_alerts()
        for alert in alerts[:20]:  # Show first 20
            alert_id = alert.get("alert_id", "?")
            name = alert.get("name", "Unnamed")
            ticker = alert.get("ticker", alert.get("ticker1", "?"))
            action = alert.get("action", "on")
            status = "ACTIVE" if action == "on" else "OFF"
            print(f"  [{status:6}] {alert_id[:8]}... | {ticker:6} | {name[:40]}")
        if len(alerts) > 20:
            print(f"\n  ... and {len(alerts) - 20} more alerts")
        print()
        return 0

    # Alert test mode
    if args.alert_id:
        result = test_alert(args.alert_id, debug=args.debug)
        print_alert_test(result)
        return 0 if result.get("success") else 1

    # Indicator history mode
    if args.indicator and args.ticker:
        result = show_indicator_history(
            args.ticker,
            args.indicator,
            bars=args.bars,
            timeframe=args.timeframe,
            debug=args.debug
        )
        print_indicator_history(result)
        return 0 if result.get("success") else 1

    # Condition test mode
    if args.condition and args.ticker:
        result = test_condition(
            args.ticker,
            args.condition,
            timeframe=args.timeframe,
            debug=args.debug
        )
        print_condition_test(result)
        return 0 if result.get("success") else 1

    # No valid combination of arguments
    parser.print_help()
    print("\nError: Please specify either:")
    print("  --ticker and --condition (to test a condition)")
    print("  --alert-id (to test an existing alert)")
    print("  --ticker and --indicator (to show indicator history)")
    print("  --list-alerts (to list available alerts)")
    return 1


if __name__ == "__main__":
    sys.exit(main())

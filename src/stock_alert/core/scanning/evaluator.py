"""Condition evaluation utilities for market scanner."""

import re
from typing import Any

import pandas as pd


def normalize_indicator_dict(ind: dict | None) -> dict | None:
    """
    Normalize indicator dict to be compatible with legacy apply_function.

    Args:
        ind: Indicator dictionary

    Returns:
        Normalized indicator dictionary or None
    """
    if not ind:
        return None
    normed = dict(ind)
    # Alias timeperiod -> period for TA-Lib calls
    if "timeperiod" in normed and "period" not in normed:
        normed["period"] = normed["timeperiod"]
    return normed


def extract_condition_values(
    df: pd.DataFrame, condition_list: list[str], simplify_func, indicator_func
) -> list[dict[str, Any] | None]:
    """
    Return latest evaluated values for each condition for display.

    Args:
        df: Price dataframe
        condition_list: List of condition strings
        simplify_func: Function to simplify conditions
        indicator_func: Function to calculate indicators

    Returns:
        List of condition value dicts or None for each condition
    """
    values = []
    for cond in condition_list:
        try:
            parsed = simplify_func(cond)
            if not parsed:
                values.append(None)
                continue
            ind1 = normalize_indicator_dict(parsed.get("ind1"))
            ind2 = normalize_indicator_dict(parsed.get("ind2"))

            def _last(val):
                if val is None:
                    return None
                if hasattr(val, "iloc"):
                    try:
                        val = val.iloc[-1]
                    except Exception:
                        return None
                return val

            val1 = _last(indicator_func(df, ind1, None, False) if ind1 else None)
            val2 = _last(indicator_func(df, ind2, None, False) if ind2 else None)

            values.append(
                {
                    "expr": cond,
                    "left": val1,
                    "right": val2,
                    "op": parsed.get("comparison"),
                }
            )
        except Exception:
            values.append(None)
    return values


def apply_zscore_indicator(indicator_expr: str, use_zscore: bool, lookback: int) -> str:
    """
    Optionally wrap a numeric indicator expression in a z-score call.

    Args:
        indicator_expr: Indicator expression string
        use_zscore: Whether to apply z-score
        lookback: Lookback period for z-score

    Returns:
        Original or z-score wrapped expression
    """
    if not use_zscore or not indicator_expr:
        return indicator_expr
    if any(op in indicator_expr for op in [">", "<", "=", " and ", " or ", ":"]):
        return indicator_expr

    base = indicator_expr.strip()
    match = re.match(r"(.+)\[(-?\d+)\]$", base)
    if match:
        base = match.group(1)
    try:
        lb = int(lookback)
    except Exception:
        lb = 20
    return f"zscore({base}, lookback={lb})[-1]"


def format_condition_values(values: list[dict[str, Any] | None]) -> str:
    """
    Format condition values for table display.

    Args:
        values: List of condition value dicts

    Returns:
        Formatted string showing all condition values
    """
    if not values:
        return ""

    def _fmt(v):
        try:
            if v is None or pd.isna(v):
                return None
        except Exception:
            if v is None:
                return None
        try:
            v = float(v)
        except Exception:
            return str(v)
        # Compact formatting
        if abs(v) >= 1000 or (abs(v) > 0 and abs(v) < 0.01):
            return f"{v:.4g}"
        return f"{v:.4f}"

    parts = []
    for idx, item in enumerate(values, 1):
        if not item:
            continue
        left = _fmt(item.get("left"))
        right = _fmt(item.get("right"))
        op = item.get("op")
        if left is None:
            continue
        if op and right is not None:
            parts.append(f"{idx}. {left} {op} {right}")
        else:
            parts.append(f"{idx}. {left}")
    return " | ".join(parts)

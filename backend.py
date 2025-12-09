"""
Compatibility shim for the missing backend module.

The original backend.py source is absent, but the compiled bytecode
still exists in __pycache__/backend_legacy.cpython-311.pyc.  Importing
this module executes that bytecode so existing imports continue working
until the real source file is restored.
"""

from __future__ import annotations

import marshal
from pathlib import Path
from types import CodeType
from typing import Any, Dict
import ast
import pandas as pd
import re
import numpy as np
from indicators_lib import (
    EWO,
    MA_SPREAD_ZSCORE,
    ICHIMOKU_BASE,
    ICHIMOKU_CLOUD_BOTTOM,
    ICHIMOKU_CLOUD_SIGNAL,
    ICHIMOKU_CLOUD_TOP,
    ICHIMOKU_CONVERSION,
    ICHIMOKU_LAGGING,
    BBANDS,
)

_LEGACY_BYTECODE = Path(__file__).parent / "__pycache__/backend_legacy.cpython-311.pyc"
_PYC_HEADER_SIZE = 16
_BOOTSTRAPPED = False


def _load_legacy_code() -> CodeType:
    if not _LEGACY_BYTECODE.exists():
        raise FileNotFoundError(
            f"Missing legacy bytecode file: {_LEGACY_BYTECODE}. Cannot initialize backend shim."
        )
    with _LEGACY_BYTECODE.open("rb") as fh:
        fh.read(_PYC_HEADER_SIZE)
        return marshal.load(fh)


def _bootstrap(namespace: Dict[str, Any]) -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    code = _load_legacy_code()
    exec(code, namespace)
    _BOOTSTRAPPED = True


_bootstrap(globals())

# --- Patch: extend indicator support (EWO) -------------------------------
_legacy_apply_function = globals().get("apply_function")
_legacy_evaluate_expression = globals().get("evaluate_expression")
_legacy_ind_to_dict = globals().get("ind_to_dict")


class _SeriesIndexer:
    """Safe Series wrapper that supports Python-style negative indexing."""

    def __init__(self, series):
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


def apply_function(df, ind, vals=None, debug_mode=False):
    """
    Override to add EWO support while delegating all other indicators to
    the legacy implementation.
    """
    func = (ind or {}).get("ind", "")
    func_lower = func.lower() if isinstance(func, str) else ""

    if func_lower == "ma_spread_zscore":
        def _as_int(value, default):
            try:
                if value is None:
                    return default
                return int(value)
            except Exception:
                return default

        def _as_bool(value, default):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                val = value.strip().lower()
                if val in ("true", "1", "yes", "y"):
                    return True
                if val in ("false", "0", "no", "n"):
                    return False
            return default

        def _as_str(value, default):
            if value is None:
                return default
            return str(value).strip("'\"")

        ma_length = _as_int(ind.get("ma_length", ind.get("period", 20)), 20)
        mean_window = _as_int(ind.get("spread_mean_window", ind.get("spread_window", ma_length)), ma_length)
        std_window = _as_int(ind.get("spread_std_window", mean_window), mean_window)
        price_col = _as_str(ind.get("price_col", "Close"), "Close")
        ma_type = _as_str(ind.get("ma_type", "SMA"), "SMA")
        use_percent = _as_bool(ind.get("use_percent", False), False)
        output = _as_str(ind.get("output", "zscore"), "zscore")

        result = MA_SPREAD_ZSCORE(
            df,
            price_col=price_col,
            ma_length=ma_length,
            spread_mean_window=mean_window,
            spread_std_window=std_window,
            ma_type=ma_type,
            spread_window=None,
            use_percent=use_percent,
        )
        if result is None or result.empty:
            return None

        target_col = None
        if output:
            for col in result.columns:
                if col.lower() == output.lower():
                    target_col = col
                    break
        if target_col is None:
            target_col = "zscore" if "zscore" in result.columns else result.columns[-1]

        series = result[target_col]
        if "specifier" in ind:
            try:
                idx = int(ind.get("specifier", -1))
            except Exception:
                return None
            if idx < -len(series) or idx >= len(series):
                return None
            value = series.iloc[idx]
            return None if pd.isna(value) else value
        return series

    if func_lower == "ewo":
        def _as_int(value, default):
            try:
                return int(value)
            except Exception:
                return default

        def _as_bool(value, default):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                if value.lower() in ("true", "1", "yes", "y"):
                    return True
                if value.lower() in ("false", "0", "no", "n"):
                    return False
            return default

        sma1 = _as_int(ind.get("sma1_length", 5), 5)
        sma2 = _as_int(ind.get("sma2_length", 35), 35)

        source = ind.get("source", ind.get("input", "Close"))
        if isinstance(source, str):
            source = source.strip("'\"")

        use_percent = _as_bool(ind.get("use_percent", True), True)

        series = EWO(df, sma1_length=sma1, sma2_length=sma2, source=source, use_percent=use_percent)
        if series is None or series.empty:
            return None

        if "specifier" in ind:
            try:
                idx = int(ind.get("specifier", -1))
            except Exception:
                return None
            if idx < -len(series) or idx >= len(series):
                return None
            value = series.iloc[idx]
            return None if pd.isna(value) else value
        return series

    if func_lower in (
        "ichimoku_cloud_top",
        "ichimoku_cloud_bottom",
        "ichimoku_cloud_signal",
        "ichimoku_conversion",
        "ichimoku_base",
        "ichimoku_lagging",
    ):
        def _as_int(value, default):
            try:
                if value is None:
                    return default
                return int(value)
            except Exception:
                return default

        conv = _as_int(ind.get("conversion_periods", ind.get("conversion", 9)), 9)
        base = _as_int(ind.get("base_periods", ind.get("base", 26)), 26)
        span_b = _as_int(ind.get("span_b_periods", ind.get("span_b", 52)), 52)
        displacement = _as_int(ind.get("displacement", 26), 26)
        visual = ind.get("visual", False)
        if isinstance(visual, str):
            visual = visual.strip().lower() in ("true", "1", "yes", "y")

        series = None
        if func_lower == "ichimoku_cloud_top":
            series = ICHIMOKU_CLOUD_TOP(df, conv, base, span_b, displacement, visual=visual)
        elif func_lower == "ichimoku_cloud_bottom":
            series = ICHIMOKU_CLOUD_BOTTOM(df, conv, base, span_b, displacement, visual=visual)
        elif func_lower == "ichimoku_cloud_signal":
            series = ICHIMOKU_CLOUD_SIGNAL(df, conv, base, span_b, displacement, visual=visual)
        elif func_lower == "ichimoku_conversion":
            series = ICHIMOKU_CONVERSION(df, conv)
        elif func_lower == "ichimoku_base":
            series = ICHIMOKU_BASE(df, base)
        elif func_lower == "ichimoku_lagging":
            series = ICHIMOKU_LAGGING(df, displacement, visual=visual)

        if series is None:
            return None
        if hasattr(series, "empty") and series.empty:
            return None

        if "specifier" in ind:
            try:
                idx = int(ind.get("specifier", -1))
            except Exception:
                return None
            if idx < -len(series) or idx >= len(series):
                return None
            value = series.iloc[idx]
            return None if pd.isna(value) else value
        return series

    if _legacy_apply_function:
        return _legacy_apply_function(df, ind, vals, debug_mode)
    return None


def ind_to_dict(ind, debug_mode=False):
    """
    Patch the legacy parser so decimal/negative numeric literals (e.g. -2.0)
    no longer raise ValueError when int() casting is attempted.
    """
    try:
        if _legacy_ind_to_dict:
            return _legacy_ind_to_dict(ind, debug_mode)
    except ValueError:
        # Legacy code attempted int(<str>) which fails for '-2.0'.
        try:
            num_val = float(ind)
        except Exception:
            raise
        # Preserve prior behaviour of returning ints when possible.
        if num_val.is_integer():
            num_val = int(num_val)
        return {
            "isNum": True,
            "number": num_val,
            "operable": True,
            "specifier": -1,
        }
    return None


# --- Custom helper: evaluate EWO zero-cross with boolean "&" strings -----
def _parse_ewo_kwargs(param_str: str) -> dict:
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


def _try_evaluate_ewo_cross(df, exp: str):
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
        use_percent = use_percent.strip("'\"")
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


def _try_evaluate_ichimoku(df, exp: str):
    """
    Evaluate Ichimoku expressions with Python's evaluator when the legacy
    parser cannot handle OR/complex logic.
    """
    if "ICHIMOKU" not in exp:
        return None

    try:
        ctx = {
            "Close": _SeriesIndexer(df["Close"]),
            "Open": _SeriesIndexer(df["Open"]) if "Open" in df else None,
            "High": _SeriesIndexer(df["High"]) if "High" in df else None,
            "Low": _SeriesIndexer(df["Low"]) if "Low" in df else None,
            "Volume": _SeriesIndexer(df["Volume"]) if "Volume" in df else None,
            "ICHIMOKU_CLOUD_TOP": lambda *args, **kwargs: _SeriesIndexer(ICHIMOKU_CLOUD_TOP(df, *args, **kwargs)),
            "ICHIMOKU_CLOUD_BOTTOM": lambda *args, **kwargs: _SeriesIndexer(ICHIMOKU_CLOUD_BOTTOM(df, *args, **kwargs)),
            "ICHIMOKU_CLOUD_SIGNAL": lambda *args, **kwargs: _SeriesIndexer(ICHIMOKU_CLOUD_SIGNAL(df, *args, **kwargs)),
            "ICHIMOKU_CONVERSION": lambda *args, **kwargs: _SeriesIndexer(ICHIMOKU_CONVERSION(df, *args, **kwargs)),
            "ICHIMOKU_BASE": lambda *args, **kwargs: _SeriesIndexer(ICHIMOKU_BASE(df, *args, **kwargs)),
            "ICHIMOKU_LAGGING": lambda *args, **kwargs: _SeriesIndexer(ICHIMOKU_LAGGING(df, *args, **kwargs)),
        }
        result = eval(exp, {"__builtins__": {}}, ctx)
    except Exception:
        return None

    try:
        return bool(result)
    except Exception:
        return None


_SIMPLE_PERIOD_FUNCS = {
    "sma",
    "ema",
    "hma",
    "frama",
    "kama",
    "rsi",
    "roc",
    "atr",
    "cci",
    "willr",
    "williamsr",
    "slope_sma",
    "slope_ema",
    "slope_hma",
}


def _make_indicator_func(df, name: str):
    lname = name.lower()

    def _wrapper(*args, **kwargs):
        # Handle BB shorthand explicitly (bb_upper, etc.)
        if lname.startswith("bb_"):
            period = args[0] if len(args) > 0 else kwargs.get("timeperiod", kwargs.get("period", 20))
            std_dev = args[1] if len(args) > 1 else kwargs.get("std_dev", kwargs.get("std", 2))
            comp = lname.replace("bb_", "")
            upper = BBANDS(df, period, std_dev, "upper")
            middle = BBANDS(df, period, std_dev, "middle")
            lower = BBANDS(df, period, std_dev, "lower")
            if comp == "upper":
                series = upper
            elif comp == "middle":
                series = middle
            elif comp == "lower":
                series = lower
            elif comp == "width":
                try:
                    series = upper - lower
                except Exception:
                    series = None
            else:
                series = None
            if series is None:
                return np.nan
            return _SeriesIndexer(series)

        ind = {"ind": lname}

        if args:
            if lname in _SIMPLE_PERIOD_FUNCS:
                ind["period"] = args[0]
            elif lname == "macd":
                if len(args) > 0:
                    ind["fast_period"] = args[0]
                if len(args) > 1:
                    ind["slow_period"] = args[1]
                if len(args) > 2:
                    ind["signal_period"] = args[2]
            elif lname == "psar":
                if len(args) > 0:
                    ind["acceleration"] = args[0]
                if len(args) > 1:
                    ind["max_acceleration"] = args[1]

        # Merge kwargs last so explicit names take precedence.
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


def _try_evaluate_python(df, exp: str):
    """
    Generic Python eval fallback for expressions the legacy parser rejects
    (e.g., simple arithmetic, OR chains, or indicators with kwargs).
    """
    try:
        ctx = {
            "Close": _SeriesIndexer(df["Close"]),
            "Open": _SeriesIndexer(df["Open"]) if "Open" in df else None,
            "High": _SeriesIndexer(df["High"]) if "High" in df else None,
            "Low": _SeriesIndexer(df["Low"]) if "Low" in df else None,
            "Volume": _SeriesIndexer(df["Volume"]) if "Volume" in df else None,
        }

        # Expose indicator helpers commonly used by the scanner.
        indicator_names = {
            "sma",
            "ema",
            "hma",
            "frama",
            "kama",
            "rsi",
            "atr",
            "cci",
            "roc",
            "williamsr",
            "willr",
            "macd",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "bb_width",
            "psar",
            "ma_spread_zscore",
            "ewo",
            "harsi_flip",
        }
        for name in indicator_names:
            ctx[name] = _make_indicator_func(df, name)

        result = eval(exp, {"__builtins__": {}}, ctx)
    except Exception:
        return None

    try:
        return bool(result)
    except Exception:
        return None


def evaluate_expression(df, exp, debug_mode=False):
    """
    Wrap legacy evaluate_expression with custom handling for EWO zero-cross
    expressions that include '&' which the legacy parser cannot handle.
    """
    custom = _try_evaluate_ewo_cross(df, exp)
    if custom is not None:
        return custom
    ichi = _try_evaluate_ichimoku(df, exp)
    if ichi is not None:
        return ichi

    legacy_exc = None
    if _legacy_evaluate_expression:
        try:
            return _legacy_evaluate_expression(df, exp, debug_mode=debug_mode)
        except Exception as exc:
            legacy_exc = exc

    fallback = _try_evaluate_python(df, exp)
    if fallback is not None:
        return fallback

    if legacy_exc:
        raise legacy_exc
    if _legacy_evaluate_expression:
        # Surface the original error if the fallback could not resolve it.
        return _legacy_evaluate_expression(df, exp, debug_mode=debug_mode)
    raise RuntimeError("Legacy evaluate_expression implementation missing.")

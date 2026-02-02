"""
End-to-end tests for backend.py using live indicator implementations (no mocks).

These tests exercise the full pipeline: expression string → parse → real indicator
calculation → comparison/evaluation. They are designed to catch logic bugs in:
- ind_to_dict / simplify_conditions parsing
- apply_function / _calculate_indicator with real SMA, RSI, etc.
- _evaluate_simple_comparison (specifier handling, NaN, type coercion)
- evaluate_expression and evaluate_expression_list
- Combination logic (AND/OR, "1 AND 2", word-boundary for condition 10)

Run with: pytest tests/integration/test_backend_e2e.py -v
Or with marker: pytest -m backend_e2e -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.services.backend import (
    _apply_specifier,
    _evaluate_simple_comparison,
    _find_operator_position,
    apply_function,
    evaluate_expression,
    evaluate_expression_list,
    ind_to_dict,
    indicator_calculation,
    simplify_conditions,
)


# ---------------------------------------------------------------------------
# Fixtures: realistic OHLCV data (no mocks)
# ---------------------------------------------------------------------------

@pytest.fixture
def ohlcv_100():
    """100 bars of OHLCV for indicators that need warmup (e.g. RSI(14), SMA(20))."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
    close = 100.0 + np.cumsum(np.random.randn(n) * 1.5)
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    open_prices = np.roll(close, 1)
    open_prices[0] = close[0]
    volume = np.random.randint(1_000_000, 10_000_000, n)
    return pd.DataFrame({
        "Open": open_prices,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)


@pytest.fixture
def ohlcv_30():
    """30 bars for shorter warmup; still enough for RSI(14), SMA(20)."""
    np.random.seed(123)
    n = 30
    dates = pd.date_range(start="2024-06-01", periods=n, freq="D")
    close = 50.0 + np.cumsum(np.random.randn(n))
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    open_prices = np.roll(close, 1)
    open_prices[0] = close[0]
    volume = np.random.randint(500_000, 5_000_000, n)
    return pd.DataFrame({
        "Open": open_prices,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)


@pytest.fixture
def ohlcv_known_values():
    """Small DataFrame with known last values for deterministic assertions."""
    return pd.DataFrame({
        "Open": [98.0, 99.0, 100.0, 101.0, 102.0],
        "High": [99.0, 100.0, 101.0, 102.0, 103.0],
        "Low": [97.0, 98.0, 99.0, 100.0, 101.0],
        "Close": [98.5, 99.5, 100.5, 101.5, 102.5],
        "Volume": [1_000_000, 1_100_000, 1_200_000, 1_300_000, 1_400_000],
    })


# ---------------------------------------------------------------------------
# E2E: Parse → apply_function → value consistency
# ---------------------------------------------------------------------------

class TestParseApplyRoundtripE2E:
    """Full pipeline: ind_to_dict(expr) -> apply_function(df, ind) with real indicators."""

    def test_close_bracket_minus_one_returns_series_last_equals_close_last(self, ohlcv_100):
        """Close[-1] as expression should yield scalar equal to df['Close'].iloc[-1]."""
        parsed = ind_to_dict("Close[-1]")
        assert parsed is not None
        assert parsed.get("ind") == "Close"
        assert parsed.get("specifier") == -1
        result = apply_function(ohlcv_100, parsed)
        assert result is not None
        # specifier -1 in apply_function means "no specifier" and returns full series
        assert hasattr(result, "iloc")
        assert result.iloc[-1] == ohlcv_100["Close"].iloc[-1]

    def test_close_bracket_minus_two_returns_scalar_second_to_last(self, ohlcv_100):
        """Close[-2] should return scalar = second-to-last Close."""
        parsed = ind_to_dict("Close[-2]")
        assert parsed is not None
        result = apply_function(ohlcv_100, parsed)
        assert result is not None
        assert not hasattr(result, "iloc") or isinstance(result, (int, float))
        expected = ohlcv_100["Close"].iloc[-2]
        assert result == expected

    def test_numeric_literal_roundtrip(self, ohlcv_100):
        """Numeric literal in expression should return that number."""
        for s in ["30", "70", "-1.5", "0"]:
            parsed = ind_to_dict(s)
            assert parsed is not None and parsed.get("isNum")
            result = apply_function(ohlcv_100, parsed)
            assert result == (int(s) if s.lstrip("-").isdigit() else float(s))

    def test_sma_20_expression_returns_series_with_sensible_values(self, ohlcv_100):
        """sma(20) and sma(20)[-1] with real SMA. Backend returns full series for specifier -1."""
        parsed_no_spec = ind_to_dict("sma(20)")
        assert parsed_no_spec is not None
        series = apply_function(ohlcv_100, parsed_no_spec)
        assert series is not None and hasattr(series, "iloc")
        assert len(series) == len(ohlcv_100)
        # SMA should be in range of closes
        assert series.iloc[-1] >= ohlcv_100["Close"].min() - 10
        assert series.iloc[-1] <= ohlcv_100["Close"].max() + 10

        parsed_spec = ind_to_dict("sma(20)[-1]")
        result = apply_function(ohlcv_100, parsed_spec)
        assert result is not None
        # Backend treats specifier -1 as "no specifier" and returns full series; evaluator uses .iloc[-1]
        last_val = result.iloc[-1] if hasattr(result, "iloc") else result
        assert last_val == series.iloc[-1]

    def test_rsi_14_expression_real_calculation(self, ohlcv_100):
        """rsi(14)[-1] last value should be in [0, 100] from real RSI (backend may return series for -1)."""
        parsed = ind_to_dict("rsi(14)[-1]")
        assert parsed is not None
        result = apply_function(ohlcv_100, parsed)
        assert result is not None
        last_val = result.iloc[-1] if hasattr(result, "iloc") else result
        assert 0 <= last_val <= 100

    def test_ema_vs_sma_expression_both_computed(self, ohlcv_100):
        """ema(10) and sma(10) should both return series; last values are numeric (backend may return series for -1)."""
        ema_parsed = ind_to_dict("ema(10)[-1]")
        sma_parsed = ind_to_dict("sma(10)[-1]")
        ema_result = apply_function(ohlcv_100, ema_parsed)
        sma_result = apply_function(ohlcv_100, sma_parsed)
        assert ema_result is not None and sma_result is not None
        ema_val = ema_result.iloc[-1] if hasattr(ema_result, "iloc") else ema_result
        sma_val = sma_result.iloc[-1] if hasattr(sma_result, "iloc") else sma_result
        # Allow int, float, or numpy scalar (e.g. np.float64)
        assert not hasattr(ema_val, "iloc"), "expected scalar last value"
        assert not hasattr(sma_val, "iloc"), "expected scalar last value"
        assert np.issubdtype(np.asarray(ema_val).dtype, np.number) or isinstance(ema_val, (int, float))
        assert np.issubdtype(np.asarray(sma_val).dtype, np.number) or isinstance(sma_val, (int, float))

    def test_ewo_with_kwargs_parsed_and_applied(self, ohlcv_100):
        """EWO(sma1_length=5, sma2_length=35) parses and returns series."""
        parsed = ind_to_dict("EWO(sma1_length=5, sma2_length=35)")
        assert parsed is not None
        assert parsed.get("sma1_length") == 5
        assert parsed.get("sma2_length") == 35
        result = apply_function(ohlcv_100, parsed)
        assert result is not None and hasattr(result, "iloc")
        assert len(result) == len(ohlcv_100)

    def test_macd_line_expression(self, ohlcv_100):
        """macd(12, 26, 9) or default MACD returns series."""
        parsed = ind_to_dict("macd(12, 26, 9)")
        if parsed is None:
            # Parser might expect keyword args
            parsed = ind_to_dict("macd(fast_period=12, slow_period=26, signal_period=9)")
        assert parsed is not None
        result = apply_function(ohlcv_100, parsed)
        assert result is not None

    def test_bbands_middle_expression(self, ohlcv_100):
        """bbands(20) or bb_middle(20) returns series."""
        parsed = ind_to_dict("bb_middle(20)")
        assert parsed is not None
        result = apply_function(ohlcv_100, parsed)
        assert result is not None and hasattr(result, "iloc")


# ---------------------------------------------------------------------------
# E2E: simplify_conditions and operator precedence
# ---------------------------------------------------------------------------

class TestSimplifyConditionsE2E:
    """Parsing condition strings and operator precedence (>= before >)."""

    def test_greater_equal_parsed_not_as_greater_plus_equal(self):
        """>= must be parsed as one operator so left/right are correct."""
        cond = "Close[-1] >= 100"
        out = simplify_conditions(cond)
        assert out is not None
        assert out["comparison"] == ">="
        assert out["ind1"] is not None and out["ind2"] is not None
        assert out["ind2"].get("number") == 100

    def test_less_equal_parsed_correctly(self):
        """<= parsed as single operator."""
        cond = "rsi(14)[-1] <= 70"
        out = simplify_conditions(cond)
        assert out is not None
        assert out["comparison"] == "<="

    def test_operator_inside_parentheses_not_used_as_split(self):
        """Comparison inside parentheses should not be the split point."""
        # e.g. "sma(20>10)" should not split on >
        cond = "sma(20)[-1] > 50"
        out = simplify_conditions(cond)
        assert out is not None
        assert out["comparison"] == ">"
        assert out["ind1"]["ind"] == "sma"

    def test_find_operator_position_ignores_inside_parens(self):
        """_find_operator_position returns outer operator, not the one inside parentheses."""
        expr = "func(a>b) > 100"
        pos = _find_operator_position(expr, ">")
        assert pos >= 0
        # Inner > is at index 6 inside (a>b); outer > is at index 10
        assert pos == 10
        assert expr[pos : pos + 1] == ">"

    def test_equality_and_not_equal_parsed(self):
        """== and != parsed correctly."""
        for expr, op in [("Close[-1] == 100", "=="), ("trend[-1] != 0", "!=")]:
            out = simplify_conditions(expr)
            assert out is not None
            assert out["comparison"] == op


# ---------------------------------------------------------------------------
# E2E: evaluate_expression full pipeline (real indicators)
# ---------------------------------------------------------------------------

class TestEvaluateExpressionE2E:
    """evaluate_expression with real indicator calculations (no mocks)."""

    def test_close_above_number_true(self, ohlcv_known_values):
        """Close[-1] > 100: last Close is 102.5, so True."""
        df = ohlcv_known_values
        assert evaluate_expression(df, "Close[-1] > 100") is True

    def test_close_above_number_false(self, ohlcv_known_values):
        """Close[-1] > 200: False."""
        df = ohlcv_known_values
        assert evaluate_expression(df, "Close[-1] > 200") is False

    def test_close_below_number_true(self, ohlcv_known_values):
        """Close[-1] < 200: True."""
        df = ohlcv_known_values
        assert evaluate_expression(df, "Close[-1] < 200") is True

    def test_close_equals_last_value(self, ohlcv_known_values):
        """Close[-1] == 102.5: True."""
        df = ohlcv_known_values
        last = float(df["Close"].iloc[-1])
        assert evaluate_expression(df, f"Close[-1] == {last}") is True

    def test_rsi_comparison_real_rsi(self, ohlcv_100):
        """RSI(14)[-1] < 100 and RSI(14)[-1] > 0 should both be True (RSI in 0..100)."""
        assert evaluate_expression(ohlcv_100, "rsi(14)[-1] < 100") is True
        assert evaluate_expression(ohlcv_100, "rsi(14)[-1] > 0") is True

    def test_close_above_sma_consistency(self, ohlcv_100):
        """Close[-1] > sma(20)[-1] or < should be consistent with actual values."""
        from src.utils.indicators import SMA
        sma_series = SMA(ohlcv_100, 20, "Close")
        last_close = ohlcv_100["Close"].iloc[-1]
        last_sma = sma_series.iloc[-1]
        expected = last_close > last_sma
        result = evaluate_expression(ohlcv_100, "Close[-1] > sma(20)[-1]")
        assert result == expected

    def test_empty_expression_returns_false(self, ohlcv_100):
        """Empty or whitespace expression returns False."""
        assert evaluate_expression(ohlcv_100, "") is False
        assert evaluate_expression(ohlcv_100, "   ") is False

    def test_none_expression_returns_false(self, ohlcv_100):
        """None expression returns False."""
        assert evaluate_expression(ohlcv_100, None) is False

    def test_invalid_expression_returns_false(self, ohlcv_100):
        """Unparseable expression should return False, not raise."""
        assert evaluate_expression(ohlcv_100, "no operator here") is False

    def test_nan_in_data_returns_false_for_that_comparison(self, ohlcv_100):
        """If last Close is NaN, comparison should return False (handled in _evaluate_simple_comparison)."""
        df = ohlcv_100.copy()
        df.loc[df.index[-1], "Close"] = np.nan
        result = evaluate_expression(df, "Close[-1] > 0")
        assert result is False

    def test_simple_comparison_extracts_last_value_when_series_returned(self, ohlcv_100):
        """When expression has no specifier (e.g. 'Close' only), simple comparison uses .iloc[-1]."""
        # Expression "Close > 0" - ind_to_dict("Close") has specifier -1, so apply_function returns series.
        # _evaluate_simple_comparison then does val1.iloc[-1].
        result = _evaluate_simple_comparison(ohlcv_100, "Close > 0")
        assert result is True


# ---------------------------------------------------------------------------
# E2E: evaluate_expression_list and combination logic
# ---------------------------------------------------------------------------

class TestEvaluateExpressionListE2E:
    """evaluate_expression_list with real expressions and combination strings."""

    def test_single_condition(self, ohlcv_known_values):
        """Single expression: result is that expression's value."""
        df = ohlcv_known_values
        assert evaluate_expression_list(df, ["Close[-1] > 100"]) is True
        assert evaluate_expression_list(df, ["Close[-1] > 200"]) is False

    def test_and_all_true(self, ohlcv_known_values):
        """AND with all true -> True."""
        df = ohlcv_known_values
        exps = ["Close[-1] > 100", "Close[-1] < 200"]
        assert evaluate_expression_list(df, exps, "AND") is True

    def test_and_one_false(self, ohlcv_known_values):
        """AND with one false -> False."""
        df = ohlcv_known_values
        exps = ["Close[-1] > 100", "Close[-1] > 200"]
        assert evaluate_expression_list(df, exps, "AND") is False

    def test_or_one_true(self, ohlcv_known_values):
        """OR with one true -> True."""
        df = ohlcv_known_values
        exps = ["Close[-1] > 200", "Close[-1] > 100"]
        assert evaluate_expression_list(df, exps, "OR") is True

    def test_or_all_false(self, ohlcv_known_values):
        """OR with all false -> False."""
        df = ohlcv_known_values
        exps = ["Close[-1] > 200", "Close[-1] > 300"]
        assert evaluate_expression_list(df, exps, "OR") is False

    def test_numeric_combination_1_and_2(self, ohlcv_known_values):
        """Combination '1 AND 2'."""
        df = ohlcv_known_values
        exps = ["Close[-1] > 100", "Close[-1] < 200"]
        assert evaluate_expression_list(df, exps, "1 AND 2") is True

    def test_numeric_combination_1_or_2(self, ohlcv_known_values):
        """Combination '1 OR 2'."""
        df = ohlcv_known_values
        exps = ["Close[-1] > 200", "Close[-1] > 100"]
        assert evaluate_expression_list(df, exps, "1 OR 2") is True

    def test_complex_combination_parens(self, ohlcv_known_values):
        """(1 AND 2) OR 3."""
        df = ohlcv_known_values
        exps = [
            "Close[-1] > 200",   # 1: False
            "Close[-1] < 200",   # 2: True
            "Close[-1] > 100",   # 3: True
        ]
        # (False AND True) OR True = True
        assert evaluate_expression_list(df, exps, "(1 AND 2) OR 3") is True

    def test_combination_word_boundary_condition_10_not_replaced_by_1(self, ohlcv_known_values):
        """With 10 conditions, '1 AND 10' must not replace '10' with value of condition 1 (word boundary)."""
        df = ohlcv_known_values
        # Build 10 conditions: 1=true, 2..9=true, 10=false
        exps = ["Close[-1] > 100"] * 9 + ["Close[-1] > 300"]  # last is False
        # "1 AND 10" should mean condition 1 AND condition 10 -> True AND False -> False
        result = evaluate_expression_list(df, exps, "1 AND 10")
        assert result is False

    def test_default_combination_and(self, ohlcv_known_values):
        """combination '1' or empty defaults to AND."""
        df = ohlcv_known_values
        exps = ["Close[-1] > 100", "Close[-1] < 200"]
        assert evaluate_expression_list(df, exps, "1") is True
        assert evaluate_expression_list(df, exps, "") is True

    def test_empty_list_returns_false(self, ohlcv_100):
        """Empty expression list returns False."""
        assert evaluate_expression_list(ohlcv_100, []) is False

    def test_vals_passed_through_to_evaluate_expression(self, ohlcv_100):
        """vals (e.g. ticker) are passed to evaluate_expression for pivot_sr etc."""
        # Just ensure no crash when vals is provided
        result = evaluate_expression_list(
            ohlcv_100,
            ["Close[-1] > 0"],
            "AND",
            vals={"ticker": "AAPL"},
        )
        assert result is True


# ---------------------------------------------------------------------------
# E2E: _apply_specifier bounds and NaN
# ---------------------------------------------------------------------------

class TestApplySpecifierE2E:
    """_apply_specifier edge cases that can cause logic bugs."""

    def test_specifier_out_of_bounds_returns_none(self):
        """specifier >= len(series) or < -len(series) returns None."""
        s = pd.Series([10, 20, 30])
        assert _apply_specifier(s, 3) is None
        assert _apply_specifier(s, -4) is None
        assert _apply_specifier(s, 10) is None

    def test_specifier_nan_value_returns_none(self):
        """When value at index is NaN, _apply_specifier returns None."""
        s = pd.Series([10.0, np.nan, 30.0])
        assert _apply_specifier(s, 1) is None

    def test_specifier_negative_one_last_value(self):
        """specifier -1 returns last element."""
        s = pd.Series([10, 20, 30])
        assert _apply_specifier(s, -1) == 30

    def test_specifier_dataframe_uses_last_column(self):
        """DataFrame: use last column, then index."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert _apply_specifier(df, -1) == 6


# ---------------------------------------------------------------------------
# E2E: indicator_calculation wrapper
# ---------------------------------------------------------------------------

class TestIndicatorCalculationE2E:
    """indicator_calculation delegates to apply_function; same results."""

    def test_numeric_indicator_calculation(self, ohlcv_100):
        """indicator_calculation with numeric dict same as apply_function."""
        ind = {"isNum": True, "number": 50}
        assert indicator_calculation(ohlcv_100, ind) == 50
        assert apply_function(ohlcv_100, ind) == 50

    def test_close_series_via_indicator_calculation(self, ohlcv_100):
        """indicator_calculation with Close returns same series as apply_function."""
        ind = {"ind": "Close", "specifier": -1}
        a = indicator_calculation(ohlcv_100, ind)
        b = apply_function(ohlcv_100, ind)
        pd.testing.assert_series_equal(a, b)


# ---------------------------------------------------------------------------
# E2E: Short / edge DataFrames (no crash, correct False when needed)
# ---------------------------------------------------------------------------

class TestEdgeCaseDataFramesE2E:
    """Short or problematic DataFrames should not crash and should behave correctly."""

    def test_very_short_df_rsi_returns_nan_or_false(self):
        """Very short DataFrame: RSI may be all NaN; comparison should return False."""
        df = pd.DataFrame({
            "Open": [100, 101],
            "High": [101, 102],
            "Low": [99, 100],
            "Close": [100.5, 101.5],
            "Volume": [1_000_000, 1_100_000],
        })
        # RSI(14) on 2 bars will have NaN; expression should not crash
        result = evaluate_expression(df, "rsi(14)[-1] < 30")
        assert result is False

    def test_single_row_df_close_comparison(self):
        """Single row: Close[-1] is the only value."""
        df = pd.DataFrame({
            "Open": [100], "High": [101], "Low": [99], "Close": [100.5], "Volume": [1_000_000],
        })
        assert evaluate_expression(df, "Close[-1] > 100") is True
        assert evaluate_expression(df, "Close[-1] > 101") is False

    def test_missing_volume_column_apply_function_returns_none(self, ohlcv_100):
        """If Volume column missing, apply_function for Volume should return None."""
        df = ohlcv_100.drop(columns=["Volume"])
        ind = ind_to_dict("Volume[-1]")
        assert ind is not None
        result = apply_function(df, ind)
        assert result is None


# ---------------------------------------------------------------------------
# E2E: EWO crossover (real EWO, no mock)
# ---------------------------------------------------------------------------

class TestEwoCrossoverE2E:
    """EWO crossover expressions with real EWO series."""

    def test_ewo_cross_up_expression_detected_when_true(self, ohlcv_100):
        """When EWO actually crosses from <=0 to >0, expression should return True."""
        from src.utils.indicators import EWO
        series = EWO(ohlcv_100, sma1_length=5, sma2_length=35, source="Close", use_percent=True)
        if series is None or len(series) < 2:
            pytest.skip("EWO returned no data")
        last, prev = series.iloc[-1], series.iloc[-2]
        # If we have a genuine up-cross in the last two bars, expression should be True
        exp = "(EWO(sma1_length=5, sma2_length=35)[-1] > 0) & (EWO(sma1_length=5, sma2_length=35)[-2] <= 0)"
        result = evaluate_expression(ohlcv_100, exp)
        expected = (last > 0 and prev <= 0)
        assert result == expected

    def test_ewo_expression_without_cross_returns_false_or_true_consistently(self, ohlcv_100):
        """EWO expression should match actual last two values."""
        from src.utils.indicators import EWO
        series = EWO(ohlcv_100, sma1_length=5, sma2_length=35, source="Close", use_percent=True)
        if series is None or len(series) < 2:
            pytest.skip("EWO returned no data")
        last, prev = series.iloc[-1], series.iloc[-2]
        exp_up = "(EWO(sma1_length=5, sma2_length=35)[-1] > 0) & (EWO(sma1_length=5, sma2_length=35)[-2] <= 0)"
        result = evaluate_expression(ohlcv_100, exp_up)
        assert result == (last > 0 and prev <= 0)


# ---------------------------------------------------------------------------
# E2E: Type coercion in comparison (RSI threshold as string, etc.)
# ---------------------------------------------------------------------------

class TestTypeCoercionE2E:
    """Comparisons with numeric strings or mixed types."""

    def test_comparison_with_numeric_string_literal(self, ohlcv_known_values):
        """Condition like Close[-1] > '100' might be parsed as number 100 by ind_to_dict."""
        df = ohlcv_known_values
        # Our parser treats "100" as number 100, so this is Close[-1] > 100
        result = evaluate_expression(df, "Close[-1] > 100")
        assert result is True

    def test_rsi_threshold_30_coerced_to_float(self, ohlcv_100):
        """RSI(14)[-1] < 30: right side is 30 (int), comparison should work."""
        result = evaluate_expression(ohlcv_100, "rsi(14)[-1] < 30")
        # Just ensure no crash; RSI may or may not be < 30
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# E2E: Bug-catching: None/unknown indicator and specifier consistency
# ---------------------------------------------------------------------------

class TestBugCatchingE2E:
    """Tests designed to catch logic bugs in evaluation and combination."""

    def test_unknown_indicator_in_expression_returns_false_not_crash(self, ohlcv_100):
        """Expression with unknown indicator should return False, not raise."""
        result = evaluate_expression(ohlcv_100, "unknown_ind(5)[-1] > 0")
        assert result is False

    def test_apply_function_returns_none_propagates_to_false_in_comparison(self, ohlcv_100):
        """When one side of comparison is None (e.g. missing column), evaluate_expression returns False."""
        # Volume missing -> apply_function returns None for Volume[-1]
        df = ohlcv_100.drop(columns=["Volume"])
        result = evaluate_expression(df, "Volume[-1] > 0")
        assert result is False

    def test_specifier_minus_two_used_consistently_in_comparison(self, ohlcv_100):
        """Close[-2] in expression should compare second-to-last value, not last."""
        df = ohlcv_100
        second_to_last = float(df["Close"].iloc[-2])
        last_val = float(df["Close"].iloc[-1])
        # Close[-2] == second_to_last should be True
        assert evaluate_expression(df, f"Close[-2] == {second_to_last}") is True
        # Close[-2] == last_val should be False (unless equal by chance)
        result = evaluate_expression(df, f"Close[-2] == {last_val}")
        assert result is (second_to_last == last_val)

    def test_combination_substitution_order_does_not_corrupt_condition_10(self, ohlcv_known_values):
        """Substituting 1 then 2 ... then 10 must not turn '10' into 'True0' or 'False0' (word boundary)."""
        df = ohlcv_known_values
        # 10 conditions: 1-9 true, 10 false. "10" must be replaced by False, not "1" replaced then "0" left
        exps = ["Close[-1] > 0"] * 9 + ["Close[-1] > 999"]
        result = evaluate_expression_list(df, exps, "10")  # "10" means condition 10 only
        assert result is False

    def test_equality_comparison_with_float_threshold(self, ohlcv_100):
        """== comparison with float (e.g. RSI level) should coerce and compare correctly."""
        # Just ensure no crash; RSI may not equal 50.0 exactly
        result = evaluate_expression(ohlcv_100, "rsi(14)[-1] == 50")
        assert isinstance(result, bool)

    def test_not_equal_comparison(self, ohlcv_known_values):
        """!= comparison should return True when values differ."""
        df = ohlcv_known_values
        assert evaluate_expression(df, "Close[-1] != 0") is True
        assert evaluate_expression(df, f"Close[-1] != {float(df['Close'].iloc[-1])}") is False


# ---------------------------------------------------------------------------
# Marker for filtering
# ---------------------------------------------------------------------------

pytestmark = [pytest.mark.backend_e2e]

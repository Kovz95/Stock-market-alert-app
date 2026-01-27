"""Unit tests for scanning evaluator."""

import pandas as pd

from src.stock_alert.core.scanning.evaluator import (
    apply_zscore_indicator,
    format_condition_values,
    normalize_indicator_dict,
)


class TestNormalizeIndicatorDict:
    """Test suite for normalize_indicator_dict function."""

    def test_returns_none_for_none_input(self):
        """Test returns None when input is None."""
        result = normalize_indicator_dict(None)
        assert result is None

    def test_aliases_timeperiod_to_period(self):
        """Test aliases timeperiod to period for TA-Lib compatibility."""
        ind = {"timeperiod": 20, "source": "Close"}
        result = normalize_indicator_dict(ind)

        assert result is not None
        assert "period" in result
        assert result["period"] == 20
        assert "timeperiod" in result  # Original should still be there

    def test_preserves_existing_period(self):
        """Test doesn't override existing period field."""
        ind = {"timeperiod": 20, "period": 30, "source": "Close"}
        result = normalize_indicator_dict(ind)

        assert result is not None
        assert result["period"] == 30  # Original period preserved

    def test_returns_copy_of_dict(self):
        """Test returns a copy, not the original dict."""
        ind = {"period": 20}
        result = normalize_indicator_dict(ind)

        assert result is not ind
        assert result == ind


class TestApplyZscoreIndicator:
    """Test suite for apply_zscore_indicator function."""

    def test_returns_original_when_disabled(self):
        """Test returns original expression when z-score disabled."""
        expr = "Close[-1]"
        result = apply_zscore_indicator(expr, use_zscore=False, lookback=20)

        assert result == expr

    def test_wraps_simple_expression(self):
        """Test wraps simple expression in z-score."""
        expr = "Close[-1]"
        result = apply_zscore_indicator(expr, use_zscore=True, lookback=20)

        assert "zscore" in result
        assert "lookback=20" in result

    def test_skips_comparison_expressions(self):
        """Test doesn't wrap expressions with comparison operators."""
        expr = "Close[-1] > sma(period=20)[-1]"
        result = apply_zscore_indicator(expr, use_zscore=True, lookback=20)

        assert result == expr  # Should remain unchanged

    def test_skips_logical_expressions(self):
        """Test doesn't wrap expressions with logical operators."""
        expr = "Close[-1] > 100 and Volume[-1] > 1000000"
        result = apply_zscore_indicator(expr, use_zscore=True, lookback=20)

        assert result == expr  # Should remain unchanged

    def test_handles_empty_expression(self):
        """Test handles empty expression."""
        expr = ""
        result = apply_zscore_indicator(expr, use_zscore=True, lookback=20)

        assert result == expr

    def test_uses_custom_lookback(self):
        """Test uses custom lookback period."""
        expr = "rsi(period=14)[-1]"
        result = apply_zscore_indicator(expr, use_zscore=True, lookback=50)

        assert "lookback=50" in result

    def test_uses_default_lookback_on_invalid(self):
        """Test uses default lookback when invalid value provided."""
        expr = "Close[-1]"
        result = apply_zscore_indicator(expr, use_zscore=True, lookback="invalid")

        assert "lookback=20" in result  # Should use default


class TestFormatConditionValues:
    """Test suite for format_condition_values function."""

    def test_formats_empty_list(self):
        """Test returns empty string for empty list."""
        values = []
        result = format_condition_values(values)

        assert result == ""

    def test_formats_single_condition(self):
        """Test formats single condition value."""
        values = [{"left": 150.5, "right": 145.0, "op": ">"}]
        result = format_condition_values(values)

        assert "150.5000" in result or "150.5" in result
        assert ">" in result
        assert "145" in result

    def test_formats_multiple_conditions(self):
        """Test formats multiple condition values with pipe separator."""
        values = [
            {"left": 150.5, "right": 145.0, "op": ">"},
            {"left": 65.5, "right": 70.0, "op": "<"},
        ]
        result = format_condition_values(values)

        assert "|" in result  # Should be separated by pipe
        assert "1." in result  # First condition
        assert "2." in result  # Second condition

    def test_skips_none_values(self):
        """Test skips conditions with None values."""
        values = [
            None,  # Should be skipped
            {"left": 150.5, "right": 145.0, "op": ">"},
        ]
        result = format_condition_values(values)

        assert "1." in result  # Should be numbered as 1st
        assert "150" in result

    def test_formats_large_numbers(self):
        """Test formats large numbers with scientific notation."""
        values = [{"left": 5000000, "right": 4000000, "op": ">"}]
        result = format_condition_values(values)

        # Should use compact formatting for large numbers
        assert "5e" in result.lower() or "5000" in result

    def test_formats_small_numbers(self):
        """Test formats small numbers with scientific notation."""
        values = [{"left": 0.0001, "right": 0.0002, "op": "<"}]
        result = format_condition_values(values)

        # Should use compact formatting for very small numbers
        assert ("e" in result.lower() or "0.000" in result)

    def test_handles_null_right_value(self):
        """Test handles condition with no right value (indicator only)."""
        values = [{"left": 65.5, "right": None, "op": None}]
        result = format_condition_values(values)

        assert "65.5" in result or "65.50" in result
        assert ">" not in result  # No operator shown

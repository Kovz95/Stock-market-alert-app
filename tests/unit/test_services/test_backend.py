"""
Unit tests for the backend module.

Tests the core functionality for:
- Parsing indicator expressions (ind_to_dict)
- Calculating indicator values (apply_function, indicator_calculation)
- Evaluating conditional expressions (evaluate_expression, evaluate_expression_list)
- Parsing condition strings (simplify_conditions)
- Helper functions (_as_int, _as_float, _as_bool, _find_column)
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, Mock


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_ohlcv_df():
    """Create a sample OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
    
    # Generate realistic price data
    close = 100 + np.cumsum(np.random.randn(n) * 2)
    high = close + np.abs(np.random.randn(n))
    low = close - np.abs(np.random.randn(n))
    open_prices = close + np.random.randn(n) * 0.5
    volume = np.random.randint(1000000, 10000000, n)
    
    return pd.DataFrame({
        "Open": open_prices,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume
    }, index=dates)


@pytest.fixture
def small_ohlcv_df():
    """Create a small OHLCV DataFrame for testing specific cases."""
    return pd.DataFrame({
        "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "High": [101.0, 102.0, 103.0, 104.0, 105.0],
        "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
        "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
        "Volume": [1000000, 1100000, 1200000, 1300000, 1400000]
    })


# =============================================================================
# Tests for _SeriesIndexer Class
# =============================================================================

class TestSeriesIndexer:
    """Tests for the _SeriesIndexer helper class."""

    def test_getitem_positive_index(self):
        """Test accessing elements with positive index."""
        from src.services.backend import _SeriesIndexer
        
        series = pd.Series([10, 20, 30, 40, 50])
        indexer = _SeriesIndexer(series)
        
        assert indexer[0] == 10
        assert indexer[2] == 30
        assert indexer[4] == 50

    def test_getitem_negative_index(self):
        """Test accessing elements with negative index (Python-style)."""
        from src.services.backend import _SeriesIndexer
        
        series = pd.Series([10, 20, 30, 40, 50])
        indexer = _SeriesIndexer(series)
        
        assert indexer[-1] == 50
        assert indexer[-2] == 40
        assert indexer[-5] == 10

    def test_getitem_out_of_bounds_returns_nan(self):
        """Test that out of bounds index returns NaN."""
        from src.services.backend import _SeriesIndexer
        
        series = pd.Series([10, 20, 30])
        indexer = _SeriesIndexer(series)
        
        assert np.isnan(indexer[10])
        assert np.isnan(indexer[-10])

    def test_getattr_delegates_to_series(self):
        """Test that attribute access delegates to underlying series."""
        from src.services.backend import _SeriesIndexer
        
        series = pd.Series([10, 20, 30, 40, 50])
        indexer = _SeriesIndexer(series)
        
        assert indexer.mean() == 30.0
        assert len(indexer) == 5

    def test_array_conversion(self):
        """Test conversion to numpy array."""
        from src.services.backend import _SeriesIndexer
        
        series = pd.Series([10, 20, 30])
        indexer = _SeriesIndexer(series)
        
        arr = np.array(indexer)
        np.testing.assert_array_equal(arr, [10, 20, 30])

    def test_len(self):
        """Test __len__ method."""
        from src.services.backend import _SeriesIndexer
        
        series = pd.Series([10, 20, 30, 40])
        indexer = _SeriesIndexer(series)
        
        assert len(indexer) == 4


# =============================================================================
# Tests for ind_to_dict Function
# =============================================================================

class TestIndToDict:
    """Tests for the ind_to_dict parsing function."""

    def test_numeric_integer(self):
        """Test parsing integer numeric values."""
        from src.services.backend import ind_to_dict
        
        result = ind_to_dict("150")
        
        assert result["isNum"] is True
        assert result["number"] == 150
        assert result["operable"] is True

    def test_numeric_float(self):
        """Test parsing float numeric values."""
        from src.services.backend import ind_to_dict
        
        result = ind_to_dict("3.14")
        
        assert result["isNum"] is True
        assert result["number"] == 3.14

    def test_numeric_negative(self):
        """Test parsing negative numeric values."""
        from src.services.backend import ind_to_dict
        
        result = ind_to_dict("-2.5")
        
        assert result["isNum"] is True
        assert result["number"] == -2.5

    def test_simple_column_name(self):
        """Test parsing simple column names like 'Close'."""
        from src.services.backend import ind_to_dict
        
        result = ind_to_dict("Close")
        
        assert result["isNum"] is False
        assert result["ind"] == "Close"
        assert result["specifier"] == -1

    def test_column_with_bracket_specifier(self):
        """Test parsing column with bracket specifier like 'Close[-1]'."""
        from src.services.backend import ind_to_dict
        
        result = ind_to_dict("Close[-1]")
        
        assert result["isNum"] is False
        assert result["ind"] == "Close"
        assert result["specifier"] == -1

    def test_column_with_positive_specifier(self):
        """Test parsing column with positive bracket specifier."""
        from src.services.backend import ind_to_dict
        
        result = ind_to_dict("Close[0]")
        
        assert result["ind"] == "Close"
        assert result["specifier"] == 0

    def test_indicator_with_period(self):
        """Test parsing indicator with period like 'sma(20)'."""
        from src.services.backend import ind_to_dict
        
        result = ind_to_dict("sma(20)")
        
        assert result["ind"] == "sma"
        assert result["period"] == 20

    def test_indicator_with_period_and_specifier(self):
        """Test parsing indicator with period and specifier like 'rsi(14)[-1]'."""
        from src.services.backend import ind_to_dict
        
        result = ind_to_dict("rsi(14)[-1]")
        
        assert result["ind"] == "rsi"
        assert result["period"] == 14
        assert result["specifier"] == -1

    def test_indicator_with_kwargs(self):
        """Test parsing indicator with keyword arguments."""
        from src.services.backend import ind_to_dict
        
        result = ind_to_dict("EWO(sma1_length=5, sma2_length=35)")
        
        assert result["ind"] == "EWO"
        assert result["sma1_length"] == 5
        assert result["sma2_length"] == 35

    def test_indicator_with_string_param(self):
        """Test parsing indicator with string parameter."""
        from src.services.backend import ind_to_dict
        
        result = ind_to_dict("sma(20, input='High')")
        
        assert result["ind"] == "sma"
        assert result["period"] == 20
        assert result["input"] == "High"

    def test_empty_string_returns_none(self):
        """Test that empty string returns None."""
        from src.services.backend import ind_to_dict
        
        assert ind_to_dict("") is None
    
    def test_whitespace_returns_empty_ind(self):
        """Test that whitespace-only string returns dict with empty ind."""
        from src.services.backend import ind_to_dict
        
        # Whitespace gets stripped and results in empty 'ind' field
        result = ind_to_dict("   ")
        
        assert result is not None
        assert result["ind"] == ""

    def test_none_input_returns_none(self):
        """Test that None input returns None."""
        from src.services.backend import ind_to_dict
        
        assert ind_to_dict(None) is None

    def test_non_string_input_returns_none(self):
        """Test that non-string input returns None."""
        from src.services.backend import ind_to_dict
        
        assert ind_to_dict(123) is None
        assert ind_to_dict([]) is None

    def test_float_period(self):
        """Test parsing indicator with float period."""
        from src.services.backend import ind_to_dict
        
        result = ind_to_dict("indicator(2.5)")
        
        assert result["period"] == 2.5

    def test_specifier_with_negative_two(self):
        """Test parsing specifier with -2."""
        from src.services.backend import ind_to_dict
        
        result = ind_to_dict("Close[-2]")
        
        assert result["specifier"] == -2


# =============================================================================
# Tests for simplify_conditions Function
# =============================================================================

class TestSimplifyConditions:
    """Tests for the simplify_conditions parsing function."""

    def test_simple_greater_than(self):
        """Test parsing simple greater than comparison."""
        from src.services.backend import simplify_conditions
        
        result = simplify_conditions("Close[-1] > 150")
        
        assert result["comparison"] == ">"
        assert result["ind1"]["ind"] == "Close"
        assert result["ind2"]["isNum"] is True
        assert result["ind2"]["number"] == 150

    def test_simple_less_than(self):
        """Test parsing simple less than comparison."""
        from src.services.backend import simplify_conditions
        
        result = simplify_conditions("rsi(14)[-1] < 30")
        
        assert result["comparison"] == "<"
        assert result["ind1"]["ind"] == "rsi"
        assert result["ind1"]["period"] == 14
        assert result["ind2"]["number"] == 30

    def test_greater_than_or_equal(self):
        """Test parsing >= comparison."""
        from src.services.backend import simplify_conditions
        
        result = simplify_conditions("Close[-1] >= sma(20)[-1]")
        
        assert result["comparison"] == ">="
        assert result["ind1"]["ind"] == "Close"
        assert result["ind2"]["ind"] == "sma"

    def test_less_than_or_equal(self):
        """Test parsing <= comparison."""
        from src.services.backend import simplify_conditions
        
        result = simplify_conditions("rsi(14)[-1] <= 70")
        
        assert result["comparison"] == "<="

    def test_equality(self):
        """Test parsing == comparison."""
        from src.services.backend import simplify_conditions
        
        result = simplify_conditions("signal[-1] == 1")
        
        assert result["comparison"] == "=="

    def test_not_equal(self):
        """Test parsing != comparison."""
        from src.services.backend import simplify_conditions
        
        result = simplify_conditions("trend[-1] != 0")
        
        assert result["comparison"] == "!="

    def test_indicator_vs_indicator(self):
        """Test parsing comparison between two indicators."""
        from src.services.backend import simplify_conditions
        
        result = simplify_conditions("ema(10)[-1] > sma(20)[-1]")
        
        assert result["ind1"]["ind"] == "ema"
        assert result["ind1"]["period"] == 10
        assert result["ind2"]["ind"] == "sma"
        assert result["ind2"]["period"] == 20

    def test_empty_string_returns_none(self):
        """Test that empty string returns None."""
        from src.services.backend import simplify_conditions
        
        assert simplify_conditions("") is None
        assert simplify_conditions("   ") is None

    def test_none_returns_none(self):
        """Test that None returns None."""
        from src.services.backend import simplify_conditions
        
        assert simplify_conditions(None) is None

    def test_no_operator_returns_none(self):
        """Test that string without operator returns None."""
        from src.services.backend import simplify_conditions
        
        result = simplify_conditions("Close[-1]")
        
        assert result is None or result["comparison"] is None


# =============================================================================
# Tests for Helper Functions
# =============================================================================

class TestAsInt:
    """Tests for the _as_int helper function."""

    def test_converts_int(self):
        """Test converting integer value."""
        from src.services.backend import _as_int
        
        assert _as_int(10, 0) == 10

    def test_converts_float_to_int(self):
        """Test converting float to integer."""
        from src.services.backend import _as_int
        
        assert _as_int(10.7, 0) == 10

    def test_converts_string_int(self):
        """Test converting string integer."""
        from src.services.backend import _as_int
        
        assert _as_int("20", 0) == 20

    def test_returns_default_for_none(self):
        """Test returning default for None."""
        from src.services.backend import _as_int
        
        assert _as_int(None, 15) == 15

    def test_returns_default_for_invalid(self):
        """Test returning default for invalid input."""
        from src.services.backend import _as_int
        
        assert _as_int("abc", 25) == 25
        assert _as_int([], 25) == 25


class TestAsFloat:
    """Tests for the _as_float helper function."""

    def test_converts_float(self):
        """Test converting float value."""
        from src.services.backend import _as_float
        
        assert _as_float(3.14, 0.0) == 3.14

    def test_converts_int_to_float(self):
        """Test converting integer to float."""
        from src.services.backend import _as_float
        
        assert _as_float(10, 0.0) == 10.0

    def test_converts_string_float(self):
        """Test converting string float."""
        from src.services.backend import _as_float
        
        assert _as_float("2.5", 0.0) == 2.5

    def test_returns_default_for_none(self):
        """Test returning default for None."""
        from src.services.backend import _as_float
        
        assert _as_float(None, 1.5) == 1.5

    def test_returns_default_for_invalid(self):
        """Test returning default for invalid input."""
        from src.services.backend import _as_float
        
        assert _as_float("abc", 2.5) == 2.5


class TestAsBool:
    """Tests for the _as_bool helper function."""

    def test_returns_bool_unchanged(self):
        """Test that bool values are returned unchanged."""
        from src.services.backend import _as_bool
        
        assert _as_bool(True, False) is True
        assert _as_bool(False, True) is False

    def test_string_true_values(self):
        """Test string values that should be True."""
        from src.services.backend import _as_bool
        
        assert _as_bool("true", False) is True
        assert _as_bool("True", False) is True
        assert _as_bool("TRUE", False) is True
        assert _as_bool("1", False) is True
        assert _as_bool("yes", False) is True
        assert _as_bool("y", False) is True

    def test_string_false_values(self):
        """Test string values that should be False."""
        from src.services.backend import _as_bool
        
        assert _as_bool("false", True) is False
        assert _as_bool("False", True) is False
        assert _as_bool("0", True) is False
        assert _as_bool("no", True) is False
        assert _as_bool("n", True) is False

    def test_numeric_values(self):
        """Test numeric values."""
        from src.services.backend import _as_bool
        
        assert _as_bool(1, False) is True
        assert _as_bool(0, True) is False
        assert _as_bool(1.5, False) is True

    def test_returns_default_for_invalid(self):
        """Test returning default for invalid input."""
        from src.services.backend import _as_bool
        
        assert _as_bool("maybe", True) is True
        assert _as_bool("maybe", False) is False


class TestFindColumn:
    """Tests for the _find_column helper function."""

    def test_finds_exact_match(self):
        """Test finding exact column match."""
        from src.services.backend import _find_column
        
        df = pd.DataFrame({"zscore": [1, 2], "value": [3, 4]})
        
        assert _find_column(df, "zscore", "value") == "zscore"

    def test_case_insensitive_match(self):
        """Test case-insensitive column matching."""
        from src.services.backend import _find_column
        
        df = pd.DataFrame({"ZScore": [1, 2], "Value": [3, 4]})
        
        assert _find_column(df, "zscore", "Value") == "ZScore"

    def test_returns_fallback_when_not_found(self):
        """Test returning fallback when column not found."""
        from src.services.backend import _find_column
        
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        
        assert _find_column(df, "missing", "b") == "b"

    def test_returns_last_column_when_fallback_missing(self):
        """Test returning last column when fallback also missing."""
        from src.services.backend import _find_column
        
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        
        assert _find_column(df, "missing", "also_missing") == "b"


# =============================================================================
# Tests for apply_function
# =============================================================================

class TestApplyFunction:
    """Tests for the apply_function indicator calculation."""

    def test_numeric_value_returns_number(self):
        """Test that numeric indicator dict returns the number."""
        from src.services.backend import apply_function
        
        df = pd.DataFrame({"Close": [100, 101, 102]})
        ind = {"isNum": True, "number": 150, "operable": True}
        
        result = apply_function(df, ind)
        
        assert result == 150

    def test_returns_none_for_invalid_ind(self):
        """Test that invalid indicator dict returns None."""
        from src.services.backend import apply_function
        
        df = pd.DataFrame({"Close": [100, 101, 102]})
        
        assert apply_function(df, None) is None
        assert apply_function(df, {}) is None
        assert apply_function(df, "not a dict") is None

    def test_price_column_close(self, small_ohlcv_df):
        """Test accessing Close price column."""
        from src.services.backend import apply_function
        
        ind = {"isNum": False, "ind": "Close", "operable": True}
        
        result = apply_function(small_ohlcv_df, ind)
        
        assert isinstance(result, pd.Series)
        assert len(result) == 5

    def test_price_column_with_specifier_minus_one(self, small_ohlcv_df):
        """Test accessing price column with specifier -1 returns full series.
        
        Note: specifier -1 is treated as 'no specifier' in apply_function,
        so it returns the full series. The caller is responsible for extracting
        the last value if needed.
        """
        from src.services.backend import apply_function
        
        ind = {"isNum": False, "ind": "Close", "specifier": -1, "operable": True}
        
        result = apply_function(small_ohlcv_df, ind)
        
        # Returns full series when specifier is -1
        assert isinstance(result, pd.Series)
        assert len(result) == 5
    
    def test_price_column_with_specifier_minus_two(self, small_ohlcv_df):
        """Test accessing price column with specifier -2."""
        from src.services.backend import apply_function
        
        ind = {"isNum": False, "ind": "Close", "specifier": -2, "operable": True}
        
        result = apply_function(small_ohlcv_df, ind)
        
        assert result == 103.5  # Second to last Close value

    @patch("src.services.backend.SMA")
    def test_sma_indicator(self, mock_sma, small_ohlcv_df):
        """Test SMA indicator calculation."""
        from src.services.backend import apply_function
        
        mock_series = pd.Series([100, 101, 102, 103, 104])
        mock_sma.return_value = mock_series
        
        ind = {"isNum": False, "ind": "sma", "period": 20, "operable": True}
        
        result = apply_function(small_ohlcv_df, ind)
        
        mock_sma.assert_called_once()
        assert isinstance(result, pd.Series)

    @patch("src.services.backend.RSI")
    def test_rsi_indicator_with_specifier_minus_two(self, mock_rsi, small_ohlcv_df):
        """Test RSI indicator with specifier -2."""
        from src.services.backend import apply_function
        
        mock_series = pd.Series([30, 40, 50, 60, 70])
        mock_rsi.return_value = mock_series
        
        ind = {"isNum": False, "ind": "rsi", "period": 14, "specifier": -2, "operable": True}
        
        result = apply_function(small_ohlcv_df, ind)
        
        assert result == 60  # Second to last RSI value


# =============================================================================
# Tests for _apply_specifier
# =============================================================================

class TestApplySpecifier:
    """Tests for the _apply_specifier helper function."""

    def test_applies_negative_specifier(self):
        """Test applying negative specifier to series."""
        from src.services.backend import _apply_specifier
        
        series = pd.Series([10, 20, 30, 40, 50])
        
        assert _apply_specifier(series, -1) == 50
        assert _apply_specifier(series, -2) == 40

    def test_applies_positive_specifier(self):
        """Test applying positive specifier to series."""
        from src.services.backend import _apply_specifier
        
        series = pd.Series([10, 20, 30, 40, 50])
        
        assert _apply_specifier(series, 0) == 10
        assert _apply_specifier(series, 2) == 30

    def test_returns_none_for_out_of_bounds(self):
        """Test returning None for out of bounds specifier."""
        from src.services.backend import _apply_specifier
        
        series = pd.Series([10, 20, 30])
        
        assert _apply_specifier(series, 10) is None
        assert _apply_specifier(series, -10) is None

    def test_returns_none_for_none_series(self):
        """Test returning None when series is None."""
        from src.services.backend import _apply_specifier
        
        assert _apply_specifier(None, -1) is None

    def test_returns_none_for_nan_value(self):
        """Test returning None when value is NaN."""
        from src.services.backend import _apply_specifier
        
        series = pd.Series([10, np.nan, 30])
        
        assert _apply_specifier(series, 1) is None

    def test_handles_dataframe(self):
        """Test handling DataFrame by using last column."""
        from src.services.backend import _apply_specifier
        
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        
        result = _apply_specifier(df, -1)
        
        assert result == 6  # Last value of last column

    def test_returns_scalar_unchanged(self):
        """Test returning scalar values unchanged."""
        from src.services.backend import _apply_specifier
        
        assert _apply_specifier(42, -1) == 42
        assert _apply_specifier(3.14, 0) == 3.14


# =============================================================================
# Tests for evaluate_expression
# =============================================================================

class TestEvaluateExpression:
    """Tests for the evaluate_expression function."""

    def test_empty_expression_returns_false(self):
        """Test that empty expression returns False."""
        from src.services.backend import evaluate_expression
        
        df = pd.DataFrame({"Close": [100, 101, 102]})
        
        assert evaluate_expression(df, "") is False
        assert evaluate_expression(df, None) is False
        assert evaluate_expression(df, "   ") is False

    def test_simple_numeric_comparison(self, small_ohlcv_df):
        """Test simple numeric comparison."""
        from src.services.backend import evaluate_expression
        
        # Close[-1] is 104.5, which is > 100
        result = evaluate_expression(small_ohlcv_df, "Close[-1] > 100")
        
        assert result is True

    def test_simple_comparison_false(self, small_ohlcv_df):
        """Test simple comparison that is false."""
        from src.services.backend import evaluate_expression
        
        # Close[-1] is 104.5, which is not > 200
        result = evaluate_expression(small_ohlcv_df, "Close[-1] > 200")
        
        assert result is False

    @patch("src.services.backend.SMA")
    def test_indicator_comparison(self, mock_sma, small_ohlcv_df):
        """Test comparison involving indicator."""
        from src.services.backend import evaluate_expression
        
        # Mock SMA to return values below Close
        mock_sma.return_value = pd.Series([98, 99, 100, 101, 102])
        
        result = evaluate_expression(small_ohlcv_df, "Close[-1] > sma(20)[-1]")
        
        assert result is True

    def test_less_than_comparison(self, small_ohlcv_df):
        """Test less than comparison."""
        from src.services.backend import evaluate_expression
        
        result = evaluate_expression(small_ohlcv_df, "Close[-1] < 200")
        
        assert result is True


# =============================================================================
# Tests for _evaluate_simple_comparison
# =============================================================================

class TestEvaluateSimpleComparison:
    """Tests for the _evaluate_simple_comparison helper function."""

    def test_greater_than_true(self, small_ohlcv_df):
        """Test greater than comparison that is true."""
        from src.services.backend import _evaluate_simple_comparison
        
        result = _evaluate_simple_comparison(small_ohlcv_df, "Close[-1] > 100")
        
        assert result is True

    def test_greater_than_false(self, small_ohlcv_df):
        """Test greater than comparison that is false."""
        from src.services.backend import _evaluate_simple_comparison
        
        result = _evaluate_simple_comparison(small_ohlcv_df, "Close[-1] > 200")
        
        assert result is False

    def test_less_than_true(self, small_ohlcv_df):
        """Test less than comparison that is true."""
        from src.services.backend import _evaluate_simple_comparison
        
        result = _evaluate_simple_comparison(small_ohlcv_df, "Close[-1] < 200")
        
        assert result is True

    def test_greater_equal_true(self, small_ohlcv_df):
        """Test >= comparison that is true."""
        from src.services.backend import _evaluate_simple_comparison
        
        result = _evaluate_simple_comparison(small_ohlcv_df, "Close[-1] >= 104.5")
        
        assert result is True

    def test_less_equal_true(self, small_ohlcv_df):
        """Test <= comparison that is true."""
        from src.services.backend import _evaluate_simple_comparison
        
        result = _evaluate_simple_comparison(small_ohlcv_df, "Close[-1] <= 104.5")
        
        assert result is True

    def test_equality_true(self, small_ohlcv_df):
        """Test == comparison that is true."""
        from src.services.backend import _evaluate_simple_comparison
        
        result = _evaluate_simple_comparison(small_ohlcv_df, "Close[-1] == 104.5")
        
        assert result is True

    def test_not_equal_true(self, small_ohlcv_df):
        """Test != comparison that is true."""
        from src.services.backend import _evaluate_simple_comparison
        
        result = _evaluate_simple_comparison(small_ohlcv_df, "Close[-1] != 100")
        
        assert result is True

    def test_returns_none_for_invalid(self, small_ohlcv_df):
        """Test returning None for invalid expression."""
        from src.services.backend import _evaluate_simple_comparison
        
        result = _evaluate_simple_comparison(small_ohlcv_df, "invalid expression")
        
        assert result is None


# =============================================================================
# Tests for evaluate_expression_list
# =============================================================================

class TestEvaluateExpressionList:
    """Tests for the evaluate_expression_list function."""

    def test_empty_list_returns_false(self, small_ohlcv_df):
        """Test that empty expression list returns False."""
        from src.services.backend import evaluate_expression_list
        
        result = evaluate_expression_list(small_ohlcv_df, [])
        
        assert result is False

    def test_single_true_expression(self, small_ohlcv_df):
        """Test single expression that is true."""
        from src.services.backend import evaluate_expression_list
        
        result = evaluate_expression_list(small_ohlcv_df, ["Close[-1] > 100"])
        
        assert result is True

    def test_single_false_expression(self, small_ohlcv_df):
        """Test single expression that is false."""
        from src.services.backend import evaluate_expression_list
        
        result = evaluate_expression_list(small_ohlcv_df, ["Close[-1] > 200"])
        
        assert result is False

    def test_and_combination_all_true(self, small_ohlcv_df):
        """Test AND combination where all expressions are true."""
        from src.services.backend import evaluate_expression_list
        
        expressions = [
            "Close[-1] > 100",
            "Close[-1] < 200"
        ]
        
        result = evaluate_expression_list(small_ohlcv_df, expressions, "AND")
        
        assert result is True

    def test_and_combination_one_false(self, small_ohlcv_df):
        """Test AND combination where one expression is false."""
        from src.services.backend import evaluate_expression_list
        
        expressions = [
            "Close[-1] > 100",
            "Close[-1] > 200"  # False
        ]
        
        result = evaluate_expression_list(small_ohlcv_df, expressions, "AND")
        
        assert result is False

    def test_or_combination_one_true(self, small_ohlcv_df):
        """Test OR combination where one expression is true."""
        from src.services.backend import evaluate_expression_list
        
        expressions = [
            "Close[-1] > 200",  # False
            "Close[-1] > 100"   # True
        ]
        
        result = evaluate_expression_list(small_ohlcv_df, expressions, "OR")
        
        assert result is True

    def test_or_combination_all_false(self, small_ohlcv_df):
        """Test OR combination where all expressions are false."""
        from src.services.backend import evaluate_expression_list
        
        expressions = [
            "Close[-1] > 200",
            "Close[-1] > 300"
        ]
        
        result = evaluate_expression_list(small_ohlcv_df, expressions, "OR")
        
        assert result is False

    def test_numeric_combination_1_and_2(self, small_ohlcv_df):
        """Test numeric combination '1 AND 2'."""
        from src.services.backend import evaluate_expression_list
        
        expressions = [
            "Close[-1] > 100",  # 1: True
            "Close[-1] < 200"   # 2: True
        ]
        
        result = evaluate_expression_list(small_ohlcv_df, expressions, "1 AND 2")
        
        assert result is True

    def test_numeric_combination_1_or_2(self, small_ohlcv_df):
        """Test numeric combination '1 OR 2'."""
        from src.services.backend import evaluate_expression_list
        
        expressions = [
            "Close[-1] > 200",  # 1: False
            "Close[-1] > 100"   # 2: True
        ]
        
        result = evaluate_expression_list(small_ohlcv_df, expressions, "1 OR 2")
        
        assert result is True

    def test_complex_combination(self, small_ohlcv_df):
        """Test complex combination like '(1 AND 2) OR 3'."""
        from src.services.backend import evaluate_expression_list
        
        expressions = [
            "Close[-1] > 200",  # 1: False
            "Close[-1] < 200",  # 2: True
            "Close[-1] > 100"   # 3: True
        ]
        
        # (False AND True) OR True = False OR True = True
        result = evaluate_expression_list(small_ohlcv_df, expressions, "(1 AND 2) OR 3")
        
        assert result is True

    def test_default_combination_is_and(self, small_ohlcv_df):
        """Test that default combination (1) is treated as AND."""
        from src.services.backend import evaluate_expression_list
        
        expressions = [
            "Close[-1] > 100",
            "Close[-1] < 200"
        ]
        
        result = evaluate_expression_list(small_ohlcv_df, expressions, "1")
        
        assert result is True

    def test_handles_invalid_expressions(self, small_ohlcv_df):
        """Test handling of invalid expressions in list."""
        from src.services.backend import evaluate_expression_list
        
        expressions = [
            "Close[-1] > 100",
            "invalid expression"
        ]
        
        # Invalid expression evaluates to False, so AND fails
        result = evaluate_expression_list(small_ohlcv_df, expressions, "AND")
        
        assert result is False


# =============================================================================
# Tests for _find_operator_position
# =============================================================================

class TestFindOperatorPosition:
    """Tests for the _find_operator_position helper function."""

    def test_finds_greater_than(self):
        """Test finding > operator."""
        from src.services.backend import _find_operator_position
        
        result = _find_operator_position("Close[-1] > 100", ">")
        
        assert result == 10

    def test_finds_less_than(self):
        """Test finding < operator."""
        from src.services.backend import _find_operator_position
        
        result = _find_operator_position("Close[-1] < 100", "<")
        
        assert result == 10

    def test_finds_greater_equal(self):
        """Test finding >= operator."""
        from src.services.backend import _find_operator_position
        
        result = _find_operator_position("Close[-1] >= 100", ">=")
        
        assert result == 10

    def test_ignores_operator_in_parentheses(self):
        """Test that operator inside parentheses is ignored."""
        from src.services.backend import _find_operator_position
        
        # The > inside func(a>b) should be ignored, find the outer one
        result = _find_operator_position("func(a>b) > 100", ">")
        
        assert result == 10  # Position of outer >

    def test_ignores_operator_in_brackets(self):
        """Test that operator inside brackets is ignored."""
        from src.services.backend import _find_operator_position
        
        result = _find_operator_position("arr[x>0] > 100", ">")
        
        assert result == 9  # Position of outer >

    def test_returns_negative_when_not_found(self):
        """Test returning -1 when operator not found."""
        from src.services.backend import _find_operator_position
        
        result = _find_operator_position("Close[-1]", ">")
        
        assert result == -1


# =============================================================================
# Tests for indicator_calculation
# =============================================================================

class TestIndicatorCalculation:
    """Tests for the indicator_calculation wrapper function."""

    def test_delegates_to_apply_function(self, small_ohlcv_df):
        """Test that indicator_calculation delegates to apply_function."""
        from src.services.backend import indicator_calculation
        
        ind = {"isNum": True, "number": 42}
        
        result = indicator_calculation(small_ohlcv_df, ind)
        
        assert result == 42

    def test_handles_price_column(self, small_ohlcv_df):
        """Test handling price column indicator."""
        from src.services.backend import indicator_calculation
        
        # specifier -1 returns full series
        ind = {"isNum": False, "ind": "Close", "specifier": -1}
        
        result = indicator_calculation(small_ohlcv_df, ind)
        
        assert isinstance(result, pd.Series)
        assert result.iloc[-1] == 104.5
    
    def test_handles_price_column_with_specifier(self, small_ohlcv_df):
        """Test handling price column with specific index."""
        from src.services.backend import indicator_calculation
        
        ind = {"isNum": False, "ind": "Close", "specifier": -2}
        
        result = indicator_calculation(small_ohlcv_df, ind)
        
        assert result == 103.5


# =============================================================================
# Tests for _parse_params
# =============================================================================

class TestParseParams:
    """Tests for the _parse_params helper function."""

    def test_single_numeric_param(self):
        """Test parsing single numeric parameter."""
        from src.services.backend import _parse_params
        
        result = _parse_params("20")
        
        assert result["period"] == 20

    def test_single_float_param(self):
        """Test parsing single float parameter."""
        from src.services.backend import _parse_params
        
        result = _parse_params("2.5")
        
        assert result["period"] == 2.5

    def test_key_value_pairs(self):
        """Test parsing key=value pairs."""
        from src.services.backend import _parse_params
        
        result = _parse_params("fast=12, slow=26")
        
        assert result["fast"] == 12
        assert result["slow"] == 26

    def test_mixed_params(self):
        """Test parsing mixed positional and keyword params."""
        from src.services.backend import _parse_params
        
        result = _parse_params("20, input='High'")
        
        assert result["period"] == 20
        assert result["input"] == "High"

    def test_boolean_string_values(self):
        """Test parsing boolean string values."""
        from src.services.backend import _parse_params
        
        result = _parse_params("use_percent=True")
        
        assert result["use_percent"] is True

    def test_nested_parentheses(self):
        """Test handling nested parentheses."""
        from src.services.backend import _parse_params
        
        result = _parse_params("func=outer(inner), value=5")
        
        assert result["func"] == "outer(inner)"
        assert result["value"] == 5


# =============================================================================
# Tests for _process_param
# =============================================================================

class TestProcessParam:
    """Tests for the _process_param helper function."""

    def test_empty_param_ignored(self):
        """Test that empty parameter is ignored."""
        from src.services.backend import _process_param
        
        params = {}
        _process_param("", params)
        
        assert params == {}

    def test_key_value_param(self):
        """Test processing key=value parameter."""
        from src.services.backend import _process_param
        
        params = {}
        _process_param("period=20", params)
        
        assert params["period"] == 20

    def test_positional_integer(self):
        """Test processing positional integer."""
        from src.services.backend import _process_param
        
        params = {}
        _process_param("14", params)
        
        assert params["period"] == 14

    def test_quoted_string_value(self):
        """Test processing quoted string value."""
        from src.services.backend import _process_param
        
        params = {}
        _process_param("input='Close'", params)
        
        assert params["input"] == "Close"

    def test_positional_string(self):
        """Test processing positional string."""
        from src.services.backend import _process_param
        
        params = {}
        _process_param("'Close'", params)
        
        assert params["input"] == "Close"


# =============================================================================
# Tests for EWO Expression Handling
# =============================================================================

class TestEwoExpression:
    """Tests for EWO crossover expression evaluation."""

    @patch("src.services.backend.EWO")
    def test_ewo_upward_cross(self, mock_ewo, small_ohlcv_df):
        """Test EWO upward crossover detection."""
        from src.services.backend import _try_evaluate_ewo_cross
        
        # EWO crosses from negative to positive
        mock_ewo.return_value = pd.Series([-0.5, -0.3, -0.1, 0.2, 0.5])
        
        exp = "(EWO(sma1_length=5, sma2_length=35)[-1] > 0) & (EWO(sma1_length=5, sma2_length=35)[-2] <= 0)"
        
        result = _try_evaluate_ewo_cross(small_ohlcv_df, exp)
        
        assert result is True

    @patch("src.services.backend.EWO")
    def test_ewo_no_cross(self, mock_ewo, small_ohlcv_df):
        """Test EWO when no crossover occurs."""
        from src.services.backend import _try_evaluate_ewo_cross
        
        # EWO stays positive - no cross
        mock_ewo.return_value = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        
        exp = "(EWO(sma1_length=5, sma2_length=35)[-1] > 0) & (EWO(sma1_length=5, sma2_length=35)[-2] <= 0)"
        
        result = _try_evaluate_ewo_cross(small_ohlcv_df, exp)
        
        assert result is False

    def test_non_ewo_expression_returns_none(self, small_ohlcv_df):
        """Test that non-EWO expression returns None."""
        from src.services.backend import _try_evaluate_ewo_cross
        
        result = _try_evaluate_ewo_cross(small_ohlcv_df, "Close[-1] > 100")
        
        assert result is None


# =============================================================================
# Tests for _parse_ewo_kwargs
# =============================================================================

class TestParseEwoKwargs:
    """Tests for the _parse_ewo_kwargs helper function."""

    def test_parses_integer_params(self):
        """Test parsing integer parameters."""
        from src.services.backend import _parse_ewo_kwargs
        
        result = _parse_ewo_kwargs("sma1_length=5, sma2_length=35")
        
        assert result["sma1_length"] == 5
        assert result["sma2_length"] == 35

    def test_parses_string_params(self):
        """Test parsing string parameters."""
        from src.services.backend import _parse_ewo_kwargs
        
        result = _parse_ewo_kwargs("source='Close'")
        
        assert result["source"] == "Close"

    def test_parses_boolean_params(self):
        """Test parsing boolean parameters."""
        from src.services.backend import _parse_ewo_kwargs
        
        result = _parse_ewo_kwargs("use_percent=True")
        
        assert result["use_percent"] is True

    def test_handles_empty_string(self):
        """Test handling empty string."""
        from src.services.backend import _parse_ewo_kwargs
        
        result = _parse_ewo_kwargs("")
        
        assert result == {}


# =============================================================================
# Tests for _build_eval_context
# =============================================================================

class TestBuildEvalContext:
    """Tests for the _build_eval_context function."""

    def test_includes_price_columns(self, small_ohlcv_df):
        """Test that context includes price columns."""
        from src.services.backend import _build_eval_context
        
        ctx = _build_eval_context(small_ohlcv_df)
        
        assert "Close" in ctx
        assert "Open" in ctx
        assert "High" in ctx
        assert "Low" in ctx
        assert "Volume" in ctx

    def test_includes_indicator_functions(self, small_ohlcv_df):
        """Test that context includes indicator functions."""
        from src.services.backend import _build_eval_context
        
        ctx = _build_eval_context(small_ohlcv_df)
        
        assert "sma" in ctx
        assert "ema" in ctx
        assert "rsi" in ctx
        assert callable(ctx["sma"])

    def test_includes_zscore_function(self, small_ohlcv_df):
        """Test that context includes zscore function."""
        from src.services.backend import _build_eval_context
        
        ctx = _build_eval_context(small_ohlcv_df)
        
        assert "zscore" in ctx
        assert callable(ctx["zscore"])


# =============================================================================
# Tests for Indicator Calculation Integration
# =============================================================================

class TestIndicatorCalculationIntegration:
    """Integration tests for indicator calculations."""

    @patch("src.services.backend.SMA")
    def test_sma_with_default_period(self, mock_sma, small_ohlcv_df):
        """Test SMA calculation with default period."""
        from src.services.backend import _calculate_indicator
        
        mock_sma.return_value = pd.Series([100, 101, 102, 103, 104])
        
        ind = {"ind": "sma"}
        result = _calculate_indicator(small_ohlcv_df, ind, "sma")
        
        mock_sma.assert_called_once_with(small_ohlcv_df, 20, "Close")

    @patch("src.services.backend.EMA")
    def test_ema_with_custom_period(self, mock_ema, small_ohlcv_df):
        """Test EMA calculation with custom period."""
        from src.services.backend import _calculate_indicator
        
        mock_ema.return_value = pd.Series([100, 101, 102, 103, 104])
        
        ind = {"ind": "ema", "period": 50, "input": "High"}
        result = _calculate_indicator(small_ohlcv_df, ind, "ema")
        
        mock_ema.assert_called_once_with(small_ohlcv_df, 50, "High")

    @patch("src.services.backend.RSI")
    def test_rsi_indicator(self, mock_rsi, small_ohlcv_df):
        """Test RSI indicator calculation."""
        from src.services.backend import _calculate_indicator
        
        mock_rsi.return_value = pd.Series([30, 40, 50, 60, 70])
        
        ind = {"ind": "rsi", "period": 14}
        result = _calculate_indicator(small_ohlcv_df, ind, "rsi")
        
        mock_rsi.assert_called_once_with(small_ohlcv_df, 14, "Close")

    @patch("src.services.backend.MACD")
    def test_macd_indicator(self, mock_macd, small_ohlcv_df):
        """Test MACD indicator calculation."""
        from src.services.backend import _calculate_indicator
        
        mock_macd.return_value = pd.Series([0.5, 0.6, 0.7, 0.8, 0.9])
        
        ind = {"ind": "macd", "fast_period": 12, "slow_period": 26, "signal_period": 9}
        result = _calculate_indicator(small_ohlcv_df, ind, "macd")
        
        mock_macd.assert_called_once()

    @patch("src.services.backend.BBANDS")
    def test_bollinger_bands(self, mock_bbands, small_ohlcv_df):
        """Test Bollinger Bands calculation."""
        from src.services.backend import _calculate_indicator
        
        mock_bbands.return_value = pd.Series([100, 101, 102, 103, 104])
        
        ind = {"ind": "bbands", "period": 20, "std_dev": 2.0, "output": "upper"}
        result = _calculate_indicator(small_ohlcv_df, ind, "bbands")
        
        mock_bbands.assert_called_once_with(small_ohlcv_df, 20, 2.0, "upper")

    def test_unknown_indicator_returns_none(self, small_ohlcv_df):
        """Test that unknown indicator returns None."""
        from src.services.backend import _calculate_indicator
        
        ind = {"ind": "unknown_indicator"}
        result = _calculate_indicator(small_ohlcv_df, ind, "unknown_indicator")
        
        assert result is None

    def test_price_column_returns_series(self, small_ohlcv_df):
        """Test that price column returns the series."""
        from src.services.backend import _calculate_indicator
        
        ind = {"ind": "close"}
        result = _calculate_indicator(small_ohlcv_df, ind, "close")
        
        assert isinstance(result, pd.Series)
        pd.testing.assert_series_equal(result, small_ohlcv_df["Close"])


# =============================================================================
# Tests for Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_ind_to_dict_with_whitespace(self):
        """Test ind_to_dict handles whitespace correctly."""
        from src.services.backend import ind_to_dict
        
        result = ind_to_dict("  Close[-1]  ")
        
        assert result["ind"] == "Close"
        assert result["specifier"] == -1

    def test_simplify_conditions_with_spaces(self):
        """Test simplify_conditions handles extra spaces."""
        from src.services.backend import simplify_conditions
        
        result = simplify_conditions("  Close[-1]   >   100  ")
        
        assert result["comparison"] == ">"

    def test_evaluate_expression_with_nan_values(self):
        """Test evaluate_expression handles NaN values."""
        from src.services.backend import evaluate_expression
        
        df = pd.DataFrame({
            "Close": [100, np.nan, 102],
            "Open": [99, 100, 101],
            "High": [101, 102, 103],
            "Low": [98, 99, 100],
            "Volume": [1000, 1100, 1200]
        })
        
        # Close[-1] is 102, should work
        result = evaluate_expression(df, "Close[-1] > 100")
        
        assert result is True

    def test_apply_function_with_missing_column(self):
        """Test apply_function handles missing column gracefully."""
        from src.services.backend import apply_function
        
        df = pd.DataFrame({"Close": [100, 101, 102]})
        ind = {"isNum": False, "ind": "Volume"}
        
        result = apply_function(df, ind)
        
        assert result is None

    def test_evaluate_expression_list_with_not_operator(self, small_ohlcv_df):
        """Test evaluate_expression_list with NOT operator."""
        from src.services.backend import evaluate_expression_list
        
        expressions = [
            "Close[-1] > 200"  # False
        ]
        
        result = evaluate_expression_list(small_ohlcv_df, expressions, "NOT 1")
        
        assert result is True

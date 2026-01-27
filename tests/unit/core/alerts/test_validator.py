"""Unit tests for alert validation."""

from src.stock_alert.core.alerts.validator import (
    _conditions_to_storage_format,
    _normalize_conditions_for_comparison,
    validate_conditions,
)


class TestValidateConditions:
    """Test suite for validate_conditions function."""

    def test_validate_empty_list(self):
        """Test validation fails for empty list."""
        assert validate_conditions([]) is False

    def test_validate_empty_string_in_list(self):
        """Test validation fails for empty string in list."""
        assert validate_conditions([""]) is False
        assert validate_conditions(["   "]) is False

    def test_validate_unclosed_brackets(self):
        """Test validation fails for unclosed brackets."""
        assert validate_conditions(["Close[-1] > sma(period=20)[-1"]) is False
        assert validate_conditions(["Close[-1 > sma(period=20)[-1]"]) is False

    def test_validate_valid_string_conditions(self):
        """Test validation succeeds for valid string conditions."""
        valid_conditions = [
            "Close[-1] > sma(period=20)[-1]",
            "rsi(period=14)[-1] < 30",
        ]
        assert validate_conditions(valid_conditions) is True

    def test_validate_dict_format(self):
        """Test validation with dict format (from Add_Alert page)."""
        dict_conditions = {
            "1": {"conditions": ["Close[-1] > sma(period=20)[-1]"]},
            "2": {"conditions": ["rsi(period=14)[-1] < 30"]},
        }
        assert validate_conditions(dict_conditions) is True

    def test_validate_dict_with_empty_conditions(self):
        """Test validation fails for dict with empty conditions."""
        dict_conditions = {
            "1": {"conditions": [""]},
        }
        assert validate_conditions(dict_conditions) is False

    def test_validate_legacy_dict_format(self):
        """Test validation with legacy dict format."""
        legacy_conditions = [
            {"index": 1, "conditions": "Close[-1] > sma(period=20)[-1]"},
            {"index": 2, "conditions": "rsi(period=14)[-1] < 30"},
        ]
        assert validate_conditions(legacy_conditions) is True


class TestNormalizeConditions:
    """Test suite for condition normalization functions."""

    def test_normalize_dict_conditions(self):
        """Test normalization of dict conditions."""
        dict_conditions = {
            "1": {"conditions": ["Close[-1] > 100", "rsi(period=14)[-1] < 30"]},
            "2": {"conditions": ["macd(fast_period=12, slow_period=26)[-1] > 0"]},
        }
        result = _normalize_conditions_for_comparison(dict_conditions)
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(item, dict) for item in result)
        assert all("index" in item and "conditions" in item for item in result)

    def test_normalize_list_conditions(self):
        """Test normalization returns list as-is."""
        list_conditions = [
            {"index": 1, "conditions": "Close[-1] > 100"},
            {"index": 2, "conditions": "rsi(period=14)[-1] < 30"},
        ]
        result = _normalize_conditions_for_comparison(list_conditions)
        assert result == list_conditions

    def test_conditions_to_storage_format_dict(self):
        """Test conversion to storage format from dict."""
        dict_conditions = {
            "1": {"conditions": ["Close[-1] > 100", "rsi(period=14)[-1] < 30"]},
        }
        result = _conditions_to_storage_format(dict_conditions)
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(item, dict) for item in result)

    def test_conditions_to_storage_format_list(self):
        """Test conversion to storage format from list."""
        list_conditions = [
            {"index": 1, "conditions": "Close[-1] > 100"},
        ]
        result = _conditions_to_storage_format(list_conditions)
        assert result == list_conditions

"""Unit tests for indicator catalog."""

from src.stock_alert.core.indicators.catalog import (
    inverse_map,
    ops,
    period_and_input,
    period_only,
    predefined_suggestions,
    supported_indicators,
)


class TestIndicatorCatalog:
    """Test suite for indicator catalog configurations."""

    def test_inverse_map_exists(self):
        """Test inverse operator map is defined."""
        assert inverse_map is not None
        assert isinstance(inverse_map, dict)
        assert ">" in inverse_map
        assert inverse_map[">"] == "<="

    def test_ops_exists(self):
        """Test operator functions map is defined."""
        assert ops is not None
        assert isinstance(ops, dict)
        assert ">" in ops
        assert callable(ops[">"])

    def test_operator_functions_work(self):
        """Test operator functions are callable and work correctly."""
        assert ops[">"](10, 5) is True
        assert ops["<"](5, 10) is True
        assert ops["=="](5, 5) is True
        assert ops[">="](10, 10) is True
        assert ops["<="](5, 5) is True
        assert ops["!="](5, 10) is True

    def test_supported_indicators_exists(self):
        """Test supported indicators dict is defined."""
        assert supported_indicators is not None
        assert isinstance(supported_indicators, dict)
        assert "sma" in supported_indicators
        assert "ema" in supported_indicators
        assert "rsi" in supported_indicators

    def test_predefined_suggestions_exists(self):
        """Test predefined suggestions list is defined."""
        assert predefined_suggestions is not None
        assert isinstance(predefined_suggestions, list)
        assert len(predefined_suggestions) > 0
        assert any("sma" in s for s in predefined_suggestions)

    def test_period_lists_exist(self):
        """Test period configuration lists are defined."""
        assert period_and_input is not None
        assert isinstance(period_and_input, list)
        assert "sma" in period_and_input

        assert period_only is not None
        assert isinstance(period_only, list)
        assert "sma" in period_only

    def test_indicator_catalog_completeness(self):
        """Test that key indicators are in the catalog."""
        required_indicators = ["sma", "ema", "rsi", "macd", "bb", "atr"]
        for indicator in required_indicators:
            assert indicator in supported_indicators, f"{indicator} not in supported_indicators"

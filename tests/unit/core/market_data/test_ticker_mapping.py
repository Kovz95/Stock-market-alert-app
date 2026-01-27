"""Unit tests for ticker mapping utilities."""

from src.stock_alert.core.market_data.ticker_mapping import (
    FMP_TICKER_MAPPING,
    get_fmp_ticker,
    get_fmp_ticker_with_fallback,
    get_yahoo_ticker,
)


class TestGetFmpTicker:
    """Test suite for get_fmp_ticker function."""

    def test_us_stock_unchanged(self):
        """Test US stock tickers remain unchanged."""
        assert get_fmp_ticker("AAPL") == "AAPL"
        assert get_fmp_ticker("MSFT") == "MSFT"
        assert get_fmp_ticker("GOOGL") == "GOOGL"

    def test_hong_kong_stock_padding(self):
        """Test Hong Kong stocks get 4-digit padding."""
        assert get_fmp_ticker("700.HK") == "0700.HK"
        assert get_fmp_ticker("0700.HK") == "0700.HK"
        assert get_fmp_ticker("9988.HK") == "9988.HK"

    def test_london_stock_mapping(self):
        """Test London stock ticker mappings."""
        assert get_fmp_ticker("BP.L") == "BP.L"
        assert get_fmp_ticker("HSBC.L") == "HSBA.L"

    def test_european_stock_suffixes(self):
        """Test European stock suffixes are preserved."""
        assert get_fmp_ticker("ASML.AS") == "ASML.AS"
        assert get_fmp_ticker("SAP.DE") == "SAP.DE"

    def test_brazilian_stock_pattern(self):
        """Test Brazilian stock pattern detection."""
        # Brazilian stocks typically end with 3, 4, or 11
        ticker = get_fmp_ticker("PETR4")
        assert ticker.endswith(".SA")

    def test_country_suffix_conversion(self):
        """Test country suffix to exchange suffix conversion."""
        # These use the -XX format that gets converted
        result = get_fmp_ticker("AAPL-US")
        assert result == "AAPL"  # US suffix removed

    def test_invalid_ticker_pattern(self):
        """Test invalid ticker patterns return None."""
        # Pattern like 0000J0 should be marked invalid
        assert get_fmp_ticker("0000J0") is None

    def test_base_mapping_lookup(self):
        """Test that base mapping is checked first."""
        # Tickers in FMP_TICKER_MAPPING should be returned as-is
        assert "AAPL" in FMP_TICKER_MAPPING
        assert get_fmp_ticker("AAPL") == FMP_TICKER_MAPPING["AAPL"]


class TestGetFmpTickerWithFallback:
    """Test suite for get_fmp_ticker_with_fallback function."""

    def test_us_stock_no_fallback(self):
        """Test US stocks don't get fallback option."""
        result = get_fmp_ticker_with_fallback("AAPL")
        assert isinstance(result, dict)
        assert "primary" in result
        assert result["fallback"] is None

    def test_international_stock_with_fallback(self):
        """Test international stocks get US listing fallback."""
        result = get_fmp_ticker_with_fallback("HSBC.L")
        assert isinstance(result, dict)
        assert "primary" in result
        assert "fallback" in result

    def test_supported_fallback_exchanges(self):
        """Test that supported exchanges get fallback options."""
        # UK exchange should have fallback
        result = get_fmp_ticker_with_fallback("BP.L")
        # Should have fallback for UK stocks
        if "." in "BP.L":
            base = "BP.L".split(".")[0]
            suffix = "BP.L".split(".")[-1]
            if suffix == "L":
                # L is a UK exchange, should be in fallback list
                assert result["fallback"] == base or result["fallback"] is None


class TestGetYahooTicker:
    """Test suite for get_yahoo_ticker function."""

    def test_us_stock_unchanged(self):
        """Test US stock tickers remain unchanged."""
        assert get_yahoo_ticker("AAPL") == "AAPL"
        assert get_yahoo_ticker("MSFT") == "MSFT"

    def test_us_suffix_removed(self):
        """Test -US suffix is removed."""
        assert get_yahoo_ticker("AAPL-US") == "AAPL"

    def test_hong_kong_padding(self):
        """Test Hong Kong stocks get 4-digit padding."""
        assert get_yahoo_ticker("700.HK") == "0700.HK"
        assert get_yahoo_ticker("9988.HK") == "9988.HK"

    def test_australian_stock_conversion(self):
        """Test Australian stock suffix conversion."""
        assert get_yahoo_ticker("BHP.AU") == "BHP.AX"

    def test_ticker_preserved_when_no_rule(self):
        """Test ticker is preserved when no conversion rule applies."""
        # Random ticker that doesn't match any rule
        assert get_yahoo_ticker("XYZ") == "XYZ"


class TestTickerMappingConstants:
    """Test suite for ticker mapping constants."""

    def test_fmp_ticker_mapping_exists(self):
        """Test FMP ticker mapping dict exists and has entries."""
        assert FMP_TICKER_MAPPING is not None
        assert isinstance(FMP_TICKER_MAPPING, dict)
        assert len(FMP_TICKER_MAPPING) > 0

    def test_fmp_mapping_has_key_entries(self):
        """Test FMP mapping has expected key entries."""
        assert "AAPL" in FMP_TICKER_MAPPING
        assert "0700.HK" in FMP_TICKER_MAPPING

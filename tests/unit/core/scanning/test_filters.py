"""Unit tests for scanning filters."""

from src.stock_alert.core.scanning.filters import (
    count_filtered_symbols,
    extract_unique_values_from_stock_db,
    get_filtered_symbols,
    symbol_matches_filters,
)


class TestSymbolMatchesFilters:
    """Test suite for symbol_matches_filters function."""

    def test_matches_all_filters(self):
        """Test symbol matches when all filters pass."""
        ticker = "AAPL"
        info = {
            "asset_type": "Stock",
            "country": "US",
            "exchange": "NASDAQ",
            "rbics_economy": "Consumer",
            "rbics_sector": "Technology",
        }

        result = symbol_matches_filters(
            ticker,
            info,
            selected_portfolio="All",
            portfolio_symbols=set(),
            asset_type_filter=["Stock"],
            selected_countries=["US"],
            selected_exchanges=["NASDAQ"],
            selected_economies=["Consumer"],
            selected_sectors=["Technology"],
            selected_subsectors=[],
            selected_industry_groups=[],
            selected_industries=[],
            selected_subindustries=[],
        )

        assert result is True

    def test_fails_portfolio_filter(self):
        """Test symbol doesn't match when not in portfolio."""
        ticker = "AAPL"
        info = {"asset_type": "Stock"}

        result = symbol_matches_filters(
            ticker,
            info,
            selected_portfolio="My Portfolio",
            portfolio_symbols={"MSFT", "GOOGL"},  # AAPL not in portfolio
            asset_type_filter=[],
            selected_countries=[],
            selected_exchanges=[],
            selected_economies=[],
            selected_sectors=[],
            selected_subsectors=[],
            selected_industry_groups=[],
            selected_industries=[],
            selected_subindustries=[],
        )

        assert result is False

    def test_fails_asset_type_filter(self):
        """Test symbol doesn't match when asset type filtered out."""
        ticker = "AAPL"
        info = {"asset_type": "Stock"}

        result = symbol_matches_filters(
            ticker,
            info,
            selected_portfolio="All",
            portfolio_symbols=set(),
            asset_type_filter=["ETF"],  # Only ETFs allowed
            selected_countries=[],
            selected_exchanges=[],
            selected_economies=[],
            selected_sectors=[],
            selected_subsectors=[],
            selected_industry_groups=[],
            selected_industries=[],
            selected_subindustries=[],
        )

        assert result is False

    def test_fails_country_filter(self):
        """Test symbol doesn't match when country filtered out."""
        ticker = "AAPL"
        info = {"asset_type": "Stock", "country": "US"}

        result = symbol_matches_filters(
            ticker,
            info,
            selected_portfolio="All",
            portfolio_symbols=set(),
            asset_type_filter=[],
            selected_countries=["UK", "DE"],  # Only UK and DE allowed
            selected_exchanges=[],
            selected_economies=[],
            selected_sectors=[],
            selected_subsectors=[],
            selected_industry_groups=[],
            selected_industries=[],
            selected_subindustries=[],
        )

        assert result is False

    def test_rbics_filters_only_for_stocks(self):
        """Test RBICS filters only apply to stocks, not ETFs."""
        ticker = "SPY"
        info = {"asset_type": "ETF", "country": "US"}

        # Should pass even though economy filter is set (ETFs ignore RBICS)
        result = symbol_matches_filters(
            ticker,
            info,
            selected_portfolio="All",
            portfolio_symbols=set(),
            asset_type_filter=["ETF"],
            selected_countries=["US"],
            selected_exchanges=[],
            selected_economies=["Technology"],  # This should be ignored for ETFs
            selected_sectors=[],
            selected_subsectors=[],
            selected_industry_groups=[],
            selected_industries=[],
            selected_subindustries=[],
        )

        assert result is True


class TestCountFilteredSymbols:
    """Test suite for count_filtered_symbols function."""

    def test_counts_all_with_no_filters(self):
        """Test counts all symbols when no filters applied."""
        stock_db = {
            "AAPL": {"asset_type": "Stock", "country": "US"},
            "MSFT": {"asset_type": "Stock", "country": "US"},
            "SPY": {"asset_type": "ETF", "country": "US"},
        }

        count = count_filtered_symbols(
            stock_db,
            selected_portfolio="All",
            portfolio_symbols=set(),
            asset_type_filter=[],
            selected_countries=[],
            selected_exchanges=[],
            selected_economies=[],
            selected_sectors=[],
            selected_subsectors=[],
            selected_industry_groups=[],
            selected_industries=[],
            selected_subindustries=[],
        )

        assert count == 3

    def test_counts_with_asset_filter(self):
        """Test counts only matching asset types."""
        stock_db = {
            "AAPL": {"asset_type": "Stock", "country": "US"},
            "MSFT": {"asset_type": "Stock", "country": "US"},
            "SPY": {"asset_type": "ETF", "country": "US"},
        }

        count = count_filtered_symbols(
            stock_db,
            selected_portfolio="All",
            portfolio_symbols=set(),
            asset_type_filter=["Stock"],  # Only stocks
            selected_countries=[],
            selected_exchanges=[],
            selected_economies=[],
            selected_sectors=[],
            selected_subsectors=[],
            selected_industry_groups=[],
            selected_industries=[],
            selected_subindustries=[],
        )

        assert count == 2  # AAPL and MSFT


class TestGetFilteredSymbols:
    """Test suite for get_filtered_symbols function."""

    def test_returns_all_with_no_filters(self):
        """Test returns all symbols when no filters applied."""
        stock_db = {
            "AAPL": {"asset_type": "Stock", "country": "US"},
            "MSFT": {"asset_type": "Stock", "country": "US"},
        }

        result = get_filtered_symbols(
            stock_db,
            selected_portfolio="All",
            portfolio_symbols=set(),
            asset_type_filter=[],
            selected_countries=[],
            selected_exchanges=[],
            selected_economies=[],
            selected_sectors=[],
            selected_subsectors=[],
            selected_industry_groups=[],
            selected_industries=[],
            selected_subindustries=[],
        )

        assert len(result) == 2
        assert ("AAPL", stock_db["AAPL"]) in result
        assert ("MSFT", stock_db["MSFT"]) in result

    def test_returns_filtered_list(self):
        """Test returns only symbols matching filters."""
        stock_db = {
            "AAPL": {"asset_type": "Stock", "country": "US"},
            "BP.L": {"asset_type": "Stock", "country": "UK"},
        }

        result = get_filtered_symbols(
            stock_db,
            selected_portfolio="All",
            portfolio_symbols=set(),
            asset_type_filter=["Stock"],
            selected_countries=["US"],  # Only US stocks
            selected_exchanges=[],
            selected_economies=[],
            selected_sectors=[],
            selected_subsectors=[],
            selected_industry_groups=[],
            selected_industries=[],
            selected_subindustries=[],
        )

        assert len(result) == 1
        assert result[0][0] == "AAPL"


class TestExtractUniqueValues:
    """Test suite for extract_unique_values_from_stock_db function."""

    def test_extracts_unique_values(self):
        """Test extracts all unique values for a field."""
        stock_db = {
            "AAPL": {"country": "US", "asset_type": "Stock"},
            "MSFT": {"country": "US", "asset_type": "Stock"},
            "BP.L": {"country": "UK", "asset_type": "Stock"},
        }

        result = extract_unique_values_from_stock_db(stock_db, "country")

        assert len(result) == 2
        assert "US" in result
        assert "UK" in result

    def test_filters_by_asset_type(self):
        """Test filters values by asset type."""
        stock_db = {
            "AAPL": {"country": "US", "asset_type": "Stock"},
            "SPY": {"country": "US", "asset_type": "ETF"},
            "BP.L": {"country": "UK", "asset_type": "Stock"},
        }

        result = extract_unique_values_from_stock_db(stock_db, "country", asset_type="Stock")

        assert len(result) == 2
        assert "US" in result
        assert "UK" in result

    def test_ignores_empty_values(self):
        """Test ignores None and empty string values."""
        stock_db = {
            "AAPL": {"country": "US", "asset_type": "Stock"},
            "UNKNOWN": {"country": "", "asset_type": "Stock"},
            "NONE": {"country": None, "asset_type": "Stock"},
        }

        result = extract_unique_values_from_stock_db(stock_db, "country")

        assert len(result) == 1
        assert result[0] == "US"

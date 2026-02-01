"""
Unit tests for portfolio_repository module.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict
from unittest.mock import MagicMock, Mock, call, patch

import pandas as pd
import pytest

from src.data_access.portfolio_repository import (
    _build_portfolio_map,
    _normalize_stock_entry,
    bulk_replace_portfolios,
    delete_portfolio,
    get_portfolio,
    list_portfolios,
    list_portfolios_cached,
    save_portfolio,
)


class TestNormalizeStockEntry:
    """Tests for _normalize_stock_entry function."""

    def test_normalize_dict_with_symbol(self):
        """Test normalizing a dict entry with 'symbol' key."""
        entry = {"symbol": "AAPL", "shares": 100}
        result = _normalize_stock_entry(entry)
        assert result == ("AAPL", {"symbol": "AAPL", "shares": 100})

    def test_normalize_dict_with_ticker(self):
        """Test normalizing a dict entry with 'ticker' key."""
        entry = {"ticker": "GOOGL", "shares": 50}
        result = _normalize_stock_entry(entry)
        assert result == ("GOOGL", {"ticker": "GOOGL", "symbol": "GOOGL", "shares": 50})

    def test_normalize_dict_with_uppercase_ticker(self):
        """Test normalizing a dict entry with 'Ticker' key."""
        entry = {"Ticker": "MSFT", "quantity": 25}
        result = _normalize_stock_entry(entry)
        assert result == ("MSFT", {"Ticker": "MSFT", "symbol": "MSFT", "quantity": 25})

    def test_normalize_dict_without_symbol(self):
        """Test normalizing a dict entry without any symbol identifier."""
        entry = {"shares": 100}
        result = _normalize_stock_entry(entry)
        assert result is None

    def test_normalize_string(self):
        """Test normalizing a string entry."""
        result = _normalize_stock_entry("TSLA")
        assert result == ("TSLA", {"symbol": "TSLA"})

    def test_normalize_invalid_type(self):
        """Test normalizing an invalid entry type."""
        result = _normalize_stock_entry(123)
        assert result is None

    def test_normalize_none(self):
        """Test normalizing None."""
        result = _normalize_stock_entry(None)
        assert result is None

    def test_normalize_empty_dict(self):
        """Test normalizing an empty dict."""
        result = _normalize_stock_entry({})
        assert result is None


class TestBuildPortfolioMap:
    """Tests for _build_portfolio_map function."""

    def test_build_empty_portfolios(self):
        """Test building map from empty dataframes."""
        portfolios_df = pd.DataFrame(columns=["id", "name", "raw_payload"])
        links_df = pd.DataFrame(columns=["portfolio_id", "ticker"])
        result = _build_portfolio_map(portfolios_df, links_df)
        assert result == {}

    def test_build_portfolio_with_raw_payload(self):
        """Test building map from portfolio with raw_payload."""
        portfolios_df = pd.DataFrame([{
            "id": "p1",
            "name": "Test Portfolio",
            "discord_webhook": "https://discord.com/webhook",
            "enabled": True,
            "created_date": "2024-01-01",
            "last_updated": "2024-01-02",
            "raw_payload": {
                "name": "Custom Name",
                "custom_field": "value"
            }
        }])
        links_df = pd.DataFrame(columns=["portfolio_id", "ticker"])

        result = _build_portfolio_map(portfolios_df, links_df)

        assert "p1" in result
        assert result["p1"]["id"] == "p1"
        assert result["p1"]["name"] == "Custom Name"
        assert result["p1"]["custom_field"] == "value"
        assert result["p1"]["stocks"] == []

    def test_build_portfolio_without_raw_payload(self):
        """Test building map from portfolio without raw_payload (None becomes empty dict)."""
        portfolios_df = pd.DataFrame([{
            "id": "p2",
            "name": "Simple Portfolio",
            "discord_webhook": "https://discord.com/webhook",
            "enabled": False,
            "created_date": "2024-01-01",
            "last_updated": "2024-01-02",
            "raw_payload": None
        }])
        links_df = pd.DataFrame(columns=["portfolio_id", "ticker"])

        result = _build_portfolio_map(portfolios_df, links_df)

        # When raw_payload is None, it becomes {}, so only id and stocks are set
        assert "p2" in result
        assert result["p2"]["id"] == "p2"
        assert result["p2"]["stocks"] == []

    def test_build_portfolio_with_invalid_raw_payload(self):
        """Test building map when raw_payload is not a dict."""
        portfolios_df = pd.DataFrame([{
            "id": "p3",
            "name": "Invalid Portfolio",
            "discord_webhook": None,
            "enabled": True,
            "created_date": "2024-01-01",
            "last_updated": "2024-01-02",
            "raw_payload": "invalid"
        }])
        links_df = pd.DataFrame(columns=["portfolio_id", "ticker"])

        result = _build_portfolio_map(portfolios_df, links_df)

        assert "p3" in result
        assert result["p3"]["id"] == "p3"
        assert result["p3"]["name"] == "Invalid Portfolio"
        assert result["p3"]["stocks"] == []

    def test_build_portfolio_with_stocks(self):
        """Test building map with portfolio stocks."""
        portfolios_df = pd.DataFrame([{
            "id": "p4",
            "name": "Portfolio with Stocks",
            "discord_webhook": None,
            "enabled": True,
            "created_date": "2024-01-01",
            "last_updated": "2024-01-02",
            "raw_payload": {"name": "Portfolio with Stocks"}
        }])
        links_df = pd.DataFrame([
            {"portfolio_id": "p4", "ticker": "AAPL"},
            {"portfolio_id": "p4", "ticker": "GOOGL"},
        ])

        result = _build_portfolio_map(portfolios_df, links_df)

        assert "p4" in result
        assert len(result["p4"]["stocks"]) == 2
        assert {"symbol": "AAPL"} in result["p4"]["stocks"]
        assert {"symbol": "GOOGL"} in result["p4"]["stocks"]

    def test_build_portfolio_with_stocks_in_raw_payload(self):
        """Test that stocks in raw_payload are replaced by links."""
        portfolios_df = pd.DataFrame([{
            "id": "p5",
            "name": "Portfolio",
            "discord_webhook": None,
            "enabled": True,
            "created_date": "2024-01-01",
            "last_updated": "2024-01-02",
            "raw_payload": {
                "name": "Portfolio",
                "stocks": "invalid"  # Not a list
            }
        }])
        links_df = pd.DataFrame([{"portfolio_id": "p5", "ticker": "TSLA"}])

        result = _build_portfolio_map(portfolios_df, links_df)

        assert "p5" in result
        # stocks should be reset to empty list then populated from links
        assert result["p5"]["stocks"] == [{"symbol": "TSLA"}]


@patch("src.data_access.portfolio_repository.get_json")
@patch("src.data_access.portfolio_repository._fetch_portfolio_frames")
@patch("src.data_access.portfolio_repository.set_json")
class TestListPortfoliosCached:
    """Tests for list_portfolios_cached function."""

    def test_cached_data_available(self, mock_set_json, mock_fetch, mock_get_json):
        """Test when cached data is available in Redis."""
        cached_data = {
            "items": {
                "p1": {"id": "p1", "name": "Test Portfolio", "stocks": []}
            }
        }
        mock_get_json.return_value = cached_data

        # Clear the LRU cache first
        list_portfolios_cached.cache_clear()

        result = list_portfolios_cached()

        assert result == cached_data["items"]
        mock_get_json.assert_called_once()
        mock_fetch.assert_not_called()
        mock_set_json.assert_not_called()

    def test_cached_data_not_available(self, mock_set_json, mock_fetch, mock_get_json):
        """Test when cached data is not available."""
        mock_get_json.return_value = None
        portfolios_df = pd.DataFrame([{
            "id": "p1",
            "name": "Test Portfolio",
            "discord_webhook": None,
            "enabled": True,
            "created_date": "2024-01-01",
            "last_updated": "2024-01-02",
            "raw_payload": {"name": "Test Portfolio"}
        }])
        links_df = pd.DataFrame(columns=["portfolio_id", "ticker"])
        mock_fetch.return_value = (portfolios_df, links_df)

        # Clear the LRU cache first
        list_portfolios_cached.cache_clear()

        result = list_portfolios_cached()

        assert "p1" in result
        mock_fetch.assert_called_once()
        mock_set_json.assert_called_once()

    def test_cached_data_invalid_format(self, mock_set_json, mock_fetch, mock_get_json):
        """Test when cached data has invalid format."""
        mock_get_json.return_value = {"wrong_key": {}}
        portfolios_df = pd.DataFrame([{
            "id": "p1",
            "name": "Test Portfolio",
            "discord_webhook": None,
            "enabled": True,
            "created_date": "2024-01-01",
            "last_updated": "2024-01-02",
            "raw_payload": {"name": "Test Portfolio"}
        }])
        links_df = pd.DataFrame(columns=["portfolio_id", "ticker"])
        mock_fetch.return_value = (portfolios_df, links_df)

        # Clear the LRU cache first
        list_portfolios_cached.cache_clear()

        result = list_portfolios_cached()

        assert "p1" in result
        mock_fetch.assert_called_once()


@patch("src.data_access.portfolio_repository.list_portfolios_cached")
class TestListPortfolios:
    """Tests for list_portfolios function."""

    def test_list_portfolios(self, mock_cached):
        """Test that list_portfolios returns a copy of cached data."""
        mock_cached.return_value = {
            "p1": {"id": "p1", "name": "Test"},
            "p2": {"id": "p2", "name": "Test2"}
        }

        result = list_portfolios()

        assert result == mock_cached.return_value
        # Verify it's a deep copy
        assert result is not mock_cached.return_value
        assert result["p1"] is not mock_cached.return_value["p1"]


@patch("src.data_access.portfolio_repository.list_portfolios")
class TestGetPortfolio:
    """Tests for get_portfolio function."""

    def test_get_existing_portfolio(self, mock_list):
        """Test getting an existing portfolio."""
        mock_list.return_value = {
            "p1": {"id": "p1", "name": "Test Portfolio"},
            "p2": {"id": "p2", "name": "Another"}
        }

        result = get_portfolio("p1")

        assert result == {"id": "p1", "name": "Test Portfolio"}

    def test_get_nonexistent_portfolio(self, mock_list):
        """Test getting a portfolio that doesn't exist."""
        mock_list.return_value = {"p1": {"id": "p1", "name": "Test"}}

        result = get_portfolio("p999")

        assert result is None


@patch("src.data_access.portfolio_repository.get_portfolio")
@patch("src.data_access.portfolio_repository._clear_cache")
@patch("src.data_access.portfolio_repository.db_config")
@patch("src.data_access.portfolio_repository.execute_values")
class TestSavePortfolio:
    """Tests for save_portfolio function."""

    def test_save_new_portfolio(self, mock_execute_values, mock_db_config, mock_clear_cache, mock_get_portfolio):
        """Test saving a new portfolio."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_db_config.get_connection.return_value = mock_conn

        portfolio = {
            "name": "New Portfolio",
            "discord_webhook": "https://discord.com/webhook",
            "enabled": True,
            "stocks": ["AAPL", "GOOGL"]
        }

        mock_get_portfolio.return_value = {
            "id": "abc123",
            "name": "New Portfolio",
            "stocks": [{"symbol": "AAPL"}, {"symbol": "GOOGL"}],
            "discord_webhook": "https://discord.com/webhook",
            "enabled": True
        }

        result = save_portfolio(portfolio)

        assert result["name"] == "New Portfolio"
        assert len(result["stocks"]) == 2
        mock_cursor.execute.assert_called()
        mock_conn.commit.assert_called_once()
        mock_clear_cache.assert_called_once()

    def test_save_portfolio_with_id(self, mock_execute_values, mock_db_config, mock_clear_cache, mock_get_portfolio):
        """Test updating an existing portfolio."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_db_config.get_connection.return_value = mock_conn

        portfolio = {
            "id": "p1",
            "name": "Updated Portfolio",
            "stocks": [{"symbol": "TSLA"}]
        }

        mock_get_portfolio.return_value = portfolio

        result = save_portfolio(portfolio)

        assert result["id"] == "p1"
        mock_clear_cache.assert_called_once()

    def test_save_portfolio_with_stock_dicts(self, mock_execute_values, mock_db_config, mock_clear_cache, mock_get_portfolio):
        """Test saving portfolio with stock dicts containing extra fields."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_db_config.get_connection.return_value = mock_conn

        portfolio = {
            "name": "Test",
            "stocks": [
                {"symbol": "AAPL", "shares": 100},
                {"ticker": "GOOGL", "shares": 50}
            ]
        }

        mock_get_portfolio.return_value = {"id": "p1", "name": "Test", "stocks": []}

        result = save_portfolio(portfolio)

        # Verify execute_values was called for portfolio_stocks
        assert mock_execute_values.called

    def test_save_portfolio_empty_stocks(self, mock_execute_values, mock_db_config, mock_clear_cache, mock_get_portfolio):
        """Test saving portfolio with no stocks."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_db_config.get_connection.return_value = mock_conn

        portfolio = {
            "name": "Empty Portfolio",
            "stocks": []
        }

        mock_get_portfolio.return_value = {"id": "p1", "name": "Empty Portfolio", "stocks": []}

        result = save_portfolio(portfolio)

        # execute_values should not be called for stocks since list is empty
        mock_cursor.execute.assert_called()


@patch("src.data_access.portfolio_repository._clear_cache")
@patch("src.data_access.portfolio_repository.db_config")
class TestDeletePortfolio:
    """Tests for delete_portfolio function."""

    def test_delete_portfolio(self, mock_db_config, mock_clear_cache):
        """Test deleting a portfolio."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_db_config.get_connection.return_value = mock_conn

        delete_portfolio("p1")

        # Should delete from both tables
        assert mock_cursor.execute.call_count == 2
        mock_conn.commit.assert_called_once()
        mock_clear_cache.assert_called_once()

    def test_delete_portfolio_connection_closed(self, mock_db_config, mock_clear_cache):
        """Test that connection is closed even on error."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("Database error")
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_db_config.get_connection.return_value = mock_conn

        with pytest.raises(Exception):
            delete_portfolio("p1")

        mock_db_config.close_connection.assert_called_once_with(mock_conn)


@patch("src.data_access.portfolio_repository._clear_cache")
@patch("src.data_access.portfolio_repository.db_config")
@patch("src.data_access.portfolio_repository.execute_values")
class TestBulkReplacePortfolios:
    """Tests for bulk_replace_portfolios function."""

    def test_bulk_replace_empty_list(self, mock_execute_values, mock_db_config, mock_clear_cache):
        """Test bulk replace with empty portfolio list."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_db_config.get_connection.return_value = mock_conn

        bulk_replace_portfolios([])

        # Should still truncate tables
        assert mock_cursor.execute.call_count == 2
        mock_execute_values.assert_not_called()
        mock_conn.commit.assert_called_once()
        mock_clear_cache.assert_called_once()

    def test_bulk_replace_multiple_portfolios(self, mock_execute_values, mock_db_config, mock_clear_cache):
        """Test bulk replace with multiple portfolios."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_db_config.get_connection.return_value = mock_conn

        portfolios = [
            {
                "id": "p1",
                "name": "Portfolio 1",
                "stocks": ["AAPL", "GOOGL"]
            },
            {
                "id": "p2",
                "name": "Portfolio 2",
                "stocks": [{"symbol": "TSLA"}]
            }
        ]

        bulk_replace_portfolios(portfolios)

        # Should call execute_values twice: once for portfolios, once for stocks
        assert mock_execute_values.call_count == 2
        mock_conn.commit.assert_called_once()
        mock_clear_cache.assert_called_once()

    def test_bulk_replace_portfolios_without_stocks(self, mock_execute_values, mock_db_config, mock_clear_cache):
        """Test bulk replace with portfolios that have no stocks."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_db_config.get_connection.return_value = mock_conn

        portfolios = [
            {"id": "p1", "name": "Empty Portfolio", "stocks": []},
            {"id": "p2", "name": "Another Empty"}
        ]

        bulk_replace_portfolios(portfolios)

        # Should call execute_values once for portfolios only
        assert mock_execute_values.call_count == 1
        mock_conn.commit.assert_called_once()

    def test_bulk_replace_generates_ids(self, mock_execute_values, mock_db_config, mock_clear_cache):
        """Test that portfolios without IDs get generated IDs."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_db_config.get_connection.return_value = mock_conn

        portfolios = [{"name": "No ID Portfolio"}]

        bulk_replace_portfolios(portfolios)

        # Verify execute_values was called with generated ID
        assert mock_execute_values.called
        call_args = mock_execute_values.call_args_list[0][0]
        rows = call_args[2]
        assert len(rows) == 1
        assert rows[0][0] is not None  # ID should be generated

    def test_bulk_replace_connection_closed_on_error(self, mock_execute_values, mock_db_config, mock_clear_cache):
        """Test that connection is closed even on error."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("Database error")
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=False)
        mock_db_config.get_connection.return_value = mock_conn

        with pytest.raises(Exception):
            bulk_replace_portfolios([{"name": "Test"}])

        mock_db_config.close_connection.assert_called_once_with(mock_conn)

"""Unit tests for metadata_repository module."""

from __future__ import annotations

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, call
from typing import Any, Dict, List

import pandas as pd


class TestQueryDataframe:
    """Tests for _query_dataframe helper function."""

    @patch("src.data_access.metadata_repository.db_config")
    @patch("src.data_access.metadata_repository.pd.read_sql_query")
    def test_executes_query_and_returns_dataframe(self, mock_read_sql, mock_db_config):
        """Test that query is executed and DataFrame is returned."""
        from src.data_access.metadata_repository import _query_dataframe

        mock_conn = MagicMock()
        mock_db_config.get_connection.return_value = mock_conn

        test_df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        mock_read_sql.return_value = test_df

        result = _query_dataframe("SELECT * FROM test")

        mock_db_config.get_connection.assert_called_once()
        mock_read_sql.assert_called_once_with("SELECT * FROM test", mock_conn, params=None)
        mock_db_config.close_connection.assert_called_once_with(mock_conn)
        pd.testing.assert_frame_equal(result, test_df)

    @patch("src.data_access.metadata_repository.db_config")
    @patch("src.data_access.metadata_repository.pd.read_sql_query")
    def test_executes_query_with_parameters(self, mock_read_sql, mock_db_config):
        """Test that query parameters are passed correctly."""
        from src.data_access.metadata_repository import _query_dataframe

        mock_conn = MagicMock()
        mock_db_config.get_connection.return_value = mock_conn

        test_df = pd.DataFrame({"id": [1]})
        mock_read_sql.return_value = test_df

        params = ("AAPL", 100)
        result = _query_dataframe("SELECT * FROM test WHERE symbol = ? AND price > ?", params)

        mock_read_sql.assert_called_once_with(
            "SELECT * FROM test WHERE symbol = ? AND price > ?",
            mock_conn,
            params=list(params)
        )

    @patch("src.data_access.metadata_repository.db_config")
    @patch("src.data_access.metadata_repository.pd.read_sql_query")
    def test_closes_connection_on_exception(self, mock_read_sql, mock_db_config):
        """Test that connection is closed even when query fails."""
        from src.data_access.metadata_repository import _query_dataframe

        mock_conn = MagicMock()
        mock_db_config.get_connection.return_value = mock_conn
        mock_read_sql.side_effect = Exception("Query failed")

        with pytest.raises(Exception, match="Query failed"):
            _query_dataframe("SELECT * FROM test")

        mock_db_config.close_connection.assert_called_once_with(mock_conn)


class TestNormaliseRecords:
    """Tests for _normalise_records helper function."""

    def test_preserves_simple_types(self):
        """Test that simple types are preserved unchanged."""
        from src.data_access.metadata_repository import _normalise_records

        records = [
            {"id": 1, "name": "Test", "active": True, "value": 123.45},
            {"id": 2, "name": None, "active": False, "value": 0},
        ]

        result = _normalise_records(records)

        assert result == records

    def test_converts_timestamps_to_isoformat(self):
        """Test that pd.Timestamp objects are converted to ISO format strings."""
        from src.data_access.metadata_repository import _normalise_records

        timestamp = pd.Timestamp("2024-01-15 10:30:00")
        records = [{"id": 1, "created_at": timestamp}]

        result = _normalise_records(records)

        assert result[0]["id"] == 1
        assert result[0]["created_at"] == "2024-01-15T10:30:00"
        assert isinstance(result[0]["created_at"], str)

    def test_converts_series_to_dict(self):
        """Test that pd.Series objects are converted to dictionaries."""
        from src.data_access.metadata_repository import _normalise_records

        series = pd.Series([1, 2, 3], index=["a", "b", "c"])
        records = [{"id": 1, "data": series}]

        result = _normalise_records(records)

        assert result[0]["id"] == 1
        assert result[0]["data"] == {"a": 1, "b": 2, "c": 3}

    def test_converts_dataframe_to_dict(self):
        """Test that pd.DataFrame objects are converted to dictionaries."""
        from src.data_access.metadata_repository import _normalise_records

        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        records = [{"id": 1, "nested_data": df}]

        result = _normalise_records(records)

        assert result[0]["id"] == 1
        assert isinstance(result[0]["nested_data"], dict)

    def test_handles_empty_list(self):
        """Test that empty list returns empty list."""
        from src.data_access.metadata_repository import _normalise_records

        result = _normalise_records([])

        assert result == []

    def test_handles_mixed_types(self):
        """Test normalisation with mixed types in one record."""
        from src.data_access.metadata_repository import _normalise_records

        records = [
            {
                "id": 1,
                "name": "Test",
                "timestamp": pd.Timestamp("2024-01-15"),
                "series": pd.Series([1, 2]),
                "value": 42,
            }
        ]

        result = _normalise_records(records)

        assert result[0]["id"] == 1
        assert result[0]["name"] == "Test"
        assert result[0]["timestamp"] == "2024-01-15T00:00:00"
        assert isinstance(result[0]["series"], dict)
        assert result[0]["value"] == 42


class TestFetchStockMetadataDF:
    """Tests for fetch_stock_metadata_df function."""

    def setup_method(self):
        """Clear cache before each test."""
        from src.data_access.metadata_repository import fetch_stock_metadata_df
        fetch_stock_metadata_df.cache_clear()

    @patch("src.data_access.metadata_repository.get_json")
    def test_returns_dataframe_from_redis_cache(self, mock_get_json):
        """Test that cached data from Redis is returned as DataFrame."""
        from src.data_access.metadata_repository import fetch_stock_metadata_df

        cached_data = {
            "records": [
                {"symbol": "AAPL", "name": "Apple Inc.", "last_updated": "2024-01-15T10:00:00"},
                {"symbol": "MSFT", "name": "Microsoft", "last_updated": "2024-01-15T10:00:00"},
            ]
        }
        mock_get_json.return_value = cached_data

        result = fetch_stock_metadata_df()

        assert len(result) == 2
        assert list(result["symbol"]) == ["AAPL", "MSFT"]
        assert pd.api.types.is_datetime64_any_dtype(result["last_updated"])

    @patch("src.data_access.metadata_repository.set_json")
    @patch("src.data_access.metadata_repository.get_json")
    @patch("src.data_access.metadata_repository._query_dataframe")
    def test_queries_database_when_cache_miss(
        self, mock_query_df, mock_get_json, mock_set_json
    ):
        """Test that database is queried when Redis cache is empty."""
        from src.data_access.metadata_repository import fetch_stock_metadata_df

        mock_get_json.return_value = None
        db_df = pd.DataFrame({
            "symbol": ["AAPL", "MSFT"],
            "name": ["Apple Inc.", "Microsoft"],
            "last_updated": pd.to_datetime(["2024-01-15", "2024-01-15"])
        })
        mock_query_df.return_value = db_df

        result = fetch_stock_metadata_df()

        mock_query_df.assert_called_once_with("SELECT * FROM stock_metadata")
        assert len(result) == 2
        mock_set_json.assert_called_once()

    @patch("src.data_access.metadata_repository.set_json")
    @patch("src.data_access.metadata_repository.get_json")
    @patch("src.data_access.metadata_repository._query_dataframe")
    def test_caches_database_result_to_redis(
        self, mock_query_df, mock_get_json, mock_set_json
    ):
        """Test that database results are cached to Redis."""
        from src.data_access.metadata_repository import (
            fetch_stock_metadata_df,
            REDIS_STOCK_METADATA_KEY
        )

        mock_get_json.return_value = None
        db_df = pd.DataFrame({
            "symbol": ["AAPL"],
            "name": ["Apple Inc."],
            "last_updated": pd.to_datetime(["2024-01-15"])
        })
        mock_query_df.return_value = db_df

        fetch_stock_metadata_df()

        mock_set_json.assert_called_once()
        call_args = mock_set_json.call_args
        assert call_args[0][0] == REDIS_STOCK_METADATA_KEY
        assert "records" in call_args[0][1]
        assert call_args[1]["ttl_seconds"] == 600

    @patch("src.data_access.metadata_repository.get_json")
    @patch("src.data_access.metadata_repository._query_dataframe")
    def test_handles_empty_database_result(self, mock_query_df, mock_get_json):
        """Test handling of empty DataFrame from database."""
        from src.data_access.metadata_repository import fetch_stock_metadata_df

        mock_get_json.return_value = None
        mock_query_df.return_value = pd.DataFrame()

        result = fetch_stock_metadata_df()

        assert len(result) == 0
        assert result.empty

    @patch("src.data_access.metadata_repository.get_json")
    def test_handles_invalid_cache_format(self, mock_get_json):
        """Test handling of invalid cache data format."""
        from src.data_access.metadata_repository import fetch_stock_metadata_df

        # Not a dict
        mock_get_json.return_value = "invalid"

        with patch("src.data_access.metadata_repository._query_dataframe") as mock_query:
            mock_query.return_value = pd.DataFrame()
            result = fetch_stock_metadata_df()

        # Should fall back to database query
        mock_query.assert_called_once()

    @patch("src.data_access.metadata_repository.get_json")
    def test_handles_cache_missing_records_key(self, mock_get_json):
        """Test handling when cache dict is missing 'records' key."""
        from src.data_access.metadata_repository import fetch_stock_metadata_df

        mock_get_json.return_value = {"wrong_key": []}

        with patch("src.data_access.metadata_repository._query_dataframe") as mock_query:
            mock_query.return_value = pd.DataFrame()
            result = fetch_stock_metadata_df()

        # Should fall back to database query
        mock_query.assert_called_once()

    @patch("src.data_access.metadata_repository.get_json")
    def test_uses_lru_cache(self, mock_get_json):
        """Test that function results are cached in memory."""
        from src.data_access.metadata_repository import fetch_stock_metadata_df

        cached_data = {"records": [{"symbol": "AAPL", "name": "Apple Inc."}]}
        mock_get_json.return_value = cached_data

        # Call twice
        result1 = fetch_stock_metadata_df()
        result2 = fetch_stock_metadata_df()

        # Should only call get_json once due to LRU cache
        assert mock_get_json.call_count == 1
        pd.testing.assert_frame_equal(result1, result2)


class TestFetchStockMetadataMap:
    """Tests for fetch_stock_metadata_map function."""

    def setup_method(self):
        """Clear caches before each test."""
        from src.data_access.metadata_repository import (
            fetch_stock_metadata_df,
            fetch_stock_metadata_map
        )
        fetch_stock_metadata_df.cache_clear()
        fetch_stock_metadata_map.cache_clear()

    @patch("src.data_access.metadata_repository.fetch_stock_metadata_df")
    def test_returns_dict_keyed_by_symbol(self, mock_fetch_df):
        """Test that DataFrame is converted to dict keyed by symbol."""
        from src.data_access.metadata_repository import fetch_stock_metadata_map

        test_df = pd.DataFrame({
            "symbol": ["AAPL", "MSFT", "GOOGL"],
            "name": ["Apple Inc.", "Microsoft", "Google"],
            "sector": ["Technology", "Technology", "Technology"]
        })
        mock_fetch_df.return_value = test_df

        result = fetch_stock_metadata_map()

        assert len(result) == 3
        assert "AAPL" in result
        assert result["AAPL"]["name"] == "Apple Inc."
        assert result["MSFT"]["name"] == "Microsoft"
        assert result["GOOGL"]["name"] == "Google"

    @patch("src.data_access.metadata_repository.fetch_stock_metadata_df")
    def test_returns_empty_dict_for_empty_dataframe(self, mock_fetch_df):
        """Test that empty DataFrame returns empty dict."""
        from src.data_access.metadata_repository import fetch_stock_metadata_map

        mock_fetch_df.return_value = pd.DataFrame()

        result = fetch_stock_metadata_map()

        assert result == {}

    @patch("src.data_access.metadata_repository.fetch_stock_metadata_df")
    def test_preserves_all_columns_in_dict_values(self, mock_fetch_df):
        """Test that all columns are preserved in the dict values."""
        from src.data_access.metadata_repository import fetch_stock_metadata_map

        test_df = pd.DataFrame({
            "symbol": ["AAPL"],
            "name": ["Apple Inc."],
            "sector": ["Technology"],
            "market_cap": [3000000000000],
            "last_updated": pd.to_datetime(["2024-01-15"])
        })
        mock_fetch_df.return_value = test_df

        result = fetch_stock_metadata_map()

        assert "symbol" in result["AAPL"]
        assert "name" in result["AAPL"]
        assert "sector" in result["AAPL"]
        assert "market_cap" in result["AAPL"]
        assert "last_updated" in result["AAPL"]


class TestFetchAlertsList:
    """Tests for fetch_alerts_list function."""

    def setup_method(self):
        """Clear cache before each test."""
        from src.data_access.metadata_repository import fetch_alerts_list
        fetch_alerts_list.cache_clear()

    @patch("src.data_access.metadata_repository.get_json")
    def test_returns_list_from_redis_cache(self, mock_get_json):
        """Test that cached alerts from Redis are returned."""
        from src.data_access.metadata_repository import fetch_alerts_list

        cached_data = {
            "items": [
                {"alert_id": "1", "name": "Alert 1", "condition": "price > 100"},
                {"alert_id": "2", "name": "Alert 2", "condition": "price < 50"},
            ]
        }
        mock_get_json.return_value = cached_data

        result = fetch_alerts_list()

        assert len(result) == 2
        assert result[0]["alert_id"] == "1"
        assert result[1]["alert_id"] == "2"

    @patch("src.data_access.metadata_repository.set_json")
    @patch("src.data_access.metadata_repository.list_alerts")
    @patch("src.data_access.metadata_repository.get_json")
    def test_queries_repository_when_cache_miss(
        self, mock_get_json, mock_list_alerts, mock_set_json
    ):
        """Test that alert repository is queried when Redis cache is empty."""
        from src.data_access.metadata_repository import fetch_alerts_list

        mock_get_json.return_value = None
        alerts_data = [
            {"alert_id": "1", "name": "Alert 1"},
            {"alert_id": "2", "name": "Alert 2"},
        ]
        mock_list_alerts.return_value = alerts_data

        result = fetch_alerts_list()

        mock_list_alerts.assert_called_once()
        assert result == alerts_data

    @patch("src.data_access.metadata_repository.set_json")
    @patch("src.data_access.metadata_repository.list_alerts")
    @patch("src.data_access.metadata_repository.get_json")
    def test_caches_alerts_to_redis(
        self, mock_get_json, mock_list_alerts, mock_set_json
    ):
        """Test that alerts are cached to Redis after fetching."""
        from src.data_access.metadata_repository import (
            fetch_alerts_list,
            REDIS_ALERTS_KEY
        )

        mock_get_json.return_value = None
        alerts_data = [{"alert_id": "1", "name": "Alert 1"}]
        mock_list_alerts.return_value = alerts_data

        fetch_alerts_list()

        mock_set_json.assert_called_once_with(
            REDIS_ALERTS_KEY,
            {"items": alerts_data},
            ttl_seconds=300
        )

    @patch("src.data_access.metadata_repository.get_json")
    def test_handles_invalid_cache_format(self, mock_get_json):
        """Test handling of invalid cache data format."""
        from src.data_access.metadata_repository import fetch_alerts_list

        mock_get_json.return_value = "invalid"

        with patch("src.data_access.metadata_repository.list_alerts") as mock_list:
            mock_list.return_value = []
            result = fetch_alerts_list()

        mock_list.assert_called_once()

    @patch("src.data_access.metadata_repository.list_alerts")
    @patch("src.data_access.metadata_repository.get_json")
    def test_returns_empty_list_when_no_alerts(self, mock_get_json, mock_list_alerts):
        """Test that empty list is returned when no alerts exist."""
        from src.data_access.metadata_repository import fetch_alerts_list

        mock_get_json.return_value = None
        mock_list_alerts.return_value = []

        result = fetch_alerts_list()

        assert result == []


class TestFetchAlertsDF:
    """Tests for fetch_alerts_df function."""

    def setup_method(self):
        """Clear caches before each test."""
        from src.data_access.metadata_repository import (
            fetch_alerts_list,
            fetch_alerts_df
        )
        fetch_alerts_list.cache_clear()
        fetch_alerts_df.cache_clear()

    @patch("src.data_access.metadata_repository.fetch_alerts_list")
    def test_converts_alerts_list_to_dataframe(self, mock_fetch_list):
        """Test that alerts list is converted to DataFrame."""
        from src.data_access.metadata_repository import fetch_alerts_df

        alerts = [
            {"alert_id": "1", "name": "Alert 1", "condition": "price > 100"},
            {"alert_id": "2", "name": "Alert 2", "condition": "price < 50"},
        ]
        mock_fetch_list.return_value = alerts

        result = fetch_alerts_df()

        assert len(result) == 2
        assert "alert_id" in result.columns
        assert "name" in result.columns
        assert list(result["alert_id"]) == ["1", "2"]

    @patch("src.data_access.metadata_repository.fetch_alerts_list")
    def test_converts_last_triggered_to_datetime(self, mock_fetch_list):
        """Test that last_triggered column is converted to datetime."""
        from src.data_access.metadata_repository import fetch_alerts_df

        alerts = [
            {
                "alert_id": "1",
                "name": "Alert 1",
                "last_triggered": "2024-01-15T10:30:00"
            },
        ]
        mock_fetch_list.return_value = alerts

        result = fetch_alerts_df()

        assert "last_triggered" in result.columns
        assert pd.api.types.is_datetime64_any_dtype(result["last_triggered"])

    @patch("src.data_access.metadata_repository.fetch_alerts_list")
    def test_returns_empty_dataframe_for_empty_list(self, mock_fetch_list):
        """Test that empty list returns empty DataFrame."""
        from src.data_access.metadata_repository import fetch_alerts_df

        mock_fetch_list.return_value = []

        result = fetch_alerts_df()

        assert len(result) == 0
        assert result.empty

    @patch("src.data_access.metadata_repository.fetch_alerts_list")
    def test_handles_missing_last_triggered_column(self, mock_fetch_list):
        """Test handling when last_triggered column is missing."""
        from src.data_access.metadata_repository import fetch_alerts_df

        alerts = [
            {"alert_id": "1", "name": "Alert 1"},
        ]
        mock_fetch_list.return_value = alerts

        result = fetch_alerts_df()

        assert len(result) == 1
        # Should not raise error if column doesn't exist


class TestFetchPortfolios:
    """Tests for fetch_portfolios function."""

    def setup_method(self):
        """Clear cache before each test."""
        from src.data_access.metadata_repository import fetch_portfolios
        fetch_portfolios.cache_clear()

    @patch("src.data_access.metadata_repository.get_json")
    def test_returns_dict_from_redis_cache(self, mock_get_json):
        """Test that cached portfolios from Redis are returned."""
        from src.data_access.metadata_repository import fetch_portfolios

        cached_data = {
            "items": {
                "portfolio1": {"name": "My Portfolio", "stocks": ["AAPL", "MSFT"]},
                "portfolio2": {"name": "Tech Stocks", "stocks": ["GOOGL", "AMZN"]},
            }
        }
        mock_get_json.return_value = cached_data

        result = fetch_portfolios()

        assert len(result) == 2
        assert "portfolio1" in result
        assert result["portfolio1"]["name"] == "My Portfolio"

    @patch("src.data_access.metadata_repository.set_json")
    @patch("src.data_access.metadata_repository.list_portfolios_map")
    @patch("src.data_access.metadata_repository.get_json")
    def test_queries_repository_when_cache_miss(
        self, mock_get_json, mock_list_portfolios, mock_set_json
    ):
        """Test that portfolio repository is queried when Redis cache is empty."""
        from src.data_access.metadata_repository import fetch_portfolios

        mock_get_json.return_value = None
        portfolios_data = {
            "portfolio1": {"name": "My Portfolio", "stocks": ["AAPL"]},
        }
        mock_list_portfolios.return_value = portfolios_data

        result = fetch_portfolios()

        mock_list_portfolios.assert_called_once()
        assert result == portfolios_data

    @patch("src.data_access.metadata_repository.set_json")
    @patch("src.data_access.metadata_repository.list_portfolios_map")
    @patch("src.data_access.metadata_repository.get_json")
    def test_caches_portfolios_to_redis(
        self, mock_get_json, mock_list_portfolios, mock_set_json
    ):
        """Test that portfolios are cached to Redis after fetching."""
        from src.data_access.metadata_repository import (
            fetch_portfolios,
            REDIS_PORTFOLIOS_KEY
        )

        mock_get_json.return_value = None
        portfolios_data = {"portfolio1": {"name": "Test"}}
        mock_list_portfolios.return_value = portfolios_data

        fetch_portfolios()

        mock_set_json.assert_called_once_with(
            REDIS_PORTFOLIOS_KEY,
            {"items": portfolios_data},
            ttl_seconds=300
        )

    @patch("src.data_access.metadata_repository.get_json")
    def test_handles_invalid_cache_format(self, mock_get_json):
        """Test handling of invalid cache data format."""
        from src.data_access.metadata_repository import fetch_portfolios

        mock_get_json.return_value = "invalid"

        with patch("src.data_access.metadata_repository.list_portfolios_map") as mock_list:
            mock_list.return_value = {}
            result = fetch_portfolios()

        mock_list.assert_called_once()

    @patch("src.data_access.metadata_repository.list_portfolios_map")
    @patch("src.data_access.metadata_repository.get_json")
    def test_returns_empty_dict_when_no_portfolios(
        self, mock_get_json, mock_list_portfolios
    ):
        """Test that empty dict is returned when no portfolios exist."""
        from src.data_access.metadata_repository import fetch_portfolios

        mock_get_json.return_value = None
        mock_list_portfolios.return_value = {}

        result = fetch_portfolios()

        assert result == {}


class TestRefreshCaches:
    """Tests for refresh_caches function."""

    def setup_method(self):
        """Clear all caches before each test."""
        from src.data_access.metadata_repository import (
            fetch_stock_metadata_df,
            fetch_stock_metadata_map,
            fetch_alerts_list,
            fetch_alerts_df,
            fetch_portfolios,
        )
        fetch_stock_metadata_df.cache_clear()
        fetch_stock_metadata_map.cache_clear()
        fetch_alerts_list.cache_clear()
        fetch_alerts_df.cache_clear()
        fetch_portfolios.cache_clear()

    @patch("src.data_access.metadata_repository.delete_key")
    @patch("src.data_access.metadata_repository.clear_portfolio_cache")
    @patch("src.data_access.metadata_repository.refresh_alert_cache")
    def test_clears_all_lru_caches(
        self, mock_refresh_alert, mock_clear_portfolio, mock_delete_key
    ):
        """Test that all LRU caches are cleared."""
        from src.data_access.metadata_repository import (
            fetch_stock_metadata_df,
            fetch_stock_metadata_map,
            fetch_alerts_list,
            fetch_alerts_df,
            fetch_portfolios,
            refresh_caches,
        )

        # Populate caches
        with patch("src.data_access.metadata_repository.get_json") as mock_get:
            mock_get.return_value = None
            with patch("src.data_access.metadata_repository._query_dataframe") as mock_query:
                mock_query.return_value = pd.DataFrame()
                fetch_stock_metadata_df()

        # Verify cache has data
        cache_info = fetch_stock_metadata_df.cache_info()
        assert cache_info.currsize > 0

        # Clear caches
        refresh_caches()

        # Verify all caches cleared
        assert fetch_stock_metadata_df.cache_info().currsize == 0
        assert fetch_stock_metadata_map.cache_info().currsize == 0
        assert fetch_alerts_list.cache_info().currsize == 0
        assert fetch_alerts_df.cache_info().currsize == 0
        assert fetch_portfolios.cache_info().currsize == 0

    @patch("src.data_access.metadata_repository.delete_key")
    @patch("src.data_access.metadata_repository.clear_portfolio_cache")
    @patch("src.data_access.metadata_repository.refresh_alert_cache")
    def test_calls_refresh_alert_cache(
        self, mock_refresh_alert, mock_clear_portfolio, mock_delete_key
    ):
        """Test that refresh_alert_cache is called."""
        from src.data_access.metadata_repository import refresh_caches

        refresh_caches()

        mock_refresh_alert.assert_called_once()

    @patch("src.data_access.metadata_repository.delete_key")
    @patch("src.data_access.metadata_repository.clear_portfolio_cache")
    @patch("src.data_access.metadata_repository.refresh_alert_cache")
    def test_calls_clear_portfolio_cache(
        self, mock_refresh_alert, mock_clear_portfolio, mock_delete_key
    ):
        """Test that clear_portfolio_cache is called."""
        from src.data_access.metadata_repository import refresh_caches

        refresh_caches()

        mock_clear_portfolio.assert_called_once()

    @patch("src.data_access.metadata_repository.delete_key")
    @patch("src.data_access.metadata_repository.clear_portfolio_cache")
    @patch("src.data_access.metadata_repository.refresh_alert_cache")
    def test_deletes_all_redis_keys(
        self, mock_refresh_alert, mock_clear_portfolio, mock_delete_key
    ):
        """Test that all Redis keys are deleted."""
        from src.data_access.metadata_repository import (
            refresh_caches,
            REDIS_STOCK_METADATA_KEY,
            REDIS_ALERTS_KEY,
            REDIS_PORTFOLIOS_KEY,
        )

        refresh_caches()

        assert mock_delete_key.call_count == 3
        mock_delete_key.assert_any_call(REDIS_STOCK_METADATA_KEY)
        mock_delete_key.assert_any_call(REDIS_ALERTS_KEY)
        mock_delete_key.assert_any_call(REDIS_PORTFOLIOS_KEY)

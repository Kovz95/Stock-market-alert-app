"""Unit tests for alert_repository module."""

from __future__ import annotations

import pytest
from datetime import datetime
from unittest.mock import MagicMock, patch, call
from uuid import uuid4

import pandas as pd


class TestParseTimestamp:
    """Tests for _parse_timestamp helper function."""

    def test_returns_none_for_empty_value(self):
        """Test that empty values return None."""
        from src.data_access.alert_repository import _parse_timestamp

        assert _parse_timestamp(None) is None
        assert _parse_timestamp("") is None
        assert _parse_timestamp(0) is None

    def test_returns_datetime_unchanged(self):
        """Test that datetime objects are returned as-is."""
        from src.data_access.alert_repository import _parse_timestamp

        dt = datetime(2024, 1, 15, 10, 30, 0)
        result = _parse_timestamp(dt)

        assert result == dt
        assert isinstance(result, datetime)

    def test_parses_iso_format_string(self):
        """Test parsing ISO format timestamp strings."""
        from src.data_access.alert_repository import _parse_timestamp

        result = _parse_timestamp("2024-01-15T10:30:00")

        assert result == datetime(2024, 1, 15, 10, 30, 0)

    def test_returns_none_for_invalid_string(self):
        """Test that invalid strings return None."""
        from src.data_access.alert_repository import _parse_timestamp

        assert _parse_timestamp("not a date") is None
        assert _parse_timestamp("2024-13-45") is None


class TestFetchAlertDataframe:
    """Tests for _fetch_alert_dataframe function."""

    @patch("src.data_access.alert_repository.pd.read_sql_query")
    @patch("src.data_access.alert_repository.db_config")
    def test_fetches_alerts_from_database(self, mock_db_config, mock_read_sql):
        """Test that alerts are fetched from database as DataFrame."""
        from src.data_access.alert_repository import _fetch_alert_dataframe

        mock_conn = MagicMock()
        mock_db_config.get_connection.return_value = mock_conn

        test_df = pd.DataFrame({
            "alert_id": ["123"],
            "name": ["Test Alert"],
            "last_triggered": ["2024-01-15T10:30:00"]
        })
        mock_read_sql.return_value = test_df

        result = _fetch_alert_dataframe()

        mock_read_sql.assert_called_once()
        assert "SELECT" in mock_read_sql.call_args[0][0]
        assert "FROM alerts" in mock_read_sql.call_args[0][0]
        mock_db_config.close_connection.assert_called_once_with(mock_conn)

    @patch("src.data_access.alert_repository.pd.read_sql_query")
    @patch("src.data_access.alert_repository.db_config")
    def test_converts_last_triggered_to_datetime(self, mock_db_config, mock_read_sql):
        """Test that last_triggered column is converted to datetime."""
        from src.data_access.alert_repository import _fetch_alert_dataframe

        mock_conn = MagicMock()
        mock_db_config.get_connection.return_value = mock_conn

        test_df = pd.DataFrame({
            "alert_id": ["123"],
            "last_triggered": ["2024-01-15T10:30:00"]
        })
        mock_read_sql.return_value = test_df

        result = _fetch_alert_dataframe()

        assert pd.api.types.is_datetime64_any_dtype(result["last_triggered"])

    @patch("src.data_access.alert_repository.pd.read_sql_query")
    @patch("src.data_access.alert_repository.db_config")
    def test_handles_empty_dataframe(self, mock_db_config, mock_read_sql):
        """Test handling of empty result set."""
        from src.data_access.alert_repository import _fetch_alert_dataframe

        mock_conn = MagicMock()
        mock_db_config.get_connection.return_value = mock_conn
        mock_read_sql.return_value = pd.DataFrame()

        result = _fetch_alert_dataframe()

        assert result.empty


class TestNormalizeAlert:
    """Tests for _normalize_alert function."""

    def test_preserves_basic_fields(self):
        """Test that basic alert fields are preserved."""
        from src.data_access.alert_repository import _normalize_alert

        record = {
            "alert_id": "123",
            "name": "Test Alert",
            "ticker": "AAPL",
            "raw_payload": None
        }

        result = _normalize_alert(record)

        assert result["alert_id"] == "123"
        assert result["name"] == "Test Alert"
        assert result["ticker"] == "AAPL"

    def test_merges_raw_payload_with_columns(self):
        """Test that raw_payload is merged with column values."""
        from src.data_access.alert_repository import _normalize_alert

        record = {
            "alert_id": "123",
            "name": "Test Alert",
            "raw_payload": {
                "custom_field": "custom_value",
                "name": "Original Name"
            }
        }

        result = _normalize_alert(record)

        assert result["alert_id"] == "123"
        assert result["name"] == "Test Alert"  # Column value overrides payload
        assert result["custom_field"] == "custom_value"

    def test_converts_alert_id_to_string(self):
        """Test that alert_id is converted to string."""
        from src.data_access.alert_repository import _normalize_alert

        record = {
            "alert_id": 123,
            "raw_payload": None
        }

        result = _normalize_alert(record)

        assert result["alert_id"] == "123"
        assert isinstance(result["alert_id"], str)

    def test_ensures_conditions_is_list(self):
        """Test that conditions defaults to empty list."""
        from src.data_access.alert_repository import _normalize_alert

        record = {
            "alert_id": "123",
            "conditions": None,
            "raw_payload": None
        }

        result = _normalize_alert(record)

        assert result["conditions"] == []

    def test_converts_pandas_timestamp_to_iso(self):
        """Test that pandas Timestamp is converted to ISO string."""
        from src.data_access.alert_repository import _normalize_alert

        record = {
            "alert_id": "123",
            "last_triggered": pd.Timestamp("2024-01-15 10:30:00"),
            "raw_payload": None
        }

        result = _normalize_alert(record)

        assert isinstance(result["last_triggered"], str)
        assert "2024-01-15" in result["last_triggered"]

    def test_converts_datetime_to_iso(self):
        """Test that datetime is converted to ISO string."""
        from src.data_access.alert_repository import _normalize_alert

        record = {
            "alert_id": "123",
            "last_triggered": datetime(2024, 1, 15, 10, 30, 0),
            "raw_payload": None
        }

        result = _normalize_alert(record)

        assert isinstance(result["last_triggered"], str)
        assert "2024-01-15" in result["last_triggered"]

    def test_sets_empty_string_for_null_last_triggered(self):
        """Test that null last_triggered becomes empty string."""
        from src.data_access.alert_repository import _normalize_alert

        record = {
            "alert_id": "123",
            "last_triggered": None,
            "raw_payload": None
        }

        result = _normalize_alert(record)

        assert result["last_triggered"] == ""

    def test_derives_ratio_from_is_ratio_when_empty(self):
        """Test that ratio field is derived from is_ratio flag."""
        from src.data_access.alert_repository import _normalize_alert

        record = {
            "alert_id": "123",
            "ratio": None,
            "is_ratio": True,
            "raw_payload": None
        }

        result = _normalize_alert(record)

        assert result["ratio"] == "Yes"

    def test_ratio_no_when_is_ratio_false(self):
        """Test that ratio is 'No' when is_ratio is False."""
        from src.data_access.alert_repository import _normalize_alert

        record = {
            "alert_id": "123",
            "ratio": "",
            "is_ratio": False,
            "raw_payload": None
        }

        result = _normalize_alert(record)

        assert result["ratio"] == "No"


class TestPreparePayload:
    """Tests for _prepare_payload function."""

    def test_generates_alert_id_if_missing(self):
        """Test that alert_id is generated when not provided."""
        from src.data_access.alert_repository import _prepare_payload

        alert = {"name": "Test Alert"}

        result = _prepare_payload(alert)

        assert "alert_id" in result
        assert len(result["alert_id"]) > 0

    def test_preserves_existing_alert_id(self):
        """Test that existing alert_id is preserved."""
        from src.data_access.alert_repository import _prepare_payload

        alert = {"alert_id": "existing-123", "name": "Test"}

        result = _prepare_payload(alert)

        assert result["alert_id"] == "existing-123"

    def test_defaults_ratio_to_no(self):
        """Test that ratio defaults to 'No'."""
        from src.data_access.alert_repository import _prepare_payload

        alert = {"name": "Test"}

        result = _prepare_payload(alert)

        assert result["ratio"] == "No"

    def test_derives_is_ratio_from_ratio_string_yes(self):
        """Test that is_ratio is derived from ratio='Yes'."""
        from src.data_access.alert_repository import _prepare_payload

        alert = {"ratio": "Yes"}

        result = _prepare_payload(alert)

        assert result["is_ratio"] is True

    def test_derives_is_ratio_from_ratio_string_true(self):
        """Test that is_ratio is derived from ratio='true'."""
        from src.data_access.alert_repository import _prepare_payload

        alert = {"ratio": "true"}

        result = _prepare_payload(alert)

        assert result["is_ratio"] is True

    def test_derives_is_ratio_from_ratio_string_no(self):
        """Test that is_ratio is False from ratio='No'."""
        from src.data_access.alert_repository import _prepare_payload

        alert = {"ratio": "No"}

        result = _prepare_payload(alert)

        assert result["is_ratio"] is False

    def test_preserves_is_ratio_when_provided(self):
        """Test that explicit is_ratio value is preserved."""
        from src.data_access.alert_repository import _prepare_payload

        alert = {"is_ratio": True}

        result = _prepare_payload(alert)

        assert result["is_ratio"] is True


class TestRowFromPayload:
    """Tests for _row_from_payload function."""

    @patch("src.data_access.alert_repository.Json")
    def test_creates_tuple_with_correct_fields(self, mock_json):
        """Test that row tuple contains all required fields."""
        from src.data_access.alert_repository import _row_from_payload

        mock_json.side_effect = lambda x: f"Json({x})"

        payload = {
            "alert_id": "123",
            "name": "Test Alert",
            "stock_name": "Apple Inc",
            "ticker": "AAPL",
            "ticker1": None,
            "ticker2": None,
            "conditions": [{"field": "price", "operator": ">", "value": 100}],
            "combination_logic": "AND",
            "last_triggered": "2024-01-15T10:30:00",
            "action": "notify",
            "timeframe": "1D",
            "exchange": "NASDAQ",
            "country": "US",
            "ratio": "No",
            "is_ratio": False,
            "adjustment_method": "split",
            "dtp_params": None,
            "multi_timeframe_params": None,
            "mixed_timeframe_params": None,
        }

        result = _row_from_payload(payload)

        assert result[0] == "123"  # alert_id
        assert result[1] == "Test Alert"  # name
        assert result[2] == "Apple Inc"  # stock_name
        assert result[3] == "AAPL"  # ticker
        assert len(result) == 20  # Total number of fields

    @patch("src.data_access.alert_repository.Json")
    def test_parses_last_triggered_timestamp(self, mock_json):
        """Test that last_triggered is parsed when provided as datetime object."""
        from src.data_access.alert_repository import _row_from_payload

        expected_dt = datetime(2024, 1, 15, 10, 30, 0)

        payload = {
            "alert_id": "123",
            "last_triggered": expected_dt,  # Pass datetime object directly
            "conditions": []
        }

        result = _row_from_payload(payload)

        assert result[8] == expected_dt  # last_triggered is at index 8

    @patch("src.data_access.alert_repository.Json")
    def test_wraps_conditions_in_json(self, mock_json):
        """Test that conditions are wrapped in psycopg2.Json."""
        from src.data_access.alert_repository import _row_from_payload

        conditions = [{"field": "price", "operator": ">", "value": 100}]
        payload = {
            "alert_id": "123",
            "conditions": conditions
        }

        _row_from_payload(payload)

        # Check that Json was called with conditions
        assert any(call(conditions) in mock_json.call_args_list for call in mock_json.call_args_list)


class TestListAlertsCached:
    """Tests for list_alerts_cached function."""

    @patch("src.data_access.alert_repository.set_json")
    @patch("src.data_access.alert_repository._fetch_alert_dataframe")
    @patch("src.data_access.alert_repository.get_json")
    def test_returns_cached_alerts_from_redis(self, mock_get_json, mock_fetch_df, mock_set_json):
        """Test that cached alerts are returned from Redis."""
        from src.data_access.alert_repository import list_alerts_cached

        # Clear LRU cache first
        list_alerts_cached.cache_clear()

        cached_data = {
            "items": [
                {"alert_id": "123", "name": "Alert 1"},
                {"alert_id": "456", "name": "Alert 2"}
            ]
        }
        mock_get_json.return_value = cached_data

        result = list_alerts_cached()

        assert len(result) == 2
        assert result[0]["name"] == "Alert 1"
        mock_fetch_df.assert_not_called()

    @patch("src.data_access.alert_repository.set_json")
    @patch("src.data_access.alert_repository._fetch_alert_dataframe")
    @patch("src.data_access.alert_repository.get_json")
    def test_fetches_from_db_when_cache_miss(self, mock_get_json, mock_fetch_df, mock_set_json):
        """Test that alerts are fetched from DB on cache miss."""
        from src.data_access.alert_repository import list_alerts_cached

        list_alerts_cached.cache_clear()

        mock_get_json.return_value = None
        test_df = pd.DataFrame([
            {
                "alert_id": "123",
                "name": "Test Alert",
                "conditions": [],
                "last_triggered": None,
                "raw_payload": None,
                "ratio": "No",
                "is_ratio": False
            }
        ])
        mock_fetch_df.return_value = test_df

        result = list_alerts_cached()

        assert len(result) == 1
        mock_set_json.assert_called_once()

    @patch("src.data_access.alert_repository.set_json")
    @patch("src.data_access.alert_repository._fetch_alert_dataframe")
    @patch("src.data_access.alert_repository.get_json")
    def test_returns_empty_list_for_empty_dataframe(self, mock_get_json, mock_fetch_df, mock_set_json):
        """Test that empty list is returned for empty DataFrame."""
        from src.data_access.alert_repository import list_alerts_cached

        list_alerts_cached.cache_clear()

        mock_get_json.return_value = None
        mock_fetch_df.return_value = pd.DataFrame()

        result = list_alerts_cached()

        assert result == []

    @patch("src.data_access.alert_repository.set_json")
    @patch("src.data_access.alert_repository._fetch_alert_dataframe")
    @patch("src.data_access.alert_repository.get_json")
    def test_caches_results_in_redis(self, mock_get_json, mock_fetch_df, mock_set_json):
        """Test that fetched alerts are cached in Redis."""
        from src.data_access.alert_repository import list_alerts_cached

        list_alerts_cached.cache_clear()

        mock_get_json.return_value = None
        test_df = pd.DataFrame([
            {
                "alert_id": "123",
                "name": "Test",
                "conditions": [],
                "last_triggered": None,
                "raw_payload": None,
                "ratio": "No",
                "is_ratio": False
            }
        ])
        mock_fetch_df.return_value = test_df

        list_alerts_cached()

        assert mock_set_json.called
        call_kwargs = mock_set_json.call_args
        assert "items" in call_kwargs[0][1]
        assert call_kwargs[1]["ttl_seconds"] == 300


class TestListAlerts:
    """Tests for list_alerts function."""

    @patch("src.data_access.alert_repository.list_alerts_cached")
    def test_returns_all_alerts_without_limit(self, mock_cached):
        """Test that all alerts are returned when no limit specified."""
        from src.data_access.alert_repository import list_alerts

        mock_cached.return_value = [
            {"alert_id": "1"},
            {"alert_id": "2"},
            {"alert_id": "3"}
        ]

        result = list_alerts()

        assert len(result) == 3

    @patch("src.data_access.alert_repository.list_alerts_cached")
    def test_limits_results_when_limit_specified(self, mock_cached):
        """Test that results are limited when limit is specified."""
        from src.data_access.alert_repository import list_alerts

        mock_cached.return_value = [
            {"alert_id": "1"},
            {"alert_id": "2"},
            {"alert_id": "3"}
        ]

        result = list_alerts(limit=2)

        assert len(result) == 2
        assert result[0]["alert_id"] == "1"
        assert result[1]["alert_id"] == "2"


class TestGetAlert:
    """Tests for get_alert function."""

    @patch("src.data_access.alert_repository.list_alerts_cached")
    def test_returns_alert_by_id(self, mock_cached):
        """Test that alert is found by ID."""
        from src.data_access.alert_repository import get_alert

        mock_cached.return_value = [
            {"alert_id": "123", "name": "Alert 1"},
            {"alert_id": "456", "name": "Alert 2"}
        ]

        result = get_alert("456")

        assert result is not None
        assert result["alert_id"] == "456"
        assert result["name"] == "Alert 2"

    @patch("src.data_access.alert_repository.list_alerts_cached")
    def test_returns_none_when_alert_not_found(self, mock_cached):
        """Test that None is returned when alert doesn't exist."""
        from src.data_access.alert_repository import get_alert

        mock_cached.return_value = [
            {"alert_id": "123", "name": "Alert 1"}
        ]

        result = get_alert("999")

        assert result is None

    @patch("src.data_access.alert_repository.list_alerts_cached")
    def test_returns_copy_of_alert(self, mock_cached):
        """Test that a copy of the alert is returned."""
        from src.data_access.alert_repository import get_alert

        original = {"alert_id": "123", "name": "Alert 1", "data": [1, 2, 3]}
        mock_cached.return_value = [original]

        result = get_alert("123")

        assert result == original
        assert result is not original  # Different object


class TestClearCache:
    """Tests for _clear_cache function."""

    @patch("src.data_access.alert_repository.delete_key")
    @patch("src.data_access.alert_repository.list_alerts_cached")
    def test_clears_lru_and_redis_cache(self, mock_cached, mock_delete_key):
        """Test that both LRU and Redis caches are cleared."""
        from src.data_access.alert_repository import _clear_cache

        _clear_cache()

        mock_cached.cache_clear.assert_called_once()
        mock_delete_key.assert_called_once()


class TestRefreshAlertCache:
    """Tests for refresh_alert_cache function."""

    @patch("src.data_access.alert_repository._clear_cache")
    def test_calls_clear_cache(self, mock_clear):
        """Test that refresh calls clear cache."""
        from src.data_access.alert_repository import refresh_alert_cache

        refresh_alert_cache()

        mock_clear.assert_called_once()


class TestCreateAlert:
    """Tests for create_alert function."""

    @patch("src.data_access.alert_repository.get_alert")
    @patch("src.data_access.alert_repository._clear_cache")
    @patch("src.data_access.alert_repository._row_from_payload")
    @patch("src.data_access.alert_repository._prepare_payload")
    @patch("src.data_access.alert_repository.db_config")
    def test_inserts_alert_into_database(
        self, mock_db_config, mock_prepare, mock_row_from, mock_clear, mock_get
    ):
        """Test that create_alert inserts into database."""
        from src.data_access.alert_repository import create_alert

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_db_config.get_connection.return_value = mock_conn

        prepared = {"alert_id": "123", "name": "Test"}
        mock_prepare.return_value = prepared
        mock_row_from.return_value = ("123", "Test")
        mock_get.return_value = prepared

        alert = {"name": "Test"}
        result = create_alert(alert)

        mock_cursor.execute.assert_called_once()
        assert "INSERT INTO alerts" in mock_cursor.execute.call_args[0][0]
        mock_conn.commit.assert_called_once()
        mock_clear.assert_called_once()

    @patch("src.data_access.alert_repository.get_alert")
    @patch("src.data_access.alert_repository._clear_cache")
    @patch("src.data_access.alert_repository._row_from_payload")
    @patch("src.data_access.alert_repository._prepare_payload")
    @patch("src.data_access.alert_repository.db_config")
    def test_returns_created_alert(
        self, mock_db_config, mock_prepare, mock_row_from, mock_clear, mock_get
    ):
        """Test that created alert is returned."""
        from src.data_access.alert_repository import create_alert

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_db_config.get_connection.return_value = mock_conn

        prepared = {"alert_id": "123", "name": "Test"}
        mock_prepare.return_value = prepared
        mock_row_from.return_value = ("123", "Test")
        mock_get.return_value = prepared

        result = create_alert({"name": "Test"})

        assert result["alert_id"] == "123"


class TestUpdateAlert:
    """Tests for update_alert function."""

    @patch("src.data_access.alert_repository.get_alert")
    @patch("src.data_access.alert_repository._clear_cache")
    @patch("src.data_access.alert_repository._row_from_payload")
    @patch("src.data_access.alert_repository._prepare_payload")
    @patch("src.data_access.alert_repository.db_config")
    def test_updates_alert_in_database(
        self, mock_db_config, mock_prepare, mock_row_from, mock_clear, mock_get
    ):
        """Test that update_alert updates the database."""
        from src.data_access.alert_repository import update_alert

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_db_config.get_connection.return_value = mock_conn

        existing = {"alert_id": "123", "name": "Old Name"}
        updated = {"alert_id": "123", "name": "New Name"}

        # First call returns existing, second call returns updated
        mock_get.side_effect = [existing, updated]
        mock_prepare.return_value = updated
        mock_row_from.return_value = ("Name", "Stock", "TICK")

        result = update_alert("123", {"name": "New Name"})

        mock_cursor.execute.assert_called_once()
        assert "UPDATE alerts" in mock_cursor.execute.call_args[0][0]
        mock_conn.commit.assert_called_once()
        mock_clear.assert_called_once()

    @patch("src.data_access.alert_repository.get_alert")
    def test_returns_none_when_alert_not_found(self, mock_get):
        """Test that None is returned when alert doesn't exist."""
        from src.data_access.alert_repository import update_alert

        mock_get.return_value = None

        result = update_alert("999", {"name": "New Name"})

        assert result is None

    @patch("src.data_access.alert_repository.get_alert")
    def test_returns_unchanged_alert_when_no_updates(self, mock_get):
        """Test that alert is returned unchanged when updates are empty."""
        from src.data_access.alert_repository import update_alert

        existing = {"alert_id": "123", "name": "Test"}
        mock_get.return_value = existing

        result = update_alert("123", {})

        assert result == existing


class TestDeleteAlert:
    """Tests for delete_alert function."""

    @patch("src.data_access.alert_repository._clear_cache")
    @patch("src.data_access.alert_repository.db_config")
    def test_deletes_alert_from_database(self, mock_db_config, mock_clear):
        """Test that delete_alert removes from database."""
        from src.data_access.alert_repository import delete_alert

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_db_config.get_connection.return_value = mock_conn

        delete_alert("123")

        mock_cursor.execute.assert_called_once()
        assert "DELETE FROM alerts" in mock_cursor.execute.call_args[0][0]
        assert mock_cursor.execute.call_args[0][1] == ("123",)
        mock_conn.commit.assert_called_once()
        mock_clear.assert_called_once()


class TestBulkReplaceAlerts:
    """Tests for bulk_replace_alerts function."""

    @patch("src.data_access.alert_repository._clear_cache")
    @patch("src.data_access.alert_repository.execute_values")
    @patch("src.data_access.alert_repository._row_from_payload")
    @patch("src.data_access.alert_repository._prepare_payload")
    @patch("src.data_access.alert_repository.db_config")
    def test_truncates_and_inserts_alerts(
        self, mock_db_config, mock_prepare, mock_row_from, mock_execute_values, mock_clear
    ):
        """Test that bulk_replace truncates table and inserts new alerts."""
        from src.data_access.alert_repository import bulk_replace_alerts

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_db_config.get_connection.return_value = mock_conn

        alerts = [
            {"name": "Alert 1"},
            {"name": "Alert 2"}
        ]
        mock_prepare.side_effect = [
            {"alert_id": "1", "name": "Alert 1"},
            {"alert_id": "2", "name": "Alert 2"}
        ]
        mock_row_from.side_effect = [
            ("1", "Alert 1"),
            ("2", "Alert 2")
        ]

        bulk_replace_alerts(alerts)

        # Check TRUNCATE was called
        truncate_call = mock_cursor.execute.call_args[0][0]
        assert "TRUNCATE TABLE alerts" in truncate_call

        # Check execute_values was called for INSERT
        mock_execute_values.assert_called_once()
        assert "INSERT INTO alerts" in mock_execute_values.call_args[0][1]

        mock_conn.commit.assert_called_once()
        mock_clear.assert_called_once()

    @patch("src.data_access.alert_repository._clear_cache")
    @patch("src.data_access.alert_repository.execute_values")
    @patch("src.data_access.alert_repository.db_config")
    def test_handles_empty_alerts_list(self, mock_db_config, mock_execute_values, mock_clear):
        """Test that empty list only truncates table."""
        from src.data_access.alert_repository import bulk_replace_alerts

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_db_config.get_connection.return_value = mock_conn

        bulk_replace_alerts([])

        # TRUNCATE should be called
        truncate_call = mock_cursor.execute.call_args[0][0]
        assert "TRUNCATE TABLE alerts" in truncate_call

        # execute_values should NOT be called for empty list
        mock_execute_values.assert_not_called()

        mock_conn.commit.assert_called_once()
        mock_clear.assert_called_once()

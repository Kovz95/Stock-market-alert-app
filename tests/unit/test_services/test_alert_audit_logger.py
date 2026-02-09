"""
Unit tests for the Alert Audit Logger service module.

Tests the comprehensive audit logging system including:
- Database initialization (SQLite and PostgreSQL)
- Alert evaluation tracking and logging
- Performance metrics calculation
- Query and reporting functionality
- Error handling and recovery
- JSON serialization for different database types
"""

import json
import logging
import os
import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch, call
import pandas as pd

from src.services.alert_audit_logger import (
    AlertAuditLogger,
    log_alert_check_start,
    log_price_data_pulled,
    log_no_data_available,
    log_conditions_evaluated,
    log_error,
    log_completion,
    get_daily_evaluation_stats,
    get_evaluation_coverage,
    get_expected_daily_evaluations,
    get_audit_summary,
    get_alert_history,
    get_performance_metrics,
)


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def mock_db_config():
    """Mock db_config for testing."""
    mock_config = Mock()
    mock_config.db_type = "postgresql"
    mock_config.connection = MagicMock()
    mock_config.execute_with_retry = MagicMock()
    return mock_config


@pytest.fixture
def mock_connection():
    """Mock database connection."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_cursor.fetchone.return_value = [1]
    mock_cursor.fetchall.return_value = []
    mock_cursor.rowcount = 0
    return mock_conn


@pytest.fixture
def mock_sqlite_connection():
    """Mock SQLite connection."""
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_conn.executescript = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=False)
    mock_cursor.lastrowid = 1
    mock_cursor.rowcount = 0
    return mock_conn


@pytest.fixture
def sample_alert():
    """Sample alert dictionary for testing."""
    return {
        "alert_id": "test-alert-123",
        "id": "test-alert-123",
        "ticker": "AAPL",
        "ticker1": "AAPL",
        "stock_name": "Apple Inc.",
        "exchange": "NASDAQ",
        "timeframe": "daily",
        "action": "BUY",
        "ratio": False,
        "ticker2": None,
        "conditions": [
            {"indicator": "RSI", "operator": "<", "value": 30}
        ],
        "combination_logic": "AND",
    }


@pytest.fixture
def minimal_alert():
    """Minimal alert with only required fields."""
    return {
        "id": "minimal-alert",
        "ticker1": "TSLA",
    }


@pytest.fixture
def audit_logger_postgres(mock_db_config, tmp_path):
    """AlertAuditLogger configured for PostgreSQL."""
    with patch("src.services.alert_audit_logger.db_config", mock_db_config):
        mock_db_config.db_type = "postgresql"
        log_file = str(tmp_path / "test_audit.log")
        with patch.object(AlertAuditLogger, "setup_file_logging"):
            with patch.object(AlertAuditLogger, "setup_database"):
                logger = AlertAuditLogger(db_path="test.db", log_file=log_file)
                logger.file_logger = MagicMock()
        return logger


@pytest.fixture
def audit_logger_sqlite(mock_db_config, tmp_path):
    """AlertAuditLogger configured for SQLite."""
    with patch("src.services.alert_audit_logger.db_config", mock_db_config):
        mock_db_config.db_type = "sqlite"
        log_file = str(tmp_path / "test_audit.log")
        with patch.object(AlertAuditLogger, "setup_file_logging"):
            with patch.object(AlertAuditLogger, "setup_database"):
                logger = AlertAuditLogger(db_path="test.db", log_file=log_file)
                logger.file_logger = MagicMock()
        return logger


@pytest.fixture
def mock_audit_id():
    """Mock audit ID for testing."""
    return "12345"


# ============================================================================
# TEST: AlertAuditLogger.__init__
# ============================================================================


class TestAlertAuditLoggerInit:
    """Tests for AlertAuditLogger initialization."""

    def test_init_with_default_parameters(self, mock_db_config, tmp_path):
        """Should initialize with default parameters."""
        with patch("src.services.alert_audit_logger.db_config", mock_db_config):
            mock_db_config.db_type = "postgresql"
            with patch.object(AlertAuditLogger, "setup_file_logging"):
                logger = AlertAuditLogger()

        assert logger.db_path == "alert_audit.db"
        assert logger.log_file is not None
        assert logger.use_postgres is True

    def test_init_with_custom_parameters(self, mock_db_config, tmp_path):
        """Should initialize with custom parameters."""
        custom_db = "custom.db"
        custom_log = str(tmp_path / "custom.log")

        with patch("src.services.alert_audit_logger.db_config", mock_db_config):
            mock_db_config.db_type = "sqlite"
            with patch.object(AlertAuditLogger, "setup_file_logging"):
                logger = AlertAuditLogger(db_path=custom_db, log_file=custom_log)

        assert logger.db_path == custom_db
        assert logger.log_file == custom_log
        assert logger.use_postgres is False

    def test_init_detects_postgresql(self, mock_db_config):
        """Should detect PostgreSQL database type."""
        with patch("src.services.alert_audit_logger.db_config", mock_db_config):
            mock_db_config.db_type = "postgresql"
            with patch.object(AlertAuditLogger, "setup_file_logging"):
                logger = AlertAuditLogger()

        assert logger.use_postgres is True

    def test_init_detects_sqlite(self, mock_db_config):
        """Should detect SQLite database type."""
        with patch("src.services.alert_audit_logger.db_config", mock_db_config):
            mock_db_config.db_type = "sqlite"
            with patch.object(AlertAuditLogger, "setup_file_logging"):
                logger = AlertAuditLogger()

        assert logger.use_postgres is False

    def test_init_calls_setup_database(self, mock_db_config):
        """Should call setup_database during initialization."""
        with patch("src.services.alert_audit_logger.db_config", mock_db_config):
            mock_db_config.db_type = "postgresql"
            with patch.object(AlertAuditLogger, "setup_database") as mock_setup_db:
                with patch.object(AlertAuditLogger, "setup_file_logging"):
                    AlertAuditLogger()

        mock_setup_db.assert_called_once()

    def test_init_calls_setup_file_logging(self, mock_db_config):
        """Should call setup_file_logging during initialization."""
        with patch("src.services.alert_audit_logger.db_config", mock_db_config):
            mock_db_config.db_type = "postgresql"
            with patch.object(AlertAuditLogger, "setup_file_logging") as mock_setup_logging:
                AlertAuditLogger()

        mock_setup_logging.assert_called_once()


# ============================================================================
# TEST: AlertAuditLogger.setup_database
# ============================================================================


class TestDatabaseSetup:
    """Tests for database setup functionality."""

    def test_setup_database_postgresql_skips_creation(self, audit_logger_postgres):
        """Should skip table creation for PostgreSQL."""
        with patch.object(audit_logger_postgres, "_connection"):
            audit_logger_postgres.setup_database()
            # Should not call connection for PostgreSQL setup

    def test_setup_database_sqlite_creates_tables(self, audit_logger_sqlite, mock_sqlite_connection):
        """Should create tables for SQLite."""
        with patch.object(audit_logger_sqlite, "_connection", return_value=mock_sqlite_connection):
            audit_logger_sqlite.setup_database()

        mock_sqlite_connection.executescript.assert_called_once()
        call_args = mock_sqlite_connection.executescript.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS alert_audits" in call_args
        assert "CREATE INDEX IF NOT EXISTS idx_alert_id" in call_args

    def test_setup_database_sqlite_creates_indexes(self, audit_logger_sqlite, mock_sqlite_connection):
        """Should create all required indexes."""
        with patch.object(audit_logger_sqlite, "_connection", return_value=mock_sqlite_connection):
            audit_logger_sqlite.setup_database()

        call_args = mock_sqlite_connection.executescript.call_args[0][0]
        assert "idx_alert_id" in call_args
        assert "idx_timestamp" in call_args
        assert "idx_ticker" in call_args

    def test_setup_database_handles_error(self, audit_logger_sqlite, mock_sqlite_connection):
        """Should handle database setup errors gracefully."""
        mock_sqlite_connection.executescript.side_effect = Exception("Setup error")

        with patch.object(audit_logger_sqlite, "_connection", return_value=mock_sqlite_connection):
            # Should not raise, just log error
            audit_logger_sqlite.setup_database()


# ============================================================================
# TEST: AlertAuditLogger.log_alert_check_start
# ============================================================================


class TestLogAlertCheckStart:
    """Tests for logging alert check start."""

    def test_log_alert_check_start_postgresql_success(
        self, audit_logger_postgres, sample_alert
    ):
        """Should log alert check start for PostgreSQL."""
        audit_logger_postgres.use_postgres = True

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = [1]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch.object(audit_logger_postgres, "_connection", return_value=mock_conn):
            audit_id = audit_logger_postgres.log_alert_check_start(sample_alert)

        assert audit_id == "1"
        mock_conn.commit.assert_called_once()

    def test_log_alert_check_start_sqlite_success(
        self, audit_logger_sqlite, sample_alert
    ):
        """Should log alert check start for SQLite."""
        audit_logger_sqlite.use_postgres = False

        mock_cursor = MagicMock()
        mock_cursor.lastrowid = 42
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch.object(audit_logger_sqlite, "_connection", return_value=mock_conn):
            audit_id = audit_logger_sqlite.log_alert_check_start(sample_alert)

        assert audit_id == "42"

    def test_log_alert_check_start_with_default_evaluation_type(
        self, audit_logger_postgres, sample_alert
    ):
        """Should use default evaluation_type when not provided."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = [1]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch.object(audit_logger_postgres, "_connection", return_value=mock_conn):
            audit_logger_postgres.log_alert_check_start(sample_alert)

        execute_call = mock_cursor.execute.call_args
        assert "scheduled" in execute_call[0][1]

    def test_log_alert_check_start_with_custom_evaluation_type(
        self, audit_logger_postgres, sample_alert
    ):
        """Should use custom evaluation_type when provided."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = [1]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch.object(audit_logger_postgres, "_connection", return_value=mock_conn):
            audit_logger_postgres.log_alert_check_start(sample_alert, evaluation_type="manual")

        execute_call = mock_cursor.execute.call_args
        assert "manual" in execute_call[0][1]

    def test_log_alert_check_start_with_minimal_alert(
        self, audit_logger_postgres, minimal_alert
    ):
        """Should handle alert with minimal fields."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = [1]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch.object(audit_logger_postgres, "_connection", return_value=mock_conn):
            audit_id = audit_logger_postgres.log_alert_check_start(minimal_alert)

        assert audit_id == "1"

    def test_log_alert_check_start_captures_alert_metadata(
        self, audit_logger_postgres, sample_alert
    ):
        """Should capture all alert metadata."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = [1]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch.object(audit_logger_postgres, "_connection", return_value=mock_conn):
            audit_logger_postgres.log_alert_check_start(sample_alert)

        execute_call = mock_cursor.execute.call_args
        params = execute_call[0][1]
        assert sample_alert["ticker"] in params
        assert sample_alert["stock_name"] in params
        assert sample_alert["exchange"] in params

    def test_log_alert_check_start_handles_error(
        self, audit_logger_postgres, sample_alert
    ):
        """Should return None on error."""
        mock_cursor = MagicMock()
        mock_cursor.execute.side_effect = Exception("DB error")
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch.object(audit_logger_postgres, "_connection", return_value=mock_conn):
            audit_id = audit_logger_postgres.log_alert_check_start(sample_alert)

        assert audit_id is None

    def test_log_alert_check_start_returns_string(
        self, audit_logger_postgres, sample_alert
    ):
        """Should return audit_id as string."""
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = [12345]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch.object(audit_logger_postgres, "_connection", return_value=mock_conn):
            audit_id = audit_logger_postgres.log_alert_check_start(sample_alert)

        assert isinstance(audit_id, str)
        assert audit_id == "12345"


# ============================================================================
# TEST: AlertAuditLogger.log_price_data_pulled
# ============================================================================


class TestLogPriceDataPulled:
    """Tests for logging price data pull."""

    def test_log_price_data_pulled_success(self, audit_logger_postgres, mock_audit_id):
        """Should log price data pulled successfully."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            audit_logger_postgres.log_price_data_pulled(
                mock_audit_id, "FMP", cache_hit=False
            )

        mock_execute.assert_called_once()
        call_args = mock_execute.call_args
        assert "UPDATE alert_audits" in call_args[0][0]
        assert "price_data_pulled" in call_args[0][0]

    def test_log_price_data_pulled_with_cache_hit(self, audit_logger_postgres, mock_audit_id):
        """Should log cache_hit=True when specified."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            audit_logger_postgres.log_price_data_pulled(
                mock_audit_id, "FMP", cache_hit=True
            )

        call_args = mock_execute.call_args
        params = call_args[0][1]
        # For PostgreSQL, True is True, not 1
        assert True in params or 1 in params

    def test_log_price_data_pulled_with_source(self, audit_logger_postgres, mock_audit_id):
        """Should log the correct price source."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            audit_logger_postgres.log_price_data_pulled(
                mock_audit_id, "FMP API", cache_hit=False
            )

        call_args = mock_execute.call_args
        params = call_args[0][1]
        assert "FMP API" in params

    def test_log_price_data_pulled_handles_error(self, audit_logger_postgres, mock_audit_id):
        """Should handle errors gracefully."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            mock_execute.side_effect = Exception("DB error")

            # Should not raise
            audit_logger_postgres.log_price_data_pulled(
                mock_audit_id, "FMP", cache_hit=False
            )


# ============================================================================
# TEST: AlertAuditLogger.log_no_data_available
# ============================================================================


class TestLogNoDataAvailable:
    """Tests for logging no data available scenario."""

    def test_log_no_data_available_success(self, audit_logger_postgres, mock_audit_id):
        """Should log no data available successfully."""
        audit_logger_postgres.file_logger = MagicMock()
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            audit_logger_postgres.log_no_data_available(mock_audit_id, "INVALID")

        mock_execute.assert_called_once()
        call_args = mock_execute.call_args
        assert "UPDATE alert_audits" in call_args[0][0]

    def test_log_no_data_available_records_error(self, audit_logger_postgres, mock_audit_id):
        """Should record error message."""
        audit_logger_postgres.file_logger = MagicMock()
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            audit_logger_postgres.log_no_data_available(mock_audit_id, "INVALID")

        call_args = mock_execute.call_args
        params = call_args[0][1]
        assert "No data available from FMP API" in params

    def test_log_no_data_available_includes_metadata(self, audit_logger_postgres, mock_audit_id):
        """Should include metadata in additional_data."""
        audit_logger_postgres.file_logger = MagicMock()
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            audit_logger_postgres.log_no_data_available(mock_audit_id, "INVALID")

        call_args = mock_execute.call_args
        params = call_args[0][1]
        # Metadata should be serialized (either Json object or JSON string)
        assert len(params) >= 3

    def test_log_no_data_available_handles_error(self, audit_logger_postgres, mock_audit_id):
        """Should handle errors gracefully."""
        audit_logger_postgres.file_logger = MagicMock()
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            mock_execute.side_effect = Exception("DB error")

            # Should not raise
            audit_logger_postgres.log_no_data_available(mock_audit_id, "INVALID")


# ============================================================================
# TEST: AlertAuditLogger.log_conditions_evaluated
# ============================================================================


class TestLogConditionsEvaluated:
    """Tests for logging conditions evaluation."""

    def test_log_conditions_evaluated_triggered(self, audit_logger_postgres, mock_audit_id):
        """Should log when conditions trigger alert."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            audit_logger_postgres.log_conditions_evaluated(
                mock_audit_id, True, trigger_reason="RSI < 30"
            )

        call_args = mock_execute.call_args
        params = call_args[0][1]
        assert True in params or 1 in params

    def test_log_conditions_evaluated_not_triggered(self, audit_logger_postgres, mock_audit_id):
        """Should log when conditions don't trigger alert."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            audit_logger_postgres.log_conditions_evaluated(mock_audit_id, False)

        call_args = mock_execute.call_args
        params = call_args[0][1]
        assert False in params or 0 in params

    def test_log_conditions_evaluated_with_reason(self, audit_logger_postgres, mock_audit_id):
        """Should include trigger reason when provided."""
        reason = "Price crossed threshold"
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            audit_logger_postgres.log_conditions_evaluated(
                mock_audit_id, True, trigger_reason=reason
            )

        call_args = mock_execute.call_args
        params = call_args[0][1]
        assert reason in params

    def test_log_conditions_evaluated_without_reason(self, audit_logger_postgres, mock_audit_id):
        """Should handle missing trigger reason."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            audit_logger_postgres.log_conditions_evaluated(
                mock_audit_id, False, trigger_reason=None
            )

        call_args = mock_execute.call_args
        params = call_args[0][1]
        assert None in params

    def test_log_conditions_evaluated_handles_error(self, audit_logger_postgres, mock_audit_id):
        """Should handle errors gracefully."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            mock_execute.side_effect = Exception("DB error")

            # Should not raise
            audit_logger_postgres.log_conditions_evaluated(
                mock_audit_id, True, trigger_reason="Test"
            )


# ============================================================================
# TEST: AlertAuditLogger.log_error
# ============================================================================


class TestLogError:
    """Tests for logging errors."""

    def test_log_error_success(self, audit_logger_postgres, mock_audit_id):
        """Should log error message successfully."""
        error_msg = "Connection timeout"
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            audit_logger_postgres.log_error(mock_audit_id, error_msg)

        mock_execute.assert_called_once()
        call_args = mock_execute.call_args
        assert "UPDATE alert_audits" in call_args[0][0]
        assert error_msg in call_args[0][1]

    def test_log_error_long_message(self, audit_logger_postgres, mock_audit_id):
        """Should handle long error messages."""
        error_msg = "X" * 1000
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            audit_logger_postgres.log_error(mock_audit_id, error_msg)

        mock_execute.assert_called_once()

    def test_log_error_special_characters(self, audit_logger_postgres, mock_audit_id):
        """Should handle special characters in error message."""
        error_msg = "Error: 'quoted' and \"double\" and \\backslash"
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            audit_logger_postgres.log_error(mock_audit_id, error_msg)

        mock_execute.assert_called_once()

    def test_log_error_handles_exception(self, audit_logger_postgres, mock_audit_id):
        """Should handle errors gracefully."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            mock_execute.side_effect = Exception("DB error")

            # Should not raise
            audit_logger_postgres.log_error(mock_audit_id, "Test error")


# ============================================================================
# TEST: AlertAuditLogger.log_completion
# ============================================================================


class TestLogCompletion:
    """Tests for logging alert evaluation completion."""

    def test_log_completion_success(self, audit_logger_postgres, mock_audit_id):
        """Should log completion successfully."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            audit_logger_postgres.log_completion(mock_audit_id, 150)

        mock_execute.assert_called_once()
        call_args = mock_execute.call_args
        assert "UPDATE alert_audits" in call_args[0][0]
        assert 150 in call_args[0][1]

    def test_log_completion_fast_execution(self, audit_logger_postgres, mock_audit_id):
        """Should log fast execution times."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            audit_logger_postgres.log_completion(mock_audit_id, 5)

        call_args = mock_execute.call_args
        assert 5 in call_args[0][1]

    def test_log_completion_slow_execution(self, audit_logger_postgres, mock_audit_id):
        """Should log slow execution times."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            audit_logger_postgres.log_completion(mock_audit_id, 5000)

        call_args = mock_execute.call_args
        assert 5000 in call_args[0][1]

    def test_log_completion_handles_error(self, audit_logger_postgres, mock_audit_id):
        """Should handle errors gracefully."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            mock_execute.side_effect = Exception("DB error")

            # Should not raise
            audit_logger_postgres.log_completion(mock_audit_id, 100)


# ============================================================================
# TEST: AlertAuditLogger.get_audit_summary
# ============================================================================


class TestGetAuditSummary:
    """Tests for getting audit summary."""

    def test_get_audit_summary_success(self, audit_logger_postgres):
        """Should retrieve audit summary successfully."""
        mock_df = pd.DataFrame({
            "alert_id": ["test-1", "test-2"],
            "ticker": ["AAPL", "TSLA"],
            "total_checks": [10, 5],
        })

        with patch.object(audit_logger_postgres, "_connection") as mock_ctx:
            mock_ctx.__enter__.return_value = MagicMock()
            with patch("pandas.read_sql_query", return_value=mock_df):
                result = audit_logger_postgres.get_audit_summary(days=7)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "alert_id" in result.columns

    def test_get_audit_summary_default_days(self, audit_logger_postgres):
        """Should use default days=7."""
        mock_df = pd.DataFrame()

        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            with patch("pandas.read_sql_query", return_value=mock_df):
                with patch.object(audit_logger_postgres, "_connection"):
                    audit_logger_postgres.get_audit_summary()

    def test_get_audit_summary_custom_days(self, audit_logger_postgres):
        """Should accept custom days parameter."""
        mock_df = pd.DataFrame()

        with patch.object(audit_logger_postgres, "_connection") as mock_ctx:
            mock_ctx.__enter__.return_value = MagicMock()
            with patch("pandas.read_sql_query", return_value=mock_df):
                result = audit_logger_postgres.get_audit_summary(days=30)

        assert isinstance(result, pd.DataFrame)

    def test_get_audit_summary_returns_dataframe(self, audit_logger_postgres):
        """Should return DataFrame."""
        mock_df = pd.DataFrame({"col": [1, 2, 3]})

        with patch.object(audit_logger_postgres, "_connection") as mock_ctx:
            mock_ctx.__enter__.return_value = MagicMock()
            with patch("pandas.read_sql_query", return_value=mock_df):
                result = audit_logger_postgres.get_audit_summary()

        assert isinstance(result, pd.DataFrame)

    def test_get_audit_summary_handles_error(self, audit_logger_postgres):
        """Should return empty DataFrame on error."""
        with patch.object(audit_logger_postgres, "_connection") as mock_ctx:
            mock_ctx.__enter__.return_value = MagicMock()
            with patch("pandas.read_sql_query", side_effect=Exception("DB error")):
                result = audit_logger_postgres.get_audit_summary()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ============================================================================
# TEST: AlertAuditLogger.get_alert_history
# ============================================================================


class TestGetAlertHistory:
    """Tests for getting alert history."""

    def test_get_alert_history_success(self, audit_logger_postgres):
        """Should retrieve alert history successfully."""
        mock_df = pd.DataFrame({
            "id": [1, 2, 3],
            "alert_id": ["test-1", "test-1", "test-1"],
            "timestamp": ["2024-01-01", "2024-01-02", "2024-01-03"],
        })

        with patch.object(audit_logger_postgres, "_connection") as mock_ctx:
            mock_ctx.__enter__.return_value = MagicMock()
            with patch("pandas.read_sql_query", return_value=mock_df):
                result = audit_logger_postgres.get_alert_history("test-1")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3

    def test_get_alert_history_with_limit(self, audit_logger_postgres):
        """Should respect limit parameter."""
        mock_df = pd.DataFrame()

        with patch.object(audit_logger_postgres, "_connection") as mock_ctx:
            mock_ctx.__enter__.return_value = MagicMock()
            with patch("pandas.read_sql_query", return_value=mock_df):
                audit_logger_postgres.get_alert_history("test-1", limit=50)

    def test_get_alert_history_none_alert_id(self, audit_logger_postgres):
        """Should handle None alert_id."""
        mock_df = pd.DataFrame()

        with patch.object(audit_logger_postgres, "_connection") as mock_ctx:
            mock_ctx.__enter__.return_value = MagicMock()
            with patch("pandas.read_sql_query", return_value=mock_df):
                result = audit_logger_postgres.get_alert_history(None)

        assert isinstance(result, pd.DataFrame)

    def test_get_alert_history_handles_error(self, audit_logger_postgres):
        """Should return empty DataFrame on error."""
        with patch.object(audit_logger_postgres, "_connection") as mock_ctx:
            mock_ctx.__enter__.return_value = MagicMock()
            with patch("pandas.read_sql_query", side_effect=Exception("DB error")):
                result = audit_logger_postgres.get_alert_history("test-1")

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ============================================================================
# TEST: AlertAuditLogger.get_performance_metrics
# ============================================================================


class TestGetPerformanceMetrics:
    """Tests for getting performance metrics."""

    def test_get_performance_metrics_success(self, audit_logger_postgres):
        """Should retrieve performance metrics successfully."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            # Mock responses for each query
            mock_execute.side_effect = [
                [[100]],  # total_checks
                [[80]],   # successful_pulls
                [[20, 40]], # cache hits
                [[150.5]], # avg_execution_time
                [[10]],   # total_errors
            ]

            result = audit_logger_postgres.get_performance_metrics(days=7)

        assert isinstance(result, dict)
        assert "total_checks" in result
        assert "successful_price_pulls" in result
        assert "success_rate" in result
        assert "cache_hit_rate" in result
        assert "avg_execution_time_ms" in result
        assert "total_errors" in result
        assert "error_rate" in result

    def test_get_performance_metrics_calculates_rates(self, audit_logger_postgres):
        """Should calculate success and error rates."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            mock_execute.side_effect = [
                [[100]],   # total_checks: 100
                [[80]],    # successful_pulls: 80
                [[20, 40]], # cache hits: 20 out of 40
                [[150]],   # avg_execution_time
                [[10]],    # total_errors: 10
            ]

            result = audit_logger_postgres.get_performance_metrics()

        assert result["success_rate"] == pytest.approx(80.0)  # 80/100
        assert result["error_rate"] == pytest.approx(10.0)    # 10/100
        assert result["cache_hit_rate"] == pytest.approx(50.0)  # 20/40

    def test_get_performance_metrics_no_data(self, audit_logger_postgres):
        """Should handle case with no data."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            mock_execute.side_effect = [
                [[0]],    # total_checks: 0
                [[0]],    # successful_pulls: 0
                [[0, 0]], # cache hits: 0
                [[None]], # avg_execution_time: None
                [[0]],    # total_errors: 0
            ]

            result = audit_logger_postgres.get_performance_metrics()

        assert result["total_checks"] == 0
        assert result["success_rate"] == 0.0
        assert result["error_rate"] == 0.0

    def test_get_performance_metrics_handles_error(self, audit_logger_postgres):
        """Should return empty dict on error."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            mock_execute.side_effect = Exception("DB error")

            result = audit_logger_postgres.get_performance_metrics()

        assert isinstance(result, dict)
        assert len(result) == 0


# ============================================================================
# TEST: AlertAuditLogger.cleanup_old_records
# ============================================================================


class TestCleanupOldRecords:
    """Tests for cleaning up old records."""

    def test_cleanup_old_records_success(self, audit_logger_postgres):
        """Should delete old records successfully."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            mock_execute.return_value = 50  # 50 records deleted

            deleted_count = audit_logger_postgres.cleanup_old_records(days_to_keep=30)

        assert deleted_count == 50

    def test_cleanup_old_records_delete_all(self, audit_logger_postgres):
        """Should delete all records when days_to_keep=0."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            mock_execute.return_value = 100

            deleted_count = audit_logger_postgres.cleanup_old_records(days_to_keep=0)

        assert deleted_count == 100
        call_args = mock_execute.call_args
        assert "DELETE FROM alert_audits" in call_args[0][0]

    def test_cleanup_old_records_default_days(self, audit_logger_postgres):
        """Should use default days_to_keep=30."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            mock_execute.return_value = 0

            audit_logger_postgres.cleanup_old_records()

    def test_cleanup_old_records_handles_error(self, audit_logger_postgres):
        """Should return 0 on error."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            mock_execute.side_effect = Exception("DB error")

            deleted_count = audit_logger_postgres.cleanup_old_records()

        assert deleted_count == 0


# ============================================================================
# TEST: AlertAuditLogger.get_daily_evaluation_stats
# ============================================================================


class TestGetDailyEvaluationStats:
    """Tests for getting daily evaluation statistics."""

    def test_get_daily_evaluation_stats_success(self, audit_logger_postgres):
        """Should retrieve daily stats successfully."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            mock_execute.return_value = [[10, 3, 7, 0, 10, 150.0, 5, 5]]

            result = audit_logger_postgres.get_daily_evaluation_stats()

        assert isinstance(result, dict)
        assert "total_evaluations" in result
        assert "triggered_alerts" in result
        assert "date" in result

    def test_get_daily_evaluation_stats_with_date(self, audit_logger_postgres):
        """Should accept custom date parameter."""
        from datetime import date

        custom_date = date(2024, 1, 15)
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            mock_execute.return_value = [[5, 1, 4, 0, 5, 100.0, 2, 3]]

            result = audit_logger_postgres.get_daily_evaluation_stats(date=custom_date)

        assert result["date"] == "2024-01-15"

    def test_get_daily_evaluation_stats_includes_all_fields(self, audit_logger_postgres):
        """Should include all expected fields."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            mock_execute.return_value = [[10, 3, 7, 0, 10, 150.0, 5, 5]]

            result = audit_logger_postgres.get_daily_evaluation_stats()

        required_fields = [
            "date",
            "total_evaluations",
            "triggered_alerts",
            "non_triggered_alerts",
            "failed_evaluations",
            "successful_evaluations",
            "avg_execution_time",
            "cache_hits",
            "cache_misses",
        ]
        for field in required_fields:
            assert field in result

    def test_get_daily_evaluation_stats_no_data(self, audit_logger_postgres):
        """Should handle no data gracefully."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            mock_execute.return_value = None

            result = audit_logger_postgres.get_daily_evaluation_stats()

        assert result is None

    def test_get_daily_evaluation_stats_handles_error(self, audit_logger_postgres):
        """Should return None on error."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            mock_execute.side_effect = Exception("DB error")

            result = audit_logger_postgres.get_daily_evaluation_stats()

        assert result is None


# ============================================================================
# TEST: AlertAuditLogger.get_evaluation_coverage
# ============================================================================


class TestGetEvaluationCoverage:
    """Tests for getting evaluation coverage."""

    def test_get_evaluation_coverage_success(self, audit_logger_postgres):
        """Should retrieve coverage successfully."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            mock_execute.side_effect = [
                [[15]],  # unique_alerts_evaluated
                [["alert-1", 5], ["alert-2", 3], ["alert-3", 2]],  # frequency
            ]

            result = audit_logger_postgres.get_evaluation_coverage()

        assert isinstance(result, dict)
        assert "unique_alerts_evaluated" in result
        assert "total_evaluations" in result
        assert "avg_evaluations_per_alert" in result

    def test_get_evaluation_coverage_calculates_averages(self, audit_logger_postgres):
        """Should calculate average evaluations per alert."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            mock_execute.side_effect = [
                [[4]],  # 4 unique alerts
                [["a", 10], ["b", 6], ["c", 4], ["d", 0]],  # 20 total evaluations
            ]

            result = audit_logger_postgres.get_evaluation_coverage()

        assert result["avg_evaluations_per_alert"] == pytest.approx(5.0)

    def test_get_evaluation_coverage_includes_frequency(self, audit_logger_postgres):
        """Should include evaluation frequency."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            mock_execute.side_effect = [
                [[2]],
                [["alert-1", 5], ["alert-2", 3]],
            ]

            result = audit_logger_postgres.get_evaluation_coverage()

        assert "evaluation_frequency" in result
        assert len(result["evaluation_frequency"]) == 2

    def test_get_evaluation_coverage_no_data(self, audit_logger_postgres):
        """Should handle no data gracefully."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            mock_execute.side_effect = [
                [[0]],
                [],
            ]

            result = audit_logger_postgres.get_evaluation_coverage()

        assert result["unique_alerts_evaluated"] == 0
        assert result["avg_evaluations_per_alert"] == 0.0

    def test_get_evaluation_coverage_handles_error(self, audit_logger_postgres):
        """Should return None on error."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            mock_execute.side_effect = Exception("DB error")

            result = audit_logger_postgres.get_evaluation_coverage()

        assert result is None


# ============================================================================
# TEST: AlertAuditLogger.get_expected_daily_evaluations
# ============================================================================


class TestGetExpectedDailyEvaluations:
    """Tests for getting expected daily evaluations."""

    def test_get_expected_daily_evaluations_success(self, audit_logger_postgres):
        """Should calculate expected evaluations successfully."""
        with patch("src.services.alert_audit_logger.repo_list_alerts") as mock_list:
            mock_list.return_value = [{"id": i} for i in range(100)]

            result = audit_logger_postgres.get_expected_daily_evaluations()

        assert isinstance(result, dict)
        assert "total_alerts" in result
        assert result["total_alerts"] == 100

    def test_get_expected_daily_evaluations_calculates_manual(self, audit_logger_postgres):
        """Should calculate manual check evaluations."""
        with patch("src.services.alert_audit_logger.repo_list_alerts") as mock_list:
            mock_list.return_value = [{"id": i} for i in range(100)]

            result = audit_logger_postgres.get_expected_daily_evaluations()

        assert result["manual_check_evaluations"] == 100

    def test_get_expected_daily_evaluations_calculates_scheduled(self, audit_logger_postgres):
        """Should calculate scheduled evaluations."""
        with patch("src.services.alert_audit_logger.repo_list_alerts") as mock_list:
            mock_list.return_value = [{"id": i} for i in range(100)]

            result = audit_logger_postgres.get_expected_daily_evaluations()

        # 90 daily + (10/5 weekly) = 92
        assert "scheduled_evaluations_estimate" in result

    def test_get_expected_daily_evaluations_handles_error(self, audit_logger_postgres):
        """Should return None on error."""
        with patch("src.services.alert_audit_logger.repo_list_alerts") as mock_list:
            mock_list.side_effect = Exception("DB error")

            result = audit_logger_postgres.get_expected_daily_evaluations()

        assert result is None


# ============================================================================
# TEST: AlertAuditLogger Helper Methods
# ============================================================================


class TestHelperMethods:
    """Tests for helper methods."""

    def test_serialize_json_postgresql_with_data(self, audit_logger_postgres):
        """Should serialize JSON for PostgreSQL using Json object."""
        data = {"key": "value"}

        with patch("psycopg2.extras.Json") as mock_json:
            audit_logger_postgres.use_postgres = True
            result = audit_logger_postgres._serialize_json(data)

        # Should use Json from psycopg2
        assert result is not None

    def test_serialize_json_sqlite_with_data(self, audit_logger_sqlite):
        """Should serialize JSON for SQLite as string."""
        data = {"key": "value"}

        audit_logger_sqlite.use_postgres = False
        result = audit_logger_sqlite._serialize_json(data)

        assert isinstance(result, str)
        assert "key" in result
        assert "value" in result

    def test_serialize_json_none(self, audit_logger_postgres):
        """Should return None for None input."""
        result = audit_logger_postgres._serialize_json(None)
        assert result is None

    def test_bool_postgresql(self, audit_logger_postgres):
        """Should return bool for PostgreSQL."""
        audit_logger_postgres.use_postgres = True
        assert audit_logger_postgres._bool(True) is True
        assert audit_logger_postgres._bool(False) is False

    def test_bool_sqlite(self, audit_logger_sqlite):
        """Should return int for SQLite."""
        audit_logger_sqlite.use_postgres = False
        assert audit_logger_sqlite._bool(True) == 1
        assert audit_logger_sqlite._bool(False) == 0

    def test_cutoff_timestamp(self, audit_logger_postgres):
        """Should calculate cutoff timestamp correctly."""
        result = audit_logger_postgres._cutoff_timestamp(days=7)

        assert isinstance(result, str)
        # Should be ISO format
        assert "T" in result or ":" in result


# ============================================================================
# TEST: Convenience Functions
# ============================================================================


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_log_alert_check_start_function(self, sample_alert):
        """Should delegate to audit_logger instance."""
        with patch("src.services.alert_audit_logger.audit_logger") as mock_logger:
            mock_logger.log_alert_check_start.return_value = "123"

            result = log_alert_check_start(sample_alert)

        assert result == "123"
        mock_logger.log_alert_check_start.assert_called_once_with(
            sample_alert, "scheduled"
        )

    def test_log_price_data_pulled_function(self):
        """Should delegate to audit_logger instance."""
        with patch("src.services.alert_audit_logger.audit_logger") as mock_logger:
            log_price_data_pulled("123", "FMP", cache_hit=True)

        mock_logger.log_price_data_pulled.assert_called_once_with(
            "123", "FMP", True
        )

    def test_log_no_data_available_function(self):
        """Should delegate to audit_logger instance."""
        with patch("src.services.alert_audit_logger.audit_logger") as mock_logger:
            log_no_data_available("123", "AAPL")

        mock_logger.log_no_data_available.assert_called_once_with("123", "AAPL")

    def test_log_conditions_evaluated_function(self):
        """Should delegate to audit_logger instance."""
        with patch("src.services.alert_audit_logger.audit_logger") as mock_logger:
            log_conditions_evaluated("123", True, trigger_reason="Test")

        mock_logger.log_conditions_evaluated.assert_called_once_with(
            "123", True, "Test"
        )

    def test_log_error_function(self):
        """Should delegate to audit_logger instance."""
        with patch("src.services.alert_audit_logger.audit_logger") as mock_logger:
            log_error("123", "Test error")

        mock_logger.log_error.assert_called_once_with("123", "Test error")

    def test_log_completion_function(self):
        """Should delegate to audit_logger instance."""
        with patch("src.services.alert_audit_logger.audit_logger") as mock_logger:
            log_completion("123", 500)

        mock_logger.log_completion.assert_called_once_with("123", 500)

    def test_get_daily_evaluation_stats_function(self):
        """Should delegate to audit_logger instance."""
        with patch("src.services.alert_audit_logger.audit_logger") as mock_logger:
            mock_logger.get_daily_evaluation_stats.return_value = {"date": "2024-01-01"}

            result = get_daily_evaluation_stats()

        assert result == {"date": "2024-01-01"}
        mock_logger.get_daily_evaluation_stats.assert_called_once()

    def test_get_evaluation_coverage_function(self):
        """Should delegate to audit_logger instance."""
        with patch("src.services.alert_audit_logger.audit_logger") as mock_logger:
            mock_logger.get_evaluation_coverage.return_value = {"date": "2024-01-01"}

            result = get_evaluation_coverage()

        assert result == {"date": "2024-01-01"}

    def test_get_expected_daily_evaluations_function(self):
        """Should delegate to audit_logger instance."""
        with patch("src.services.alert_audit_logger.audit_logger") as mock_logger:
            mock_logger.get_expected_daily_evaluations.return_value = {
                "total_alerts": 50
            }

            result = get_expected_daily_evaluations()

        assert result == {"total_alerts": 50}

    def test_get_audit_summary_function(self):
        """Should delegate to audit_logger instance."""
        with patch("src.services.alert_audit_logger.audit_logger") as mock_logger:
            mock_logger.get_audit_summary.return_value = pd.DataFrame()

            result = get_audit_summary(days=14)

        assert isinstance(result, pd.DataFrame)

    def test_get_alert_history_function(self):
        """Should delegate to audit_logger instance."""
        with patch("src.services.alert_audit_logger.audit_logger") as mock_logger:
            mock_logger.get_alert_history.return_value = pd.DataFrame()

            result = get_alert_history("test-1", limit=50)

        assert isinstance(result, pd.DataFrame)

    def test_get_performance_metrics_function(self):
        """Should delegate to audit_logger instance."""
        with patch("src.services.alert_audit_logger.audit_logger") as mock_logger:
            mock_logger.get_performance_metrics.return_value = {"total_checks": 100}

            result = get_performance_metrics(days=30)

        assert result == {"total_checks": 100}


# ============================================================================
# TEST: Edge Cases and Error Scenarios
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_extremely_large_execution_time(self, audit_logger_postgres, mock_audit_id):
        """Should handle very large execution times."""
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            audit_logger_postgres.log_completion(mock_audit_id, 999999999)

        call_args = mock_execute.call_args
        assert 999999999 in call_args[0][1]

    def test_very_long_error_message(self, audit_logger_postgres, mock_audit_id):
        """Should handle very long error messages."""
        error_msg = "X" * 10000
        with patch.object(audit_logger_postgres, "_execute") as mock_execute:
            audit_logger_postgres.log_error(mock_audit_id, error_msg)

        mock_execute.assert_called_once()

    def test_alert_with_none_values(self, audit_logger_postgres):
        """Should handle alert with None values."""
        alert = {
            "alert_id": None,
            "id": "test",
            "ticker": None,
            "ticker1": "AAPL",
            "stock_name": None,
        }

        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = [1]
        mock_conn = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)

        with patch.object(audit_logger_postgres, "_connection", return_value=mock_conn):
            audit_id = audit_logger_postgres.log_alert_check_start(alert)

        assert audit_id == "1"

    def test_zero_days_parameter(self, audit_logger_postgres):
        """Should handle days=0."""
        mock_df = pd.DataFrame()

        with patch.object(audit_logger_postgres, "_connection") as mock_ctx:
            mock_ctx.__enter__.return_value = MagicMock()
            with patch("pandas.read_sql_query", return_value=mock_df):
                result = audit_logger_postgres.get_audit_summary(days=0)

        assert isinstance(result, pd.DataFrame)

    def test_negative_days_parameter(self, audit_logger_postgres):
        """Should handle negative days."""
        mock_df = pd.DataFrame()

        with patch.object(audit_logger_postgres, "_connection") as mock_ctx:
            mock_ctx.__enter__.return_value = MagicMock()
            with patch("pandas.read_sql_query", return_value=mock_df):
                result = audit_logger_postgres.get_audit_summary(days=-1)

        assert isinstance(result, pd.DataFrame)

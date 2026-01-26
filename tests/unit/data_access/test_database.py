"""
Unit tests for database configuration and connection management.

Tests the DatabaseConfig class and connection pooling logic.
"""

import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.stock_alert.data_access.database import DatabaseConfig, PostgresCursorProxy


class TestDatabaseConfig:
    """Test suite for DatabaseConfig class"""

    def test_database_config_initialization(self, monkeypatch):
        """Test that DatabaseConfig initializes with correct defaults"""
        monkeypatch.setenv("ENVIRONMENT", "development")
        config = DatabaseConfig()

        assert config.environment == "development"
        assert config.is_production is False
        assert config.db_type == "postgresql"
        assert config._pg_pool_min == 5
        assert config._pg_pool_max == 50

    def test_database_config_production_mode(self, monkeypatch):
        """Test that production environment is detected correctly"""
        monkeypatch.setenv("ENVIRONMENT", "production")
        config = DatabaseConfig()

        assert config.is_production is True
        assert config.environment == "production"

    def test_database_url_from_environment(self, monkeypatch):
        """Test that DATABASE_URL is loaded from environment"""
        test_url = "postgresql://testuser:testpass@localhost:5432/testdb"
        monkeypatch.setenv("DATABASE_URL", test_url)
        config = DatabaseConfig()

        assert config.database_url == test_url

    def test_pool_configuration_from_environment(self, monkeypatch):
        """Test that pool settings are loaded from environment"""
        monkeypatch.setenv("POSTGRES_POOL_MIN", "10")
        monkeypatch.setenv("POSTGRES_POOL_MAX", "100")
        monkeypatch.setenv("POSTGRES_CONN_LIMIT", "80")

        config = DatabaseConfig()

        assert config._pg_pool_min == 10
        assert config._pg_pool_max == 100

    @patch("psycopg2.pool.ThreadedConnectionPool")
    def test_get_postgresql_connection(self, mock_pool_class, monkeypatch):
        """Test that PostgreSQL connection is retrieved from pool"""
        # Setup
        mock_pool = Mock()
        mock_conn = Mock()
        mock_conn.autocommit = False
        mock_conn.cursor.return_value.__enter__ = Mock(return_value=Mock())
        mock_conn.cursor.return_value.__exit__ = Mock(return_value=None)
        mock_pool.getconn.return_value = mock_conn
        mock_pool_class.return_value = mock_pool

        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")
        config = DatabaseConfig()

        # Execute
        connection = config.get_postgresql_connection()

        # Verify
        assert connection is not None
        assert mock_pool.getconn.called
        assert config._pg_pool is not None

    def test_connection_context_manager(self, monkeypatch):
        """Test that connection context manager works correctly"""
        monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/test")
        config = DatabaseConfig()

        mock_conn = Mock()
        mock_conn.is_postgres = True

        with patch.object(config, "get_connection", return_value=mock_conn):
            with config.connection() as conn:
                assert conn is mock_conn

            # Verify connection was closed
            assert mock_conn.close.called


class TestPostgresCursorProxy:
    """Test suite for PostgresCursorProxy parameter translation"""

    def test_translate_question_marks_to_percent_s(self):
        """Test that ? placeholders are translated to %s"""
        mock_cursor = Mock()
        proxy = PostgresCursorProxy(mock_cursor)

        query = "SELECT * FROM table WHERE id = ? AND name = ?"
        params = (1, "test")

        proxy.execute(query, params)

        # Verify translation occurred
        called_query = mock_cursor.execute.call_args[0][0]
        assert "%s" in called_query
        assert "?" not in called_query

    def test_pass_through_percent_s_queries(self):
        """Test that queries already using %s are not modified"""
        mock_cursor = Mock()
        proxy = PostgresCursorProxy(mock_cursor)

        query = "SELECT * FROM table WHERE id = %s"
        params = (1,)

        proxy.execute(query, params)

        # Verify query was passed through unchanged
        called_query = mock_cursor.execute.call_args[0][0]
        assert called_query == query

    def test_handle_mapping_params(self):
        """Test that dictionary params are passed through"""
        mock_cursor = Mock()
        proxy = PostgresCursorProxy(mock_cursor)

        query = "SELECT * FROM table WHERE id = %(id)s"
        params = {"id": 1}

        proxy.execute(query, params)

        # Verify mapping params were passed through
        called_params = mock_cursor.execute.call_args[0][1]
        assert called_params == params

    def test_executemany_translation(self):
        """Test that executemany translates parameters correctly"""
        mock_cursor = Mock()
        proxy = PostgresCursorProxy(mock_cursor)

        query = "INSERT INTO table (id, name) VALUES (?, ?)"
        param_sets = [(1, "test1"), (2, "test2")]

        proxy.executemany(query, param_sets)

        # Verify translation occurred
        called_query = mock_cursor.executemany.call_args[0][0]
        assert "%s" in called_query
        assert "?" not in called_query


class TestDatabaseConfigGlobalInstance:
    """Test the global db_config instance"""

    def test_db_config_singleton_exists(self):
        """Test that db_config global instance is available"""
        from src.stock_alert.data_access.database import db_config

        assert db_config is not None
        assert isinstance(db_config, DatabaseConfig)

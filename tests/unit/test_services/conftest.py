"""
Pytest configuration for test_services.

This module provides shared fixtures and patches to prevent database
connections during test collection.
"""

import sys
import pytest
from unittest.mock import Mock, MagicMock, patch


def pytest_configure(config):
    """Configure pytest before test collection."""
    # Create mock modules to prevent import errors
    sys.modules["psycopg2"] = MagicMock()
    sys.modules["psycopg2.extras"] = MagicMock()
    sys.modules["psycopg2.pool"] = MagicMock()


@pytest.fixture(scope="session", autouse=True)
def mock_db_config_module():
    """Mock db_config module to prevent database connection on import."""
    mock_config = Mock()
    mock_config.db_type = "sqlite"
    mock_config.connection = MagicMock()
    mock_config.execute_with_retry = MagicMock()
    mock_config.get_connection = MagicMock()

    # Patch db_config in sys.modules
    mock_db_config_module_obj = MagicMock()
    mock_db_config_module_obj.db_config = mock_config
    sys.modules["src.data_access.db_config"] = mock_db_config_module_obj

    # Also patch the discord_routing and stock_alert_checker that cause issues
    sys.modules["src.services.discord_routing"] = MagicMock()
    sys.modules["src.services.stock_alert_checker"] = MagicMock()

    yield mock_config


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration between tests."""
    import logging
    # Clear all handlers
    logging.root.handlers = []
    yield
    logging.root.handlers = []

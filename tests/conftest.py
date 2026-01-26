"""
Shared pytest fixtures for the Stock Market Alert App test suite.

This module provides common fixtures used across unit and integration tests.
"""

import os
import sys
from collections.abc import Generator
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest
from faker import Faker

# Add src directory to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Set test environment variables before importing modules
os.environ.setdefault("FMP_API_KEY", "test_fmp_api_key_12345")
os.environ.setdefault("WEBHOOK_URL_LOGGING", "https://discord.com/api/webhooks/test/test_webhook_1")
os.environ.setdefault(
    "WEBHOOK_URL_LOGGING_2", "https://discord.com/api/webhooks/test/test_webhook_2"
)
os.environ.setdefault("ENVIRONMENT", "testing")


@pytest.fixture
def faker_instance() -> Faker:
    """Provide a Faker instance for generating test data"""
    return Faker()


@pytest.fixture
def sample_ticker() -> str:
    """Provide a sample ticker symbol"""
    return "AAPL"


@pytest.fixture
def sample_tickers() -> list[str]:
    """Provide a list of sample ticker symbols"""
    return ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]


@pytest.fixture
def sample_price_data() -> pd.DataFrame:
    """
    Provide sample OHLCV price data for testing

    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
    return pd.DataFrame(
        {
            "date": dates,
            "open": [100 + i * 0.5 for i in range(100)],
            "high": [102 + i * 0.5 for i in range(100)],
            "low": [99 + i * 0.5 for i in range(100)],
            "close": [101 + i * 0.5 for i in range(100)],
            "volume": [1000000 + i * 10000 for i in range(100)],
        }
    )


@pytest.fixture
def sample_alert_config() -> dict[str, Any]:
    """
    Provide a sample alert configuration

    Returns:
        Dictionary representing a basic alert configuration
    """
    return {
        "ticker": "AAPL",
        "stock_name": "Apple Inc.",
        "exchange": "NASDAQ",
        "condition": "close > 150",
        "notification_message": "AAPL price above $150",
        "enabled": True,
        "created_at": datetime.now().isoformat(),
    }


@pytest.fixture
def sample_alert_configs() -> list[dict[str, Any]]:
    """
    Provide multiple sample alert configurations

    Returns:
        List of alert configuration dictionaries
    """
    return [
        {
            "ticker": "AAPL",
            "condition": "close > 150",
            "enabled": True,
        },
        {
            "ticker": "MSFT",
            "condition": "rsi_14 < 30",
            "enabled": True,
        },
        {
            "ticker": "GOOGL",
            "condition": "volume > sma_volume_20 * 2",
            "enabled": False,
        },
    ]


@pytest.fixture
def mock_database_connection() -> Generator[Mock]:
    """
    Provide a mocked database connection

    Yields:
        Mock database connection object
    """
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    mock_cursor.__enter__.return_value = mock_cursor
    mock_cursor.__exit__.return_value = None

    yield mock_conn


@pytest.fixture
def mock_fmp_api() -> Mock:
    """
    Provide a mocked FMP API client

    Returns:
        Mock FMP API client with common methods
    """
    mock_api = Mock()

    # Mock successful price data response
    mock_api.get_historical_price_data.return_value = {
        "historical": [
            {
                "date": "2024-01-26",
                "open": 150.0,
                "high": 152.0,
                "low": 149.0,
                "close": 151.0,
                "volume": 50000000,
            }
        ]
    }

    # Mock stock profile response
    mock_api.get_stock_profile.return_value = {
        "symbol": "AAPL",
        "companyName": "Apple Inc.",
        "exchange": "NASDAQ",
        "sector": "Technology",
    }

    return mock_api


@pytest.fixture
def mock_discord_webhook() -> Mock:
    """
    Provide a mocked Discord webhook client

    Returns:
        Mock Discord webhook that simulates successful posts
    """
    mock_webhook = Mock()
    mock_webhook.post.return_value = Mock(status_code=200, ok=True)
    return mock_webhook


@pytest.fixture
def mock_redis_client() -> Mock:
    """
    Provide a mocked Redis client

    Returns:
        Mock Redis client with common cache operations
    """
    mock_redis = Mock()
    cache_store = {}

    def mock_get(key):
        return cache_store.get(key)

    def mock_set(key, value, ex=None):
        cache_store[key] = value
        return True

    def mock_delete(key):
        cache_store.pop(key, None)
        return True

    mock_redis.get.side_effect = mock_get
    mock_redis.set.side_effect = mock_set
    mock_redis.delete.side_effect = mock_delete

    return mock_redis


@pytest.fixture
def freeze_time_now():
    """
    Provide a frozen current datetime for testing

    Returns:
        Datetime object representing a fixed point in time
    """
    return datetime(2024, 1, 26, 12, 0, 0)


@pytest.fixture(autouse=True)
def reset_environment():
    """
    Automatically reset environment state between tests

    This fixture runs before and after each test to ensure clean state.
    """
    # Setup: Save original environment
    original_env = os.environ.copy()

    yield

    # Teardown: Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

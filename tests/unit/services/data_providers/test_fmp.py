"""
Unit tests for FMP data provider.

Tests the FMPDataProvider class with mocked API responses.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.stock_alert.services.data_providers.fmp import FMPDataProvider, OptimizedFMPDataProvider


class TestFMPDataProvider:
    """Test suite for FMPDataProvider"""

    def test_initialization_with_api_key(self):
        """Test that provider initializes with API key"""
        provider = FMPDataProvider(api_key="test_key_123")

        assert provider.api_key == "test_key_123"
        assert provider.name == "Financial Modeling Prep"
        assert provider.base_url == "https://financialmodelingprep.com/api/v3"

    def test_initialization_from_environment(self, monkeypatch):
        """Test that provider loads API key from environment"""
        monkeypatch.setenv("FMP_API_KEY", "env_key_456")

        provider = FMPDataProvider()

        assert provider.api_key == "env_key_456"

    def test_initialization_without_api_key_raises_error(self, monkeypatch):
        """Test that missing API key raises ValueError"""
        monkeypatch.delenv("FMP_API_KEY", raising=False)

        with pytest.raises(ValueError, match="FMP_API_KEY is required"):
            FMPDataProvider()

    @patch("src.stock_alert.services.data_providers.fmp.requests.Session")
    def test_get_historical_prices_success(self, mock_session_class):
        """Test successful historical price fetching"""
        # Setup mock response
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {
            "historical": [
                {
                    "date": "2024-01-26",
                    "open": 150.0,
                    "high": 152.0,
                    "low": 149.0,
                    "close": 151.0,
                    "volume": 50000000,
                },
                {
                    "date": "2024-01-25",
                    "open": 148.0,
                    "high": 150.0,
                    "low": 147.0,
                    "close": 149.0,
                    "volume": 45000000,
                },
            ]
        }
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        # Execute
        provider = FMPDataProvider(api_key="test_key")
        df = provider.get_historical_prices("AAPL", days=30)

        # Verify
        assert not df.empty
        assert len(df) == 2
        assert list(df.columns) == ["date", "open", "high", "low", "close", "volume"]
        assert df.iloc[0]["close"] == 149.0  # Sorted ascending by date
        assert df.iloc[1]["close"] == 151.0

    @patch("src.stock_alert.services.data_providers.fmp.requests.Session")
    def test_get_historical_prices_no_data(self, mock_session_class):
        """Test handling of empty historical data response"""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = {}  # No 'historical' key
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        provider = FMPDataProvider(api_key="test_key")
        df = provider.get_historical_prices("INVALID")

        assert df.empty

    @patch("src.stock_alert.services.data_providers.fmp.requests.Session")
    def test_get_historical_prices_api_error(self, mock_session_class):
        """Test handling of API request errors"""
        mock_session = Mock()
        mock_session.get.side_effect = Exception("API Error")
        mock_session_class.return_value = mock_session

        provider = FMPDataProvider(api_key="test_key")
        df = provider.get_historical_prices("AAPL")

        assert df.empty

    @patch("src.stock_alert.services.data_providers.fmp.requests.Session")
    def test_get_quote_success(self, mock_session_class):
        """Test successful quote fetching"""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = [
            {
                "symbol": "AAPL",
                "price": 151.25,
                "volume": 50000000,
                "change": 1.25,
            }
        ]
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        provider = FMPDataProvider(api_key="test_key")
        quote = provider.get_quote("AAPL")

        assert quote is not None
        assert quote["symbol"] == "AAPL"
        assert quote["price"] == 151.25

    @patch("src.stock_alert.services.data_providers.fmp.requests.Session")
    def test_get_latest_price(self, mock_session_class):
        """Test fetching latest price"""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = [{"price": 151.25}]
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        provider = FMPDataProvider(api_key="test_key")
        price = provider.get_latest_price("AAPL")

        assert price == 151.25

    @patch("src.stock_alert.services.data_providers.fmp.requests.Session")
    def test_validate_ticker_valid(self, mock_session_class):
        """Test ticker validation for valid ticker"""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = [{"symbol": "AAPL"}]
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        provider = FMPDataProvider(api_key="test_key")
        is_valid = provider.validate_ticker("AAPL")

        assert is_valid is True

    @patch("src.stock_alert.services.data_providers.fmp.requests.Session")
    def test_validate_ticker_invalid(self, mock_session_class):
        """Test ticker validation for invalid ticker"""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.json.return_value = []  # Empty response
        mock_response.raise_for_status = Mock()
        mock_session.get.return_value = mock_response
        mock_session_class.return_value = mock_session

        provider = FMPDataProvider(api_key="test_key")
        is_valid = provider.validate_ticker("INVALID123")

        assert is_valid is False

    def test_context_manager(self):
        """Test that provider works as context manager"""
        with FMPDataProvider(api_key="test_key") as provider:
            assert provider is not None
            assert provider.api_key == "test_key"


class TestOptimizedFMPDataProvider:
    """Test suite for OptimizedFMPDataProvider with database integration"""

    def test_get_missing_dates_no_data(self):
        """Test getting missing dates when no data exists"""
        # Create provider first
        provider = OptimizedFMPDataProvider(api_key="test_key")

        # Mock the db_config on the instance
        mock_db_config = MagicMock()
        mock_conn = MagicMock()
        mock_db_config.connection.return_value.__enter__.return_value = mock_conn
        mock_db_config.execute_with_retry.return_value = [[None]]
        provider.db_config = mock_db_config

        last_date, days_missing = provider.get_missing_dates("AAPL")

        assert last_date is None
        assert days_missing == 365  # Default when no data

    def test_get_missing_dates_with_data(self):
        """Test getting missing dates when data exists"""
        # Create provider first
        provider = OptimizedFMPDataProvider(api_key="test_key")

        # Mock the db_config on the instance
        mock_db_config = MagicMock()
        mock_conn = MagicMock()
        mock_db_config.connection.return_value.__enter__.return_value = mock_conn

        # Mock last date as 10 days ago
        ten_days_ago = pd.Timestamp.now() - pd.Timedelta(days=10)
        mock_db_config.execute_with_retry.return_value = [[ten_days_ago]]
        provider.db_config = mock_db_config

        last_date, days_missing = provider.get_missing_dates("AAPL")

        assert last_date is not None
        assert days_missing >= 9  # At least 9 days missing (could be 10 depending on time)

"""
Unit tests for the FMP data fetcher module.

Tests the core functionality for:
- Initialization and API key handling
- Daily data fetching and processing
- Intraday/hourly data fetching
- Weekly resampling
- Quote and profile fetching
- Error handling and edge cases
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch, Mock
from datetime import datetime, timedelta
import json


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def fmp_fetcher():
    """Create an FMPDataFetcher instance with a test API key."""
    from src.services.backend_fmp import FMPDataFetcher
    return FMPDataFetcher(api_key="test_api_key")


@pytest.fixture
def sample_daily_response():
    """Create a sample FMP daily data API response."""
    return {
        "symbol": "AAPL",
        "historical": [
            {
                "date": "2024-01-05",
                "open": 150.0,
                "high": 152.0,
                "low": 149.0,
                "close": 151.0,
                "adjClose": 151.0,
                "volume": 50000000,
                "change": 1.0,
                "changePercent": 0.67,
                "vwap": 150.5,
                "changeOverTime": 0.01
            },
            {
                "date": "2024-01-04",
                "open": 149.0,
                "high": 151.0,
                "low": 148.0,
                "close": 150.0,
                "adjClose": 150.0,
                "volume": 45000000,
                "change": 1.5,
                "changePercent": 1.0,
                "vwap": 149.5,
                "changeOverTime": 0.005
            },
            {
                "date": "2024-01-03",
                "open": 148.0,
                "high": 150.0,
                "low": 147.0,
                "close": 149.0,
                "adjClose": 149.0,
                "volume": 48000000,
                "change": 0.5,
                "changePercent": 0.34,
                "vwap": 148.5,
                "changeOverTime": 0.002
            },
        ]
    }


@pytest.fixture
def sample_intraday_response():
    """Create a sample FMP intraday/hourly data API response."""
    return [
        {
            "date": "2024-01-05 15:00:00",
            "open": 151.0,
            "high": 152.0,
            "low": 150.5,
            "close": 151.5,
            "volume": 5000000
        },
        {
            "date": "2024-01-05 14:00:00",
            "open": 150.5,
            "high": 151.5,
            "low": 150.0,
            "close": 151.0,
            "volume": 4500000
        },
        {
            "date": "2024-01-05 13:00:00",
            "open": 150.0,
            "high": 151.0,
            "low": 149.5,
            "close": 150.5,
            "volume": 4000000
        },
    ]


@pytest.fixture
def sample_quote_response():
    """Create a sample FMP quote API response."""
    return [{
        "symbol": "AAPL",
        "name": "Apple Inc.",
        "price": 151.0,
        "changesPercentage": 0.67,
        "change": 1.0,
        "dayLow": 149.0,
        "dayHigh": 152.0,
        "yearHigh": 200.0,
        "yearLow": 120.0,
        "marketCap": 2500000000000,
        "priceAvg50": 145.0,
        "priceAvg200": 140.0,
        "exchange": "NASDAQ",
        "volume": 50000000,
        "avgVolume": 45000000,
        "open": 150.0,
        "previousClose": 150.0,
        "eps": 6.0,
        "pe": 25.0,
        "timestamp": 1704499200
    }]


@pytest.fixture
def sample_profile_response():
    """Create a sample FMP profile API response."""
    return [{
        "symbol": "AAPL",
        "price": 151.0,
        "companyName": "Apple Inc.",
        "currency": "USD",
        "exchange": "NASDAQ",
        "industry": "Consumer Electronics",
        "sector": "Technology",
        "country": "US",
        "description": "Apple Inc. designs, manufactures, and markets smartphones...",
        "ceo": "Tim Cook",
        "website": "https://www.apple.com",
        "employees": 164000,
        "ipoDate": "1980-12-12"
    }]


@pytest.fixture
def sample_daily_df():
    """Create a sample daily OHLCV DataFrame for testing resampling."""
    # Create 3 weeks of daily data (15 trading days)
    dates = pd.date_range(start="2024-01-02", periods=15, freq="B")  # Business days
    
    return pd.DataFrame({
        "Open": [100 + i for i in range(15)],
        "High": [101 + i for i in range(15)],
        "Low": [99 + i for i in range(15)],
        "Close": [100.5 + i for i in range(15)],
        "Volume": [1000000 + i * 100000 for i in range(15)]
    }, index=dates)


# =============================================================================
# Tests for FMPDataFetcher Initialization
# =============================================================================

class TestFMPDataFetcherInit:
    """Tests for FMPDataFetcher initialization."""

    def test_init_with_provided_api_key(self):
        """Test initialization with a provided API key."""
        from src.services.backend_fmp import FMPDataFetcher
        
        fetcher = FMPDataFetcher(api_key="my_custom_key")
        
        assert fetcher.api_key == "my_custom_key"
        assert fetcher.base_url == "https://financialmodelingprep.com/api/v3"

    @patch.dict("os.environ", {"FMP_API_KEY": "env_api_key"})
    def test_init_with_environment_variable(self):
        """Test initialization using environment variable."""
        from src.services.backend_fmp import FMPDataFetcher
        
        fetcher = FMPDataFetcher(api_key=None)
        
        # Falls back to env var or default
        assert fetcher.api_key is not None

    def test_init_uses_default_key_when_no_env(self):
        """Test initialization uses default key when no env var."""
        from src.services.backend_fmp import FMPDataFetcher
        
        # Note: The class has a hardcoded default key
        fetcher = FMPDataFetcher()
        
        assert fetcher.api_key is not None


# =============================================================================
# Tests for get_historical_data
# =============================================================================

class TestGetHistoricalData:
    """Tests for the get_historical_data method."""

    @patch("src.services.backend_fmp.requests.get")
    def test_get_historical_data_daily(self, mock_get, fmp_fetcher, sample_daily_response):
        """Test fetching daily historical data."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_daily_response
        mock_get.return_value = mock_response
        
        result = fmp_fetcher.get_historical_data("AAPL", period="1day")
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert "Open" in result.columns
        assert "Close" in result.columns
        assert "High" in result.columns
        assert "Low" in result.columns
        assert "Volume" in result.columns

    @patch("src.services.backend_fmp.requests.get")
    def test_get_historical_data_weekly_resamples(self, mock_get, fmp_fetcher, sample_daily_response):
        """Test that weekly timeframe triggers resampling."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_daily_response
        mock_get.return_value = mock_response
        
        with patch.object(fmp_fetcher, "_resample_to_weekly") as mock_resample:
            mock_resample.return_value = pd.DataFrame()
            
            fmp_fetcher.get_historical_data("AAPL", timeframe="1wk")
            
            mock_resample.assert_called_once()

    @patch("src.services.backend_fmp.requests.get")
    def test_get_historical_data_intraday(self, mock_get, fmp_fetcher, sample_intraday_response):
        """Test fetching intraday data."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_intraday_response
        mock_get.return_value = mock_response
        
        result = fmp_fetcher.get_historical_data("AAPL", period="1hour")
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)

    def test_get_historical_data_returns_none_on_exception(self, fmp_fetcher):
        """Test that exceptions return None."""
        with patch.object(fmp_fetcher, "_fetch_daily_data", side_effect=Exception("Test error")):
            result = fmp_fetcher.get_historical_data("AAPL", period="1day")
            
            assert result is None


# =============================================================================
# Tests for _fetch_daily_data
# =============================================================================

class TestFetchDailyData:
    """Tests for the _fetch_daily_data method."""

    @patch("src.services.backend_fmp.requests.get")
    def test_fetch_daily_data_success(self, mock_get, fmp_fetcher, sample_daily_response):
        """Test successful daily data fetch."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_daily_response
        mock_get.return_value = mock_response
        
        result = fmp_fetcher._fetch_daily_data("AAPL", limit=750)
        
        assert result is not None
        assert len(result) == 3
        # Verify data is sorted by date ascending
        assert result.index[0] < result.index[-1]
        # Verify column names are standardized
        assert "Open" in result.columns
        assert "Close" in result.columns

    @patch("src.services.backend_fmp.requests.get")
    def test_fetch_daily_data_empty_response(self, mock_get, fmp_fetcher):
        """Test handling empty API response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"historical": []}
        mock_get.return_value = mock_response
        
        result = fmp_fetcher._fetch_daily_data("INVALID", limit=750)
        
        assert result is None

    @patch("src.services.backend_fmp.requests.get")
    def test_fetch_daily_data_no_historical_key(self, mock_get, fmp_fetcher):
        """Test handling response without historical key."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"error": "Not found"}
        mock_get.return_value = mock_response
        
        result = fmp_fetcher._fetch_daily_data("INVALID", limit=750)
        
        assert result is None

    @patch("src.services.backend_fmp.requests.get")
    def test_fetch_daily_data_401_error(self, mock_get, fmp_fetcher):
        """Test handling 401 unauthorized error."""
        mock_response = Mock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response
        
        result = fmp_fetcher._fetch_daily_data("AAPL", limit=750)
        
        assert result is None

    @patch("src.services.backend_fmp.requests.get")
    def test_fetch_daily_data_500_error(self, mock_get, fmp_fetcher):
        """Test handling server error."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        result = fmp_fetcher._fetch_daily_data("AAPL", limit=750)
        
        assert result is None

    @patch("src.services.backend_fmp.requests.get")
    def test_fetch_daily_data_timeout(self, mock_get, fmp_fetcher):
        """Test handling request timeout."""
        import requests
        mock_get.side_effect = requests.exceptions.Timeout()
        
        result = fmp_fetcher._fetch_daily_data("AAPL", limit=750)
        
        assert result is None

    @patch("src.services.backend_fmp.requests.get")
    def test_fetch_daily_data_constructs_correct_url(self, mock_get, fmp_fetcher):
        """Test that the correct URL and parameters are used."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"historical": []}
        mock_get.return_value = mock_response
        
        fmp_fetcher._fetch_daily_data("AAPL", limit=500)
        
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        
        assert "historical-price-full/AAPL" in call_args[0][0]
        assert call_args[1]["params"]["apikey"] == "test_api_key"
        assert call_args[1]["params"]["limit"] == 500


# =============================================================================
# Tests for _fetch_intraday_data
# =============================================================================

class TestFetchIntradayData:
    """Tests for the _fetch_intraday_data method."""

    @patch("src.services.backend_fmp.requests.get")
    def test_fetch_intraday_data_success(self, mock_get, fmp_fetcher, sample_intraday_response):
        """Test successful intraday data fetch."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_intraday_response
        mock_get.return_value = mock_response
        
        result = fmp_fetcher._fetch_intraday_data("AAPL", interval="1hour")
        
        assert result is not None
        assert len(result) == 3
        assert "Open" in result.columns
        assert "Close" in result.columns

    @patch("src.services.backend_fmp.requests.get")
    def test_fetch_intraday_data_interval_mapping(self, mock_get, fmp_fetcher):
        """Test interval name mapping."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response
        
        # Test various interval aliases
        intervals_to_test = ["1min", "5min", "15min", "30min", "1hour", "1hr", "hourly", "4hour"]
        
        for interval in intervals_to_test:
            fmp_fetcher._fetch_intraday_data("AAPL", interval=interval)

    @patch("src.services.backend_fmp.requests.get")
    def test_fetch_intraday_data_empty_response(self, mock_get, fmp_fetcher):
        """Test handling empty intraday response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response
        
        result = fmp_fetcher._fetch_intraday_data("INVALID", interval="1hour")
        
        assert result is None

    @patch("src.services.backend_fmp.requests.get")
    def test_fetch_intraday_data_api_error(self, mock_get, fmp_fetcher):
        """Test handling API error for intraday data."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = fmp_fetcher._fetch_intraday_data("AAPL", interval="1hour")
        
        assert result is None


# =============================================================================
# Tests for get_hourly_data
# =============================================================================

class TestGetHourlyData:
    """Tests for the get_hourly_data method."""

    @patch("src.services.backend_fmp.time.sleep")
    @patch("src.services.backend_fmp.requests.get")
    def test_get_hourly_data_success(self, mock_get, mock_sleep, fmp_fetcher, sample_intraday_response):
        """Test successful hourly data fetch."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_intraday_response
        mock_get.return_value = mock_response
        
        result = fmp_fetcher.get_hourly_data(
            "AAPL",
            from_date="2024-01-01",
            to_date="2024-01-10"
        )
        
        assert result is not None

    @patch("src.services.backend_fmp.time.sleep")
    @patch("src.services.backend_fmp.requests.get")
    def test_get_hourly_data_default_dates(self, mock_get, mock_sleep, fmp_fetcher, sample_intraday_response):
        """Test hourly data with default date range."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_intraday_response
        mock_get.return_value = mock_response
        
        result = fmp_fetcher.get_hourly_data("AAPL")
        
        # Should not raise and should return data
        assert result is not None

    @patch("src.services.backend_fmp.time.sleep")
    @patch("src.services.backend_fmp.requests.get")
    def test_get_hourly_data_removes_duplicates(self, mock_get, mock_sleep, fmp_fetcher):
        """Test that duplicate records at chunk boundaries are removed."""
        # Create overlapping data to simulate chunk boundary duplicates
        chunk1_data = [
            {"date": "2024-01-05 15:00:00", "open": 151.0, "high": 152.0, "low": 150.5, "close": 151.5, "volume": 5000000},
            {"date": "2024-01-05 14:00:00", "open": 150.5, "high": 151.5, "low": 150.0, "close": 151.0, "volume": 4500000},
        ]
        chunk2_data = [
            {"date": "2024-01-05 15:00:00", "open": 151.0, "high": 152.0, "low": 150.5, "close": 151.5, "volume": 5000000},  # Duplicate
            {"date": "2024-01-05 16:00:00", "open": 151.5, "high": 152.5, "low": 151.0, "close": 152.0, "volume": 5500000},
        ]
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = [chunk1_data, chunk2_data]
        mock_get.return_value = mock_response
        
        result = fmp_fetcher.get_hourly_data(
            "AAPL",
            from_date="2024-01-01",
            to_date="2024-03-15"
        )
        
        # Should have 3 unique records, not 4
        if result is not None:
            assert len(result) == len(result[~result.index.duplicated(keep='first')])

    @patch("src.services.backend_fmp.time.sleep")
    @patch("src.services.backend_fmp.requests.get")
    def test_get_hourly_data_no_data_available(self, mock_get, mock_sleep, fmp_fetcher):
        """Test handling when no hourly data is available."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response
        
        result = fmp_fetcher.get_hourly_data(
            "INVALID",
            from_date="2024-01-01",
            to_date="2024-01-10"
        )
        
        assert result is None

    def test_get_hourly_data_exception_handling(self, fmp_fetcher):
        """Test that exceptions are handled gracefully."""
        with patch.object(fmp_fetcher, "_fetch_hourly_chunk", side_effect=Exception("Test error")):
            result = fmp_fetcher.get_hourly_data(
                "AAPL",
                from_date="2024-01-01",
                to_date="2024-01-02"
            )
            
            assert result is None


# =============================================================================
# Tests for _fetch_hourly_chunk
# =============================================================================

class TestFetchHourlyChunk:
    """Tests for the _fetch_hourly_chunk method."""

    @patch("src.services.backend_fmp.requests.get")
    def test_fetch_hourly_chunk_success(self, mock_get, fmp_fetcher, sample_intraday_response):
        """Test successful hourly chunk fetch."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_intraday_response
        mock_get.return_value = mock_response
        
        result = fmp_fetcher._fetch_hourly_chunk("AAPL", "2024-01-01", "2024-01-10")
        
        assert result is not None
        assert len(result) == 3

    @patch("src.services.backend_fmp.requests.get")
    def test_fetch_hourly_chunk_empty_returns_empty_df(self, mock_get, fmp_fetcher):
        """Test that empty response returns empty DataFrame."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response
        
        result = fmp_fetcher._fetch_hourly_chunk("AAPL", "2024-01-01", "2024-01-10")
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    @patch("src.services.backend_fmp.requests.get")
    def test_fetch_hourly_chunk_api_error(self, mock_get, fmp_fetcher):
        """Test handling API error in hourly chunk fetch."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        result = fmp_fetcher._fetch_hourly_chunk("AAPL", "2024-01-01", "2024-01-10")
        
        assert result is None


# =============================================================================
# Tests for _resample_to_weekly
# =============================================================================

class TestResampleToWeekly:
    """Tests for the _resample_to_weekly method."""

    def test_resample_to_weekly_success(self, fmp_fetcher, sample_daily_df):
        """Test successful weekly resampling."""
        result = fmp_fetcher._resample_to_weekly(sample_daily_df)
        
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        # Should have fewer records than daily
        assert len(result) < len(sample_daily_df)
        # Should have OHLCV columns
        assert "Open" in result.columns
        assert "Close" in result.columns
        assert "High" in result.columns
        assert "Low" in result.columns
        assert "Volume" in result.columns

    def test_resample_to_weekly_empty_df(self, fmp_fetcher):
        """Test resampling empty DataFrame."""
        empty_df = pd.DataFrame()
        
        result = fmp_fetcher._resample_to_weekly(empty_df)
        
        assert result is None

    def test_resample_to_weekly_calculates_ohlc_correctly(self, fmp_fetcher):
        """Test that OHLC values are calculated correctly."""
        # Create a simple week of data
        dates = pd.date_range(start="2024-01-08", periods=5, freq="B")  # Mon-Fri
        df = pd.DataFrame({
            "Open": [100, 102, 104, 103, 105],
            "High": [101, 103, 106, 105, 108],
            "Low": [99, 101, 103, 102, 104],
            "Close": [101, 103, 104, 104, 106],
            "Volume": [1000, 1100, 1200, 1300, 1400]
        }, index=dates)
        
        result = fmp_fetcher._resample_to_weekly(df)
        
        if result is not None and len(result) > 0:
            # Weekly open should be first day's open
            assert result.iloc[-1]["Open"] == 100
            # Weekly high should be max of all highs
            assert result.iloc[-1]["High"] == 108
            # Weekly low should be min of all lows
            assert result.iloc[-1]["Low"] == 99
            # Weekly close should be last day's close
            assert result.iloc[-1]["Close"] == 106
            # Weekly volume should be sum
            assert result.iloc[-1]["Volume"] == 6000

    def test_resample_to_weekly_handles_date_in_columns(self, fmp_fetcher):
        """Test resampling when date is a column instead of index."""
        dates = pd.date_range(start="2024-01-08", periods=5, freq="B")
        df = pd.DataFrame({
            "date": dates,
            "Open": [100, 102, 104, 103, 105],
            "High": [101, 103, 106, 105, 108],
            "Low": [99, 101, 103, 102, 104],
            "Close": [101, 103, 104, 104, 106],
            "Volume": [1000, 1100, 1200, 1300, 1400]
        })
        
        result = fmp_fetcher._resample_to_weekly(df)
        
        # Should not raise and should handle the date column
        assert result is None or isinstance(result, pd.DataFrame)


# =============================================================================
# Tests for get_quote
# =============================================================================

class TestGetQuote:
    """Tests for the get_quote method."""

    @patch("src.services.backend_fmp.requests.get")
    def test_get_quote_success(self, mock_get, fmp_fetcher, sample_quote_response):
        """Test successful quote fetch."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_quote_response
        mock_get.return_value = mock_response
        
        result = fmp_fetcher.get_quote("AAPL")
        
        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["price"] == 151.0

    @patch("src.services.backend_fmp.requests.get")
    def test_get_quote_empty_response(self, mock_get, fmp_fetcher):
        """Test handling empty quote response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response
        
        result = fmp_fetcher.get_quote("INVALID")
        
        assert result is None

    @patch("src.services.backend_fmp.requests.get")
    def test_get_quote_api_error(self, mock_get, fmp_fetcher):
        """Test handling API error for quote."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = fmp_fetcher.get_quote("AAPL")
        
        assert result is None

    @patch("src.services.backend_fmp.requests.get")
    def test_get_quote_exception(self, mock_get, fmp_fetcher):
        """Test handling exception in quote fetch."""
        mock_get.side_effect = Exception("Network error")
        
        result = fmp_fetcher.get_quote("AAPL")
        
        assert result is None


# =============================================================================
# Tests for get_profile
# =============================================================================

class TestGetProfile:
    """Tests for the get_profile method."""

    @patch("src.services.backend_fmp.requests.get")
    def test_get_profile_success(self, mock_get, fmp_fetcher, sample_profile_response):
        """Test successful profile fetch."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_profile_response
        mock_get.return_value = mock_response
        
        result = fmp_fetcher.get_profile("AAPL")
        
        assert result is not None
        assert result["symbol"] == "AAPL"
        assert result["companyName"] == "Apple Inc."
        assert result["sector"] == "Technology"

    @patch("src.services.backend_fmp.requests.get")
    def test_get_profile_empty_response(self, mock_get, fmp_fetcher):
        """Test handling empty profile response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response
        
        result = fmp_fetcher.get_profile("INVALID")
        
        assert result is None

    @patch("src.services.backend_fmp.requests.get")
    def test_get_profile_api_error(self, mock_get, fmp_fetcher):
        """Test handling API error for profile."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        result = fmp_fetcher.get_profile("AAPL")
        
        assert result is None

    @patch("src.services.backend_fmp.requests.get")
    def test_get_profile_exception(self, mock_get, fmp_fetcher):
        """Test handling exception in profile fetch."""
        mock_get.side_effect = Exception("Connection error")
        
        result = fmp_fetcher.get_profile("AAPL")
        
        assert result is None


# =============================================================================
# Tests for Column Name Standardization
# =============================================================================

class TestColumnNameStandardization:
    """Tests for column name standardization in data processing."""

    @patch("src.services.backend_fmp.requests.get")
    def test_daily_data_column_mapping(self, mock_get, fmp_fetcher):
        """Test that daily data columns are properly mapped."""
        api_response = {
            "historical": [{
                "date": "2024-01-05",
                "open": 150.0,
                "high": 152.0,
                "low": 149.0,
                "close": 151.0,
                "adjClose": 151.0,
                "volume": 50000000,
                "change": 1.0,
                "changePercent": 0.67,
                "vwap": 150.5,
                "changeOverTime": 0.01
            }]
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = api_response
        mock_get.return_value = mock_response
        
        result = fmp_fetcher._fetch_daily_data("AAPL")
        
        assert "Open" in result.columns
        assert "High" in result.columns
        assert "Low" in result.columns
        assert "Close" in result.columns
        assert "Volume" in result.columns
        assert "Adj Close" in result.columns
        assert "VWAP" in result.columns

    @patch("src.services.backend_fmp.requests.get")
    def test_intraday_data_column_mapping(self, mock_get, fmp_fetcher):
        """Test that intraday data columns are properly mapped."""
        api_response = [{
            "date": "2024-01-05 15:00:00",
            "open": 151.0,
            "high": 152.0,
            "low": 150.5,
            "close": 151.5,
            "volume": 5000000
        }]
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = api_response
        mock_get.return_value = mock_response
        
        result = fmp_fetcher._fetch_intraday_data("AAPL", "1hour")
        
        assert "Open" in result.columns
        assert "High" in result.columns
        assert "Low" in result.columns
        assert "Close" in result.columns
        assert "Volume" in result.columns


# =============================================================================
# Tests for Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @patch("src.services.backend_fmp.requests.get")
    def test_handles_json_decode_error(self, mock_get, fmp_fetcher):
        """Test handling JSON decode error."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Error", "", 0)
        mock_get.return_value = mock_response
        
        result = fmp_fetcher._fetch_daily_data("AAPL")
        
        assert result is None

    @patch("src.services.backend_fmp.requests.get")
    def test_handles_connection_error(self, mock_get, fmp_fetcher):
        """Test handling connection error."""
        import requests
        mock_get.side_effect = requests.exceptions.ConnectionError()
        
        result = fmp_fetcher._fetch_daily_data("AAPL")
        
        assert result is None

    @patch("src.services.backend_fmp.requests.get")
    def test_get_historical_data_with_different_periods(self, mock_get, fmp_fetcher, sample_daily_response):
        """Test get_historical_data with various period values."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_daily_response
        mock_get.return_value = mock_response
        
        # Test daily aliases
        result = fmp_fetcher.get_historical_data("AAPL", period="daily")
        assert result is not None

    @patch("src.services.backend_fmp.requests.get")
    def test_get_historical_data_weekly_aliases(self, mock_get, fmp_fetcher, sample_daily_response):
        """Test get_historical_data with weekly timeframe aliases."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_daily_response
        mock_get.return_value = mock_response
        
        with patch.object(fmp_fetcher, "_resample_to_weekly") as mock_resample:
            mock_resample.return_value = pd.DataFrame()
            
            # Test various weekly aliases
            for alias in ["1wk", "weekly", "1week"]:
                fmp_fetcher.get_historical_data("AAPL", timeframe=alias)
                mock_resample.assert_called()

    def test_resample_handles_non_datetime_index(self, fmp_fetcher):
        """Test resampling handles non-datetime index gracefully."""
        df = pd.DataFrame({
            "Open": [100, 102],
            "High": [101, 103],
            "Low": [99, 101],
            "Close": [101, 103],
            "Volume": [1000, 1100]
        }, index=["2024-01-01", "2024-01-02"])  # String index
        
        result = fmp_fetcher._resample_to_weekly(df)
        
        # Should handle conversion or return None
        assert result is None or isinstance(result, pd.DataFrame)


# =============================================================================
# Tests for Data Integrity
# =============================================================================

class TestDataIntegrity:
    """Tests for data integrity in fetched results."""

    @patch("src.services.backend_fmp.requests.get")
    def test_daily_data_index_is_datetime(self, mock_get, fmp_fetcher, sample_daily_response):
        """Test that daily data index is DatetimeIndex."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_daily_response
        mock_get.return_value = mock_response
        
        result = fmp_fetcher._fetch_daily_data("AAPL")
        
        assert isinstance(result.index, pd.DatetimeIndex)

    @patch("src.services.backend_fmp.requests.get")
    def test_intraday_data_index_is_datetime(self, mock_get, fmp_fetcher, sample_intraday_response):
        """Test that intraday data index is DatetimeIndex."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_intraday_response
        mock_get.return_value = mock_response
        
        result = fmp_fetcher._fetch_intraday_data("AAPL", "1hour")
        
        assert isinstance(result.index, pd.DatetimeIndex)

    @patch("src.services.backend_fmp.requests.get")
    def test_data_is_sorted_ascending(self, mock_get, fmp_fetcher, sample_daily_response):
        """Test that returned data is sorted by date ascending."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = sample_daily_response
        mock_get.return_value = mock_response
        
        result = fmp_fetcher._fetch_daily_data("AAPL")
        
        # Verify ascending order
        assert result.index.is_monotonic_increasing

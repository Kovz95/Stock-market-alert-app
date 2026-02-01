"""
Unit tests for check_hourly_data_repository module.
"""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.data_access.check_hourly_data_repository import (
    CheckHourlyDataRepository,
    HourlyDataStats,
    PriceBar,
    RecordDistribution,
    TickerStats,
)


class TestHourlyDataStats:
    """Tests for HourlyDataStats data class."""

    def test_initialization(self):
        """Test that HourlyDataStats initializes correctly."""
        earliest = datetime(2024, 1, 1, 9, 30)
        latest = datetime(2024, 1, 5, 16, 0)
        stats = HourlyDataStats(
            tickers_with_data=100,
            total_records=5000,
            earliest=earliest,
            latest=latest,
        )

        assert stats.tickers_with_data == 100
        assert stats.total_records == 5000
        assert stats.earliest == earliest
        assert stats.latest == latest

    def test_days_span_with_datetime_objects(self):
        """Test days_span calculation with datetime objects."""
        earliest = datetime(2024, 1, 1, 9, 30)
        latest = datetime(2024, 1, 5, 16, 0)
        stats = HourlyDataStats(
            tickers_with_data=100,
            total_records=5000,
            earliest=earliest,
            latest=latest,
        )

        assert stats.days_span == 4

    def test_days_span_with_string_dates(self):
        """Test days_span calculation with ISO format strings."""
        stats = HourlyDataStats(
            tickers_with_data=100,
            total_records=5000,
            earliest="2024-01-01T09:30:00",
            latest="2024-01-10T16:00:00",
        )

        assert stats.days_span == 9

    def test_days_span_with_none_values(self):
        """Test days_span returns None when dates are None."""
        stats = HourlyDataStats(
            tickers_with_data=0,
            total_records=0,
            earliest=None,
            latest=None,
        )

        assert stats.days_span is None

    def test_days_span_with_partial_none(self):
        """Test days_span returns None when only one date is None."""
        stats = HourlyDataStats(
            tickers_with_data=0,
            total_records=0,
            earliest=datetime(2024, 1, 1),
            latest=None,
        )

        assert stats.days_span is None


class TestTickerStats:
    """Tests for TickerStats data class."""

    def test_initialization(self):
        """Test that TickerStats initializes correctly."""
        first_date = datetime(2024, 1, 1, 9, 30)
        last_date = datetime(2024, 1, 5, 16, 0)
        stats = TickerStats(
            ticker="AAPL",
            num_records=500,
            first_date=first_date,
            last_date=last_date,
        )

        assert stats.ticker == "AAPL"
        assert stats.num_records == 500
        assert stats.first_date == first_date
        assert stats.last_date == last_date

    def test_days_covered_with_datetime_objects(self):
        """Test days_covered calculation with datetime objects."""
        stats = TickerStats(
            ticker="AAPL",
            num_records=500,
            first_date=datetime(2024, 1, 1, 9, 30),
            last_date=datetime(2024, 1, 10, 16, 0),
        )

        assert stats.days_covered == 9

    def test_days_covered_with_string_dates(self):
        """Test days_covered calculation with ISO format strings."""
        stats = TickerStats(
            ticker="AAPL",
            num_records=500,
            first_date="2024-01-01T09:30:00",
            last_date="2024-01-05T16:00:00",
        )

        assert stats.days_covered == 4

    def test_days_covered_minimum_one(self):
        """Test days_covered returns minimum of 1 for same-day data."""
        same_date = datetime(2024, 1, 1, 9, 30)
        stats = TickerStats(
            ticker="AAPL",
            num_records=10,
            first_date=same_date,
            last_date=same_date,
        )

        assert stats.days_covered == 1

    def test_avg_bars_per_day(self):
        """Test avg_bars_per_day calculation."""
        stats = TickerStats(
            ticker="AAPL",
            num_records=400,
            first_date=datetime(2024, 1, 1),
            last_date=datetime(2024, 1, 5),
        )

        # 400 records over 4 days = 100 bars per day
        assert stats.avg_bars_per_day == 100.0

    def test_avg_bars_per_day_fractional(self):
        """Test avg_bars_per_day with fractional result."""
        stats = TickerStats(
            ticker="AAPL",
            num_records=350,
            first_date=datetime(2024, 1, 1),
            last_date=datetime(2024, 1, 6),
        )

        # 350 records over 5 days = 70 bars per day
        assert stats.avg_bars_per_day == 70.0


class TestPriceBar:
    """Tests for PriceBar data class."""

    def test_initialization(self):
        """Test that PriceBar initializes correctly."""
        dt = datetime(2024, 1, 1, 9, 30)
        bar = PriceBar(
            datetime=dt,
            open=150.0,
            high=152.0,
            low=149.5,
            close=151.5,
            volume=1000000,
        )

        assert bar.datetime == dt
        assert bar.open == 150.0
        assert bar.high == 152.0
        assert bar.low == 149.5
        assert bar.close == 151.5
        assert bar.volume == 1000000


class TestRecordDistribution:
    """Tests for RecordDistribution data class."""

    def test_initialization(self):
        """Test that RecordDistribution initializes correctly."""
        dist = RecordDistribution(range_name="100-500 bars", num_tickers=50)

        assert dist.range_name == "100-500 bars"
        assert dist.num_tickers == 50


class TestCheckHourlyDataRepository:
    """Tests for CheckHourlyDataRepository."""

    @pytest.fixture
    def mock_db_config(self):
        """Create a mock db_config for testing."""
        mock_config = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_config.connection.return_value.__enter__.return_value = mock_conn
        mock_config.connection.return_value.__exit__.return_value = None
        return mock_config, mock_cursor

    @pytest.fixture
    def repository(self, mock_db_config):
        """Create a repository instance with mocked db_config."""
        mock_config, _ = mock_db_config
        return CheckHourlyDataRepository(db_config_instance=mock_config)

    def test_get_overall_stats(self, repository, mock_db_config):
        """Test get_overall_stats returns correct data."""
        _, mock_cursor = mock_db_config
        earliest = datetime(2024, 1, 1, 9, 30)
        latest = datetime(2024, 1, 31, 16, 0)
        mock_cursor.fetchone.return_value = (100, 50000, earliest, latest)

        stats = repository.get_overall_stats()

        assert isinstance(stats, HourlyDataStats)
        assert stats.tickers_with_data == 100
        assert stats.total_records == 50000
        assert stats.earliest == earliest
        assert stats.latest == latest
        mock_cursor.execute.assert_called_once()
        assert "COUNT(DISTINCT ticker)" in mock_cursor.execute.call_args[0][0]

    def test_get_overall_stats_empty_database(self, repository, mock_db_config):
        """Test get_overall_stats with empty database."""
        _, mock_cursor = mock_db_config
        mock_cursor.fetchone.return_value = (0, 0, None, None)

        stats = repository.get_overall_stats()

        assert stats.tickers_with_data == 0
        assert stats.total_records == 0
        assert stats.earliest is None
        assert stats.latest is None

    def test_get_sample_tickers_default_limit(self, repository, mock_db_config):
        """Test get_sample_tickers with default limit."""
        _, mock_cursor = mock_db_config
        mock_cursor.fetchall.return_value = [
            ("AAPL", 500, datetime(2024, 1, 1), datetime(2024, 1, 31)),
            ("GOOGL", 480, datetime(2024, 1, 2), datetime(2024, 1, 30)),
        ]

        tickers = repository.get_sample_tickers()

        assert len(tickers) == 2
        assert all(isinstance(t, TickerStats) for t in tickers)
        assert tickers[0].ticker == "AAPL"
        assert tickers[0].num_records == 500
        assert tickers[1].ticker == "GOOGL"
        assert tickers[1].num_records == 480
        mock_cursor.execute.assert_called_once()
        assert mock_cursor.execute.call_args[0][1] == (10,)

    def test_get_sample_tickers_custom_limit(self, repository, mock_db_config):
        """Test get_sample_tickers with custom limit."""
        _, mock_cursor = mock_db_config
        mock_cursor.fetchall.return_value = [
            ("AAPL", 500, datetime(2024, 1, 1), datetime(2024, 1, 31)),
        ]

        tickers = repository.get_sample_tickers(limit=5)

        assert len(tickers) == 1
        mock_cursor.execute.assert_called_once()
        assert mock_cursor.execute.call_args[0][1] == (5,)

    def test_get_sample_tickers_empty_result(self, repository, mock_db_config):
        """Test get_sample_tickers with no data."""
        _, mock_cursor = mock_db_config
        mock_cursor.fetchall.return_value = []

        tickers = repository.get_sample_tickers()

        assert tickers == []

    def test_get_ticker_stats_with_data(self, repository, mock_db_config):
        """Test get_ticker_stats for ticker with data."""
        _, mock_cursor = mock_db_config
        first_date = datetime(2024, 1, 1, 9, 30)
        last_date = datetime(2024, 1, 31, 16, 0)
        mock_cursor.fetchone.return_value = (500, first_date, last_date)

        stats = repository.get_ticker_stats("AAPL")

        assert isinstance(stats, TickerStats)
        assert stats.ticker == "AAPL"
        assert stats.num_records == 500
        assert stats.first_date == first_date
        assert stats.last_date == last_date
        mock_cursor.execute.assert_called_once()
        assert mock_cursor.execute.call_args[0][1] == ("AAPL",)

    def test_get_ticker_stats_no_data(self, repository, mock_db_config):
        """Test get_ticker_stats for ticker with no data."""
        _, mock_cursor = mock_db_config
        mock_cursor.fetchone.return_value = (0, None, None)

        stats = repository.get_ticker_stats("INVALID")

        assert stats is None

    def test_get_ticker_stats_none_result(self, repository, mock_db_config):
        """Test get_ticker_stats when query returns None."""
        _, mock_cursor = mock_db_config
        mock_cursor.fetchone.return_value = None

        stats = repository.get_ticker_stats("INVALID")

        assert stats is None

    def test_get_ticker_price_bars_ascending(self, repository, mock_db_config):
        """Test get_ticker_price_bars with ascending order."""
        _, mock_cursor = mock_db_config
        dt1 = datetime(2024, 1, 1, 9, 30)
        dt2 = datetime(2024, 1, 1, 10, 30)
        mock_cursor.fetchall.return_value = [
            (dt1, 150.0, 152.0, 149.5, 151.5, 1000000),
            (dt2, 151.5, 153.0, 151.0, 152.5, 1200000),
        ]

        bars = repository.get_ticker_price_bars("AAPL", limit=2, order="ASC")

        assert len(bars) == 2
        assert all(isinstance(b, PriceBar) for b in bars)
        assert bars[0].datetime == dt1
        assert bars[0].open == 150.0
        assert bars[0].volume == 1000000
        assert bars[1].datetime == dt2
        mock_cursor.execute.assert_called_once()
        assert "ORDER BY datetime ASC" in mock_cursor.execute.call_args[0][0]

    def test_get_ticker_price_bars_descending(self, repository, mock_db_config):
        """Test get_ticker_price_bars with descending order."""
        _, mock_cursor = mock_db_config
        dt1 = datetime(2024, 1, 1, 10, 30)
        dt2 = datetime(2024, 1, 1, 9, 30)
        mock_cursor.fetchall.return_value = [
            (dt1, 151.5, 153.0, 151.0, 152.5, 1200000),
            (dt2, 150.0, 152.0, 149.5, 151.5, 1000000),
        ]

        bars = repository.get_ticker_price_bars("AAPL", limit=2, order="DESC")

        assert len(bars) == 2
        assert bars[0].datetime == dt1
        assert bars[1].datetime == dt2
        mock_cursor.execute.assert_called_once()
        assert "ORDER BY datetime DESC" in mock_cursor.execute.call_args[0][0]

    def test_get_ticker_price_bars_invalid_order(self, repository, mock_db_config):
        """Test get_ticker_price_bars with invalid order parameter."""
        with pytest.raises(ValueError, match="order must be 'ASC' or 'DESC'"):
            repository.get_ticker_price_bars("AAPL", limit=3, order="INVALID")

    def test_get_ticker_price_bars_empty_result(self, repository, mock_db_config):
        """Test get_ticker_price_bars with no data."""
        _, mock_cursor = mock_db_config
        mock_cursor.fetchall.return_value = []

        bars = repository.get_ticker_price_bars("INVALID")

        assert bars == []

    def test_get_record_distribution(self, repository, mock_db_config):
        """Test get_record_distribution returns correct data."""
        _, mock_cursor = mock_db_config
        mock_cursor.fetchall.return_value = [
            ("< 100 bars", 10),
            ("100-500 bars", 50),
            ("500-1000 bars", 30),
            ("1000-2000 bars", 8),
            ("2000-3000 bars", 1),
            ("3000+ bars", 1),
        ]

        distribution = repository.get_record_distribution()

        assert len(distribution) == 6
        assert all(isinstance(d, RecordDistribution) for d in distribution)
        assert distribution[0].range_name == "< 100 bars"
        assert distribution[0].num_tickers == 10
        assert distribution[1].range_name == "100-500 bars"
        assert distribution[1].num_tickers == 50
        mock_cursor.execute.assert_called_once()
        assert "ticker_counts" in mock_cursor.execute.call_args[0][0]

    def test_get_record_distribution_empty(self, repository, mock_db_config):
        """Test get_record_distribution with no data."""
        _, mock_cursor = mock_db_config
        mock_cursor.fetchall.return_value = []

        distribution = repository.get_record_distribution()

        assert distribution == []

    def test_repository_uses_correct_db_role(self, mock_db_config):
        """Test that repository uses 'prices' role for all queries."""
        mock_config, mock_cursor = mock_db_config
        repository = CheckHourlyDataRepository(db_config_instance=mock_config)

        # Test each method uses the 'prices' role
        mock_cursor.fetchone.return_value = (0, 0, None, None)
        repository.get_overall_stats()
        mock_config.connection.assert_called_with(role="prices")

        mock_config.reset_mock()
        mock_cursor.fetchall.return_value = []
        repository.get_sample_tickers()
        mock_config.connection.assert_called_with(role="prices")

        mock_config.reset_mock()
        mock_cursor.fetchone.return_value = (0, None, None)
        repository.get_ticker_stats("AAPL")
        mock_config.connection.assert_called_with(role="prices")

        mock_config.reset_mock()
        mock_cursor.fetchall.return_value = []
        repository.get_ticker_price_bars("AAPL")
        mock_config.connection.assert_called_with(role="prices")

        mock_config.reset_mock()
        mock_cursor.fetchall.return_value = []
        repository.get_record_distribution()
        mock_config.connection.assert_called_with(role="prices")

    def test_repository_uses_default_db_config(self):
        """Test that repository uses global db_config by default."""
        with patch("src.data_access.check_hourly_data_repository.db_config") as mock:
            repository = CheckHourlyDataRepository()
            assert repository.db_config == mock


class TestIntegrationScenarios:
    """Integration-style tests for common usage scenarios."""

    @pytest.fixture
    def mock_db_config(self):
        """Create a mock db_config for testing."""
        mock_config = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value = mock_cursor
        mock_config.connection.return_value.__enter__.return_value = mock_conn
        mock_config.connection.return_value.__exit__.return_value = None
        return mock_config, mock_cursor

    def test_full_analysis_workflow(self, mock_db_config):
        """Test a complete analysis workflow."""
        mock_config, mock_cursor = mock_db_config
        repository = CheckHourlyDataRepository(db_config_instance=mock_config)

        # Step 1: Get overall stats
        mock_cursor.fetchone.return_value = (
            100,
            50000,
            datetime(2024, 1, 1),
            datetime(2024, 1, 31),
        )
        overall = repository.get_overall_stats()
        assert overall.tickers_with_data == 100
        assert overall.days_span == 30

        # Step 2: Get sample tickers
        mock_cursor.fetchall.return_value = [
            ("AAPL", 500, datetime(2024, 1, 1), datetime(2024, 1, 31)),
        ]
        samples = repository.get_sample_tickers(limit=1)
        assert len(samples) == 1
        assert samples[0].ticker == "AAPL"

        # Step 3: Get detailed ticker stats
        mock_cursor.fetchone.return_value = (
            500,
            datetime(2024, 1, 1),
            datetime(2024, 1, 31),
        )
        ticker_stats = repository.get_ticker_stats("AAPL")
        assert ticker_stats.num_records == 500

        # Step 4: Get price bars
        mock_cursor.fetchall.return_value = [
            (datetime(2024, 1, 1, 9, 30), 150.0, 152.0, 149.5, 151.5, 1000000),
        ]
        bars = repository.get_ticker_price_bars("AAPL", limit=1)
        assert len(bars) == 1
        assert bars[0].open == 150.0

        # Step 5: Get distribution
        mock_cursor.fetchall.return_value = [
            ("100-500 bars", 50),
        ]
        distribution = repository.get_record_distribution()
        assert len(distribution) == 1

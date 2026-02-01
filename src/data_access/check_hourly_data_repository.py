"""
Repository for checking and analyzing hourly price data in the database.
"""

from datetime import datetime
from typing import Optional

from db_config import db_config


class HourlyDataStats:
    """Container for overall hourly data statistics."""

    def __init__(
        self,
        tickers_with_data: int,
        total_records: int,
        earliest: Optional[datetime],
        latest: Optional[datetime],
    ):
        self.tickers_with_data = tickers_with_data
        self.total_records = total_records
        self.earliest = earliest
        self.latest = latest

    @property
    def days_span(self) -> Optional[int]:
        """Calculate the span in days between earliest and latest dates."""
        if not self.earliest or not self.latest:
            return None
        earliest_dt = self._to_datetime(self.earliest)
        latest_dt = self._to_datetime(self.latest)
        return (latest_dt - earliest_dt).days

    @staticmethod
    def _to_datetime(dt: datetime | str) -> datetime:
        """Convert string or datetime to datetime object."""
        if isinstance(dt, str):
            return datetime.fromisoformat(dt)
        return dt


class TickerStats:
    """Container for per-ticker statistics."""

    def __init__(
        self,
        ticker: str,
        num_records: int,
        first_date: datetime,
        last_date: datetime,
    ):
        self.ticker = ticker
        self.num_records = num_records
        self.first_date = first_date
        self.last_date = last_date

    @property
    def days_covered(self) -> int:
        """Calculate days covered, minimum 1."""
        first = self._to_datetime(self.first_date)
        last = self._to_datetime(self.last_date)
        return max((last - first).days, 1)

    @property
    def avg_bars_per_day(self) -> float:
        """Calculate average bars per day."""
        return self.num_records / self.days_covered

    @staticmethod
    def _to_datetime(dt: datetime | str) -> datetime:
        """Convert string or datetime to datetime object."""
        if isinstance(dt, str):
            return datetime.fromisoformat(dt)
        return dt


class PriceBar:
    """Container for a single price bar."""

    def __init__(
        self,
        datetime: datetime,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: int,
    ):
        self.datetime = datetime
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


class RecordDistribution:
    """Container for record count distribution."""

    def __init__(self, range_name: str, num_tickers: int):
        self.range_name = range_name
        self.num_tickers = num_tickers


class CheckHourlyDataRepository:
    """Repository for analyzing hourly price data."""

    def __init__(self, db_config_instance=None):
        """Initialize the repository with optional db_config override for testing."""
        self.db_config = db_config_instance or db_config

    def get_overall_stats(self) -> HourlyDataStats:
        """Get overall statistics for hourly price data.

        Returns:
            HourlyDataStats object containing overall statistics.
        """
        with self.db_config.connection(role="prices") as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    COUNT(DISTINCT ticker) AS tickers_with_data,
                    COUNT(*) AS total_records,
                    MIN(datetime) AS earliest,
                    MAX(datetime) AS latest
                FROM hourly_prices
                """
            )
            stats = cursor.fetchone()
            return HourlyDataStats(*stats)

    def get_sample_tickers(self, limit: int = 10) -> list[TickerStats]:
        """Get a sample of tickers with their statistics.

        Args:
            limit: Maximum number of tickers to return.

        Returns:
            List of TickerStats objects.
        """
        with self.db_config.connection(role="prices") as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    ticker,
                    COUNT(*) AS num_records,
                    MIN(datetime) AS first_date,
                    MAX(datetime) AS last_date
                FROM hourly_prices
                GROUP BY ticker
                ORDER BY ticker
                LIMIT %s
                """,
                (limit,),
            )
            return [TickerStats(*row) for row in cursor.fetchall()]

    def get_ticker_stats(self, ticker: str) -> Optional[TickerStats]:
        """Get statistics for a specific ticker.

        Args:
            ticker: The ticker symbol to query.

        Returns:
            TickerStats object if data exists, None otherwise.
        """
        with self.db_config.connection(role="prices") as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    COUNT(*) AS num_records,
                    MIN(datetime) AS first_datetime,
                    MAX(datetime) AS last_datetime
                FROM hourly_prices
                WHERE ticker = %s
                """,
                (ticker,),
            )
            row = cursor.fetchone()
            if row and row[0] > 0:
                return TickerStats(ticker, row[0], row[1], row[2])
            return None

    def get_ticker_price_bars(
        self,
        ticker: str,
        limit: int = 3,
        order: str = "ASC",
    ) -> list[PriceBar]:
        """Get price bars for a specific ticker.

        Args:
            ticker: The ticker symbol to query.
            limit: Maximum number of bars to return.
            order: Sort order - 'ASC' for oldest first, 'DESC' for newest first.

        Returns:
            List of PriceBar objects.
        """
        if order not in ("ASC", "DESC"):
            raise ValueError("order must be 'ASC' or 'DESC'")

        with self.db_config.connection(role="prices") as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""
                SELECT datetime, open, high, low, close, volume
                FROM hourly_prices
                WHERE ticker = %s
                ORDER BY datetime {order}
                LIMIT %s
                """,
                (ticker, limit),
            )
            return [PriceBar(*row) for row in cursor.fetchall()]

    def get_record_distribution(self) -> list[RecordDistribution]:
        """Get the distribution of records by count ranges.

        Returns:
            List of RecordDistribution objects.
        """
        with self.db_config.connection(role="prices") as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                WITH ticker_counts AS (
                    SELECT ticker, COUNT(*) AS cnt
                    FROM hourly_prices
                    GROUP BY ticker
                )
                SELECT
                    CASE
                        WHEN cnt < 100 THEN '< 100 bars'
                        WHEN cnt < 500 THEN '100-500 bars'
                        WHEN cnt < 1000 THEN '500-1000 bars'
                        WHEN cnt < 2000 THEN '1000-2000 bars'
                        WHEN cnt < 3000 THEN '2000-3000 bars'
                        ELSE '3000+ bars'
                    END AS range,
                    COUNT(*) AS num_tickers
                FROM ticker_counts
                GROUP BY range
                ORDER BY
                    CASE range
                        WHEN '< 100 bars' THEN 1
                        WHEN '100-500 bars' THEN 2
                        WHEN '500-1000 bars' THEN 3
                        WHEN '1000-2000 bars' THEN 4
                        WHEN '2000-3000 bars' THEN 5
                        ELSE 6
                    END
                """
            )
            return [RecordDistribution(*row) for row in cursor.fetchall()]

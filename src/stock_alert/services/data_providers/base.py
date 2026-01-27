"""
Abstract base class for data providers.

Defines the interface that all data providers must implement.
"""

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class AbstractDataProvider(ABC):
    """
    Abstract interface for market data providers.

    All data providers (FMP, Interactive Brokers, etc.) must implement
    this interface to ensure consistent behavior.
    """

    @abstractmethod
    def get_historical_prices(
        self,
        ticker: str,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
        days: int | None = None,
    ) -> pd.DataFrame:
        """
        Fetch historical price data for a ticker.

        Args:
            ticker: Stock symbol (e.g., "AAPL")
            from_date: Start date for data (optional)
            to_date: End date for data (optional)
            days: Number of days to fetch (alternative to date range)

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        pass

    @abstractmethod
    def get_latest_price(self, ticker: str) -> float | None:
        """
        Fetch the latest price for a ticker.

        Args:
            ticker: Stock symbol

        Returns:
            Latest close price or None if unavailable
        """
        pass

    @abstractmethod
    def get_quote(self, ticker: str) -> dict | None:
        """
        Fetch current quote data for a ticker.

        Args:
            ticker: Stock symbol

        Returns:
            Dictionary with quote data (price, volume, bid, ask, etc.)
        """
        pass

    @abstractmethod
    def validate_ticker(self, ticker: str) -> bool:
        """
        Check if a ticker symbol is valid.

        Args:
            ticker: Stock symbol to validate

        Returns:
            True if ticker exists, False otherwise
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the data provider (e.g., 'FMP', 'Interactive Brokers')"""
        pass

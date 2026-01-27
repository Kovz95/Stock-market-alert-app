"""
Financial Modeling Prep (FMP) data provider implementation.

This consolidates the FMP data fetching logic from multiple backend files
into a single, well-tested provider class.
"""

import logging
import os
from datetime import datetime, timedelta

import pandas as pd
import requests

from .base import AbstractDataProvider

logger = logging.getLogger(__name__)


class FMPDataProvider(AbstractDataProvider):
    """
    Financial Modeling Prep API data provider.

    Provides access to historical and real-time market data from FMP.
    Optimized to fetch only missing data rather than full historical ranges.
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize FMP data provider.

        Args:
            api_key: FMP API key (defaults to FMP_API_KEY environment variable)
        """
        self.api_key = api_key or os.getenv("FMP_API_KEY")
        if not self.api_key:
            raise ValueError("FMP_API_KEY is required (via parameter or environment)")

        self.base_url = "https://financialmodelingprep.com/api/v3"
        self._session = requests.Session()

    @property
    def name(self) -> str:
        """Return provider name"""
        return "Financial Modeling Prep"

    def get_historical_prices(
        self,
        ticker: str,
        from_date: datetime | None = None,
        to_date: datetime | None = None,
        days: int | None = None,
    ) -> pd.DataFrame:
        """
        Fetch historical price data from FMP.

        Args:
            ticker: Stock symbol
            from_date: Start date for data
            to_date: End date for data
            days: Alternative to date range - fetch last N days

        Returns:
            DataFrame with OHLCV data
        """
        # Determine date range
        if days:
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
        elif not from_date:
            from_date = datetime.now() - timedelta(days=365)

        if not to_date:
            to_date = datetime.now()

        # Format dates for API
        from_str = from_date.strftime("%Y-%m-%d")
        to_str = to_date.strftime("%Y-%m-%d")

        # Build URL
        url = f"{self.base_url}/historical-price-full/{ticker}"
        params = {
            "from": from_str,
            "to": to_str,
            "apikey": self.api_key,
        }

        try:
            response = self._session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "historical" not in data:
                logger.warning(f"No historical data for {ticker}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(data["historical"])

            if df.empty:
                return df

            # Standardize column names
            df = df.rename(
                columns={
                    "date": "date",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                }
            )

            # Convert date column
            df["date"] = pd.to_datetime(df["date"])

            # Sort by date ascending
            df = df.sort_values("date").reset_index(drop=True)

            # Select only needed columns
            columns = ["date", "open", "high", "low", "close", "volume"]
            df = df[[col for col in columns if col in df.columns]]

            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch data for {ticker}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error fetching data for {ticker}: {e}")
            return pd.DataFrame()

    def get_latest_price(self, ticker: str) -> float | None:
        """
        Fetch the latest price for a ticker.

        Args:
            ticker: Stock symbol

        Returns:
            Latest close price or None
        """
        quote = self.get_quote(ticker)
        if quote:
            return quote.get("price")
        return None

    def get_quote(self, ticker: str) -> dict | None:
        """
        Fetch current quote data from FMP.

        Args:
            ticker: Stock symbol

        Returns:
            Dictionary with quote data
        """
        url = f"{self.base_url}/quote/{ticker}"
        params = {"apikey": self.api_key}

        try:
            response = self._session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data and isinstance(data, list) and len(data) > 0:
                return data[0]

            return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch quote for {ticker}: {e}")
            return None

    def validate_ticker(self, ticker: str) -> bool:
        """
        Check if a ticker symbol is valid on FMP.

        Args:
            ticker: Stock symbol

        Returns:
            True if ticker exists
        """
        quote = self.get_quote(ticker)
        return quote is not None

    def get_stock_profile(self, ticker: str) -> dict | None:
        """
        Fetch company profile information.

        Args:
            ticker: Stock symbol

        Returns:
            Dictionary with company profile data
        """
        url = f"{self.base_url}/profile/{ticker}"
        params = {"apikey": self.api_key}

        try:
            response = self._session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data and isinstance(data, list) and len(data) > 0:
                return data[0]

            return None

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch profile for {ticker}: {e}")
            return None

    def close(self):
        """Close the session"""
        if self._session:
            self._session.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


class OptimizedFMPDataProvider(FMPDataProvider):
    """
    Optimized FMP provider that integrates with database to fetch only missing data.

    This is a migration of the OptimizedFMPDataFetcher class from backend_fmp_optimized.py
    """

    def __init__(self, api_key: str | None = None):
        """Initialize optimized FMP provider"""
        super().__init__(api_key)
        # Import here to avoid circular dependency
        from db_config import db_config

        self.db_config = db_config

    def get_missing_dates(self, ticker: str) -> tuple[datetime | None, int]:
        """
        Determine what dates are missing for a ticker.

        Returns:
            Tuple of (last_date_in_db, days_to_fetch)
        """
        try:
            with self.db_config.connection(role="price_data") as conn:
                result = self.db_config.execute_with_retry(
                    conn,
                    """
                    SELECT MAX(date) FROM daily_prices
                    WHERE ticker = %s
                    """,
                    (ticker,),
                )

            last_value = result[0][0] if result else None

            if last_value:
                last_date = pd.to_datetime(last_value)
                if getattr(last_date, "tzinfo", None):
                    last_date = last_date.tz_convert(None)

                # Calculate days since last date
                today = pd.Timestamp.now().normalize()
                days_missing = (today - last_date).days

                return last_date, max(0, days_missing)

            # No data found, fetch 365 days
            return None, 365

        except Exception as e:
            logger.error(f"Error checking missing dates for {ticker}: {e}")
            return None, 365

    def fetch_and_store(self, ticker: str, days_back: int = 7) -> bool:
        """
        Fetch missing data and store in database.

        Args:
            ticker: Stock symbol
            days_back: Number of days to fetch

        Returns:
            True if successful
        """
        # Check what's missing
        last_date, days_missing = self.get_missing_dates(ticker)

        if days_missing == 0:
            logger.info(f"{ticker}: Data is up to date")
            return True

        # Fetch the data
        days_to_fetch = min(days_missing + days_back, 365)
        df = self.get_historical_prices(ticker, days=days_to_fetch)

        if df.empty:
            logger.warning(f"{ticker}: No data fetched")
            return False

        # Store in database
        try:
            with self.db_config.connection(role="price_data") as conn:
                # Prepare data for insertion
                df["ticker"] = ticker
                records = df.to_dict("records")

                cursor = conn.cursor()
                for record in records:
                    cursor.execute(
                        """
                        INSERT INTO daily_prices (ticker, date, open, high, low, close, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (ticker, date) DO UPDATE SET
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume
                        """,
                        (
                            record["ticker"],
                            record["date"],
                            record["open"],
                            record["high"],
                            record["low"],
                            record["close"],
                            record["volume"],
                        ),
                    )
                conn.commit()

            logger.info(f"{ticker}: Stored {len(df)} rows")
            return True

        except Exception as e:
            logger.error(f"Error storing data for {ticker}: {e}")
            return False

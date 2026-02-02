"""
Hourly Price Data Collector
Collects and stores hourly price data for stocks during trading hours
Used by the hourly data scheduler for real-time monitoring
"""

import logging
import time as time_module
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from src.services.backend_fmp import FMPDataFetcher
from src.data_access.metadata_repository import fetch_stock_metadata_map
from src.data_access.daily_price_repository import DailyPriceRepository

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HourlyPriceCollector:
    """
    Collects hourly price data for stocks
    Uses the same database as DailyPriceCollector but stores hourly intervals
    """

    def __init__(self):
        """
        Initialize the hourly price collector
        """
        self.db = DailyPriceRepository()
        self.fetcher = FMPDataFetcher()
        self.stats = {
            'updated': 0,
            'skipped': 0,
            'failed': 0,
            'new': 0,
        }
        self.last_error: Optional[str] = None

    def load_stock_database(self) -> Dict[str, Dict]:
        """
        Load stock metadata from the database

        Returns:
            Dictionary mapping ticker symbols to their metadata
        """
        try:
            return fetch_stock_metadata_map()
        except Exception as e:
            logger.error(f"Error loading stock database: {e}")
            return {}

    def get_all_tickers(self) -> List[str]:
        """
        Get list of all tickers from the stock database

        Returns:
            Sorted list of ticker symbols
        """
        stock_db = self.load_stock_database()
        return sorted(stock_db.keys())

    def update_ticker_hourly(
        self,
        ticker: str,
        days_back: int = 7,
        skip_existing: bool = True
    ) -> bool:
        """
        Update hourly price data for a single ticker

        Args:
            ticker: Stock symbol to update
            days_back: Number of days of history to fetch (default: 7)
            skip_existing: If True, skip if recent data exists (default: True)

        Returns:
            True if update successful, False otherwise
        """
        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)

            # Check if we should skip (if recent data exists)
            if skip_existing:
                existing_df = self.db.get_hourly_prices(
                    ticker,
                    start_datetime=from_date,
                    end_datetime=to_date
                )

                if not existing_df.empty:
                    # Check if we have very recent data (within last 2 hours)
                    latest_datetime = existing_df.index.max()
                    hours_since_update = (pd.Timestamp.now() - latest_datetime).total_seconds() / 3600

                    if hours_since_update < 2:
                        logger.debug(f"{ticker}: Up to date (last: {latest_datetime})")
                        self.stats['skipped'] += 1
                        return True

            # Fetch hourly data from API
            logger.debug(f"Updating hourly data for {ticker}...")
            df = self.fetcher.get_hourly_data(
                ticker,
                from_date=from_date.strftime('%Y-%m-%d'),
                to_date=to_date.strftime('%Y-%m-%d')
            )

            if df is None or df.empty:
                logger.debug(f"{ticker}: No hourly data received")
                self.stats['failed'] += 1
                self.last_error = getattr(self.fetcher, "last_error", None) or "No hourly data from API"
                return False

            # Store hourly data
            records = self.db.store_hourly_prices(ticker, df)

            if records > 0:
                logger.debug(f"{ticker}: Stored {records} hourly records")
                self.stats['updated'] += 1
                return True
            else:
                self.stats['failed'] += 1
                self.last_error = "Store returned 0 records"
                return False

        except Exception as e:
            logger.error(f"Error updating hourly data for {ticker}: {e}")
            self.stats['failed'] += 1
            self.last_error = str(e)
            return False

    def update_multiple_tickers(
        self,
        tickers: List[str],
        days_back: int = 7,
        skip_existing: bool = True,
        rate_limit_delay: float = 0.2
    ) -> Dict[str, int]:
        """
        Update hourly data for multiple tickers

        Args:
            tickers: List of ticker symbols to update
            days_back: Number of days of history to fetch
            skip_existing: If True, skip tickers with recent data
            rate_limit_delay: Delay between requests in seconds (default: 0.2)

        Returns:
            Dictionary with statistics (updated, skipped, failed)
        """
        logger.info(f"Starting hourly update for {len(tickers)} tickers")
        start_time = time_module.time()

        # Reset stats
        self.stats = {
            'updated': 0,
            'skipped': 0,
            'failed': 0,
            'new': 0,
        }

        for i, ticker in enumerate(tickers, 1):
            self.update_ticker_hourly(ticker, days_back, skip_existing)

            # Rate limiting
            if rate_limit_delay > 0:
                time_module.sleep(rate_limit_delay)

            # Progress logging every 50 tickers
            if i % 50 == 0 or i == len(tickers):
                elapsed = time_module.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Progress: {i}/{len(tickers)} | "
                    f"Updated={self.stats['updated']}, "
                    f"Skipped={self.stats['skipped']}, "
                    f"Failed={self.stats['failed']} | "
                    f"Rate: {rate:.1f} tickers/sec"
                )

        elapsed = time_module.time() - start_time
        logger.info(
            f"Hourly update complete in {elapsed:.1f}s | "
            f"Final stats: {self.stats}"
        )

        return self.stats

    def get_hourly_prices(
        self,
        ticker: str,
        start_datetime: Optional[datetime] = None,
        end_datetime: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Retrieve hourly prices for a ticker

        Args:
            ticker: Stock symbol
            start_datetime: Start datetime for data range
            end_datetime: End datetime for data range
            limit: Maximum number of records to return

        Returns:
            DataFrame with hourly OHLCV data
        """
        return self.db.get_hourly_prices(ticker, start_datetime, end_datetime, limit)

    def get_statistics(self) -> Dict:
        """
        Get collection statistics

        Returns:
            Dictionary with update statistics
        """
        return self.stats.copy()

    def get_database_statistics(self) -> Dict:
        """
        Get database statistics

        Returns:
            Dictionary with database statistics
        """
        try:
            return self.db.get_statistics()
        except Exception as e:
            logger.error(f"Error getting database statistics: {e}")
            return {}

    def close(self):
        """Clean up database connection"""
        if self.db:
            self.db.close()


def run_hourly_update(
    exchanges: Optional[List[str]] = None,
    days_back: int = 7,
    skip_existing: bool = True
):
    """
    Run hourly update for tickers from specified exchanges

    Args:
        exchanges: List of exchange names to filter tickers (default: all)
        days_back: Number of days of history to fetch
        skip_existing: If True, skip tickers with recent data
    """
    logger.info("=" * 70)
    logger.info(f"Hourly Price Update - {datetime.now()}")
    logger.info("=" * 70)

    collector = HourlyPriceCollector()

    try:
        # Load stock database
        stock_db = collector.load_stock_database()

        # Filter by exchanges if specified
        if exchanges:
            tickers = [
                ticker for ticker, info in stock_db.items()
                if info.get('exchange') in exchanges
            ]
            logger.info(f"Filtered {len(tickers)} tickers for exchanges: {exchanges}")
        else:
            tickers = list(stock_db.keys())
            logger.info(f"Updating all {len(tickers)} tickers")

        if not tickers:
            logger.warning("No tickers to update")
            return

        # Run update
        stats = collector.update_multiple_tickers(
            tickers,
            days_back=days_back,
            skip_existing=skip_existing
        )

        # Display final statistics
        logger.info("\nFinal Statistics:")
        logger.info(f"  Updated: {stats['updated']}")
        logger.info(f"  Skipped: {stats['skipped']}")
        logger.info(f"  Failed: {stats['failed']}")

        # Database statistics
        db_stats = collector.get_database_statistics()
        if db_stats:
            logger.info(f"\nDatabase Statistics:")
            if 'total_tickers' in db_stats:
                logger.info(f"  Total tickers: {db_stats['total_tickers']:,}")
            if 'total_daily_records' in db_stats:
                logger.info(f"  Daily records: {db_stats['total_daily_records']:,}")

    finally:
        collector.close()


def test_hourly_collector():
    """Test the hourly price collector"""
    logger.info("Testing Hourly Price Collector")
    logger.info("=" * 70)

    collector = HourlyPriceCollector()

    try:
        # Test with a single ticker
        test_ticker = "AAPL"
        logger.info(f"\nTesting with {test_ticker}")

        # Update hourly data
        success = collector.update_ticker_hourly(test_ticker, days_back=7, skip_existing=False)
        logger.info(f"Update result: {'Success' if success else 'Failed'}")

        # Retrieve data
        df = collector.get_hourly_prices(
            test_ticker,
            start_datetime=datetime.now() - timedelta(days=7)
        )

        if not df.empty:
            logger.info(f"\nRetrieved {len(df)} hourly records")
            logger.info(f"Date range: {df.index.min()} to {df.index.max()}")
            logger.info("\nSample data (last 5 records):")
            logger.info(df.tail())
        else:
            logger.warning("No data retrieved")

        # Display statistics
        stats = collector.get_statistics()
        logger.info(f"\nCollection Statistics: {stats}")

    finally:
        collector.close()

    logger.info("\n" + "=" * 70)
    logger.info("Test complete")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_hourly_collector()
    else:
        run_hourly_update()

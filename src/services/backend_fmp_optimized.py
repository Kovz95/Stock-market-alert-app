"""
Optimized FMP Data Fetcher for Scheduler
- Fetches only missing data instead of 300 days
- Caches data to Redis for fast web app access
- Supports date range queries
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
import os
import json
from src.data_access.db_config import db_config
from src.data_access import redis_support
from src.utils.cache_helpers import get_cache_ttl, build_cache_key

logger = logging.getLogger(__name__)

class OptimizedFMPDataFetcher:
    """Optimized FMP data fetcher that only fetches missing data"""

    def __init__(self):
        self.api_key = os.getenv('FMP_API_KEY')
        self.base_url = "https://financialmodelingprep.com/api/v3"
        if not self.api_key:
            logger.warning("FMP_API_KEY not set - API calls will fail")

    def get_missing_dates(self, ticker: str) -> tuple[Optional[datetime], int]:
        """
        Determine what dates are missing for a ticker
        Returns: (last_date_in_db, days_to_fetch)
        """
        try:
            with db_config.connection(role="price_data") as conn:
                result = db_config.execute_with_retry(
                    conn,
                    """
                    SELECT MAX(date) FROM daily_prices
                    WHERE ticker = ?
                    """,
                    (ticker,),
                )

            last_value = result[0][0] if result else None

            if last_value:
                last_date = pd.to_datetime(last_value)
                if getattr(last_date, "tzinfo", None):
                    last_date = last_date.tz_convert(None)
                # Normalize to tz-naive to avoid tz-aware comparisons
                today = pd.Timestamp.utcnow().tz_localize(None).normalize()

                # Calculate business days missing (excluding weekends)
                days_missing = pd.bdate_range(start=last_date, end=today).shape[0] - 1

                # If we have today's data already (after market close), no update needed
                if last_date >= today:
                    return last_date, 0

                # Fetch a few extra days for safety (holidays, etc)
                days_to_fetch = min(days_missing + 5, 30)  # Max 30 days

                return last_date, days_to_fetch
            else:
                # No data for this ticker, fetch initial dataset
                # For new tickers, get 3 years (750 trading days) for comprehensive history
                return None, 750

        except Exception as e:
            logger.error(f"Error checking missing dates for {ticker}: {e}")
            # On error, fetch 3 years as fallback
            return None, 750

    def get_historical_data_optimized(self, ticker: str, force_days: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Fetch only missing historical data for a ticker

        Args:
            ticker: Stock symbol
            force_days: Override the number of days to fetch (for testing)
        """
        try:
            # Determine how much data we need
            if force_days:
                days_to_fetch = force_days
                last_date = None
            else:
                last_date, days_to_fetch = self.get_missing_dates(ticker)

            # Skip if no update needed
            if days_to_fetch == 0:
                logger.debug(f"{ticker}: Already up to date")
                return pd.DataFrame()  # Return empty df to indicate no update needed

            # Log what we're fetching
            if last_date:
                logger.info(f"{ticker}: Fetching {days_to_fetch} days (last: {last_date.strftime('%Y-%m-%d')})")
            else:
                logger.info(f"{ticker}: Fetching initial {days_to_fetch} days")

            # Use ticker directly from main_database_with_etfs.json (no mapping)
            logger.debug(f"Using symbol directly: {ticker}")

            # Build API request with appropriate limit
            url = f"{self.base_url}/historical-price-full/{ticker}"
            params = {
                "apikey": self.api_key,
                "limit": days_to_fetch
            }

            # Make API request
            response = requests.get(url, params=params, timeout=10)

            if response.status_code == 200:
                data = response.json()

                if "historical" in data and data["historical"]:
                    df = pd.DataFrame(data["historical"])

                    # Process the data
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')

                    # Standardize column names
                    column_mapping = {
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume',
                        'adjClose': 'Adj Close',
                        'change': 'Change',
                        'changePercent': 'Change %',
                        'vwap': 'VWAP',
                        'changeOverTime': 'Change Over Time'
                    }
                    df = df.rename(columns=column_mapping)

                    # If we have existing data, filter to only new dates
                    if last_date:
                        df = df[df['date'] > last_date]
                        logger.info(f"{ticker}: Fetched {len(df)} new records")

                    # Set date as index (required by store_daily_prices)
                    df = df.set_index('date')

                    return df
                else:
                    # Empty or missing "historical" — common for delisted, renamed, or unsupported symbols
                    logger.warning(
                        "%s: No historical data in API response (may be delisted, renamed, or unsupported)",
                        ticker,
                    )
                    return None
            else:
                logger.error(f"{ticker}: API error {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None

    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time quote for a symbol (for live checks)
        No caching for real-time quotes
        """
        # Use symbol directly (no mapping)
        url = f"{self.base_url}/quote/{symbol}"
        params = {"apikey": self.api_key}

        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    return data[0]
            return None
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return None


class OptimizedDailyPriceCollector:
    """
    Optimized version that only fetches missing data
    """

    def __init__(self):
        self.fetcher = OptimizedFMPDataFetcher()
        self.db = self._get_db()
        self.stats : dict[str, Any] = {
            'updated': 0,
            'skipped': 0,
            'skipped_tickers': [],
            'failed': 0,
            'failed_tickers': [],
            'new': 0,
            'records_fetched': 0,
            'api_calls': 0,
            'weekly_updated': 0  # Track weekly price updates
        }

    def _get_db(self):
        """Get database instance from data access layer."""
        from src.data_access.daily_price_repository import DailyPriceRepository
        return DailyPriceRepository()

    def update_ticker(
        self, ticker: str, resample_weekly: bool = False, force_refresh: bool = False
    ) -> bool:
        """
        Update daily prices for a single ticker (optimized version).

        Args:
            ticker: The ticker to update.
            resample_weekly: If True, also resample to weekly data (should only be True on Fridays).
            force_refresh: If True, skip "already up to date" checks and fetch the last N days
                so that an explicit refresh from the UI always writes to the database.
        """
        try:
            if not force_refresh:
                needs_update, last_update = self.db.needs_update(ticker, force_after_close=True)
                if not needs_update:
                    logger.debug(f"{ticker}: Already up to date")
                    self.stats['skipped'] += 1
                    self.stats['skipped_tickers'].append(ticker)
                    return True
            else:
                last_update = None

            # Fetch data: force_refresh pulls last 30 days so we always have something to write
            logger.debug(f"Updating {ticker}..." + (" (force refresh)" if force_refresh else ""))
            df = (
                self.fetcher.get_historical_data_optimized(ticker, force_days=30)
                if force_refresh
                else self.fetcher.get_historical_data_optimized(ticker)
            )
            self.stats['api_calls'] += 1

            if df is None:
                logger.warning(f"{ticker}: Failed to fetch data")
                self.stats['failed'] += 1
                self.stats['failed_tickers'].append(ticker)
                return False

            if df.empty:
                # Empty df means already up to date
                logger.debug(f"{ticker}: No new data")
                self.stats['skipped'] += 1
                self.stats['skipped_tickers'].append(ticker)
                return True

            # Store only the new data
            try:
                records = self.db.store_daily_prices(ticker, df)
                self.stats['records_fetched'] += records

                # Cache the daily data to Redis for fast web app access
                if records > 0:
                    try:
                        # Get full dataset for caching (250 days optimized for performance)
                        full_df = self.db.get_daily_prices(ticker, limit=250)
                        if full_df is not None and not full_df.empty:
                            self._cache_price_data(ticker, full_df, timeframe='1d')
                    except Exception as e:
                        logger.debug(f"{ticker}: Failed to cache daily data: {e}")

            except Exception as e:
                logger.error(f"{ticker}: Failed to store daily prices: {e}", exc_info=True)
                self.stats['failed'] += 1
                self.stats['failed_tickers'].append(ticker)
                return False

            # Update weekly data on Fridays (when resample_weekly is True)
            # Resample if: new daily data was fetched OR weekly data is missing/incomplete
            should_resample = False
            if resample_weekly:
                if records > 0:
                    should_resample = True  # New daily data was fetched
                else:
                    # Check if weekly data exists and is current
                    weekly_check = self.db.get_weekly_prices(ticker, limit=1)
                    if weekly_check is None or weekly_check.empty:
                        should_resample = True  # No weekly data exists
                        logger.info(f"{ticker}: No weekly data found, will regenerate")

            if should_resample:
                # Get sufficient data for weekly calculation (at least 250 days for proper weekly aggregation)
                # This ensures we have enough data to generate meaningful weekly records
                recent_df = self.db.get_daily_prices(ticker, limit=250)
                if recent_df is not None and not recent_df.empty:
                    # Check if we have data from the current week
                    from datetime import datetime, timedelta, timezone
                    today = datetime.now(tz=timezone.utc).date()
                    # Find Sunday of current week
                    days_since_sunday = today.weekday() + 1 if today.weekday() != 6 else 0
                    current_week_start = today - timedelta(days=days_since_sunday)

                    # Check if the latest date is from the current week
                    latest_date = recent_df.index[-1].date() if isinstance(recent_df.index[-1], pd.Timestamp) else pd.to_datetime(recent_df.index[-1]).date()

                    # Only resample if we have data from the current week
                    if latest_date >= current_week_start:
                        weekly_df = self._resample_to_weekly(recent_df, ticker)
                        if weekly_df is not None and not weekly_df.empty:
                            try:
                                weekly_records = self.db.store_weekly_prices(ticker, weekly_df)
                                if weekly_records > 0:
                                    self.stats['weekly_updated'] += 1

                                    # Cache the weekly data to Redis
                                    try:
                                        full_weekly_df = self.db.get_weekly_prices(ticker, limit=250)
                                        if full_weekly_df is not None and not full_weekly_df.empty:
                                            self._cache_price_data(ticker, full_weekly_df, timeframe='1wk')
                                    except Exception as e:
                                        logger.debug(f"{ticker}: Failed to cache weekly data: {e}")

                                logger.info(f"{ticker}: Generated {len(weekly_df)} weekly records")
                            except Exception as e:
                                logger.error(f"{ticker}: Failed to store weekly prices: {e}", exc_info=True)
                                # Don't fail the entire update just because weekly failed
                    else:
                        logger.debug(f"{ticker}: Skipping weekly resample - no current week data (latest: {latest_date})")

            if last_update:
                self.stats['updated'] += 1
            else:
                self.stats['new'] += 1

            logger.info(f"{ticker}: Added {records} new records")
            return True

        except Exception as e:
            logger.error(f"Error updating {ticker}: {e}")
            self.stats['failed'] += 1
            return False

    def _resample_to_weekly(self, df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        """Resample daily data to weekly using ACTUAL last trading day"""
        try:
            if df.empty:
                return None

            # Ensure we have a datetime index
            if 'date' in df.columns:
                df = df.set_index('date')
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # Create a copy to avoid modifying the original
            df_copy = df.copy()

            # Add week identifier (year-week number using Sunday start)
            df_copy['year_week'] = df_copy.index.strftime('%Y-%U')

            # Group by week and aggregate
            weekly_data = []
            today = pd.Timestamp(datetime.now(tz=timezone.utc)).normalize()
            current_week_start = today - pd.Timedelta(days=today.weekday())

            for year_week, week_df in df_copy.groupby('year_week'):
                if week_df.empty:
                    continue

                # Use the ACTUAL last trading day of the week as the date
                # This ensures we use Thursday if there's no Friday trading, etc.
                last_trading_day = week_df.index[-1]

                # Skip partial weeks (current week Mon-Thu only)
                # A week is "complete" if:
                # 1. It's from a past week (before current week), OR
                # 2. It's current week and today is Friday or later
                if last_trading_day >= current_week_start:
                    # This is current week data
                    if today.weekday() < 4:  # Today is Mon-Thu
                        continue  # Skip current week's partial data
                    # If today is Fri/Sat/Sun, include current week

                # Calculate weekly OHLCV using actual trading days
                weekly_record = {
                    'date': last_trading_day,  # Use actual last trading day of the week
                    'Open': week_df['Open'].iloc[0],  # First open of the week
                    'High': week_df['High'].max(),     # Highest high of the week
                    'Low': week_df['Low'].min(),       # Lowest low of the week
                    'Close': week_df['Close'].iloc[-1], # Last close of the week
                    'Volume': week_df['Volume'].sum()   # Total volume for the week
                }
                weekly_data.append(weekly_record)

            if not weekly_data:
                return None

            # Create DataFrame from weekly data
            weekly_df = pd.DataFrame(weekly_data)
            weekly_df.set_index('date', inplace=True)

            logger.info(f"{ticker}: Generated {len(weekly_df)} weekly records using actual trading dates")
            return weekly_df

        except Exception as e:
            logger.error(f"Error resampling {ticker} to weekly: {e}")
            return None

    def _cache_price_data(
        self,
        ticker: str,
        df: pd.DataFrame,
        timeframe: str = '1d',
        lookback_days: int = 250
    ) -> None:
        """
        Cache price data to Redis for fast web app access.

        Args:
            ticker: Stock ticker symbol.
            timeframe: Timeframe string ('1h', '1d', '1wk').
            df: Price DataFrame to cache.
            lookback_days: Lookback period for cache key (default: 250 for performance).
        """
        try:
            cache_key = build_cache_key(ticker, timeframe, lookback_days)
            ttl = get_cache_ttl(timeframe)

            # Prepare data for caching (convert to JSON-serializable format)
            df_reset = df.reset_index()
            # Rename index column to 'Date' for consistency
            if df_reset.columns[0] in ['date', 'datetime', 'week_ending']:
                df_reset.rename(columns={df_reset.columns[0]: 'Date'}, inplace=True)

            cache_data = {
                'data': df_reset.to_dict('records'),
                'last_update': df.index.max().isoformat() if not df.empty else None
            }

            # Store to Redis
            success = redis_support.set_json(
                cache_key,
                cache_data,
                ttl_seconds=ttl
            )

            if success:
                logger.debug(f"{ticker}: Cached {timeframe} data ({len(df)} rows, TTL: {ttl}s)")

        except Exception as e:
            logger.debug(f"{ticker}: Cache error: {e}")

    def get_statistics(self) -> Dict:
        """Get collection statistics"""
        return {
            **self.stats,
            'avg_records_per_ticker': (
                self.stats['records_fetched'] / max(self.stats['updated'] + self.stats['new'], 1)
            )
        }

    def close(self):
        """Close database connection"""
        self.db.close()

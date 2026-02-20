"""
Smart Price Fetcher Service

Intelligent price data fetcher with 3-tier retrieval strategy:
Cache (Redis) → Database (PostgreSQL) → FMP API

This service minimizes API calls by checking cache and database first,
only falling back to the FMP API when necessary. It also provides
data freshness information for transparency.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

import pandas as pd

from src.data_access.daily_price_repository import DailyPriceRepository
from src.data_access import redis_support
from src.models.price_data_result import PriceDataResult
from src.services.backend_fmp import FMPDataFetcher
from src.utils.cache_helpers import (
    get_cache_ttl,
    build_cache_key,
)
from src.utils.stale_data import is_data_stale

logger = logging.getLogger(__name__)


class SmartPriceFetcher:
    """
    Intelligent price data fetcher with 3-tier retrieval:
    Cache (Redis) → Database (PostgreSQL) → FMP API

    This service implements a cascading data retrieval strategy:
    1. Check Redis cache first (fastest, 5-60min TTL)
    2. Check PostgreSQL database if cache miss (fast, may be stale)
    3. Fetch from FMP API if database is stale (slowest, always fresh)

    Attributes:
        price_repo: Repository for database price access.
        fmp_fetcher: FMP API client for external data fetching.

    Example:
        >>> fetcher = SmartPriceFetcher()
        >>> result = fetcher.get_price_data('AAPL', timeframe='1d')
        >>> if result.is_successful:
        ...     print(f"Got {len(result.df)} bars from {result.source}")
    """

    def __init__(
        self,
        price_repo: Optional[DailyPriceRepository] = None,
        fmp_fetcher: Optional[FMPDataFetcher] = None
    ):
        """
        Initialize smart price fetcher.

        Args:
            price_repo: Optional price repository. If None, creates default.
            fmp_fetcher: Optional FMP fetcher. If None, creates default.
        """
        self.price_repo = price_repo or DailyPriceRepository()
        self.fmp_fetcher = fmp_fetcher or FMPDataFetcher()

        logger.debug("Initialized SmartPriceFetcher")

    def get_price_data(
        self,
        ticker: str,
        timeframe: str = "1d",
        lookback_days: int = 250,
        force_refresh: bool = False
    ) -> PriceDataResult:
        """
        Get price data with intelligent 3-tier fetching.

        Retrieval order (unless force_refresh=True):
        1. Redis cache (if available and not expired)
        2. Database (if data is fresh enough)
        3. FMP API (if cache miss or stale data)

        Args:
            ticker: Stock ticker symbol.
            timeframe: Timeframe ('1h', '1d', '1wk').
            lookback_days: Number of days of historical data (default: 250 for performance).
            force_refresh: If True, skip cache and fetch from API.

        Returns:
            PriceDataResult with df, source, freshness, and metadata.

        Example:
            >>> result = fetcher.get_price_data('AAPL', '1d', 250)
            >>> print(f"Source: {result.source}, Freshness: {result.freshness}")
        """
        ticker = ticker.upper().strip()
        normalized_timeframe = timeframe.lower().strip()

        logger.debug(
            f"Fetching {ticker} {timeframe} data (lookback: {lookback_days} days, "
            f"force_refresh: {force_refresh})"
        )

        # Skip cache if force refresh
        if not force_refresh:
            # Tier 1: Check cache
            cache_result = self._check_cache(ticker, timeframe, lookback_days)
            if cache_result is not None:
                return cache_result

        # Tier 2: Check database
        db_result = self._check_database(ticker, timeframe, lookback_days)

        # If database data is fresh, return it (and cache it)
        if db_result is not None and db_result.freshness == 'fresh':
            # Cache the fresh database data
            if not force_refresh:
                self._cache_data(ticker, timeframe, lookback_days, db_result.df)
            return db_result

        # Tier 3: Fetch from API (database is stale or empty)
        logger.info(f"{ticker}: Fetching from FMP API (database stale or empty)")
        api_result = self._fetch_from_api(ticker, timeframe, lookback_days)

        # Store API data to database if successful
        if api_result.is_successful:
            try:
                self._store_to_database(ticker, timeframe, api_result.df)
            except Exception as e:
                logger.error(f"Failed to store {ticker} to database: {e}")

            # Cache the API data
            if not force_refresh:
                self._cache_data(ticker, timeframe, lookback_days, api_result.df)

        return api_result

    def get_pair_price_data(
        self,
        ticker1: str,
        ticker2: str,
        timeframe: str = "1d",
        lookback_days: int = 250
    ) -> Tuple[PriceDataResult, PriceDataResult]:
        """
        Atomically fetch both symbols for pair consistency.

        Fetches both symbols and ensures they have the same data source
        and timeframe for reliable pair comparisons.

        Args:
            ticker1: First ticker symbol.
            ticker2: Second ticker symbol.
            timeframe: Timeframe ('1h', '1d', '1wk').
            lookback_days: Number of days of historical data.

        Returns:
            Tuple of (result1, result2).

        Example:
            >>> result1, result2 = fetcher.get_pair_price_data('AAPL', 'MSFT')
            >>> if result1.is_successful and result2.is_successful:
            ...     print(f"Both fetched from {result1.source}")
        """
        logger.debug(f"Fetching pair: {ticker1}/{ticker2} ({timeframe})")

        # Fetch both symbols
        result1 = self.get_price_data(ticker1, timeframe, lookback_days)
        result2 = self.get_price_data(ticker2, timeframe, lookback_days)

        return result1, result2

    def _check_cache(
        self,
        ticker: str,
        timeframe: str,
        lookback_days: int
    ) -> Optional[PriceDataResult]:
        """
        Check Redis cache for price data.

        Args:
            ticker: Stock ticker symbol.
            timeframe: Timeframe string.
            lookback_days: Number of days of lookback.

        Returns:
            PriceDataResult if cache hit, None if cache miss.
        """
        try:
            cache_key = build_cache_key(ticker, timeframe, lookback_days)
            cached_data = redis_support.get_json(cache_key)

            if cached_data is not None:
                # Reconstruct DataFrame from cached dict
                df = pd.DataFrame(cached_data['data'])
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')

                last_update = pd.to_datetime(cached_data['last_update'])

                logger.debug(f"{ticker}: Cache hit (key: {cache_key})")

                # Check freshness based on cached data
                freshness = self._determine_freshness(
                    last_update=last_update,
                    timeframe=timeframe
                )

                return PriceDataResult(
                    df=df,
                    source='cache',
                    freshness=freshness,
                    last_update=last_update,
                    error=None
                )

        except Exception as e:
            logger.debug(f"{ticker}: Cache check failed: {e}")

        return None

    def _check_database(
        self,
        ticker: str,
        timeframe: str,
        lookback_days: int
    ) -> Optional[PriceDataResult]:
        """
        Check database for price data.

        Args:
            ticker: Stock ticker symbol.
            timeframe: Timeframe string.
            lookback_days: Number of days of lookback.

        Returns:
            PriceDataResult with database data and freshness status.
        """
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)

            # Fetch from appropriate table based on timeframe
            if timeframe in ['1h', 'hourly', '1hr', 'hour']:
                df = self.price_repo.get_hourly_prices(
                    ticker,
                    start_datetime=start_date,
                    end_datetime=end_date
                )
            elif timeframe in ['1wk', 'weekly', 'week', '1w']:
                df = self.price_repo.get_weekly_prices(
                    ticker,
                    start_date=start_date,
                    end_date=end_date
                )
            else:  # Daily
                df = self.price_repo.get_daily_prices(
                    ticker,
                    start_date=start_date,
                    end_date=end_date
                )

            if df is None or df.empty:
                logger.debug(f"{ticker}: No data in database")
                return None

            # Get last update timestamp
            last_update = df.index.max()
            if isinstance(last_update, pd.Timestamp):
                last_update = last_update.to_pydatetime()

            # Check if data is stale
            is_stale = is_data_stale(last_update, timeframe=timeframe)

            freshness = self._determine_freshness(
                last_update=last_update,
                timeframe=timeframe
            )

            logger.debug(
                f"{ticker}: Database hit ({len(df)} rows, last: {last_update}, "
                f"freshness: {freshness})"
            )

            # Rename index to 'Date' for consistency
            df.index.name = 'Date'

            return PriceDataResult(
                df=df,
                source='database',
                freshness=freshness,
                last_update=last_update,
                error=None
            )

        except Exception as e:
            logger.error(f"{ticker}: Database query failed: {e}")
            return None

    def _fetch_from_api(
        self,
        ticker: str,
        timeframe: str,
        lookback_days: int
    ) -> PriceDataResult:
        """
        Fetch price data from FMP API.

        Args:
            ticker: Stock ticker symbol.
            timeframe: Timeframe string.
            lookback_days: Number of days of lookback.

        Returns:
            PriceDataResult with API data or error.
        """
        try:
            # Fetch from FMP API
            df = self.fmp_fetcher.get_historical_data(
                ticker=ticker,
                period=self._timeframe_to_period(timeframe),
                timeframe=timeframe
            )

            if df is None or df.empty:
                error_msg = f"No data from FMP API for {ticker}"
                logger.warning(error_msg)
                return PriceDataResult(
                    df=None,
                    source='api',
                    freshness='error',
                    last_update=None,
                    error=error_msg
                )

            # Trim to requested lookback
            if len(df) > lookback_days:
                df = df.tail(lookback_days)

            # Get last update
            last_update = df.index.max()
            if isinstance(last_update, pd.Timestamp):
                last_update = last_update.to_pydatetime()

            freshness = self._determine_freshness(
                last_update=last_update,
                timeframe=timeframe
            )

            logger.info(
                f"{ticker}: API fetch successful ({len(df)} rows, last: {last_update})"
            )

            # Rename index to 'Date' for consistency
            df.index.name = 'Date'

            return PriceDataResult(
                df=df,
                source='api',
                freshness=freshness,
                last_update=last_update,
                error=None
            )

        except Exception as e:
            error_msg = f"API fetch failed: {e}"
            logger.error(f"{ticker}: {error_msg}")
            return PriceDataResult(
                df=None,
                source='api',
                freshness='error',
                last_update=None,
                error=error_msg
            )

    def _cache_data(
        self,
        ticker: str,
        timeframe: str,
        lookback_days: int,
        df: pd.DataFrame
    ) -> None:
        """
        Cache price data to Redis.

        Args:
            ticker: Stock ticker symbol.
            timeframe: Timeframe string.
            lookback_days: Number of days of lookback.
            df: Price DataFrame to cache.
        """
        try:
            cache_key = build_cache_key(ticker, timeframe, lookback_days)
            ttl = get_cache_ttl(timeframe)

            # Prepare data for caching (convert to JSON-serializable format)
            df_reset = df.reset_index()
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
                logger.debug(f"{ticker}: Cached data (TTL: {ttl}s)")
            else:
                logger.debug(f"{ticker}: Cache storage failed")

        except Exception as e:
            logger.debug(f"{ticker}: Cache error: {e}")

    def _store_to_database(
        self,
        ticker: str,
        timeframe: str,
        df: pd.DataFrame
    ) -> None:
        """
        Store price data to database.

        Args:
            ticker: Stock ticker symbol.
            timeframe: Timeframe string.
            df: Price DataFrame to store.
        """
        if timeframe in ['1h', 'hourly', '1hr', 'hour']:
            records = self.price_repo.store_hourly_prices(ticker, df)
            logger.debug(f"{ticker}: Stored {records} hourly records to database")
        elif timeframe in ['1wk', 'weekly', 'week', '1w']:
            records = self.price_repo.store_weekly_prices(ticker, df)
            logger.debug(f"{ticker}: Stored {records} weekly records to database")
        else:  # Daily
            records = self.price_repo.store_daily_prices(ticker, df)
            logger.debug(f"{ticker}: Stored {records} daily records to database")

    def _determine_freshness(
        self,
        last_update: datetime,
        timeframe: str
    ) -> str:
        """
        Determine data freshness status.

        Args:
            last_update: Timestamp of last data point.
            timeframe: Timeframe string.

        Returns:
            Freshness status: 'fresh', 'recent', or 'stale'.
        """
        # Use existing stale data logic
        is_stale = is_data_stale(last_update, timeframe=timeframe)

        if is_stale:
            return 'stale'

        # Calculate age
        age_seconds = (datetime.now() - last_update).total_seconds()

        # Determine freshness based on timeframe
        if timeframe in ['1h', 'hourly', '1hr', 'hour']:
            # Hourly: fresh if < 1 hour old
            return 'fresh' if age_seconds < 3600 else 'recent'
        elif timeframe in ['1wk', 'weekly', 'week', '1w']:
            # Weekly: fresh if < 1 week old
            return 'fresh' if age_seconds < 604800 else 'recent'
        else:  # Daily
            # Daily: fresh if < 24 hours old
            return 'fresh' if age_seconds < 86400 else 'recent'

    def _timeframe_to_period(self, timeframe: str) -> str:
        """
        Convert timeframe string to FMP period parameter.

        Args:
            timeframe: Timeframe string ('1h', '1d', '1wk').

        Returns:
            FMP period string.
        """
        normalized = timeframe.lower().strip()

        if normalized in ['1h', 'hourly', '1hr', 'hour']:
            return '1hour'
        elif normalized in ['1wk', 'weekly', 'week', '1w']:
            return '1day'  # FMP doesn't have weekly, we resample daily
        else:  # Daily or default
            return '1day'


def create_smart_price_fetcher() -> SmartPriceFetcher:
    """
    Factory function to create a configured SmartPriceFetcher instance.

    Returns:
        Configured SmartPriceFetcher.

    Example:
        >>> fetcher = create_smart_price_fetcher()
        >>> result = fetcher.get_price_data('AAPL')
    """
    return SmartPriceFetcher()

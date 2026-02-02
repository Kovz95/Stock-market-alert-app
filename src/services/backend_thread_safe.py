"""
Thread-safe backend functions for alert processing
"""

import threading
from functools import lru_cache
from datetime import datetime, timedelta
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Thread-local storage for database connections
_thread_local = threading.local()

from src.data_access.db_config import db_config
from src.utils.stale_data import is_data_stale, get_last_trading_day

DAILY_ALIASES = {'1d', 'daily', 'day'}
WEEKLY_ALIASES = {'1wk', 'weekly', '1week', 'week'}
HOURLY_ALIASES = {'1h', '1hr', 'hourly', 'hour'}

def get_thread_safe_db_connection():
    """Get or create a thread-local PostgreSQL connection"""
    if not hasattr(_thread_local, 'db_conn'):
        _thread_local.db_conn = db_config.get_connection(role="prices")
    return _thread_local.db_conn

def close_thread_db_connection():
    """Close the thread-local database connection if it exists"""
    if hasattr(_thread_local, 'db_conn'):
        try:
            db_config.close_connection(_thread_local.db_conn)
        except Exception:
            pass
        delattr(_thread_local, 'db_conn')

def get_cached_stock_data_thread_safe(ticker, timeframe, adjustment_method=None):
    """
    Thread-safe version of get_cached_stock_data that uses thread-local connections
    Supports both stocks and futures
    """
    try:
        # This function is for stocks/ETFs only - no futures support needed
        # Stock handling code starts here
        # Try to use stored price data first if available
        use_storage = False
        data = None
        data_is_stale = False

        try:
            conn = get_thread_safe_db_connection()

            tf = str(timeframe).lower() if timeframe else '1d'

            if tf in WEEKLY_ALIASES:
                # Determine closing day (Thursday=4 for some exchanges, Friday=5 otherwise)
                closing_dow = 4 if any(suffix in ticker for suffix in ['.TA', '.IS', '.QA', '.KW', '.SA']) else 5
                query = """
                    SELECT date, open, high, low, close, volume
                    FROM daily_prices
                    WHERE ticker = %s
                      AND EXTRACT(DOW FROM date) = %s
                    ORDER BY date DESC
                    LIMIT 250
                """
                params = (ticker, closing_dow)
            elif tf in HOURLY_ALIASES:
                query = """
                    SELECT datetime, open, high, low, close, volume
                    FROM hourly_prices
                    WHERE ticker = %s
                    ORDER BY datetime DESC
                    LIMIT 1500
                """
                params = (ticker,)
            else:
                query = """
                    SELECT date, open, high, low, close, volume
                    FROM daily_prices
                    WHERE ticker = %s
                    ORDER BY date DESC
                    LIMIT 1500
                """
                params = (ticker,)

            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            cursor.close()

            if rows:
                if tf in HOURLY_ALIASES:
                    columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
                else:
                    columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                data = pd.DataFrame(rows, columns=columns)

                if 'datetime' in data.columns:
                    data = data.rename(columns={'datetime': 'date'})

                data['date'] = pd.to_datetime(data['date'])
                data = data.sort_values('date')
                data = data.rename(
                    columns={
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume',
                        'date': 'Date',
                    }
                )
                data.set_index('Date', inplace=True)

                use_storage = True

                most_recent_date = data.index.max()
                today = datetime.now()
                days_stale = (today - most_recent_date).days

                if tf in WEEKLY_ALIASES:
                    normalized_timeframe = '1w'
                elif tf in HOURLY_ALIASES:
                    normalized_timeframe = '1h'
                else:
                    normalized_timeframe = '1d'

                if normalized_timeframe == '1h':
                    data_is_stale = False
                else:
                    data_is_stale = is_data_stale(most_recent_date, timeframe=normalized_timeframe)

                if data_is_stale:
                    last_expected = get_last_trading_day() if normalized_timeframe == '1d' else None
                    logger.debug(f"{ticker}: Data from {most_recent_date.date()} is stale for timeframe {timeframe}")
                    from industry_logging import log_alert_check
                    log_alert_check(
                        ticker,
                        "Data Check",
                        f"⚠️ Data is {days_stale} days old (last: {most_recent_date.strftime('%Y-%m-%d')}). Treating as stale data.",
                    )
        except Exception as e:
            logger.warning(f"Storage fetch failed for {ticker}: {e}")
            logger.warning(f"Exception type: {type(e).__name__}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            pass  # Fall back to API silently
        
        # If data is stale, return special indicator
        if data_is_stale:
            return 'stale_data'  # Return 'stale_data' indicator
        
        # Fall back to API if storage not available or no data
        if not use_storage or data is None or data.empty:
            # Import FMP backend data fetcher
            from backend_fmp import FMPDataFetcher
            
            # Create fetcher instance
            fetcher = FMPDataFetcher()
            
            # Map timeframe to period for FMP API
            period_map = {
                "daily": "1day",
                "weekly": "1week",
                "hourly": "1hour",
                "1d": "1day",
                "1wk": "1week",
                "1h": "1hour"
            }
            
            period = period_map.get(timeframe, "1day")
            
            # Fetch historical data using FMP backend
            data = fetcher.get_historical_data(ticker, period)
        
        # Handle failed data fetch
        if data is None or data.empty:
            logger.warning(f"No data returned for {ticker} - data is {'None' if data is None else 'empty DataFrame'}")
            logger.warning(f"Storage used: {use_storage}, Timeframe: {timeframe}")
            return None  # Return None if no data
        
        if data is not None and not data.empty:
            # Ensure proper column names (FMP uses lowercase)
            # Rename columns to match expected format if needed
            if 'open' in data.columns:
                column_mapping = {
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume',
                    'date': 'Date'
                }
                data = data.rename(columns=column_mapping)
            
            # Reset index to have Date as a column if needed
            if data.index.name == 'Date':
                data = data.reset_index()
            
            # Ensure Date column is datetime
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
                data = data.set_index('Date')
            
            return data
        
        return None
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None

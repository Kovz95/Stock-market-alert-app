"""
Daily Price Data Collector
Collects and stores daily closing prices for all alert tickers
Runs once per day after market close to update the database
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import time as time_module
from backend_fmp import FMPDataFetcher
from db_config import db_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _drop_tz(ts):
    """Return a tz-naive Timestamp/Datetime if tz-aware, otherwise unchanged."""
    if ts is None:
        return pd.NaT
    try:
        if isinstance(ts, pd.Timestamp):
            if ts.tzinfo:
                return ts.tz_convert(None)
            return ts
        if hasattr(ts, "tzinfo") and ts.tzinfo:
            return pd.Timestamp(ts).tz_convert(None)
        return pd.Timestamp(ts)
    except Exception:
        return pd.to_datetime(ts, errors="coerce")

class DailyPriceDatabase:
    """
    SQLite database for efficient storage and retrieval of daily price data
    Optimized for time-series queries and indicator calculations
    """
    
    def __init__(self, db_path: str = "price_data.db"):
        self.db_path = db_path
        self.conn = None
        self.use_postgres = db_config.db_type == "postgresql"
        self._sqlalchemy_engine = None
        self.initialize_database()

    def _get_sqlalchemy_engine(self):
        """Get SQLAlchemy engine for pandas operations (avoids UserWarning)"""
        if self._sqlalchemy_engine is not None:
            return self._sqlalchemy_engine
        if self.use_postgres:
            self._sqlalchemy_engine = db_config.get_sqlalchemy_engine()
        else:
            # For SQLite, use the connection string
            from sqlalchemy import create_engine
            self._sqlalchemy_engine = create_engine(f"sqlite:///{self.db_path}")
        return self._sqlalchemy_engine

    def _read_sql(self, query, params=None, parse_dates=None):
        """
        Execute SQL query using SQLAlchemy engine (preferred by pandas).
        Handles both ? (SQLite) and %s (PostgreSQL) placeholders.
        """
        engine = self._get_sqlalchemy_engine()
        if params:
            from sqlalchemy import text
            # Convert placeholders to named parameters
            query_modified = query
            param_count = query.count('?')
            if param_count > 0:
                param_names = [f'p{i}' for i in range(param_count)]
                for name in param_names:
                    query_modified = query_modified.replace('?', f':{name}', 1)
                params_dict = dict(zip(param_names, params))
                # Use connection from engine for proper parameter binding
                with engine.connect() as conn:
                    return pd.read_sql_query(text(query_modified), conn, params=params_dict, parse_dates=parse_dates)
        return pd.read_sql_query(query, engine, parse_dates=parse_dates)

    @staticmethod
    def _sanitize_value(value):
        if value is None:
            return None
        if isinstance(value, (pd.Timestamp, datetime)):
            return value
        if isinstance(value, (np.generic,)):
            return value.item()
        if pd.isna(value):
            return None
        return value

    def initialize_database(self):
        """Create database tables if they don't exist"""
        # Use the new db_config for connection with WAL mode and optimizations
        if self.use_postgres:
            self.conn = db_config.get_connection(role="price_data")
        else:
            self.conn = db_config.get_connection(db_path=self.db_path)
        
        # Create price data table with optimal indexing
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS daily_prices (
                ticker TEXT NOT NULL,
                date DATE NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL NOT NULL,
                volume INTEGER,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, date)
            )
        """)
        
        # Create indexes for fast queries
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticker_date 
            ON daily_prices(ticker, date DESC)
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_date 
            ON daily_prices(date DESC)
        """)
        
        # Create metadata table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS ticker_metadata (
                ticker TEXT PRIMARY KEY,
                first_date DATE,
                last_date DATE,
                total_records INTEGER,
                last_update TIMESTAMP,
                exchange TEXT,
                asset_type TEXT
            )
        """)
        
        # Create weekly prices table (pre-computed)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS weekly_prices (
                ticker TEXT NOT NULL,
                week_ending DATE NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL NOT NULL,
                volume INTEGER,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, week_ending)
            )
        """)

        # Create hourly prices table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS hourly_prices (
                ticker TEXT NOT NULL,
                datetime TIMESTAMP NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL NOT NULL,
                volume INTEGER,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (ticker, datetime)
            )
        """)

        # Create index for hourly data
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_hourly_ticker_datetime
            ON hourly_prices(ticker, datetime DESC)
        """)

        self.conn.commit()
    
    def store_daily_prices(self, ticker: str, df: pd.DataFrame) -> int:
        """
        Store or update daily prices for a ticker
        Returns number of records inserted/updated
        """
        if df.empty:
            return 0

        # Ensure the DataFrame has a proper date index
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error(f"{ticker}: DataFrame index is not DatetimeIndex, type: {type(df.index)}")
            return 0

        # Prepare data for insertion
        records = []
        for date, row in df.iterrows():
            # Validate date before storing
            if isinstance(date, pd.Timestamp):
                date_str = date.strftime('%Y-%m-%d')
            elif isinstance(date, datetime):
                date_str = date.strftime('%Y-%m-%d')
            elif isinstance(date, str) and len(date) == 10 and date[4] == '-' and date[7] == '-':
                date_str = date
            else:
                logger.error(f"{ticker}: Invalid date format: {date} (type: {type(date)})")
                continue  # Skip this record

            records.append((
                ticker,
                date_str,  # Always store as string in YYYY-MM-DD format
                self._sanitize_value(row.get('Open', row.get('open', None))),
                self._sanitize_value(row.get('High', row.get('high', None))),
                self._sanitize_value(row.get('Low', row.get('low', None))),
                self._sanitize_value(row.get('Close', row.get('close'))),
                self._sanitize_value(row.get('Volume', row.get('volume', None)))
            ))

        try:
            # Insert or update records using UPSERT logic
            upsert_query = """
                INSERT INTO daily_prices
                (ticker, date, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (ticker, date) DO UPDATE SET
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close,
                    volume = excluded.volume,
                    updated_at = CURRENT_TIMESTAMP
            """
            self.conn.executemany(upsert_query, records)

            # Update metadata - only if we have valid records
            if records:
                first_date = df.index.min().strftime('%Y-%m-%d') if isinstance(df.index.min(), pd.Timestamp) else str(df.index.min())
                last_date = df.index.max().strftime('%Y-%m-%d') if isinstance(df.index.max(), pd.Timestamp) else str(df.index.max())

                self.conn.execute("""
                    INSERT INTO ticker_metadata
                    (ticker, first_date, last_date, total_records, last_update)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT (ticker) DO UPDATE SET
                        first_date = excluded.first_date,
                        last_date = excluded.last_date,
                        total_records = excluded.total_records,
                        last_update = CURRENT_TIMESTAMP
                """, (ticker, first_date, last_date, len(records)))

            self.conn.commit()
            return len(records)
        except Exception as e:
            logger.error(f"{ticker}: Error storing daily prices: {e}")
            # Roll back the transaction to clear the aborted state
            self.conn.rollback()
            raise
    
    def get_daily_prices(self, ticker: str, 
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        limit: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieve daily prices for a ticker
        """
        query = "SELECT date, open, high, low, close, volume FROM daily_prices WHERE ticker = ?"
        params = [ticker]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date.date() if isinstance(start_date, datetime) else start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date.date() if isinstance(end_date, datetime) else end_date)
        
        query += " ORDER BY date"
        
        if limit:
            query += f" DESC LIMIT {limit}"
            # Need to re-sort after limit
            query = f"SELECT * FROM ({query}) AS limited ORDER BY date"

        df = self._read_sql(query, params=params, parse_dates=['date'])
        
        if not df.empty:
            df.set_index('date', inplace=True)
            # Rename columns to match expected format
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        return df
    
    def store_weekly_prices(self, ticker: str, df: pd.DataFrame) -> int:
        """
        Store pre-computed weekly prices
        """
        if df.empty:
            return 0

        # Ensure the DataFrame has a proper date index
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error(f"{ticker}: Weekly DataFrame index is not DatetimeIndex, type: {type(df.index)}")
            return 0

        records = []
        for date, row in df.iterrows():
            # Validate date before storing
            if isinstance(date, pd.Timestamp):
                date_str = date.strftime('%Y-%m-%d')
            elif isinstance(date, datetime):
                date_str = date.strftime('%Y-%m-%d')
            elif isinstance(date, str) and len(date) == 10 and date[4] == '-' and date[7] == '-':
                date_str = date
            else:
                logger.error(f"{ticker}: Invalid weekly date format: {date} (type: {type(date)})")
                continue  # Skip this record

            records.append((
                ticker,
                date_str,  # Always store as string in YYYY-MM-DD format
                self._sanitize_value(row.get('Open', row.get('open', None))),
                self._sanitize_value(row.get('High', row.get('high', None))),
                self._sanitize_value(row.get('Low', row.get('low', None))),
                self._sanitize_value(row.get('Close', row.get('close'))),
                self._sanitize_value(row.get('Volume', row.get('volume', None)))
            ))

        try:
            upsert_query = """
                INSERT INTO weekly_prices
                (ticker, week_ending, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (ticker, week_ending) DO UPDATE SET
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close,
                    volume = excluded.volume,
                    updated_at = CURRENT_TIMESTAMP
            """
            self.conn.executemany(upsert_query, records)

            self.conn.commit()
            return len(records)
        except Exception as e:
            logger.error(f"{ticker}: Error storing weekly prices: {e}")
            # Roll back the transaction to clear the aborted state
            self.conn.rollback()
            raise
    
    def get_weekly_prices(self, ticker: str,
                         start_date: Optional[datetime] = None,
                         end_date: Optional[datetime] = None,
                         limit: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieve pre-computed weekly prices
        """
        query = "SELECT week_ending, open, high, low, close, volume FROM weekly_prices WHERE ticker = ?"
        params = [ticker]
        
        if start_date:
            query += " AND week_ending >= ?"
            params.append(start_date.date() if isinstance(start_date, datetime) else start_date)
        
        if end_date:
            query += " AND week_ending <= ?"
            params.append(end_date.date() if isinstance(end_date, datetime) else end_date)
        
        query += " ORDER BY week_ending"
        
        if limit:
            query += f" DESC LIMIT {limit}"
            query = f"SELECT * FROM ({query}) AS limited ORDER BY week_ending"

        df = self._read_sql(query, params=params, parse_dates=['week_ending'])
        
        if not df.empty:
            df.set_index('week_ending', inplace=True)
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        return df

    def store_hourly_prices(self, ticker: str, df: pd.DataFrame) -> int:
        """
        Store hourly OHLC prices
        """
        if df.empty:
            return 0

        # Ensure the DataFrame has a proper datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error(f"{ticker}: Hourly DataFrame index is not DatetimeIndex, type: {type(df.index)}")
            return 0

        records = []
        for dt, row in df.iterrows():
            # Validate datetime before storing
            if isinstance(dt, pd.Timestamp):
                datetime_str = dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                logger.warning(f"{ticker}: Invalid datetime {dt}, skipping")
                continue

            records.append((
                ticker,
                datetime_str,
                self._sanitize_value(row.get('Open', row.get('open', None))),
                self._sanitize_value(row.get('High', row.get('high', None))),
                self._sanitize_value(row.get('Low', row.get('low', None))),
                self._sanitize_value(row.get('Close', row.get('close'))),
                self._sanitize_value(row.get('Volume', row.get('volume', None)))
            ))

        try:
            upsert_query = """
                INSERT INTO hourly_prices
                (ticker, datetime, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (ticker, datetime) DO UPDATE SET
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close,
                    volume = excluded.volume,
                    updated_at = CURRENT_TIMESTAMP
            """
            self.conn.executemany(upsert_query, records)

            self.conn.commit()
            return len(records)
        except Exception as e:
            logger.error(f"{ticker}: Error storing hourly prices: {e}")
            # Roll back the transaction to clear the aborted state
            self.conn.rollback()
            raise

    def get_hourly_prices(self, ticker: str,
                         start_datetime: Optional[datetime] = None,
                         end_datetime: Optional[datetime] = None,
                         limit: Optional[int] = None) -> pd.DataFrame:
        """
        Retrieve hourly prices
        """
        query = "SELECT datetime, open, high, low, close, volume FROM hourly_prices WHERE ticker = ?"
        params = [ticker]

        if start_datetime:
            query += " AND datetime >= ?"
            params.append(start_datetime.strftime('%Y-%m-%d %H:%M:%S') if isinstance(start_datetime, datetime) else start_datetime)

        if end_datetime:
            query += " AND datetime <= ?"
            params.append(end_datetime.strftime('%Y-%m-%d %H:%M:%S') if isinstance(end_datetime, datetime) else end_datetime)

        query += " ORDER BY datetime"

        if limit:
            query += f" DESC LIMIT {limit}"
            query = f"SELECT * FROM ({query}) AS limited ORDER BY datetime"

        df = self._read_sql(query, params=params, parse_dates=['datetime'])

        if not df.empty:
            df.set_index('datetime', inplace=True)
            df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        return df

    def needs_update(self, ticker: str, force_after_close: bool = False) -> Tuple[bool, Optional[datetime]]:
        """
        Check if ticker needs updating
        Always update if data is more than 1 day old for active trading days
        
        Args:
            ticker: The ticker symbol to check
            force_after_close: If True, force update for post-market close checks
        """
        result = self.conn.execute("""
            SELECT last_date, last_update 
            FROM ticker_metadata 
            WHERE ticker = ?
        """, (ticker,)).fetchone()
        
        if not result:
            return True, None
        
        last_date, last_update = result
        last_date = _drop_tz(last_date)
        last_update_time = _drop_tz(last_update)
        now = _drop_tz(pd.Timestamp.utcnow())
        today = now.normalize()
        if pd.isna(last_date):
            return True, None
        if pd.isna(last_update_time):
            last_update_time = None

        # Calculate days since last update
        days_since_update = (today - last_date).days

        # Force update if requested (for post-market close checks)
        if force_after_close:
            # For same-day updates, check if we have today's closing data
            # Allow re-fetching if last update was more than 30 minutes ago
            # This ensures we get fresh closing prices after market close
            if last_date >= today:
                if last_update_time is None:
                    return True, None
                minutes_since_update = (now - last_update_time).total_seconds() / 60
                if minutes_since_update < 30:
                    return False, last_update_time
            return True, last_update_time

        # For weekends, use Friday's date
        if today.weekday() >= 5:  # Saturday = 5, Sunday = 6
            # If it's weekend, check if we have Friday's data
            last_friday = today - pd.Timedelta(days=(today.weekday() - 4))
            if last_date < last_friday:
                return True, last_update_time
        else:
            # For weekdays, need update if data is more than 1 day old
            # This ensures we always try to get the latest data regardless of market hours
            if days_since_update > 1:
                return True, last_update_time
            # Also update if it's a new trading day (even if only 1 day old)
            elif days_since_update == 1:
                # Always try to update for current day after any market might have closed
                return True, last_update_time

        return False, last_update_time
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        stats = {}
        
        # Total tickers
        stats['total_tickers'] = self.conn.execute(
            "SELECT COUNT(DISTINCT ticker) FROM daily_prices"
        ).fetchone()[0]
        
        # Total records
        stats['total_daily_records'] = self.conn.execute(
            "SELECT COUNT(*) FROM daily_prices"
        ).fetchone()[0]
        
        stats['total_weekly_records'] = self.conn.execute(
            "SELECT COUNT(*) FROM weekly_prices"
        ).fetchone()[0]
        
        # Date range
        date_range = self.conn.execute("""
            SELECT MIN(date), MAX(date) 
            FROM daily_prices
        """).fetchone()
        
        if date_range[0]:
            stats['earliest_date'] = date_range[0]
            stats['latest_date'] = date_range[1]
        
        # Database size (only available for local SQLite)
        if not self.use_postgres and os.path.exists(self.db_path):
            stats['db_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024)
        else:
            stats['db_size_mb'] = None
        
        return stats
    
    def close(self):
        """Close database connection"""
        if self.conn:
            # Use db_config to properly close and checkpoint WAL
            db_config.close_connection(self.conn)
            self.conn = None


class DailyPriceCollector:
    """
    Collects daily price data for all tickers in alerts
    """
    
    def __init__(self):
        self.db = DailyPriceDatabase()
        self.fetcher = FMPDataFetcher()
        self.stats = {
            'updated': 0,
            'skipped': 0,
            'failed': 0,
            'new': 0,
            'weekly_updated': 0  # Track weekly price updates
        }
    
    def get_all_tickers(self) -> List[str]:
        """Get unique tickers from metadata and alert repositories."""
        tickers = set(fetch_stock_metadata_map().keys())
        if tickers:
            logger.info(f"Found {len(tickers)} tickers from stock metadata")

        try:
            alerts = repo_list_alerts()
        except Exception as exc:
            logger.error(f"Error loading alerts from repository: {exc}")
            alerts = []

        for alert in alerts:
            ticker = alert.get('ticker')
            if ticker:
                tickers.add(ticker)

            if (alert.get('ratio') == 'Yes') or alert.get('is_ratio'):
                ticker1 = alert.get('ticker1')
                ticker2 = alert.get('ticker2')
                if ticker1:
                    tickers.add(ticker1)
                if ticker2:
                    tickers.add(ticker2)

        return sorted(tickers)
    
    def update_ticker(self, ticker: str) -> bool:
        """
        Update daily prices for a single ticker
        """
        try:
            # Check if update needed
            needs_update, last_update = self.db.needs_update(ticker)
            
            if not needs_update:
                logger.debug(f"{ticker}: Up to date (last: {last_update})")
                self.stats['skipped'] += 1
                return True
            
            # Fetch latest data
            logger.info(f"Updating {ticker}...")
            df = self.fetcher.get_historical_data(ticker, period="1day", timeframe="1d")
            
            if df is None or df.empty:
                logger.warning(f"{ticker}: No data received")
                self.stats['failed'] += 1
                return False
            
            # Store daily data
            records = self.db.store_daily_prices(ticker, df)
            
            # Compute and store weekly data
            weekly_df = self._resample_to_weekly(df, ticker)
            if weekly_df is not None:
                weekly_records = self.db.store_weekly_prices(ticker, weekly_df)
                if weekly_records > 0:
                    self.stats['weekly_updated'] += 1
            
            if last_update:
                self.stats['updated'] += 1
            else:
                self.stats['new'] += 1
            
            logger.info(f"{ticker}: Stored {records} daily records")
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

            # Make sure we have a datetime index
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)

            # Create a copy to avoid modifying the original
            df_copy = df.copy()

            # Add week identifier (year-week number)
            # Use %U for week starting Sunday (0-53) or %W for week starting Monday
            df_copy['year_week'] = df_copy.index.strftime('%Y-%U')

            # Group by week and aggregate
            weekly_data = []
            for year_week, week_df in df_copy.groupby('year_week'):
                if week_df.empty:
                    continue

                # Use the ACTUAL last trading day of the week as the date
                # This ensures we use Thursday if there's no Friday trading
                last_trading_day = week_df.index[-1]

                # Calculate weekly OHLCV using actual trading days
                weekly_record = {
                    'date': last_trading_day,  # Use actual last trading day
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

            logger.debug(f"{ticker}: Resampled to {len(weekly_df)} weekly records using actual trading dates")
            return weekly_df

        except Exception as e:
            logger.error(f"Error resampling {ticker}: {e}")
            return None
    
    def update_all(self, batch_size: int = 50):
        """
        Update all tickers in batches
        """
        tickers = self.get_all_tickers()
        total = len(tickers)
        
        logger.info(f"Starting update for {total} tickers")
        start_time = time_module.time()
        
        for i in range(0, total, batch_size):
            batch = tickers[i:i+batch_size]
            batch_end = min(i+batch_size, total)
            
            logger.info(f"Processing batch {i//batch_size + 1} ({i+1}-{batch_end}/{total})")
            
            for ticker in batch:
                self.update_ticker(ticker)
                
                # Rate limiting
                time_module.sleep(0.2)  # 5 requests per second
            
            # Progress update
            elapsed = time_module.time() - start_time
            rate = (i + len(batch)) / elapsed if elapsed > 0 else 0
            remaining = (total - (i + len(batch))) / rate if rate > 0 else 0
            
            logger.info(f"Progress: Updated={self.stats['updated']}, "
                       f"New={self.stats['new']}, "
                       f"Skipped={self.stats['skipped']}, "
                       f"Failed={self.stats['failed']}")
            logger.info(f"Est. remaining: {remaining/60:.1f} minutes")
        
        # Final statistics
        elapsed = time_module.time() - start_time
        logger.info(f"\nUpdate complete in {elapsed/60:.1f} minutes")
        logger.info(f"Final stats: {self.stats}")
        
        # Database statistics
        db_stats = self.db.get_statistics()
        logger.info(f"Database: {db_stats['total_tickers']} tickers, "
                   f"{db_stats['total_daily_records']:,} daily records, "
                   f"{db_stats['total_weekly_records']:,} weekly records, "
                   f"{db_stats['db_size_mb']:.1f} MB")
    
    def close(self):
        """Clean up"""
        self.db.close()


def run_daily_update():
    """
    Run the daily update process
    This should be scheduled to run after market close
    """
    logger.info("=" * 70)
    logger.info(f"Daily Price Update - {datetime.now()}")
    logger.info("=" * 70)
    
    collector = DailyPriceCollector()
    
    try:
        # Check if it's the right time to run
        now = datetime.now()
        market_close = time(16, 30)  # 4:30 PM ET
        
        if now.time() < market_close and now.weekday() < 5:
            logger.warning("Market is still open, skipping update")
            return
        
        # Run update
        collector.update_all()
        
    finally:
        collector.close()


def test_retrieval():
    """Test data retrieval"""
    db = DailyPriceDatabase()
    
    try:
        # Test daily data retrieval
        ticker = "AAPL"
        print(f"\nTesting retrieval for {ticker}")
        
        # Get last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        daily_df = db.get_daily_prices(ticker, start_date, end_date)
        print(f"Daily data: {len(daily_df)} records")
        if not daily_df.empty:
            print(daily_df.tail())
        
        # Get weekly data
        weekly_df = db.get_weekly_prices(ticker, start_date, end_date)
        print(f"\nWeekly data: {len(weekly_df)} records")
        if not weekly_df.empty:
            print(weekly_df.tail())
        
        # Check if needs update
        needs_update, last_update = db.needs_update(ticker)
        print(f"\nNeeds update: {needs_update}")
        print(f"Last update: {last_update}")
        
    finally:
        db.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_retrieval()
    else:
        run_daily_update()

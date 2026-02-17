"""
Data access layer for daily, weekly, and hourly price storage and retrieval.
Replaces the legacy DailyPriceDatabase with a repository in src.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.data_access.db_config import db_config

logger = logging.getLogger(__name__)


def _drop_tz(ts: Any) -> Any:
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


class DailyPriceRepository:
    """
    Repository for daily, weekly, and hourly price data.
    Optimized for time-series queries and indicator calculations.
    """

    def __init__(self, db_path: str = "price_data.db") -> None:
        self.db_path = db_path
        self.conn: Any = None
        self.use_postgres = db_config.db_type == "postgresql"
        self._sqlalchemy_engine: Any = None
        self.initialize_database()

    def _get_sqlalchemy_engine(self):
        """Get SQLAlchemy engine for pandas operations (avoids UserWarning)."""
        if self._sqlalchemy_engine is not None:
            return self._sqlalchemy_engine
        if self.use_postgres:
            self._sqlalchemy_engine = db_config.get_sqlalchemy_engine()
        else:
            from sqlalchemy import create_engine
            self._sqlalchemy_engine = create_engine(f"sqlite:///{self.db_path}")
        return self._sqlalchemy_engine

    def _read_sql(
        self,
        query: str,
        params: Optional[List[Any]] = None,
        parse_dates: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Execute SQL query using SQLAlchemy engine.
        Handles both ? (SQLite) and %s (PostgreSQL) placeholders.
        """
        engine = self._get_sqlalchemy_engine()
        if params:
            query_modified = query
            param_count = query.count("?")
            if param_count > 0:
                param_names = [f"p{i}" for i in range(param_count)]
                for name in param_names:
                    query_modified = query_modified.replace("?", f":{name}", 1)
                params_dict = dict(zip(param_names, params))
                with engine.connect() as conn:
                    return pd.read_sql_query(
                        text(query_modified),
                        conn,
                        params=params_dict,
                        parse_dates=parse_dates or [],
                    )
        return pd.read_sql_query(query, engine, parse_dates=parse_dates or [])

    @staticmethod
    def _sanitize_value(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (pd.Timestamp, datetime)):
            return value
        if isinstance(value, (np.generic,)):
            return value.item()
        if pd.isna(value):
            return None
        return value

    def initialize_database(self) -> None:
        """Create database tables if they don't exist."""
        if self.use_postgres:
            self.conn = db_config.get_connection(role="price_data")
        else:
            self.conn = db_config.get_connection(db_path=self.db_path)

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
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ticker_date
            ON daily_prices(ticker, date DESC)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_date
            ON daily_prices(date DESC)
        """)
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
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_hourly_ticker_datetime
            ON hourly_prices(ticker, datetime DESC)
        """)
        self.conn.commit()

    def store_daily_prices(self, ticker: str, df: pd.DataFrame) -> int:
        """
        Store or update daily prices for a ticker.
        Returns number of records inserted/updated.
        """
        if df.empty:
            return 0
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error(
                "%s: DataFrame index is not DatetimeIndex, type: %s",
                ticker,
                type(df.index),
            )
            return 0

        records: List[Tuple[Any, ...]] = []
        for date, row in df.iterrows():
            if isinstance(date, pd.Timestamp):
                date_str = date.strftime("%Y-%m-%d")
            elif isinstance(date, datetime):
                date_str = date.strftime("%Y-%m-%d")
            elif isinstance(date, str) and len(date) == 10 and date[4] == "-" and date[7] == "-":
                date_str = date
            else:
                logger.error("%s: Invalid date format: %s (type: %s)", ticker, date, type(date))
                continue
            records.append((
                ticker,
                date_str,
                self._sanitize_value(row.get("Open", row.get("open", None))),
                self._sanitize_value(row.get("High", row.get("high", None))),
                self._sanitize_value(row.get("Low", row.get("low", None))),
                self._sanitize_value(row.get("Close", row.get("close"))),
                self._sanitize_value(row.get("Volume", row.get("volume", None))),
            ))

        try:
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
            if records:
                first_date = (
                    df.index.min().strftime("%Y-%m-%d")
                    if isinstance(df.index.min(), pd.Timestamp)
                    else str(df.index.min())
                )
                last_date = (
                    df.index.max().strftime("%Y-%m-%d")
                    if isinstance(df.index.max(), pd.Timestamp)
                    else str(df.index.max())
                )
                self.conn.execute(
                    """
                    INSERT INTO ticker_metadata
                    (ticker, first_date, last_date, total_records, last_update)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT (ticker) DO UPDATE SET
                        first_date = excluded.first_date,
                        last_date = excluded.last_date,
                        total_records = excluded.total_records,
                        last_update = CURRENT_TIMESTAMP
                    """,
                    (ticker, first_date, last_date, len(records)),
                )
            self.conn.commit()
            return len(records)
        except Exception as e:
            logger.error("%s: Error storing daily prices: %s", ticker, e)
            self.conn.rollback()
            raise

    def get_daily_prices(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Retrieve daily prices for a ticker."""
        query = "SELECT date, open, high, low, close, volume FROM daily_prices WHERE ticker = ?"
        params: List[Any] = [ticker]
        if start_date:
            query += " AND date >= ?"
            params.append(start_date.date() if isinstance(start_date, datetime) else start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date.date() if isinstance(end_date, datetime) else end_date)
        query += " ORDER BY date"
        if limit:
            query += f" DESC LIMIT {limit}"
            query = f"SELECT * FROM ({query}) AS limited ORDER BY date"
        df = self._read_sql(query, params=params, parse_dates=["date"])
        if not df.empty:
            df.set_index("date", inplace=True)
            df.columns = ["Open", "High", "Low", "Close", "Volume"]
        return df

    def store_weekly_prices(self, ticker: str, df: pd.DataFrame) -> int:
        """Store pre-computed weekly prices."""
        if df.empty:
            return 0
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error(
                "%s: Weekly DataFrame index is not DatetimeIndex, type: %s",
                ticker,
                type(df.index),
            )
            return 0
        records = []
        for date, row in df.iterrows():
            if isinstance(date, pd.Timestamp):
                date_str = date.strftime("%Y-%m-%d")
            elif isinstance(date, datetime):
                date_str = date.strftime("%Y-%m-%d")
            elif isinstance(date, str) and len(date) == 10 and date[4] == "-" and date[7] == "-":
                date_str = date
            else:
                logger.error("%s: Invalid weekly date format: %s (type: %s)", ticker, date, type(date))
                continue
            records.append((
                ticker,
                date_str,
                self._sanitize_value(row.get("Open", row.get("open", None))),
                self._sanitize_value(row.get("High", row.get("high", None))),
                self._sanitize_value(row.get("Low", row.get("low", None))),
                self._sanitize_value(row.get("Close", row.get("close"))),
                self._sanitize_value(row.get("Volume", row.get("volume", None))),
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
            logger.error("%s: Error storing weekly prices: %s", ticker, e)
            self.conn.rollback()
            raise

    def get_weekly_prices(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Retrieve pre-computed weekly prices."""
        query = "SELECT week_ending, open, high, low, close, volume FROM weekly_prices WHERE ticker = ?"
        params: List[Any] = [ticker]
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
        df = self._read_sql(query, params=params, parse_dates=["week_ending"])
        if not df.empty:
            df.set_index("week_ending", inplace=True)
            df.columns = ["Open", "High", "Low", "Close", "Volume"]
        return df

    def store_hourly_prices(self, ticker: str, df: pd.DataFrame) -> int:
        """Store hourly OHLC prices."""
        if df.empty:
            return 0
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error(
                "%s: Hourly DataFrame index is not DatetimeIndex, type: %s",
                ticker,
                type(df.index),
            )
            return 0
        records = []
        for dt, row in df.iterrows():
            if isinstance(dt, pd.Timestamp):
                datetime_str = dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                logger.warning("%s: Invalid datetime %s, skipping", ticker, dt)
                continue
            records.append((
                ticker,
                datetime_str,
                self._sanitize_value(row.get("Open", row.get("open", None))),
                self._sanitize_value(row.get("High", row.get("high", None))),
                self._sanitize_value(row.get("Low", row.get("low", None))),
                self._sanitize_value(row.get("Close", row.get("close"))),
                self._sanitize_value(row.get("Volume", row.get("volume", None))),
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
            logger.error("%s: Error storing hourly prices: %s", ticker, e)
            self.conn.rollback()
            raise

    def get_hourly_prices(
        self,
        ticker: str,
        start_datetime: Optional[datetime] = None,
        end_datetime: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """Retrieve hourly prices."""
        query = "SELECT datetime, open, high, low, close, volume FROM hourly_prices WHERE ticker = ?"
        params: List[Any] = [ticker]
        if start_datetime:
            query += " AND datetime >= ?"
            params.append(
                start_datetime.strftime("%Y-%m-%d %H:%M:%S")
                if isinstance(start_datetime, datetime)
                else start_datetime
            )
        if end_datetime:
            query += " AND datetime <= ?"
            params.append(
                end_datetime.strftime("%Y-%m-%d %H:%M:%S")
                if isinstance(end_datetime, datetime)
                else end_datetime
            )
        query += " ORDER BY datetime"
        if limit:
            query += f" DESC LIMIT {limit}"
            query = f"SELECT * FROM ({query}) AS limited ORDER BY datetime"
        df = self._read_sql(query, params=params, parse_dates=["datetime"])
        if not df.empty:
            df.set_index("datetime", inplace=True)
            df.columns = ["Open", "High", "Low", "Close", "Volume"]
        return df

    def get_daily_prices_batch(
        self,
        tickers: List[str],
        limit: int = 200,
    ) -> Dict[str, pd.DataFrame]:
        """Retrieve daily prices for multiple tickers in a single query.

        Args:
            tickers: List of ticker symbols to fetch.
            limit: Max rows per ticker (most recent). Defaults to 200.

        Returns:
            Dict mapping ticker to its OHLCV DataFrame (DatetimeIndex).
        """
        return self._batch_query(
            table="daily_prices",
            date_col="date",
            tickers=tickers,
            limit=limit,
        )

    def get_weekly_prices_batch(
        self,
        tickers: List[str],
        limit: int = 200,
    ) -> Dict[str, pd.DataFrame]:
        """Retrieve weekly prices for multiple tickers in a single query.

        Args:
            tickers: List of ticker symbols to fetch.
            limit: Max rows per ticker (most recent). Defaults to 200.

        Returns:
            Dict mapping ticker to its OHLCV DataFrame (DatetimeIndex).
        """
        return self._batch_query(
            table="weekly_prices",
            date_col="week_ending",
            tickers=tickers,
            limit=limit,
        )

    def get_hourly_prices_batch(
        self,
        tickers: List[str],
        limit: int = 200,
    ) -> Dict[str, pd.DataFrame]:
        """Retrieve hourly prices for multiple tickers in a single query.

        Args:
            tickers: List of ticker symbols to fetch.
            limit: Max rows per ticker (most recent). Defaults to 200.

        Returns:
            Dict mapping ticker to its OHLCV DataFrame (DatetimeIndex).
        """
        return self._batch_query(
            table="hourly_prices",
            date_col="datetime",
            tickers=tickers,
            limit=limit,
        )

    def _batch_query(
        self,
        table: str,
        date_col: str,
        tickers: List[str],
        limit: int,
        chunk_size: int = 500,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data for many tickers via chunked IN-queries.

        Args:
            table: DB table name.
            date_col: Name of the date/datetime column.
            tickers: Ticker symbols to fetch.
            limit: Max rows to keep per ticker (tail).
            chunk_size: Max tickers per SQL IN-clause.

        Returns:
            Dict mapping ticker -> DataFrame with DatetimeIndex and
            columns ["Open", "High", "Low", "Close", "Volume"].
        """
        if not tickers:
            return {}

        engine = self._get_sqlalchemy_engine()
        all_frames: List[pd.DataFrame] = []

        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i : i + chunk_size]
            param_names = [f"p{j}" for j in range(len(chunk))]
            placeholders = ", ".join(f":{n}" for n in param_names)
            query = (
                f"SELECT ticker, {date_col}, open, high, low, close, volume "
                f"FROM {table} "
                f"WHERE ticker IN ({placeholders}) "
                f"ORDER BY ticker, {date_col}"
            )
            params = dict(zip(param_names, chunk))
            try:
                with engine.connect() as conn:
                    df = pd.read_sql_query(
                        text(query), conn, params=params, parse_dates=[date_col]
                    )
                if not df.empty:
                    all_frames.append(df)
            except Exception as e:
                logger.error("Batch query failed for chunk of %d tickers: %s", len(chunk), e)

        if not all_frames:
            return {}

        combined = pd.concat(all_frames, ignore_index=True)
        result: Dict[str, pd.DataFrame] = {}
        for ticker, group in combined.groupby("ticker"):
            group = group.sort_values(date_col)
            if limit and len(group) > limit:
                group = group.tail(limit)
            group = group.drop(columns=["ticker"]).set_index(date_col)
            group.columns = ["Open", "High", "Low", "Close", "Volume"]
            result[ticker] = group

        return result

    def needs_update(
        self, ticker: str, force_after_close: bool = False
    ) -> Tuple[bool, Optional[datetime]]:
        """
        Check if ticker needs updating.
        Always update if data is more than 1 day old for active trading days.
        """
        result = self.conn.execute(
            """
            SELECT last_date, last_update
            FROM ticker_metadata
            WHERE ticker = ?
            """,
            (ticker,),
        ).fetchone()

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

        days_since_update = (today - last_date).days

        if force_after_close:
            if last_date >= today:
                if last_update_time is None:
                    return True, None
                minutes_since_update = (now - last_update_time).total_seconds() / 60
                if minutes_since_update < 30:
                    return False, last_update_time
            return True, last_update_time

        if today.weekday() >= 5:
            last_friday = today - pd.Timedelta(days=(today.weekday() - 4))
            if last_date < last_friday:
                return True, last_update_time
        else:
            if days_since_update > 1:
                return True, last_update_time
            if days_since_update == 1:
                return True, last_update_time

        return False, last_update_time

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats: Dict[str, Any] = {}
        stats["total_tickers"] = self.conn.execute(
            "SELECT COUNT(DISTINCT ticker) FROM daily_prices"
        ).fetchone()[0]
        stats["total_daily_records"] = self.conn.execute(
            "SELECT COUNT(*) FROM daily_prices"
        ).fetchone()[0]
        stats["total_weekly_records"] = self.conn.execute(
            "SELECT COUNT(*) FROM weekly_prices"
        ).fetchone()[0]
        date_range = self.conn.execute(
            "SELECT MIN(date), MAX(date) FROM daily_prices"
        ).fetchone()
        if date_range[0]:
            stats["earliest_date"] = date_range[0]
            stats["latest_date"] = date_range[1]
        if not self.use_postgres and os.path.exists(self.db_path):
            stats["db_size_mb"] = os.path.getsize(self.db_path) / (1024 * 1024)
        else:
            stats["db_size_mb"] = None
        return stats

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            db_config.close_connection(self.conn)
            self.conn = None

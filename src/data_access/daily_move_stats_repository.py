"""
Data access layer for daily_move_stats table operations.
"""

from __future__ import annotations

from typing import List, Tuple

import pandas as pd

TABLE_NAME = "daily_move_stats"


def ensure_table(conn) -> None:
    """Create daily_move_stats table if it does not exist."""
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                ticker TEXT NOT NULL,
                date DATE NOT NULL,
                pct_change DOUBLE PRECISION NOT NULL,
                mean_change DOUBLE PRECISION NOT NULL,
                std_change DOUBLE PRECISION NOT NULL,
                zscore DOUBLE PRECISION NOT NULL,
                sigma_level INTEGER NOT NULL,
                direction TEXT NOT NULL,
                magnitude TEXT NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                PRIMARY KEY (ticker, date)
            )
            """
        )
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_sigma ON {TABLE_NAME} (sigma_level)"
        )
        cursor.execute(
            f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_date ON {TABLE_NAME} (date)"
        )
    conn.commit()


def fetch_daily_prices(conn, tickers: List[str]) -> pd.DataFrame:
    """
    Fetch daily prices for the specified tickers.

    Args:
        conn: Database connection.
        tickers: List of ticker symbols.

    Returns:
        DataFrame with columns: ticker, date, close.
    """
    if not tickers:
        return pd.DataFrame()

    placeholders = ",".join(["%s"] * len(tickers))
    query = (
        f"SELECT ticker, date, close FROM daily_prices "
        f"WHERE ticker IN ({placeholders}) ORDER BY ticker, date"
    )
    return pd.read_sql_query(query, conn, params=tuple(tickers), parse_dates=["date"])


def delete_stats_for_tickers(
    conn, tickers: List[str], min_date: str
) -> None:
    """
    Delete stats for specified tickers from a given date onward.

    Args:
        conn: Database connection.
        tickers: List of ticker symbols.
        min_date: Minimum date string (YYYY-MM-DD) from which to delete.
    """
    with conn.cursor() as cursor:
        cursor.executemany(
            f"DELETE FROM {TABLE_NAME} WHERE ticker = %s AND date >= %s",
            [(ticker, min_date) for ticker in tickers],
        )


def delete_stats_before_date(conn, cutoff_date: str) -> None:
    """
    Delete stats older than the cutoff date.

    Args:
        conn: Database connection.
        cutoff_date: Date string (YYYY-MM-DD) before which to delete.
    """
    with conn.cursor() as cursor:
        cursor.execute(
            f"DELETE FROM {TABLE_NAME} WHERE date < %s",
            (cutoff_date,),
        )


def upsert_stats(conn, rows: List[Tuple]) -> None:
    """
    Insert or update daily move stats rows.

    Args:
        conn: Database connection.
        rows: List of tuples containing:
            (ticker, date, pct_change, mean_change, std_change, zscore,
             sigma_level, direction, magnitude, updated_at)
    """
    with conn.cursor() as cursor:
        cursor.executemany(
            f"""
            INSERT INTO {TABLE_NAME} (
                ticker, date, pct_change, mean_change, std_change, zscore,
                sigma_level, direction, magnitude, updated_at
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (ticker, date) DO UPDATE SET
                pct_change = EXCLUDED.pct_change,
                mean_change = EXCLUDED.mean_change,
                std_change = EXCLUDED.std_change,
                zscore = EXCLUDED.zscore,
                sigma_level = EXCLUDED.sigma_level,
                direction = EXCLUDED.direction,
                magnitude = EXCLUDED.magnitude,
                updated_at = EXCLUDED.updated_at
            """,
            rows,
        )


def fetch_stats_by_date(conn, date: str) -> pd.DataFrame:
    """
    Fetch all stats for a specific date.

    Args:
        conn: Database connection.
        date: Date string (YYYY-MM-DD).

    Returns:
        DataFrame with all stats for the given date.
    """
    query = f"SELECT * FROM {TABLE_NAME} WHERE date = %s ORDER BY ticker"
    return pd.read_sql_query(query, conn, params=[date])


def fetch_stats_by_sigma(conn, sigma_level: int, limit: int = 100) -> pd.DataFrame:
    """
    Fetch stats filtered by minimum sigma level.

    Args:
        conn: Database connection.
        sigma_level: Minimum sigma level to filter by.
        limit: Maximum number of rows to return.

    Returns:
        DataFrame with stats at or above the sigma level.
    """
    query = (
        f"SELECT * FROM {TABLE_NAME} "
        f"WHERE sigma_level >= %s "
        f"ORDER BY date DESC, ABS(zscore) DESC "
        f"LIMIT %s"
    )
    return pd.read_sql_query(query, conn, params=[sigma_level, limit])

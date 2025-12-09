from datetime import datetime, timezone
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from data_access.metadata_repository import fetch_stock_metadata_map
from db_config import db_config

TABLE_NAME = "daily_move_stats"
HISTORY_WINDOW_DAYS = 120


def _load_tickers_for_exchanges(exchanges: Optional[Iterable[str]] = None) -> List[str]:
    """
    Load tickers for the specified exchanges from main_database_with_etfs.json.
    If exchanges is None, all tickers are returned.
    """
    data = fetch_stock_metadata_map() or {}

    if not exchanges:
        return list(data.keys())

    exchanges_upper = {ex.upper() for ex in exchanges if ex}

    tickers = [
        symbol
        for symbol, info in data.items()
        if isinstance(info, dict) and info.get("exchange", "").upper() in exchanges_upper
    ]
    return tickers


def _ensure_table(conn) -> None:
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


def _compute_ticker_stats(group: pd.DataFrame, history_window: int) -> pd.DataFrame:
    group = group.sort_values("date")
    group["pct_change"] = group["close"].pct_change() * 100.0
    group = group.dropna(subset=["pct_change"])
    if group.empty:
        return pd.DataFrame()

    group["mean_change"] = group["pct_change"].expanding().mean()
    std_series = group["pct_change"].expanding().std(ddof=0)
    group["std_change"] = std_series.fillna(0.0)

    denominators = group["std_change"].to_numpy()
    numerators = (group["pct_change"] - group["mean_change"]).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        zscores = np.where(np.abs(denominators) > 1e-9, numerators / denominators, 0.0)

    zscores = np.nan_to_num(zscores, nan=0.0, posinf=0.0, neginf=0.0)
    group["zscore"] = zscores
    sigma_levels = np.where(
        np.abs(zscores) >= 2,
        2,
        np.where(np.abs(zscores) >= 1, 1, 0),
    ).astype(int)
    group["sigma_level"] = sigma_levels
    group["direction"] = np.where(group["pct_change"] >= 0, "up", "down")
    group["magnitude"] = np.where(
        sigma_levels >= 2,
        "2σ",
        np.where(sigma_levels == 1, "1σ", "normal"),
    )
    group["updated_at"] = datetime.now(timezone.utc).isoformat()

    result = group[
        [
            "ticker",
            "date",
            "pct_change",
            "mean_change",
            "std_change",
            "zscore",
            "sigma_level",
            "direction",
            "magnitude",
            "updated_at",
        ]
    ]

    if history_window:
        result = result.tail(history_window)

    return result


def _calculate_stats_for_tickers(conn, tickers: List[str], history_window: int) -> pd.DataFrame:
    """Calculate daily percent-change statistics for the provided ticker list."""
    if not tickers:
        return pd.DataFrame()

    placeholders = ",".join(["%s"] * len(tickers))
    query = (
        f"SELECT ticker, date, close FROM daily_prices "
        f"WHERE ticker IN ({placeholders}) ORDER BY ticker, date"
    )
    prices = pd.read_sql_query(query, conn, params=tickers, parse_dates=["date"])

    if prices.empty:
        return pd.DataFrame()

    stats_frames = []
    for ticker, group in prices.groupby("ticker"):
        stats = _compute_ticker_stats(group, history_window)
        if not stats.empty:
            stats_frames.append(stats)

    if not stats_frames:
        return pd.DataFrame()

    result = pd.concat(stats_frames, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    return result


def update_daily_move_stats(exchanges: Optional[Iterable[str]] = None) -> int:
    """
    Compute one- and two-standard-deviation daily move stats and persist them.

    Args:
        exchanges: Optional iterable of exchange names to restrict computation.

    Returns:
        Number of tickers updated.
    """
    with db_config.connection(role="prices") as conn:
        _ensure_table(conn)

        tickers = _load_tickers_for_exchanges(exchanges)
        if not tickers:
            return 0

        stats_df = _calculate_stats_for_tickers(conn, tickers, HISTORY_WINDOW_DAYS)
        if stats_df.empty:
            return 0

        stats_df["date_str"] = stats_df["date"].dt.strftime("%Y-%m-%d")
        min_date = stats_df["date_str"].min()
        unique_tickers = stats_df["ticker"].unique()

        with conn.cursor() as cursor:
            cursor.executemany(
                f"DELETE FROM {TABLE_NAME} WHERE ticker = %s AND date >= %s",
                [(ticker, min_date) for ticker in unique_tickers],
            )

            cutoff_date = (
                (stats_df["date"].max() - pd.Timedelta(days=HISTORY_WINDOW_DAYS * 2))
                if not stats_df.empty
                else None
            )
            if cutoff_date is not None:
                cursor.execute(
                    f"DELETE FROM {TABLE_NAME} WHERE date < %s",
                    (cutoff_date.strftime("%Y-%m-%d"),),
                )

            rows = [
                (
                    row["ticker"],
                    row["date_str"],
                    row["pct_change"],
                    row["mean_change"],
                    row["std_change"],
                    row["zscore"],
                    row["sigma_level"],
                    row["direction"],
                    row["magnitude"],
                    row["updated_at"],
                )
                for row in stats_df.to_dict(orient="records")
            ]

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
        conn.commit()
        return len(unique_tickers)


if __name__ == "__main__":
    updated = update_daily_move_stats()
    print(f"Updated daily move stats for {updated} tickers.")

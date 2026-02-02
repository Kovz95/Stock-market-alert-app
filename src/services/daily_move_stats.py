from datetime import datetime, timezone
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from src.data_access.metadata_repository import fetch_stock_metadata_map
from src.data_access.db_config import db_config
from src.data_access.daily_move_stats_repository import (
    delete_stats_before_date,
    delete_stats_for_tickers,
    ensure_table,
    fetch_daily_prices,
    upsert_stats,
)

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
    if len(tickers) == 0:
        return pd.DataFrame()

    prices = fetch_daily_prices(conn, tickers)

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
        ensure_table(conn)

        tickers = _load_tickers_for_exchanges(exchanges)
        if not tickers:
            return 0

        stats_df = _calculate_stats_for_tickers(conn, tickers, HISTORY_WINDOW_DAYS)
        if stats_df.empty:
            return 0

        stats_df["date_str"] = stats_df["date"].dt.strftime("%Y-%m-%d")
        min_date = stats_df["date_str"].min()
        unique_tickers = list(stats_df["ticker"].unique())

        delete_stats_for_tickers(conn, unique_tickers, min_date)

        cutoff_date = (
            (stats_df["date"].max() - pd.Timedelta(days=HISTORY_WINDOW_DAYS * 2))
            if not stats_df.empty
            else None
        )
        if cutoff_date is not None:
            delete_stats_before_date(conn, cutoff_date.strftime("%Y-%m-%d"))

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

        upsert_stats(conn, rows)
        conn.commit()
        return len(unique_tickers)


if __name__ == "__main__":
    updated = update_daily_move_stats()
    print(f"Updated daily move stats for {updated} tickers.")

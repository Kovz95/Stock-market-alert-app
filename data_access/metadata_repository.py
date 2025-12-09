"""
Helpers for reading application metadata (stocks, alerts, portfolios) from Postgres.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Iterable, List

import pandas as pd

from db_config import db_config
from .alert_repository import list_alerts, refresh_alert_cache
from .portfolio_repository import list_portfolios as list_portfolios_map, _clear_cache as clear_portfolio_cache
from redis_support import build_key, delete_key, get_json, set_json

REDIS_STOCK_METADATA_KEY = build_key("stocks:metadata")
REDIS_PORTFOLIOS_KEY = build_key("portfolios:map")
REDIS_ALERTS_KEY = build_key("alerts:list")


def _query_dataframe(sql: str, params: Iterable[Any] | None = None) -> pd.DataFrame:
    conn = db_config.get_connection()
    try:
        df = pd.read_sql_query(sql, conn, params=params)
        return df
    finally:
        db_config.close_connection(conn)


def _normalise_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    serialised: List[Dict[str, Any]] = []
    for record in records:
        normalised = {}
        for key, value in record.items():
            if isinstance(value, pd.Timestamp):
                normalised[key] = value.to_pydatetime().isoformat()
            elif isinstance(value, (pd.Series, pd.DataFrame)):
                normalised[key] = value.to_dict()
            else:
                normalised[key] = value
        serialised.append(normalised)
    return serialised


@lru_cache(maxsize=1)
def fetch_stock_metadata_df() -> pd.DataFrame:
    """
    Return stock metadata as a pandas DataFrame.
    """
    cached = get_json(REDIS_STOCK_METADATA_KEY)
    if isinstance(cached, dict):
        records = cached.get("records")
        if isinstance(records, list):
            df = pd.DataFrame(records)
            if not df.empty and "last_updated" in df.columns:
                df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
            return df

    df = _query_dataframe("SELECT * FROM stock_metadata")
    if not df.empty:
        if "last_updated" in df.columns:
            df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")
        set_json(
            REDIS_STOCK_METADATA_KEY,
            {"records": _normalise_records(df.to_dict(orient="records"))},
            ttl_seconds=600,
        )
    return df


@lru_cache(maxsize=1)
def fetch_stock_metadata_map() -> Dict[str, Dict[str, Any]]:
    """
    Return stock metadata keyed by symbol.
    """
    df = fetch_stock_metadata_df()
    if df.empty:
        return {}
    records = df.to_dict(orient="records")
    return {record["symbol"]: record for record in records}


@lru_cache(maxsize=1)
def fetch_alerts_list() -> List[Dict[str, Any]]:
    cached = get_json(REDIS_ALERTS_KEY)
    if isinstance(cached, dict):
        items = cached.get("items")
        if isinstance(items, list):
            return items
    alerts = list_alerts()
    set_json(
        REDIS_ALERTS_KEY,
        {"items": alerts},
        ttl_seconds=300,
    )
    return alerts


@lru_cache(maxsize=1)
def fetch_alerts_df() -> pd.DataFrame:
    alerts = fetch_alerts_list()
    if not alerts:
        return pd.DataFrame()
    df = pd.DataFrame(alerts)
    if "last_triggered" in df.columns:
        df["last_triggered"] = pd.to_datetime(df["last_triggered"], errors="coerce")
    return df


@lru_cache(maxsize=1)
def fetch_portfolios() -> Dict[str, Dict[str, Any]]:
    cached = get_json(REDIS_PORTFOLIOS_KEY)
    if isinstance(cached, dict):
        payload = cached.get("items")
        if isinstance(payload, dict):
            return payload
    portfolios = list_portfolios_map()
    set_json(
        REDIS_PORTFOLIOS_KEY,
        {"items": portfolios},
        ttl_seconds=300,
    )
    return portfolios


def refresh_caches() -> None:
    """Clear cache entries (call after mutations)."""
    fetch_stock_metadata_df.cache_clear()
    fetch_stock_metadata_map.cache_clear()
    fetch_alerts_list.cache_clear()
    fetch_alerts_df.cache_clear()
    fetch_portfolios.cache_clear()
    refresh_alert_cache()
    clear_portfolio_cache()
    delete_key(REDIS_STOCK_METADATA_KEY)
    delete_key(REDIS_ALERTS_KEY)
    delete_key(REDIS_PORTFOLIOS_KEY)

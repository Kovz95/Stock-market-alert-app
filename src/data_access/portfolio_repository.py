"""
Portfolio repository utilities backed by PostgreSQL.
"""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

import pandas as pd
try:
    from psycopg2.extras import Json, execute_values
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "psycopg2 is required for portfolio repository access. Activate the "
        "project virtual environment (source venv/bin/activate) before running."
    ) from exc

from src.data_access.db_config import db_config
from src.data_access.redis_support import build_key, delete_key, get_json, set_json

REDIS_PORTFOLIO_KEY = build_key("portfolios:map")


def _normalize_stock_entry(entry: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
    if isinstance(entry, dict):
        symbol = entry.get("symbol") or entry.get("ticker") or entry.get("Ticker")
        if not symbol:
            return None
        payload = dict(entry)
        payload.setdefault("symbol", symbol)
        return symbol, payload
    if isinstance(entry, str):
        return entry, {"symbol": entry}
    return None


def _fetch_portfolio_frames() -> Tuple[pd.DataFrame, pd.DataFrame]:
    conn = db_config.get_connection()
    try:
        portfolios_df = pd.read_sql_query("SELECT * FROM portfolios", conn)
        links_df = pd.read_sql_query("SELECT * FROM portfolio_stocks", conn)
    finally:
        db_config.close_connection(conn)
    return portfolios_df, links_df


def _build_portfolio_map(portfolios_df: pd.DataFrame, links_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    portfolio_map: Dict[str, Dict[str, Any]] = {}
    for _, row in portfolios_df.iterrows():
        portfolio_id = row["id"]
        raw_payload = row.get("raw_payload") or {}
        if isinstance(raw_payload, dict):
            portfolio = raw_payload.copy()
            portfolio.setdefault("id", portfolio_id)
        else:
            portfolio = {
                "id": portfolio_id,
                "name": row.get("name"),
                "discord_webhook": row.get("discord_webhook"),
                "enabled": bool(row.get("enabled", True)),
                "created_date": row.get("created_date"),
                "last_updated": row.get("last_updated"),
                "stocks": [],
            }
        stocks = portfolio.get("stocks")
        if not isinstance(stocks, list):
            portfolio["stocks"] = []
        portfolio_map[portfolio_id] = portfolio

    if not links_df.empty:
        for _, row in links_df.iterrows():
            portfolio_id = row["portfolio_id"]
            ticker = row["ticker"]
            if portfolio_id in portfolio_map:
                portfolio_map[portfolio_id].setdefault("stocks", [])
                portfolio_map[portfolio_id]["stocks"].append({"symbol": ticker})

    return portfolio_map


@lru_cache(maxsize=1)
def list_portfolios_cached() -> Dict[str, Dict[str, Any]]:
    cached = get_json(REDIS_PORTFOLIO_KEY)
    if isinstance(cached, dict):
        payload = cached.get("items")
        if isinstance(payload, dict):
            return payload

    portfolios_df, links_df = _fetch_portfolio_frames()
    portfolio_map = _build_portfolio_map(portfolios_df, links_df)
    set_json(
        REDIS_PORTFOLIO_KEY,
        {"items": portfolio_map},
        ttl_seconds=300,
    )
    return portfolio_map


def list_portfolios() -> Dict[str, Dict[str, Any]]:
    return {k: dict(v) for k, v in list_portfolios_cached().items()}


def get_portfolio(portfolio_id: str) -> Optional[Dict[str, Any]]:
    return list_portfolios().get(portfolio_id)


def _clear_cache() -> None:
    list_portfolios_cached.cache_clear()
    delete_key(REDIS_PORTFOLIO_KEY)


def save_portfolio(portfolio: Dict[str, Any]) -> Dict[str, Any]:
    portfolio_id = portfolio.get("id") or str(uuid4())[:8]
    name = portfolio.get("name") or f"Portfolio {portfolio_id}"
    webhook = portfolio.get("discord_webhook")
    enabled = bool(portfolio.get("enabled", True))
    created = portfolio.get("created_date")
    updated = datetime.now()

    stocks_entries = portfolio.get("stocks", []) or []
    normalized_rows: List[Tuple[str, Dict[str, Any]]] = []
    for entry in stocks_entries:
        parsed = _normalize_stock_entry(entry)
        if parsed:
            normalized_rows.append(parsed)

    conn = db_config.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO portfolios (id, name, discord_webhook, enabled, created_date, last_updated, raw_payload)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    discord_webhook = EXCLUDED.discord_webhook,
                    enabled = EXCLUDED.enabled,
                    created_date = COALESCE(EXCLUDED.created_date, portfolios.created_date),
                    last_updated = EXCLUDED.last_updated,
                    raw_payload = EXCLUDED.raw_payload
                """,
                (
                    portfolio_id,
                    name,
                    webhook,
                    enabled,
                    created,
                    updated,
                    Json({**portfolio, "id": portfolio_id}),
                ),
            )

            cur.execute("DELETE FROM portfolio_stocks WHERE portfolio_id = %s", (portfolio_id,))
            if normalized_rows:
                execute_values(
                    cur,
                    "INSERT INTO portfolio_stocks (portfolio_id, ticker) VALUES %s",
                    [(portfolio_id, symbol) for symbol, _ in normalized_rows],
                )
        conn.commit()
    finally:
        db_config.close_connection(conn)

    _clear_cache()
    return get_portfolio(portfolio_id) or {
        "id": portfolio_id,
        "name": name,
        "stocks": [payload for _, payload in normalized_rows],
        "discord_webhook": webhook,
        "enabled": enabled,
        "created_date": created,
        "last_updated": updated.isoformat(),
    }


def delete_portfolio(portfolio_id: str) -> None:
    conn = db_config.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM portfolio_stocks WHERE portfolio_id = %s", (portfolio_id,))
            cur.execute("DELETE FROM portfolios WHERE id = %s", (portfolio_id,))
        conn.commit()
    finally:
        db_config.close_connection(conn)
    _clear_cache()


def bulk_replace_portfolios(portfolios: Iterable[Dict[str, Any]]) -> None:
    conn = db_config.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE portfolio_stocks")
            cur.execute("TRUNCATE TABLE portfolios")

            rows: List[Tuple[Any, ...]] = []
            stocks_rows: List[Tuple[str, str]] = []
            for entry in portfolios:
                portfolio_id = entry.get("id") or str(uuid4())[:8]
                created = entry.get("created_date")
                updated = entry.get("last_updated") or datetime.now().isoformat()
                rows.append(
                    (
                        portfolio_id,
                        entry.get("name"),
                        entry.get("discord_webhook"),
                        bool(entry.get("enabled", True)),
                        created,
                        updated,
                        Json({**entry, "id": portfolio_id}),
                    )
                )
                for stock_entry in entry.get("stocks", []) or []:
                    parsed = _normalize_stock_entry(stock_entry)
                    if parsed:
                        symbol, _ = parsed
                        stocks_rows.append((portfolio_id, symbol))

            if rows:
                execute_values(
                    cur,
                    """
                    INSERT INTO portfolios (id, name, discord_webhook, enabled, created_date, last_updated, raw_payload)
                    VALUES %s
                    """,
                    rows,
                )
            if stocks_rows:
                execute_values(
                    cur,
                    "INSERT INTO portfolio_stocks (portfolio_id, ticker) VALUES %s",
                    stocks_rows,
                )
        conn.commit()
    finally:
        db_config.close_connection(conn)
    _clear_cache()

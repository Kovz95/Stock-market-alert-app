#!/usr/bin/env python3
"""
Utility script to seed PostgreSQL with data that historically lived in JSON files.

Tables populated:
    - stock_metadata        (from main_database_with_etfs.json)
    - alerts                (from alerts.json)
    - portfolios / portfolio_stocks  (from portfolios.json)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from psycopg2.extras import Json, execute_values  # type: ignore

from db_config import db_config

STOCK_JSON_PATH = BASE_DIR / "main_database_with_etfs.json"
ALERTS_JSON_PATH = BASE_DIR / "alerts.json"
PORTFOLIOS_JSON_PATH = BASE_DIR / "portfolios.json"
DOCUMENT_PATHS = {
    "discord_channels_config": BASE_DIR / "discord_channels_config.json",
    "custom_discord_channels": BASE_DIR / "custom_discord_channels.json",
    "database_filters": BASE_DIR / "database_filters.json",
    "enhanced_fmp_ticker_mapping": BASE_DIR / "enhanced_fmp_ticker_mapping.json",
    "fmp_ticker_mapping": BASE_DIR / "fmp_ticker_mapping.json",
    "yahoo_ticker_mapping": BASE_DIR / "yahoo_ticker_mapping.json",
    "futures_alerts": BASE_DIR / "futures_alerts.json",
    "futures_database": BASE_DIR / "futures_database.json",
    "futures_scheduler_config": BASE_DIR / "futures_scheduler_config.json",
    "futures_scheduler_status": BASE_DIR / "futures_scheduler_status.json",
    "ib_futures_config": BASE_DIR / "ib_futures_config.json",
    "industry_filters": BASE_DIR / "industry_filters.json",
    "local_exchange_mappings": BASE_DIR / "local_exchange_mappings.json",
    "logging_config": BASE_DIR / "logging_config.json",
    "notifications": BASE_DIR / "notifications.json",
    "saved_scans": BASE_DIR / "saved_scans.json",
    "scheduler_config": BASE_DIR / "scheduler_config.json",
    "alert_check_results": BASE_DIR / "alert_check_results.json",
    "hourly_scheduler_status": BASE_DIR / "hourly_scheduler_status.json",
    "scheduler_status": BASE_DIR / "scheduler_status.json",
    "alerts_legacy": BASE_DIR / "alerts.json",
    "job_locks": BASE_DIR / "job_locks.json",
}


def _parse_float(value: Any) -> float | None:
    if value in ("", None, "N/A", "No", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def load_stock_metadata(cur) -> int:
    if not STOCK_JSON_PATH.exists():
        return 0

    data: Dict[str, Dict[str, Any]] = json.loads(STOCK_JSON_PATH.read_text(encoding="utf-8"))
    rows: List[Tuple[Any, ...]] = []
    for symbol, info in data.items():
        last_updated = _parse_datetime(info.get("last_updated"))
        rows.append(
            (
                symbol,
                info.get("isin"),
                info.get("name"),
                info.get("exchange"),
                info.get("country"),
                info.get("rbics_economy"),
                info.get("rbics_sector"),
                info.get("rbics_subsector"),
                info.get("rbics_industry_group"),
                info.get("rbics_industry"),
                info.get("rbics_subindustry"),
                _parse_float(info.get("closing_price")),
                _parse_float(info.get("market_value")),
                _parse_float(info.get("sales")),
                _parse_float(info.get("avg_daily_volume")),
                info.get("data_source"),
                last_updated,
                info.get("asset_type"),
                Json(info),
            )
        )

    if not rows:
        return 0

    query = """
        INSERT INTO stock_metadata (
            symbol, isin, name, exchange, country,
            rbics_economy, rbics_sector, rbics_subsector,
            rbics_industry_group, rbics_industry, rbics_subindustry,
            closing_price, market_value, sales, avg_daily_volume,
            data_source, last_updated, asset_type, raw_payload
        )
        VALUES %s
        ON CONFLICT (symbol) DO UPDATE SET
            isin = EXCLUDED.isin,
            name = EXCLUDED.name,
            exchange = EXCLUDED.exchange,
            country = EXCLUDED.country,
            rbics_economy = EXCLUDED.rbics_economy,
            rbics_sector = EXCLUDED.rbics_sector,
            rbics_subsector = EXCLUDED.rbics_subsector,
            rbics_industry_group = EXCLUDED.rbics_industry_group,
            rbics_industry = EXCLUDED.rbics_industry,
            rbics_subindustry = EXCLUDED.rbics_subindustry,
            closing_price = EXCLUDED.closing_price,
            market_value = EXCLUDED.market_value,
            sales = EXCLUDED.sales,
            avg_daily_volume = EXCLUDED.avg_daily_volume,
            data_source = EXCLUDED.data_source,
            last_updated = EXCLUDED.last_updated,
            asset_type = EXCLUDED.asset_type,
            raw_payload = EXCLUDED.raw_payload
    """
    execute_values(cur, query, rows)
    return len(rows)


def load_alerts(cur) -> int:
    if not ALERTS_JSON_PATH.exists():
        return 0

    alerts: Iterable[Dict[str, Any]] = json.loads(ALERTS_JSON_PATH.read_text(encoding="utf-8"))
    rows: List[Tuple[Any, ...]] = []
    for alert in alerts:
        last_triggered = _parse_datetime(alert.get("last_triggered"))
        ratio_flag = (alert.get("ratio") or "").lower() in {"yes", "true", "1"}
        rows.append(
            (
                alert.get("alert_id"),
                alert.get("name"),
                alert.get("stock_name"),
                alert.get("ticker"),
                alert.get("ticker1"),
                alert.get("ticker2"),
                Json(alert.get("conditions", [])),
                alert.get("combination_logic"),
                last_triggered,
                alert.get("action"),
                alert.get("timeframe"),
                alert.get("exchange"),
                alert.get("country"),
                alert.get("ratio"),
                ratio_flag,
                alert.get("adjustment_method"),
                Json(alert.get("dtp_params")) if alert.get("dtp_params") else None,
                Json(alert.get("multi_timeframe_params")) if alert.get("multi_timeframe_params") else None,
                Json(alert.get("mixed_timeframe_params")) if alert.get("mixed_timeframe_params") else None,
                Json(alert),
            )
        )

    if not rows:
        return 0

    query = """
        INSERT INTO alerts (
            alert_id, name, stock_name, ticker, ticker1, ticker2, conditions,
            combination_logic, last_triggered, action, timeframe,
            exchange, country, ratio, is_ratio, adjustment_method,
            dtp_params, multi_timeframe_params, mixed_timeframe_params, raw_payload
        )
        VALUES %s
        ON CONFLICT (alert_id) DO UPDATE SET
            name = EXCLUDED.name,
            stock_name = EXCLUDED.stock_name,
            ticker = EXCLUDED.ticker,
            ticker1 = EXCLUDED.ticker1,
            ticker2 = EXCLUDED.ticker2,
            conditions = EXCLUDED.conditions,
            combination_logic = EXCLUDED.combination_logic,
            last_triggered = EXCLUDED.last_triggered,
            action = EXCLUDED.action,
            timeframe = EXCLUDED.timeframe,
            exchange = EXCLUDED.exchange,
            country = EXCLUDED.country,
            ratio = EXCLUDED.ratio,
            is_ratio = EXCLUDED.is_ratio,
            adjustment_method = EXCLUDED.adjustment_method,
            dtp_params = EXCLUDED.dtp_params,
            multi_timeframe_params = EXCLUDED.multi_timeframe_params,
            mixed_timeframe_params = EXCLUDED.mixed_timeframe_params,
            raw_payload = EXCLUDED.raw_payload,
            updated_at = NOW()
    """
    execute_values(cur, query, rows)
    return len(rows)


def load_portfolios(cur) -> Tuple[int, int]:
    if not PORTFOLIOS_JSON_PATH.exists():
        return 0, 0

    payload = json.loads(PORTFOLIOS_JSON_PATH.read_text(encoding="utf-8"))
    portfolios = payload.get("portfolios", {})
    portfolio_rows: List[Tuple[Any, ...]] = []
    link_rows: List[Tuple[Any, Any]] = []

    for portfolio_id, info in portfolios.items():
        created = _parse_datetime(info.get("created_date"))
        updated = _parse_datetime(info.get("last_updated"))
        portfolio_rows.append(
            (
                portfolio_id,
                info.get("name"),
                info.get("discord_webhook"),
                bool(info.get("enabled", True)),
                created,
                updated,
                Json(info),
            )
        )
        for stock_info in info.get("stocks", []):
            symbol = None
            if isinstance(stock_info, dict):
                symbol = stock_info.get("symbol")
            elif isinstance(stock_info, str):
                symbol = stock_info
            if symbol:
                link_rows.append((portfolio_id, symbol))

    if portfolio_rows:
        query = """
            INSERT INTO portfolios (
                id, name, discord_webhook, enabled,
                created_date, last_updated, raw_payload
            )
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                discord_webhook = EXCLUDED.discord_webhook,
                enabled = EXCLUDED.enabled,
                created_date = COALESCE(EXCLUDED.created_date, portfolios.created_date),
                last_updated = EXCLUDED.last_updated,
                raw_payload = EXCLUDED.raw_payload
        """
        execute_values(cur, query, portfolio_rows)

    if link_rows:
        cur.execute("DELETE FROM portfolio_stocks")
        execute_values(
            cur,
            """
            INSERT INTO portfolio_stocks (portfolio_id, ticker)
            VALUES %s
            ON CONFLICT (portfolio_id, ticker) DO NOTHING
            """,
            link_rows,
        )

    return len(portfolio_rows), len(link_rows)


def load_documents(cur) -> int:
    """Load auxiliary JSON documents into the app_documents table."""
    inserted = 0
    for key, path in DOCUMENT_PATHS.items():
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        cur.execute(
            """
            INSERT INTO app_documents (document_key, payload, source_path, updated_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (document_key) DO UPDATE SET
                payload = EXCLUDED.payload,
                source_path = COALESCE(EXCLUDED.source_path, app_documents.source_path),
                updated_at = EXCLUDED.updated_at
            """,
            (key, Json(payload), str(path)),
        )
        inserted += 1
    return inserted


def main() -> None:
    conn = db_config.get_connection()
    try:
        cur = conn.cursor()
        inserted_stocks = load_stock_metadata(cur)
        inserted_alerts = load_alerts(cur)
        portfolios_count, portfolio_links = load_portfolios(cur)
        documents_loaded = load_documents(cur)
        conn.commit()

        print(f"Loaded {inserted_stocks} stock metadata rows.")
        print(f"Loaded {inserted_alerts} alerts.")
        print(f"Loaded {portfolios_count} portfolios with {portfolio_links} stock links.")
        print(f"Loaded {documents_loaded} auxiliary documents into app_documents.")
    except Exception:
        conn.rollback()
        raise
    finally:
        db_config.close_connection(conn)


if __name__ == "__main__":
    main()

"""Alert repository backed by PostgreSQL."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional
from uuid import uuid4

import pandas as pd
try:
    from psycopg2.extras import Json, execute_values  # type: ignore
except ImportError as exc:  # pragma: no cover - enforce virtualenv usage
    raise ImportError(
        "psycopg2 is required for alert repository access. Activate the "
        "project virtual environment (source venv/bin/activate) before running."
    ) from exc

from db_config import db_config
from redis_support import build_key, delete_key, get_json, set_json


ALERT_SELECT_COLUMNS = """
    SELECT
        alert_id,
        name,
        stock_name,
        ticker,
        ticker1,
        ticker2,
        conditions,
        combination_logic,
        last_triggered,
        action,
        timeframe,
        exchange,
        country,
        ratio,
        is_ratio,
        adjustment_method,
        dtp_params,
        multi_timeframe_params,
        mixed_timeframe_params,
        raw_payload
    FROM alerts
"""

REDIS_ALERT_CACHE_KEY = build_key("alerts:list")
REDIS_ALERT_SINGLE_PREFIX = build_key("alerts:item:")


def _parse_timestamp(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def _fetch_alert_dataframe() -> pd.DataFrame:
    conn = db_config.get_connection()
    try:
        df = pd.read_sql_query(f"{ALERT_SELECT_COLUMNS} ORDER BY updated_at DESC, name ASC", conn)
    finally:
        db_config.close_connection(conn)

    if not df.empty and "last_triggered" in df.columns:
        df["last_triggered"] = pd.to_datetime(df["last_triggered"])
    return df


def _normalize_alert(record: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(record)
    raw_payload = row.pop("raw_payload", None)
    payload: Dict[str, Any]
    if isinstance(raw_payload, dict):
        payload = {**raw_payload, **{k: v for k, v in row.items() if v is not None}}
    else:
        payload = row

    payload.setdefault("alert_id", row.get("alert_id"))
    payload["alert_id"] = str(payload.get("alert_id"))
    payload["conditions"] = payload.get("conditions") or []

    last_triggered = payload.get("last_triggered")
    if isinstance(last_triggered, pd.Timestamp):
        payload["last_triggered"] = last_triggered.to_pydatetime().isoformat()
    elif isinstance(last_triggered, datetime):
        payload["last_triggered"] = last_triggered.isoformat()
    elif not last_triggered:
        payload["last_triggered"] = ""
    else:
        payload["last_triggered"] = str(last_triggered)

    if payload.get("ratio") in (None, ""):
        payload["ratio"] = "Yes" if payload.get("is_ratio") else "No"

    return payload


def _prepare_payload(alert: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(alert)
    payload.setdefault("alert_id", str(payload.get("alert_id") or uuid4()))
    payload.setdefault("ratio", payload.get("ratio", "No"))
    # Normalise boolean flag
    if "is_ratio" not in payload:
        payload["is_ratio"] = str(payload.get("ratio", ""))
        payload["is_ratio"] = str(payload["is_ratio"]).lower() in {"yes", "true", "1"}
    else:
        payload["is_ratio"] = bool(payload["is_ratio"])
    return payload


def _row_from_payload(payload: Dict[str, Any]) -> tuple:
    last_triggered = _parse_timestamp(payload.get("last_triggered"))
    return (
        payload["alert_id"],
        payload.get("name"),
        payload.get("stock_name"),
        payload.get("ticker"),
        payload.get("ticker1"),
        payload.get("ticker2"),
        Json(payload.get("conditions", [])),
        payload.get("combination_logic"),
        last_triggered,
        payload.get("action"),
        payload.get("timeframe"),
        payload.get("exchange"),
        payload.get("country"),
        payload.get("ratio"),
        payload.get("is_ratio"),
        payload.get("adjustment_method"),
        Json(payload.get("dtp_params")) if payload.get("dtp_params") is not None else None,
        Json(payload.get("multi_timeframe_params")) if payload.get("multi_timeframe_params") is not None else None,
        Json(payload.get("mixed_timeframe_params")) if payload.get("mixed_timeframe_params") is not None else None,
        Json(payload),
    )


@lru_cache(maxsize=1)
def list_alerts_cached() -> List[Dict[str, Any]]:
    cached = get_json(REDIS_ALERT_CACHE_KEY)
    if isinstance(cached, dict):
        items = cached.get("items")
        if isinstance(items, list):
            return items

    df = _fetch_alert_dataframe()
    if df.empty:
        alerts: List[Dict[str, Any]] = []
    else:
        records = df.to_dict(orient="records")
        alerts = [_normalize_alert(rec) for rec in records]

    set_json(
        REDIS_ALERT_CACHE_KEY,
        {"items": alerts, "cached_at": datetime.utcnow().isoformat()},
        ttl_seconds=300,
    )
    return alerts


def list_alerts(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    alerts = list_alerts_cached()
    if limit is not None:
        return alerts[:limit]
    return alerts


def get_alert(alert_id: str) -> Optional[Dict[str, Any]]:
    for alert in list_alerts_cached():
        if alert["alert_id"] == str(alert_id):
            return dict(alert)
    return None


def _clear_cache() -> None:
    list_alerts_cached.cache_clear()
    delete_key(REDIS_ALERT_CACHE_KEY)


def refresh_alert_cache() -> None:
    _clear_cache()


def create_alert(alert: Dict[str, Any]) -> Dict[str, Any]:
    payload = _prepare_payload(alert)
    row = _row_from_payload(payload)

    conn = db_config.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO alerts (
                    alert_id,
                    name,
                    stock_name,
                    ticker,
                    ticker1,
                    ticker2,
                    conditions,
                    combination_logic,
                    last_triggered,
                    action,
                    timeframe,
                    exchange,
                    country,
                    ratio,
                    is_ratio,
                    adjustment_method,
                    dtp_params,
                    multi_timeframe_params,
                    mixed_timeframe_params,
                    raw_payload
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (alert_id) DO NOTHING
                """,
                row,
            )
        conn.commit()
    finally:
        db_config.close_connection(conn)

    _clear_cache()
    return get_alert(payload["alert_id"]) or payload


def update_alert(alert_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not updates:
        return get_alert(alert_id)

    existing = get_alert(alert_id)
    if not existing:
        return None

    payload = _prepare_payload({**existing, **updates, "alert_id": alert_id})
    row = _row_from_payload(payload)

    conn = db_config.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE alerts
                SET
                    name = %s,
                    stock_name = %s,
                    ticker = %s,
                    ticker1 = %s,
                    ticker2 = %s,
                    conditions = %s,
                    combination_logic = %s,
                    last_triggered = %s,
                    action = %s,
                    timeframe = %s,
                    exchange = %s,
                    country = %s,
                    ratio = %s,
                    is_ratio = %s,
                    adjustment_method = %s,
                    dtp_params = %s,
                    multi_timeframe_params = %s,
                    mixed_timeframe_params = %s,
                    raw_payload = %s,
                    updated_at = NOW()
                WHERE alert_id = %s
                """,
                (*row, alert_id),
            )
        conn.commit()
    finally:
        db_config.close_connection(conn)

    _clear_cache()
    return get_alert(alert_id)


def delete_alert(alert_id: str) -> None:
    conn = db_config.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM alerts WHERE alert_id = %s", (alert_id,))
        conn.commit()
    finally:
        db_config.close_connection(conn)
    _clear_cache()


def bulk_replace_alerts(alerts: Iterable[Dict[str, Any]]) -> None:
    payloads = [_prepare_payload(alert) for alert in alerts]
    rows = [_row_from_payload(payload) for payload in payloads]

    conn = db_config.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE alerts")
            if rows:
                execute_values(
                    cur,
                    """
                    INSERT INTO alerts (
                        alert_id,
                        name,
                        stock_name,
                        ticker,
                        ticker1,
                        ticker2,
                        conditions,
                        combination_logic,
                        last_triggered,
                        action,
                        timeframe,
                        exchange,
                        country,
                        ratio,
                        is_ratio,
                        adjustment_method,
                        dtp_params,
                        multi_timeframe_params,
                        mixed_timeframe_params,
                        raw_payload
                    )
                    VALUES %s
                    """,
                    rows,
                )
        conn.commit()
    finally:
        db_config.close_connection(conn)
    _clear_cache()

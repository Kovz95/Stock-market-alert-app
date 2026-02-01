"""
PostgreSQL-backed JSON document store with optional Redis caching.

This module replaces legacy JSON file persistence by providing a simple
key/value interface.  Documents are stored in the ``app_documents`` table
and mirrored to Redis when available for fast reads.
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    from psycopg2.extras import Json
except ImportError as exc:  # pragma: no cover - enforce virtualenv usage
    raise ImportError(
        "psycopg2 is required for document storage. Activate the project "
        "virtual environment (source venv/bin/activate) before running."
    ) from exc

from db_config import db_config
from redis_support import build_key, delete_key, get_json, get_client

try:
    from redis_support import set_json
except ImportError:  # pragma: no cover - fallback in case helpers move
    set_json = None  # type: ignore

DOCUMENT_CACHE_NAMESPACE = "document:"
DEFAULT_CACHE_TTL_SECONDS = 300


def _build_cache_key(document_key: str) -> str:
    return build_key(f"{DOCUMENT_CACHE_NAMESPACE}{document_key}")


def _fetch_document_from_db(document_key: str) -> Optional[Any]:
    conn = db_config.get_connection()
    try:
        cur = conn.cursor()
        try:
            cur.execute(
                "SELECT payload FROM app_documents WHERE document_key = %s",
                (document_key,),
            )
            row = cur.fetchone()
        finally:
            cur.close()
    finally:
        db_config.close_connection(conn)
    if not row:
        return None
    payload = row[0]
    return deepcopy(payload)


def _write_document_to_db(
    document_key: str,
    payload: Any,
    *,
    source_path: Optional[str] = None,
) -> None:
    conn = db_config.get_connection()
    try:
        cur = conn.cursor()
        try:
            cur.execute(
                """
                INSERT INTO app_documents (document_key, payload, source_path, updated_at)
                VALUES (%s, %s, %s, NOW())
                ON CONFLICT (document_key) DO UPDATE SET
                    payload = EXCLUDED.payload,
                    source_path = COALESCE(EXCLUDED.source_path, app_documents.source_path),
                    updated_at = EXCLUDED.updated_at
                """,
                (document_key, Json(payload), source_path),
            )
        finally:
            cur.close()
        conn.commit()
    finally:
        db_config.close_connection(conn)


def load_document(
    document_key: str,
    *,
    default: Any = None,
    fallback_path: Optional[str] = None,
    cache_ttl: int = DEFAULT_CACHE_TTL_SECONDS,
) -> Any:
    """
    Load a document from PostgreSQL (with Redis caching when available).

    Args:
        document_key: Logical identifier for the document.
        default: Value returned when the document is missing.
        fallback_path: Optional legacy JSON path used for bootstrap.
        cache_ttl: Redis TTL in seconds.
    """
    redis_key = _build_cache_key(document_key)
    cached = get_json(redis_key)
    if cached is not None:
        payload = cached.get("payload") if isinstance(cached, dict) else cached
        return deepcopy(payload)

    payload = _fetch_document_from_db(document_key)
    if payload is None:
        return deepcopy(default)

    if set_json:
        set_json(
            redis_key,
            {"payload": payload, "cached_at": datetime.now(timezone.utc).isoformat()},
            ttl_seconds=cache_ttl,
        )

    return deepcopy(payload)


def save_document(
    document_key: str,
    payload: Any,
    *,
    fallback_path: Optional[str] = None,
    cache_ttl: int = DEFAULT_CACHE_TTL_SECONDS,
    persist_to_disk: bool = False,
) -> None:
    """
    Persist a document to PostgreSQL and update Redis cache.

    Args:
        document_key: Logical identifier for the document.
        payload: JSON-serialisable object.
        fallback_path: Optional path for backwards-compatible writes.
        cache_ttl: Redis TTL in seconds.
        persist_to_disk: When True, also write the payload back to the fallback path.
    """
    _write_document_to_db(document_key, payload, source_path=fallback_path)

    redis_key = _build_cache_key(document_key)
    delete_key(redis_key)
    if set_json:
        set_json(
            redis_key,
            {"payload": payload, "cached_at": datetime.now(timezone.utc).isoformat()},
            ttl_seconds=cache_ttl,
        )

    # Disk persistence is deliberately disabled now that PostgreSQL is the source of truth.


def delete_document(document_key: str) -> None:
    """Remove a document from the store and invalidate caches."""
    conn = db_config.get_connection()
    try:
        cur = conn.cursor()
        try:
            cur.execute("DELETE FROM app_documents WHERE document_key = %s", (document_key,))
        finally:
            cur.close()
        conn.commit()
    finally:
        db_config.close_connection(conn)

    redis_key = _build_cache_key(document_key)
    delete_key(redis_key)


def document_exists(document_key: str) -> bool:
    """Return True when a document is present in the database."""
    conn = db_config.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM app_documents WHERE document_key = %s LIMIT 1",
                (document_key,),
            )
            row = cur.fetchone()
            return row is not None
    finally:
        db_config.close_connection(conn)


def clear_cache(document_key: Optional[str] = None) -> None:
    """
    Clear Redis cache entries for documents.

    Args:
        document_key: When provided, only clears this document. Otherwise, flushes
        all document entries currently cached (best-effort).
    """
    client = get_client()
    if client is None:
        return

    if document_key:
        delete_key(_build_cache_key(document_key))
        return

    pattern = build_key(f"{DOCUMENT_CACHE_NAMESPACE}*")
    try:
        keys = client.scan_iter(match=pattern)
        for key in keys:
            client.delete(key)
    except Exception:
        pass

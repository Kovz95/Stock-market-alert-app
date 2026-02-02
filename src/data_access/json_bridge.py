"""Compatibility bridge for legacy JSON file access.

This module intercepts ``open`` calls for specific JSON files and redirects
them to the PostgreSQL-backed document store.  Existing code that still uses
``with open('foo.json')`` can continue to operate without modification while
the underlying persistence is handled by PostgreSQL (with optional Redis
cache).
"""

from __future__ import annotations

import builtins
import io
import json
from pathlib import Path
from threading import RLock
from typing import Optional

from src.data_access.document_store import load_document, save_document


FILE_KEY_MAP = {
    "alert_check_results.json": "alert_check_results",
    "alerts.json": "alerts_legacy",
    "custom_discord_channels.json": "custom_discord_channels",
    "database_filters.json": "database_filters",
    "discord_channels_config.json": "discord_channels_config",
    "enhanced_fmp_ticker_mapping.json": "enhanced_fmp_ticker_mapping",
    "fmp_ticker_mapping.json": "fmp_ticker_mapping",
    "futures_alerts.json": "futures_alerts",
    "futures_database.json": "futures_database",
    "futures_scheduler_config.json": "futures_scheduler_config",
    "futures_scheduler_status.json": "futures_scheduler_status",
    "hourly_scheduler_status.json": "hourly_scheduler_status",
    "ib_futures_config.json": "ib_futures_config",
    "industry_filters.json": "industry_filters",
    "job_locks.json": "job_locks",
    "local_exchange_mappings.json": "local_exchange_mappings",
    "logging_config.json": "logging_config",
    "main_database_with_etfs.json": "main_database_with_etfs",
    "notifications.json": "notifications",
    "saved_scans.json": "saved_scans",
    "scheduler_config.json": "scheduler_config",
    "scheduler_status.json": "scheduler_status",
}

FILE_DEFAULTS = {
    "alerts_legacy": [],
    "alert_check_results": [],
    "custom_discord_channels": {},
    "database_filters": {},
    "discord_channels_config": {},
    "enhanced_fmp_ticker_mapping": {},
    "fmp_ticker_mapping": {},
    "futures_alerts": [],
    "futures_database": {},
    "futures_scheduler_config": {},
    "futures_scheduler_status": {},
    "hourly_scheduler_status": {},
    "ib_futures_config": {},
    "industry_filters": {},
    "job_locks": {},
    "local_exchange_mappings": {},
    "logging_config": {},
    "main_database_with_etfs": {},
    "notifications": [],
    "saved_scans": {},
    "scheduler_config": {},
    "scheduler_status": {},
}

_original_open = builtins.open
_bridge_lock = RLock()
_bridge_enabled = False


class _DocumentStoreWriter(io.StringIO):
    """Capture writes and persist the JSON payload to the document store."""

    def __init__(self, document_key: str, fallback_path: str):
        super().__init__()
        self._document_key = document_key
        self._fallback_path = fallback_path
        self._closed = False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        content = self.getvalue()
        super().close()
        if not content.strip():
            payload = None
        else:
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                payload = None

        if payload is not None:
            save_document(
                self._document_key,
                payload,
                fallback_path=self._fallback_path,
            )


def _is_supported_json(path: Path) -> Optional[str]:
    key = FILE_KEY_MAP.get(path.name)
    return key


def _bridge_open(file, mode="r", *args, **kwargs):
    # Only intercept path-like file arguments (str, Path, or __fspath__).
    # File descriptors (int) and other non-path args must pass through unchanged
    # (e.g. multiprocessing on Windows uses open(wfd, 'wb') with integer wfd).
    path = None
    if isinstance(file, (str, Path)):
        path = Path(file)
    elif hasattr(file, "__fspath__"):
        try:
            path = Path(file.__fspath__())
        except (TypeError, AttributeError):
            pass
    if path is None:
        return _original_open(file, mode, *args, **kwargs)
    document_key = _is_supported_json(path)
    if document_key is None or "b" in mode:
        return _original_open(file, mode, *args, **kwargs)

    if mode.startswith("r"):
        payload = load_document(
            document_key,
            default=FILE_DEFAULTS.get(document_key),
            fallback_path=str(path),
        )
        if payload is None:
            json_str = "null"
        else:
            json_str = json.dumps(payload, indent=2, sort_keys=True)
        stream = io.StringIO(json_str)
        stream.name = str(path)
        return stream

    if mode.startswith("w"):
        return _DocumentStoreWriter(document_key, str(path))

    # For other modes (append, update), fallback to the original open
    return _original_open(file, mode, *args, **kwargs)


def enable_json_bridge() -> None:
    """Enable interception of JSON file access."""
    global _bridge_enabled
    with _bridge_lock:
        if _bridge_enabled:
            return
        builtins.open = _bridge_open  # type: ignore[assignment]
        _bridge_enabled = True


def disable_json_bridge() -> None:
    """Restore the original ``open`` implementation (mainly for tests)."""
    global _bridge_enabled
    with _bridge_lock:
        if not _bridge_enabled:
            return
        builtins.open = _original_open
        _bridge_enabled = False

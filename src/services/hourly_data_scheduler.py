#!/usr/bin/env python3
"""Hourly data scheduler driven by exchange_calendars."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import psutil
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Ensure project modules are importable when run as a script
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from auto_scheduler_v2 import run_alert_checks  # noqa: E402
from calendar_adapter import (  # noqa: E402
    get_hourly_alignment,
    get_session_bounds,
    is_exchange_open as calendar_is_open,
)
from src.config.exchange_schedule_config import EXCHANGE_SCHEDULES  # noqa: E402
from src.services.hourly_price_collector import HourlyPriceCollector  # noqa: E402
from src.services.hourly_scheduler_discord import HourlySchedulerDiscord  # noqa: E402
from src.data_access.redis_support import build_key, delete_key, get_json, set_json  # noqa: E402
from src.data_access.document_store import delete_document, load_document, save_document  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger("hourly_data_scheduler")
logger.setLevel(logging.INFO)
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(stream_handler)
    # Use LOG_DIR environment variable if set
    log_dir = Path(os.getenv("LOG_DIR", "."))
    file_handler = logging.FileHandler(log_dir / "hourly_data_scheduler.log")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

LOCK_FILE = Path("hourly_scheduler.lock")
STATUS_FILE = Path("hourly_scheduler_status.json")
REDIS_STATUS_KEY = build_key("hourly_scheduler_status")
DOCUMENT_STATUS_KEY = "hourly_scheduler_status"
STATUS_CACHE_TTL_SECONDS = 3600

scheduler: BackgroundScheduler | None = None


# ---------------------------------------------------------------------------
# Lock helpers
# ---------------------------------------------------------------------------


def _process_matches(pid: int) -> bool:
    try:
        process = psutil.Process(pid)
        cmdline = process.cmdline() or []
        return any("hourly_data_scheduler" in segment for segment in cmdline)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False


def acquire_lock() -> bool:
    if LOCK_FILE.exists():
        try:
            info = json.loads(LOCK_FILE.read_text())
            existing_pid = info.get("pid")
        except Exception:
            existing_pid = None

        if existing_pid and _process_matches(existing_pid):
            logger.warning("Another hourly scheduler instance (PID %s) is running.", existing_pid)
            return False

        LOCK_FILE.unlink(missing_ok=True)

    payload = {
        "pid": os.getpid(),
        "timestamp": datetime.utcnow().isoformat(),
        "started_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "type": "hourly_data_scheduler",
    }
    LOCK_FILE.write_text(json.dumps(payload, indent=2))
    return True


def release_lock() -> None:
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except OSError as exc:
        logger.error("Failed to remove lock file: %s", exc)
    delete_document(DOCUMENT_STATUS_KEY)
    delete_key(REDIS_STATUS_KEY)


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------


def update_status(
    *,
    status: str,
    last_run: str | None = None,
    next_run: str | None = None,
    stats: dict | None = None,
    current_progress: dict | None = None,
) -> None:
    payload: Dict[str, Any] = {}
    try:
        payload = load_document(
            DOCUMENT_STATUS_KEY,
            default={},
            fallback_path=str(STATUS_FILE),
        )
        if not isinstance(payload, dict):
            payload = {}
    except Exception as exc:
        logger.error("Error loading persisted status: %s", exc)
        try:
            if STATUS_FILE.exists():
                payload_from_file = json.loads(STATUS_FILE.read_text())
                if isinstance(payload_from_file, dict):
                    payload = payload_from_file
        except Exception as file_exc:  # noqa: BLE001
            logger.error("Error reading fallback status file: %s", file_exc)
            payload = {}

    payload.update(
        {
            "status": status,
            "pid": os.getpid(),
            "type": "hourly_data_scheduler",
            "last_update": datetime.utcnow().isoformat(),
        }
    )

    if last_run is not None:
        payload["last_run"] = last_run
    if next_run is not None:
        payload["next_run"] = next_run
    elif status in {"stopped", "error"}:
        payload.pop("next_run", None)

    if stats is not None:
        payload["last_run_stats"] = stats

    if current_progress is not None:
        payload["current_progress"] = current_progress
    else:
        payload.pop("current_progress", None)

    try:
        save_document(
            DOCUMENT_STATUS_KEY,
            payload,
            fallback_path=str(STATUS_FILE),
            cache_ttl=STATUS_CACHE_TTL_SECONDS,
        )
    except Exception as exc:
        logger.error("Error writing status to document store: %s", exc)

    try:
        STATUS_FILE.write_text(json.dumps(payload, indent=2))
    except Exception as file_exc:  # noqa: BLE001
        logger.error("Failed to write fallback status file: %s", file_exc)

    try:
        set_json(REDIS_STATUS_KEY, payload, ttl_seconds=STATUS_CACHE_TTL_SECONDS)
    except Exception as exc:
        logger.error("Error caching status to Redis: %s", exc)


# ---------------------------------------------------------------------------
# Calendar helpers
# ---------------------------------------------------------------------------


def get_exchange_market_hours(exchanges: Iterable[str] | None = None) -> Dict[str, Tuple[float, float, str]]:
    """
    Return mapping of exchange -> (open_hour_utc, close_hour_utc, alignment).

    open/close are floats representing UTC hours (e.g. 13.5 for 13:30).
    """
    now = pd.Timestamp.utcnow()
    hours: Dict[str, Tuple[float, float, str]] = {}
    for exchange in exchanges or EXCHANGE_SCHEDULES.keys():
        try:
            open_dt, close_dt = get_session_bounds(exchange, now, next_if_closed=True)
        except Exception:
            continue

        open_utc = open_dt.tz_convert("UTC")
        close_utc = close_dt.tz_convert("UTC")
        style = get_hourly_alignment(exchange)
        hours[exchange] = (
            open_utc.hour + open_utc.minute / 60.0,
            close_utc.hour + close_utc.minute / 60.0,
            style,
        )
    return hours


def is_exchange_open(exchange: str, timestamp: datetime | None = None) -> bool:
    if not exchange:
        return False
    if timestamp is None:
        ts = pd.Timestamp.utcnow()
    elif isinstance(timestamp, pd.Timestamp):
        ts = timestamp.tz_convert("UTC") if timestamp.tzinfo else timestamp.tz_localize("UTC")
    else:
        ts = pd.Timestamp(timestamp)
        ts = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    return calendar_is_open(exchange, ts)


def any_market_open() -> bool:
    """Return True if any exchange is currently open; calendar errors for one exchange are skipped."""
    now = pd.Timestamp.utcnow()
    for exchange in EXCHANGE_SCHEDULES.keys():
        try:
            if calendar_is_open(exchange, now):
                return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skip exchange %s for open check: %s", exchange, exc)
    return False


def determine_candle_description(exchanges: Iterable[str]) -> str:
    styles = {get_hourly_alignment(ex) for ex in exchanges if ex}
    if not styles:
        return "Mixed cadence"
    if styles == {"quarter"}:
        return ":15 candles"
    if styles == {"half"}:
        return ":30 candles"
    if styles == {"hour"}:
        return ":00 candles"
    return "Mixed cadence"


def exchanges_grouped_by_style() -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = defaultdict(list)
    for exchange in EXCHANGE_SCHEDULES.keys():
        mapping[get_hourly_alignment(exchange)].append(exchange)
    return mapping


# ---------------------------------------------------------------------------
# Hourly job
# ---------------------------------------------------------------------------


def update_hourly_data(exchange_filter: Iterable[str] | None = None) -> None:
    collector = HourlyPriceCollector()
    discord_logger = HourlySchedulerDiscord()
    run_started = datetime.utcnow()

    try:
        exchanges = list(exchange_filter) if exchange_filter else list(EXCHANGE_SCHEDULES.keys())
        exchange_label = ", ".join(exchanges) if exchange_filter else "All exchanges"
        candle_desc = determine_candle_description(exchanges)

        logger.info("=" * 80)
        logger.info("Hourly update triggered (%s) - %s", candle_desc, exchange_label)

        if not any_market_open():
            reason = "No exchanges currently open"
            logger.info("%s - skipping run", reason)
            discord_logger.notify_skipped(run_started, reason)
            update_status(
                status="idle",
                last_run="Skipped - Markets closed",
                next_run=(datetime.utcnow() + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
            )
            return

        stock_db = collector.load_stock_database()
        now = datetime.utcnow()

        open_tickers = {}
        skipped_closed = defaultdict(int)
        for ticker, info in stock_db.items():
            exchange = info.get("exchange")
            if exchange_filter and exchange not in exchange_filter:
                continue
            if exchange and is_exchange_open(exchange, now):
                open_tickers[ticker] = info
            else:
                skipped_closed[exchange or "Unknown"] += 1

        if not open_tickers:
            reason = "No tickers from open exchanges"
            logger.info("%s - skipping run", reason)
            discord_logger.notify_skipped(run_started, reason)
            update_status(
                status="idle",
                last_run="Skipped - No open exchanges",
                next_run=(datetime.utcnow() + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
            )
            return

        open_exchanges = sorted({info.get("exchange") for info in open_tickers.values() if info.get("exchange")})
        exchange_hours = get_exchange_market_hours(open_exchanges)
        close_lines = [
            f"{exchange}: {close:.2f}h UTC"
            for exchange, (_, close, _) in exchange_hours.items()
        ]

        discord_logger.notify_start(
            run_started,
            candle_desc,
            open_exchanges or ["Unknown"],
            len(open_tickers),
            close_info=", ".join(close_lines) if close_lines else None,
        )

        stats = {
            "total": len(open_tickers),
            "success": 0,
            "failed": 0,
            "skipped_closed": sum(skipped_closed.values()),
        }

        start_time = time.time()
        for idx, (ticker, info) in enumerate(sorted(open_tickers.items()), 1):
            try:
                if collector.update_ticker_hourly(ticker, days_back=7, skip_existing=True):
                    stats["success"] += 1
                else:
                    stats["failed"] += 1
            except Exception as exc:
                stats["failed"] += 1
                logger.error("Failed to update %s: %s", ticker, exc)

            if idx % 50 == 0:
                update_status(
                    status="updating",
                    current_progress={
                        "current": idx,
                        "total": stats["total"],
                        "current_ticker": ticker,
                        "success": stats["success"],
                        "failed": stats["failed"],
                        "percentage": round(idx / stats["total"] * 100, 1),
                    },
                )

        alert_stats = run_alert_checks(open_exchanges, "hourly")
        stats.update(
            {
                "alerts_total": alert_stats.get("total", 0),
                "alerts_triggered": alert_stats.get("success", 0),
                "alerts_errors": alert_stats.get("errors", 0),
                "alerts_no_data": alert_stats.get("no_data", 0),
                "alerts_stale": alert_stats.get("stale_data", 0),
            }
        )

        elapsed = time.time() - start_time
        logger.info(
            "Hourly update finished: %s tickers (success=%s / failed=%s) in %.1fs",
            stats["total"],
            stats["success"],
            stats["failed"],
            elapsed,
        )

        first_failure = getattr(collector, "last_error", None) if stats.get("failed") else None
        discord_logger.notify_complete(
            run_started,
            elapsed,
            stats,
            alert_stats,
            open_exchanges or ["Unknown"],
            first_failure_reason=first_failure,
        )
        update_status(
            status="running",
            last_run=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            next_run=(datetime.utcnow() + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S"),
            stats=stats,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Hourly update failed: %s", exc)
        discord_logger.notify_error(run_started, str(exc))
        update_status(
            status="error",
            last_run=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        )
    finally:
        db_handle = getattr(collector, "db", None)
        close_fn = getattr(db_handle, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to close hourly price database handle: %s", exc)


# ---------------------------------------------------------------------------
# Scheduler orchestration
# ---------------------------------------------------------------------------


STYLE_MINUTES = {
    "hour": 5,
    "half": 35,
    "quarter": 20,
}


def build_scheduler() -> BackgroundScheduler:
    sched = BackgroundScheduler(timezone="UTC")
    style_map = exchanges_grouped_by_style()

    for style, minute in STYLE_MINUTES.items():
        exchanges = style_map.get(style, [])
        if not exchanges:
            continue
        sched.add_job(
            update_hourly_data,
            trigger=CronTrigger(minute=minute),
            kwargs={"exchange_filter": exchanges},
            id=f"hourly_{style}",
            replace_existing=True,
            misfire_grace_time=600,
            coalesce=True,
        )

    # Any exchanges that did not fall into the predefined styles
    classified = set().union(*style_map.values()) if style_map else set()
    miscellaneous = [ex for ex in EXCHANGE_SCHEDULES.keys() if ex not in classified]
    if miscellaneous:
        sched.add_job(
            update_hourly_data,
            trigger=CronTrigger(minute=5),
            kwargs={"exchange_filter": miscellaneous},
            id="hourly_misc",
            replace_existing=True,
            misfire_grace_time=600,
            coalesce=True,
        )

    return sched


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    global scheduler

    if not acquire_lock():
        return

    logger.info("Hourly scheduler starting (PID %s)", os.getpid())
    update_status(status="starting")

    scheduler = build_scheduler()
    scheduler.start()
    update_status(status="running")

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Hourly scheduler stopping (KeyboardInterrupt)")
    finally:
        if scheduler:
            scheduler.shutdown(wait=False)
        update_status(status="stopped")
        release_lock()


if __name__ == "__main__":
    main()

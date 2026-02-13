#!/usr/bin/env python3
"""
Database cleanup scheduler service.

Runs once daily at 2 AM UTC. Deletes stale price records:
  - daily_prices: rows older than 5 years (``date`` column)
  - hourly_prices: rows older than 1 year (``datetime`` column)

Usage:
    python -m src.services.data_cleanup_scheduler
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import psutil
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Ensure project modules are importable when run as a script
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.data_access.db_config import db_config  # noqa: E402

logger = logging.getLogger("data_cleanup_scheduler")
logger.setLevel(logging.INFO)

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

if not logger.handlers:
    # Console handler for Docker logs
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    logger.addHandler(stream_handler)

    # File handler with error handling
    log_dir = Path(os.getenv("LOG_DIR", "."))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "data_cleanup_scheduler.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
        logger.addHandler(file_handler)
    except Exception as e:
        logger.warning(f"Could not create file handler: {e}")

# Configure APScheduler logger to use the same format
apscheduler_logger = logging.getLogger("apscheduler")
apscheduler_logger.setLevel(logging.INFO)
if not apscheduler_logger.handlers:
    aps_handler = logging.StreamHandler(sys.stdout)
    aps_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT))
    apscheduler_logger.addHandler(aps_handler)

LOCK_FILE = Path("data_cleanup_scheduler.lock")  # relative → /app/ in Docker

# Configurable via env (useful for testing without waiting years)
DAILY_PRICES_RETENTION_YEARS: int = int(os.getenv("CLEANUP_DAILY_PRICES_RETENTION_YEARS", "5"))
HOURLY_PRICES_RETENTION_YEARS: int = int(os.getenv("CLEANUP_HOURLY_PRICES_RETENTION_YEARS", "1"))

# Default: 2:00 AM UTC daily
CLEANUP_HOUR: int = int(os.getenv("CLEANUP_SCHEDULE_HOUR", "2"))
CLEANUP_MINUTE: int = int(os.getenv("CLEANUP_SCHEDULE_MINUTE", "0"))

scheduler: BackgroundScheduler | None = None


# ---------------------------------------------------------------------------
# Lock helpers
# ---------------------------------------------------------------------------


def _process_matches(pid: int) -> bool:
    try:
        process = psutil.Process(pid)
        cmdline = process.cmdline() or []
        return any("data_cleanup_scheduler" in segment for segment in cmdline)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False


def acquire_lock() -> bool:
    current_pid = os.getpid()

    if LOCK_FILE.exists():
        try:
            info = json.loads(LOCK_FILE.read_text())
            existing_pid = info.get("pid")
        except Exception as e:
            logger.warning(f"Could not read lock file: {e}")
            existing_pid = None

        if existing_pid == current_pid:
            logger.debug("Lock file exists for current process (PID %s), updating timestamp", current_pid)
        elif existing_pid and _process_matches(existing_pid):
            logger.warning("Another data_cleanup_scheduler instance (PID %s) is already running.", existing_pid)
            return False
        else:
            logger.info("Removing stale lock file (PID %s not running)", existing_pid)
            LOCK_FILE.unlink(missing_ok=True)

    payload = {
        "pid": current_pid,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "started_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "type": "data_cleanup_scheduler",
    }
    LOCK_FILE.write_text(json.dumps(payload, indent=2))
    logger.info("Lock acquired for PID %s", current_pid)
    return True


def release_lock() -> None:
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except OSError as exc:
        logger.error("Failed to remove lock file: %s", exc)


# ---------------------------------------------------------------------------
# Core cleanup job
# ---------------------------------------------------------------------------


def run_cleanup_job() -> None:
    """Delete stale price records from daily_prices and hourly_prices tables."""
    run_started = datetime.now(tz=timezone.utc)
    logger.info("=" * 72)
    logger.info("Data cleanup job starting at %s UTC", run_started.strftime("%Y-%m-%d %H:%M:%S"))

    daily_deleted = 0
    hourly_deleted = 0

    try:
        with db_config.connection(role="cleanup") as conn:
            # daily_prices cleanup
            try:
                cursor = conn.cursor()
                cursor.execute(
                    f"DELETE FROM daily_prices WHERE date < NOW() - INTERVAL '{DAILY_PRICES_RETENTION_YEARS} years'"
                )
                daily_deleted = cursor.rowcount
                conn.commit()
                logger.info(
                    "daily_prices: deleted %d row(s) older than %d year(s).",
                    daily_deleted,
                    DAILY_PRICES_RETENTION_YEARS,
                )
            except Exception as exc:
                logger.error("Failed to clean daily_prices: %s", exc, exc_info=True)
                try:
                    conn.rollback()
                except Exception:
                    pass

            # hourly_prices cleanup
            try:
                cursor = conn.cursor()
                cursor.execute(
                    f"DELETE FROM hourly_prices WHERE datetime < NOW() - INTERVAL '{HOURLY_PRICES_RETENTION_YEARS} year'"
                )
                hourly_deleted = cursor.rowcount
                conn.commit()
                logger.info(
                    "hourly_prices: deleted %d row(s) older than %d year(s).",
                    hourly_deleted,
                    HOURLY_PRICES_RETENTION_YEARS,
                )
            except Exception as exc:
                logger.error("Failed to clean hourly_prices: %s", exc, exc_info=True)
                try:
                    conn.rollback()
                except Exception:
                    pass

    except Exception as exc:
        logger.error("Failed to obtain DB connection for cleanup: %s", exc, exc_info=True)

    elapsed = (datetime.now(tz=timezone.utc) - run_started).total_seconds()
    logger.info(
        "Cleanup finished in %.1fs | daily_prices deleted: %d | hourly_prices deleted: %d",
        elapsed,
        daily_deleted,
        hourly_deleted,
    )
    logger.info("=" * 72)


# ---------------------------------------------------------------------------
# Scheduler builder
# ---------------------------------------------------------------------------


def build_scheduler() -> BackgroundScheduler:
    sched = BackgroundScheduler(timezone="UTC")
    sched.add_job(
        run_cleanup_job,
        trigger=CronTrigger(hour=CLEANUP_HOUR, minute=CLEANUP_MINUTE, timezone="UTC"),
        id="data_cleanup",
        replace_existing=True,
        misfire_grace_time=3600,  # run even if container restarted ≤1h late
        coalesce=True,
        max_instances=1,
    )
    return sched


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    global scheduler

    if not acquire_lock():
        logger.error("Could not acquire lock — another instance running. Exiting.")
        sys.exit(1)

    logger.info("=" * 72)
    logger.info("Data cleanup scheduler starting (PID %s)", os.getpid())
    logger.info("Schedule: daily at %02d:%02d UTC", CLEANUP_HOUR, CLEANUP_MINUTE)
    logger.info(
        "Retention: daily_prices=%d years, hourly_prices=%d year(s)",
        DAILY_PRICES_RETENTION_YEARS,
        HOURLY_PRICES_RETENTION_YEARS,
    )
    logger.info("=" * 72)

    scheduler = build_scheduler()
    scheduler.start()

    for job in scheduler.get_jobs():
        next_run = job.next_run_time.strftime("%Y-%m-%d %H:%M:%S") if job.next_run_time else "Not scheduled"
        logger.info("Job '%s' next run: %s UTC", job.id, next_run)

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — stopping.")
    except Exception as exc:
        logger.error("Unexpected error in main loop: %s", exc, exc_info=True)
    finally:
        if scheduler and scheduler.running:
            scheduler.shutdown(wait=False)
        release_lock()
        logger.info("Data cleanup scheduler stopped.")


if __name__ == "__main__":
    main()

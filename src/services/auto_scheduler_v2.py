#!/usr/bin/env python3
"""
Auto Scheduler V2 - Calendar-Aware Stock Alert Scheduler

Automatically schedules daily and weekly alert checks for global exchanges
based on their actual trading schedules using exchange-calendars.

Features:
- Calendar-aware scheduling for 39+ global exchanges
- Daily checks after market close (with 40-minute data delay)
- Weekly checks on last trading day of the week
- Discord notifications for job events
- Job locking to prevent duplicate execution
- Comprehensive status tracking
- Heartbeat monitoring for watchdog integration

Usage:
    python auto_scheduler_v2.py
"""

from __future__ import annotations

import json
import logging
import threading
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import psutil
import pytz
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Exchange run times from get_exchanges_by_closing_time() are in Eastern Time
SCHEDULER_EXCHANGE_TZ = pytz.timezone("America/New_York")

# Ensure project modules are importable (project root so "from src.xxx" works)
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.append(str(BASE_DIR))

from src.data_access.alert_repository import list_alerts  # noqa: E402
from src.data_access.document_store import load_document, save_document  # noqa: E402
from src.data_access.metadata_repository import fetch_stock_metadata_df  # noqa: E402
from src.config.exchange_schedule_config import (  # noqa: E402
    EXCHANGE_SCHEDULES,
    get_exchanges_by_closing_time,
    get_market_days_for_exchange,
)
from src.services.scheduled_price_updater import update_prices_for_exchanges  # noqa: E402
from src.services.daily_price_service import run_full_daily_update  # noqa: E402
from src.services.stock_alert_checker import StockAlertChecker  # noqa: E402
from src.services.scheduler_discord import create_scheduler_discord  # noqa: E402

# Constants
LOCK_FILE = BASE_DIR / "scheduler_v2.lock"
STATUS_DOCUMENT_KEY = "scheduler_status"
CONFIG_DOCUMENT_KEY = "scheduler_config"
JOB_TIMEOUT_SECONDS = int(os.getenv("SCHEDULER_JOB_TIMEOUT", "900"))  # 15 minutes
HEARTBEAT_INTERVAL = int(os.getenv("SCHEDULER_HEARTBEAT_INTERVAL", "60"))  # 1 minute

# Logging
logger = logging.getLogger("auto_scheduler_v2")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler - use LOG_DIR environment variable if set
    log_dir = Path(os.getenv("LOG_DIR", str(BASE_DIR)))
    log_file = log_dir / "auto_scheduler_v2.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Global scheduler instance
scheduler: Optional[BackgroundScheduler] = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_scheduler_config() -> Dict[str, Any]:
    """Load scheduler configuration from document store."""
    default_config = {
        "scheduler_webhook": {
            "url": "",
            "enabled": False,
            "name": "Scheduler Status",
        },
        "notification_settings": {
            "send_start_notification": True,
            "send_completion_notification": True,
            "send_progress_updates": False,
            "progress_update_interval": 300,
            "include_summary_stats": True,
            "job_timeout_seconds": JOB_TIMEOUT_SECONDS,
        },
    }

    try:
        config = load_document(
            CONFIG_DOCUMENT_KEY,
            default=default_config,
            fallback_path=str(BASE_DIR / "scheduler_config.json"),
        )
        if not isinstance(config, dict):
            return default_config
        return config
    except Exception as exc:
        logger.warning("Failed to load scheduler config: %s", exc)
        return default_config


# ---------------------------------------------------------------------------
# Discord Notifications
# ---------------------------------------------------------------------------

def send_scheduler_notification(message: str, event: str = "info") -> bool:
    """Send a scheduler status message to Discord (if configured)."""
    config = load_scheduler_config()
    webhook_cfg = (config or {}).get("scheduler_webhook", {})

    if not webhook_cfg.get("enabled") or not webhook_cfg.get("url"):
        return False
    from src.utils.discord_env import get_discord_environment_tag, is_discord_send_enabled
    if not is_discord_send_enabled():
        return False

    payload = {
        "content": get_discord_environment_tag() + message,
        "username": webhook_cfg.get("name") or "Scheduler",
    }

    try:
        response = requests.post(webhook_cfg["url"], json=payload, timeout=10)
        if response.status_code not in (200, 204):
            logger.warning(
                "Failed to send scheduler notification (%s): HTTP %s",
                event,
                response.status_code,
            )
            return False
        return True
    except Exception as exc:
        logger.warning("Error sending scheduler notification (%s): %s", event, exc)
        return False


def _format_stats_for_message(price_stats, alert_stats) -> str:
    """Build a compact one-line summary for Discord."""
    price_parts = []
    if isinstance(price_stats, dict):
        price_parts.append(f"upd {price_stats.get('updated', 0):,}")
        price_parts.append(f"fail {price_stats.get('failed', 0):,}")
        price_parts.append(f"skip {price_stats.get('skipped', 0):,}")
    else:
        price_parts.append(str(price_stats))

    alert_parts = []
    if isinstance(alert_stats, dict):
        alert_parts.append(f"total {alert_stats.get('total', 0):,}")
        triggered = alert_stats.get('success', alert_stats.get('triggered', 0))
        errors = alert_stats.get('errors', 0)
        alert_parts.append(f"trig {triggered:,}")
        alert_parts.append(f"err {errors:,}")
    else:
        alert_parts.append(str(alert_stats))

    return f"Price ({', '.join(price_parts)}) | Alerts ({', '.join(alert_parts)})"


def _format_duration(seconds: float) -> str:
    """Return a human-friendly duration string."""
    seconds = int(max(0, round(seconds)))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Alert Checking
# ---------------------------------------------------------------------------

def run_alert_checks(
    exchanges: List[str],
    timeframe_key: str = "daily",
) -> Dict[str, Any]:
    """
    Run alert checks for the given exchanges.

    Args:
        exchanges: List of exchange names to check alerts for
        timeframe_key: The timeframe to check ('daily' or 'weekly')

    Returns:
        Statistics dictionary with alert check results
    """
    stats = {
        "total": 0,
        "success": 0,
        "triggered": 0,
        "errors": 0,
        "no_data": 0,
        "stale_data": 0,
    }

    try:
        # Get metadata to filter by exchange
        metadata_df = fetch_stock_metadata_df()
        if metadata_df is None or metadata_df.empty:
            logger.warning("No metadata available for alert checks")
            return stats

        # Get tickers for these exchanges
        exchange_tickers = set()
        for exchange in exchanges:
            exchange_df = metadata_df[metadata_df["exchange"] == exchange]
            if not exchange_df.empty and "symbol" in exchange_df.columns:
                exchange_tickers.update(exchange_df["symbol"].tolist())

        # Get all alerts
        all_alerts = list_alerts()
        relevant_alerts = []

        for alert in all_alerts:
            if not isinstance(alert, dict):
                continue

            alert_ticker = alert.get("ticker")
            alert_exchange = alert.get("exchange")
            alert_timeframe = alert.get("timeframe", "daily")
            alert_action = alert.get("action", "on")

            # Skip disabled alerts (only skip if explicitly set to "off")
            if alert_action == "off":
                continue

            # Check if alert matches the exchange filter
            if alert_exchange in exchanges or alert_ticker in exchange_tickers:
                # Check if alert matches the timeframe
                if timeframe_key == "weekly" and alert_timeframe.lower() in ("weekly", "1wk"):
                    relevant_alerts.append(alert)
                elif timeframe_key == "daily" and alert_timeframe.lower() in ("daily", "1d"):
                    relevant_alerts.append(alert)
                elif timeframe_key == "hourly" and alert_timeframe.lower() in ("hourly", "1h", "1hr"):
                    relevant_alerts.append(alert)

        stats["total"] = len(relevant_alerts)

        logger.info(
            "Evaluating %d %s alerts for exchanges %s",
            len(relevant_alerts),
            timeframe_key,
            exchanges,
        )

        # Create stock alert checker and evaluate alerts
        checker = StockAlertChecker()
        check_stats = checker.check_alerts(relevant_alerts, timeframe_key)

        # Update stats from checker results
        stats["success"] = check_stats.get("success", 0)
        stats["triggered"] = check_stats.get("triggered", 0)
        stats["errors"] = check_stats.get("errors", 0)
        stats["no_data"] = check_stats.get("no_data", 0)

        logger.info(
            "Alert evaluation complete: %d triggered, %d errors, %d no_data",
            stats["triggered"],
            stats["errors"],
            stats["no_data"],
        )

    except Exception as exc:
        logger.error("Error running alert checks: %s", exc)
        stats["errors"] = 1

    return stats


# ---------------------------------------------------------------------------
# Job Locking
# ---------------------------------------------------------------------------

_job_locks: Dict[str, bool] = {}
_lock_lock = threading.Lock()


def sanitize_job_id(job_type: str, exchange_name: str) -> str:
    """Create a consistent job ID from type and exchange."""
    return f"{job_type}_{exchange_name}".lower().replace(" ", "_")


def acquire_job_lock(job_id: str) -> bool:
    """Acquire a lock for a job. Returns True if acquired, False if already locked."""
    with _lock_lock:
        if _job_locks.get(job_id):
            return False
        _job_locks[job_id] = True
        return True


def release_job_lock(job_id: str) -> None:
    """Release a job lock."""
    with _lock_lock:
        _job_locks.pop(job_id, None)


# ---------------------------------------------------------------------------
# Status Management
# ---------------------------------------------------------------------------

def update_scheduler_status(
    status: str = "running",
    current_job: Optional[Dict[str, Any]] = None,
    last_run: Optional[Dict[str, Any]] = None,
    last_result: Optional[Dict[str, Any]] = None,
    last_error: Optional[Dict[str, Any]] = None,
) -> None:
    """Update scheduler status in document store."""
    try:
        existing_status = load_document(
            STATUS_DOCUMENT_KEY,
            default={},
            fallback_path=str(BASE_DIR / "scheduler_status.json"),
        )
        if not isinstance(existing_status, dict):
            existing_status = {}

        # Update fields
        existing_status["status"] = status
        existing_status["heartbeat"] = datetime.now(tz=timezone.utc).isoformat()

        if current_job is not None:
            existing_status["current_job"] = current_job
        elif "current_job" not in existing_status:
            existing_status["current_job"] = None

        if last_run is not None:
            existing_status["last_run"] = last_run

        if last_result is not None:
            existing_status["last_result"] = last_result

        if last_error is not None:
            existing_status["last_error"] = last_error

        # Save to document store
        save_document(
            STATUS_DOCUMENT_KEY,
            existing_status,
            fallback_path=str(BASE_DIR / "scheduler_status.json"),
        )

        # Also update lock file with heartbeat
        if LOCK_FILE.exists():
            try:
                lock_data = json.loads(LOCK_FILE.read_text())
                lock_data["heartbeat"] = datetime.now(tz=timezone.utc).isoformat()
                LOCK_FILE.write_text(json.dumps(lock_data, indent=2))
            except Exception:
                pass

    except Exception as exc:
        logger.warning("Failed to update scheduler status: %s", exc)


def get_scheduler_info() -> Optional[Dict[str, Any]]:
    """Get current scheduler status and information."""
    try:
        status = load_document(
            STATUS_DOCUMENT_KEY,
            default={},
            fallback_path=str(BASE_DIR / "scheduler_status.json"),
        )

        if not isinstance(status, dict):
            return None

        # Add job counts
        if scheduler:
            jobs = scheduler.get_jobs()
            daily_jobs = [j for j in jobs if "daily" in j.id]
            weekly_jobs = [j for j in jobs if "weekly" in j.id]

            status["total_daily_jobs"] = len(daily_jobs)
            status["total_weekly_jobs"] = len(weekly_jobs)

            # Get next run time
            if jobs:
                next_job = min(jobs, key=lambda j: j.next_run_time or datetime.max)
                if next_job.next_run_time:
                    status["next_run"] = next_job.next_run_time.isoformat()

        return status

    except Exception as exc:
        logger.warning("Failed to get scheduler info: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Process Management
# ---------------------------------------------------------------------------

def _process_matches(pid: int) -> bool:
    """Check if PID matches our scheduler process (running as main script)."""
    try:
        process = psutil.Process(pid)
        cmdline = process.cmdline() or []
        return any(segment.endswith("auto_scheduler_v2.py") for segment in cmdline)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False


def _acquire_process_lock() -> bool:
    """Acquire process-level lock file."""
    current_pid = os.getpid()

    if LOCK_FILE.exists():
        try:
            info = json.loads(LOCK_FILE.read_text())
            existing_pid = info.get("pid")
        except Exception:
            existing_pid = None

        # If the existing PID is the same as current PID, it's the same process
        # This can happen in Docker containers where PID 1 is the main process
        if existing_pid == current_pid:
            logger.debug("Lock file exists for current process (PID %s), updating timestamp", current_pid)
        elif existing_pid and _process_matches(existing_pid):
            logger.warning("Another scheduler instance (PID %s) is running.", existing_pid)
            return False
        else:
            LOCK_FILE.unlink(missing_ok=True)

    payload = {
        "pid": current_pid,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "heartbeat": datetime.now(tz=timezone.utc).isoformat(),
        "started_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "type": "auto_scheduler_v2",
    }
    LOCK_FILE.write_text(json.dumps(payload, indent=2))
    return True


def _release_process_lock() -> None:
    """Release process-level lock file."""
    try:
        LOCK_FILE.unlink(missing_ok=True)
    except OSError as exc:
        logger.error("Failed to remove lock file: %s", exc)


def is_scheduler_running() -> bool:
    """Check if scheduler is currently running as a main script."""
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            # Check that auto_scheduler_v2.py is being run as the main script
            # (not just imported by another script like check_scheduler_status.py)
            if any(part.endswith("auto_scheduler_v2.py") for part in cmdline):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


# ---------------------------------------------------------------------------
# Job Execution with Timeout
# ---------------------------------------------------------------------------

def _exchange_worker(exchange_name: str, resample_weekly: bool):
    """Execute price update + alert checks for a single exchange."""
    price_stats = update_prices_for_exchanges([exchange_name], resample_weekly=resample_weekly)
    timeframe_key = "weekly" if resample_weekly else "daily"
    alert_stats = run_alert_checks([exchange_name], timeframe_key)
    return price_stats, alert_stats


def _run_exchange_job(exchange_name: str, resample_weekly: bool, timeout_seconds: int):
    """
    Run a single exchange job on a thread with a soft timeout.

    Uses ThreadPoolExecutor so the caller is not blocked indefinitely.
    Unlike the old subprocess approach the async Discord queue lives in
    the same process and drains naturally — no explicit wait required.

    Returns (price_stats, alert_stats, error_message)
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_exchange_worker, exchange_name, resample_weekly)
        try:
            price_stats, alert_stats = future.result(timeout=timeout_seconds)
            return price_stats, alert_stats, None
        except TimeoutError:
            logger.warning(
                "Job for %s timed out after %ds (worker thread will finish in background)",
                exchange_name,
                timeout_seconds,
            )
            return None, None, f"timeout after {timeout_seconds}s"
        except Exception as exc:
            return None, None, str(exc)


def execute_exchange_job(exchange_name: str, job_type: str):
    """
    Execute a single exchange job (daily or weekly).

    Args:
        exchange_name: Name of the exchange
        job_type: 'daily' or 'weekly'
    """
    job_id = sanitize_job_id(job_type, exchange_name)
    if not acquire_job_lock(job_id):
        logger.warning("Job %s is already running; skipping duplicate execution.", job_id)
        return None

    notify_settings = (load_scheduler_config() or {}).get("notification_settings", {})
    send_start = notify_settings.get("send_start_notification", True)
    send_complete = notify_settings.get("send_completion_notification", True)
    job_timeout = max(int(notify_settings.get("job_timeout_seconds", JOB_TIMEOUT_SECONDS)), 60)

    run_time_utc = datetime.now(tz=timezone.utc)
    discord_notifier = create_scheduler_discord(job_type)

    start_time = time.time()
    if send_start:
        discord_notifier.notify_start(run_time_utc, exchange_name)

    try:
        update_scheduler_status(
            status="running",
            current_job={
                "id": job_id,
                "exchange": exchange_name,
                "job_type": job_type,
                "started": run_time_utc.isoformat(),
            },
        )

        resample_weekly = job_type == "weekly"
        price_stats, alert_stats, worker_err = _run_exchange_job(
            exchange_name,
            resample_weekly=resample_weekly,
            timeout_seconds=job_timeout,
        )

        if worker_err:
            raise TimeoutError(worker_err) if "timeout" in worker_err else RuntimeError(worker_err)

        duration = round(time.time() - start_time, 2)

        update_scheduler_status(
            status="running",
            current_job=None,
            last_run={
                "job_id": job_id,
                "exchange": exchange_name,
                "job_type": job_type,
                "completed_at": datetime.now(tz=timezone.utc).isoformat(),
                "duration_seconds": duration,
            },
            last_result={
                "price_stats": price_stats,
                "alert_stats": alert_stats,
            },
        )

        logger.info(
            "%s job finished for %s (price updates: %s, alerts: %s)",
            job_type.upper(),
            exchange_name,
            price_stats.get("updated", 0) if isinstance(price_stats, dict) else price_stats,
            alert_stats.get("total", 0) if isinstance(alert_stats, dict) else alert_stats,
        )

        if send_complete:
            first_failure_reason = (
                price_stats.get("first_failure_reason")
                if isinstance(price_stats, dict)
                else None
            )
            discord_notifier.notify_complete(
                run_time_utc,
                duration,
                price_stats if isinstance(price_stats, dict) else {},
                alert_stats if isinstance(alert_stats, dict) else {},
                exchange_name,
                first_failure_reason=first_failure_reason,
            )

        return {
            "price_stats": price_stats,
            "alert_stats": alert_stats,
            "duration": duration,
        }

    except Exception as exc:
        err_msg = (
            f"Timeout after {job_timeout}s"
            if isinstance(exc, TimeoutError)
            else str(exc)
        )
        logger.exception("Error running %s job for %s: %s", job_type, exchange_name, err_msg)
        update_scheduler_status(
            status="error",
            current_job=None,
            last_error={
                "time": datetime.now(tz=timezone.utc).isoformat(),
                "job_id": job_id,
                "exchange": exchange_name,
                "job_type": job_type,
                "message": err_msg,
            },
        )
        discord_notifier.notify_error(run_time_utc, err_msg)
        return None
    finally:
        release_job_lock(job_id)


def run_daily_job(exchange_name: str):
    """Execute a daily job for an exchange."""
    return execute_exchange_job(exchange_name, "daily")


def run_weekly_job(exchange_name: str):
    """Execute a weekly job for an exchange."""
    return execute_exchange_job(exchange_name, "weekly")


# ---------------------------------------------------------------------------
# Scheduler Setup
# ---------------------------------------------------------------------------

def _log_scheduler_startup_summary(configured_runs: List[Dict[str, Any]]):
    """Log a comprehensive startup summary of all scheduled jobs."""
    et_tz = pytz.timezone("America/New_York")
    utc_tz = pytz.timezone("UTC")
    now_utc = datetime.now(tz=timezone.utc)
    now_et = now_utc.astimezone(et_tz)

    # Count jobs from scheduler
    total_jobs = len(scheduler.get_jobs()) if scheduler else 0
    exchange_count = len(configured_runs)

    # Startup banner
    logger.info("=" * 70)
    logger.info("DAILY SCHEDULER STARTUP SUMMARY")
    logger.info("=" * 70)
    logger.info("Current Time: %s UTC / %s ET", now_utc.strftime("%Y-%m-%d %H:%M:%S"), now_et.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Scheduler Timezone: America/New_York (Eastern Time)")
    logger.info("Total Jobs Scheduled: %d (%d daily, %d weekly, plus system jobs)", total_jobs, exchange_count, exchange_count)
    logger.info("Exchanges Configured: %d", exchange_count)
    logger.info("-" * 70)

    # Group by time for cleaner display
    time_grouped: Dict[str, List[str]] = {}
    for entry in configured_runs:
        time_key = f"{entry['hour']:02d}:{entry['minute']:02d}"
        time_grouped.setdefault(time_key, []).append(entry["exchange"])

    # Log scheduled times grouped by time slot
    logger.info("SCHEDULED RUN TIMES (Eastern Time):")
    logger.info("-" * 70)
    for time_key in sorted(time_grouped.keys()):
        exchanges = sorted(time_grouped[time_key])
        # Convert to UTC for reference
        h, m = map(int, time_key.split(":"))
        sample_et = now_et.replace(hour=h, minute=m, second=0, microsecond=0)
        sample_utc = sample_et.astimezone(utc_tz)
        utc_time = sample_utc.strftime("%H:%M")

        logger.info("  %s ET (%s UTC): %d exchange(s)", time_key, utc_time, len(exchanges))
        # List all exchanges at this time
        for exchange in exchanges:
            logger.info("    - %s (daily + weekly)", exchange)

    logger.info("-" * 70)

    # Log next scheduled runs (up to 15)
    if scheduler:
        jobs = scheduler.get_jobs()
        # Sort by next run time
        jobs_with_time = [(j, j.next_run_time) for j in jobs if j.next_run_time]
        jobs_with_time.sort(key=lambda x: x[1])

        logger.info("NEXT SCHEDULED RUNS (up to 15):")
        logger.info("-" * 70)
        for job, next_run in jobs_with_time[:15]:
            next_et = next_run.astimezone(et_tz)
            next_utc = next_run.astimezone(utc_tz)
            time_until = next_run - now_utc
            hours_until = time_until.total_seconds() / 3600

            logger.info(
                "  %s: %s ET (%s UTC) - in %.1f hours",
                job.id,
                next_et.strftime("%Y-%m-%d %H:%M"),
                next_utc.strftime("%H:%M"),
                hours_until
            )

    logger.info("-" * 70)
    logger.info("Scheduler configuration complete. Waiting for scheduled jobs...")
    logger.info("=" * 70)


def _schedule_exchange_jobs():
    """Schedule all exchange jobs based on their closing times."""
    if not scheduler:
        logger.error("Scheduler not initialized")
        return

    time_groups = get_exchanges_by_closing_time()
    scheduled_count = 0
    # Collect configured run times for startup logging
    configured_runs: List[Dict[str, Any]] = []

    for time_str, exchanges_info in time_groups.items():
        for info in exchanges_info:
            exchange = info["exchange"]
            config = EXCHANGE_SCHEDULES.get(exchange, {})
            market_days = get_market_days_for_exchange(config)

            # Parse time string (format: "HH:MM")
            try:
                hour, minute = map(int, time_str.split(":"))
            except ValueError:
                logger.warning("Invalid time format for %s: %s", exchange, time_str)
                continue

            daily_days = market_days.get("daily", "mon-fri")
            weekly_day = market_days.get("weekly", "fri")
            time_display = f"{hour:02d}:{minute:02d}"
            configured_runs.append({
                "exchange": exchange,
                "hour": hour,
                "minute": minute,
                "daily_days": daily_days,
                "weekly_day": weekly_day,
                "daily_et": f"{time_display} ET ({daily_days})",
                "weekly_et": f"{time_display} ET ({weekly_day})",
            })

            # Schedule daily job (hour/minute are Eastern Time per get_exchanges_by_closing_time)
            daily_job_id = sanitize_job_id("daily", exchange)
            scheduler.add_job(
                run_daily_job,
                CronTrigger(
                    day_of_week=daily_days,
                    hour=hour,
                    minute=minute,
                    timezone=SCHEDULER_EXCHANGE_TZ,
                ),
                id=daily_job_id,
                args=[exchange],
                replace_existing=True,
                max_instances=1,
                misfire_grace_time=3600,
            )
            scheduled_count += 1

            # Schedule weekly job (same ET interpretation)
            weekly_job_id = sanitize_job_id("weekly", exchange)
            scheduler.add_job(
                run_weekly_job,
                CronTrigger(
                    day_of_week=weekly_day,
                    hour=hour,
                    minute=minute,
                    timezone=SCHEDULER_EXCHANGE_TZ,
                ),
                id=weekly_job_id,
                args=[exchange],
                replace_existing=True,
                max_instances=1,
                misfire_grace_time=3600,
            )
            scheduled_count += 1

    logger.info("Scheduled %d jobs for %d exchanges", scheduled_count, len(configured_runs))
    return configured_runs


def _heartbeat_job():
    """Update heartbeat timestamp in lock file and status."""
    try:
        if LOCK_FILE.exists():
            lock_data = json.loads(LOCK_FILE.read_text())
            lock_data["heartbeat"] = datetime.now(tz=timezone.utc).isoformat()
            LOCK_FILE.write_text(json.dumps(lock_data, indent=2))

        update_scheduler_status(status="running")
    except Exception as exc:
        logger.debug("Heartbeat update failed: %s", exc)


# ---------------------------------------------------------------------------
# Start/Stop Functions
# ---------------------------------------------------------------------------

def start_auto_scheduler(foreground: bool = False) -> bool:
    """
    Start the auto scheduler.
    
    Args:
        foreground: If True, run scheduler directly in this process (for Docker/CLI).
                   If False, spawn a background subprocess (for Streamlit UI).
    """
    global scheduler

    if is_scheduler_running():
        logger.info("Scheduler is already running")
        return True

    # If not running in foreground mode, spawn a background subprocess
    # This is used when starting from Streamlit UI to detach the scheduler
    if not foreground:
        try:
            # Run from project root so Python can resolve "from src.xxx" (cwd on path)
            script_path = BASE_DIR / "auto_scheduler_v2.py"
            popen_kw: dict = {
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
            }
            if sys.platform == "win32":
                popen_kw["creationflags"] = subprocess.CREATE_NO_WINDOW
            else:
                popen_kw["start_new_session"] = True

            subprocess.Popen(
                [sys.executable, str(script_path)],
                cwd=str(PROJECT_ROOT),
                **popen_kw,
            )
            logger.info("Started scheduler in background process")
            return True
        except Exception as exc:
            logger.error("Failed to start scheduler: %s", exc)
            return False

    # Running in foreground - start scheduler directly in this process
    if not _acquire_process_lock():
        return False

    try:
        scheduler = BackgroundScheduler(timezone="UTC")

        # Schedule all exchange jobs
        configured_runs = _schedule_exchange_jobs() or []

        # Add heartbeat job
        scheduler.add_job(
            _heartbeat_job,
            "interval",
            seconds=HEARTBEAT_INTERVAL,
            id="heartbeat",
            replace_existing=True,
        )

        # Full daily price update (all tickers) at 11 PM UTC
        scheduler.add_job(
            run_full_daily_update,
            CronTrigger(hour=23, minute=0, timezone="UTC"),
            id="daily_full_update",
            name="Daily Full Database Update",
            replace_existing=True,
            misfire_grace_time=3600,
        )

        # Start scheduler
        scheduler.start()

        # Log the complete startup summary after scheduler is started
        # so we have accurate next_run_time information
        _log_scheduler_startup_summary(configured_runs)

        # Update status
        update_scheduler_status(
            status="running",
            current_job=None,
        )

        logger.info("Auto scheduler v2 started successfully")
        send_scheduler_notification("✅ Scheduler started", "success")

        return True

    except Exception as exc:
        logger.exception("Failed to start scheduler: %s", exc)
        _release_process_lock()
        return False


def stop_auto_scheduler() -> bool:
    """Stop the auto scheduler."""
    global scheduler

    try:
        if scheduler and scheduler.running:
            scheduler.shutdown(wait=True)
            scheduler = None

        update_scheduler_status(status="stopped", current_job=None)
        _release_process_lock()

        logger.info("Auto scheduler v2 stopped")
        send_scheduler_notification("⏹️ Scheduler stopped", "info")

        return True

    except Exception as exc:
        logger.error("Failed to stop scheduler: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    """Main entry point for running scheduler as a standalone process."""
    # Run in foreground mode when executed directly (Docker, CLI)
    # This ensures logs are visible and scheduler runs in main process
    if not start_auto_scheduler(foreground=True):
        logger.error("Failed to start scheduler")
        sys.exit(1)

    logger.info("Scheduler running. Press Ctrl+C to stop.")

    def signal_handler(sig, frame):
        logger.info("Received signal %s, shutting down...", sig)
        stop_auto_scheduler()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Keep the process alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        stop_auto_scheduler()


if __name__ == "__main__":
    main()

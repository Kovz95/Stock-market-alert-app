#!/usr/bin/env python3
"""
Auto Scheduler V2 - Calendar-Aware Stock Alert Scheduler

Thin orchestrator/facade that preserves the original public API while
delegating job execution to scheduler_job_handler.py (Template Method)
and shared infrastructure to scheduler_services.py.

Public API (unchanged):
    run_daily_job(exchange_name)
    run_weekly_job(exchange_name)
    execute_exchange_job(exchange_name, job_type)
    start_auto_scheduler(foreground=False)
    stop_auto_scheduler()
    get_scheduler_info()
    is_scheduler_running()
    run_alert_checks(exchanges, timeframe_key="daily")
    load_scheduler_config()
    send_scheduler_notification(message, event="info")

Usage:
    python auto_scheduler_v2.py
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytz
from apscheduler.executors.pool import ThreadPoolExecutor as APThreadPoolExecutor
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

from src.data_access.alert_repository import list_alerts, refresh_alert_cache  # noqa: E402
from src.data_access.metadata_repository import fetch_stock_metadata_df  # noqa: E402
from src.config.exchange_schedule_config import (  # noqa: E402
    EXCHANGE_SCHEDULES,
    get_exchanges_by_closing_time,
    get_market_days_for_exchange,
)
from src.services.daily_price_service import run_full_daily_update  # noqa: E402
from src.services.stock_alert_checker import StockAlertChecker  # noqa: E402
from src.services.scheduler_services import (  # noqa: E402
    HEARTBEAT_INTERVAL,
    JOB_TIMEOUT_SECONDS,
    LOCK_FILE,
    SchedulerServices,
    format_duration,
    format_stats_for_message,
)
from src.services.scheduler_job_handler import (  # noqa: E402
    DailyJobHandler,
    HourlyJobHandler,
    WeeklyJobHandler,
    _exchange_worker,
    _run_exchange_job,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger = logging.getLogger("auto_scheduler_v2")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_dir = Path(os.getenv("LOG_DIR", str(BASE_DIR)))
    log_file = log_dir / "auto_scheduler_v2.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# ---------------------------------------------------------------------------
# Singleton instances (mode-specific lock file when SCHEDULER_MODE is set)
# ---------------------------------------------------------------------------

_scheduler_mode = os.getenv("SCHEDULER_MODE", "").strip().lower() or None
if _scheduler_mode and _scheduler_mode in ("daily", "weekly", "hourly"):
    _services = SchedulerServices(
        lock_file=BASE_DIR / f"scheduler_v2_{_scheduler_mode}.lock",
        mode=_scheduler_mode
    )
else:
    _services = SchedulerServices()
_daily_handler = DailyJobHandler(_services)
_weekly_handler = WeeklyJobHandler(_services)
_hourly_handler = HourlyJobHandler(_services)

# Global scheduler instance
scheduler: Optional[BackgroundScheduler] = None


# ---------------------------------------------------------------------------
# Backward-compatible public API: Configuration
# ---------------------------------------------------------------------------

def load_scheduler_config() -> Dict[str, Any]:
    """Load scheduler configuration from document store."""
    return _services.load_config()


# ---------------------------------------------------------------------------
# Backward-compatible public API: Discord Notifications
# ---------------------------------------------------------------------------

def send_scheduler_notification(message: str, event: str = "info") -> bool:
    """Send a scheduler status message to Discord (if configured)."""
    return _services.send_notification(message, event)


# ---------------------------------------------------------------------------
# Backward-compatible public API: Formatting helpers
# ---------------------------------------------------------------------------

def _format_stats_for_message(price_stats: Any, alert_stats: Any) -> str:
    """Build a compact one-line summary for Discord."""
    return format_stats_for_message(price_stats, alert_stats)


def _format_duration(seconds: float) -> str:
    """Return a human-friendly duration string."""
    return format_duration(seconds)


# ---------------------------------------------------------------------------
# Alert Checking (stays here — imported by hourly_data_scheduler.py)
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
        # Clear stale in-process lru_cache so the scheduler always sees
        # the latest alerts from Redis / the database.
        refresh_alert_cache()

        metadata_df = fetch_stock_metadata_df()
        if metadata_df is None or metadata_df.empty:
            logger.warning("No metadata available for alert checks")
            return stats

        exchange_tickers = set()
        for exchange in exchanges:
            exchange_df = metadata_df[metadata_df["exchange"] == exchange]
            if not exchange_df.empty and "symbol" in exchange_df.columns:
                exchange_tickers.update(exchange_df["symbol"].tolist())

        all_alerts = list_alerts()
        relevant_alerts = []

        for alert in all_alerts:
            if not isinstance(alert, dict):
                continue

            alert_ticker = alert.get("ticker")
            alert_exchange = alert.get("exchange")
            alert_timeframe = alert.get("timeframe", "daily")
            alert_action = alert.get("action", "on")

            if alert_action == "off":
                continue

            if alert_exchange in exchanges or alert_ticker in exchange_tickers:
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

        checker = StockAlertChecker()
        check_stats = checker.check_alerts(relevant_alerts, timeframe_key)

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
# Backward-compatible public API: Job Locking
# ---------------------------------------------------------------------------

# Expose the services instance's job locks for tests that access _job_locks
_job_locks = _services._job_locks
_lock_lock = _services._lock_lock


def sanitize_job_id(job_type: str, exchange_name: str) -> str:
    """Create a consistent job ID from type and exchange."""
    return _services.sanitize_job_id(job_type, exchange_name)


def acquire_job_lock(job_id: str) -> bool:
    """Acquire a lock for a job. Returns True if acquired, False if already locked."""
    return _services.acquire_job_lock(job_id)


def release_job_lock(job_id: str) -> None:
    """Release a job lock."""
    _services.release_job_lock(job_id)


# ---------------------------------------------------------------------------
# Backward-compatible public API: Status Management
# ---------------------------------------------------------------------------

def update_scheduler_status(
    status: str = "running",
    current_job: Optional[Dict[str, Any]] = None,
    last_run: Optional[Dict[str, Any]] = None,
    last_result: Optional[Dict[str, Any]] = None,
    last_error: Optional[Dict[str, Any]] = None,
) -> None:
    """Update scheduler status in document store."""
    _services.update_status(
        status=status,
        current_job=current_job,
        last_run=last_run,
        last_result=last_result,
        last_error=last_error,
    )


def get_scheduler_info() -> Optional[Dict[str, Any]]:
    """Get current scheduler status and information."""
    return _services.get_info(scheduler_instance=scheduler)


# ---------------------------------------------------------------------------
# Backward-compatible public API: Process Management
# ---------------------------------------------------------------------------

def _process_matches(pid: int) -> bool:
    """Check if PID matches our scheduler process."""
    return _services.process_matches(pid)


def _acquire_process_lock() -> bool:
    """Acquire process-level lock file."""
    return _services.acquire_process_lock()


def _release_process_lock() -> None:
    """Release process-level lock file."""
    _services.release_process_lock()


def is_scheduler_running() -> bool:
    """Check if scheduler is currently running as a main script."""
    return _services.is_scheduler_running()


# ---------------------------------------------------------------------------
# Backward-compatible public API: Job Execution
# ---------------------------------------------------------------------------

def execute_exchange_job(exchange_name: str, job_type: str) -> Optional[Dict[str, Any]]:
    """
    Execute a single exchange job (daily, weekly, or hourly).

    Args:
        exchange_name: Name of the exchange
        job_type: 'daily', 'weekly', or 'hourly'
    """
    if job_type == "daily":
        return _daily_handler.execute(exchange_name)
    if job_type == "weekly":
        return _weekly_handler.execute(exchange_name)
    if job_type == "hourly":
        return _hourly_handler.execute(exchange_name)
    logger.warning("Unknown job_type %r; expected daily, weekly, or hourly", job_type)
    return None


def run_daily_job(exchange_name: str):
    """Execute a daily job for an exchange."""
    return execute_exchange_job(exchange_name, "daily")


def run_weekly_job(exchange_name: str):
    """Execute a weekly job for an exchange."""
    return execute_exchange_job(exchange_name, "weekly")


def run_hourly_job(exchange_name: str):
    """Execute an hourly job for an exchange."""
    return execute_exchange_job(exchange_name, "hourly")


# ---------------------------------------------------------------------------
# Scheduler Setup
# ---------------------------------------------------------------------------

def _log_scheduler_startup_summary(configured_runs: List[Dict[str, Any]]):
    """Log a comprehensive startup summary of all scheduled jobs."""
    et_tz = pytz.timezone("America/New_York")
    utc_tz = pytz.timezone("UTC")
    now_utc = datetime.now(tz=timezone.utc)
    now_et = now_utc.astimezone(et_tz)

    total_jobs = len(scheduler.get_jobs()) if scheduler else 0
    exchange_count = len(configured_runs)

    logger.info("=" * 70)
    logger.info("SCHEDULER STARTUP SUMMARY (mode=%s)", _scheduler_mode or "all")
    logger.info("=" * 70)
    logger.info("Current Time: %s UTC / %s ET", now_utc.strftime("%Y-%m-%d %H:%M:%S"), now_et.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("Total Jobs Scheduled: %d | Exchanges Configured: %d", total_jobs, exchange_count)
    logger.info("-" * 70)

    # Hourly mode: entries have exchange + hourly_et
    if configured_runs and "hourly_et" in configured_runs[0]:
        logger.info("HOURLY RUNS (UTC):")
        for entry in configured_runs:
            logger.info("  - %s: %s", entry["exchange"], entry.get("hourly_et", "every hour"))
    else:
        time_grouped: Dict[str, List[str]] = {}
        for entry in configured_runs:
            time_key = f"{entry['hour']:02d}:{entry['minute']:02d}"
            time_grouped.setdefault(time_key, []).append(entry["exchange"])

        logger.info("SCHEDULED RUN TIMES (Eastern Time):")
        logger.info("-" * 70)
        for time_key in sorted(time_grouped.keys()):
            exchanges = sorted(time_grouped[time_key])
            h, m = map(int, time_key.split(":"))
            sample_et = now_et.replace(hour=h, minute=m, second=0, microsecond=0)
            sample_utc = sample_et.astimezone(utc_tz)
            utc_time = sample_utc.strftime("%H:%M")

            logger.info("  %s ET (%s UTC): %d exchange(s)", time_key, utc_time, len(exchanges))
            for exchange in exchanges:
                logger.info("    - %s (daily + weekly)", exchange)

    logger.info("-" * 70)

    if scheduler:
        jobs = scheduler.get_jobs()
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


def _schedule_exchange_jobs(mode: Optional[str] = None):
    """Schedule exchange jobs based on closing times (or hourly when mode=hourly).

    When mode is 'daily', 'weekly', or 'hourly', only that job type is scheduled.
    When mode is None, both daily and weekly are scheduled (backward compatible).
    """
    if not scheduler:
        logger.error("Scheduler not initialized")
        return

    scheduled_count = 0
    configured_runs: List[Dict[str, Any]] = []
    schedule_daily = mode is None or mode == "daily"
    schedule_weekly = mode is None or mode == "weekly"
    schedule_hourly = mode == "hourly"

    if schedule_hourly:
        # Hourly: one job per exchange at minute 5 past every hour (UTC)
        for exchange in sorted(EXCHANGE_SCHEDULES.keys()):
            hourly_job_id = sanitize_job_id("hourly", exchange)
            scheduler.add_job(
                run_hourly_job,
                CronTrigger(minute=5, timezone="UTC"),
                id=hourly_job_id,
                args=[exchange],
                replace_existing=True,
                max_instances=1,
                coalesce=True,
                misfire_grace_time=600,
            )
            scheduled_count += 1
            configured_runs.append({
                "exchange": exchange,
                "hourly_et": "every hour at :05 UTC",
            })
        logger.info("Scheduled %d hourly jobs for %d exchanges", scheduled_count, len(configured_runs))
        return configured_runs

    time_groups = get_exchanges_by_closing_time()
    for time_str, exchanges_info in time_groups.items():
        for info in exchanges_info:
            exchange = info["exchange"]
            config = EXCHANGE_SCHEDULES.get(exchange, {})
            market_days = get_market_days_for_exchange(config)

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

            if schedule_daily:
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
                    coalesce=True,
                    misfire_grace_time=3600,
                )
                scheduled_count += 1

            if schedule_weekly:
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
                    coalesce=True,
                    misfire_grace_time=3600,
                )
                scheduled_count += 1

    logger.info("Scheduled %d jobs for %d exchanges", scheduled_count, len(configured_runs))
    return configured_runs


def _heartbeat_job():
    """Update heartbeat timestamp in lock file and status."""
    _services.heartbeat()


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

    if not foreground:
        try:
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

    if not _acquire_process_lock():
        return False

    try:
        # Run one exchange job at a time to avoid thread/connection exhaustion.
        # Parallelism happens inside each job (10-20 price-update workers),
        # not across jobs. Queued jobs run sequentially as slots free up.
        max_concurrent_jobs = int(os.getenv("SCHEDULER_MAX_CONCURRENT_JOBS", "1"))
        scheduler = BackgroundScheduler(
            timezone="UTC",
            executors={
                "default": APThreadPoolExecutor(max_workers=max_concurrent_jobs),
            },
        )

        configured_runs = _schedule_exchange_jobs(_scheduler_mode) or []

        scheduler.add_job(
            _heartbeat_job,
            "interval",
            seconds=HEARTBEAT_INTERVAL,
            id="heartbeat",
            replace_existing=True,
            max_instances=2,
            coalesce=True,
            misfire_grace_time=30,
        )

        # Daily full update only when running in combined mode or daily-only mode
        if _scheduler_mode is None or _scheduler_mode == "daily":
            scheduler.add_job(
                run_full_daily_update,
                CronTrigger(hour=23, minute=0, timezone="UTC"),
                id="daily_full_update",
                name="Daily Full Database Update",
                replace_existing=True,
                coalesce=True,
                max_instances=1,
                misfire_grace_time=3600,
            )

        scheduler.start()
        _log_scheduler_startup_summary(configured_runs)

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
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        stop_auto_scheduler()


if __name__ == "__main__":
    main()

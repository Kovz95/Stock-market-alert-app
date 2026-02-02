#!/usr/bin/env python3
"""
Futures Scheduler - Automated futures price updates and alert checking

Automatically schedules futures price updates and alert checks based on
configured intervals and Interactive Brokers trading hours.

Features:
- APScheduler-based scheduling (aligned with auto_scheduler_v2.py)
- Respects IB trading hours (configurable, default 05:00-23:00 UTC)
- Discord notifications for job events
- Job locking to prevent duplicate execution
- Comprehensive status tracking
- Heartbeat monitoring for watchdog integration
- Subprocess-based job execution with timeout protection

Usage:
    python src/services/futures_scheduler.py
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

import psutil
import requests
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Ensure project modules are importable
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from src.data_access.document_store import load_document, save_document  # noqa: E402
from src.services.futures_alert_checker import FuturesAlertChecker  # noqa: E402

# Constants
LOCK_FILE = BASE_DIR / "futures_scheduler.lock"
STATUS_DOCUMENT_KEY = "futures_scheduler_status"
CONFIG_DOCUMENT_KEY = "futures_scheduler_config"
JOB_TIMEOUT_SECONDS = int(os.getenv("FUTURES_SCHEDULER_JOB_TIMEOUT", "900"))  # 15 minutes
HEARTBEAT_INTERVAL = int(os.getenv("FUTURES_SCHEDULER_HEARTBEAT_INTERVAL", "60"))  # 1 minute

# Logging
logger = logging.getLogger("futures_scheduler")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    log_file = BASE_DIR / "futures_scheduler.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

# Global scheduler instance
scheduler: Optional[BackgroundScheduler] = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_scheduler_config() -> Dict[str, Any]:
    """Load futures scheduler configuration from document store."""
    default_config = {
        "scheduler_webhook": {
            "url": "",
            "enabled": False,
            "name": "Futures Scheduler",
        },
        "update_times": ["06:00", "12:00", "16:00", "20:00"],
        "ib_hours": {
            "start": "05:00",
            "end": "23:00",
        },
        "enabled": True,
        "update_on_start": True,
        "notification_settings": {
            "send_start_notification": True,
            "send_completion_notification": True,
            "include_summary_stats": True,
            "job_timeout_seconds": JOB_TIMEOUT_SECONDS,
        },
    }

    try:
        config = load_document(
            CONFIG_DOCUMENT_KEY,
            default=default_config,
            fallback_path=str(BASE_DIR / "futures_scheduler_config.json"),
        )
        if not isinstance(config, dict):
            return default_config

        # Merge with defaults to ensure all keys exist
        merged = {**default_config, **config}
        merged.setdefault("ib_hours", default_config["ib_hours"])
        merged.setdefault("notification_settings", default_config["notification_settings"])

        return merged
    except Exception as exc:
        logger.warning("Failed to load futures scheduler config: %s", exc)
        return default_config


def save_scheduler_config(config: Dict[str, Any]) -> bool:
    """Save configuration to document store."""
    try:
        save_document(
            CONFIG_DOCUMENT_KEY,
            config,
            fallback_path=str(BASE_DIR / "futures_scheduler_config.json"),
        )
        return True
    except Exception as exc:
        logger.error("Failed to save scheduler config: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Discord Notifications
# ---------------------------------------------------------------------------

def send_scheduler_notification(message: str, event: str = "info") -> bool:
    """Send a scheduler status message to Discord (if configured)."""
    config = load_scheduler_config()
    webhook_cfg = (config or {}).get("scheduler_webhook", {})

    if not webhook_cfg.get("enabled") or not webhook_cfg.get("url"):
        return False

    payload = {
        "content": message,
        "username": webhook_cfg.get("name") or "Futures Scheduler",
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
# IB Trading Hours Check
# ---------------------------------------------------------------------------

def is_ib_available() -> bool:
    """Check if current time is within IB trading hours."""
    try:
        config = load_scheduler_config()
        ib_hours = config.get("ib_hours", {})

        now = datetime.utcnow()
        current_time = now.strftime("%H:%M")

        ib_start = ib_hours.get("start", "05:00")
        ib_end = ib_hours.get("end", "23:00")

        # Simple time comparison (assumes same day)
        return ib_start <= current_time <= ib_end

    except Exception as exc:
        logger.error("Error checking IB availability: %s", exc)
        return True  # Assume available if can't determine


# ---------------------------------------------------------------------------
# Job Locking
# ---------------------------------------------------------------------------

_job_locks: Dict[str, bool] = {}
_lock_lock = mp.Lock()


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
    """Update futures scheduler status in document store."""
    try:
        existing_status = load_document(
            STATUS_DOCUMENT_KEY,
            default={},
            fallback_path=str(BASE_DIR / "futures_scheduler_status.json"),
        )
        if not isinstance(existing_status, dict):
            existing_status = {}

        # Update fields
        existing_status["status"] = status
        existing_status["heartbeat"] = datetime.utcnow().isoformat()
        existing_status["pid"] = os.getpid()
        existing_status["type"] = "futures_scheduler"

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
            fallback_path=str(BASE_DIR / "futures_scheduler_status.json"),
        )

        # Also update lock file with heartbeat
        if LOCK_FILE.exists():
            try:
                lock_data = json.loads(LOCK_FILE.read_text())
                lock_data["heartbeat"] = datetime.utcnow().isoformat()
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
            fallback_path=str(BASE_DIR / "futures_scheduler_status.json"),
        )

        if not isinstance(status, dict):
            return None

        # Add job counts
        if scheduler:
            jobs = scheduler.get_jobs()
            status["total_jobs"] = len(jobs)

            # Get next run time
            if jobs:
                next_job = min(jobs, key=lambda j: j.next_run_time or datetime.max.replace(tzinfo=None))
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
        return any(segment.endswith("futures_scheduler.py") for segment in cmdline)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False


def _acquire_process_lock() -> bool:
    """Acquire process-level lock file."""
    if LOCK_FILE.exists():
        try:
            info = json.loads(LOCK_FILE.read_text())
            existing_pid = info.get("pid")
        except Exception:
            existing_pid = None

        if existing_pid and _process_matches(existing_pid):
            logger.warning("Another futures scheduler instance (PID %s) is running.", existing_pid)
            return False

        LOCK_FILE.unlink(missing_ok=True)

    payload = {
        "pid": os.getpid(),
        "timestamp": datetime.utcnow().isoformat(),
        "heartbeat": datetime.utcnow().isoformat(),
        "started_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "type": "futures_scheduler",
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
    """Check if futures scheduler is currently running as a main script."""
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            # Check that futures_scheduler.py is being run as the main script
            if any(part.endswith("futures_scheduler.py") for part in cmdline):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


# ---------------------------------------------------------------------------
# Price Update Execution
# ---------------------------------------------------------------------------

def run_price_update() -> Dict[str, Any]:
    """
    Run futures price update via subprocess.

    Returns:
        Statistics dictionary with update results
    """
    stats = {"updated": 0, "failed": 0, "error": None}

    try:
        if not is_ib_available():
            logger.info("Skipping price update - outside IB hours")
            stats["error"] = "Outside IB hours"
            return stats

        logger.info("Starting futures price update...")
        start_time = time.time()

        # Run the price updater
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "futures_price_updater.py")],
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            cwd=str(BASE_DIR),
        )

        if result.returncode == 0:
            # Parse output for counts if available
            updated_count = 0
            failed_count = 0
            if "succeeded" in result.stdout:
                try:
                    for line in result.stdout.split("\n"):
                        if "succeeded" in line and "failed" in line:
                            # Parse: "Update complete: 60 succeeded, 5 failed"
                            parts = line.split(":")
                            if len(parts) > 1:
                                counts = parts[1].strip().split(",")
                                for count in counts:
                                    if "succeeded" in count:
                                        updated_count = int(count.split()[0])
                                    elif "failed" in count:
                                        failed_count = int(count.split()[0])
                            break
                except Exception:
                    pass

            stats["updated"] = updated_count
            stats["failed"] = failed_count
            duration = time.time() - start_time
            logger.info(
                "Price update completed: %d updated, %d failed (%.1fs)",
                updated_count,
                failed_count,
                duration,
            )
        else:
            stats["error"] = result.stderr[:500] if result.stderr else "Unknown error"
            logger.error("Price update failed: %s", stats["error"])

    except subprocess.TimeoutExpired:
        stats["error"] = "Timeout after 600s"
        logger.error("Price update timed out")
    except Exception as exc:
        stats["error"] = str(exc)
        logger.error("Error running price update: %s", exc)

    return stats


# ---------------------------------------------------------------------------
# Alert Checking
# ---------------------------------------------------------------------------

def run_alert_checks() -> Dict[str, Any]:
    """
    Run futures alert checks.

    Returns:
        Statistics dictionary with alert check results
    """
    stats = {
        "total": 0,
        "triggered": 0,
        "errors": 0,
        "skipped": 0,
        "no_data": 0,
        "success": 0,
    }

    try:
        logger.info("Starting futures alert check...")
        start_time = time.time()

        # Create alert checker and run checks
        checker = FuturesAlertChecker()
        check_stats = checker.check_all_alerts()

        # Update stats from checker results
        stats.update(check_stats)

        duration = time.time() - start_time
        logger.info(
            "Alert check complete: %d total, %d triggered, %d errors (%.1fs)",
            stats["total"],
            stats["triggered"],
            stats["errors"],
            duration,
        )

    except Exception as exc:
        logger.error("Error running alert checks: %s", exc)
        stats["errors"] = 1

    return stats


# ---------------------------------------------------------------------------
# Job Execution with Timeout
# ---------------------------------------------------------------------------

def _job_worker(queue):
    """Worker executed in a subprocess to isolate hangs/crashes."""
    try:
        price_stats = run_price_update()
        time.sleep(5)  # Let database settle
        alert_stats = run_alert_checks()
        queue.put({"ok": True, "price_stats": price_stats, "alert_stats": alert_stats})
    except Exception as exc:
        queue.put({"ok": False, "error": str(exc)})


def _run_job_subprocess(timeout_seconds: int):
    """
    Run a futures job in a subprocess with a hard timeout.
    Returns (price_stats, alert_stats, error_message)
    """
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_job_worker, args=(queue,))
    proc.start()
    start = time.time()

    while True:
        elapsed = time.time() - start
        remaining = timeout_seconds - elapsed
        if remaining <= 0:
            proc.terminate()
            proc.join(5)
            if proc.is_alive():
                proc.kill()
            return None, None, f"timeout after {timeout_seconds}s"

        wait_time = min(remaining, 5)
        proc.join(wait_time)
        if not proc.is_alive():
            break

    if queue.empty():
        return None, None, "no result from worker"

    result = queue.get()
    if not result.get("ok"):
        return None, None, result.get("error", "unknown error")

    return result.get("price_stats"), result.get("alert_stats"), None


def execute_futures_job():
    """Execute a combined futures price update and alert check job."""
    job_id = "futures_combined"

    if not acquire_job_lock(job_id):
        logger.warning("Job %s is already running; skipping duplicate execution.", job_id)
        return None

    config = load_scheduler_config()
    notify_settings = config.get("notification_settings", {})
    send_start = notify_settings.get("send_start_notification", True)
    send_complete = notify_settings.get("send_completion_notification", True)
    job_timeout = max(int(notify_settings.get("job_timeout_seconds", JOB_TIMEOUT_SECONDS)), 60)

    start_time = time.time()

    if send_start:
        send_scheduler_notification(
            "\n".join(
                [
                    "🚀 **Futures job started**",
                    f"• Job ID: `{job_id}`",
                    f"• Start (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}Z",
                    f"• Timeout: {job_timeout}s",
                ]
            ),
            event="start",
        )

    try:
        update_scheduler_status(
            status="running",
            current_job={
                "id": job_id,
                "job_type": "futures_combined",
                "started": datetime.utcnow().isoformat(),
            },
        )

        # Check IB availability before running
        if not is_ib_available():
            logger.info("Skipping job - outside IB hours")
            update_scheduler_status(
                status="running",
                current_job=None,
                last_run={
                    "job_id": job_id,
                    "completed_at": datetime.utcnow().isoformat(),
                    "skipped": True,
                    "skip_reason": "Outside IB hours",
                },
            )
            release_job_lock(job_id)
            return

        price_stats, alert_stats, worker_err = _run_job_subprocess(timeout_seconds=job_timeout)

        if worker_err:
            raise TimeoutError(worker_err) if "timeout" in worker_err else RuntimeError(worker_err)

        duration = round(time.time() - start_time, 2)

        update_scheduler_status(
            status="running",
            current_job=None,
            last_run={
                "job_id": job_id,
                "completed_at": datetime.utcnow().isoformat(),
                "duration_seconds": duration,
            },
            last_result={
                "price_stats": price_stats,
                "alert_stats": alert_stats,
            },
        )

        logger.info(
            "Futures job finished (price updates: %s, alerts: %s)",
            price_stats.get("updated", 0) if isinstance(price_stats, dict) else price_stats,
            alert_stats.get("total", 0) if isinstance(alert_stats, dict) else alert_stats,
        )

        if send_complete:
            send_scheduler_notification(
                "\n".join(
                    [
                        "🏁 **Futures job complete**",
                        f"• Duration: {duration}s",
                        f"• Summary: {_format_stats_for_message(price_stats, alert_stats)}",
                    ]
                ),
                event="complete",
            )

        return {
            "price_stats": price_stats,
            "alert_stats": alert_stats,
            "duration": duration,
        }

    except Exception as exc:
        err_msg = f"Timeout after {job_timeout}s" if isinstance(exc, TimeoutError) else str(exc)
        logger.exception("Error running futures job: %s", err_msg)
        update_scheduler_status(
            status="error",
            current_job=None,
            last_error={
                "time": datetime.utcnow().isoformat(),
                "job_id": job_id,
                "message": err_msg,
            },
        )
        send_scheduler_notification(
            "\n".join(
                [
                    "❌ **Futures job failed**",
                    f"• Error: {err_msg}",
                    f"• Duration: {round(time.time() - start_time, 2)}s",
                ]
            ),
            event="error",
        )
        return None
    finally:
        release_job_lock(job_id)


# ---------------------------------------------------------------------------
# Scheduler Setup
# ---------------------------------------------------------------------------

def _schedule_futures_jobs():
    """Schedule all futures jobs based on configured times."""
    if not scheduler:
        logger.error("Scheduler not initialized")
        return

    config = load_scheduler_config()
    update_times = config.get("update_times", ["06:00", "12:00", "16:00", "20:00"])

    scheduled_count = 0

    for time_str in update_times:
        try:
            hour, minute = map(int, time_str.split(":"))
        except ValueError:
            logger.warning("Invalid time format: %s", time_str)
            continue

        job_id = f"futures_{time_str.replace(':', '')}"
        scheduler.add_job(
            execute_futures_job,
            CronTrigger(
                hour=hour,
                minute=minute,
                timezone="UTC",
            ),
            id=job_id,
            replace_existing=True,
            max_instances=1,
            misfire_grace_time=3600,
        )
        scheduled_count += 1
        logger.info("Scheduled futures job at %s UTC", time_str)

    logger.info("Scheduled %d futures jobs", scheduled_count)


def _heartbeat_job():
    """Update heartbeat timestamp in lock file and status."""
    try:
        if LOCK_FILE.exists():
            lock_data = json.loads(LOCK_FILE.read_text())
            lock_data["heartbeat"] = datetime.utcnow().isoformat()
            LOCK_FILE.write_text(json.dumps(lock_data, indent=2))

        update_scheduler_status(status="running")
    except Exception as exc:
        logger.debug("Heartbeat update failed: %s", exc)


# ---------------------------------------------------------------------------
# Start/Stop Functions
# ---------------------------------------------------------------------------

def start_futures_scheduler() -> bool:
    """Start the futures scheduler."""
    global scheduler

    if is_scheduler_running():
        logger.info("Futures scheduler is already running")
        return True

    # If we're being called from another process, spawn a new process
    if os.getpid() != os.getppid():
        try:
            subprocess.Popen(
                [sys.executable, str(Path(__file__))],
                cwd=str(BASE_DIR),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            logger.info("Started futures scheduler in background process")
            return True
        except Exception as exc:
            logger.error("Failed to start futures scheduler: %s", exc)
            return False

    # Otherwise, start in this process
    if not _acquire_process_lock():
        return False

    try:
        scheduler = BackgroundScheduler(timezone="UTC")

        # Schedule all futures jobs
        _schedule_futures_jobs()

        # Add heartbeat job
        scheduler.add_job(
            _heartbeat_job,
            "interval",
            seconds=HEARTBEAT_INTERVAL,
            id="heartbeat",
            replace_existing=True,
        )

        # Start scheduler
        scheduler.start()

        # Update status
        update_scheduler_status(
            status="running",
            current_job=None,
        )

        # Run initial job if configured
        config = load_scheduler_config()
        if config.get("update_on_start", True):
            logger.info("Running initial futures job on startup...")
            import threading
            threading.Thread(target=execute_futures_job, daemon=True).start()

        logger.info("Futures scheduler started successfully")
        send_scheduler_notification("✅ Futures scheduler started", "success")

        return True

    except Exception as exc:
        logger.exception("Failed to start futures scheduler: %s", exc)
        _release_process_lock()
        return False


def stop_futures_scheduler() -> bool:
    """Stop the futures scheduler."""
    global scheduler

    try:
        if scheduler and scheduler.running:
            scheduler.shutdown(wait=True)
            scheduler = None

        update_scheduler_status(status="stopped", current_job=None)
        _release_process_lock()

        logger.info("Futures scheduler stopped")
        send_scheduler_notification("⏹️ Futures scheduler stopped", "info")

        return True

    except Exception as exc:
        logger.error("Failed to stop futures scheduler: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    """Main entry point for running scheduler as a standalone process."""
    if not start_futures_scheduler():
        logger.error("Failed to start futures scheduler")
        sys.exit(1)

    logger.info("Futures scheduler running. Press Ctrl+C to stop.")

    def signal_handler(sig, frame):
        logger.info("Received signal %s, shutting down...", sig)
        stop_futures_scheduler()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Keep the process alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        stop_futures_scheduler()


if __name__ == "__main__":
    main()

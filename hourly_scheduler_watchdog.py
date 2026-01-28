#!/usr/bin/env python3
"""
Lightweight watchdog that monitors the hourly data scheduler heartbeat and
restarts it if the heartbeat goes stale or the process dies.

This watchdog monitors:
- hourly_scheduler.lock for process info and heartbeat
- hourly_data_scheduler.py process status

Usage:
    python3 hourly_scheduler_watchdog.py

Environment variables:
    HOURLY_WATCHDOG_INTERVAL      Poll interval in seconds (default: 60)
    HOURLY_HEARTBEAT_MAX_AGE      Max allowed heartbeat age in seconds (default: 900)
"""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import psutil

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
LOCK_FILE = BASE_DIR / "hourly_scheduler.lock"
LOG_FILE = BASE_DIR / "hourly_scheduler_watchdog.log"

# Configuration
POLL_INTERVAL = int(os.getenv("HOURLY_WATCHDOG_INTERVAL", "60"))
MAX_AGE = int(os.getenv("HOURLY_HEARTBEAT_MAX_AGE", "900"))  # 15 minutes

# Logging
logger = logging.getLogger("hourly_scheduler_watchdog")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def read_lock() -> Optional[dict]:
    """Read and parse the hourly scheduler lock file."""
    if not LOCK_FILE.exists():
        return None
    try:
        return json.loads(LOCK_FILE.read_text())
    except Exception as exc:
        logger.warning("Failed to read lock file: %s", exc)
        return None


def is_process_running(pid: int) -> bool:
    """Check if a process with the given PID is running."""
    try:
        process = psutil.Process(pid)
        cmdline = process.cmdline() or []
        # Verify it's actually the hourly scheduler
        return any("hourly_data_scheduler" in segment for segment in cmdline)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False


def kill_process(pid: int) -> None:
    """Terminate a process gracefully, then forcefully if needed."""
    try:
        os.kill(pid, signal.SIGTERM)
        logger.info("Sent SIGTERM to PID %s", pid)
    except Exception as exc:
        logger.warning("Failed to send SIGTERM to PID %s: %s", pid, exc)

    time.sleep(2)

    # Force kill if still alive
    try:
        if is_process_running(pid):
            os.kill(pid, signal.SIGTERM)
            logger.info("Sent SIGKILL to PID %s", pid)
    except Exception as exc:
        logger.debug("Failed to send SIGKILL to PID %s: %s", pid, exc)


def get_heartbeat_age(lock_info: dict) -> float:
    """Calculate the age of the last heartbeat in seconds."""
    # Try multiple timestamp fields that might be used
    ts_str = lock_info.get("last_update") or lock_info.get("timestamp") or lock_info.get("started_at")

    if not ts_str:
        return float("inf")

    try:
        # Handle ISO format timestamps
        ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - ts).total_seconds()
    except Exception as exc:
        logger.debug("Failed to parse timestamp %s: %s", ts_str, exc)
        return float("inf")


def start_hourly_scheduler() -> bool:
    """Start the hourly data scheduler as a background process."""
    try:
        subprocess.Popen(
            [sys.executable, str(BASE_DIR / "hourly_data_scheduler.py")],
            cwd=str(BASE_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        logger.info("Started hourly scheduler via subprocess")
        return True
    except Exception as exc:
        logger.error("Failed to start hourly scheduler: %s", exc)
        return False


def cleanup_stale_lock() -> None:
    """Remove stale lock file if the process is not running."""
    if not LOCK_FILE.exists():
        return

    lock_info = read_lock()
    if not lock_info:
        try:
            LOCK_FILE.unlink(missing_ok=True)
            logger.info("Removed invalid lock file")
        except Exception as exc:
            logger.warning("Failed to remove invalid lock file: %s", exc)
        return

    pid = lock_info.get("pid")
    if pid and is_process_running(int(pid)):
        return

    try:
        LOCK_FILE.unlink(missing_ok=True)
        logger.info("Removed stale lock file (PID: %s)", pid)
    except Exception as exc:
        logger.warning("Failed to remove stale lock file: %s", exc)


def check_and_restart_scheduler() -> None:
    """Check scheduler health and restart if needed."""
    lock_info = read_lock()

    if not lock_info:
        logger.warning("No lock file found; attempting to start scheduler")
        cleanup_stale_lock()
        if start_hourly_scheduler():
            logger.info("Hourly scheduler started successfully")
        else:
            logger.error("Failed to start hourly scheduler")
        return

    pid = lock_info.get("pid")

    # Check if process is running
    if not pid or not is_process_running(int(pid)):
        logger.warning("Hourly scheduler process not running (PID: %s); restarting", pid)
        cleanup_stale_lock()
        if start_hourly_scheduler():
            logger.info("Hourly scheduler restarted successfully")
        else:
            logger.error("Failed to restart hourly scheduler")
        return

    # Check heartbeat age
    age = get_heartbeat_age(lock_info)

    if age > MAX_AGE:
        logger.warning(
            "Hourly scheduler has stale heartbeat (age=%.0fs, max=%ds); restarting",
            age,
            MAX_AGE,
        )
        kill_process(int(pid))
        cleanup_stale_lock()
        if start_hourly_scheduler():
            logger.info("Hourly scheduler restarted successfully after stale heartbeat")
        else:
            logger.error("Failed to restart hourly scheduler after stale heartbeat")
    else:
        logger.debug("Hourly scheduler healthy (PID: %s, heartbeat age: %.0fs)", pid, age)


def main() -> None:
    """Main watchdog loop."""
    logger.info(
        "Hourly Scheduler Watchdog starting (interval=%ss, max_heartbeat_age=%ss)",
        POLL_INTERVAL,
        MAX_AGE,
    )

    try:
        iteration = 0
        while True:
            iteration += 1

            try:
                check_and_restart_scheduler()
            except Exception as exc:
                logger.exception("Error during health check iteration %d: %s", iteration, exc)

            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        logger.info("Hourly Scheduler Watchdog stopping (KeyboardInterrupt)")
    except Exception as exc:
        logger.exception("Hourly Scheduler Watchdog crashed: %s", exc)
        raise


if __name__ == "__main__":
    main()

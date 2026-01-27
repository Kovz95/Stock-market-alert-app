#!/usr/bin/env python3
"""
Enhanced scheduler watchdog (v2) that monitors both daily/weekly and hourly schedulers.

This watchdog combines process monitoring with heartbeat validation to ensure
schedulers remain healthy and auto-restart them when issues are detected.

Features:
- Monitors both auto_scheduler_v2 (daily/weekly) and hourly_data_scheduler
- Checks process existence and heartbeat timestamps
- Auto-restarts failed or stale schedulers
- Discord notifications for important events
- Comprehensive logging

Usage:
    python scheduler_watchdog_v2.py

Environment variables:
    SCHEDULER_WATCHDOG_INTERVAL      Poll interval in seconds (default: 60)
    SCHEDULER_HEARTBEAT_MAX_AGE      Max heartbeat age in seconds (default: 900)
    WATCHDOG_ENABLE_DISCORD          Enable Discord notifications (default: false)
    WATCHDOG_DISCORD_WEBHOOK         Discord webhook URL for notifications
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
from typing import Dict, Iterable, Optional

import psutil
import requests

BASE_DIR = Path(__file__).resolve().parent
POLL_INTERVAL = int(os.getenv("SCHEDULER_WATCHDOG_INTERVAL", "60"))
MAX_HEARTBEAT_AGE = int(os.getenv("SCHEDULER_HEARTBEAT_MAX_AGE", "900"))  # 15 minutes

# Lock file paths
DAILY_LOCK = BASE_DIR / "scheduler_v2.lock"
HOURLY_LOCK = BASE_DIR / "hourly_scheduler.lock"

# Logging
LOG_FILE = BASE_DIR / "scheduler_watchdog_v2.log"
logger = logging.getLogger("scheduler_watchdog_v2")
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

# Discord configuration
ENABLE_DISCORD = os.getenv("WATCHDOG_ENABLE_DISCORD", "false").lower() in ("true", "1", "yes")
DISCORD_WEBHOOK = os.getenv("WATCHDOG_DISCORD_WEBHOOK", "")


# ---------------------------------------------------------------------------
# Discord notifications
# ---------------------------------------------------------------------------

def send_discord_notification(message: str, level: str = "info") -> bool:
    """Send a notification to Discord webhook if configured."""
    if not ENABLE_DISCORD or not DISCORD_WEBHOOK:
        return False
    
    emoji_map = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌",
        "restart": "🔄",
    }
    
    emoji = emoji_map.get(level, "📝")
    payload = {
        "content": f"{emoji} **Scheduler Watchdog v2**\n{message}",
        "username": "Watchdog v2",
    }
    
    try:
        response = requests.post(DISCORD_WEBHOOK, json=payload, timeout=10)
        return response.status_code in (200, 204)
    except Exception as exc:
        logger.debug("Failed to send Discord notification: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------

def _proc_matches(proc: psutil.Process, substrings: Iterable[str]) -> bool:
    """Check if process command line contains all specified substrings."""
    try:
        cmdline = proc.cmdline() or []
        joined = " ".join(cmdline)
        return all(sub in joined for sub in substrings)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False


def find_process(substrings: Iterable[str]) -> Optional[psutil.Process]:
    """Find a running process that matches all given substrings."""
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        if _proc_matches(proc, substrings):
            return proc
    return None


def pid_alive(pid: int, substrings: Iterable[str]) -> bool:
    """Check if a specific PID exists and matches the expected process."""
    try:
        proc = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False
    return _proc_matches(proc, substrings)


def kill_process(pid: int) -> None:
    """Forcefully terminate a process."""
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)
    except Exception:
        pass
    
    # Force kill if still alive
    try:
        os.kill(pid, signal.SIGKILL)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Lock file helpers
# ---------------------------------------------------------------------------

def read_lock(path: Path) -> Dict:
    """Read and parse a scheduler lock file."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        logger.warning("Failed to read lock file %s: %s", path, exc)
        return {}


def cleanup_stale_lock(path: Path, substrings: Iterable[str]) -> None:
    """Remove lock file if the process is not running."""
    if not path.exists():
        return
    
    info = read_lock(path)
    pid = info.get("pid")
    
    if pid and pid_alive(int(pid), substrings):
        return
    
    try:
        path.unlink(missing_ok=True)
        logger.info("Removed stale lock file: %s (PID: %s)", path.name, pid)
    except Exception as exc:
        logger.warning("Failed to remove stale lock %s: %s", path, exc)


def get_heartbeat_age(lock_info: Dict) -> float:
    """Calculate the age of the last heartbeat in seconds."""
    # Try multiple timestamp fields
    ts_str = lock_info.get("heartbeat") or lock_info.get("timestamp") or lock_info.get("started_at")
    
    if not ts_str:
        return float("inf")
    
    try:
        ts = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - ts).total_seconds()
    except Exception as exc:
        logger.debug("Failed to parse timestamp %s: %s", ts_str, exc)
        return float("inf")


# ---------------------------------------------------------------------------
# Scheduler start helpers
# ---------------------------------------------------------------------------

def start_daily_weekly_scheduler() -> bool:
    """Start the daily/weekly scheduler using auto_scheduler_v2."""
    try:
        # Try importing and using the start function
        from auto_scheduler_v2 import start_auto_scheduler
        
        if start_auto_scheduler():
            logger.info("Started daily/weekly scheduler via start_auto_scheduler()")
            send_discord_notification("Daily/Weekly scheduler started", "restart")
            return True
    except Exception as exc:
        logger.warning("start_auto_scheduler() failed: %s", exc)
    
    # Fallback: launch as subprocess
    try:
        subprocess.Popen(
            [sys.executable, str(BASE_DIR / "auto_scheduler_v2.py")],
            cwd=str(BASE_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        logger.info("Started daily/weekly scheduler via subprocess")
        send_discord_notification("Daily/Weekly scheduler started (subprocess)", "restart")
        return True
    except Exception as exc:
        logger.error("Failed to start daily/weekly scheduler: %s", exc)
        send_discord_notification(f"Failed to start Daily/Weekly scheduler: {exc}", "error")
        return False


def start_hourly_scheduler() -> bool:
    """Start the hourly data scheduler."""
    try:
        subprocess.Popen(
            [sys.executable, str(BASE_DIR / "hourly_data_scheduler.py")],
            cwd=str(BASE_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        logger.info("Started hourly scheduler via subprocess")
        send_discord_notification("Hourly scheduler started", "restart")
        return True
    except Exception as exc:
        logger.error("Failed to start hourly scheduler: %s", exc)
        send_discord_notification(f"Failed to start Hourly scheduler: {exc}", "error")
        return False


# ---------------------------------------------------------------------------
# Health check and restart logic
# ---------------------------------------------------------------------------

def check_and_restart_daily_weekly() -> None:
    """Check daily/weekly scheduler health and restart if needed."""
    scheduler_name = "Daily/Weekly"
    process_pattern = ["auto_scheduler_v2"]
    
    # Check if process is running
    proc = find_process(process_pattern)
    lock_info = read_lock(DAILY_LOCK)
    
    if proc and proc.is_running():
        # Process is running - check heartbeat
        if lock_info:
            age = get_heartbeat_age(lock_info)
            if age > MAX_HEARTBEAT_AGE:
                logger.warning(
                    "%s scheduler has stale heartbeat (age=%.0fs). Restarting...",
                    scheduler_name,
                    age,
                )
                send_discord_notification(
                    f"{scheduler_name} scheduler heartbeat stale ({age:.0f}s). Restarting.",
                    "warning",
                )
                # Kill the process and restart
                kill_process(proc.pid)
                cleanup_stale_lock(DAILY_LOCK, process_pattern)
                start_daily_weekly_scheduler()
            else:
                logger.debug("%s scheduler healthy (heartbeat age: %.0fs)", scheduler_name, age)
        else:
            logger.debug("%s scheduler running but no lock file", scheduler_name)
    else:
        # Process is not running
        logger.warning("%s scheduler not running. Starting...", scheduler_name)
        cleanup_stale_lock(DAILY_LOCK, process_pattern)
        
        if start_daily_weekly_scheduler():
            logger.info("%s scheduler restarted successfully", scheduler_name)
        else:
            logger.error("%s scheduler failed to restart", scheduler_name)


def check_and_restart_hourly() -> None:
    """Check hourly scheduler health and restart if needed."""
    scheduler_name = "Hourly"
    process_pattern = ["hourly_data_scheduler"]
    
    # Check if process is running
    proc = find_process(process_pattern)
    lock_info = read_lock(HOURLY_LOCK)
    
    if proc and proc.is_running():
        # Process is running - check heartbeat
        if lock_info:
            age = get_heartbeat_age(lock_info)
            if age > MAX_HEARTBEAT_AGE:
                logger.warning(
                    "%s scheduler has stale heartbeat (age=%.0fs). Restarting...",
                    scheduler_name,
                    age,
                )
                send_discord_notification(
                    f"{scheduler_name} scheduler heartbeat stale ({age:.0f}s). Restarting.",
                    "warning",
                )
                # Kill the process and restart
                kill_process(proc.pid)
                cleanup_stale_lock(HOURLY_LOCK, process_pattern)
                start_hourly_scheduler()
            else:
                logger.debug("%s scheduler healthy (heartbeat age: %.0fs)", scheduler_name, age)
        else:
            logger.debug("%s scheduler running but no lock file", scheduler_name)
    else:
        # Process is not running
        logger.warning("%s scheduler not running. Starting...", scheduler_name)
        cleanup_stale_lock(HOURLY_LOCK, process_pattern)
        
        if start_hourly_scheduler():
            logger.info("%s scheduler restarted successfully", scheduler_name)
        else:
            logger.error("%s scheduler failed to restart", scheduler_name)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    """Main watchdog loop."""
    logger.info(
        "Scheduler Watchdog v2 starting (interval=%ss, max_heartbeat_age=%ss)",
        POLL_INTERVAL,
        MAX_HEARTBEAT_AGE,
    )
    
    send_discord_notification(
        f"Watchdog v2 started\n• Poll interval: {POLL_INTERVAL}s\n• Max heartbeat age: {MAX_HEARTBEAT_AGE}s",
        "success",
    )
    
    try:
        iteration = 0
        while True:
            iteration += 1
            
            try:
                check_and_restart_daily_weekly()
                check_and_restart_hourly()
            except Exception as exc:
                logger.exception("Error during health check iteration %d: %s", iteration, exc)
            
            time.sleep(POLL_INTERVAL)
    
    except KeyboardInterrupt:
        logger.info("Scheduler Watchdog v2 stopping (KeyboardInterrupt)")
        send_discord_notification("Watchdog v2 stopped (manual)", "info")
    except Exception as exc:
        logger.exception("Scheduler Watchdog v2 crashed: %s", exc)
        send_discord_notification(f"Watchdog v2 crashed: {exc}", "error")
        raise


if __name__ == "__main__":
    main()

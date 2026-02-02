#!/usr/bin/env python3
"""
Unified scheduler watchdog service that monitors both daily/weekly and hourly schedulers.

This service monitors scheduler processes and their heartbeat timestamps to ensure
they remain healthy and can auto-restart them when issues are detected.

Features:
- Monitors both auto_scheduler_v2 (daily/weekly) and hourly_data_scheduler
- Checks process existence and heartbeat timestamps
- Auto-restarts failed or stale schedulers
- Optional Discord notifications for important events
- Comprehensive logging

This is the core service logic. Use scripts/maintenance/run_scheduler_watchdog.py
as the entry point to run the watchdog.
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
from typing import Callable, Dict, Iterable, Optional

import psutil

logger = logging.getLogger("scheduler_watchdog")


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


class WatchdogConfig:
    """Configuration for the scheduler watchdog."""

    def __init__(
        self,
        base_dir: Path,
        daily_lock_path: Path,
        hourly_lock_path: Path,
        poll_interval: int = 60,
        max_heartbeat_age: int = 900,
        log_file: Optional[Path] = None,
        enable_discord: bool = False,
        discord_webhook: str = "",
        start_daily_scheduler: Optional[Callable[[], bool]] = None,
        start_hourly_scheduler: Optional[Callable[[], bool]] = None,
    ):
        """Initialize watchdog configuration.

        Args:
            base_dir: Project root directory (used as cwd when starting schedulers)
            daily_lock_path: Path to daily/weekly scheduler lock file
            hourly_lock_path: Path to hourly scheduler lock file
            poll_interval: How often to check schedulers (seconds)
            max_heartbeat_age: Maximum allowed heartbeat age (seconds)
            log_file: Optional log file path
            enable_discord: Whether to enable Discord notifications
            discord_webhook: Discord webhook URL for notifications
            start_daily_scheduler: Function to start daily scheduler (optional)
            start_hourly_scheduler: Function to start hourly scheduler (optional)
        """
        self.base_dir = base_dir
        self.daily_lock_path = daily_lock_path
        self.hourly_lock_path = hourly_lock_path
        self.poll_interval = poll_interval
        self.max_heartbeat_age = max_heartbeat_age
        self.log_file = log_file
        self.enable_discord = enable_discord
        self.discord_webhook = discord_webhook
        self.start_daily_scheduler = start_daily_scheduler
        self.start_hourly_scheduler = start_hourly_scheduler


# ---------------------------------------------------------------------------
# Discord notifications
# ---------------------------------------------------------------------------


def send_discord_notification(config: WatchdogConfig, message: str, level: str = "info") -> bool:
    """Send a notification to Discord webhook if configured."""
    if not config.enable_discord or not config.discord_webhook:
        return False

    try:
        import requests
    except ImportError:
        logger.warning("requests not available for Discord notifications")
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
        "content": f"{emoji} **Scheduler Watchdog**\n{message}",
        "username": "Watchdog",
    }

    try:
        response = requests.post(config.discord_webhook, json=payload, timeout=10)
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
        logger.info("Sent SIGTERM to PID %s", pid)
        time.sleep(2)
    except Exception as exc:
        logger.warning("Failed to send SIGTERM to PID %s: %s", pid, exc)

    # Force kill if still alive
    try:
        if pid_alive(pid, ["python"]):  # Basic check if still running
            os.kill(pid, signal.SIGTERM)
            logger.info("Sent SIGKILL to PID %s", pid)
    except Exception as exc:
        logger.debug("Failed to send SIGKILL to PID %s: %s", pid, exc)


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
    # Try multiple timestamp fields that schedulers might use
    ts_str = (
        lock_info.get("heartbeat")
        or lock_info.get("last_update")
        or lock_info.get("timestamp")
        or lock_info.get("started_at")
    )

    if not ts_str:
        return float("inf")

    try:
        # Handle ISO format timestamps (with or without Z)
        ts_str_clean = str(ts_str).replace("Z", "+00:00")
        ts = datetime.fromisoformat(ts_str_clean)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - ts).total_seconds()
    except Exception as exc:
        logger.debug("Failed to parse timestamp %s: %s", ts_str, exc)
        return float("inf")


# ---------------------------------------------------------------------------
# Scheduler start helpers
# ---------------------------------------------------------------------------


def start_daily_weekly_scheduler(config: WatchdogConfig) -> bool:
    """Start the daily/weekly scheduler using auto_scheduler_v2."""
    # Try using the provided start function first
    if config.start_daily_scheduler:
        try:
            if config.start_daily_scheduler():
                logger.info("Started daily/weekly scheduler via provided function")
                send_discord_notification(config, "Daily/Weekly scheduler started", "restart")
                return True
        except Exception as exc:
            logger.warning("Provided start_daily_scheduler() failed: %s", exc)

    # Fallback: try importing from the service module
    try:
        # Import from the new architecture location
        sys.path.insert(0, str(config.base_dir))
        from src.services.auto_scheduler_v2 import start_auto_scheduler

        if start_auto_scheduler():
            logger.info("Started daily/weekly scheduler via start_auto_scheduler()")
            send_discord_notification(config, "Daily/Weekly scheduler started", "restart")
            return True
    except Exception as exc:
        logger.warning("start_auto_scheduler() import failed: %s", exc)

    # Final fallback: launch as subprocess
    try:
        scheduler_script = config.base_dir / "src" / "services" / "auto_scheduler_v2.py"
        if not scheduler_script.exists():
            # Try old location
            scheduler_script = config.base_dir / "auto_scheduler_v2.py"

        subprocess.Popen(
            [sys.executable, str(scheduler_script)],
            cwd=str(config.base_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        logger.info("Started daily/weekly scheduler via subprocess")
        send_discord_notification(config, "Daily/Weekly scheduler started (subprocess)", "restart")
        return True
    except Exception as exc:
        logger.error("Failed to start daily/weekly scheduler: %s", exc)
        send_discord_notification(config, f"Failed to start Daily/Weekly scheduler: {exc}", "error")
        return False


def start_hourly_scheduler(config: WatchdogConfig) -> bool:
    """Start the hourly data scheduler."""
    # Try using the provided start function first
    if config.start_hourly_scheduler:
        try:
            if config.start_hourly_scheduler():
                logger.info("Started hourly scheduler via provided function")
                send_discord_notification(config, "Hourly scheduler started", "restart")
                return True
        except Exception as exc:
            logger.warning("Provided start_hourly_scheduler() failed: %s", exc)

    # Fallback: launch as subprocess
    try:
        scheduler_script = config.base_dir / "src" / "services" / "hourly_data_scheduler.py"
        if not scheduler_script.exists():
            # Try old location
            scheduler_script = config.base_dir / "hourly_data_scheduler.py"

        subprocess.Popen(
            [sys.executable, str(scheduler_script)],
            cwd=str(config.base_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        logger.info("Started hourly scheduler via subprocess")
        send_discord_notification(config, "Hourly scheduler started", "restart")
        return True
    except Exception as exc:
        logger.error("Failed to start hourly scheduler: %s", exc)
        send_discord_notification(config, f"Failed to start Hourly scheduler: {exc}", "error")
        return False


# ---------------------------------------------------------------------------
# Health check and restart logic
# ---------------------------------------------------------------------------


def check_and_restart_daily_weekly(config: WatchdogConfig) -> None:
    """Check daily/weekly scheduler health and restart if needed."""
    scheduler_name = "Daily/Weekly"
    process_pattern = ["auto_scheduler_v2"]

    # Check if process is running
    proc = find_process(process_pattern)
    lock_info = read_lock(config.daily_lock_path)

    if proc and proc.is_running():
        # Process is running - check heartbeat
        if lock_info:
            age = get_heartbeat_age(lock_info)
            if age > config.max_heartbeat_age:
                logger.warning(
                    "%s scheduler has stale heartbeat (age=%.0fs, max=%ds). Restarting...",
                    scheduler_name,
                    age,
                    config.max_heartbeat_age,
                )
                send_discord_notification(
                    config,
                    f"{scheduler_name} scheduler heartbeat stale ({age:.0f}s). Restarting.",
                    "warning",
                )
                # Kill the process and restart
                kill_process(proc.pid)
                cleanup_stale_lock(config.daily_lock_path, process_pattern)
                start_daily_weekly_scheduler(config)
            else:
                logger.debug("%s scheduler healthy (heartbeat age: %.0fs)", scheduler_name, age)
        else:
            logger.debug("%s scheduler running but no lock file", scheduler_name)
    else:
        # Process is not running
        logger.warning("%s scheduler not running. Starting...", scheduler_name)
        cleanup_stale_lock(config.daily_lock_path, process_pattern)

        if start_daily_weekly_scheduler(config):
            logger.info("%s scheduler restarted successfully", scheduler_name)
        else:
            logger.error("%s scheduler failed to restart", scheduler_name)


def check_and_restart_hourly(config: WatchdogConfig) -> None:
    """Check hourly scheduler health and restart if needed."""
    scheduler_name = "Hourly"
    process_pattern = ["hourly_data_scheduler"]

    # Check if process is running
    proc = find_process(process_pattern)
    lock_info = read_lock(config.hourly_lock_path)

    if proc and proc.is_running():
        # Process is running - check heartbeat
        if lock_info:
            age = get_heartbeat_age(lock_info)
            if age > config.max_heartbeat_age:
                logger.warning(
                    "%s scheduler has stale heartbeat (age=%.0fs, max=%ds). Restarting...",
                    scheduler_name,
                    age,
                    config.max_heartbeat_age,
                )
                send_discord_notification(
                    config,
                    f"{scheduler_name} scheduler heartbeat stale ({age:.0f}s). Restarting.",
                    "warning",
                )
                # Kill the process and restart
                kill_process(proc.pid)
                cleanup_stale_lock(config.hourly_lock_path, process_pattern)
                start_hourly_scheduler(config)
            else:
                logger.debug("%s scheduler healthy (heartbeat age: %.0fs)", scheduler_name, age)
        else:
            logger.debug("%s scheduler running but no lock file", scheduler_name)
    else:
        # Process is not running
        logger.warning("%s scheduler not running. Starting...", scheduler_name)
        cleanup_stale_lock(config.hourly_lock_path, process_pattern)

        if start_hourly_scheduler(config):
            logger.info("%s scheduler restarted successfully", scheduler_name)
        else:
            logger.error("%s scheduler failed to restart", scheduler_name)


# ---------------------------------------------------------------------------
# Main watchdog loop
# ---------------------------------------------------------------------------


def run_watchdog(config: WatchdogConfig) -> None:
    """Run the main watchdog loop."""
    logger.info(
        "Scheduler Watchdog starting (interval=%ss, max_heartbeat_age=%ss)",
        config.poll_interval,
        config.max_heartbeat_age,
    )

    send_discord_notification(
        config,
        f"Watchdog started\n• Poll interval: {config.poll_interval}s\n• Max heartbeat age: {config.max_heartbeat_age}s",
        "success",
    )

    try:
        iteration = 0
        while True:
            iteration += 1

            try:
                check_and_restart_daily_weekly(config)
                check_and_restart_hourly(config)
            except Exception as exc:
                logger.exception("Error during health check iteration %d: %s", iteration, exc)

            time.sleep(config.poll_interval)

    except KeyboardInterrupt:
        logger.info("Scheduler Watchdog stopping (KeyboardInterrupt)")
        send_discord_notification(config, "Watchdog stopped (manual)", "info")
    except Exception as exc:
        logger.exception("Scheduler Watchdog crashed: %s", exc)
        send_discord_notification(config, f"Watchdog crashed: {exc}", "error")
        raise

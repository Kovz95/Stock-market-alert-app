#!/usr/bin/env python3
"""
Lightweight watchdog to auto-restart schedulers if they stop.

Monitors:
- Daily/Weekly scheduler (auto_scheduler_v2)
- Hourly data scheduler (hourly_data_scheduler)

If a target process is not running, it removes stale lock files (to avoid
false positives) and spawns a fresh process with a new PID. Runs in a loop
until stopped. Intended to be launched via systemd/cron/tmux/screen.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

import psutil

BASE_DIR = Path(__file__).resolve().parent
CHECK_INTERVAL = int(os.getenv("SCHEDULER_WATCHDOG_INTERVAL", "60"))
DAILY_LOCK = BASE_DIR / "scheduler_v2.lock"
HOURLY_LOCK = BASE_DIR / "hourly_scheduler.lock"
LOG_FILE = BASE_DIR / "scheduler_watchdog.log"

# Configure logging
logger = logging.getLogger("scheduler_auto_restart")
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------
def _proc_matches(proc: psutil.Process, substrings: Iterable[str]) -> bool:
    try:
        cmdline = proc.cmdline() or []
        joined = " ".join(cmdline)
        return all(sub in joined for sub in substrings)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False


def find_process(substrings: Iterable[str]) -> Optional[psutil.Process]:
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        if _proc_matches(proc, substrings):
            return proc
    return None


def pid_alive(pid: int, substrings: Iterable[str]) -> bool:
    try:
        proc = psutil.Process(pid)
    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
        return False
    return _proc_matches(proc, substrings)


# ---------------------------------------------------------------------------
# Lock helpers
# ---------------------------------------------------------------------------
def _read_lock(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def cleanup_stale_lock(path: Path, substrings: Iterable[str]) -> None:
    if not path.exists():
        return
    info = _read_lock(path)
    pid = info.get("pid")
    if pid and pid_alive(int(pid), substrings):
        return
    try:
        path.unlink(missing_ok=True)
        logger.info("Removed stale lock %s (pid=%s)", path.name, pid)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to remove stale lock %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Start helpers
# ---------------------------------------------------------------------------
def _launch_subprocess(script: str, friendly_name: str) -> bool:
    try:
        subprocess.Popen(
            [sys.executable, str(BASE_DIR / script)],
            cwd=str(BASE_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        logger.info("Spawned %s via %s", friendly_name, script)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to start %s: %s", friendly_name, exc)
        return False


def start_daily_weekly_scheduler() -> bool:
    # Try the public helper if available, fall back to running the script.
    try:
        from auto_scheduler_v2 import start_auto_scheduler  # type: ignore

        if start_auto_scheduler():
            logger.info("Started daily/weekly scheduler via start_auto_scheduler()")
            return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("start_auto_scheduler() failed, falling back to script: %s", exc)

    return _launch_subprocess("auto_scheduler_v2.py", "daily/weekly scheduler")


def start_hourly_scheduler() -> bool:
    return _launch_subprocess("hourly_data_scheduler.py", "hourly scheduler")


# ---------------------------------------------------------------------------
# Ensure functions
# ---------------------------------------------------------------------------
def ensure_daily_weekly() -> None:
    proc = find_process(["auto_scheduler_v2"])
    if proc and proc.is_running():
        return
    cleanup_stale_lock(DAILY_LOCK, ["auto_scheduler_v2"])
    if start_daily_weekly_scheduler():
        logger.info("Daily/Weekly scheduler restarted")


def ensure_hourly() -> None:
    proc = find_process(["hourly_data_scheduler"])
    if proc and proc.is_running():
        return
    cleanup_stale_lock(HOURLY_LOCK, ["hourly_data_scheduler"])
    if start_hourly_scheduler():
        logger.info("Hourly scheduler restarted")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main() -> None:
    logger.info("Scheduler auto-restart watchdog starting (interval=%ss)", CHECK_INTERVAL)
    try:
        while True:
            ensure_daily_weekly()
            ensure_hourly()
            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        logger.info("Scheduler auto-restart watchdog stopping (KeyboardInterrupt)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Lightweight watchdog that monitors the daily/weekly scheduler heartbeat and
restarts it if the heartbeat goes stale.

Usage:
    python3 scheduler_heartbeat_watchdog.py

Environment variables:
    SCHEDULER_WATCHDOG_INTERVAL   Poll interval in seconds (default: 60)
    SCHEDULER_HEARTBEAT_MAX_AGE   Max allowed heartbeat age in seconds (default: 900)
"""

from __future__ import annotations

import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

from auto_scheduler_v2 import start_auto_scheduler

LOCK_FILE = Path("scheduler_v2.lock")
POLL_INTERVAL = int(os.getenv("SCHEDULER_WATCHDOG_INTERVAL", "60"))
MAX_AGE = int(os.getenv("SCHEDULER_HEARTBEAT_MAX_AGE", "900"))  # 15 minutes


def _read_lock():
    if not LOCK_FILE.exists():
        return None
    try:
        return json.loads(LOCK_FILE.read_text())
    except Exception:
        return None


def _is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _kill_process(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception:
        pass
    time.sleep(2)
    try:
        os.kill(pid, signal.SIGTERM)
    except Exception:
        pass


def _age_seconds(ts_str: str) -> float:
    try:
        ts = datetime.fromisoformat(ts_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - ts).total_seconds()
    except Exception:
        return float("inf")


def main():
    print(f"[watchdog] Starting (interval={POLL_INTERVAL}s, max_age={MAX_AGE}s)")
    while True:
        lock = _read_lock()
        if not lock:
            print("[watchdog] No lock file found; attempting restart")
            start_auto_scheduler()
            time.sleep(POLL_INTERVAL)
            continue

        pid = lock.get("pid")
        hb = lock.get("heartbeat") or lock.get("timestamp") or lock.get("started")
        age = _age_seconds(hb) if hb else float("inf")

        if age > MAX_AGE:
            print(f"[watchdog] Stale heartbeat detected (age={age:.0f}s). Restarting scheduler.")
            if pid and _is_process_running(pid):
                _kill_process(pid)
            start_auto_scheduler()
        else:
            print(f"[watchdog] Heartbeat OK (age={age:.0f}s)")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()

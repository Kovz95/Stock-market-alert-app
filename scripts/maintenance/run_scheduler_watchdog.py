#!/usr/bin/env python3
"""
Maintenance script to run the unified scheduler watchdog.

This script sets up the watchdog with proper paths and configuration,
then runs the watchdog service to monitor both daily/weekly and hourly schedulers.

Usage:
    python scripts/maintenance/run_scheduler_watchdog.py

Environment variables:
    SCHEDULER_WATCHDOG_INTERVAL      Poll interval in seconds (default: 60)
    SCHEDULER_HEARTBEAT_MAX_AGE      Max allowed heartbeat age in seconds (default: 900)
    WATCHDOG_ENABLE_DISCORD          Enable Discord notifications (default: false)
    WATCHDOG_DISCORD_WEBHOOK         Discord webhook URL for notifications
    WATCHDOG_LOG_FILE                Optional log file path (default: scheduler_watchdog.log in project root)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import codecs

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.services.scheduler_watchdog import WatchdogConfig, run_watchdog

# Configuration from environment
POLL_INTERVAL = int(os.getenv("SCHEDULER_WATCHDOG_INTERVAL", "60"))
MAX_HEARTBEAT_AGE = int(os.getenv("SCHEDULER_HEARTBEAT_MAX_AGE", "900"))  # 15 minutes
ENABLE_DISCORD = os.getenv("WATCHDOG_ENABLE_DISCORD", "false").lower() in ("true", "1", "yes")
DISCORD_WEBHOOK = os.getenv("WATCHDOG_DISCORD_WEBHOOK", "")
LOG_FILE = os.getenv("WATCHDOG_LOG_FILE", str(BASE_DIR / "scheduler_watchdog.log"))

# Lock file paths (in project root where schedulers write them)
DAILY_LOCK = BASE_DIR / "scheduler_v2.lock"
HOURLY_LOCK = BASE_DIR / "hourly_scheduler.lock"

# Setup logging
logger = logging.getLogger("scheduler_watchdog")
logger.setLevel(logging.INFO)

if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    log_path = Path(LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def main() -> None:
    """Main entry point for the watchdog script."""
    # Create watchdog configuration
    config = WatchdogConfig(
        base_dir=BASE_DIR,
        daily_lock_path=DAILY_LOCK,
        hourly_lock_path=HOURLY_LOCK,
        poll_interval=POLL_INTERVAL,
        max_heartbeat_age=MAX_HEARTBEAT_AGE,
        log_file=Path(LOG_FILE),
        enable_discord=ENABLE_DISCORD,
        discord_webhook=DISCORD_WEBHOOK,
    )

    # Run the watchdog
    run_watchdog(config)


if __name__ == "__main__":
    main()

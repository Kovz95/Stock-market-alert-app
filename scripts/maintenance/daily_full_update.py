#!/usr/bin/env python3
"""
Daily full database update for all tickers.
Run once per day (e.g., at night) to ensure all tickers are current.

Usage:
    python scripts/maintenance/daily_full_update.py

Examples:
    python scripts/maintenance/daily_full_update.py
"""

import sys
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

# Add project root to path so src is importable (script is in scripts/maintenance/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def add_to_scheduler() -> str:
    """Return example code to add this job to auto_scheduler_v2."""
    return """
# Add this import at the top
from src.services.daily_price_service import run_full_daily_update

# Add this scheduled job (runs at 11 PM every day)
scheduler.add_job(
    run_full_daily_update,
    'cron',
    hour=23,
    minute=0,
    id='daily_full_update',
    name='Daily Full Database Update',
    misfire_grace_time=3600
)

# Or after US market close (5 PM ET)
scheduler.add_job(
    run_full_daily_update,
    'cron',
    hour=17,
    minute=30,
    id='daily_full_update',
    name='Daily Full Database Update',
    misfire_grace_time=3600
)
"""


def main() -> None:
    """Run the full daily update or print scheduler snippet."""
    if len(sys.argv) > 1 and sys.argv[1] == "scheduler":
        print(add_to_scheduler())
        return

    from src.services.daily_price_service import run_full_daily_update
    run_full_daily_update()


if __name__ == "__main__":
    main()

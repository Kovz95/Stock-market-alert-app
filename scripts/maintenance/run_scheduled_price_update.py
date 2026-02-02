#!/usr/bin/env python3
"""
Run scheduled price update for one or more exchanges.

Usage:
    python scripts/maintenance/run_scheduled_price_update.py <EXCHANGE> [--weekly]
    python scripts/maintenance/run_scheduled_price_update.py NASDAQ
    python scripts/maintenance/run_scheduled_price_update.py NYSE --weekly
"""

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

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run scheduled price update for one or more exchanges.",
    )
    parser.add_argument(
        "exchange",
        nargs="+",
        help="Exchange name(s), e.g. NASDAQ, NYSE",
    )
    parser.add_argument(
        "--weekly",
        action="store_true",
        help="Also resample and update weekly data",
    )
    args = parser.parse_args()

    from src.services.scheduled_price_updater import update_prices_for_exchanges

    exchanges = [e.strip() for e in args.exchange]
    logger.info("Updating prices for %s (weekly=%s)", exchanges, args.weekly)
    stats = update_prices_for_exchanges(exchanges, resample_weekly=args.weekly)
    print("Results:", stats)


if __name__ == "__main__":
    main()

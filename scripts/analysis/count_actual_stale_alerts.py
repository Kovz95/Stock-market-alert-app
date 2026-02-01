#!/usr/bin/env python3
"""
Count alerts with truly stale daily price data.

Uses shared stale-data logic (stale_data_utils.is_data_stale) to determine
which American daily alerts have price data older than the reference cutoff.
Read-only analysis script for data quality audits.

Usage:
    python scripts/analysis/count_actual_stale_alerts.py
    python scripts/analysis/count_actual_stale_alerts.py --cutoff 2025-09-16
    python scripts/analysis/count_actual_stale_alerts.py --cutoff 2026-02-01

Examples:
    # Default: use 2025-09-16 as reference date (historical audit)
    python scripts/analysis/count_actual_stale_alerts.py

    # Use today as reference (current staleness)
    python scripts/analysis/count_actual_stale_alerts.py --cutoff $(date +%Y-%m-%d)
"""

import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import argparse

# Set UTF-8 encoding for Windows console
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_access.alert_repository import list_alerts as repo_list_alerts
from db_config import db_config
from stale_data_utils import is_data_stale

# American exchanges from the scheduler
AMERICAN_EXCHANGES = [
    "NASDAQ",
    "NYSE",
    "NYSE AMERICAN",
    "NYSE ARCA",
    "CBOE BZX",
    "NASDAQ GLOBAL MARKET",
    "NASDAQ GLOBAL SELECT",
    "TORONTO",
    "TSX",
    "MEXICO",
    "BUENOS AIRES",
    "SANTIAGO",
]

AMERICAN_COUNTRIES = [
    "UNITED STATES",
    "USA",
    "CANADA",
    "MEXICO",
    "ARGENTINA",
    "CHILE",
]


def _filter_american_daily_alerts(alerts: list[dict]) -> list[dict]:
    """Return alerts that are daily timeframe and American exchange/country."""
    result = []
    for alert in alerts:
        if alert.get("timeframe") not in ["1d", "Daily", "daily"]:
            continue
        exchange = str(alert.get("exchange", "")).upper()
        if any(ex in exchange for ex in AMERICAN_EXCHANGES):
            result.append(alert)
            continue
        country = (alert.get("country") or "").upper()
        if country in AMERICAN_COUNTRIES:
            result.append(alert)
    return result


def _parse_last_date(result):
    """Parse MAX(date) result from cursor into date/datetime for is_data_stale."""
    if not result or not result[0]:
        return None
    raw = result[0]
    if hasattr(raw, "date") and callable(getattr(raw, "date", None)):
        return raw.date()
    if hasattr(raw, "year"):  # already a date
        return raw
    return datetime.strptime(str(raw), "%Y-%m-%d").date()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count American daily alerts with stale price data."
    )
    parser.add_argument(
        "--cutoff",
        type=str,
        default="2025-09-16",
        metavar="YYYY-MM-DD",
        help="Reference date for staleness (default: 2025-09-16)",
    )
    args = parser.parse_args()

    try:
        cutoff_date = datetime.strptime(args.cutoff, "%Y-%m-%d").date()
    except ValueError:
        sys.stderr.write(f"Invalid --cutoff: expected YYYY-MM-DD, got {args.cutoff!r}\n")
        sys.exit(1)

    reference_time = datetime.combine(cutoff_date, datetime.min.time())

    all_alerts = repo_list_alerts()
    american_daily_alerts = _filter_american_daily_alerts(all_alerts)
    print(f"Total American daily alerts: {len(american_daily_alerts)}")

    conn = db_config.get_connection(role="prices")
    try:
        cursor = conn.cursor()

        stale_alerts: list[dict] = []
        ticker_status: dict[str, str] = {}

        for alert in american_daily_alerts:
            ticker = alert.get("ticker")
            if not ticker:
                continue

            if ticker in ticker_status:
                if ticker_status[ticker] == "stale":
                    stale_alerts.append(alert)
                continue

            cursor.execute(
                """
                SELECT MAX(date) FROM daily_prices
                WHERE ticker = %s
                """,
                (ticker,),
            )
            result = cursor.fetchone()
            last_date = _parse_last_date(result)

            if last_date is None:
                ticker_status[ticker] = "no_data"
                stale_alerts.append(alert)
            elif is_data_stale(last_date, "1d", reference_time=reference_time):
                ticker_status[ticker] = "stale"
                stale_alerts.append(alert)
            else:
                ticker_status[ticker] = "current"

        print(f"\nStale alerts (data stale as of {cutoff_date}): {len(stale_alerts)}")

        exchange_groups: dict[str, list] = defaultdict(list)
        for alert in stale_alerts:
            exchange = alert.get("exchange", "Unknown")
            exchange_groups[exchange].append(alert)

        print("\nStale alerts by exchange:")
        for exchange, alerts in sorted(exchange_groups.items()):
            print(f"  {exchange}: {len(alerts)} alerts")

        stale_tickers = {a["ticker"] for a in stale_alerts if a.get("ticker")}
        print(f"\nUnique stale tickers: {len(stale_tickers)}")

        print("\nSample stale tickers:")
        for ticker in list(stale_tickers)[:10]:
            cursor.execute(
                """
                SELECT MAX(date) FROM daily_prices
                WHERE ticker = %s
                """,
                (ticker,),
            )
            result = cursor.fetchone()
            if result and result[0]:
                print(f"  {ticker}: Last update {result[0]}")
            else:
                print(f"  {ticker}: No data")

    finally:
        db_config.close_connection(conn)

    print("\n" + "=" * 50)
    print(f"ANSWER: {len(stale_alerts)} alerts have stale data")
    print("=" * 50)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Initialize the hourly_prices table in PostgreSQL and display basic info.

Ensures daily_prices, weekly_prices, ticker_metadata, and hourly_prices tables
exist (via DailyPriceRepository), then verifies hourly_prices and prints
schema and row count.

Usage:
    python scripts/maintenance/init_hourly_table.py

Examples:
    python scripts/maintenance/init_hourly_table.py
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

from src.data_access.db_config import db_config
from src.data_access.daily_price_repository import DailyPriceRepository


def main() -> None:
    print("Initializing hourly_prices table...")
    DailyPriceRepository()

    with db_config.connection(role="prices") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT to_regclass('public.hourly_prices')")
        exists = cursor.fetchone()[0] is not None

        if not exists:
            print("ERROR: Failed to create hourly_prices table")
            return

        print("SUCCESS: hourly_prices table created successfully!")

        cursor.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'hourly_prices'
            ORDER BY ordinal_position
            """
        )
        columns = cursor.fetchall()

        print("\nTable schema:")
        for name, data_type in columns:
            print(f"  - {name} ({data_type})")

        cursor.execute("SELECT COUNT(*) FROM hourly_prices")
        count = cursor.fetchone()[0]
        print(f"\nCurrent records: {count}")

    print("\nDone!")


if __name__ == "__main__":
    main()

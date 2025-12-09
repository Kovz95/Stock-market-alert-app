"""
Initialize the hourly_prices table in PostgreSQL and display basic info.
"""

from daily_price_collector import DailyPriceDatabase
from db_config import db_config


def main() -> None:
    print("Initializing hourly_prices table...")
    DailyPriceDatabase()

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

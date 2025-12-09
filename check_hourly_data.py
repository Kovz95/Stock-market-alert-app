"""
Check hourly data in the database - detailed stats (PostgreSQL only)
"""

from datetime import datetime

from data_access.metadata_repository import fetch_stock_metadata_map
from db_config import db_config

# Load main database from PostgreSQL
stock_db = fetch_stock_metadata_map() or {}

print("=" * 80)
print("HOURLY DATA DATABASE ANALYSIS")
print("=" * 80)
print()

with db_config.connection(role="prices") as conn:
    cursor = conn.cursor()

    # Overall stats
    cursor.execute(
        """
        SELECT
            COUNT(DISTINCT ticker) AS tickers_with_data,
            COUNT(*) AS total_records,
            MIN(datetime) AS earliest,
            MAX(datetime) AS latest
        FROM hourly_prices
        """
    )

    stats = cursor.fetchone()
    tickers_with_data, total_records, earliest, latest = stats

    print("OVERALL STATISTICS:")
    print(f"  Total tickers in main database: {len(stock_db):,}")
    print(f"  Tickers with hourly data: {tickers_with_data:,}")
    print(f"  Total hourly records: {total_records:,}")
    print(f"  Earliest datetime: {earliest}")
    print(f"  Latest datetime: {latest}")
    print()

    # Calculate date range
    if earliest and latest:
        earliest_dt = earliest
        latest_dt = latest
        if isinstance(earliest_dt, str):
            earliest_dt = datetime.fromisoformat(earliest_dt)
        if isinstance(latest_dt, str):
            latest_dt = datetime.fromisoformat(latest_dt)
        days_span = (latest_dt - earliest_dt).days
        print(f"  Date range span: {days_span} days")
        print()

    # Per-ticker breakdown (sample)
    print("SAMPLE TICKERS (First 10 with data):")
    cursor.execute(
        """
        SELECT
            ticker,
            COUNT(*) AS num_records,
            MIN(datetime) AS first_date,
            MAX(datetime) AS last_date
        FROM hourly_prices
        GROUP BY ticker
        ORDER BY ticker
        LIMIT 10
        """
    )

    for row in cursor.fetchall():
        ticker, num_records, first_date, last_date = row
        print(f"  {ticker:10} | Records: {num_records:5,} | From: {first_date} | To: {last_date}")

    print()

    # Check for the specific ticker you tested (AAPL)
    print("DETAILED CHECK - AAPL:")
    cursor.execute(
        """
        SELECT
            COUNT(*) AS num_records,
            MIN(datetime) AS first_datetime,
            MAX(datetime) AS last_datetime
        FROM hourly_prices
        WHERE ticker = %s
        """,
        ("AAPL",),
    )

    aapl_stats = cursor.fetchone()
    if aapl_stats and aapl_stats[0] > 0:
        num_records, first_dt, last_dt = aapl_stats
        first = first_dt if isinstance(first_dt, datetime) else datetime.fromisoformat(str(first_dt))
        last = last_dt if isinstance(last_dt, datetime) else datetime.fromisoformat(str(last_dt))
        days = max((last - first).days, 1)

        print(f"  Total records: {num_records:,}")
        print(f"  First datetime: {first_dt}")
        print(f"  Last datetime: {last_dt}")
        print(f"  Days covered: {days}")
        print(f"  Avg bars per day: {num_records/days:.1f}")

        # Show first and last 3 records
        print()
        print("  First 3 records:")
        cursor.execute(
            """
            SELECT datetime, open, high, low, close, volume
            FROM hourly_prices
            WHERE ticker = %s
            ORDER BY datetime ASC
            LIMIT 3
            """,
            ("AAPL",),
        )
        for row in cursor.fetchall():
            print(f"    {row[0]} | O:{row[1]:7.2f} H:{row[2]:7.2f} L:{row[3]:7.2f} C:{row[4]:7.2f} V:{row[5]:,}")

        print()
        print("  Last 3 records:")
        cursor.execute(
            """
            SELECT datetime, open, high, low, close, volume
            FROM hourly_prices
            WHERE ticker = %s
            ORDER BY datetime DESC
            LIMIT 3
            """,
            ("AAPL",),
        )
        for row in cursor.fetchall():
            print(f"    {row[0]} | O:{row[1]:7.2f} H:{row[2]:7.2f} L:{row[3]:7.2f} C:{row[4]:7.2f} V:{row[5]:,}")
    else:
        print("  NO DATA FOUND FOR AAPL")

    print()

    # Distribution by number of records
    print("RECORD COUNT DISTRIBUTION:")
    cursor.execute(
        """
        WITH ticker_counts AS (
            SELECT ticker, COUNT(*) AS cnt
            FROM hourly_prices
            GROUP BY ticker
        )
        SELECT
            CASE
                WHEN cnt < 100 THEN '< 100 bars'
                WHEN cnt < 500 THEN '100-500 bars'
                WHEN cnt < 1000 THEN '500-1000 bars'
                WHEN cnt < 2000 THEN '1000-2000 bars'
                WHEN cnt < 3000 THEN '2000-3000 bars'
                ELSE '3000+ bars'
            END AS range,
            COUNT(*) AS num_tickers
        FROM ticker_counts
        GROUP BY range
        ORDER BY
            CASE range
                WHEN '< 100 bars' THEN 1
                WHEN '100-500 bars' THEN 2
                WHEN '500-1000 bars' THEN 3
                WHEN '1000-2000 bars' THEN 4
                WHEN '2000-3000 bars' THEN 5
                ELSE 6
            END
        """
    )

    for row in cursor.fetchall():
        range_name, num_tickers = row
        print(f"  {range_name:20}: {num_tickers:5,} tickers")

print()
print("=" * 80)
print("Analysis complete!")
print("=" * 80)

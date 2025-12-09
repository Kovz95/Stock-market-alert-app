"""
Check how many tickers have hourly data through the last available hour (Postgres)
"""

from db_config import db_config

with db_config.connection(role="prices") as conn:
    cursor = conn.cursor()

    cursor.execute("SELECT MAX(datetime) AS latest_hour FROM hourly_prices")
    latest_hour = cursor.fetchone()[0]

    if latest_hour:
        print(f"Latest hour in database: {latest_hour}")
        print("=" * 80)
        print()

        cursor.execute(
            """
            SELECT COUNT(DISTINCT ticker) AS ticker_count
            FROM hourly_prices
            WHERE datetime = %s
            """,
            (latest_hour,),
        )
        tickers_at_latest = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT ticker) FROM hourly_prices")
        total_tickers = cursor.fetchone()[0]

        print(f"Total tickers with hourly data: {total_tickers:,}")
        print(f"Tickers with data at latest hour ({latest_hour}): {tickers_at_latest:,}")
        coverage = (tickers_at_latest / total_tickers * 100) if total_tickers else 0
        print(f"Coverage: {coverage:.1f}%")
        print()

        print("Distribution of last update times (last 10 hours):")
        print("-" * 80)

        cursor.execute(
            """
            WITH latest_per_ticker AS (
                SELECT ticker, MAX(datetime) AS datetime
                FROM hourly_prices
                GROUP BY ticker
            )
            SELECT datetime, COUNT(*) AS ticker_count
            FROM latest_per_ticker
            GROUP BY datetime
            ORDER BY datetime DESC
            LIMIT 10
            """
        )

        for row in cursor.fetchall():
            dt, count = row
            print(f"{dt}: {count:5,} tickers")

        print()

        cursor.execute(
            """
            WITH latest_per_ticker AS (
                SELECT ticker, MAX(datetime) AS latest_datetime
                FROM hourly_prices
                GROUP BY ticker
            )
            SELECT COUNT(*) FROM latest_per_ticker WHERE latest_datetime < %s
            """,
            (latest_hour,),
        )
        behind_count = cursor.fetchone()[0]

        print(f"Tickers behind the latest hour: {behind_count:,}")
        print(f"Tickers up to date: {tickers_at_latest:,}")

        if behind_count > 0:
            print()
            print("Sample of tickers behind (first 20):")
            print("-" * 80)

            cursor.execute(
                """
                WITH latest_per_ticker AS (
                    SELECT ticker, MAX(datetime) AS latest_datetime
                    FROM hourly_prices
                    GROUP BY ticker
                )
                SELECT ticker, latest_datetime
                FROM latest_per_ticker
                WHERE latest_datetime < %s
                ORDER BY ticker
                LIMIT 20
                """,
                (latest_hour,),
            )

            for row in cursor.fetchall():
                ticker, latest_dt = row
                print(f"{ticker:15} | Last hour: {latest_dt}")
    else:
        print("No hourly data found in database")

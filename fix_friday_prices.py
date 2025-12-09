"""
Emergency fix for Friday 10/10/2025 closing prices
Deletes incorrect intraday data and re-fetches correct closing prices
"""

from backend_fmp_optimized import OptimizedDailyPriceCollector
import logging
from datetime import datetime
from data_access.metadata_repository import fetch_stock_metadata_map
from db_config import db_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Load stock database from PostgreSQL
    stock_db = fetch_stock_metadata_map() or {}

    # Filter for US and Sao Paulo exchanges
    us_sao_paulo_tickers = [
        ticker for ticker, info in stock_db.items()
        if isinstance(info, dict) and info.get('exchange') in ['NASDAQ', 'NYSE', 'AMEX', 'SAO PAULO']
    ]

    total = len(us_sao_paulo_tickers)
    logger.info(f"Found {total} tickers from US and Sao Paulo exchanges")
    logger.info("Deleting incorrect 10/10/2025 data and re-fetching...")

    collector = OptimizedDailyPriceCollector()

    success = 0
    failed = 0
    deleted = 0

    with db_config.connection(role="prices") as conn:
        cursor = conn.cursor()

        for i, ticker in enumerate(us_sao_paulo_tickers, 1):
            try:
                cursor.execute(
                    """
                    DELETE FROM daily_prices
                    WHERE ticker = %s AND date = '2025-10-10'
                    """,
                    (ticker,),
                )

                cursor.execute(
                    """
                    DELETE FROM weekly_prices
                    WHERE ticker = %s AND week_ending = '2025-10-10'
                    """,
                    (ticker,),
                )

                if cursor.rowcount:
                    deleted += cursor.rowcount

                cursor.execute(
                    """
                    UPDATE ticker_metadata
                    SET last_date = '2025-10-09', last_update = '2025-10-09 16:00:00'
                    WHERE ticker = %s
                    """,
                    (ticker,),
                )

                conn.commit()

                result = collector.update_ticker(ticker, resample_weekly=True)

                if result:
                    success += 1
                else:
                    failed += 1

                if i % 100 == 0:
                    logger.info(
                        f"Progress: {i}/{total} | Success: {success} | Failed: {failed} | Deleted: {deleted} records"
                    )

            except Exception as e:
                logger.error(f"Error updating {ticker}: {e}")
                failed += 1

    logger.info("="*60)
    logger.info(f"COMPLETED")
    logger.info(f"Total: {total} | Success: {success} | Failed: {failed}")
    logger.info(f"Records deleted: {deleted}")
    logger.info(f"Success rate: {success/total*100:.1f}%")

    conn.close()
    collector.close()

if __name__ == "__main__":
    main()

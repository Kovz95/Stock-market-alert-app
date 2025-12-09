"""
Daily full database update for all tickers
Run this once per day (e.g., at night) to ensure all tickers are current
"""

import logging
from datetime import datetime, time as datetime_time
from daily_price_collector import DailyPriceCollector, DailyPriceDatabase
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_full_daily_update():
    """
    Update price data for ALL tickers in the database
    Best run at night (e.g., 11 PM) after all markets have closed
    """
    logger.info("=" * 70)
    logger.info(f"FULL DAILY PRICE UPDATE - {datetime.now()}")
    logger.info("=" * 70)
    
    collector = DailyPriceCollector()
    
    try:
        # Get all tickers from main database
        tickers = collector.get_all_tickers()
        logger.info(f"Found {len(tickers)} tickers to update")
        
        if len(tickers) == 0:
            logger.error("No tickers found")
            return
        
        # Track statistics
        stats = {
            'updated': 0,
            'skipped': 0,
            'failed': 0,
            'already_current': 0
        }
        
        start_time = time.time()
        
        # Process in batches to avoid overwhelming the API
        batch_size = 50
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            batch_end = min(i+batch_size, len(tickers))
            
            logger.info(f"Processing batch {i//batch_size + 1} ({i+1}-{batch_end}/{len(tickers)})")
            
            for ticker in batch:
                try:
                    # Check if already updated today
                    db = DailyPriceDatabase()
                    needs_update, last_update = db.needs_update(ticker)
                    db.close()
                    
                    if not needs_update:
                        stats['already_current'] += 1
                        continue
                    
                    # Update the ticker
                    if collector.update_ticker(ticker):
                        stats['updated'] += 1
                    else:
                        stats['failed'] += 1
                    
                    # Rate limiting (5 requests per second max)
                    time.sleep(0.2)
                    
                except Exception as e:
                    logger.error(f"Error updating {ticker}: {e}")
                    stats['failed'] += 1
            
            # Progress report every 500 tickers
            if (i + len(batch)) % 500 == 0:
                elapsed = time.time() - start_time
                rate = (i + len(batch)) / elapsed if elapsed > 0 else 0
                remaining = len(tickers) - (i + len(batch))
                eta = remaining / rate if rate > 0 else 0
                
                logger.info(f"Progress: {i + len(batch)}/{len(tickers)} tickers")
                logger.info(f"Stats: Updated={stats['updated']}, Current={stats['already_current']}, Failed={stats['failed']}")
                logger.info(f"ETA: {eta/60:.1f} minutes")
        
        # Final report
        elapsed = time.time() - start_time
        logger.info("=" * 70)
        logger.info("UPDATE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total time: {elapsed/60:.1f} minutes")
        logger.info(f"Updated: {stats['updated']} tickers")
        logger.info(f"Already current: {stats['already_current']} tickers")
        logger.info(f"Failed: {stats['failed']} tickers")
        
        # Get database statistics
        db = DailyPriceDatabase()
        db_stats = db.get_statistics()
        logger.info(f"Database: {db_stats['total_tickers']} tickers, "
                   f"{db_stats['total_daily_records']:,} daily records, "
                   f"{db_stats['total_weekly_records']:,} weekly records, "
                   f"{db_stats['db_size_mb']:.1f} MB")
        db.close()
        
    except Exception as e:
        logger.error(f"Error during update: {e}")
    finally:
        collector.close()


def add_to_scheduler():
    """
    Add this to your auto_scheduler_v2.py
    """
    code = '''
# Add this import at the top
from daily_full_update import run_full_daily_update

# Add this scheduled job (runs at 11 PM every day)
scheduler.add_job(
    run_full_daily_update,
    'cron',
    hour=23,  # 11 PM
    minute=0,
    id='daily_full_update',
    name='Daily Full Database Update',
    misfire_grace_time=3600
)

# Or if you prefer after US market close (5 PM ET)
scheduler.add_job(
    run_full_daily_update,
    'cron',
    hour=17,  # 5 PM
    minute=30,  # 30 minutes after close
    id='daily_full_update',
    name='Daily Full Database Update',
    misfire_grace_time=3600
)
'''
    return code


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "scheduler":
        print(add_to_scheduler())
    else:
        # Run the update
        run_full_daily_update()
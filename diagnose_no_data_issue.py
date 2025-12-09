"""
Diagnose why alerts are returning "No Data" across all exchanges (Postgres only)
"""
import logging
from collections import Counter

import pandas as pd

from backend_thread_safe import get_cached_stock_data_thread_safe
from data_access.alert_repository import list_alerts
from db_config import db_config

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load alerts from PostgreSQL
alerts = list_alerts()

# Get exchange breakdown
exchanges = Counter(a.get('exchange', 'Unknown') for a in alerts)
print("=" * 60)
print("ALERT DISTRIBUTION BY EXCHANGE")
print("=" * 60)
for exchange, count in exchanges.most_common(10):
    print(f"  {exchange}: {count} alerts")

# Test a sample from each major exchange
test_samples = {
    'NYSE': 'AAPL',
    'NASDAQ': 'MSFT',
    'NSE INDIA': 'RELIANCE.NS',
    'TORONTO': 'TD.TO',
    'LSE': 'BP.L',
    'ASX': 'BHP.AX'
}

print("\n" + "=" * 60)
print("TESTING SAMPLE TICKERS")
print("=" * 60)

for exchange, ticker in test_samples.items():
    print(f"\nTesting {ticker} ({exchange}):")

    # Check if ticker is in alerts
    ticker_alerts = [a for a in alerts if a.get('ticker') == ticker]
    print(f"  Alerts for this ticker: {len(ticker_alerts)}")

    # Check database directly
    with db_config.connection(role="prices") as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT COUNT(*), MAX(date) as last_date
            FROM daily_prices
            WHERE ticker = %s
            """,
            (ticker,),
        )
        count, last_date = cursor.fetchone()

    print(f"  Records in DB: {count}, Last date: {last_date}")

    # Test get_cached_stock_data_thread_safe
    print(f"  Testing backend_thread_safe fetch...")
    try:
        data = get_cached_stock_data_thread_safe(ticker, 'daily')

        if data == 'stale_data':
            print(f"    Result: STALE DATA marker")
        elif data is None:
            print(f"    Result: None (No data)")
        elif isinstance(data, pd.DataFrame):
            if not data.empty:
                print(f"    Result: DataFrame with {len(data)} rows")
                print(f"    Date range: {data.index.min()} to {data.index.max()}")
            else:
                print(f"    Result: Empty DataFrame")
        else:
            print(f"    Result: Unknown type {type(data)}")
    except Exception as e:
        print(f"    ERROR: {e}")

# Check stale data detection
print("\n" + "=" * 60)
print("STALE DATA DETECTION CHECK")
print("=" * 60)

from stale_data_utils import is_data_stale
from datetime import datetime, timedelta

# Test dates
test_dates = [
    datetime.now().date(),  # Today
    datetime.now().date() - timedelta(days=1),  # Yesterday
    datetime.now().date() - timedelta(days=2),  # 2 days ago
    datetime.now().date() - timedelta(days=3),  # 3 days ago
]

for test_date in test_dates:
    is_stale = is_data_stale(test_date, timeframe='1d')
    print(f"  {test_date}: {'STALE' if is_stale else 'FRESH'}")

print("\n" + "=" * 60)
print("DIAGNOSIS SUMMARY")
print("=" * 60)

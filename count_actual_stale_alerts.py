"""
Count alerts with truly stale data (price data before 9/16/2025)
"""

from datetime import datetime
from collections import defaultdict

import pandas as pd

from data_access.alert_repository import list_alerts as repo_list_alerts
from db_config import db_config

# American exchanges from the scheduler
AMERICAN_EXCHANGES = [
    'NASDAQ', 'NYSE', 'NYSE AMERICAN', 'NYSE ARCA', 'CBOE BZX',
    'NASDAQ GLOBAL MARKET', 'NASDAQ GLOBAL SELECT',
    'TORONTO', 'TSX', 'MEXICO', 'BUENOS AIRES', 'SANTIAGO'
]

# Filter for American daily alerts only
all_alerts = repo_list_alerts()

american_daily_alerts = []
for alert in all_alerts:
    if alert.get('timeframe') in ['1d', 'Daily', 'daily']:
        exchange = str(alert.get('exchange', '')).upper()
        # Check if it's an American exchange
        if any(ex in exchange for ex in AMERICAN_EXCHANGES):
            american_daily_alerts.append(alert)
        # Also check country-based
        elif alert.get('country', '').upper() in ['UNITED STATES', 'USA', 'CANADA', 'MEXICO', 'ARGENTINA', 'CHILE']:
            american_daily_alerts.append(alert)

print(f"Total American daily alerts: {len(american_daily_alerts)}")

# Connect to price database
# Connect to Postgres price database
conn = db_config.get_connection(role="prices")
cursor = conn.cursor()

# Last trading day was Tuesday 9/16/2025
last_trading_day = datetime(2025, 9, 16).date()

# Count stale alerts
stale_alerts = []
ticker_status = {}  # Track each ticker's status

for alert in american_daily_alerts:
    ticker = alert.get('ticker')
    if not ticker:
        continue

    # Skip if we already checked this ticker
    if ticker in ticker_status:
        if ticker_status[ticker] == 'stale':
            stale_alerts.append(alert)
        continue

    # Get most recent price date for this ticker
    cursor.execute("""
        SELECT MAX(date) FROM daily_prices
        WHERE ticker = %s
    """, (ticker,))

    result = cursor.fetchone()
    if result and result[0]:
        last_date = datetime.strptime(str(result[0]), '%Y-%m-%d').date()

        # Check if data is before last trading day (9/16/2025)
        if last_date < last_trading_day:
            ticker_status[ticker] = 'stale'
            stale_alerts.append(alert)
        else:
            ticker_status[ticker] = 'current'
    else:
        # No data at all
        ticker_status[ticker] = 'no_data'
        stale_alerts.append(alert)  # Count no data as stale

print(f"\nStale alerts (data before {last_trading_day}): {len(stale_alerts)}")

# Group stale alerts by exchange
exchange_groups = defaultdict(list)
for alert in stale_alerts:
    exchange = alert.get('exchange', 'Unknown')
    exchange_groups[exchange].append(alert)

print("\nStale alerts by exchange:")
for exchange, alerts in sorted(exchange_groups.items()):
    print(f"  {exchange}: {len(alerts)} alerts")

# Show unique stale tickers
stale_tickers = set()
for alert in stale_alerts:
    if alert.get('ticker'):
        stale_tickers.add(alert['ticker'])

print(f"\nUnique stale tickers: {len(stale_tickers)}")

# Show first 10 stale tickers with their last update date
print("\nSample stale tickers:")
for ticker in list(stale_tickers)[:10]:
    cursor.execute("""
        SELECT MAX(date) FROM daily_prices
        WHERE ticker = %s
    """, (ticker,))
    result = cursor.fetchone()
    if result and result[0]:
        print(f"  {ticker}: Last update {result[0]}")
    else:
        print(f"  {ticker}: No data")

db_config.close_connection(conn)

print("\n" + "="*50)
print(f"ANSWER: {len(stale_alerts)} alerts have stale data")
print("="*50)

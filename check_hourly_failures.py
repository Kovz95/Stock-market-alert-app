"""
Check which tickers have hourly data and which don't (PostgreSQL only)
"""

from data_access.metadata_repository import fetch_stock_metadata_map
from db_config import db_config

# Load main database from PostgreSQL
stock_db = fetch_stock_metadata_map() or {}

with db_config.connection(role="prices") as conn:
    cursor = conn.cursor()

    # Get tickers with hourly data
    cursor.execute("SELECT DISTINCT ticker FROM hourly_prices")
    tickers_with_hourly = {row[0] for row in cursor.fetchall()}

all_tickers = set(stock_db.keys())
tickers_without_hourly = all_tickers - tickers_with_hourly

print(f"Total tickers in database: {len(all_tickers)}")
print(f"Tickers with hourly data: {len(tickers_with_hourly)}")
print(f"Tickers without hourly data: {len(tickers_without_hourly)}")
print()

if tickers_without_hourly:
    print("Sample tickers WITHOUT hourly data (first 20):")
    for i, ticker in enumerate(sorted(tickers_without_hourly)[:20], 1):
        info = stock_db.get(ticker, {})
        exchange = info.get("exchange", "N/A")
        country = info.get("country", "N/A")
        asset_type = info.get("asset_type", "N/A")
        print(f"  {i}. {ticker:15} | {exchange:10} | {country:10} | {asset_type}")

    print("\nBreakdown by exchange (tickers WITHOUT hourly data):")
    exchanges = {}
    for ticker in tickers_without_hourly:
        exchange = stock_db.get(ticker, {}).get("exchange", "Unknown")
        exchanges[exchange] = exchanges.get(exchange, 0) + 1

    for exchange, count in sorted(exchanges.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {exchange:20}: {count:4} tickers")

    print("\nBreakdown by country (tickers WITHOUT hourly data):")
    countries = {}
    for ticker in tickers_without_hourly:
        country = stock_db.get(ticker, {}).get("country", "Unknown")
        countries[country] = countries.get(country, 0) + 1

    for country, count in sorted(countries.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {country:20}: {count:4} tickers")

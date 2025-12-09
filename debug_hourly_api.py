"""
Debug why hourly API isn't returning 2 years of data for some tickers
"""

from backend_fmp import FMPDataFetcher
from datetime import datetime, timedelta
import time

fetcher = FMPDataFetcher()

# Test with multiple tickers
test_tickers = [
    'AAPL',      # Should have 2 years but doesn't
    'SCHA.OL',   # Has 2 years in DB
    'NVDA',      # Should have 2 years
    'TSLA',      # Should have 2 years
]

to_date = datetime.now().strftime('%Y-%m-%d')
from_date = (datetime.now() - timedelta(days=756)).strftime('%Y-%m-%d')

print(f"Testing hourly data fetch from {from_date} to {to_date} (756 days)")
print("=" * 80)
print()

for ticker in test_tickers:
    print(f"Testing {ticker}...")
    print(f"  Requesting: {from_date} to {to_date}")

    df = fetcher.get_hourly_data(ticker, from_date=from_date, to_date=to_date)

    if df is not None and not df.empty:
        earliest = df.index.min()
        latest = df.index.max()
        days_span = (latest - earliest).days

        print(f"  Got: {len(df):,} bars")
        print(f"  Range: {earliest} to {latest}")
        print(f"  Days span: {days_span}")

        # Check if we got less than expected
        expected_bars = 756 * 6.5  # Rough estimate
        if len(df) < expected_bars * 0.5:
            print(f"  WARNING: Only got {len(df)}/{expected_bars:.0f} expected bars ({len(df)/expected_bars*100:.1f}%)")
    else:
        print(f"  FAILED: No data returned")

    print()
    time.sleep(0.5)

print("=" * 80)
print("\nNow testing with smaller chunks to see exact API behavior...")
print()

# Test AAPL with specific date ranges
ticker = 'AAPL'
print(f"Testing {ticker} with different date ranges:")
print()

# Test 1: Last 30 days
print("1. Last 30 days:")
from_30 = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
df = fetcher._fetch_hourly_chunk(ticker, from_30, to_date)
print(f"   Result: {len(df) if df is not None else 0} bars")

# Test 2: 30-60 days ago
print("2. 30-60 days ago:")
from_60 = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
to_30 = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
df = fetcher._fetch_hourly_chunk(ticker, from_60, to_30)
print(f"   Result: {len(df) if df is not None else 0} bars")

# Test 3: 90-120 days ago
print("3. 90-120 days ago:")
from_120 = (datetime.now() - timedelta(days=120)).strftime('%Y-%m-%d')
to_90 = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
df = fetcher._fetch_hourly_chunk(ticker, from_120, to_90)
print(f"   Result: {len(df) if df is not None else 0} bars")

# Test 4: 1 year ago (30 days)
print("4. 365-395 days ago (1 year back):")
from_395 = (datetime.now() - timedelta(days=395)).strftime('%Y-%m-%d')
to_365 = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
df = fetcher._fetch_hourly_chunk(ticker, from_395, to_365)
print(f"   Result: {len(df) if df is not None else 0} bars")

# Test 5: 2 years ago (30 days)
print("5. 730-760 days ago (2 years back):")
from_760 = (datetime.now() - timedelta(days=760)).strftime('%Y-%m-%d')
to_730 = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
df = fetcher._fetch_hourly_chunk(ticker, from_760, to_730)
print(f"   Result: {len(df) if df is not None else 0} bars")

print()
print("Analysis complete!")

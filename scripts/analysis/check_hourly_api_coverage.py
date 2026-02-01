#!/usr/bin/env python3
"""
Debug why hourly API isn't returning expected data coverage for some tickers.

Tests the FMP hourly data API with various tickers and date ranges to identify
coverage gaps and understand API behavior.

Usage:
    python scripts/analysis/check_hourly_api_coverage.py [options]

Examples:
    python scripts/analysis/check_hourly_api_coverage.py
    python scripts/analysis/check_hourly_api_coverage.py --tickers AAPL NVDA TSLA
    python scripts/analysis/check_hourly_api_coverage.py --days 365
"""

import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

import argparse
import time
from datetime import datetime, timedelta

from src.services.backend_fmp import FMPDataFetcher


def test_tickers_coverage(
    fetcher: FMPDataFetcher, tickers: list[str], days: int
) -> None:
    """Test hourly data coverage for multiple tickers."""
    to_date = datetime.now().strftime("%Y-%m-%d")
    from_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    print(f"Testing hourly data fetch from {from_date} to {to_date} ({days} days)")
    print("=" * 80)
    print()

    for ticker in tickers:
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
            expected_bars = days * 6.5  # Rough estimate (6.5 trading hours per day)
            if len(df) < expected_bars * 0.5:
                print(
                    f"  WARNING: Only got {len(df)}/{expected_bars:.0f} expected bars "
                    f"({len(df)/expected_bars*100:.1f}%)"
                )
        else:
            print("  FAILED: No data returned")

        print()
        time.sleep(0.5)


def test_date_range_chunks(fetcher: FMPDataFetcher, ticker: str) -> None:
    """Test specific date ranges to understand exact API behavior."""
    print("=" * 80)
    print(f"\nTesting {ticker} with different date ranges to analyze API behavior...")
    print()

    to_date = datetime.now().strftime("%Y-%m-%d")

    # Define test ranges: (description, days_ago_start, days_ago_end)
    test_ranges = [
        ("Last 30 days", 30, 0),
        ("30-60 days ago", 60, 30),
        ("90-120 days ago", 120, 90),
        ("365-395 days ago (1 year back)", 395, 365),
        ("730-760 days ago (2 years back)", 760, 730),
    ]

    for i, (description, days_start, days_end) in enumerate(test_ranges, 1):
        print(f"{i}. {description}:")
        from_dt = (datetime.now() - timedelta(days=days_start)).strftime("%Y-%m-%d")
        to_dt = (
            to_date
            if days_end == 0
            else (datetime.now() - timedelta(days=days_end)).strftime("%Y-%m-%d")
        )

        df = fetcher._fetch_hourly_chunk(ticker, from_dt, to_dt)
        bar_count = len(df) if df is not None else 0
        print(f"   Range: {from_dt} to {to_dt}")
        print(f"   Result: {bar_count} bars")
        print()
        time.sleep(0.3)


def main():
    """Main script logic."""
    parser = argparse.ArgumentParser(
        description="Debug hourly API data coverage for tickers"
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["AAPL", "SCHA.OL", "NVDA", "TSLA"],
        help="Ticker symbols to test (default: AAPL SCHA.OL NVDA TSLA)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=756,
        help="Number of days of history to request (default: 756, ~2 years)",
    )
    parser.add_argument(
        "--chunk-test",
        type=str,
        default="AAPL",
        help="Ticker to use for date range chunk testing (default: AAPL)",
    )
    parser.add_argument(
        "--skip-chunk-test",
        action="store_true",
        help="Skip the detailed date range chunk testing",
    )
    args = parser.parse_args()

    fetcher = FMPDataFetcher()

    # Test ticker coverage
    test_tickers_coverage(fetcher, args.tickers, args.days)

    # Test date range chunks
    if not args.skip_chunk_test:
        test_date_range_chunks(fetcher, args.chunk_test)

    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

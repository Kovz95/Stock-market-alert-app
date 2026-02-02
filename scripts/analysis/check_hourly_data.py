#!/usr/bin/env python3
"""
Check hourly data in the database - detailed stats (PostgreSQL only).

Usage:
    python scripts/analysis/check_hourly_data.py [--ticker TICKER]

Examples:
    python scripts/analysis/check_hourly_data.py
    python scripts/analysis/check_hourly_data.py --ticker AAPL
"""

import sys
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

import argparse

from src.data_access.metadata_repository import fetch_stock_metadata_map
from src.data_access.check_hourly_data_repository import CheckHourlyDataRepository


def print_overall_stats(repository: CheckHourlyDataRepository, stock_db: dict):
    """Print overall statistics."""
    stats = repository.get_overall_stats()
    
    print("=" * 80)
    print("HOURLY DATA DATABASE ANALYSIS")
    print("=" * 80)
    print()
    print("OVERALL STATISTICS:")
    print(f"  Total tickers in main database: {len(stock_db):,}")
    print(f"  Tickers with hourly data: {stats.tickers_with_data:,}")
    print(f"  Total hourly records: {stats.total_records:,}")
    print(f"  Earliest datetime: {stats.earliest}")
    print(f"  Latest datetime: {stats.latest}")
    
    if stats.days_span is not None:
        print(f"  Date range span: {stats.days_span} days")
    print()


def print_sample_tickers(repository: CheckHourlyDataRepository):
    """Print sample tickers with data."""
    print("SAMPLE TICKERS (First 10 with data):")
    samples = repository.get_sample_tickers(limit=10)
    
    for ticker_stats in samples:
        print(
            f"  {ticker_stats.ticker:10} | Records: {ticker_stats.num_records:5,} | "
            f"From: {ticker_stats.first_date} | To: {ticker_stats.last_date}"
        )
    print()


def print_ticker_details(repository: CheckHourlyDataRepository, ticker: str):
    """Print detailed information for a specific ticker."""
    print(f"DETAILED CHECK - {ticker}:")
    stats = repository.get_ticker_stats(ticker)
    
    if not stats:
        print(f"  NO DATA FOUND FOR {ticker}")
        print()
        return
    
    print(f"  Total records: {stats.num_records:,}")
    print(f"  First datetime: {stats.first_date}")
    print(f"  Last datetime: {stats.last_date}")
    print(f"  Days covered: {stats.days_covered}")
    print(f"  Avg bars per day: {stats.avg_bars_per_day:.1f}")
    print()
    
    # Show first 3 records
    print("  First 3 records:")
    first_bars = repository.get_ticker_price_bars(ticker, limit=3, order="ASC")
    for bar in first_bars:
        print(
            f"    {bar.datetime} | O:{bar.open:7.2f} H:{bar.high:7.2f} "
            f"L:{bar.low:7.2f} C:{bar.close:7.2f} V:{bar.volume:,}"
        )
    print()
    
    # Show last 3 records
    print("  Last 3 records:")
    last_bars = repository.get_ticker_price_bars(ticker, limit=3, order="DESC")
    for bar in last_bars:
        print(
            f"    {bar.datetime} | O:{bar.open:7.2f} H:{bar.high:7.2f} "
            f"L:{bar.low:7.2f} C:{bar.close:7.2f} V:{bar.volume:,}"
        )
    print()


def print_record_distribution(repository: CheckHourlyDataRepository):
    """Print record count distribution."""
    print("RECORD COUNT DISTRIBUTION:")
    distribution = repository.get_record_distribution()
    
    for dist in distribution:
        print(f"  {dist.range_name:20}: {dist.num_tickers:5,} tickers")
    print()


def main():
    """Main script logic."""
    parser = argparse.ArgumentParser(
        description="Analyze hourly price data in the database"
    )
    parser.add_argument(
        "--ticker",
        type=str,
        default="AAPL",
        help="Ticker symbol for detailed analysis (default: AAPL)",
    )
    args = parser.parse_args()
    
    # Load main database
    stock_db = fetch_stock_metadata_map() or {}
    
    # Create repository
    repository = CheckHourlyDataRepository()
    
    # Run analysis
    print_overall_stats(repository, stock_db)
    print_sample_tickers(repository)
    print_ticker_details(repository, args.ticker)
    print_record_distribution(repository)
    
    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

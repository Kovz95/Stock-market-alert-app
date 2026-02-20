"""
Debug Script: Scanner Performance Analysis

Run this to understand where time is spent during scanning.
"""

import time
import sys
from datetime import datetime

# Add project to path
sys.path.append('.')

from src.services.smart_price_fetcher import SmartPriceFetcher
from src.services.backend import evaluate_expression_list


def test_data_fetching():
    """Test price data fetching speed"""
    print("\n" + "="*60)
    print("TEST 1: Price Data Fetching Speed")
    print("="*60)

    fetcher = SmartPriceFetcher()
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

    for symbol in test_symbols:
        # Test 1: First fetch (cold cache)
        start = time.time()
        result = fetcher.get_price_data(symbol, timeframe='1d', force_refresh=False)
        elapsed = (time.time() - start) * 1000

        print(f"\n{symbol}:")
        print(f"  Source: {result.source.upper()}")
        print(f"  Freshness: {result.freshness}")
        print(f"  Time: {elapsed:.1f}ms")
        print(f"  Rows: {len(result.df) if result.df is not None else 0}")

        if elapsed > 1000:
            print(f"  ⚠️  SLOW! (>1000ms)")
        elif elapsed > 200:
            print(f"  ⚡ Database query")
        else:
            print(f"  🚀 Cache hit!")


def test_indicator_calculation():
    """Test indicator calculation speed"""
    print("\n" + "="*60)
    print("TEST 2: Indicator Calculation Speed")
    print("="*60)

    fetcher = SmartPriceFetcher()

    # Get sample data
    result = fetcher.get_price_data('AAPL', timeframe='1d')
    if result.df is None:
        print("❌ Failed to get data")
        return

    df = result.df
    print(f"\nDataFrame: {len(df)} rows")

    # Test different condition complexities
    test_conditions = [
        ("Simple: Close[-1] > 100", ["Close[-1] > 100"]),
        ("SMA: SMA(Close, 20)[-1] > SMA(Close, 50)[-1]", ["SMA(Close, 20)[-1] > SMA(Close, 50)[-1]"]),
        ("RSI: RSI(Close, 14)[-1] < 30", ["RSI(Close, 14)[-1] < 30"]),
        ("Multiple: 3 conditions", [
            "Close[-1] > SMA(Close, 20)[-1]",
            "RSI(Close, 14)[-1] > 50",
            "Volume[-1] > SMA(Volume, 20)[-1]"
        ]),
    ]

    for name, conditions in test_conditions:
        start = time.time()
        try:
            result = evaluate_expression_list(df, conditions, combination='1')
            elapsed = (time.time() - start) * 1000
            print(f"\n{name}:")
            print(f"  Time: {elapsed:.1f}ms")
            print(f"  Result: {result}")

            if elapsed > 100:
                print(f"  ⚠️  Slow indicator calculation")
        except Exception as e:
            print(f"\n{name}:")
            print(f"  ❌ Error: {e}")


def test_full_scan_workflow():
    """Test complete scan workflow for one symbol"""
    print("\n" + "="*60)
    print("TEST 3: Full Scan Workflow (Single Symbol)")
    print("="*60)

    fetcher = SmartPriceFetcher()
    symbol = 'AAPL'
    conditions = ["Close[-1] > SMA(Close, 20)[-1]"]

    print(f"\nScanning {symbol}...")

    # Step 1: Fetch data
    t1 = time.time()
    result = fetcher.get_price_data(symbol, timeframe='1d')
    fetch_time = (time.time() - t1) * 1000
    print(f"  1. Fetch data: {fetch_time:.1f}ms ({result.source})")

    if result.df is None:
        print("  ❌ No data")
        return

    # Step 2: Validate data
    t2 = time.time()
    df = result.df
    has_enough_data = len(df) >= 50
    validate_time = (time.time() - t2) * 1000
    print(f"  2. Validate data: {validate_time:.1f}ms ({len(df)} rows, sufficient: {has_enough_data})")

    # Step 3: Evaluate conditions
    t3 = time.time()
    match = evaluate_expression_list(df, conditions, combination='1')
    eval_time = (time.time() - t3) * 1000
    print(f"  3. Evaluate conditions: {eval_time:.1f}ms (match: {match})")

    # Total time
    total_time = fetch_time + validate_time + eval_time
    print(f"\n  Total: {total_time:.1f}ms")

    # Breakdown
    print(f"\n  Breakdown:")
    print(f"    Data fetch: {fetch_time/total_time*100:.1f}%")
    print(f"    Validation: {validate_time/total_time*100:.1f}%")
    print(f"    Evaluation: {eval_time/total_time*100:.1f}%")


def test_parallel_simulation():
    """Simulate parallel scanning of multiple symbols"""
    print("\n" + "="*60)
    print("TEST 4: Parallel Scan Simulation (20 symbols)")
    print("="*60)

    import concurrent.futures

    fetcher = SmartPriceFetcher()
    test_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
        'META', 'NVDA', 'AMD', 'INTC', 'NFLX',
        'DIS', 'BA', 'GE', 'F', 'GM',
        'JPM', 'BAC', 'WFC', 'GS', 'MS'
    ]

    def scan_one(symbol):
        start = time.time()
        result = fetcher.get_price_data(symbol, timeframe='1d')
        elapsed = (time.time() - start) * 1000
        return symbol, elapsed, result.source if result else 'error'

    # Simulate ThreadPoolExecutor with 20 workers
    print(f"\nScanning {len(test_symbols)} symbols with 20 workers...")
    overall_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(scan_one, symbol): symbol for symbol in test_symbols}
        results = []

        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    overall_elapsed = (time.time() - overall_start) * 1000

    print(f"\n  Total time: {overall_elapsed:.1f}ms ({overall_elapsed/1000:.2f}s)")
    print(f"  Per symbol avg: {overall_elapsed/len(test_symbols):.1f}ms")

    # Breakdown by source
    sources = {}
    for symbol, elapsed, source in results:
        if source not in sources:
            sources[source] = []
        sources[source].append(elapsed)

    print(f"\n  Breakdown by source:")
    for source, times in sources.items():
        avg_time = sum(times) / len(times)
        print(f"    {source.upper()}: {len(times)} symbols, avg {avg_time:.1f}ms")

    # Identify slow symbols
    slow_symbols = [(s, t, src) for s, t, src in results if t > 500]
    if slow_symbols:
        print(f"\n  ⚠️  Slow symbols (>500ms):")
        for symbol, elapsed, source in slow_symbols:
            print(f"    {symbol}: {elapsed:.1f}ms ({source})")


def main():
    print("\n" + "="*60)
    print("Scanner Performance Diagnostic Tool")
    print("="*60)
    print(f"Started at: {datetime.now()}")

    try:
        # Run all tests
        test_data_fetching()
        test_indicator_calculation()
        test_full_scan_workflow()
        test_parallel_simulation()

        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("\nIf you see:")
        print("  • Cache hits (🚀): Scanner should be fast (<5s for 20 symbols)")
        print("  • Database queries (⚡): Moderate speed (5-10s for 20 symbols)")
        print("  • API calls (⚠️): Slow first time (20-30s for 20 symbols)")
        print("  • Slow indicators (⚠️): Consider simpler conditions")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

    print(f"\nFinished at: {datetime.now()}")


if __name__ == "__main__":
    main()

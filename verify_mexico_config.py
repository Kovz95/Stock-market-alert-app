"""Verify Mexico configuration in hourly scheduler"""
import sys
sys.path.append('.')

from hourly_data_scheduler import get_exchange_market_hours

# Get market hours
market_hours = get_exchange_market_hours()

# Check Mexico configuration
if 'MEXICO' in market_hours:
    open_hour, close_hour, candle_type = market_hours['MEXICO']

    def to_time(h):
        hours = int(h)
        minutes = int((h - hours) * 60)
        return f'{hours:02d}:{minutes:02d}'

    print("Mexico Stock Exchange Configuration:")
    print(f"  Open:  {to_time(open_hour)} UTC")
    print(f"  Close: {to_time(close_hour)} UTC")
    print(f"  Candle Type: {'Half-hour (:30)' if candle_type == 'half' else 'On-the-hour (:00)'}")
    update_time = ':35' if candle_type == 'half' else ':05'
    print(f"  Update Time: {update_time}")

    # Verify it's correct
    if open_hour == 14.5 and close_hour == 21.0 and candle_type == 'half':
        print("\n✓ Mexico configuration is CORRECT:")
        print("  - Hours: 14:30-21:00 UTC (8:30 AM - 3:00 PM CST)")
        print("  - Candle Type: :30 (half-hour)")
        print("  - Updates at :35")
    else:
        print("\n✗ Mexico configuration is INCORRECT")
        print(f"  Expected: 14.5-21.0 hours, 'half' candles")
        print(f"  Got: {open_hour}-{close_hour} hours, '{candle_type}' candles")
else:
    print("✗ Mexico not found in market hours!")

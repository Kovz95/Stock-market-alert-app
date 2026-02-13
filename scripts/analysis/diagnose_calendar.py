#!/usr/bin/env python3
"""
Diagnose calendar/market hours issues in the scheduler.

Usage:
    python scripts/analysis/diagnose_calendar.py
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd

# Test exchange_calendars availability and date ranges
print("=" * 70)
print("CALENDAR DIAGNOSTICS")
print("=" * 70)

try:
    import exchange_calendars as xcals
    print(f"exchange_calendars version: {xcals.__version__}")
    XCALS_AVAILABLE = True
except ImportError as e:
    print(f"exchange_calendars NOT AVAILABLE: {e}")
    XCALS_AVAILABLE = False

print()

# Test some key exchanges
TEST_EXCHANGES = {
    "NYSE": "XNYS",
    "NASDAQ": "XNAS",
    "LONDON": "XLON",
    "TOKYO": "XTKS",
    "HONG KONG": "XHKG",
    "SINGAPORE": "XSES",
}

now_utc = pd.Timestamp.now(tz="UTC")
now_et = now_utc.tz_convert("America/New_York")

print(f"Current time (UTC): {now_utc}")
print(f"Current time (ET):  {now_et}")
print(f"Day of week: {now_utc.day_name()}")
print()

if XCALS_AVAILABLE:
    print("-" * 70)
    print("EXCHANGE CALENDAR DATE RANGES AND STATUS")
    print("-" * 70)

    for exchange_name, calendar_code in TEST_EXCHANGES.items():
        try:
            cal = xcals.get_calendar(calendar_code)
            print(f"\n{exchange_name} ({calendar_code}):")
            print(f"  Calendar range: {cal.first_session} to {cal.last_session}")

            # Check if current date is within range (convert to same type for comparison)
            current_date = now_utc.date()
            last_session_date = pd.Timestamp(cal.last_session).date() if hasattr(cal.last_session, 'date') else cal.last_session
            if isinstance(last_session_date, pd.Timestamp):
                last_session_date = last_session_date.date()

            if current_date > last_session_date:
                print(f"  WARNING: Current date {current_date} is BEYOND calendar end ({last_session_date})!")
            else:
                print(f"  Current date {current_date} is within calendar range")

            # Try to check if open
            try:
                is_open = cal.is_open_at_time(now_utc)
                print(f"  is_open_at_time({now_utc}): {is_open}")
            except Exception as e:
                print(f"  is_open_at_time FAILED: {type(e).__name__}: {e}")

            # Try to get session info for today
            try:
                if cal.is_session(current_date):
                    session_open = cal.session_open(current_date)
                    session_close = cal.session_close(current_date)
                    print(f"  Today's session: {session_open} to {session_close}")
                else:
                    print(f"  Today ({current_date}) is NOT a trading session")
            except Exception as e:
                print(f"  Session info FAILED: {type(e).__name__}: {e}")

        except Exception as e:
            print(f"\n{exchange_name} ({calendar_code}): FAILED TO LOAD - {type(e).__name__}: {e}")

print()
print("-" * 70)
print("TESTING calendar_adapter FUNCTIONS")
print("-" * 70)

# Import directly to avoid services/__init__.py which requires database
import importlib.util
calendar_adapter_path = BASE_DIR / "src" / "services" / "calendar_adapter.py"
spec = importlib.util.spec_from_file_location("calendar_adapter", calendar_adapter_path)
calendar_adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(calendar_adapter)

is_exchange_open = calendar_adapter.is_exchange_open
get_session_bounds = calendar_adapter.get_session_bounds
_get_session_bounds_fallback = calendar_adapter._get_session_bounds_fallback
EXCHANGE_CALENDARS_AVAILABLE = calendar_adapter.EXCHANGE_CALENDARS_AVAILABLE

print(f"\nEXCHANGE_CALENDARS_AVAILABLE: {EXCHANGE_CALENDARS_AVAILABLE}")
print()

for exchange_name in ["NYSE", "NASDAQ", "LONDON", "TOKYO"]:
    print(f"\n{exchange_name}:")
    try:
        is_open = is_exchange_open(exchange_name, now_utc)
        print(f"  is_exchange_open: {is_open}")
    except Exception as e:
        print(f"  is_exchange_open FAILED: {e}")

    try:
        session_open, session_close = get_session_bounds(exchange_name, now_utc, next_if_closed=False)
        print(f"  Session bounds (current/recent): {session_open} to {session_close}")

        # Check if current time is between bounds
        in_session = session_open <= now_utc <= session_close
        print(f"  Current time between bounds: {in_session}")
    except Exception as e:
        print(f"  get_session_bounds FAILED: {e}")

    # Test fallback directly
    try:
        fb_open, fb_close = _get_session_bounds_fallback(exchange_name, now_utc, next_if_closed=False)
        print(f"  Fallback bounds: {fb_open} to {fb_close}")
    except Exception as e:
        print(f"  Fallback FAILED: {e}")

print()
print("-" * 70)
print("TESTING any_market_open()")
print("-" * 70)

from src.services.hourly_data_scheduler import any_market_open

try:
    result = any_market_open()
    print(f"any_market_open(): {result}")
except Exception as e:
    print(f"any_market_open() FAILED: {e}")

print()
print("=" * 70)
print("DIAGNOSTICS COMPLETE")
print("=" * 70)

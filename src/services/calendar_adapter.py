#!/usr/bin/env python3
"""
Calendar Adapter for Exchange Trading Hours

Provides calendar-aware functions for market hours using exchange_calendars library.
Handles DST transitions, holidays, and different hourly alignments across global markets.

Uses EXCHANGE_SCHEDULES from exchange_schedule_config_v2 for exchange metadata.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional, Tuple

import pandas as pd
import pytz

# Import exchange_calendars if available
try:
    import exchange_calendars as xcals
    EXCHANGE_CALENDARS_AVAILABLE = True
except ImportError:
    EXCHANGE_CALENDARS_AVAILABLE = False
    xcals = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import exchange schedules and configuration
from src.config.exchange_schedule_config import (
    EXCHANGE_SCHEDULES,
    EXCHANGE_TIMEZONES,
    get_exchange_close_time,
    is_dst_active,
)

# Mapping of exchange names to exchange_calendars calendar codes
EXCHANGE_TO_CALENDAR = {
    "NYSE": "XNYS",
    "NASDAQ": "XNAS",
    "NYSE AMERICAN": "XASE",
    "NYSE ARCA": "ARCX",
    "CBOE BZX": "BATS",
    "TORONTO": "XTSE",
    "LONDON": "XLON",
    "XETRA": "XFRA",
    "EURONEXT PARIS": "XPAR",
    "EURONEXT AMSTERDAM": "XAMS",
    "EURONEXT BRUSSELS": "XBRU",
    "EURONEXT LISBON": "XLIS",
    "EURONEXT DUBLIN": "XDUB",
    "MILAN": "XMIL",
    "SPAIN": "XMAD",
    "SIX SWISS": "XSWX",
    "SIX": "XSWX",
    "OMX NORDIC STOCKHOLM": "XSTO",
    "OMX NORDIC COPENHAGEN": "XCSE",
    "OMX NORDIC HELSINKI": "XHEL",
    "OMX NORDIC ICELAND": "XICE",
    "OSLO": "XOSL",
    "WARSAW": "XWAR",
    "VIENNA": "XWBO",
    "ATHENS": "ASEX",
    "PRAGUE": "XPRA",
    "BUDAPEST": "XBUD",
    "ISTANBUL": "XIST",
    "TOKYO": "XTKS",
    "HONG KONG": "XHKG",
    "SINGAPORE": "XSES",
    "ASX": "XASX",
    "TAIWAN": "XTAI",
    "NSE INDIA": "XNSE",
    "BSE INDIA": "XBOM",
    "INDONESIA": "XIDX",
    "THAILAND": "XBKK",
    "MALAYSIA": "XKLS",
    "SAO PAULO": "BVMF",
    "MEXICO": "XMEX",
    "BUENOS AIRES": "XBUE",
    "SANTIAGO": "XSGO",
    "JSE": "XJSE",
}

# Hourly alignment for exchanges (when candles close)
# "hour" = :00 candles (updates at :05)
# "quarter" = :15 candles (updates at :20)
# "half" = :30 candles (updates at :35)
HOURLY_ALIGNMENT = {
    # :00 candles - Most exchanges
    "hour": [
        "NYSE", "NASDAQ", "NYSE AMERICAN", "NYSE ARCA", "CBOE BZX",
        "TORONTO", "MEXICO", "SANTIAGO", "BUENOS AIRES",
        "LONDON", "XETRA", "EURONEXT AMSTERDAM", "EURONEXT BRUSSELS",
        "EURONEXT LISBON", "EURONEXT DUBLIN", "MILAN", "SPAIN",
        "SIX SWISS", "SIX", "OMX NORDIC STOCKHOLM", "OMX NORDIC COPENHAGEN",
        "OMX NORDIC HELSINKI", "OSLO", "WARSAW", "VIENNA", "PRAGUE",
        "BUDAPEST", "ISTANBUL", "TOKYO", "TAIWAN", "ASX",
        "SINGAPORE", "MALAYSIA", "THAILAND", "INDONESIA",
        "SAO PAULO", "JSE",
    ],
    # :15 candles - India exchanges
    "quarter": [
        "BSE INDIA", "NSE INDIA",
    ],
    # :30 candles - Some exchanges
    "half": [
        "HONG KONG", "EURONEXT PARIS", "ATHENS", "OMX NORDIC ICELAND",
    ],
}

# Create reverse mapping for quick lookup
_alignment_lookup = {}
for style, exchanges in HOURLY_ALIGNMENT.items():
    for exchange in exchanges:
        _alignment_lookup[exchange] = style


def get_calendar_timezone(exchange: str) -> Optional[str]:
    """
    Get the timezone for an exchange.

    Args:
        exchange: Exchange code (e.g., "NYSE", "LONDON")

    Returns:
        Timezone string (e.g., "America/New_York") or None
    """
    if exchange in EXCHANGE_SCHEDULES:
        config = EXCHANGE_SCHEDULES[exchange]
        # Check if timezone is explicitly set in config
        if "timezone" in config:
            return config["timezone"]

    # Fall back to EXCHANGE_TIMEZONES mapping
    return EXCHANGE_TIMEZONES.get(exchange)


def get_hourly_alignment(exchange: str) -> str:
    """
    Get the hourly candle alignment for an exchange.

    Args:
        exchange: Exchange code (e.g., "NYSE", "BSE INDIA")

    Returns:
        Alignment style: "hour", "quarter", or "half"
        - "hour" = :00 candles (updates at :05)
        - "quarter" = :15 candles (updates at :20)
        - "half" = :30 candles (updates at :35)
    """
    return _alignment_lookup.get(exchange, "hour")


def _get_exchange_calendar(exchange: str):
    """
    Get the exchange_calendars calendar object for an exchange.

    Args:
        exchange: Exchange code

    Returns:
        ExchangeCalendar object or None if not available
    """
    if not EXCHANGE_CALENDARS_AVAILABLE:
        return None

    calendar_code = EXCHANGE_TO_CALENDAR.get(exchange)
    if not calendar_code:
        return None

    try:
        return xcals.get_calendar(calendar_code)
    except Exception as e:
        logger.debug(f"Could not get calendar for {exchange} ({calendar_code}): {e}")
        return None


def get_session_bounds(
    exchange: str,
    timestamp: pd.Timestamp,
    next_if_closed: bool = False
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Get the session open and close times for an exchange at a given timestamp.

    Args:
        exchange: Exchange code (e.g., "NYSE", "LONDON")
        timestamp: Reference timestamp (should be timezone-aware)
        next_if_closed: If True, return next session if market is currently closed

    Returns:
        Tuple of (session_open, session_close) as timezone-aware pandas Timestamps
    """
    # Ensure timestamp is timezone-aware UTC
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    # Try using exchange_calendars if available
    cal = _get_exchange_calendar(exchange)
    if cal is not None:
        try:
            # Get the session for this timestamp
            date = timestamp.date()

            if next_if_closed:
                # If requesting next session and we're after close, move to next valid session
                try:
                    session_open = cal.session_open(date)
                    session_close = cal.session_close(date)

                    # If timestamp is after today's close, get next session
                    if timestamp > session_close:
                        next_date = cal.next_session(date)
                        session_open = cal.session_open(next_date)
                        session_close = cal.session_close(next_date)
                except Exception:
                    # If date is not a valid trading day, get next session. If calendar
                    # rejects the date (e.g. beyond calendar end like XSES 2025-12-31),
                    # do not call calendar again—re-raise so outer handler falls back.
                    try:
                        next_date = cal.next_session(date)
                        session_open = cal.session_open(next_date)
                        session_close = cal.session_close(next_date)
                    except Exception:
                        raise
            else:
                # Get current or most recent session
                if cal.is_session(date):
                    session_open = cal.session_open(date)
                    session_close = cal.session_close(date)
                else:
                    # Get previous session
                    prev_date = cal.previous_session(date)
                    session_open = cal.session_open(prev_date)
                    session_close = cal.session_close(prev_date)

            return session_open, session_close

        except Exception as e:
            # Calendar can reject dates beyond its valid range (e.g. XSES only through 2025-12-31)
            if "cannot be later than" in str(e) or "last trading minute" in str(e).lower():
                logger.info(
                    "exchange_calendars date range limit for %s: %s; using schedule fallback",
                    exchange,
                    str(e).split("\n")[0][:80],
                )
            else:
                logger.debug(
                    "exchange_calendars failed for %s: %s, falling back to manual calculation",
                    exchange,
                    e,
                )

    # Fallback: Use EXCHANGE_SCHEDULES configuration
    return _get_session_bounds_fallback(exchange, timestamp, next_if_closed)


def _get_session_bounds_fallback(
    exchange: str,
    timestamp: pd.Timestamp,
    next_if_closed: bool
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Fallback method to calculate session bounds using EXCHANGE_SCHEDULES.

    Args:
        exchange: Exchange code
        timestamp: Reference timestamp (UTC)
        next_if_closed: If True, return next session if currently closed

    Returns:
        Tuple of (session_open, session_close)
    """
    if exchange not in EXCHANGE_SCHEDULES:
        raise ValueError(f"Unknown exchange: {exchange}")

    config = EXCHANGE_SCHEDULES[exchange]

    # Get timezone
    tz_name = get_calendar_timezone(exchange)
    if not tz_name:
        # Default to US Eastern for US exchanges
        if exchange in ["NYSE", "NASDAQ", "NYSE AMERICAN", "NYSE ARCA", "CBOE BZX"]:
            tz_name = "America/New_York"
        else:
            tz_name = "UTC"

    tz = pytz.timezone(tz_name)

    # Convert timestamp to local timezone
    local_time = timestamp.tz_convert(tz)
    current_date = local_time.date()

    # Standard market hours (these are approximate - exchange_calendars would be more accurate)
    # Most markets: 9:30 AM - 4:00 PM local time
    open_hour, open_minute = 9, 30
    close_hour, close_minute = 16, 0

    # Adjust for specific exchanges
    if exchange in ["LONDON", "XETRA"]:
        open_hour, open_minute = 8, 0
        close_hour, close_minute = 16, 30
    elif exchange in ["TOKYO"]:
        open_hour, open_minute = 9, 0
        close_hour, close_minute = 15, 0
    elif exchange in ["HONG KONG"]:
        open_hour, open_minute = 9, 30
        close_hour, close_minute = 16, 0

    # Create open/close timestamps for current date
    session_open = tz.localize(datetime.combine(current_date, datetime.min.time().replace(hour=open_hour, minute=open_minute)))
    session_close = tz.localize(datetime.combine(current_date, datetime.min.time().replace(hour=close_hour, minute=close_minute)))

    # Convert to pandas Timestamps
    session_open = pd.Timestamp(session_open)
    session_close = pd.Timestamp(session_close)

    # If we're past close and next_if_closed is True, move to next day
    if next_if_closed and timestamp > session_close:
        # Move to next business day (simple approximation)
        next_date = current_date + timedelta(days=1)
        # Skip weekends
        while next_date.weekday() >= 5:  # Saturday=5, Sunday=6
            next_date += timedelta(days=1)

        session_open = tz.localize(datetime.combine(next_date, datetime.min.time().replace(hour=open_hour, minute=open_minute)))
        session_close = tz.localize(datetime.combine(next_date, datetime.min.time().replace(hour=close_hour, minute=close_minute)))
        session_open = pd.Timestamp(session_open)
        session_close = pd.Timestamp(session_close)

    return session_open, session_close


def is_exchange_open(exchange: str, timestamp: pd.Timestamp) -> bool:
    """
    Check if an exchange is open at a given timestamp.

    Args:
        exchange: Exchange code (e.g., "NYSE", "LONDON")
        timestamp: Timestamp to check (should be timezone-aware)

    Returns:
        True if exchange is open, False otherwise
    """
    # Ensure timestamp is timezone-aware
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    # Try using exchange_calendars
    cal = _get_exchange_calendar(exchange)
    if cal is not None:
        try:
            return cal.is_open_at_time(timestamp)
        except Exception as e:
            logger.debug(f"exchange_calendars failed for {exchange}: {e}, falling back")

    # Fallback: Check against session bounds
    try:
        session_open, session_close = get_session_bounds(exchange, timestamp, next_if_closed=False)
        return session_open <= timestamp <= session_close
    except Exception:
        return False


def get_next_daily_run_time(exchange: str, reference_time: pd.Timestamp) -> datetime:
    """
    Get the next daily alert check run time for an exchange.

    This is typically 40 minutes after market close to allow for data availability.

    Args:
        exchange: Exchange code
        reference_time: Reference timestamp (timezone-aware)

    Returns:
        Next run time as datetime
    """
    if exchange not in EXCHANGE_SCHEDULES:
        raise ValueError(f"Unknown exchange: {exchange}")

    # Get the next session close
    _, session_close = get_session_bounds(exchange, reference_time, next_if_closed=True)

    # Add 40 minutes for data availability
    run_time = session_close + pd.Timedelta(minutes=40)

    return run_time.to_pydatetime()


def get_market_open_close_times(
    exchange: str,
    date: Optional[datetime.date] = None
) -> Tuple[datetime, datetime]:
    """
    Get market open and close times for a specific date.

    Args:
        exchange: Exchange code
        date: Date to check (defaults to today)

    Returns:
        Tuple of (open_time, close_time) as datetime objects
    """
    if date is None:
        date = datetime.now().date()

    # Create a timestamp for the date
    timestamp = pd.Timestamp(date).tz_localize("UTC")

    # Get session bounds
    session_open, session_close = get_session_bounds(exchange, timestamp, next_if_closed=False)

    return session_open.to_pydatetime(), session_close.to_pydatetime()


# Export the EXCHANGE_SCHEDULES for convenience
__all__ = [
    "EXCHANGE_SCHEDULES",
    "get_calendar_timezone",
    "get_hourly_alignment",
    "get_session_bounds",
    "is_exchange_open",
    "get_next_daily_run_time",
    "get_market_open_close_times",
]

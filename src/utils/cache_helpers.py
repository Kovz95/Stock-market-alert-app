"""
Cache helper utilities for smart price fetching.

This module provides helper functions for cache TTL management and
data freshness formatting.
"""

from datetime import datetime
from typing import Literal


def get_cache_ttl(timeframe: str) -> int:
    """
    Get appropriate TTL (time-to-live) in seconds based on timeframe.

    Different timeframes require different cache durations:
    - Hourly data: 5 minutes (data changes frequently)
    - Daily data: 60 minutes (stable during trading day)
    - Weekly data: 2 hours (very stable)

    Args:
        timeframe: Timeframe string ('1h', '1d', '1wk', etc.)

    Returns:
        TTL in seconds.

    Example:
        >>> get_cache_ttl('1h')
        300
        >>> get_cache_ttl('1d')
        3600
        >>> get_cache_ttl('1wk')
        7200
    """
    # Normalize timeframe
    normalized = str(timeframe).lower().strip()

    # Map timeframes to TTL
    ttl_map = {
        '1h': 300,      # 5 minutes (hourly data changes fast)
        'hourly': 300,
        '1hr': 300,
        'hour': 300,
        '1d': 3600,     # 60 minutes (daily data stable)
        'daily': 3600,
        'day': 3600,
        '1wk': 7200,    # 2 hours (weekly data very stable)
        'weekly': 7200,
        'week': 7200,
        '1w': 7200,
    }

    return ttl_map.get(normalized, 1800)  # Default: 30 minutes


def format_age(last_update: datetime) -> str:
    """
    Format timestamp as human-readable age (e.g., '5 min ago').

    Args:
        last_update: Timestamp of last update.

    Returns:
        Human-readable age string.

    Example:
        >>> format_age(datetime.now() - timedelta(minutes=5))
        '5 min ago'
        >>> format_age(datetime.now() - timedelta(hours=2))
        '2 hours ago'
    """
    if not last_update:
        return "Unknown"

    try:
        age = datetime.now() - last_update

        # Format based on age
        seconds = age.total_seconds()

        if seconds < 0:
            return "Future"

        if seconds < 60:
            return f"{int(seconds)} sec ago"

        minutes = seconds / 60
        if minutes < 60:
            return f"{int(minutes)} min ago"

        hours = minutes / 60
        if hours < 24:
            return f"{int(hours)} hour{'s' if hours >= 2 else ''} ago"

        days = hours / 24
        if days < 7:
            return f"{int(days)} day{'s' if days >= 2 else ''} ago"

        weeks = days / 7
        return f"{int(weeks)} week{'s' if weeks >= 2 else ''} ago"

    except Exception:
        return "Unknown"


def get_freshness_icon(freshness: Literal['fresh', 'recent', 'stale', 'error']) -> str:
    """
    Return icon/badge for freshness status.

    Args:
        freshness: Freshness status.

    Returns:
        Icon string with status label.

    Example:
        >>> get_freshness_icon('fresh')
        '🟢 Fresh'
        >>> get_freshness_icon('stale')
        '🔴 Stale'
    """
    icon_map = {
        'fresh': '🟢 Fresh',
        'recent': '🟡 Recent',
        'stale': '🔴 Stale',
        'error': '⚠️ Error'
    }

    return icon_map.get(freshness, '❓ Unknown')


def build_cache_key(ticker: str, timeframe: str, lookback_days: int) -> str:
    """
    Build a cache key for price data.

    Args:
        ticker: Stock ticker symbol.
        timeframe: Timeframe string ('1h', '1d', '1wk').
        lookback_days: Number of days of historical data.

    Returns:
        Cache key string.

    Example:
        >>> build_cache_key('AAPL', '1d', 500)
        'scanner:price:AAPL:1d:500'
    """
    # Normalize inputs
    ticker = ticker.upper().strip()
    timeframe = timeframe.lower().strip()

    return f"scanner:price:{ticker}:{timeframe}:{lookback_days}"

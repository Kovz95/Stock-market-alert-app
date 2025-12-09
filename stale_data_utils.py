"""
Shared utilities for determining stale data
Ensures consistency between price updates and alert checking
"""

from datetime import datetime, timedelta
import pandas as pd

def is_data_stale(last_data_date, timeframe='1d', reference_time=None):
    """
    Determine if data is stale based on the last data date and timeframe

    Args:
        last_data_date: The date of the last available data (datetime, date, or Timestamp)
        timeframe: The timeframe being checked ('1d' for daily, '1w' for weekly)
        reference_time: The time to compare against (default: now)

    Returns:
        bool: True if data is stale, False otherwise
    """
    if reference_time is None:
        reference_time = datetime.now()

    normalized_timeframe = str(timeframe).lower() if timeframe else '1d'

    if normalized_timeframe in {'1h', 'hourly', '1hr', 'hour'}:
        # Work with full datetime granularity
        if isinstance(last_data_date, pd.Timestamp):
            last_timestamp = last_data_date.to_pydatetime()
        elif isinstance(last_data_date, datetime):
            last_timestamp = last_data_date
        else:
            # Attempt to parse generic objects (e.g., string)
            try:
                last_timestamp = pd.to_datetime(last_data_date).to_pydatetime()
            except Exception:
                last_timestamp = datetime.min

        if not isinstance(reference_time, datetime):
            reference_time = pd.to_datetime(reference_time).to_pydatetime()

        diff_hours = (reference_time - last_timestamp).total_seconds() / 3600.0
        diff_hours = max(diff_hours, 0)

        weekday = reference_time.weekday()
        if weekday <= 4:
            # During trading week expect data within the last 3 hours
            return diff_hours > 3
        else:
            # On weekends allow more slack (72 hours ~= 3 days)
            return diff_hours > 72

    # Convert to comparable formats
    if hasattr(last_data_date, 'date'):
        last_date = last_data_date.date()
    else:
        last_date = last_data_date

    if hasattr(reference_time, 'date'):
        today = reference_time.date()
    else:
        today = reference_time

    # Determine the last expected trading day based on timeframe
    if normalized_timeframe in {'1w', '1wk', 'weekly', 'week'}:
        # For weekly data: should be from the most recent Friday that has passed
        # Mon-Thu: Expect last week's Friday
        # Fri: Expect this Friday (today)
        # Sat-Sun: Expect this past Friday

        weekday = reference_time.weekday()

        if weekday < 4:  # Monday through Thursday
            # Calculate last week's Friday (previous week)
            days_since_last_friday = weekday + 3  # Mon=3, Tue=4, Wed=5, Thu=6 days since last Friday
            last_friday = reference_time - timedelta(days=days_since_last_friday)
            expected_date = last_friday.date() if hasattr(last_friday, 'date') else last_friday
        elif weekday == 4:  # Friday
            # Expect today's data (this Friday)
            expected_date = today
        else:  # Saturday or Sunday
            # Use this past Friday
            days_since_friday = weekday - 4  # Sat=1, Sun=2 days since Friday
            last_friday = reference_time - timedelta(days=days_since_friday)
            expected_date = last_friday.date() if hasattr(last_friday, 'date') else last_friday

        # Weekly data is stale if it's not from the expected Friday
        return last_date != expected_date
    else:
        # For daily data: stale if missing today's data
        # Fresh means we have today's data (for trading days Mon-Fri)
        weekday = reference_time.weekday() if hasattr(reference_time, 'weekday') else today.weekday()

        # If data is from today, it's fresh
        if last_date >= today:
            return False

        # On trading days (Mon-Fri), we need today's data
        # On weekends, we need Friday's data
        if weekday <= 4:  # Monday through Friday
            # Need today's data to be fresh
            return True  # If we got here, last_date < today, so it's stale
        else:  # Saturday or Sunday
            # Need Friday's data
            days_since_friday = weekday - 4  # Sat=2, Sun=3
            last_friday = reference_time - timedelta(days=days_since_friday)
            last_expected_date = last_friday.date() if hasattr(last_friday, 'date') else last_friday
            return last_date < last_expected_date

def get_last_trading_day(reference_time=None):
    """
    Get the last expected trading day

    Args:
        reference_time: The time to calculate from (default: now)

    Returns:
        date: The last trading day
    """
    if reference_time is None:
        reference_time = datetime.now()

    weekday = reference_time.weekday() if hasattr(reference_time, 'weekday') else reference_time.weekday()

    if weekday == 0:  # Monday
        last_trading_day = reference_time - timedelta(days=3)  # Friday
    elif weekday == 6:  # Sunday
        last_trading_day = reference_time - timedelta(days=2)  # Friday
    else:
        last_trading_day = reference_time - timedelta(days=1)  # Previous day

    return last_trading_day.date() if hasattr(last_trading_day, 'date') else last_trading_day

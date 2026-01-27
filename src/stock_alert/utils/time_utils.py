"""Time and timezone conversion utilities."""

import datetime

import pytz


def get_dst_adjusted_time(
    local_time_str: str, timezone_str: str, target_timezone: str = "America/New_York"
) -> tuple[int, int]:
    """
    Convert local market time to target timezone with DST awareness.

    Args:
        local_time_str: Time string in local market time (e.g., "4:00 PM")
        timezone_str: Market timezone (e.g., "Asia/Hong_Kong")
        target_timezone: Target timezone for conversion (default: EST/EDT)

    Returns:
        tuple: (hour, minute) in target timezone
    """
    try:
        # Parse the local time
        if "PM" in local_time_str.upper():
            time_part = local_time_str.replace(" PM", "").replace(" pm", "")
            hour, minute = map(int, time_part.split(":"))
            if hour != 12:
                hour += 12
        elif "AM" in local_time_str.upper():
            time_part = local_time_str.replace(" AM", "").replace(" am", "")
            hour, minute = map(int, time_part.split(":"))
            if hour == 12:
                hour = 0
        else:
            # Handle 24-hour format
            hour, minute = map(int, local_time_str.split(":"))

        # Create a datetime object in the market's timezone
        market_tz = pytz.timezone(timezone_str)
        target_tz = pytz.timezone(target_timezone)

        # Use today's date for the conversion (DST will be automatically handled)
        today = datetime.datetime.now().date()
        local_dt = datetime.datetime.combine(today, datetime.time(hour, minute))

        # Localize to market timezone
        local_dt = market_tz.localize(local_dt)

        # Convert to target timezone
        target_dt = local_dt.astimezone(target_tz)

        return target_dt.hour, target_dt.minute

    except Exception as e:
        print(f"Error converting time {local_time_str} from {timezone_str}: {e}")
        # Fallback to manual conversion
        return convert_time_manual(local_time_str, timezone_str, target_timezone)


def convert_time_manual(
    local_time_str: str, timezone_str: str, target_timezone: str = "America/New_York"
) -> tuple[int, int]:
    """
    Manual time conversion with DST handling for common markets.

    Args:
        local_time_str: Time string in local market time
        timezone_str: Market timezone
        target_timezone: Target timezone for conversion

    Returns:
        tuple: (hour, minute) in target timezone
    """
    # Parse local time
    if "PM" in local_time_str.upper():
        time_part = local_time_str.replace(" PM", "").replace(" pm", "")
        hour, minute = map(int, time_part.split(":"))
        if hour != 12:
            hour += 12
    elif "AM" in local_time_str.upper():
        time_part = local_time_str.replace(" AM", "").replace(" am", "")
        hour, minute = map(int, time_part.split(":"))
        if hour == 12:
            hour = 0
    else:
        hour, minute = map(int, local_time_str.split(":"))

    # DST-aware conversion table
    # Format: (market_tz, local_hour, local_minute, est_hour_dst, est_minute_dst, est_hour_standard, est_minute_standard)
    conversion_table = {
        "Asia/Hong_Kong": (hour, minute, hour - 13, minute, hour - 12, minute),  # HK: UTC+8
        "Asia/Singapore": (hour, minute, hour - 13, minute, hour - 12, minute),  # SG: UTC+8
        "Asia/Taipei": (hour, minute, hour - 13, minute, hour - 12, minute),  # TW: UTC+8
        "Asia/Kuala_Lumpur": (hour, minute, hour - 13, minute, hour - 12, minute),  # MY: UTC+8
        "Asia/Tokyo": (hour, minute, hour - 14, minute, hour - 13, minute),  # JP: UTC+9
        "Europe/London": (hour, minute, hour - 5, minute, hour - 6, minute),  # UK: UTC+0/+1
        "Europe/Paris": (hour, minute, hour - 6, minute, hour - 7, minute),  # FR: UTC+1/+2
        "Europe/Berlin": (hour, minute, hour - 6, minute, hour - 7, minute),  # DE: UTC+1/+2
        "America/Toronto": (hour, minute, hour + 0, minute, hour + 0, minute),  # CA: Same as US
    }

    # Check if current time is in DST
    now = datetime.datetime.now(pytz.timezone(target_timezone))
    is_dst = now.dst() != datetime.timedelta(0)

    if timezone_str in conversion_table:
        _, _, _, dst_hour, dst_minute, std_hour, std_minute = conversion_table[timezone_str]

        if is_dst:
            return dst_hour, dst_minute
        else:
            return std_hour, std_minute

    # Default fallback
    return hour - 5, minute  # Assume UTC-5 for unknown timezones


def get_market_timezone(exchange_name: str) -> str:
    """
    Get the timezone for a given exchange.

    Args:
        exchange_name: Exchange name

    Returns:
        Timezone string
    """
    timezone_map = {
        "Hong Kong": "Asia/Hong_Kong",
        "Singapore": "Asia/Singapore",
        "Taiwan": "Asia/Taipei",
        "Malaysia": "Asia/Kuala_Lumpur",
        "Tokyo": "Asia/Tokyo",
        "London": "Europe/London",
        "Euronext Paris": "Europe/Paris",
        "Xetra": "Europe/Berlin",
        "Toronto": "America/Toronto",
        "Nasdaq": "America/New_York",
        "NYSE": "America/New_York",
        "NYSE American": "America/New_York",
    }

    return timezone_map.get(exchange_name, "UTC")


def is_dst_active() -> bool:
    """
    Check if Daylight Saving Time is currently active in EST/EDT.

    Returns:
        True if DST is active, False otherwise
    """
    now = datetime.datetime.now(pytz.timezone("America/New_York"))
    return now.dst() != datetime.timedelta(0)


def get_dst_status() -> dict:
    """
    Get current DST status and next transition dates.

    Returns:
        Dict with DST status information
    """
    ny_tz = pytz.timezone("America/New_York")
    now = datetime.datetime.now(ny_tz)

    # Get next DST transitions
    transitions = ny_tz._utc_transition_times
    current_year = now.year

    # Find next spring forward (DST starts)
    spring_forward = None
    fall_back = None

    for transition in transitions:
        if transition.year >= current_year:
            transition_dt = datetime.datetime.fromtimestamp(transition.timestamp(), ny_tz)
            if transition_dt > now:
                if spring_forward is None:
                    spring_forward = transition_dt
                elif fall_back is None:
                    fall_back = transition_dt
                    break

    return {
        "is_dst": now.dst() != datetime.timedelta(0),
        "current_offset": now.utcoffset(),
        "spring_forward": spring_forward,
        "fall_back": fall_back,
        "current_time": now,
    }

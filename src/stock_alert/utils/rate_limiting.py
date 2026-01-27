"""Rate limiting and cache management utilities."""

import time


def calculate_cache_hit_rate(cache_size: int = 0) -> float:
    """
    Calculate cache hit rate (simplified - you'd need to track hits/misses).

    Args:
        cache_size: Current cache size

    Returns:
        Estimated cache hit rate percentage
    """
    # This is a simplified calculation - in practice you'd track actual hits/misses
    if cache_size > 0:
        # Estimate based on cache size and age
        return min(85, cache_size / 100)  # Rough estimate
    return 0


def estimate_daily_requests(requests_last_hour: int = 100) -> int:
    """
    Estimate daily request volume based on current patterns.

    Args:
        requests_last_hour: Number of requests in the last hour

    Returns:
        Estimated daily request count
    """
    # Estimate daily requests (assuming consistent pattern)
    estimated_daily = requests_last_hour * 24

    return estimated_daily


def get_rate_limit_recommendations(minute_utilization: float, hour_utilization: float) -> list[str]:
    """
    Get recommendations based on current utilization.

    Args:
        minute_utilization: Utilization percentage for minute limit
        hour_utilization: Utilization percentage for hour limit

    Returns:
        List of recommendation strings
    """
    recommendations = []

    if minute_utilization > 80:
        recommendations.append(
            "[CRITICAL] Minute limit nearly reached - consider pausing processing"
        )
    elif minute_utilization > 60:
        recommendations.append("[WARNING] High minute utilization - monitor closely")

    if hour_utilization > 80:
        recommendations.append(
            "[CRITICAL] Hour limit nearly reached - implement aggressive caching"
        )
    elif hour_utilization > 60:
        recommendations.append(
            "[WARNING] High hour utilization - consider increasing cache duration"
        )

    if minute_utilization < 30 and hour_utilization < 30:
        recommendations.append("[GOOD] Rate limits well within safe range")

        recommendations.append("[SUGGESTION] Low cache usage - consider reducing cache duration")
        recommendations.append("[SUGGESTION] High cache usage - consider increasing cache cleanup")

    return recommendations


def emergency_rate_limit_pause():
    """Emergency function to pause processing when rate limits are exceeded."""
    print("[EMERGENCY] Rate limits exceeded - pausing processing for 5 minutes")
    time.sleep(300)  # Wait 5 minutes
    print("[RESUMED] Processing resumed")


def adjust_rate_limits_for_high_volume():
    """
    Automatically adjust rate limits for high volume scenarios.
    This function is deprecated.
    """
    pass

"""
Performance Configuration Settings
Adjust these values to optimize performance for your specific use case
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Batch Processing Settings
BATCH_SIZE = 1000  # Number of alerts to process in each batch
MAX_WORKERS = 10  # Number of parallel workers for processing

# Caching Settings
CACHE_TTL = 300  # Cache time-to-live in seconds (5 minutes)
CACHE_MAX_SIZE = 1000  # Maximum number of cached items

# API Rate Limiting
API_RATE_LIMIT = 3  # Requests per second for API calls
API_TIMEOUT = 30  # Timeout for API requests in seconds

# Memory Management
MAX_MEMORY_USAGE = 80  # Maximum memory usage percentage before cleanup
MEMORY_CLEANUP_INTERVAL = 1000  # Cleanup memory every N alerts

# Performance Thresholds
LARGE_ALERT_THRESHOLD = 1000  # Switch to batch processing above this number
OPTIMIZATION_THRESHOLD = 5000  # Enable advanced optimizations above this number

# Monitoring Settings
PERFORMANCE_MONITORING_ENABLED = True
METRICS_SAVE_INTERVAL = 60  # Save metrics every N seconds
SYSTEM_METRICS_INTERVAL = 10  # Record system metrics every N seconds

# Error Handling
MAX_RETRIES = 3  # Maximum number of retries for failed operations
ERROR_LOG_THRESHOLD = 10  # Log errors only if more than this many occur

# Database/Storage Settings
USE_DATABASE = False  # Set to True to use database instead of JSON files
DATABASE_BATCH_SIZE = 100  # Database operations batch size

# Network Settings
CONNECTION_POOL_SIZE = 20  # Number of connections in the pool
REQUEST_TIMEOUT = 30  # Timeout for HTTP requests

# Alert Processing Settings
ALERT_CHECK_INTERVAL = 60  # Seconds between alert checks
PARALLEL_ALERT_CHECKING = True  # Enable parallel alert checking
STOCK_GROUPING_ENABLED = True  # Group alerts by stock to minimize API calls

# Performance Profiles
PERFORMANCE_PROFILES: dict[str, dict[str, Any]] = {
    "development": {
        "batch_size": 100,
        "max_workers": 5,
        "cache_ttl": 60,
        "monitoring_enabled": True,
    },
    "production": {
        "batch_size": 1000,
        "max_workers": 10,
        "cache_ttl": 300,
        "monitoring_enabled": True,
    },
    "high_volume": {
        "batch_size": 2000,
        "max_workers": 20,
        "cache_ttl": 600,
        "monitoring_enabled": True,
    },
}


def get_performance_profile(profile_name: str = "production") -> dict[str, Any]:
    """Get performance settings for a specific profile."""
    return PERFORMANCE_PROFILES.get(profile_name, PERFORMANCE_PROFILES["production"]).copy()


def update_performance_settings(settings_dict: dict[str, Any]) -> None:
    """Update performance settings dynamically."""
    global BATCH_SIZE, MAX_WORKERS, CACHE_TTL, CACHE_MAX_SIZE

    if "batch_size" in settings_dict:
        BATCH_SIZE = settings_dict["batch_size"]
    if "max_workers" in settings_dict:
        MAX_WORKERS = settings_dict["max_workers"]
    if "cache_ttl" in settings_dict:
        CACHE_TTL = settings_dict["cache_ttl"]
    if "cache_max_size" in settings_dict:
        CACHE_MAX_SIZE = settings_dict["cache_max_size"]

    logger.info("Updated performance settings: %s", settings_dict)


def get_optimal_settings(alert_count: int) -> dict[str, int]:
    """Get optimal settings based on alert count."""
    if alert_count < 100:
        return {"batch_size": 50, "max_workers": 5, "cache_ttl": 60}
    if alert_count < 1000:
        return {"batch_size": 200, "max_workers": 8, "cache_ttl": 180}
    if alert_count < 10000:
        return {"batch_size": 1000, "max_workers": 10, "cache_ttl": 300}
    return {"batch_size": 2000, "max_workers": 15, "cache_ttl": 600}

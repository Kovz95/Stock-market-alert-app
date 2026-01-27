"""
Legacy utils.py - maintains backward compatibility.

This file now serves as a compatibility layer, re-exporting functions from their new locations
in the core/ and utils/ modules. New code should import directly from the new modules.

Week 4 Refactoring: Business logic extracted to:
- core/market_data/ - Market data loading and ticker mapping
- core/alerts/ - Alert models, validation, and processing
- core/indicators/ - Indicator catalog and configurations
- utils/ - Formatting, time utilities, rate limiting
"""

import os
import threading

from dotenv import load_dotenv

from data_access.json_bridge import enable_json_bridge
from indicators_lib import *

enable_json_bridge()


# Load environment variables from .env file
load_dotenv()

# Try to import streamlit for caching, but don't fail if not available
try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Enhanced Caching System:
# - Custom in-memory cache: 10 minutes, up to 10,000 entries (for rate limiting)
# - Streamlit cache: 5 minutes (300s) for price data, 3 minutes (180s) for stock data
# - Dual-layer caching provides both performance and persistence across sessions

MAX_DISCORD_MESSAGE_LENGTH = 2000

# Load configuration from settings module (replaces hardcoded secrets)
try:
    from src.stock_alert.config.settings import get_settings

    _settings = get_settings()
    FMP_API_KEY = _settings.fmp_api_key
    WEBHOOK_URL = _settings.webhook_url
    WEBHOOK_URL_2 = _settings.webhook_url_2
    WEBHOOK_URL_LOGGING = _settings.webhook_url_logging
    WEBHOOK_URL_LOGGING_2 = _settings.webhook_url_logging_2
except ImportError:
    # Fallback to environment variables directly if settings module not available
    FMP_API_KEY = os.getenv("FMP_API_KEY")
    if not FMP_API_KEY:
        raise ValueError("FMP_API_KEY environment variable is required")
    WEBHOOK_URL = os.getenv("WEBHOOK_URL")
    WEBHOOK_URL_2 = os.getenv("WEBHOOK_URL_2")
    WEBHOOK_URL_LOGGING = os.getenv("WEBHOOK_URL_LOGGING")
    if not WEBHOOK_URL_LOGGING:
        raise ValueError("WEBHOOK_URL_LOGGING environment variable is required")
    WEBHOOK_URL_LOGGING_2 = os.getenv("WEBHOOK_URL_LOGGING_2")
    if not WEBHOOK_URL_LOGGING_2:
        raise ValueError("WEBHOOK_URL_LOGGING_2 environment variable is required")

LOG_BUFFER = []

# Enhanced cache with TTL and size limits
fmp_cache = {}
fmp_cache_lock = threading.Lock()
MAX_CACHE_SIZE = 10000  # Maximum number of cached entries

# FMP API client is initialized later when needed
client = None

# Path to CSV file for storing exchange and stock data
CSV_FILE_PATH = "cleaned_data.csv"

# Path to main database with all securities
MAIN_DATABASE_PATH = "main_database_with_etfs.json"

# Path to CSV file for storing stock alerts
FUTURES_ALERTS_FILE_PATH = "futures_alerts.json"

# ============================================================================
# BACKWARD COMPATIBILITY IMPORTS
# Import functions from their new locations and re-export them
# ============================================================================

# Market Data & Ticker Mapping (core/market_data/)
# Alert Processing (core/alerts/)
from src.stock_alert.core.alerts.processor import (
    check_database,
    get_alert_by_id,
    get_all_alerts_for_stock,
    get_all_stocks,
    get_stock_exchange,
    load_alert_data,
    save_alert,
    save_ratio_alert,
    send_alert,
    send_stock_alert,
    update_alert,
    update_ratio_alert,
    update_stock_database,
)

# Alert Validation (core/alerts/)
from src.stock_alert.core.alerts.validator import (
    check_similar_alerts,
    check_similar_ratio_alerts,
    get_stock_alerts_summary,
    suggest_alert_update,
    suggest_ratio_alert_update,
    validate_conditions,
)

# Indicators (core/indicators/)
from src.stock_alert.core.indicators.catalog import (
    inverse_map,
    ops,
    period_and_input,
    period_only,
    predefined_suggestions,
    predefined_suggestions_alt,
    supported_indicators,
)
from src.stock_alert.core.market_data.calculations import (
    calculate_cross_exchange_ratio,
    calculate_ratio,
    normalize_dataframe,
)
from src.stock_alert.core.market_data.data_fetcher import (
    get_latest_stock_data as _get_latest_stock_data,
)
from src.stock_alert.core.market_data.data_fetcher import (
    grab_new_data_fmp as _grab_new_data_fmp,
)
from src.stock_alert.core.market_data.data_fetcher import (
    grab_new_data_universal as _grab_new_data_universal,
)
from src.stock_alert.core.market_data.data_fetcher import (
    load_market_data,
    process_alerts_in_batches,
)
from src.stock_alert.core.market_data.ticker_mapping import (
    FMP_TICKER_MAPPING,
    YAHOO_TICKER_MAPPING,
    get_fmp_ticker,
    get_fmp_ticker_with_fallback,
    get_yahoo_ticker,
    load_fmp_ticker_mapping,
    load_yahoo_ticker_mapping,
)
from src.stock_alert.core.market_data.ticker_mapping import (
    test_fmp_ticker_availability as _test_fmp_ticker_availability,
)

# Discord logging (services/discord/)
from src.stock_alert.services.discord.client import (
    flush_logs_to_discord,
    log_to_discord,
)

# Formatting Utilities (utils/)
from src.stock_alert.utils.formatting import (
    bl_sp,
    get_asset_type,
    is_us_stock,
    split_message,
)

# Rate Limiting (utils/)
from src.stock_alert.utils.rate_limiting import (
    adjust_rate_limits_for_high_volume,
    calculate_cache_hit_rate,
    emergency_rate_limit_pause,
    estimate_daily_requests,
    get_rate_limit_recommendations,
)

# Time Utilities (utils/)
from src.stock_alert.utils.time_utils import (
    convert_time_manual,
    get_dst_adjusted_time,
    get_dst_status,
    get_market_timezone,
    is_dst_active,
)


# Wrapper functions to maintain exact same API signatures
def test_fmp_ticker_availability(ticker):
    """Test if a ticker is available in FMP API - wrapper for backward compatibility."""
    return _test_fmp_ticker_availability(ticker, FMP_API_KEY)


def grab_new_data_fmp(ticker, timespan="1d", period="1y"):
    """Fetch historical data from FMP API - wrapper for backward compatibility."""
    return _grab_new_data_fmp(ticker, timespan, period, FMP_API_KEY)


def grab_new_data_universal(ticker, timespan="1d", period="1y"):
    """Universal data fetching function - wrapper for backward compatibility."""
    return _grab_new_data_universal(ticker, timespan, period, FMP_API_KEY)


@st.cache_data(ttl=900) if STREAMLIT_AVAILABLE else lambda func: func
def get_latest_stock_data(stock, exchange, timespan):
    """Fetch the latest stock data - wrapper for backward compatibility."""
    return _get_latest_stock_data(stock, exchange, timespan, FMP_API_KEY)


# Note: The following functions have been moved but are imported above:
# - _derive_country (now in core/alerts/processor.py)
# - _is_futures_symbol (now in core/alerts/processor.py)
# These are intentionally not re-exported as they were private functions

__all__ = [
    # Constants
    "MAX_DISCORD_MESSAGE_LENGTH",
    "FMP_API_KEY",
    "WEBHOOK_URL",
    "WEBHOOK_URL_2",
    "WEBHOOK_URL_LOGGING",
    "WEBHOOK_URL_LOGGING_2",
    "LOG_BUFFER",
    "fmp_cache",
    "fmp_cache_lock",
    "MAX_CACHE_SIZE",
    "CSV_FILE_PATH",
    "MAIN_DATABASE_PATH",
    "FUTURES_ALERTS_FILE_PATH",
    "FMP_TICKER_MAPPING",
    "YAHOO_TICKER_MAPPING",
    "predefined_suggestions",
    "predefined_suggestions_alt",
    "inverse_map",
    "ops",
    "supported_indicators",
    "period_and_input",
    "period_only",
    # Market Data & Ticker Mapping
    "get_fmp_ticker",
    "get_fmp_ticker_with_fallback",
    "test_fmp_ticker_availability",
    "load_fmp_ticker_mapping",
    "load_yahoo_ticker_mapping",
    "get_yahoo_ticker",
    "load_market_data",
    "grab_new_data_fmp",
    "grab_new_data_universal",
    "calculate_ratio",
    "calculate_cross_exchange_ratio",
    "normalize_dataframe",
    "process_alerts_in_batches",
    # Alert Processing
    "validate_conditions",
    "save_alert",
    "save_ratio_alert",
    "update_alert",
    "update_ratio_alert",
    "get_alert_by_id",
    "load_alert_data",
    "get_all_stocks",
    "get_stock_exchange",
    "get_all_alerts_for_stock",
    "get_latest_stock_data",
    "check_database",
    "update_stock_database",
    "send_alert",
    "send_stock_alert",
    # Alert Validation
    "check_similar_alerts",
    "check_similar_ratio_alerts",
    "suggest_alert_update",
    "suggest_ratio_alert_update",
    "get_stock_alerts_summary",
    # Formatting
    "bl_sp",
    "split_message",
    "get_asset_type",
    "is_us_stock",
    # Time Utilities
    "get_dst_adjusted_time",
    "convert_time_manual",
    "get_market_timezone",
    "is_dst_active",
    "get_dst_status",
    # Rate Limiting
    "calculate_cache_hit_rate",
    "estimate_daily_requests",
    "get_rate_limit_recommendations",
    "emergency_rate_limit_pause",
    "adjust_rate_limits_for_high_volume",
    # Discord
    "log_to_discord",
    "flush_logs_to_discord",
]

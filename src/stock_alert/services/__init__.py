"""
Services layer for Stock Market Alert App.

This module provides external service integrations including:
- Data providers (FMP, Interactive Brokers)
- Discord notifications
- Cache management (Redis)
"""

# Data providers
from .data_providers import (
    AbstractDataProvider,
    FMPDataProvider,
    OptimizedFMPDataProvider,
)
from .data_providers.factory import (
    DataProviderFactory,
    get_futures_provider,
    get_provider_for_symbol,
    get_stock_provider,
)

# Discord
from .discord import DiscordClient

# Cache
try:
    from .cache import RedisCache

    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    RedisCache = None

# IB Futures (optional)
try:
    from .data_providers import IBFuturesProvider

    IB_AVAILABLE = True
except (ImportError, AttributeError):
    IB_AVAILABLE = False
    IBFuturesProvider = None

__all__ = [
    # Data providers
    "AbstractDataProvider",
    "FMPDataProvider",
    "OptimizedFMPDataProvider",
    "DataProviderFactory",
    "get_stock_provider",
    "get_futures_provider",
    "get_provider_for_symbol",
    # Discord
    "DiscordClient",
]

if IB_AVAILABLE:
    __all__.append("IBFuturesProvider")

if CACHE_AVAILABLE:
    __all__.append("RedisCache")

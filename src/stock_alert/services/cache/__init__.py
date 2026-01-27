"""
Cache services for Stock Market Alert App.

Provides caching mechanisms for market data and application state.
"""

try:
    from .redis_client import RedisCache

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    RedisCache = None

__all__ = []

if REDIS_AVAILABLE:
    __all__.append("RedisCache")

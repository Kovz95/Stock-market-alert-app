"""
Redis cache client for Stock Market Alert App.

Provides a simple interface for caching market data and application state.
"""

import json
import logging
from typing import Any

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

from src.stock_alert.config.settings import Settings

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis cache client with JSON serialization support.

    Provides simple get/set operations with automatic serialization.
    """

    def __init__(self, redis_url: str | None = None, ttl: int = 3600):
        """
        Initialize Redis cache.

        Args:
            redis_url: Redis connection URL (default from settings)
            ttl: Default time-to-live in seconds (default 1 hour)
        """
        if not REDIS_AVAILABLE:
            logger.warning("redis-py not installed. Caching disabled.")
            self.client = None
            return

        try:
            settings = Settings()
            url = redis_url or settings.REDIS_URL
            self.client = redis.from_url(url, decode_responses=True)
            self.ttl = ttl
            logger.info(f"Connected to Redis at {url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.client = None

    def is_available(self) -> bool:
        """Check if Redis is available"""
        if not self.client:
            return False
        try:
            self.client.ping()
            return True
        except Exception:
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from cache.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        if not self.client:
            return default

        try:
            value = self.client.get(key)
            if value is None:
                return default
            # Try to deserialize JSON
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
        except Exception as e:
            logger.error(f"Error getting key {key}: {e}")
            return default

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """
        Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache (will be JSON serialized)
            ttl: Time-to-live in seconds (overrides default)

        Returns:
            True if successful
        """
        if not self.client:
            return False

        try:
            # Serialize value to JSON
            if isinstance(value, (dict, list, tuple)) or isinstance(value, (int, float, str, bool)):
                serialized = json.dumps(value)
            else:
                serialized = str(value)

            ttl_seconds = ttl if ttl is not None else self.ttl
            self.client.setex(key, ttl_seconds, serialized)
            return True
        except Exception as e:
            logger.error(f"Error setting key {key}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete a key from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        if not self.client:
            return False

        try:
            self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting key {key}: {e}")
            return False

    def exists(self, key: str) -> bool:
        """
        Check if a key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if exists
        """
        if not self.client:
            return False

        try:
            return bool(self.client.exists(key))
        except Exception as e:
            logger.error(f"Error checking key {key}: {e}")
            return False

    def clear(self) -> bool:
        """
        Clear all keys in cache.

        WARNING: This clears the entire Redis database!

        Returns:
            True if successful
        """
        if not self.client:
            return False

        try:
            self.client.flushdb()
            logger.info("Redis cache cleared")
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    def get_ttl(self, key: str) -> int:
        """
        Get time-to-live for a key.

        Args:
            key: Cache key

        Returns:
            TTL in seconds, -1 if no expiry, -2 if key doesn't exist
        """
        if not self.client:
            return -2

        try:
            return self.client.ttl(key)
        except Exception as e:
            logger.error(f"Error getting TTL for key {key}: {e}")
            return -2

    def close(self):
        """Close Redis connection"""
        if self.client:
            self.client.close()
            logger.info("Redis connection closed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


# Global cache instance
_cache_instance: RedisCache | None = None


def get_cache() -> RedisCache:
    """
    Get global Redis cache instance.

    Returns:
        RedisCache instance
    """
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache()
    return _cache_instance

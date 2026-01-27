"""
Redis support utilities for caching and key-value storage.

Provides a simple interface for JSON-based caching with Redis.
Gracefully handles missing Redis connections by returning None/no-op.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Redis key prefix for namespacing
KEY_PREFIX = os.getenv("REDIS_KEY_PREFIX", "stockalert:")

# Redis connection settings
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_URL = os.getenv("REDIS_URL")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

# Connection timeout settings
REDIS_SOCKET_TIMEOUT = float(os.getenv("REDIS_SOCKET_TIMEOUT", "5.0"))
REDIS_SOCKET_CONNECT_TIMEOUT = float(os.getenv("REDIS_SOCKET_CONNECT_TIMEOUT", "5.0"))

# Global client instance
_client: Optional[Any] = None
_client_initialized = False


def _get_redis_module():
    """Import redis module, returning None if not available."""
    try:
        import redis
        return redis
    except ImportError:
        return None


def get_client() -> Optional[Any]:
    """
    Get the Redis client instance.

    Returns:
        Redis client if available and connected, None otherwise.
    """
    global _client, _client_initialized

    if _client_initialized:
        return _client

    redis_module = _get_redis_module()
    if redis_module is None:
        logger.debug("Redis module not installed; caching disabled")
        _client_initialized = True
        return None

    try:
        if REDIS_URL:
            _client = redis_module.from_url(
                REDIS_URL,
                socket_timeout=REDIS_SOCKET_TIMEOUT,
                socket_connect_timeout=REDIS_SOCKET_CONNECT_TIMEOUT,
                decode_responses=True,
            )
        else:
            _client = redis_module.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB,
                password=REDIS_PASSWORD,
                socket_timeout=REDIS_SOCKET_TIMEOUT,
                socket_connect_timeout=REDIS_SOCKET_CONNECT_TIMEOUT,
                decode_responses=True,
            )
        # Test connection
        _client.ping()
        logger.debug("Redis connection established to %s:%s", REDIS_HOST, REDIS_PORT)
    except Exception as exc:
        logger.warning("Failed to connect to Redis: %s", exc)
        _client = None

    _client_initialized = True
    return _client


def build_key(key: str) -> str:
    """
    Build a Redis key with the configured prefix.

    Args:
        key: The base key name.

    Returns:
        Prefixed key string.
    """
    if key.startswith(KEY_PREFIX):
        return key
    return f"{KEY_PREFIX}{key}"


def get_json(key: str) -> Optional[Any]:
    """
    Retrieve a JSON value from Redis.

    Args:
        key: Redis key (will be prefixed if needed).

    Returns:
        Deserialized JSON value, or None if key doesn't exist or Redis unavailable.
    """
    client = get_client()
    if client is None:
        return None

    try:
        full_key = build_key(key) if not key.startswith(KEY_PREFIX) else key
        value = client.get(full_key)
        if value is None:
            return None
        return json.loads(value)
    except Exception as exc:
        logger.debug("Error getting key %s from Redis: %s", key, exc)
        return None


def set_json(
    key: str,
    value: Any,
    *,
    ttl_seconds: Optional[int] = None,
) -> bool:
    """
    Store a JSON value in Redis.

    Args:
        key: Redis key (will be prefixed if needed).
        value: JSON-serializable value to store.
        ttl_seconds: Optional TTL in seconds.

    Returns:
        True if successful, False otherwise.
    """
    client = get_client()
    if client is None:
        return False

    try:
        full_key = build_key(key) if not key.startswith(KEY_PREFIX) else key
        serialized = json.dumps(value, default=str)
        if ttl_seconds:
            client.setex(full_key, ttl_seconds, serialized)
        else:
            client.set(full_key, serialized)
        return True
    except Exception as exc:
        logger.debug("Error setting key %s in Redis: %s", key, exc)
        return False


def delete_key(key: str) -> bool:
    """
    Delete a key from Redis.

    Args:
        key: Redis key (will be prefixed if needed).

    Returns:
        True if successful, False otherwise.
    """
    client = get_client()
    if client is None:
        return False

    try:
        full_key = build_key(key) if not key.startswith(KEY_PREFIX) else key
        client.delete(full_key)
        return True
    except Exception as exc:
        logger.debug("Error deleting key %s from Redis: %s", key, exc)
        return False


def reset_client() -> None:
    """
    Reset the Redis client connection.

    Call this to force reconnection on next operation.
    """
    global _client, _client_initialized
    if _client is not None:
        try:
            _client.close()
        except Exception:
            pass
    _client = None
    _client_initialized = False

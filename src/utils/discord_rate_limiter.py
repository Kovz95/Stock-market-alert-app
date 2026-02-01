#!/usr/bin/env python3
"""
Discord Rate Limiter

Manages Discord webhook rate limiting to prevent hitting API limits.
Discord rate limits:
- 5 requests per 2 seconds per webhook
- 30 requests per minute per webhook
"""

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Dict, Tuple, Any, Optional

import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiscordRateLimiter:
    """
    Rate limiter for Discord webhooks to prevent hitting API limits.
    Tracks requests per webhook URL and enforces rate limits.
    """

    def __init__(
        self,
        short_limit: int = 5,
        short_window: float = 2.0,
        long_limit: int = 30,
        long_window: float = 60.0,
    ):
        """
        Initialize the rate limiter.

        Args:
            short_limit: Maximum requests in short window (default: 5)
            short_window: Short window duration in seconds (default: 2.0)
            long_limit: Maximum requests in long window (default: 30)
            long_window: Long window duration in seconds (default: 60.0)
        """
        self.short_limit = short_limit
        self.short_window = short_window
        self.long_limit = long_limit
        self.long_window = long_window

        # Track request timestamps per webhook URL
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque())

        # Lock for thread safety
        self.lock = threading.Lock()

        # Statistics
        self.total_sent = 0
        self.total_rate_limited = 0
        self.total_errors = 0

        logger.info(
            f"Discord rate limiter initialized: "
            f"{short_limit} requests per {short_window}s, "
            f"{long_limit} requests per {long_window}s"
        )

    def _clean_old_requests(self, webhook_url: str, current_time: float) -> None:
        """
        Remove timestamps older than the long window.

        Args:
            webhook_url: The webhook URL to clean
            current_time: Current timestamp
        """
        history = self.request_history[webhook_url]
        cutoff = current_time - self.long_window

        while history and history[0] < cutoff:
            history.popleft()

    def _can_send(self, webhook_url: str, current_time: float) -> Tuple[bool, Optional[float]]:
        """
        Check if a request can be sent without violating rate limits.

        Args:
            webhook_url: The webhook URL to check
            current_time: Current timestamp

        Returns:
            Tuple of (can_send, wait_time)
            - can_send: True if request can be sent
            - wait_time: Seconds to wait if can_send is False, None otherwise
        """
        history = self.request_history[webhook_url]

        # Clean old requests
        self._clean_old_requests(webhook_url, current_time)

        # Check short window (last 2 seconds)
        short_cutoff = current_time - self.short_window
        short_count = sum(1 for ts in history if ts >= short_cutoff)

        if short_count >= self.short_limit:
            # Calculate wait time based on oldest request in short window
            oldest_in_short = min((ts for ts in history if ts >= short_cutoff), default=current_time)
            wait_time = self.short_window - (current_time - oldest_in_short) + 0.1
            return False, wait_time

        # Check long window (last 60 seconds)
        if len(history) >= self.long_limit:
            # Calculate wait time based on oldest request
            oldest = history[0]
            wait_time = self.long_window - (current_time - oldest) + 0.1
            return False, wait_time

        return True, None

    def send_with_rate_limit(
        self,
        webhook_url: str,
        payload: Dict[str, Any],
        timeout: float = 10.0,
        max_retries: int = 3,
    ) -> Tuple[bool, int]:
        """
        Send a Discord webhook message with rate limiting.

        Args:
            webhook_url: Discord webhook URL
            payload: Message payload (dict with 'content', 'username', etc.)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries on rate limit

        Returns:
            Tuple of (success, status_code)
            - success: True if message sent successfully
            - status_code: HTTP status code or 0 for errors
        """
        with self.lock:
            current_time = time.time()

            # Check if we can send
            can_send, wait_time = self._can_send(webhook_url, current_time)

            if not can_send and wait_time:
                if max_retries > 0:
                    logger.debug(
                        f"Rate limit hit for {webhook_url[:50]}... "
                        f"Waiting {wait_time:.2f}s (retries left: {max_retries})"
                    )
                    self.total_rate_limited += 1

                    # Release lock before sleeping
                    self.lock.release()
                    time.sleep(wait_time)
                    self.lock.acquire()

                    # Retry
                    return self.send_with_rate_limit(
                        webhook_url, payload, timeout, max_retries - 1
                    )
                else:
                    logger.warning(
                        f"Rate limit exceeded for {webhook_url[:50]}..., "
                        f"no retries left"
                    )
                    self.total_rate_limited += 1
                    return False, 429  # Too Many Requests

            # Send the request
            try:
                response = requests.post(webhook_url, json=payload, timeout=timeout)
                status_code = response.status_code

                if status_code == 204:
                    # Success - record the timestamp
                    self.request_history[webhook_url].append(current_time)
                    self.total_sent += 1
                    logger.debug(f"Successfully sent webhook to {webhook_url[:50]}...")
                    return True, status_code

                elif status_code == 429:
                    # Discord returned rate limit response
                    self.total_rate_limited += 1

                    # Try to get retry_after from response
                    try:
                        retry_after = response.json().get('retry_after', 1.0)
                    except Exception:
                        retry_after = 1.0

                    if max_retries > 0:
                        logger.warning(
                            f"Discord rate limit response for {webhook_url[:50]}... "
                            f"Waiting {retry_after}s"
                        )

                        # Release lock before sleeping
                        self.lock.release()
                        time.sleep(retry_after)
                        self.lock.acquire()

                        # Retry
                        return self.send_with_rate_limit(
                            webhook_url, payload, timeout, max_retries - 1
                        )
                    else:
                        logger.error(
                            f"Discord rate limit, no retries left for {webhook_url[:50]}..."
                        )
                        return False, status_code

                else:
                    # Other error
                    self.total_errors += 1
                    logger.error(
                        f"Failed to send webhook to {webhook_url[:50]}...: "
                        f"{status_code} - {response.text[:200]}"
                    )
                    return False, status_code

            except requests.exceptions.Timeout:
                self.total_errors += 1
                logger.error(f"Timeout sending webhook to {webhook_url[:50]}...")
                return False, 0

            except Exception as e:
                self.total_errors += 1
                logger.error(f"Error sending webhook to {webhook_url[:50]}...: {e}")
                return False, 0

    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current status of the rate limiter.

        Returns:
            Dictionary with rate limiter statistics
        """
        with self.lock:
            current_time = time.time()

            # Calculate current load per webhook
            webhook_stats = {}
            for webhook_url, history in self.request_history.items():
                # Clean old requests
                self._clean_old_requests(webhook_url, current_time)

                # Count recent requests
                short_cutoff = current_time - self.short_window
                short_count = sum(1 for ts in history if ts >= short_cutoff)

                webhook_stats[webhook_url[:50]] = {
                    'short_window_count': short_count,
                    'short_window_limit': self.short_limit,
                    'long_window_count': len(history),
                    'long_window_limit': self.long_limit,
                    'utilization_short': f"{(short_count / self.short_limit * 100):.1f}%",
                    'utilization_long': f"{(len(history) / self.long_limit * 100):.1f}%",
                }

            return {
                'rate_limiting': 'enabled',
                'short_limit': f"{self.short_limit} requests per {self.short_window}s",
                'long_limit': f"{self.long_limit} requests per {self.long_window}s",
                'total_sent': self.total_sent,
                'total_rate_limited': self.total_rate_limited,
                'total_errors': self.total_errors,
                'tracked_webhooks': len(self.request_history),
                'webhook_stats': webhook_stats,
            }

    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        with self.lock:
            self.total_sent = 0
            self.total_rate_limited = 0
            self.total_errors = 0
            logger.info("Rate limiter statistics reset")


# Global rate limiter instance
_rate_limiter: Optional[DiscordRateLimiter] = None
_limiter_lock = threading.Lock()


def get_rate_limiter() -> DiscordRateLimiter | None:
    """
    Get the global Discord rate limiter instance (singleton pattern).

    Returns:
        DiscordRateLimiter instance
    """
    global _rate_limiter

    if _rate_limiter is None:
        with _limiter_lock:
            if _rate_limiter is None:
                _rate_limiter = DiscordRateLimiter()

    return _rate_limiter


def create_rate_limiter(
    short_limit: int = 5,
    short_window: float = 2.0,
    long_limit: int = 30,
    long_window: float = 60.0,
) -> DiscordRateLimiter:
    """
    Create a new Discord rate limiter with custom limits.

    Args:
        short_limit: Maximum requests in short window
        short_window: Short window duration in seconds
        long_limit: Maximum requests in long window
        long_window: Long window duration in seconds

    Returns:
        New DiscordRateLimiter instance
    """
    return DiscordRateLimiter(short_limit, short_window, long_limit, long_window)

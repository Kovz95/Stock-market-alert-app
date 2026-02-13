#!/usr/bin/env python3
"""
Discord Rate Limiter

Manages Discord webhook rate limiting by parsing API response headers
rather than hardcoding limits. Per Discord's documentation, rate limits
should not be hardcoded -- instead, apps should parse response headers
to prevent hitting limits and respond accordingly when they do.

See: https://discord.com/developers/docs/topics/rate-limits
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Tuple, Any, Optional

import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BucketState:
    """Tracks the rate limit state for a specific rate limit bucket."""

    bucket_id: str
    limit: int = 0
    remaining: int = 1  # Assume we can send until told otherwise
    reset: float = 0.0  # Epoch time when the bucket resets
    reset_after: float = 0.0  # Seconds until the bucket resets
    last_updated: float = field(default_factory=time.time)

    def is_exhausted(self) -> bool:
        """Check if this bucket has no remaining requests."""
        if self.remaining > 0:
            return False
        # Check if the reset time has passed
        if time.time() >= self.reset:
            return False
        return True

    def wait_time(self) -> float:
        """Calculate how long to wait before this bucket resets."""
        if not self.is_exhausted():
            return 0.0
        remaining_wait = self.reset - time.time()
        return max(remaining_wait + 0.1, 0.0)  # Small buffer


class DiscordRateLimiter:
    """
    Rate limiter for Discord webhooks that respects API response headers.

    Instead of hardcoding rate limits, this tracks per-bucket limits using
    the X-RateLimit-* headers returned by Discord's API. On 429 responses,
    it uses the retry_after value from the response body to determine
    when to retry.
    """

    def __init__(self, max_retries: int = 5):
        """
        Initialize the rate limiter.

        Args:
            max_retries: Default maximum retries on rate limit (default: 5).
        """
        self.default_max_retries = max_retries

        # Map webhook URL -> bucket ID (discovered from response headers)
        self._url_to_bucket: Dict[str, str] = {}

        # Map bucket ID -> BucketState
        self._bucket_states: Dict[str, BucketState] = {}

        # Global rate limit: epoch time when the global cooldown expires
        self._global_retry_after: float = 0.0

        # Lock for thread safety
        self.lock = threading.Lock()

        # Statistics
        self.total_sent = 0
        self.total_rate_limited = 0
        self.total_errors = 0

        logger.info(
            "Discord rate limiter initialized (header-based, "
            f"max_retries={max_retries})"
        )

    def _update_rate_limit_state(
        self, webhook_url: str, response: requests.Response
    ) -> None:
        """
        Parse rate limit headers from a Discord API response and update
        internal tracking state.

        Args:
            webhook_url: The webhook URL that was called.
            response: The HTTP response from Discord.
        """
        headers = response.headers

        bucket_id = headers.get("X-RateLimit-Bucket")
        if not bucket_id:
            return

        # Map this URL to the discovered bucket
        self._url_to_bucket[webhook_url] = bucket_id

        # Parse rate limit headers
        try:
            limit = int(headers.get("X-RateLimit-Limit", 0))
            remaining = int(headers.get("X-RateLimit-Remaining", 1))
            reset = float(headers.get("X-RateLimit-Reset", 0.0))
            reset_after = float(headers.get("X-RateLimit-Reset-After", 0.0))
        except (ValueError, TypeError):
            logger.warning(
                f"Failed to parse rate limit headers for {webhook_url[:50]}..."
            )
            return

        if bucket_id in self._bucket_states:
            state = self._bucket_states[bucket_id]
            state.limit = limit
            state.remaining = remaining
            state.reset = reset
            state.reset_after = reset_after
            state.last_updated = time.time()
        else:
            self._bucket_states[bucket_id] = BucketState(
                bucket_id=bucket_id,
                limit=limit,
                remaining=remaining,
                reset=reset,
                reset_after=reset_after,
            )

        logger.debug(
            f"Rate limit state for bucket {bucket_id}: "
            f"{remaining}/{limit} remaining, resets in {reset_after:.2f}s"
        )

    def _get_preemptive_wait(self, webhook_url: str) -> float:
        """
        Check if we should wait before sending based on known bucket state.

        Args:
            webhook_url: The webhook URL to check.

        Returns:
            Seconds to wait (0.0 if no wait is needed).
        """
        now = time.time()

        # Check global rate limit first
        if now < self._global_retry_after:
            return self._global_retry_after - now + 0.1

        # Check per-bucket rate limit
        bucket_id = self._url_to_bucket.get(webhook_url)
        if bucket_id and bucket_id in self._bucket_states:
            state = self._bucket_states[bucket_id]
            return state.wait_time()

        return 0.0

    def send_with_rate_limit(
        self,
        webhook_url: str,
        payload: Dict[str, Any],
        timeout: float = 10.0,
        max_retries: Optional[int] = None,
    ) -> Tuple[bool, int]:
        """
        Send a Discord webhook message with rate limit handling.

        Preemptively waits if a known bucket is exhausted, and retries
        with the server-provided retry_after value on 429 responses.

        Args:
            webhook_url: Discord webhook URL.
            payload: Message payload (dict with 'content', 'embeds', etc.).
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retries on rate limit.
                         Defaults to self.default_max_retries.

        Returns:
            Tuple of (success, status_code)
            - success: True if message sent successfully
            - status_code: HTTP status code or 0 for errors
        """
        if max_retries is None:
            max_retries = self.default_max_retries

        for attempt in range(max_retries + 1):
            with self.lock:
                # Preemptive wait based on known bucket state
                wait_time = self._get_preemptive_wait(webhook_url)

            if wait_time > 0:
                logger.debug(
                    f"Preemptive wait {wait_time:.2f}s for "
                    f"{webhook_url[:50]}... (attempt {attempt + 1})"
                )
                time.sleep(wait_time)

            # Send the request
            try:
                response = requests.post(
                    webhook_url, json=payload, timeout=timeout
                )
                status_code = response.status_code

                with self.lock:
                    # Always update state from response headers
                    self._update_rate_limit_state(webhook_url, response)

                if status_code in (200, 204):
                    # Success
                    with self.lock:
                        self.total_sent += 1
                    logger.debug(
                        f"Successfully sent webhook to {webhook_url[:50]}..."
                    )
                    return True, status_code

                elif status_code == 429:
                    # Rate limited -- parse retry_after from response
                    with self.lock:
                        self.total_rate_limited += 1

                    retry_after = self._parse_retry_after(response)
                    is_global = self._is_global_rate_limit(response)
                    scope = response.headers.get("X-RateLimit-Scope", "unknown")

                    if is_global:
                        with self.lock:
                            self._global_retry_after = time.time() + retry_after
                        logger.warning(
                            f"Global rate limit hit! "
                            f"Waiting {retry_after:.2f}s (scope: {scope})"
                        )
                    else:
                        logger.warning(
                            f"Rate limited on {webhook_url[:50]}... "
                            f"Waiting {retry_after:.2f}s "
                            f"(scope: {scope}, "
                            f"attempt {attempt + 1}/{max_retries + 1})"
                        )

                    if attempt < max_retries:
                        time.sleep(retry_after)
                        continue
                    else:
                        logger.error(
                            f"Rate limit exceeded, no retries left "
                            f"for {webhook_url[:50]}..."
                        )
                        return False, status_code

                else:
                    # Other HTTP error
                    with self.lock:
                        self.total_errors += 1
                    logger.error(
                        f"Failed to send webhook to {webhook_url[:50]}...: "
                        f"{status_code} - {response.text[:200]}"
                    )
                    return False, status_code

            except requests.exceptions.Timeout:
                with self.lock:
                    self.total_errors += 1
                logger.error(
                    f"Timeout sending webhook to {webhook_url[:50]}..."
                )
                return False, 0

            except requests.exceptions.ConnectionError as e:
                with self.lock:
                    self.total_errors += 1
                logger.error(
                    f"Connection error sending webhook to "
                    f"{webhook_url[:50]}...: {e}"
                )
                return False, 0

            except Exception as e:
                with self.lock:
                    self.total_errors += 1
                logger.error(
                    f"Error sending webhook to {webhook_url[:50]}...: {e}"
                )
                return False, 0

        # Should not reach here, but just in case
        return False, 429

    @staticmethod
    def _parse_retry_after(response: requests.Response) -> float:
        """
        Extract retry_after from a 429 response.

        Prefers the JSON body's retry_after field, falls back to the
        Retry-After header, then defaults to 1.0s.

        Args:
            response: The 429 HTTP response.

        Returns:
            Seconds to wait before retrying.
        """
        # Try JSON body first (more precise, can have decimals)
        try:
            body = response.json()
            retry_after = body.get("retry_after")
            if retry_after is not None:
                return float(retry_after)
        except (ValueError, KeyError, AttributeError):
            pass

        # Fall back to Retry-After header
        retry_header = response.headers.get("Retry-After")
        if retry_header:
            try:
                return float(retry_header)
            except (ValueError, TypeError):
                pass

        # Default fallback
        logger.warning("Could not parse retry_after, defaulting to 1.0s")
        return 1.0

    @staticmethod
    def _is_global_rate_limit(response: requests.Response) -> bool:
        """
        Check if a 429 response indicates a global rate limit.

        Args:
            response: The 429 HTTP response.

        Returns:
            True if this is a global rate limit.
        """
        # Check header
        if response.headers.get("X-RateLimit-Global", "").lower() == "true":
            return True

        # Check JSON body
        try:
            body = response.json()
            return body.get("global", False) is True
        except (ValueError, KeyError, AttributeError):
            pass

        return False

    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get current status of the rate limiter.

        Returns:
            Dictionary with rate limiter statistics and bucket states.
        """
        with self.lock:
            now = time.time()

            bucket_info = {}
            for bucket_id, state in self._bucket_states.items():
                bucket_info[bucket_id] = {
                    "limit": state.limit,
                    "remaining": state.remaining,
                    "reset_after": f"{max(state.reset - now, 0):.1f}s",
                    "exhausted": state.is_exhausted(),
                    "last_updated": f"{now - state.last_updated:.1f}s ago",
                }

            global_active = now < self._global_retry_after
            global_wait = (
                f"{self._global_retry_after - now:.1f}s"
                if global_active
                else "none"
            )

            return {
                "rate_limiting": "enabled (header-based)",
                "total_sent": self.total_sent,
                "total_rate_limited": self.total_rate_limited,
                "total_errors": self.total_errors,
                "tracked_buckets": len(self._bucket_states),
                "tracked_urls": len(self._url_to_bucket),
                "global_rate_limit_active": global_active,
                "global_wait_remaining": global_wait,
                "buckets": bucket_info,
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

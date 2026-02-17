"""
Synchronous Discord Message Accumulator

Collects Discord embeds per webhook URL across an entire alert-checking job,
auto-flushes when a bucket reaches 10 embeds (Discord's per-message limit),
and sends remaining partial batches via flush_all() after all alerts finish.

This replaces the async queue path for alert jobs, ensuring maximum batching
and minimal webhook POST calls.
"""

import logging
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Discord allows up to 10 embeds per message
_MAX_EMBEDS_PER_MESSAGE = 10


class DiscordMessageAccumulator:
    """
    Thread-safe accumulator that batches Discord embeds by webhook URL.

    Usage:
        accumulator = DiscordMessageAccumulator(rate_limiter=get_rate_limiter())
        # ... inside each alert check:
        accumulator.add(webhook_url, embed)
        # ... after all alerts finish:
        accumulator.flush_all()
        logger.info("Accumulator stats: %s", accumulator.get_stats())
    """

    def __init__(self, rate_limiter=None, auto_flush: bool = True):
        """
        Initialize the accumulator.

        Args:
            rate_limiter: Optional DiscordRateLimiter instance. Passed through
                          to send_batch_embeds() for 429 handling.
            auto_flush: If True (default), automatically send batches when a
                        bucket reaches 10 embeds. If False, only append to
                        buckets — all sends happen in flush_all().
        """
        self._lock = threading.Lock()
        self._buckets: Dict[str, List[Dict[str, Any]]] = {}
        self._rate_limiter = rate_limiter
        self._auto_flush = auto_flush

        # Stats
        self._added = 0
        self._sent = 0
        self._failed = 0
        self._flushes = 0

    def add(self, webhook_url: str, embed: Dict[str, Any]) -> None:
        """
        Append an embed to the bucket for *webhook_url*.

        If the bucket reaches the 10-embed limit, the full batch is extracted
        under the lock and sent outside the lock (so HTTP I/O does not block
        other threads from adding embeds).

        Args:
            webhook_url: Discord webhook URL.
            embed: Discord embed dict.
        """
        batch_to_send: Optional[List[Dict[str, Any]]] = None

        with self._lock:
            self._added += 1
            bucket = self._buckets.setdefault(webhook_url, [])
            bucket.append(embed)

            if self._auto_flush and len(bucket) >= _MAX_EMBEDS_PER_MESSAGE:
                batch_to_send = bucket[:_MAX_EMBEDS_PER_MESSAGE]
                self._buckets[webhook_url] = bucket[_MAX_EMBEDS_PER_MESSAGE:]

        if batch_to_send is not None:
            self._send_batch(webhook_url, batch_to_send)

    def flush_all(self) -> None:
        """
        Send all remaining partial batches.

        Call this after every alert in the job has been processed so that
        no embeds are left unsent.
        """
        pending: Dict[str, List[Dict[str, Any]]] = {}

        with self._lock:
            pending = dict(self._buckets)
            self._buckets.clear()

        for webhook_url, embeds in pending.items():
            # Chunk into groups of 10 (should usually be < 10)
            for i in range(0, len(embeds), _MAX_EMBEDS_PER_MESSAGE):
                chunk = embeds[i:i + _MAX_EMBEDS_PER_MESSAGE]
                self._send_batch(webhook_url, chunk)

    def _send_batch(self, webhook_url: str, embeds: List[Dict[str, Any]]) -> None:
        """
        Send a batch of embeds via send_batch_embeds().

        Args:
            webhook_url: Discord webhook URL.
            embeds: List of embed dicts (max 10).
        """
        from src.services.discord_routing import send_batch_embeds

        try:
            success = send_batch_embeds(webhook_url, embeds, self._rate_limiter)
            with self._lock:
                self._flushes += 1
                if success:
                    self._sent += len(embeds)
                else:
                    self._failed += len(embeds)

            if success:
                logger.info(
                    "Sent batch of %d embeds to %s…",
                    len(embeds), webhook_url[:50],
                )
            else:
                logger.warning(
                    "Failed batch of %d embeds to %s…",
                    len(embeds), webhook_url[:50],
                )
        except Exception as exc:
            with self._lock:
                self._flushes += 1
                self._failed += len(embeds)
            logger.error("Error sending embed batch: %s", exc)

    def get_stats(self) -> Dict[str, int]:
        """
        Return accumulator statistics.

        Returns:
            Dictionary with added, sent, failed, and flushes counts.
        """
        with self._lock:
            return {
                "added": self._added,
                "sent": self._sent,
                "failed": self._failed,
                "flushes": self._flushes,
            }

#!/usr/bin/env python3
"""
Async Discord Notification Queue

Queues Discord notifications to be sent in a background thread,
preventing Discord rate limits from blocking the main scheduler jobs.
"""

import logging
import os
import queue
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class NotificationTask:
    """A queued notification task."""
    alert: Dict[str, Any]
    message: str
    send_func: Callable[[Dict[str, Any], str], bool]
    queued_at: datetime
    priority: int = 0  # Lower = higher priority
    webhook_url: Optional[str] = None  # Pre-resolved webhook URL for batching
    embed: Optional[Dict[str, Any]] = None  # Discord embed dict for batching


class AsyncDiscordQueue:
    """
    Background queue for Discord notifications.

    Notifications are queued and sent by a background thread,
    allowing the main job to complete without waiting for rate limits.
    """

    _instance: Optional['AsyncDiscordQueue'] = None
    _lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern - only one queue instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._initialized = True

        # Statistics
        self.stats = {
            'queued': 0,
            'sent': 0,
            'failed': 0,
            'dropped': 0,
        }
        self._stats_lock = threading.Lock()

        # Configuration
        self.max_queue_size = 1000  # Max pending notifications
        self.batch_delay = 0.1  # Seconds between sends (on top of rate limiter)
        self.batch_linger_seconds = float(
            os.getenv("DISCORD_BATCH_LINGER_SECONDS", "3.0")
        )  # Wait this long after the first item for more to accumulate

        logger.info("AsyncDiscordQueue initialized")

    def start(self) -> None:
        """Start the background worker thread."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            logger.debug("Worker thread already running")
            return

        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="AsyncDiscordWorker",
            daemon=True
        )
        self._worker_thread.start()
        logger.info("AsyncDiscordQueue worker started")

    def stop(self, timeout: float = 30.0) -> None:
        """Stop the background worker thread."""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            return

        logger.info("Stopping AsyncDiscordQueue worker...")
        self._stop_event.set()
        self._worker_thread.join(timeout=timeout)

        if self._worker_thread.is_alive():
            logger.warning("Worker thread did not stop gracefully")
        else:
            logger.info("AsyncDiscordQueue worker stopped")

    def enqueue(
        self,
        alert: Dict[str, Any],
        message: str,
        send_func: Callable[[Dict[str, Any], str], bool],
        priority: int = 0,
        webhook_url: Optional[str] = None,
        embed: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Add a notification to the queue.

        Args:
            alert: Alert dictionary
            message: Formatted message to send
            send_func: Function to call to send the notification
            priority: Lower = higher priority (default 0)
            webhook_url: Pre-resolved Discord webhook URL (enables batching)
            embed: Discord embed dict (enables batching with embeds)

        Returns:
            True if queued successfully, False if queue is full
        """
        # Start worker if not running
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self.start()

        # Check queue size
        if self._queue.qsize() >= self.max_queue_size:
            with self._stats_lock:
                self.stats['dropped'] += 1
            logger.warning(
                f"Discord queue full ({self.max_queue_size}), dropping notification for "
                f"{alert.get('ticker', 'unknown')}"
            )
            return False

        task = NotificationTask(
            alert=alert,
            message=message,
            send_func=send_func,
            queued_at=datetime.now(tz=timezone.utc),
            priority=priority,
            webhook_url=webhook_url,
            embed=embed,
        )

        # PriorityQueue uses (priority, item) tuples
        self._queue.put((priority, task))

        with self._stats_lock:
            self.stats['queued'] += 1

        ticker = alert.get('ticker', 'unknown')
        logger.debug(f"Queued notification for {ticker} (queue size: {self._queue.qsize()})")
        return True

    # -- batching helpers -----------------------------------------------------

    _MAX_BATCH_SIZE = 10  # Discord allows up to 10 embeds per message

    def _send_embed_batch(self, webhook_url: str, tasks: List[NotificationTask]) -> None:
        """Send a list of embed tasks as a single batched webhook POST."""
        from src.services.discord_routing import send_batch_embeds
        from src.utils.discord_rate_limiter import get_rate_limiter

        embeds = [t.embed for t in tasks if t.embed]
        if not embeds:
            return

        rate_limiter = get_rate_limiter()
        try:
            success = send_batch_embeds(webhook_url, embeds, rate_limiter)
            with self._stats_lock:
                if success:
                    self.stats['sent'] += len(tasks)
                else:
                    self.stats['failed'] += len(tasks)

            tickers = ", ".join(t.alert.get("ticker", "?") for t in tasks)
            if success:
                logger.info(
                    "Sent batch of %d embeds to %s… [%s]",
                    len(embeds), webhook_url[:50], tickers,
                )
            else:
                logger.warning(
                    "Failed batch of %d embeds to %s… [%s]",
                    len(embeds), webhook_url[:50], tickers,
                )
        except Exception as exc:
            with self._stats_lock:
                self.stats['failed'] += len(tasks)
            logger.error("Error sending embed batch: %s", exc)

    def _send_legacy_task(self, task: NotificationTask) -> None:
        """Send a single notification using the legacy send_func path."""
        ticker = task.alert.get('ticker', 'unknown')
        try:
            success = task.send_func(task.alert, task.message)
            with self._stats_lock:
                if success:
                    self.stats['sent'] += 1
                else:
                    self.stats['failed'] += 1
            if success:
                logger.debug(f"Sent queued notification for {ticker}")
            else:
                logger.warning(f"Failed to send queued notification for {ticker}")
        except Exception as e:
            with self._stats_lock:
                self.stats['failed'] += 1
            logger.error(f"Error sending notification for {ticker}: {e}")

    def _drain_batch(self) -> List[NotificationTask]:
        """
        Drain up to ``_MAX_BATCH_SIZE`` items from the queue.

        The first item is retrieved with a 1-second timeout (so the worker
        can periodically check the stop event).  After the first item
        arrives we *linger* for up to ``batch_linger_seconds``, polling in
        short intervals, to give parallel alert-checker threads time to
        queue their notifications before the batch is dispatched.
        """
        batch: List[NotificationTask] = []
        try:
            _priority, task = self._queue.get(timeout=1.0)
            batch.append(task)
        except queue.Empty:
            return batch

        # Linger: keep collecting items until the batch is full or the
        # linger window expires, giving parallel workers time to enqueue.
        deadline = time.monotonic() + self.batch_linger_seconds
        while len(batch) < self._MAX_BATCH_SIZE:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                _priority, task = self._queue.get(timeout=min(remaining, 0.5))
                batch.append(task)
            except queue.Empty:
                # Poll interval expired but deadline not yet reached; keep waiting
                if time.monotonic() >= deadline:
                    break
                continue

        return batch

    def _process_batch(self, batch: List[NotificationTask]) -> None:
        """
        Process a batch of tasks: group embed-ready tasks by webhook URL
        and send them as batched embeds; fall back to legacy for the rest.
        """
        by_webhook: Dict[str, List[NotificationTask]] = defaultdict(list)
        legacy: List[NotificationTask] = []

        for task in batch:
            if task.webhook_url and task.embed:
                by_webhook[task.webhook_url].append(task)
            else:
                legacy.append(task)

        # Send batched embeds (1 POST per webhook URL)
        for url, tasks in by_webhook.items():
            self._send_embed_batch(url, tasks)

        # Fallback: send legacy tasks one at a time
        for task in legacy:
            self._send_legacy_task(task)
            if self.batch_delay > 0:
                time.sleep(self.batch_delay)

        # Mark all items as done
        for _ in batch:
            self._queue.task_done()

    # -- worker loop --------------------------------------------------------

    def _worker_loop(self) -> None:
        """Background worker that processes the queue with batching."""
        logger.info(
            "Discord notification worker started (batch size up to %d, linger %.1fs)",
            self._MAX_BATCH_SIZE,
            self.batch_linger_seconds,
        )

        while not self._stop_event.is_set():
            try:
                batch = self._drain_batch()
                if not batch:
                    continue
                self._process_batch(batch)

                # Small delay between batches
                if self.batch_delay > 0:
                    time.sleep(self.batch_delay)

            except Exception as e:
                logger.error(f"Error in notification worker: {e}")
                time.sleep(1.0)  # Avoid tight loop on errors

        # Drain remaining items when stopping
        remaining = self._queue.qsize()
        if remaining > 0:
            logger.info(f"Processing {remaining} remaining notifications before shutdown...")
            while not self._queue.empty():
                try:
                    _priority, task = self._queue.get_nowait()
                    if task.webhook_url and task.embed:
                        # Send individually during shutdown to keep it simple
                        self._send_embed_batch(task.webhook_url, [task])
                    else:
                        self._send_legacy_task(task)
                    self._queue.task_done()
                except queue.Empty:
                    break

        logger.info("Discord notification worker stopped")

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        with self._stats_lock:
            return {
                **self.stats,
                'pending': self._queue.qsize(),
                'worker_alive': self._worker_thread.is_alive() if self._worker_thread else False,
            }

    def wait_for_empty(self, timeout: float = 300.0) -> bool:
        """
        Wait for the queue to be fully drained *and* all in-flight items
        to finish processing (i.e. every ``get()`` has a matching
        ``task_done()``).

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if queue emptied and all items processed, False if timeout
        """
        # queue.join() blocks until all task_done() calls are made, but
        # does not accept a timeout.  Run it in a helper thread so we
        # can enforce a deadline.
        done_event = threading.Event()

        def _join():
            self._queue.join()
            done_event.set()

        joiner = threading.Thread(target=_join, daemon=True)
        joiner.start()
        return done_event.wait(timeout=timeout)

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        with self._stats_lock:
            self.stats = {
                'queued': 0,
                'sent': 0,
                'failed': 0,
                'dropped': 0,
            }


# Global queue instance
_discord_queue: Optional[AsyncDiscordQueue] = None


def get_discord_queue() -> AsyncDiscordQueue:
    """Get the global async Discord queue instance."""
    global _discord_queue
    if _discord_queue is None:
        _discord_queue = AsyncDiscordQueue()
    return _discord_queue


def queue_discord_notification(
    alert: Dict[str, Any],
    message: str,
    send_func: Callable[[Dict[str, Any], str], bool],
    priority: int = 0,
    webhook_url: Optional[str] = None,
    embed: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Queue a Discord notification for async sending.

    When *webhook_url* and *embed* are provided the background worker can
    batch multiple notifications into a single webhook POST (up to 10
    embeds per message), dramatically reducing the number of API calls.

    Args:
        alert: Alert dictionary
        message: Formatted message (used as fallback if embed is None)
        send_func: Function to send the notification (fallback path)
        priority: Lower = higher priority
        webhook_url: Pre-resolved Discord webhook URL (enables batching)
        embed: Discord embed dict (enables batching)

    Returns:
        True if queued successfully
    """
    return get_discord_queue().enqueue(
        alert, message, send_func, priority,
        webhook_url=webhook_url, embed=embed,
    )

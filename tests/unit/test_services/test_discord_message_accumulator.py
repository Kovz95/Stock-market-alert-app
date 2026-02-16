"""
Unit tests for src/utils/discord_message_accumulator.py

Covers:
- add(): single embed, multiple embeds, multiple webhook URLs
- add(): auto-flush when bucket reaches 10 embeds
- add(): auto-flush extracts exactly 10 and keeps remainder
- flush_all(): sends all remaining partial batches
- flush_all(): no-op when accumulator is empty
- flush_all(): chunks batches > 10 correctly
- _send_batch(): delegates to send_batch_embeds with rate_limiter
- _send_batch(): handles send_batch_embeds returning False
- _send_batch(): handles send_batch_embeds raising exception
- get_stats(): returns correct counts after various operations
- Thread safety: concurrent adds from multiple threads
"""

from __future__ import annotations

import threading
import pytest
from unittest.mock import MagicMock, patch, call

from src.utils.discord_message_accumulator import (
    DiscordMessageAccumulator,
    _MAX_EMBEDS_PER_MESSAGE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embed(ticker: str = "AAPL", idx: int = 0) -> dict:
    """Create a minimal Discord embed dict for testing."""
    return {
        "title": f"{ticker} — Alert #{idx}",
        "description": "Test condition",
        "color": 0x00FF00,
    }


WEBHOOK_A = "https://discord.com/api/webhooks/111/aaa"
WEBHOOK_B = "https://discord.com/api/webhooks/222/bbb"


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestInit:
    """Tests for DiscordMessageAccumulator.__init__."""

    def test_initial_stats_are_zero(self):
        """All stats should start at zero."""
        acc = DiscordMessageAccumulator()
        stats = acc.get_stats()

        assert stats == {"added": 0, "sent": 0, "failed": 0, "flushes": 0}

    def test_rate_limiter_stored(self):
        """Rate limiter should be stored and passed through to send."""
        rl = MagicMock()
        acc = DiscordMessageAccumulator(rate_limiter=rl)

        assert acc._rate_limiter is rl

    def test_rate_limiter_defaults_to_none(self):
        """Rate limiter should default to None when not provided."""
        acc = DiscordMessageAccumulator()

        assert acc._rate_limiter is None


# ---------------------------------------------------------------------------
# add() – basic accumulation
# ---------------------------------------------------------------------------

class TestAddBasic:
    """Tests for add() without triggering auto-flush."""

    @patch("src.services.discord_routing.send_batch_embeds")
    def test_single_add_does_not_send(self, mock_send):
        """Adding one embed should not trigger a send."""
        acc = DiscordMessageAccumulator()

        acc.add(WEBHOOK_A, _make_embed())

        mock_send.assert_not_called()
        assert acc.get_stats()["added"] == 1

    @patch("src.services.discord_routing.send_batch_embeds")
    def test_multiple_adds_below_threshold_do_not_send(self, mock_send):
        """Adding fewer than 10 embeds should not trigger a send."""
        acc = DiscordMessageAccumulator()

        for i in range(9):
            acc.add(WEBHOOK_A, _make_embed(idx=i))

        mock_send.assert_not_called()
        assert acc.get_stats()["added"] == 9

    @patch("src.services.discord_routing.send_batch_embeds")
    def test_adds_to_separate_buckets(self, mock_send):
        """Embeds for different webhook URLs should be in separate buckets."""
        acc = DiscordMessageAccumulator()

        acc.add(WEBHOOK_A, _make_embed(ticker="AAPL"))
        acc.add(WEBHOOK_B, _make_embed(ticker="MSFT"))

        mock_send.assert_not_called()
        assert acc.get_stats()["added"] == 2

        # Both buckets exist internally
        assert WEBHOOK_A in acc._buckets
        assert WEBHOOK_B in acc._buckets
        assert len(acc._buckets[WEBHOOK_A]) == 1
        assert len(acc._buckets[WEBHOOK_B]) == 1


# ---------------------------------------------------------------------------
# add() – auto-flush at threshold
# ---------------------------------------------------------------------------

class TestAddAutoFlush:
    """Tests for auto-flush when a bucket reaches 10 embeds."""

    @patch("src.services.discord_routing.send_batch_embeds", return_value=True)
    def test_auto_flush_at_10(self, mock_send):
        """Adding the 10th embed should trigger an automatic send."""
        acc = DiscordMessageAccumulator()

        for i in range(10):
            acc.add(WEBHOOK_A, _make_embed(idx=i))

        mock_send.assert_called_once()
        args = mock_send.call_args
        assert args[0][0] == WEBHOOK_A
        assert len(args[0][1]) == 10

    @patch("src.services.discord_routing.send_batch_embeds", return_value=True)
    def test_auto_flush_clears_bucket(self, mock_send):
        """After auto-flush, the bucket should be empty."""
        acc = DiscordMessageAccumulator()

        for i in range(10):
            acc.add(WEBHOOK_A, _make_embed(idx=i))

        # Bucket should be empty after auto-flush
        assert len(acc._buckets.get(WEBHOOK_A, [])) == 0

    @patch("src.services.discord_routing.send_batch_embeds", return_value=True)
    def test_auto_flush_keeps_remainder(self, mock_send):
        """Adding 12 embeds should flush 10 and keep 2."""
        acc = DiscordMessageAccumulator()

        for i in range(12):
            acc.add(WEBHOOK_A, _make_embed(idx=i))

        mock_send.assert_called_once()
        assert len(acc._buckets[WEBHOOK_A]) == 2
        assert acc.get_stats()["added"] == 12

    @patch("src.services.discord_routing.send_batch_embeds", return_value=True)
    def test_auto_flush_triggers_twice_for_20(self, mock_send):
        """Adding 20 embeds should trigger two auto-flushes."""
        acc = DiscordMessageAccumulator()

        for i in range(20):
            acc.add(WEBHOOK_A, _make_embed(idx=i))

        assert mock_send.call_count == 2
        assert len(acc._buckets.get(WEBHOOK_A, [])) == 0

    @patch("src.services.discord_routing.send_batch_embeds", return_value=True)
    def test_auto_flush_only_affects_full_bucket(self, mock_send):
        """Auto-flush for one URL should not affect other URLs."""
        acc = DiscordMessageAccumulator()

        # Add 10 to WEBHOOK_A (triggers flush)
        for i in range(10):
            acc.add(WEBHOOK_A, _make_embed(ticker="AAPL", idx=i))

        # Add 3 to WEBHOOK_B (no flush)
        for i in range(3):
            acc.add(WEBHOOK_B, _make_embed(ticker="MSFT", idx=i))

        # Only one send call (for WEBHOOK_A)
        assert mock_send.call_count == 1
        assert mock_send.call_args[0][0] == WEBHOOK_A
        assert len(acc._buckets[WEBHOOK_B]) == 3

    @patch("src.services.discord_routing.send_batch_embeds", return_value=True)
    def test_auto_flush_stats_updated(self, mock_send):
        """Stats should reflect auto-flushed embeds."""
        acc = DiscordMessageAccumulator()

        for i in range(10):
            acc.add(WEBHOOK_A, _make_embed(idx=i))

        stats = acc.get_stats()
        assert stats["added"] == 10
        assert stats["sent"] == 10
        assert stats["flushes"] == 1
        assert stats["failed"] == 0


# ---------------------------------------------------------------------------
# flush_all()
# ---------------------------------------------------------------------------

class TestFlushAll:
    """Tests for flush_all()."""

    @patch("src.services.discord_routing.send_batch_embeds", return_value=True)
    def test_flush_all_sends_partial_batch(self, mock_send):
        """flush_all should send remaining embeds that didn't reach threshold."""
        acc = DiscordMessageAccumulator()

        for i in range(5):
            acc.add(WEBHOOK_A, _make_embed(idx=i))

        acc.flush_all()

        mock_send.assert_called_once()
        assert len(mock_send.call_args[0][1]) == 5

    @patch("src.services.discord_routing.send_batch_embeds", return_value=True)
    def test_flush_all_sends_multiple_urls(self, mock_send):
        """flush_all should send batches for each webhook URL."""
        acc = DiscordMessageAccumulator()

        acc.add(WEBHOOK_A, _make_embed(ticker="AAPL"))
        acc.add(WEBHOOK_B, _make_embed(ticker="MSFT"))

        acc.flush_all()

        assert mock_send.call_count == 2
        urls_sent = {c[0][0] for c in mock_send.call_args_list}
        assert urls_sent == {WEBHOOK_A, WEBHOOK_B}

    @patch("src.services.discord_routing.send_batch_embeds", return_value=True)
    def test_flush_all_chunks_large_remainder(self, mock_send):
        """flush_all should chunk embeds > 10 into multiple sends."""
        acc = DiscordMessageAccumulator()

        # Manually stuff 15 embeds into a bucket (bypassing auto-flush for test)
        acc._buckets[WEBHOOK_A] = [_make_embed(idx=i) for i in range(15)]
        acc._added = 15

        acc.flush_all()

        assert mock_send.call_count == 2
        first_batch = mock_send.call_args_list[0][0][1]
        second_batch = mock_send.call_args_list[1][0][1]
        assert len(first_batch) == 10
        assert len(second_batch) == 5

    @patch("src.services.discord_routing.send_batch_embeds", return_value=True)
    def test_flush_all_clears_buckets(self, mock_send):
        """flush_all should empty all buckets."""
        acc = DiscordMessageAccumulator()

        acc.add(WEBHOOK_A, _make_embed())
        acc.add(WEBHOOK_B, _make_embed())

        acc.flush_all()

        assert len(acc._buckets) == 0

    @patch("src.services.discord_routing.send_batch_embeds")
    def test_flush_all_on_empty_accumulator_is_noop(self, mock_send):
        """flush_all on empty accumulator should not send anything."""
        acc = DiscordMessageAccumulator()

        acc.flush_all()

        mock_send.assert_not_called()
        assert acc.get_stats()["flushes"] == 0

    @patch("src.services.discord_routing.send_batch_embeds", return_value=True)
    def test_flush_all_idempotent(self, mock_send):
        """Calling flush_all twice should only send once."""
        acc = DiscordMessageAccumulator()

        acc.add(WEBHOOK_A, _make_embed())
        acc.flush_all()
        acc.flush_all()

        mock_send.assert_called_once()

    @patch("src.services.discord_routing.send_batch_embeds", return_value=True)
    def test_flush_all_after_auto_flush_sends_remainder(self, mock_send):
        """flush_all should only send embeds not already auto-flushed."""
        acc = DiscordMessageAccumulator()

        # Add 13 embeds: 10 auto-flushed + 3 remaining
        for i in range(13):
            acc.add(WEBHOOK_A, _make_embed(idx=i))

        mock_send.assert_called_once()  # auto-flush
        mock_send.reset_mock()

        acc.flush_all()

        mock_send.assert_called_once()
        assert len(mock_send.call_args[0][1]) == 3


# ---------------------------------------------------------------------------
# _send_batch() – delegation and error handling
# ---------------------------------------------------------------------------

class TestSendBatch:
    """Tests for _send_batch() behavior."""

    @patch("src.services.discord_routing.send_batch_embeds", return_value=True)
    def test_passes_rate_limiter(self, mock_send):
        """_send_batch should pass rate_limiter to send_batch_embeds."""
        rl = MagicMock()
        acc = DiscordMessageAccumulator(rate_limiter=rl)

        embeds = [_make_embed()]
        acc._send_batch(WEBHOOK_A, embeds)

        mock_send.assert_called_once_with(WEBHOOK_A, embeds, rl)

    @patch("src.services.discord_routing.send_batch_embeds", return_value=True)
    def test_passes_none_rate_limiter_when_unset(self, mock_send):
        """_send_batch should pass None rate_limiter when not configured."""
        acc = DiscordMessageAccumulator()

        embeds = [_make_embed()]
        acc._send_batch(WEBHOOK_A, embeds)

        mock_send.assert_called_once_with(WEBHOOK_A, embeds, None)

    @patch("src.services.discord_routing.send_batch_embeds", return_value=False)
    def test_failure_increments_failed_stat(self, mock_send):
        """When send_batch_embeds returns False, failed count should increase."""
        acc = DiscordMessageAccumulator()

        embeds = [_make_embed(idx=i) for i in range(3)]
        acc._send_batch(WEBHOOK_A, embeds)

        stats = acc.get_stats()
        assert stats["failed"] == 3
        assert stats["sent"] == 0
        assert stats["flushes"] == 1

    @patch("src.services.discord_routing.send_batch_embeds", return_value=True)
    def test_success_increments_sent_stat(self, mock_send):
        """When send_batch_embeds returns True, sent count should increase."""
        acc = DiscordMessageAccumulator()

        embeds = [_make_embed(idx=i) for i in range(5)]
        acc._send_batch(WEBHOOK_A, embeds)

        stats = acc.get_stats()
        assert stats["sent"] == 5
        assert stats["failed"] == 0
        assert stats["flushes"] == 1

    @patch(
        "src.services.discord_routing.send_batch_embeds",
        side_effect=Exception("Network error"),
    )
    def test_exception_increments_failed_stat(self, mock_send):
        """When send_batch_embeds raises, failed count should increase."""
        acc = DiscordMessageAccumulator()

        embeds = [_make_embed(idx=i) for i in range(2)]
        acc._send_batch(WEBHOOK_A, embeds)

        stats = acc.get_stats()
        assert stats["failed"] == 2
        assert stats["sent"] == 0
        assert stats["flushes"] == 1

    @patch(
        "src.services.discord_routing.send_batch_embeds",
        side_effect=Exception("boom"),
    )
    def test_exception_does_not_propagate(self, mock_send):
        """Exceptions in _send_batch should be caught, not raised."""
        acc = DiscordMessageAccumulator()

        # Should not raise
        acc._send_batch(WEBHOOK_A, [_make_embed()])


# ---------------------------------------------------------------------------
# get_stats()
# ---------------------------------------------------------------------------

class TestGetStats:
    """Tests for get_stats()."""

    def test_stats_keys(self):
        """get_stats should return dict with expected keys."""
        acc = DiscordMessageAccumulator()
        stats = acc.get_stats()

        assert set(stats.keys()) == {"added", "sent", "failed", "flushes"}

    @patch("src.services.discord_routing.send_batch_embeds", return_value=True)
    def test_stats_after_full_lifecycle(self, mock_send):
        """Stats should reflect adds, auto-flushes, and final flush."""
        acc = DiscordMessageAccumulator()

        # Add 13 embeds to URL A (auto-flush at 10, 3 remain)
        for i in range(13):
            acc.add(WEBHOOK_A, _make_embed(idx=i))

        # Add 2 embeds to URL B
        acc.add(WEBHOOK_B, _make_embed(idx=0))
        acc.add(WEBHOOK_B, _make_embed(idx=1))

        # Flush remaining
        acc.flush_all()

        stats = acc.get_stats()
        assert stats["added"] == 15
        assert stats["sent"] == 15  # 10 + 3 + 2
        assert stats["failed"] == 0
        assert stats["flushes"] == 3  # auto-flush(10) + flush(3) + flush(2)

    @patch("src.services.discord_routing.send_batch_embeds", return_value=False)
    def test_stats_with_all_failures(self, mock_send):
        """Stats should correctly count failures."""
        acc = DiscordMessageAccumulator()

        for i in range(10):
            acc.add(WEBHOOK_A, _make_embed(idx=i))

        acc.flush_all()  # bucket was cleared by auto-flush, nothing left

        stats = acc.get_stats()
        assert stats["added"] == 10
        assert stats["sent"] == 0
        assert stats["failed"] == 10
        assert stats["flushes"] == 1

    @patch("src.services.discord_routing.send_batch_embeds")
    def test_stats_mixed_success_and_failure(self, mock_send):
        """Stats should track mixed outcomes across separate batches."""
        # First call succeeds, second fails
        mock_send.side_effect = [True, False]

        acc = DiscordMessageAccumulator()

        # Add 5 to each URL
        for i in range(5):
            acc.add(WEBHOOK_A, _make_embed(idx=i))
        for i in range(5):
            acc.add(WEBHOOK_B, _make_embed(idx=i))

        acc.flush_all()

        stats = acc.get_stats()
        assert stats["added"] == 10
        assert stats["sent"] + stats["failed"] == 10
        assert stats["flushes"] == 2


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:
    """Tests for concurrent access from multiple threads."""

    @patch("src.services.discord_routing.send_batch_embeds", return_value=True)
    def test_concurrent_adds_no_lost_embeds(self, mock_send):
        """Concurrent adds should not lose any embeds."""
        acc = DiscordMessageAccumulator()
        num_threads = 10
        adds_per_thread = 8  # Total 80, should trigger 8 auto-flushes
        barrier = threading.Barrier(num_threads)

        def worker():
            barrier.wait()  # Synchronize start
            for i in range(adds_per_thread):
                acc.add(WEBHOOK_A, _make_embed(idx=i))

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Flush any remaining
        acc.flush_all()

        stats = acc.get_stats()
        total = num_threads * adds_per_thread
        assert stats["added"] == total
        assert stats["sent"] + stats["failed"] == total

    @patch("src.services.discord_routing.send_batch_embeds", return_value=True)
    def test_concurrent_adds_to_different_urls(self, mock_send):
        """Concurrent adds to different URLs should not interfere."""
        acc = DiscordMessageAccumulator()
        num_per_url = 5
        barrier = threading.Barrier(2)

        def worker_a():
            barrier.wait()
            for i in range(num_per_url):
                acc.add(WEBHOOK_A, _make_embed(ticker="AAPL", idx=i))

        def worker_b():
            barrier.wait()
            for i in range(num_per_url):
                acc.add(WEBHOOK_B, _make_embed(ticker="MSFT", idx=i))

        t1 = threading.Thread(target=worker_a)
        t2 = threading.Thread(target=worker_b)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        acc.flush_all()

        stats = acc.get_stats()
        assert stats["added"] == num_per_url * 2
        assert stats["sent"] == num_per_url * 2

    @patch("src.services.discord_routing.send_batch_embeds", return_value=True)
    def test_concurrent_add_and_flush(self, mock_send):
        """flush_all should be safe to call while adds are in progress."""
        acc = DiscordMessageAccumulator()
        done = threading.Event()

        def adder():
            for i in range(50):
                acc.add(WEBHOOK_A, _make_embed(idx=i))
            done.set()

        t = threading.Thread(target=adder)
        t.start()

        # Flush while adder is running
        done.wait(timeout=5)
        acc.flush_all()
        t.join()

        stats = acc.get_stats()
        assert stats["added"] == 50
        # All should be sent (some via auto-flush, rest via flush_all)
        assert stats["sent"] + stats["failed"] == 50


# ---------------------------------------------------------------------------
# MAX_EMBEDS_PER_MESSAGE constant
# ---------------------------------------------------------------------------

class TestConstants:
    """Tests for module-level constants."""

    def test_max_embeds_per_message_is_10(self):
        """Discord limit is 10 embeds per message."""
        assert _MAX_EMBEDS_PER_MESSAGE == 10


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @patch("src.services.discord_routing.send_batch_embeds", return_value=True)
    def test_exactly_10_embeds_sends_one_batch(self, mock_send):
        """Exactly 10 embeds should send one batch with nothing remaining."""
        acc = DiscordMessageAccumulator()

        for i in range(10):
            acc.add(WEBHOOK_A, _make_embed(idx=i))

        assert mock_send.call_count == 1
        acc.flush_all()
        # flush_all should not send anything extra
        assert mock_send.call_count == 1

    @patch("src.services.discord_routing.send_batch_embeds", return_value=True)
    def test_single_embed_flush(self, mock_send):
        """A single embed should be sent as a batch of 1 on flush."""
        acc = DiscordMessageAccumulator()

        acc.add(WEBHOOK_A, _make_embed())
        acc.flush_all()

        mock_send.assert_called_once()
        assert len(mock_send.call_args[0][1]) == 1

    @patch("src.services.discord_routing.send_batch_embeds", return_value=True)
    def test_add_after_flush_all_works(self, mock_send):
        """Adding embeds after flush_all should work normally."""
        acc = DiscordMessageAccumulator()

        acc.add(WEBHOOK_A, _make_embed(idx=0))
        acc.flush_all()

        mock_send.reset_mock()

        acc.add(WEBHOOK_A, _make_embed(idx=1))
        acc.flush_all()

        mock_send.assert_called_once()
        assert len(mock_send.call_args[0][1]) == 1
        assert acc.get_stats()["added"] == 2

    @patch("src.services.discord_routing.send_batch_embeds", return_value=True)
    def test_many_different_urls(self, mock_send):
        """Accumulator should handle many different webhook URLs."""
        acc = DiscordMessageAccumulator()

        urls = [f"https://discord.com/api/webhooks/{i}/token{i}" for i in range(20)]
        for url in urls:
            acc.add(url, _make_embed())

        acc.flush_all()

        assert mock_send.call_count == 20
        assert acc.get_stats()["added"] == 20
        assert acc.get_stats()["sent"] == 20

    @patch(
        "src.services.discord_routing.send_batch_embeds",
        side_effect=Exception("fail"),
    )
    def test_failed_auto_flush_does_not_block_further_adds(self, mock_send):
        """If auto-flush fails, subsequent adds should still work."""
        acc = DiscordMessageAccumulator()

        # Add 10 (triggers auto-flush which fails)
        for i in range(10):
            acc.add(WEBHOOK_A, _make_embed(idx=i))

        # Should still be able to add more
        acc.add(WEBHOOK_A, _make_embed(idx=10))

        assert acc.get_stats()["added"] == 11
        assert acc.get_stats()["failed"] == 10
        assert len(acc._buckets[WEBHOOK_A]) == 1

    @patch("src.services.discord_routing.send_batch_embeds", return_value=True)
    def test_embed_ordering_preserved_in_batch(self, mock_send):
        """Embeds within a batch should maintain insertion order."""
        acc = DiscordMessageAccumulator()

        embeds = [_make_embed(idx=i) for i in range(5)]
        for e in embeds:
            acc.add(WEBHOOK_A, e)

        acc.flush_all()

        sent_embeds = mock_send.call_args[0][1]
        for i, embed in enumerate(sent_embeds):
            assert embed["title"] == f"AAPL — Alert #{i}"

    @patch("src.services.discord_routing.send_batch_embeds", return_value=True)
    def test_auto_flush_preserves_order_of_first_10(self, mock_send):
        """Auto-flushed batch of 10 should preserve insertion order."""
        acc = DiscordMessageAccumulator()

        for i in range(10):
            acc.add(WEBHOOK_A, _make_embed(idx=i))

        sent_embeds = mock_send.call_args[0][1]
        for i, embed in enumerate(sent_embeds):
            assert embed["title"] == f"AAPL — Alert #{i}"

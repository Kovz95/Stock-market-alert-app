"""
Unit tests for the Discord Logger module.

Tests the asynchronous Discord webhook logging functionality including:
- Message splitting
- Webhook sending (with mocking)
- Async queuing and batching
- Thread management
- Error handling
"""

import os
import queue
import threading
import time
from unittest.mock import Mock, patch, MagicMock, call
import pytest

from src.services import discord_logger
from src.services.discord_logger import (
    split_message,
    log_to_discord_async,
    flush_discord_logs,
    shutdown_discord_logger,
    _send_to_discord,
    _ensure_worker_thread,
    MAX_DISCORD_MESSAGE_LENGTH,
    BATCH_DELAY,
    MAX_BATCH_SIZE,
)


class TestSplitMessage:
    """Tests for the split_message function."""

    def test_short_message_no_splitting(self):
        """Test that short messages are not split."""
        message = "This is a short message"
        result = split_message(message)

        assert len(result) == 1
        assert result[0] == f"```{message}```"

    def test_empty_message(self):
        """Test that empty messages return code block with empty content."""
        result = split_message("")
        assert len(result) == 1
        # Empty message still gets wrapped in code blocks
        assert result[0] == "``````"

    def test_long_message_splitting(self):
        """Test that long multi-line messages are split into multiple chunks."""
        # Create multiple lines where combined they exceed MAX_DISCORD_MESSAGE_LENGTH
        # Each line is short enough to fit, but together they're too long
        lines = ["Line " + "x" * 100 for _ in range(30)]
        message = "\n".join(lines)

        result = split_message(message)

        # Should be split into multiple chunks
        assert len(result) > 1

    def test_multiline_message_preserves_lines(self):
        """Test that multiline messages are split at line boundaries."""
        lines = ["Line 1", "Line 2", "Line 3"]
        message = "\n".join(lines)

        result = split_message(message)

        # Should be in a single chunk since it's short
        assert len(result) == 1
        for line in lines:
            assert line in result[0]

    def test_custom_max_length(self):
        """Test splitting with custom max_length parameter."""
        # Create multiple short lines that together exceed custom max
        lines = ["x" * 20 for _ in range(10)]
        message = "\n".join(lines)
        custom_max = 100

        result = split_message(message, max_length=custom_max)

        # Should split into multiple chunks
        assert len(result) > 1

    def test_code_block_formatting(self):
        """Test that chunks are wrapped in code blocks."""
        message = "Test message"
        result = split_message(message)

        assert result[0].startswith("```")
        assert result[0].endswith("```")

    def test_very_long_single_line(self):
        """Test handling of a very long single line.

        Note: The split_message function splits by newlines, not by length.
        A single line longer than max_length will still be kept in one chunk.
        This is a known limitation of the current implementation.
        """
        long_line = "a" * 3000
        result = split_message(long_line)

        # Single line is kept intact even if it exceeds max_length
        # This is the actual behavior of the function
        assert len(result) == 1
        # The chunk will be wrapped in code blocks
        assert result[0].startswith("```")
        assert result[0].endswith("```")


class TestSendToDiscord:
    """Tests for the _send_to_discord function."""

    def setup_method(self):
        """Reset module state before each test."""
        # Store original values
        self.original_webhook = discord_logger.WEBHOOK_URL_LOGGING
        self.original_webhook_2 = discord_logger.WEBHOOK_URL_LOGGING_2

    def teardown_method(self):
        """Restore original values after each test."""
        discord_logger.WEBHOOK_URL_LOGGING = self.original_webhook
        discord_logger.WEBHOOK_URL_LOGGING_2 = self.original_webhook_2

    @patch('src.services.discord_logger.requests.post')
    def test_successful_send(self, mock_post):
        """Test successful message sending to Discord."""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        discord_logger.WEBHOOK_URL_LOGGING = "https://discord.com/api/webhooks/test"

        result = _send_to_discord("Test message")

        assert result is True
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[1]['json']['content'] == "```Test message```"

    @patch('src.services.discord_logger.requests.post')
    def test_failed_send_non_204_status(self, mock_post):
        """Test handling of non-204 status code."""
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_post.return_value = mock_response

        discord_logger.WEBHOOK_URL_LOGGING = "https://discord.com/api/webhooks/test"

        result = _send_to_discord("Test message")

        # Should return False since status is not 204
        assert result is False

    def test_no_webhook_configured(self):
        """Test handling when no webhook is configured."""
        discord_logger.WEBHOOK_URL_LOGGING = ""

        result = _send_to_discord("Test message")

        assert result is False

    def test_placeholder_webhook_url(self):
        """Test handling of placeholder webhook URL."""
        discord_logger.WEBHOOK_URL_LOGGING = "YOUR_WEBHOOK_URL_HERE"

        result = _send_to_discord("Test message")

        assert result is False

    @patch('src.services.discord_logger.requests.post')
    def test_fallback_to_secondary_webhook(self, mock_post):
        """Test fallback to secondary webhook when primary fails."""
        # Primary fails, secondary succeeds
        primary_response = Mock()
        primary_response.status_code = 500

        secondary_response = Mock()
        secondary_response.status_code = 204

        # First call fails, second succeeds
        mock_post.side_effect = [Exception("Primary failed"), secondary_response]

        discord_logger.WEBHOOK_URL_LOGGING = "https://discord.com/primary"
        discord_logger.WEBHOOK_URL_LOGGING_2 = "https://discord.com/secondary"

        result = _send_to_discord("Test message")

        assert result is True
        assert mock_post.call_count == 2

    @patch('src.services.discord_logger.requests.post')
    def test_both_webhooks_fail(self, mock_post):
        """Test handling when both primary and secondary webhooks fail."""
        mock_post.side_effect = [Exception("Primary failed"), Exception("Secondary failed")]

        discord_logger.WEBHOOK_URL_LOGGING = "https://discord.com/primary"
        discord_logger.WEBHOOK_URL_LOGGING_2 = "https://discord.com/secondary"

        result = _send_to_discord("Test message")

        assert result is False
        assert mock_post.call_count == 2

    @patch('src.services.discord_logger.requests.post')
    @patch('src.services.discord_logger.time.sleep')
    def test_multiple_message_chunks_with_delay(self, mock_sleep, mock_post):
        """Test that multiple message chunks are sent with delays."""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        discord_logger.WEBHOOK_URL_LOGGING = "https://discord.com/api/webhooks/test"

        # Create a long message that will be split
        long_message = "\n".join(["x" * 500 for _ in range(10)])

        result = _send_to_discord(long_message)

        assert result is True
        # Should have multiple calls to post
        assert mock_post.call_count > 1
        # Should have called sleep between messages
        assert mock_sleep.call_count >= mock_post.call_count - 1

    @patch('src.services.discord_logger.requests.post')
    def test_timeout_parameter(self, mock_post):
        """Test that requests are made with timeout."""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        discord_logger.WEBHOOK_URL_LOGGING = "https://discord.com/api/webhooks/test"

        _send_to_discord("Test message")

        # Verify timeout is set
        call_args = mock_post.call_args
        assert call_args[1]['timeout'] == 10


class TestLogToDiscordAsync:
    """Tests for the log_to_discord_async function."""

    def setup_method(self):
        """Setup before each test."""
        # Clear the message queue
        while not discord_logger._message_queue.empty():
            try:
                discord_logger._message_queue.get_nowait()
                discord_logger._message_queue.task_done()
            except queue.Empty:
                break

        # Shutdown any existing worker thread
        shutdown_discord_logger()

    def teardown_method(self):
        """Cleanup after each test."""
        shutdown_discord_logger()

    @patch('src.services.discord_logger._ensure_worker_thread')
    def test_queues_message(self, mock_ensure_thread):
        """Test that messages are queued."""
        log_to_discord_async("Test message")

        mock_ensure_thread.assert_called_once()
        # Message should be in queue
        assert not discord_logger._message_queue.empty()

    def test_ignores_empty_message(self):
        """Test that empty messages are not queued."""
        log_to_discord_async("")

        # Queue should remain empty
        assert discord_logger._message_queue.empty()

    @patch('src.services.discord_logger._send_to_discord')
    def test_worker_thread_processes_messages(self, mock_send):
        """Test that worker thread processes queued messages."""
        mock_send.return_value = True

        # Queue a message
        log_to_discord_async("Test message")

        # Wait for processing (with timeout)
        time.sleep(BATCH_DELAY + 0.5)

        # Flush to ensure processing
        flush_discord_logs()

        # Message should have been sent
        mock_send.assert_called()


class TestFlushDiscordLogs:
    """Tests for the flush_discord_logs function."""

    def setup_method(self):
        """Setup before each test."""
        shutdown_discord_logger()

    def teardown_method(self):
        """Cleanup after each test."""
        shutdown_discord_logger()

    @patch('src.services.discord_logger._ensure_worker_thread')
    def test_ensures_worker_thread(self, mock_ensure_thread):
        """Test that flush ensures worker thread is running."""
        # Create a mock queue join that returns immediately
        with patch.object(discord_logger._message_queue, 'join'):
            flush_discord_logs()

        mock_ensure_thread.assert_called_once()

    @patch('src.services.discord_logger._send_to_discord')
    def test_waits_for_queue_empty(self, mock_send):
        """Test that flush waits for all messages to be processed."""
        mock_send.return_value = True

        # Queue multiple messages
        for i in range(3):
            log_to_discord_async(f"Message {i}")

        # Flush should wait for all messages
        flush_discord_logs()

        # Queue should be empty
        assert discord_logger._message_queue.empty()


class TestShutdownDiscordLogger:
    """Tests for the shutdown_discord_logger function."""

    def setup_method(self):
        """Setup before each test."""
        # Ensure clean state
        shutdown_discord_logger()

    def test_sets_shutdown_event(self):
        """Test that shutdown sets the shutdown event."""
        # Start a worker thread
        _ensure_worker_thread()

        # Shutdown
        shutdown_discord_logger()

        # Event should be set
        assert discord_logger._shutdown_event.is_set()

    def test_waits_for_thread_to_finish(self):
        """Test that shutdown waits for worker thread to finish."""
        # Start a worker thread
        _ensure_worker_thread()
        original_thread = discord_logger._worker_thread

        # Shutdown
        shutdown_discord_logger()

        # Thread should no longer be alive
        if original_thread:
            assert not original_thread.is_alive()

    def test_clears_worker_thread_reference(self):
        """Test that shutdown clears the worker thread reference."""
        # Start a worker thread
        _ensure_worker_thread()

        # Shutdown
        shutdown_discord_logger()

        # Reference should be None
        assert discord_logger._worker_thread is None

    @patch('src.services.discord_logger._send_to_discord')
    def test_sends_remaining_messages_before_shutdown(self, mock_send):
        """Test that remaining messages are sent before shutdown."""
        mock_send.return_value = True

        # Queue messages
        log_to_discord_async("Message 1")
        log_to_discord_async("Message 2")

        # Shutdown immediately (before batch delay)
        shutdown_discord_logger()

        # Should have sent the messages
        mock_send.assert_called()


class TestEnsureWorkerThread:
    """Tests for the _ensure_worker_thread function."""

    def setup_method(self):
        """Setup before each test."""
        shutdown_discord_logger()

    def teardown_method(self):
        """Cleanup after each test."""
        shutdown_discord_logger()

    def test_starts_thread_if_not_running(self):
        """Test that _ensure_worker_thread starts thread if not running."""
        assert discord_logger._worker_thread is None

        _ensure_worker_thread()

        assert discord_logger._worker_thread is not None
        assert discord_logger._worker_thread.is_alive()

    def test_does_not_create_duplicate_threads(self):
        """Test that multiple calls don't create duplicate threads."""
        _ensure_worker_thread()
        first_thread = discord_logger._worker_thread

        _ensure_worker_thread()
        second_thread = discord_logger._worker_thread

        # Should be the same thread
        assert first_thread is second_thread

    def test_restarts_dead_thread(self):
        """Test that _ensure_worker_thread restarts a dead thread."""
        # Start and stop thread
        _ensure_worker_thread()
        shutdown_discord_logger()

        # Ensure thread again
        _ensure_worker_thread()

        # Should have a new, alive thread
        assert discord_logger._worker_thread is not None
        assert discord_logger._worker_thread.is_alive()

    def test_clears_shutdown_event(self):
        """Test that _ensure_worker_thread clears shutdown event."""
        # Set shutdown event
        discord_logger._shutdown_event.set()

        _ensure_worker_thread()

        # Event should be cleared
        assert not discord_logger._shutdown_event.is_set()

    def test_thread_is_daemon(self):
        """Test that worker thread is created as daemon."""
        _ensure_worker_thread()

        assert discord_logger._worker_thread is not None
        assert discord_logger._worker_thread.daemon is True

    def test_thread_has_correct_name(self):
        """Test that worker thread has correct name."""
        _ensure_worker_thread()

        assert discord_logger._worker_thread is not None
        assert discord_logger._worker_thread.name == "DiscordLogger"


class TestWorkerLoop:
    """Tests for the _worker_loop function."""

    def setup_method(self):
        """Setup before each test."""
        shutdown_discord_logger()

    def teardown_method(self):
        """Cleanup after each test."""
        shutdown_discord_logger()

    @patch('src.services.discord_logger._send_to_discord')
    def test_batches_messages(self, mock_send):
        """Test that worker batches multiple messages together."""
        mock_send.return_value = True

        # Queue multiple messages quickly
        for i in range(3):
            log_to_discord_async(f"Message {i}")

        # Wait for batch delay
        time.sleep(BATCH_DELAY + 0.2)
        flush_discord_logs()

        # Should have combined messages
        mock_send.assert_called()
        call_args = mock_send.call_args[0][0]
        assert "Message 0" in call_args
        assert "Message 1" in call_args
        assert "Message 2" in call_args

    @patch('src.services.discord_logger._send_to_discord')
    def test_sends_when_batch_size_reached(self, mock_send):
        """Test that messages are sent when batch size is reached."""
        mock_send.return_value = True

        # Queue MAX_BATCH_SIZE messages
        for i in range(MAX_BATCH_SIZE):
            log_to_discord_async(f"Message {i}")

        # Should send immediately without waiting for delay
        flush_discord_logs()

        mock_send.assert_called()

    @patch('src.services.discord_logger._send_to_discord')
    def test_handles_exceptions_gracefully(self, mock_send):
        """Test that worker handles exceptions without crashing."""
        # Make send raise exception
        mock_send.side_effect = Exception("Test error")

        log_to_discord_async("Test message")

        # Wait for processing
        time.sleep(BATCH_DELAY + 0.2)

        # Worker should still be alive despite error
        assert discord_logger._worker_thread is not None
        assert discord_logger._worker_thread.is_alive()

    @patch('src.services.discord_logger._send_to_discord')
    def test_clears_buffer_on_error(self, mock_send):
        """Test that buffer is cleared on error to prevent infinite loop."""
        # First call fails, second should work
        mock_send.side_effect = [Exception("Test error"), True]

        log_to_discord_async("Message 1")
        time.sleep(BATCH_DELAY + 0.2)

        log_to_discord_async("Message 2")
        time.sleep(BATCH_DELAY + 0.2)

        flush_discord_logs()

        # Should have attempted to send twice (once failed, once succeeded)
        assert mock_send.call_count == 2


class TestIntegration:
    """Integration tests for the Discord logger."""

    def setup_method(self):
        """Setup before each test."""
        shutdown_discord_logger()

    def teardown_method(self):
        """Cleanup after each test."""
        shutdown_discord_logger()

    @patch('src.services.discord_logger._send_to_discord')
    def test_full_async_logging_flow(self, mock_send):
        """Test complete async logging flow from queue to send."""
        mock_send.return_value = True

        # Log messages asynchronously
        log_to_discord_async("First message")
        log_to_discord_async("Second message")
        log_to_discord_async("Third message")

        # Flush all messages
        flush_discord_logs()

        # All messages should be sent
        mock_send.assert_called()

        # Verify messages were combined
        call_args = mock_send.call_args[0][0]
        assert "First message" in call_args
        assert "Second message" in call_args
        assert "Third message" in call_args

    @patch('src.services.discord_logger._send_to_discord')
    def test_graceful_shutdown_with_pending_messages(self, mock_send):
        """Test graceful shutdown sends pending messages."""
        mock_send.return_value = True

        # Queue messages
        log_to_discord_async("Pending message 1")
        log_to_discord_async("Pending message 2")

        # Shutdown immediately
        shutdown_discord_logger()

        # Messages should have been sent during shutdown
        mock_send.assert_called()

    @patch('src.services.discord_logger.requests.post')
    def test_rate_limiting_between_chunks(self, mock_post):
        """Test rate limiting delays between message chunks."""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        discord_logger.WEBHOOK_URL_LOGGING = "https://discord.com/api/webhooks/test"

        # Create message that will split into multiple chunks
        long_message = "\n".join(["x" * 400 for _ in range(10)])

        start_time = time.time()
        _send_to_discord(long_message)
        elapsed_time = time.time() - start_time

        # Should have taken time due to rate limiting delays
        # Multiple chunks should have 0.5s delays between them
        if mock_post.call_count > 1:
            assert elapsed_time >= (mock_post.call_count - 1) * 0.5

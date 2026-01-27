"""
Unit tests for Discord client.

Tests the DiscordClient class with mocked webhook requests.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Mock async_discord_logger before importing client
sys.modules["async_discord_logger"] = MagicMock()

from src.stock_alert.services.discord.client import (
    DiscordClient,
    flush_logs_to_discord,
    log_to_discord,
    send_stock_alert,
)


class TestDiscordClient:
    """Test suite for DiscordClient"""

    def test_initialization_with_webhooks(self):
        """Test that client initializes with provided webhooks"""
        client = DiscordClient(
            primary_webhook="https://webhook1.com",
            secondary_webhook="https://webhook2.com",
            log_webhook="https://log1.com",
            log_webhook_2="https://log2.com",
        )

        assert client.primary_webhook == "https://webhook1.com"
        assert client.secondary_webhook == "https://webhook2.com"
        assert client.log_webhook == "https://log1.com"
        assert client.log_webhook_2 == "https://log2.com"
        assert client.log_buffer == []

    @patch("src.stock_alert.services.discord.client.Settings")
    def test_initialization_from_settings(self, mock_settings_class):
        """Test that client loads webhooks from settings"""
        mock_settings = Mock()
        mock_settings.WEBHOOK_URL = "https://settings-webhook.com"
        mock_settings.WEBHOOK_URL_2 = "https://settings-webhook2.com"
        mock_settings.WEBHOOK_URL_LOGGING = "https://settings-log.com"
        mock_settings.WEBHOOK_URL_LOGGING_2 = "https://settings-log2.com"
        mock_settings_class.return_value = mock_settings

        client = DiscordClient()

        assert client.primary_webhook == "https://settings-webhook.com"
        assert client.secondary_webhook == "https://settings-webhook2.com"

    @patch("src.stock_alert.services.discord.client.requests.post")
    def test_send_alert_success(self, mock_post):
        """Test successful alert sending"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        client = DiscordClient(primary_webhook="https://test-webhook.com")
        result = client.send_alert(
            alert_name="Test Alert",
            ticker="AAPL",
            triggered_condition="RSI > 70",
            current_price=150.50,
            action="Sell",
            timeframe="1d",
            exchange="NASDAQ",
        )

        assert result is True
        mock_post.assert_called_once()

        # Verify payload structure
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert "embeds" in payload
        assert len(payload["embeds"]) == 1
        assert payload["embeds"][0]["title"] == "[ALERT] Test Alert (AAPL)"
        assert payload["embeds"][0]["color"] == 0xFF0000  # Red for Sell

    @patch("src.stock_alert.services.discord.client.requests.post")
    def test_send_alert_buy_action_green_color(self, mock_post):
        """Test that Buy action uses green color"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        client = DiscordClient(primary_webhook="https://test-webhook.com")
        client.send_alert(
            alert_name="Buy Alert",
            ticker="TSLA",
            triggered_condition="SMA crossed",
            current_price=200.00,
            action="Buy",
            timeframe="1d",
        )

        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["embeds"][0]["color"] == 0x00FF00  # Green for Buy

    @patch("src.stock_alert.services.discord.client.requests.post")
    def test_send_alert_no_webhook(self, mock_post):
        """Test that alert fails gracefully without webhook"""
        client = DiscordClient(primary_webhook=None)
        result = client.send_alert(
            alert_name="Test Alert",
            ticker="AAPL",
            triggered_condition="RSI > 70",
            current_price=150.50,
            action="Sell",
            timeframe="1d",
        )

        assert result is False
        mock_post.assert_not_called()

    @patch("src.stock_alert.services.discord.client.requests.post")
    def test_send_alert_request_failure(self, mock_post):
        """Test handling of request failures"""
        mock_post.side_effect = Exception("Network error")

        client = DiscordClient(primary_webhook="https://test-webhook.com")
        result = client.send_alert(
            alert_name="Test Alert",
            ticker="AAPL",
            triggered_condition="RSI > 70",
            current_price=150.50,
            action="Sell",
            timeframe="1d",
        )

        assert result is False

    @patch("src.stock_alert.services.discord.client.requests.post")
    def test_send_to_multiple_webhooks(self, mock_post):
        """Test sending to multiple webhooks"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        client = DiscordClient(
            primary_webhook="https://webhook1.com", secondary_webhook="https://webhook2.com"
        )

        success_count = client.send_to_multiple_webhooks(
            alert_name="Test Alert",
            ticker="AAPL",
            triggered_condition="RSI > 70",
            current_price=150.50,
            action="Sell",
            timeframe="1d",
        )

        assert success_count == 2
        assert mock_post.call_count == 2

    @patch("src.stock_alert.services.discord.client.requests.post")
    def test_send_to_multiple_webhooks_skips_duplicates(self, mock_post):
        """Test that duplicate webhooks are not called twice"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        client = DiscordClient(
            primary_webhook="https://webhook.com",
            secondary_webhook="https://webhook.com",  # Same as primary
        )

        success_count = client.send_to_multiple_webhooks(
            alert_name="Test Alert",
            ticker="AAPL",
            triggered_condition="RSI > 70",
            current_price=150.50,
            action="Sell",
            timeframe="1d",
        )

        assert success_count == 1
        assert mock_post.call_count == 1

    def test_log_message(self):
        """Test adding messages to log buffer"""
        client = DiscordClient()

        client.log_message("Test log 1")
        client.log_message("Test log 2")

        assert len(client.log_buffer) == 2
        assert client.log_buffer[0] == "Test log 1"
        assert client.log_buffer[1] == "Test log 2"

    @patch("src.stock_alert.services.discord.client.requests.post")
    def test_flush_logs_success(self, mock_post):
        """Test flushing logs to Discord"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response

        client = DiscordClient(log_webhook="https://log-webhook.com")
        client.log_message("Log 1")
        client.log_message("Log 2")

        client.flush_logs()

        assert len(client.log_buffer) == 0  # Buffer cleared
        mock_post.assert_called()

    def test_flush_logs_no_webhook(self):
        """Test flush logs clears buffer even without webhook"""
        client = DiscordClient(log_webhook=None)
        client.log_message("Log 1")

        client.flush_logs()

        assert len(client.log_buffer) == 0

    def test_split_message_short(self):
        """Test that short messages are not split"""
        message = "Short message"
        chunks = DiscordClient._split_message(message, 2000)

        assert len(chunks) == 1
        assert "Short message" in chunks[0]

    def test_split_message_long(self):
        """Test that long messages are split correctly"""
        # Create a message longer than max length
        lines = ["Line " + str(i) for i in range(100)]
        message = "\n".join(lines)

        chunks = DiscordClient._split_message(message, 200)

        assert len(chunks) > 1
        # Verify all chunks have code block fences
        for chunk in chunks:
            assert chunk.startswith("```")
            assert chunk.endswith("```")

    def test_format_timeframe(self):
        """Test timeframe formatting"""
        assert DiscordClient._format_timeframe("1d") == "1D (Daily)"
        assert DiscordClient._format_timeframe("1wk") == "1W (Weekly)"
        assert DiscordClient._format_timeframe("daily") == "1D (Daily)"
        assert DiscordClient._format_timeframe("custom") == "custom"


class TestBackwardCompatibilityFunctions:
    """Test backward compatibility functions"""

    @patch("src.stock_alert.services.discord.client.requests.post")
    def test_send_stock_alert_legacy(self, mock_post):
        """Test legacy send_stock_alert function"""
        mock_response = Mock()
        mock_response.status_code = 204
        mock_post.return_value = mock_response

        result = send_stock_alert(
            webhook_url="https://test.com",
            timeframe="1d",
            alert_name="Test",
            ticker="AAPL",
            triggered_condition="RSI > 70",
            current_price=150.0,
            action="Buy",
        )

        mock_post.assert_called_once()

    def test_log_to_discord_legacy(self):
        """Test legacy log_to_discord function"""
        # This should not raise an error
        log_to_discord("Test message")

    def test_flush_logs_to_discord_legacy(self):
        """Test legacy flush_logs_to_discord function"""
        # This test verifies the function completes without errors
        # Since async_discord_logger is mocked, it will succeed via the async path
        log_to_discord("Test log 1")
        log_to_discord("Test log 2")
        flush_logs_to_discord()

        # No assertion needed - just verify no exception is raised

"""
Discord webhook client for sending alerts and logs.

Provides a clean interface for Discord webhook interactions.
"""

import datetime
import logging
import time

import requests

from src.stock_alert.config.settings import Settings

logger = logging.getLogger(__name__)

# Maximum Discord message length
MAX_DISCORD_MESSAGE_LENGTH = 2000


class DiscordClient:
    """
    Client for interacting with Discord webhooks.

    Handles:
    - Sending stock alerts
    - Logging messages
    - Message chunking for long messages
    - Multiple webhook support
    """

    def __init__(
        self,
        primary_webhook: str | None = None,
        secondary_webhook: str | None = None,
        log_webhook: str | None = None,
        log_webhook_2: str | None = None,
    ):
        """
        Initialize Discord client.

        Args:
            primary_webhook: Primary webhook URL for alerts
            secondary_webhook: Secondary webhook URL for alerts
            log_webhook: Primary webhook for logging
            log_webhook_2: Secondary webhook for logging
        """
        # Try to load from settings if webhooks not provided
        if not any([primary_webhook, secondary_webhook, log_webhook, log_webhook_2]):
            try:
                settings = Settings()
                self.primary_webhook = primary_webhook or settings.WEBHOOK_URL
                self.secondary_webhook = secondary_webhook or settings.WEBHOOK_URL_2
                self.log_webhook = log_webhook or settings.WEBHOOK_URL_LOGGING
                self.log_webhook_2 = log_webhook_2 or settings.WEBHOOK_URL_LOGGING_2
            except Exception:
                # If settings can't be loaded, use provided values
                self.primary_webhook = primary_webhook
                self.secondary_webhook = secondary_webhook
                self.log_webhook = log_webhook
                self.log_webhook_2 = log_webhook_2
        else:
            self.primary_webhook = primary_webhook
            self.secondary_webhook = secondary_webhook
            self.log_webhook = log_webhook
            self.log_webhook_2 = log_webhook_2

        # Log buffer for batching
        self.log_buffer: list[str] = []

    def send_alert(
        self,
        alert_name: str,
        ticker: str,
        triggered_condition: str,
        current_price: float,
        action: str,
        timeframe: str,
        exchange: str = "Unknown",
        webhook_url: str | None = None,
    ) -> bool:
        """
        Send a stock alert to Discord.

        Args:
            alert_name: Name of the alert
            ticker: Stock ticker symbol
            triggered_condition: Description of triggered condition
            current_price: Current stock price
            action: Action (Buy/Sell)
            timeframe: Timeframe of alert
            exchange: Exchange name
            webhook_url: Override webhook URL (uses primary if not provided)

        Returns:
            True if sent successfully
        """
        webhook = webhook_url or self.primary_webhook

        if not webhook:
            logger.warning(f"No webhook URL configured for alert: {alert_name}")
            return False

        # Color based on action
        color = 0x00FF00 if action == "Buy" else 0xFF0000

        # Format timeframe for display
        timeframe_display = self._format_timeframe(timeframe)

        embed = {
            "title": f"[ALERT] {alert_name} ({ticker})",
            "description": f"The condition **{triggered_condition}** was triggered. \n Action: {action}",
            "fields": [
                {"name": "Timeframe", "value": timeframe_display, "inline": True},
                {"name": "Exchange", "value": exchange, "inline": True},
                {"name": "Current Price", "value": f"${current_price:.2f}", "inline": True},
            ],
            "color": color,
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        }

        payload = {"embeds": [embed]}

        try:
            response = requests.post(webhook, json=payload, timeout=10)
            if response.status_code == 204:
                logger.info(f"Alert sent successfully for {ticker}")
                return True
            else:
                logger.error(
                    f"Failed to send alert. Status: {response.status_code}, "
                    f"Response: {response.text}"
                )
                return False
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            return False

    def send_to_multiple_webhooks(
        self,
        alert_name: str,
        ticker: str,
        triggered_condition: str,
        current_price: float,
        action: str,
        timeframe: str,
        exchange: str = "Unknown",
    ) -> int:
        """
        Send alert to both primary and secondary webhooks.

        Returns:
            Number of successful sends
        """
        success_count = 0

        # Send to primary webhook
        if self.send_alert(
            alert_name,
            ticker,
            triggered_condition,
            current_price,
            action,
            timeframe,
            exchange,
            self.primary_webhook,
        ):
            success_count += 1

        # Send to secondary if different from primary
        if self.secondary_webhook and self.secondary_webhook != self.primary_webhook:
            if self.send_alert(
                alert_name,
                ticker,
                triggered_condition,
                current_price,
                action,
                timeframe,
                exchange,
                self.secondary_webhook,
            ):
                success_count += 1

        return success_count

    def log_message(self, message: str):
        """
        Add a message to the log buffer.

        Args:
            message: Message to log
        """
        self.log_buffer.append(str(message))

    def flush_logs(self):
        """
        Flush buffered logs to Discord webhooks.
        """
        if not self.log_buffer:
            return

        # Check if webhook URL is configured
        if not self.log_webhook:
            self.log_buffer.clear()
            logger.warning("No log webhook configured, clearing buffer")
            return

        # Combine all logs
        full_message = "\n".join(self.log_buffer)
        chunks = self._split_message(full_message, MAX_DISCORD_MESSAGE_LENGTH)

        # Send to primary log webhook
        for chunk in chunks:
            payload = {"content": chunk}
            try:
                response = requests.post(self.log_webhook, json=payload, timeout=10)
                response.raise_for_status()
                time.sleep(0.1)  # Rate limiting
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to send log to Discord: {e}")
                break

            # Send to secondary log webhook if different
            if self.log_webhook_2 and self.log_webhook_2 != self.log_webhook:
                try:
                    response_2 = requests.post(self.log_webhook_2, json=payload, timeout=10)
                    response_2.raise_for_status()
                    time.sleep(0.1)
                except requests.exceptions.RequestException as e:
                    logger.error(f"Failed to send log to secondary webhook: {e}")
                    break

        # Clear buffer after sending
        self.log_buffer.clear()

    @staticmethod
    def _split_message(message: str, max_length: int) -> list[str]:
        """
        Split a long message into multiple code blocks.

        Args:
            message: Message to split
            max_length: Maximum length per chunk

        Returns:
            List of message chunks
        """
        lines = message.split("\n")
        chunks = []
        current_chunk = ""

        for line in lines:
            # 6 characters for code block fences (```)
            if len(current_chunk) + len(line) + 1 < max_length - 6:
                current_chunk += line + "\n"
            else:
                if current_chunk:
                    chunks.append(f"```{current_chunk.strip()}```")
                current_chunk = line + "\n"

        if current_chunk:
            chunks.append(f"```{current_chunk.strip()}```")

        return chunks

    @staticmethod
    def _format_timeframe(timeframe: str) -> str:
        """
        Format timeframe for display.

        Args:
            timeframe: Raw timeframe string

        Returns:
            Formatted timeframe
        """
        timeframe_map = {
            "1d": "1D (Daily)",
            "1wk": "1W (Weekly)",
            "1w": "1W (Weekly)",
            "1D": "1D (Daily)",
            "1W": "1W (Weekly)",
            "daily": "1D (Daily)",
            "weekly": "1W (Weekly)",
        }

        tf_lower = timeframe.lower() if isinstance(timeframe, str) else str(timeframe)
        return timeframe_map.get(tf_lower, timeframe)


# Convenience functions for backward compatibility
def send_stock_alert(
    webhook_url: str,
    timeframe: str,
    alert_name: str,
    ticker: str,
    triggered_condition: str,
    current_price: float,
    action: str,
    exchange: str = "Unknown",
):
    """
    Legacy function for sending stock alerts.

    Maintained for backward compatibility with existing code.
    """
    client = DiscordClient()
    return client.send_alert(
        alert_name=alert_name,
        ticker=ticker,
        triggered_condition=triggered_condition,
        current_price=current_price,
        action=action,
        timeframe=timeframe,
        exchange=exchange,
        webhook_url=webhook_url,
    )


# Module-level log buffer for backward compatibility
_global_log_buffer: list[str] = []


def log_to_discord(message: str):
    """
    Legacy function for logging to Discord.

    Maintained for backward compatibility with existing code.
    """
    # Try async logger first (if available)
    try:
        from async_discord_logger import log_to_discord_async

        log_to_discord_async(str(message))
        return
    except ImportError:
        pass

    # Fallback to buffer
    global _global_log_buffer
    _global_log_buffer.append(str(message))


def flush_logs_to_discord():
    """
    Legacy function for flushing logs to Discord.

    Maintained for backward compatibility with existing code.
    """
    # Try async logger first (if available)
    try:
        from async_discord_logger import flush_discord_logs

        flush_discord_logs()
        return
    except ImportError:
        pass

    # Fallback to client
    global _global_log_buffer
    if _global_log_buffer:
        client = DiscordClient()
        client.log_buffer = _global_log_buffer.copy()
        client.flush_logs()
        _global_log_buffer.clear()

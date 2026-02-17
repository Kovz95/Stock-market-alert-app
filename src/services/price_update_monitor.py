"""
Price Update Monitor
Reports failed price updates to Discord channel
"""

import logging
import requests
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

from src.data_access.document_store import load_document

logger = logging.getLogger(__name__)

# Discord enforces a hard 6000-character total limit per embed.
_MAX_EMBED_SIZE = 5800  # Leave margin for JSON overhead


def _estimate_embed_size(embed: dict) -> int:
    """Estimate the total character count of a Discord embed."""
    size = len(embed.get("title", ""))
    size += len(embed.get("description", ""))
    for field in embed.get("fields", []):
        size += len(field.get("name", ""))
        size += len(field.get("value", ""))
    footer = embed.get("footer", {})
    size += len(footer.get("text", ""))
    return size


def _split_embed_if_needed(embed: dict) -> List[Dict]:
    """Split an embed into multiple embeds if it exceeds Discord's 6000 char limit.

    Keeps the title/description/footer on the first embed only, then distributes
    fields across continuation embeds.

    Args:
        embed: A Discord embed dict with fields that may exceed the size limit.

    Returns:
        List of embed dicts, each under the 6000-character limit.
    """
    if _estimate_embed_size(embed) <= _MAX_EMBED_SIZE:
        return [embed]

    # Split by fields: keep adding fields until we approach the limit
    base_size = (
        len(embed.get("title", ""))
        + len(embed.get("description", ""))
        + len(embed.get("footer", {}).get("text", ""))
        + 100  # overhead for timestamp, color, etc.
    )

    all_fields = embed.get("fields", [])
    embeds: List[Dict] = []
    current_fields: List[Dict] = []
    current_size = base_size

    for field in all_fields:
        field_size = len(field.get("name", "")) + len(field.get("value", ""))

        if current_size + field_size > _MAX_EMBED_SIZE and current_fields:
            # Flush current embed
            new_embed: dict = {
                "color": embed.get("color"),
                "fields": current_fields,
            }
            if not embeds:
                # First embed gets title/description/timestamp
                new_embed["title"] = embed.get("title", "")
                if embed.get("description"):
                    new_embed["description"] = embed["description"]
                if embed.get("timestamp"):
                    new_embed["timestamp"] = embed["timestamp"]
            else:
                new_embed["title"] = embed.get("title", "…") + " (continued)"
            embeds.append(new_embed)
            current_fields = []
            current_size = 200  # continuation embed overhead

        current_fields.append(field)
        current_size += field_size

    # Flush remaining fields
    if current_fields:
        new_embed = {
            "color": embed.get("color"),
            "fields": current_fields,
        }
        if not embeds:
            new_embed["title"] = embed.get("title", "")
            if embed.get("description"):
                new_embed["description"] = embed["description"]
            if embed.get("timestamp"):
                new_embed["timestamp"] = embed["timestamp"]
        else:
            new_embed["title"] = embed.get("title", "…") + " (continued)"
        # Footer goes on the last embed
        if embed.get("footer"):
            new_embed["footer"] = embed["footer"]
        embeds.append(new_embed)

    logger.info(
        "Split oversized embed (%d chars) into %d embeds",
        _estimate_embed_size(embed),
        len(embeds),
    )
    return embeds


class PriceUpdateMonitor:
    """Monitor and report failed price updates to Discord"""

    def __init__(self):
        # Load Discord configuration
        self.discord_config = load_document(
            'discord_channels_config',
            default={},
            fallback_path='discord_channels_config.json',
        ) or {}

        # Get the failed price updates channel
        self.failed_channel = self.discord_config.get('channel_mappings', {}).get('Failed_Price_Updates', {})
        self.webhook_url = self.failed_channel.get('webhook_url')

        if not self.webhook_url:
            logger.warning("Failed_Price_Updates webhook URL not found in config")

    def send_to_discord(self, message: str, embeds: Optional[List[Dict]] = None):
        """Send a message to the failed price updates Discord channel.

        Automatically splits oversized embeds to stay under Discord's
        6000-character embed limit.
        """
        if not self.webhook_url:
            logger.warning("No webhook URL configured for failed price updates")
            return False
        from src.utils.discord_env import get_discord_environment_tag, is_discord_send_enabled
        if not is_discord_send_enabled():
            return False

        # Split any oversized embeds into smaller ones
        safe_embeds: List[Dict] = []
        if embeds:
            for embed in embeds:
                safe_embeds.extend(_split_embed_if_needed(embed))

        # Discord allows max 10 embeds per message — chunk if needed
        success = True
        env_tag = get_discord_environment_tag()
        embed_chunks = [
            safe_embeds[i:i + 10]
            for i in range(0, max(len(safe_embeds), 1), 10)
        ] if safe_embeds else [None]

        for chunk in embed_chunks:
            try:
                payload: dict[str, Any] = {"content": env_tag + message}
                if chunk:
                    payload["embeds"] = chunk

                response = requests.post(self.webhook_url, json=payload, timeout=10)

                if response.status_code in (200, 204):
                    logger.debug("Discord webhook sent successfully")
                else:
                    logger.error(
                        "Discord webhook failed: %s - %s",
                        response.status_code,
                        response.text[:200],
                    )
                    success = False

            except Exception as e:
                logger.error(f"Error sending to Discord: {e}")
                success = False

        return success

    def report_failed_updates(self, failed_tickers: List[Dict], exchange: str | None = None):
        """
        Report failed price updates to Discord

        Args:
            failed_tickers: List of dicts with ticker info and error details
            exchange: Optional exchange name for context
        """
        if not failed_tickers:
            return

        # Create embed for better formatting
        embed: dict[str, Any] = {
            "title": "⚠️ Failed Price Updates",
            "color": 15158332,  # Red color
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "fields": []
        }

        if exchange:
            embed["description"] = f"Exchange: **{exchange}**"

        # Add summary field
        embed["fields"].append({
            "name": "Summary",
            "value": f"Failed to update {len(failed_tickers)} ticker(s)",
            "inline": False
        })

        # Group failures by error type
        error_groups = {}
        for ticker_info in failed_tickers:  # Process ALL failed tickers
            error = ticker_info.get('error', 'Unknown error')
            # Clean up the error message
            if 'Cannot convert input' in error and 'bytes' in error:
                error = 'Database corruption - timestamp format issue'
            elif 'No data available' in error:
                error = 'No data available from API'
            elif len(error) > 100:
                error = error[:100] + '...'

            if error not in error_groups:
                error_groups[error] = []
            error_groups[error].append(ticker_info.get('ticker', 'Unknown'))

        # Add fields for each error type - show ALL tickers
        for error, tickers in error_groups.items():
            # Discord has a 1024 character limit per field value
            # Split tickers into manageable chunks
            all_tickers = sorted(tickers)  # Sort for easier reading

            # Create the initial field with error description
            chunk_size = 50  # Conservative to keep total embed size manageable
            chunks = [all_tickers[i:i + chunk_size] for i in range(0, len(all_tickers), chunk_size)]

            for i, chunk in enumerate(chunks):
                if len(embed["fields"]) >= 24:  # Discord has 25 field limit
                    # Add final field indicating more exist
                    embed["fields"].append({
                        "name": "Note",
                        "value": f"...and {len(all_tickers) - (i * chunk_size)} more tickers not shown due to Discord limits",
                        "inline": False
                    })
                    break

                field_name = f"Error: {error}" if i == 0 else f"└─ Continued ({i+1})"
                ticker_list = ', '.join(chunk)

                embed["fields"].append({
                    "name": field_name,
                    "value": ticker_list[:1024],  # Ensure we don't exceed Discord's limit
                    "inline": False
                })

        # Send to Discord
        self.send_to_discord("", embeds=[embed])

    def report_skipped_tickers(self, skipped_tickers: List[str], exchange: str | None = None):
        """
        Report skipped tickers to Discord

        Args:
            skipped_tickers: List of ticker symbols that were skipped
            exchange: Optional exchange name for context
        """
        if not skipped_tickers:
            return

        # Create embed for skipped tickers
        embed: dict[str, Any] = {
            "title": "⏭️ Skipped Price Updates",
            "color": 16776960,  # Yellow color for skipped
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),

            "fields": []
        }

        if exchange:
            embed["description"] = f"Exchange: **{exchange}**"

        # Add summary field
        embed["fields"].append({
            "name": "Summary",
            "value": f"Skipped {len(skipped_tickers)} ticker(s) (already up-to-date)",
            "inline": False
        })

        # Add tickers in chunks
        all_tickers = sorted(skipped_tickers)
        chunk_size = 50  # Conservative to keep total embed size manageable
        chunks = [all_tickers[i:i + chunk_size] for i in range(0, len(all_tickers), chunk_size)]

        for i, chunk in enumerate(chunks):
            if len(embed["fields"]) >= 24:  # Discord limit
                embed["fields"].append({
                    "name": "Note",
                    "value": f"...and {len(all_tickers) - (i * chunk_size)} more tickers not shown",
                    "inline": False
                })
                break

            field_name = f"Skipped Tickers" if i == 0 else f"└─ Continued ({i+1})"
            ticker_list = ', '.join(chunk)

            embed["fields"].append({
                "name": field_name,
                "value": ticker_list[:1024],
                "inline": False
            })

        # Send to Discord
        self.send_to_discord("", embeds=[embed])

    def report_update_summary(self, summary: Dict):
        """
        Report a summary of price update run

        Args:
            summary: Dict with update statistics
        """
        total = summary.get('total', 0)
        successful = summary.get('successful', 0)
        failed = summary.get('failed', 0)
        skipped = summary.get('skipped', 0)
        skipped_tickers = summary.get('skipped_tickers', [])
        exchange = summary.get('exchange', 'Unknown')
        duration = summary.get('duration_seconds', 0)

        # Report failures if any
        if failed > 0 or skipped > 0:
            # Create embed
            embed = {
                "title": "📊 Price Update Summary",
                "color": 15158332 if failed > 0 else 16776960,  # Red if failures, yellow if only skipped
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "fields": [
                    {
                        "name": "Exchange",
                        "value": exchange,
                        "inline": True
                    },
                    {
                        "name": "Total Tickers",
                        "value": str(total),
                        "inline": True
                    },
                    {
                        "name": "Duration",
                        "value": f"{duration:.1f}s" if duration < 60 else f"{duration/60:.1f}m",
                        "inline": True
                    },
                    {
                        "name": "✅ Successful",
                        "value": str(successful),
                        "inline": True
                    },
                    {
                        "name": "❌ Failed",
                        "value": str(failed),
                        "inline": True
                    },
                    {
                        "name": "⏭️ Skipped",
                        "value": str(skipped),
                        "inline": True
                    }
                ]
            }

            # Add footer with rates
            if total > 0:
                failure_rate = (failed / total) * 100
                skip_rate = (skipped / total) * 100
                embed["footer"] = {
                    "text": f"Failure rate: {failure_rate:.1f}% | Skip rate: {skip_rate:.1f}%"
                }

            # Send summary to Discord
            self.send_to_discord("", embeds=[embed])

            # If there are skipped tickers, report them separately
            if skipped_tickers:
                self.report_skipped_tickers(skipped_tickers, exchange)


def integrate_with_scheduler():
    """
    Integration code for scheduled_price_updater.py
    Add this to your price update process
    """
    integration_code = '''
# Add this import at the top of scheduled_price_updater.py
from src.services.price_update_monitor import PriceUpdateMonitor

# Add this to the ScheduledPriceUpdater class __init__:
self.monitor = PriceUpdateMonitor()

# In the update_exchange_prices method, track failures:
failed_updates = []
for ticker in batch:
    try:
        if not self.collector.update_ticker(ticker):
            failed_updates.append({
                'ticker': ticker,
                'error': 'No data available'
            })
    except Exception as e:
        failed_updates.append({
            'ticker': ticker,
            'error': str(e)
        })

# After processing, report failures:
if failed_updates:
    self.monitor.report_failed_updates(failed_updates, exchange_names[0])

# At the end of run_scheduled_update, report summary:
summary = {
    'exchange': ', '.join(countries),
    'total': len(tickers),
    'successful': stats['updated'] + stats['new'],
    'failed': stats['failed'],
    'skipped': stats['skipped'],
    'duration_seconds': elapsed
}
self.monitor.report_update_summary(summary)
'''
    return integration_code


if __name__ == "__main__":
    # Test the monitor
    monitor = PriceUpdateMonitor()

    # Test with sample failed updates
    test_failures = [
        {'ticker': 'TEST1', 'error': 'API rate limit exceeded'},
        {'ticker': 'TEST2', 'error': 'No data available'},
        {'ticker': 'TEST3', 'error': 'Invalid symbol'}
    ]

    print("Testing failed price update reporting...")
    monitor.report_failed_updates(test_failures, "TEST_EXCHANGE")

    # Test summary report
    test_summary = {
        'exchange': 'TEST_EXCHANGE',
        'total': 100,
        'successful': 85,
        'failed': 10,
        'skipped': 5,
        'duration_seconds': 120.5
    }

    print("Testing summary reporting...")
    monitor.report_update_summary(test_summary)

    print("\nCheck your Discord #failed-price-updates channel for test messages")

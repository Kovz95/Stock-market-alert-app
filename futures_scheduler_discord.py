"""
Send futures scheduler status updates to Discord
"""

import requests
from datetime import datetime
import logging

from src.data_access.document_store import load_document

logger = logging.getLogger(__name__)

class FuturesSchedulerDiscord:
    def __init__(self):
        self.config = self.load_discord_config()

    def load_discord_config(self):
        """Load Discord webhook configuration"""
        try:
            config = load_document(
                "discord_channels_config",
                default={},
                fallback_path='discord_channels_config.json',
            ) or {}
            return config.get('logging_channels', {}).get('Futures_Scheduler_Status', {})
        except Exception as e:
            logger.error(f"Error loading Discord config: {e}")
            return {}

    def send_status_update(self, status_type, details):
        """Send status update to Discord"""
        webhook_url = self.config.get('webhook_url')

        if not webhook_url or webhook_url == "YOUR_WEBHOOK_URL_HERE":
            logger.info(f"Discord webhook not configured for futures scheduler status")
            return False

        try:
            # Determine color based on status type
            color_map = {
                'started': 0x00ff00,  # Green
                'stopped': 0xff0000,  # Red
                'price_update_complete': 0x00ff00,  # Green
                'price_update_failed': 0xffa500,  # Orange
                'alert_check_complete': 0x3498db,  # Blue
                'alert_triggered': 0xffff00,  # Yellow
                'error': 0xff0000,  # Red
            }

            color = color_map.get(status_type, 0x808080)  # Default gray

            # Create embed
            embed = {
                "title": f"🔮 Futures Scheduler Status",
                "color": color,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "fields": []
            }

            # Add status-specific fields
            if status_type == 'started':
                embed["description"] = "✅ Futures scheduler has started"
                embed["fields"].append({
                    "name": "Update Schedule",
                    "value": details.get('schedule', 'Not configured'),
                    "inline": False
                })

            elif status_type == 'stopped':
                embed["description"] = "🛑 Futures scheduler has stopped"

            elif status_type == 'price_update_complete':
                embed["description"] = "📊 Price update completed successfully"
                embed["fields"].append({
                    "name": "Contracts Updated",
                    "value": str(details.get('updated_count', 0)),
                    "inline": True
                })
                embed["fields"].append({
                    "name": "Duration",
                    "value": details.get('duration', 'Unknown'),
                    "inline": True
                })

            elif status_type == 'price_update_failed':
                embed["description"] = "⚠️ Price update encountered errors"
                embed["fields"].append({
                    "name": "Failed Contracts",
                    "value": str(details.get('failed_count', 0)),
                    "inline": True
                })
                embed["fields"].append({
                    "name": "Error",
                    "value": details.get('error', 'Unknown error')[:1024],
                    "inline": False
                })

            elif status_type == 'alert_check_complete':
                embed["description"] = "🔔 Alert check completed"
                embed["fields"].append({
                    "name": "Alerts Checked",
                    "value": str(details.get('alert_count', 0)),
                    "inline": True
                })
                embed["fields"].append({
                    "name": "Triggered",
                    "value": str(details.get('triggered_count', 0)),
                    "inline": True
                })

            elif status_type == 'alert_triggered':
                embed["description"] = "🚨 Futures alert triggered!"
                embed["fields"].append({
                    "name": "Symbol",
                    "value": details.get('symbol', 'Unknown'),
                    "inline": True
                })
                embed["fields"].append({
                    "name": "Alert",
                    "value": details.get('alert_name', 'Unknown'),
                    "inline": True
                })
                embed["fields"].append({
                    "name": "Condition",
                    "value": details.get('condition', 'Unknown')[:1024],
                    "inline": False
                })

            elif status_type == 'error':
                embed["description"] = "❌ Scheduler error occurred"
                embed["fields"].append({
                    "name": "Error",
                    "value": details.get('error', 'Unknown error')[:1024],
                    "inline": False
                })

            # Add next run time if available
            if 'next_run' in details:
                embed["fields"].append({
                    "name": "Next Run",
                    "value": details['next_run'],
                    "inline": False
                })

            # Send to Discord
            payload = {"embeds": [embed]}
            response = requests.post(webhook_url, json=payload)

            if response.status_code == 204:
                logger.info(f"Sent {status_type} status to Discord")
                return True
            else:
                logger.error(f"Failed to send Discord status: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error sending Discord status: {e}")
            return False

    def send_daily_summary(self, summary_data):
        """Send daily summary to Discord"""
        webhook_url = self.config.get('webhook_url')

        if not webhook_url or webhook_url == "YOUR_WEBHOOK_URL_HERE":
            return False

        try:
            embed = {
                "title": "📊 Futures Daily Summary",
                "description": f"Summary for {datetime.now().strftime('%Y-%m-%d')}",
                "color": 0x00ff00,
                "fields": [
                    {
                        "name": "Total Updates",
                        "value": str(summary_data.get('total_updates', 0)),
                        "inline": True
                    },
                    {
                        "name": "Successful",
                        "value": str(summary_data.get('successful_updates', 0)),
                        "inline": True
                    },
                    {
                        "name": "Failed",
                        "value": str(summary_data.get('failed_updates', 0)),
                        "inline": True
                    },
                    {
                        "name": "Alert Checks",
                        "value": str(summary_data.get('alert_checks', 0)),
                        "inline": True
                    },
                    {
                        "name": "Alerts Triggered",
                        "value": str(summary_data.get('alerts_triggered', 0)),
                        "inline": True
                    },
                    {
                        "name": "Uptime",
                        "value": summary_data.get('uptime', 'Unknown'),
                        "inline": True
                    }
                ],
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }

            payload = {"embeds": [embed]}
            response = requests.post(webhook_url, json=payload)

            return response.status_code == 204

        except Exception as e:
            logger.error(f"Error sending daily summary: {e}")
            return False

# Create singleton instance
futures_discord = FuturesSchedulerDiscord()

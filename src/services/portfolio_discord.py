"""
Portfolio Discord integration module
Manages portfolio-based Discord alerts
"""

import requests
from typing import List, Tuple, Dict, Optional

from src.data_access.portfolio_repository import list_portfolios as repo_list_portfolios


class PortfolioManager:
    """Manages portfolio-based Discord alerts"""

    def __init__(self):
        self.portfolios = {}
        self.load_portfolios()

    def load_portfolios(self):
        """Load portfolio configurations from the repository."""
        try:
            self.portfolios = repo_list_portfolios()
            print(f"Loaded {len(self.portfolios)} portfolios")
        except Exception as e:
            print(f"Error loading portfolios: {e}")
            self.portfolios = {}

    def get_portfolios_for_stock(self, ticker: str) -> List[Tuple[str, Dict]]:
        """
        Get portfolios containing a specific stock

        Args:
            ticker: Stock ticker symbol

        Returns:
            List of tuples (portfolio_id, portfolio_data) containing this stock
        """
        result = []

        # Only reload if portfolios are empty (first time)
        if not self.portfolios:
            self.load_portfolios()

        for portfolio_id, portfolio_data in self.portfolios.items():
            # Check if portfolio is enabled
            if not portfolio_data.get('enabled', True):
                continue

            # Check if ticker is in portfolio stocks
            stocks = portfolio_data.get('stocks', [])
            for stock in stocks:
                if stock.get('symbol', '').upper() == ticker.upper():
                    result.append((portfolio_id, portfolio_data))
                    break

        return result

    def send_portfolio_alert(self, alert_data: Dict, portfolio_id: str, portfolio_data: Dict):
        """
        Send alert to portfolio Discord channel

        Args:
            alert_data: Alert information dictionary (must contain 'message' key with full formatted message)
            portfolio_id: Portfolio ID
            portfolio_data: Portfolio configuration data
        """
        try:
            webhook_url = portfolio_data.get('discord_webhook')
            if not webhook_url:
                return

            portfolio_name = portfolio_data.get('name', 'Portfolio')

            # If we have a pre-formatted message, use it directly (same as industry alerts)
            if 'message' in alert_data:
                # Use the exact same message format as industry alerts
                formatted_message = f"**#{portfolio_name.lower().replace(' ', '-')}-alerts**\n{alert_data['message']}"

                payload = {
                    "content": formatted_message,
                    "username": "Stock Alert Bot",
                    "avatar_url": "https://cdn.discordapp.com/attachments/123456789/stock_alert_bot.png"
                }
            else:
                # Fallback to old embed format if no message provided
                embed = {
                    "title": f"📊 {portfolio_name} Alert",
                    "description": f"**{alert_data.get('ticker', 'Unknown')}** - {alert_data.get('name', 'Alert')}",
                    "color": 0x00ff00,  # Green color
                    "fields": [
                        {
                            "name": "Condition Met",
                            "value": alert_data.get('condition', 'N/A'),
                            "inline": False
                        },
                        {
                            "name": "Current Price",
                            "value": f"${alert_data.get('price', 'N/A')}",
                            "inline": True
                        },
                        {
                            "name": "Timeframe",
                            "value": alert_data.get('timeframe', 'N/A'),
                            "inline": True
                        }
                    ]
                }

                payload = {
                    "embeds": [embed]
                }

            # Send to Discord
            response = requests.post(webhook_url, json=payload)

            if response.status_code == 204:
                print(f"[PORTFOLIO ALERT] Successfully sent alert to portfolio: {portfolio_name} for {alert_data.get('ticker')}")
            else:
                print(f"[PORTFOLIO ALERT ERROR] Failed to send alert to portfolio {portfolio_name}: {response.status_code}")
                print(f"  Response: {response.text}")

        except Exception as e:
            print(f"Error sending portfolio alert: {e}")


# Create singleton instance
portfolio_manager = PortfolioManager()

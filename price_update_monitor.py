"""
Price Update Monitor
Reports failed price updates to Discord channel
"""

import logging
import requests
from datetime import datetime
from typing import List, Dict, Optional

from data_access.document_store import load_document

logger = logging.getLogger(__name__)

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
        """Send a message to the failed price updates Discord channel"""
        if not self.webhook_url:
            logger.warning("No webhook URL configured for failed price updates")
            return False
        
        try:
            payload = {"content": message}
            if embeds:
                payload["embeds"] = embeds
            
            response = requests.post(self.webhook_url, json=payload)
            
            if response.status_code == 204:
                return True
            else:
                logger.error(f"Discord webhook failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Error sending to Discord: {e}")
            return False
    
    def report_failed_updates(self, failed_tickers: List[Dict], exchange: str = None):
        """
        Report failed price updates to Discord
        
        Args:
            failed_tickers: List of dicts with ticker info and error details
            exchange: Optional exchange name for context
        """
        if not failed_tickers:
            return
        
        # Create embed for better formatting
        embed = {
            "title": "⚠️ Failed Price Updates",
            "color": 15158332,  # Red color
            "timestamp": datetime.utcnow().isoformat(),
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
            chunk_size = 80  # Approximate number of tickers that fit in 1024 chars
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
    
    def report_skipped_tickers(self, skipped_tickers: List[str], exchange: str = None):
        """
        Report skipped tickers to Discord
        
        Args:
            skipped_tickers: List of ticker symbols that were skipped
            exchange: Optional exchange name for context
        """
        if not skipped_tickers:
            return
        
        # Create embed for skipped tickers
        embed = {
            "title": "⏭️ Skipped Price Updates",
            "color": 16776960,  # Yellow color for skipped
            "timestamp": datetime.utcnow().isoformat(),
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
        chunk_size = 100
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
                "timestamp": datetime.utcnow().isoformat(),
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
from price_update_monitor import PriceUpdateMonitor

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

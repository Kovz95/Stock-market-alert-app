#!/usr/bin/env python3
"""
Alert Discord Routing Checker
Shows which Discord channel(s) an alert will be sent to
"""

import logging
import sys
from typing import Dict, Any, List

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

from discord_routing import discord_router
from data_access.alert_repository import list_alerts

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Suppress info logs for cleaner output
    format='%(message)s'
)
logger = logging.getLogger(__name__)


class AlertRoutingChecker:
    """Check Discord routing for alerts"""
    
    def __init__(self):
        self.router = discord_router
    
    def mask_webhook(self, url: str) -> str:
        """Mask webhook URL for security"""
        if not url:
            return "❌ NOT CONFIGURED"
        if "YOUR_" in url or url == "YOUR_GENERAL_WEBHOOK_URL_HERE":
            return "⚠️ PLACEHOLDER (not configured)"
        
        # Show first 30 and last 10 chars
        if len(url) > 50:
            return f"{url[:30]}...{url[-10:]}"
        return url[:40] + "..."
    
    def explain_routing(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """
        Explain where an alert will be routed and why
        
        Returns:
            Dictionary with routing information
        """
        ticker = alert.get('ticker', alert.get('ticker1', 'N/A'))
        timeframe = alert.get('timeframe', '1d')
        
        result = {
            'alert_id': alert.get('alert_id', 'N/A'),
            'ticker': ticker,
            'timeframe': timeframe,
            'primary_channel': None,
            'primary_webhook': None,
            'routing_reason': None,
            'custom_channels': [],
            'total_channels': 0,
        }
        
        # Determine primary channel routing
        if alert.get('ratio') == 'Yes':
            result['routing_reason'] = 'Ratio Alert'
            result['primary_channel'] = 'Pairs'
        elif self.router.is_etf(ticker):
            result['routing_reason'] = f'ETF (Asset Type)'
            result['primary_channel'] = 'ETFs'
        else:
            # Check economy classification
            economy = self.router.get_stock_economy(ticker)
            if economy:
                result['routing_reason'] = f'Economy: {economy}'
                result['primary_channel'] = economy
            else:
                result['routing_reason'] = 'No economy found - using default'
                default_channel = self.router.config.get('default_channel', 'General')
                result['primary_channel'] = default_channel
        
        # Get the actual channel info
        channel_name, webhook_url = self.router.determine_alert_channel(alert)
        result['primary_channel_display'] = channel_name
        result['primary_webhook'] = webhook_url
        
        # Check custom channels
        custom_channels = self.router.get_custom_channels_for_alert(alert)
        for custom_name, custom_info in custom_channels:
            result['custom_channels'].append({
                'name': custom_name,
                'display_name': custom_info.get('channel_name', custom_name),
                'webhook': custom_info.get('webhook_url', ''),
                'condition': custom_info.get('condition', 'Unknown'),
            })
        
        result['total_channels'] = 1 + len(result['custom_channels'])
        
        return result
    
    def check_alert(self, alert_id: str) -> None:
        """Check routing for a specific alert"""
        print("=" * 70)
        print(f"DISCORD ROUTING CHECK: {alert_id}")
        print("=" * 70)
        
        # Find the alert
        alerts = list_alerts()
        alert = None
        
        for a in alerts:
            if a.get('alert_id') == alert_id:
                alert = a
                break
        
        if not alert:
            print(f"❌ Alert {alert_id} not found")
            return
        
        # Get routing info
        routing = self.explain_routing(alert)
        
        # Display alert info
        print(f"\n📋 Alert Information:")
        print(f"   ID: {routing['alert_id']}")
        print(f"   Name: {alert.get('name', alert.get('stock_name', 'Unnamed'))}")
        print(f"   Ticker: {routing['ticker']}")
        print(f"   Timeframe: {routing['timeframe']}")
        
        # Display routing decision
        print(f"\n🎯 Primary Channel Routing:")
        print(f"   Reason: {routing['routing_reason']}")
        print(f"   Channel: {routing['primary_channel_display']}")
        print(f"   Webhook: {self.mask_webhook(routing['primary_webhook'])}")
        
        # Check if webhook is configured
        if not routing['primary_webhook'] or "YOUR_" in routing['primary_webhook']:
            print(f"   ⚠️  WARNING: Webhook not properly configured!")
        else:
            print(f"   ✅ Webhook configured")
        
        # Display custom channels
        if routing['custom_channels']:
            print(f"\n📢 Additional Custom Channels ({len(routing['custom_channels'])}):")
            for i, custom in enumerate(routing['custom_channels'], 1):
                print(f"\n   {i}. {custom['display_name']}")
                print(f"      Matched Condition: {custom['condition']}")
                print(f"      Webhook: {self.mask_webhook(custom['webhook'])}")
        else:
            print(f"\n📢 Additional Custom Channels: None")
        
        # Summary
        print(f"\n📊 Summary:")
        print(f"   Total channels this alert will send to: {routing['total_channels']}")
        
        # Show conditions for context
        conditions = alert.get('conditions', [])
        if conditions:
            print(f"\n🔍 Alert Conditions (for reference):")
            for i, cond in enumerate(conditions[:3], 1):  # Show first 3
                if isinstance(cond, dict):
                    cond_str = cond.get('conditions', '')
                elif isinstance(cond, str):
                    cond_str = cond
                else:
                    cond_str = str(cond)
                
                # Truncate long conditions
                if len(cond_str) > 70:
                    cond_str = cond_str[:67] + "..."
                print(f"   {i}. {cond_str}")
            
            if len(conditions) > 3:
                print(f"   ... and {len(conditions) - 3} more")
        
        print("\n" + "=" * 70)
    
    def check_all_alerts(self) -> None:
        """Show routing for all alerts"""
        print("=" * 70)
        print("DISCORD ROUTING SUMMARY - ALL ALERTS")
        print("=" * 70)
        
        alerts = list_alerts()
        
        if not alerts:
            print("No alerts found")
            return
        
        # Group by channel
        channel_groups: Dict[str, List[Dict]] = {}
        
        for alert in alerts:
            routing = self.explain_routing(alert)
            
            channel_key = routing['primary_channel_display']
            if channel_key not in channel_groups:
                channel_groups[channel_key] = []
            
            channel_groups[channel_key].append({
                'alert_id': alert.get('alert_id', 'N/A'),
                'name': alert.get('name', alert.get('stock_name', 'Unnamed')),
                'ticker': routing['ticker'],
                'reason': routing['routing_reason'],
                'custom_count': len(routing['custom_channels']),
                'webhook_configured': bool(routing['primary_webhook']) and "YOUR_" not in routing['primary_webhook'],
            })
        
        # Display by channel
        print(f"\nTotal Alerts: {len(alerts)}")
        print(f"Unique Channels: {len(channel_groups)}\n")
        
        for channel, alert_list in sorted(channel_groups.items()):
            print(f"\n📢 {channel} ({len(alert_list)} alerts)")
            print("   " + "-" * 66)
            
            for i, alert_info in enumerate(alert_list[:5], 1):  # Show first 5 per channel
                status = "✅" if alert_info['webhook_configured'] else "⚠️"
                custom_str = f" +{alert_info['custom_count']} custom" if alert_info['custom_count'] > 0 else ""
                print(f"   {status} {alert_info['ticker']:8} | {alert_info['name'][:35]:35}{custom_str}")
            
            if len(alert_list) > 5:
                print(f"   ... and {len(alert_list) - 5} more")
        
        # Show configuration warnings
        unconfigured = sum(
            1 for alerts in channel_groups.values()
            for alert in alerts
            if not alert['webhook_configured']
        )
        
        if unconfigured > 0:
            print(f"\n⚠️  WARNING: {unconfigured} alerts have unconfigured webhooks!")
            print("   Update discord_channels_config.json with proper webhook URLs")
        
        print("\n" + "=" * 70)
    
    def check_ticker_routing(self, ticker: str) -> None:
        """Check where a ticker would be routed"""
        print("=" * 70)
        print(f"DISCORD ROUTING CHECK FOR TICKER: {ticker}")
        print("=" * 70)
        
        # Create a dummy alert
        test_alert = {
            'ticker': ticker,
            'name': f'Test Alert for {ticker}',
            'timeframe': '1d',
        }
        
        # Check economy
        economy = self.router.get_stock_economy(ticker)
        is_etf = self.router.is_etf(ticker)
        is_futures = self.router.is_futures(ticker)
        
        print(f"\n📊 Ticker Analysis:")
        print(f"   Ticker: {ticker}")
        print(f"   Economy Classification: {economy or 'Not found'}")
        print(f"   Is ETF: {'Yes' if is_etf else 'No'}")
        print(f"   Is Futures: {'Yes' if is_futures else 'No'}")
        
        # Get routing
        channel_name, webhook_url = self.router.determine_alert_channel(test_alert)
        
        print(f"\n🎯 Routing Decision:")
        print(f"   Channel: {channel_name}")
        print(f"   Webhook: {self.mask_webhook(webhook_url)}")
        
        if not webhook_url or "YOUR_" in webhook_url:
            print(f"   ⚠️  WARNING: Webhook not configured for this channel!")
        else:
            print(f"   ✅ Webhook is configured")
        
        print("\n" + "=" * 70)
    
    def show_channel_config(self) -> None:
        """Show Discord channel configuration"""
        print("=" * 70)
        print("DISCORD CHANNEL CONFIGURATION")
        print("=" * 70)
        
        print("\n📢 Daily Alert Channels:")
        daily_channels = self.router.get_available_channels(timeframe='daily')
        
        for channel in daily_channels:
            status = "✅" if channel['configured'] else "⚠️"
            print(f"   {status} {channel['channel_name']:30} | {channel['description'][:35]}")
        
        print("\n⏰ Hourly Alert Channels:")
        hourly_channels = self.router.get_available_channels(timeframe='hourly')
        
        for channel in hourly_channels:
            status = "✅" if channel['configured'] else "⚠️"
            print(f"   {status} {channel['channel_name']:30} | {channel['description'][:35]}")
        
        print("\n🎯 Custom Channels:")
        custom_channels = self.router.custom_channels
        
        if custom_channels:
            for name, info in custom_channels.items():
                enabled = info.get('enabled', False)
                status = "✅" if enabled else "❌"
                condition = info.get('condition', 'N/A')
                print(f"   {status} {info.get('channel_name', name):30} | Condition: {condition[:30]}")
        else:
            print("   No custom channels configured")
        
        # Summary
        total_daily = len(daily_channels)
        configured_daily = sum(1 for c in daily_channels if c['configured'])
        total_hourly = len(hourly_channels)
        configured_hourly = sum(1 for c in hourly_channels if c['configured'])
        
        print(f"\n📊 Configuration Summary:")
        print(f"   Daily Channels: {configured_daily}/{total_daily} configured")
        print(f"   Hourly Channels: {configured_hourly}/{total_hourly} configured")
        print(f"   Custom Channels: {len(custom_channels)} defined")
        
        print("\n" + "=" * 70)


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Check Discord routing for alerts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check routing for a specific alert
  python check_alert_routing.py --alert ALERT_ID
  
  # Check routing for all alerts (grouped by channel)
  python check_alert_routing.py --all
  
  # Check where a ticker would be routed
  python check_alert_routing.py --ticker AAPL
  
  # Show all Discord channel configuration
  python check_alert_routing.py --config
        """
    )
    
    parser.add_argument(
        '--alert',
        type=str,
        metavar='ALERT_ID',
        help='Check routing for a specific alert'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Show routing summary for all alerts'
    )
    
    parser.add_argument(
        '--ticker',
        type=str,
        metavar='TICKER',
        help='Check routing for a specific ticker'
    )
    
    parser.add_argument(
        '--config',
        action='store_true',
        help='Show Discord channel configuration'
    )
    
    args = parser.parse_args()
    
    checker = AlertRoutingChecker()
    
    # Handle commands
    if args.alert:
        checker.check_alert(args.alert)
        return 0
    
    elif args.all:
        checker.check_all_alerts()
        return 0
    
    elif args.ticker:
        checker.check_ticker_routing(args.ticker)
        return 0
    
    elif args.config:
        checker.show_channel_config()
        return 0
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

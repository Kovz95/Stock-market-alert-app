#!/usr/bin/env python3
"""
Alert Trigger Test Script
Tests the full alert triggering flow including Discord notifications

This script provides multiple testing strategies:
1. Test a specific alert with current market data
2. Temporarily override alert conditions to force a trigger
3. Create a temporary test alert
4. Test Discord routing for different asset types
"""

import logging
import sys
from datetime import datetime
from typing import Dict, Any, Optional

from stock_alert_checker import StockAlertChecker
from futures_alert_checker import FuturesAlertChecker
from data_access.alert_repository import list_alerts, update_alert, create_alert
from src.services.discord_routing import send_economy_discord_alert, discord_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AlertTriggerTester:
    """Test alert triggering and Discord notification functionality"""
    
    def __init__(self):
        self.stock_checker = StockAlertChecker()
        self.futures_checker = FuturesAlertChecker()
        self.router = discord_router
    
    def _show_routing_info(self, alert: Dict[str, Any]) -> None:
        """Show Discord routing information for an alert"""
        try:
            channel_name, webhook_url = self.router.determine_alert_channel(alert)
            
            # Check custom channels
            custom_channels = self.router.get_custom_channels_for_alert(alert)
            
            logger.info(f"\n📢 Discord Routing:")
            logger.info(f"   Primary Channel: {channel_name}")
            
            if custom_channels:
                logger.info(f"   Additional Custom Channels: {len(custom_channels)}")
                for custom_name, custom_info in custom_channels:
                    logger.info(f"      • {custom_info.get('channel_name', custom_name)}")
            
            logger.info("")  # Blank line for readability
        except Exception as e:
            logger.debug(f"Could not determine routing: {e}")
    
    def list_available_alerts(self) -> None:
        """List all available alerts for testing"""
        logger.info("=" * 60)
        logger.info("AVAILABLE ALERTS FOR TESTING")
        logger.info("=" * 60)
        
        try:
            alerts = list_alerts()
            
            if not alerts:
                logger.warning("No alerts found in the system")
                return
            
            for i, alert in enumerate(alerts, 1):
                alert_id = alert.get('alert_id', 'N/A')
                ticker = alert.get('ticker', alert.get('ticker1', 'N/A'))
                name = alert.get('name', alert.get('stock_name', 'Unnamed'))
                status = alert.get('action', 'on')
                timeframe = alert.get('timeframe', 'daily')
                
                # Get conditions
                conditions = []
                if 'conditions' in alert:
                    cond_list = alert.get('conditions', [])
                    for cond in cond_list[:2]:  # Show first 2
                        if isinstance(cond, dict):
                            conditions.append(cond.get('conditions', ''))
                        elif isinstance(cond, str):
                            conditions.append(cond)
                
                logger.info(f"\n{i}. Alert ID: {alert_id}")
                logger.info(f"   Ticker: {ticker}")
                logger.info(f"   Name: {name}")
                logger.info(f"   Status: {status}")
                logger.info(f"   Timeframe: {timeframe}")
                if conditions:
                    logger.info(f"   Conditions: {conditions[0][:80]}...")
                logger.info("-" * 60)
                
        except Exception as e:
            logger.error(f"Error listing alerts: {e}")
    
    def test_specific_alert(self, alert_id: str) -> bool:
        """
        Test a specific alert with current market data
        
        Args:
            alert_id: ID of the alert to test
            
        Returns:
            True if alert triggered, False otherwise
        """
        logger.info("=" * 60)
        logger.info(f"TESTING ALERT: {alert_id}")
        logger.info("=" * 60)
        
        try:
            # Load the specific alert
            alerts = list_alerts()
            alert = None
            
            for a in alerts:
                if a.get('alert_id') == alert_id:
                    alert = a
                    break
            
            if not alert:
                logger.error(f"Alert {alert_id} not found")
                return False
            
            # Display alert info
            ticker = alert.get('ticker', alert.get('ticker1', 'N/A'))
            name = alert.get('name', alert.get('stock_name', 'Unnamed'))
            action = alert.get('action', 'on')
            
            logger.info(f"Alert: {name}")
            logger.info(f"Ticker: {ticker}")
            logger.info(f"Status: {action}")
            
            # Warn if explicitly disabled
            if action == 'off':
                logger.warning(f"\n⚠️  WARNING: This alert is currently DISABLED (action='off')")
                logger.warning(f"For override testing, we'll temporarily enable it to test Discord delivery\n")
            
            # Check if it's a futures alert
            is_futures = alert.get('alert_type') == 'futures' or 'adjustment_method' in alert
            
            if is_futures:
                logger.info("Alert Type: Futures")
                result = self._test_futures_alert(alert)
            else:
                logger.info("Alert Type: Stock")
                result = self._test_stock_alert(alert)
            
            return result
            
        except Exception as e:
            logger.error(f"Error testing alert: {e}")
            return False
    
    def _test_stock_alert(self, alert: Dict[str, Any]) -> bool:
        """Test a stock alert"""
        # Show routing info before testing
        self._show_routing_info(alert)
        
        result = self.stock_checker.check_alert(alert)
        
        if result.get('triggered'):
            logger.info("✅ ALERT TRIGGERED!")
            logger.info(f"Discord notification sent: {result.get('discord_sent', 'Unknown')}")
            return True
        elif result.get('error'):
            logger.warning(f"❌ Error: {result.get('error')}")
            return False
        elif result.get('skipped'):
            logger.info(f"⏭️ Alert skipped: {result.get('skip_reason', 'Unknown')}")
            return False
        else:
            logger.info("⏸️ Conditions not met - alert did not trigger")
            return False
    
    def _test_futures_alert(self, alert: Dict[str, Any]) -> bool:
        """Test a futures alert"""
        triggered = self.futures_checker.check_alert(alert)
        
        if triggered:
            logger.info("✅ ALERT TRIGGERED!")
            
            # Get price data for notification
            symbol = alert.get('ticker', '')
            df = self.futures_checker.get_price_data(
                symbol,
                days=1,
                adjustment_method=alert.get('adjustment_method', 'none')
            )
            
            if df is not None:
                symbol_data = {'close': df['close'].iloc[-1]}
                success = self.futures_checker.send_discord_notification(alert, symbol_data)
                logger.info(f"Discord notification sent: {success}")
            
            return True
        else:
            logger.info("⏸️ Conditions not met - alert did not trigger")
            return False
    
    def test_with_condition_override(
        self,
        alert_id: str,
        test_condition: str = "Close[-1] > 0"
    ) -> bool:
        """
        Test an alert by temporarily overriding its condition with one that will trigger
        
        Args:
            alert_id: ID of the alert to test
            test_condition: Condition that will definitely trigger (default: "Close[-1] > 0")
            
        Returns:
            True if test successful, False otherwise
        """
        logger.info("=" * 60)
        logger.info(f"TESTING WITH CONDITION OVERRIDE: {alert_id}")
        logger.info("=" * 60)
        
        try:
            # Load the alert
            alerts = list_alerts()
            alert = None
            original_conditions = None
            
            for a in alerts:
                if a.get('alert_id') == alert_id:
                    alert = a.copy()
                    original_conditions = a.get('conditions', []).copy() if 'conditions' in a else []
                    break
            
            if not alert:
                logger.error(f"Alert {alert_id} not found")
                return False
            
            # Override conditions and enable alert for testing
            logger.info(f"Original conditions: {original_conditions}")
            logger.info(f"Test condition: {test_condition}")
            
            # Temporarily modify the alert conditions and enable it
            alert['conditions'] = [{"conditions": test_condition}]
            alert['action'] = 'on'  # Temporarily enable for testing
            
            # Test the alert
            ticker = alert.get('ticker', alert.get('ticker1', 'N/A'))
            name = alert.get('name', alert.get('stock_name', 'Unnamed'))
            
            logger.info(f"\nTesting alert: {name} ({ticker})")
            logger.info("This will send a real Discord notification if successful!\n")
            
            # Check if it's a futures alert
            is_futures = alert.get('alert_type') == 'futures' or 'adjustment_method' in alert
            
            if is_futures:
                result = self._test_futures_alert(alert)
            else:
                result = self._test_stock_alert(alert)
            
            logger.info("\n⚠️ Note: Original alert was NOT modified in the database")
            logger.info("This was a temporary override for testing purposes only")
            logger.info("The alert's actual conditions and status remain unchanged")
            
            return result
            
        except Exception as e:
            logger.error(f"Error testing with condition override: {e}")
            return False
    
    def create_test_alert(
        self,
        ticker: str = "AAPL",
        condition: str = "Close[-1] > 0"
    ) -> Optional[str]:
        """
        Create a temporary test alert that will definitely trigger
        
        Args:
            ticker: Stock ticker to test with
            condition: Condition that will trigger
            
        Returns:
            Alert ID if created successfully, None otherwise
        """
        logger.info("=" * 60)
        logger.info("CREATING TEST ALERT")
        logger.info("=" * 60)
        
        try:
            test_alert = {
                "ticker": ticker,
                "name": f"TEST ALERT - {ticker} - {datetime.now().strftime('%H:%M:%S')}",
                "stock_name": f"Test Alert for {ticker}",
                "conditions": [{"conditions": condition}],
                "combination_logic": "AND",
                "timeframe": "1d",
                "action": "on",
                "created_at": datetime.now().isoformat(),
            }
            
            logger.info(f"Creating test alert for {ticker}")
            logger.info(f"Condition: {condition}")
            
            # Create the alert
            result = create_alert(test_alert)
            
            if result and result.get('alert_id'):
                alert_id = result['alert_id']
                logger.info(f"✅ Test alert created with ID: {alert_id}")
                logger.info(f"\nYou can now test it with:")
                logger.info(f"  python test_alert_trigger.py --test-alert {alert_id}")
                return alert_id
            else:
                logger.error("Failed to create test alert")
                return None
                
        except Exception as e:
            logger.error(f"Error creating test alert: {e}")
            return None
    
    def test_discord_routing(self, ticker: str = "AAPL") -> bool:
        """
        Test Discord routing for a specific ticker without evaluating conditions
        
        Args:
            ticker: Stock ticker to test routing for
            
        Returns:
            True if message sent successfully, False otherwise
        """
        logger.info("=" * 60)
        logger.info(f"TESTING DISCORD ROUTING: {ticker}")
        logger.info("=" * 60)
        
        try:
            # Create a test alert
            test_alert = {
                "ticker": ticker,
                "name": f"Discord Routing Test - {ticker}",
                "stock_name": f"Test for {ticker}",
                "timeframe": "1d",
            }
            
            # Create a test message
            test_message = f"""🧪 **Discord Routing Test**

**Ticker:** `{ticker}`
**Test Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This is a test message to verify Discord routing is working correctly.
If you see this message, the alert system can successfully send notifications to Discord!"""
            
            logger.info(f"Sending test message for {ticker}...")
            
            # Send the message
            success = send_economy_discord_alert(test_alert, test_message)
            
            if success:
                logger.info("✅ Test message sent successfully!")
                logger.info("Check your Discord channel to verify receipt")
                return True
            else:
                logger.error("❌ Failed to send test message")
                return False
                
        except Exception as e:
            logger.error(f"Error testing Discord routing: {e}")
            return False


def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test alert triggering and Discord notifications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available alerts
  python test_alert_trigger.py --list
  
  # Test a specific alert with current market data
  python test_alert_trigger.py --test-alert ALERT_ID
  
  # Test with a condition override (always triggers)
  python test_alert_trigger.py --test-override ALERT_ID
  
  # Create a test alert that will trigger
  python test_alert_trigger.py --create-test AAPL
  
  # Test Discord routing without condition evaluation
  python test_alert_trigger.py --test-discord AAPL
        """
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all available alerts'
    )
    
    parser.add_argument(
        '--test-alert',
        type=str,
        metavar='ALERT_ID',
        help='Test a specific alert with current market data'
    )
    
    parser.add_argument(
        '--test-override',
        type=str,
        metavar='ALERT_ID',
        help='Test an alert with condition override (will trigger)'
    )
    
    parser.add_argument(
        '--override-condition',
        type=str,
        default='Close[-1] > 0',
        help='Custom condition for override test (default: "Close[-1] > 0")'
    )
    
    parser.add_argument(
        '--create-test',
        type=str,
        metavar='TICKER',
        help='Create a test alert for a specific ticker'
    )
    
    parser.add_argument(
        '--test-condition',
        type=str,
        default='Close[-1] > 0',
        help='Condition for test alert (default: "Close[-1] > 0")'
    )
    
    parser.add_argument(
        '--test-discord',
        type=str,
        metavar='TICKER',
        help='Test Discord routing for a ticker'
    )
    
    args = parser.parse_args()
    
    tester = AlertTriggerTester()
    
    # Handle commands
    if args.list:
        tester.list_available_alerts()
        return 0
    
    elif args.test_alert:
        success = tester.test_specific_alert(args.test_alert)
        return 0 if success else 1
    
    elif args.test_override:
        success = tester.test_with_condition_override(
            args.test_override,
            args.override_condition
        )
        return 0 if success else 1
    
    elif args.create_test:
        alert_id = tester.create_test_alert(
            args.create_test,
            args.test_condition
        )
        return 0 if alert_id else 1
    
    elif args.test_discord:
        success = tester.test_discord_routing(args.test_discord)
        return 0 if success else 1
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

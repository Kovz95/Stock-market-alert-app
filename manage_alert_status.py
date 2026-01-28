#!/usr/bin/env python3
"""
Alert Status Manager
Quick utility to check and toggle alert enabled/disabled status
"""

import sys
from typing import Optional

from data_access.alert_repository import list_alerts, get_alert, update_alert


def show_alert_status(alert_id: str) -> None:
    """Show the status of a specific alert"""
    alert = get_alert(alert_id)
    
    if not alert:
        print(f"❌ Alert {alert_id} not found")
        return
    
    name = alert.get('name', alert.get('stock_name', 'Unnamed'))
    ticker = alert.get('ticker', alert.get('ticker1', 'N/A'))
    action = alert.get('action', 'on')
    
    print("=" * 60)
    print(f"Alert: {name}")
    print(f"ID: {alert_id}")
    print(f"Ticker: {ticker}")
    print(f"Status: {'✅ ENABLED' if action == 'on' else '❌ DISABLED'}")
    print("=" * 60)


def enable_alert(alert_id: str) -> bool:
    """Enable an alert"""
    alert = get_alert(alert_id)
    
    if not alert:
        print(f"❌ Alert {alert_id} not found")
        return False
    
    if alert.get('action') == 'on':
        print(f"✅ Alert is already enabled")
        return True
    
    result = update_alert(alert_id, {'action': 'on'})
    
    if result:
        print(f"✅ Alert enabled successfully")
        return True
    else:
        print(f"❌ Failed to enable alert")
        return False


def disable_alert(alert_id: str) -> bool:
    """Disable an alert"""
    alert = get_alert(alert_id)
    
    if not alert:
        print(f"❌ Alert {alert_id} not found")
        return False
    
    if alert.get('action') != 'on':
        print(f"ℹ️  Alert is already disabled")
        return True
    
    result = update_alert(alert_id, {'action': 'off'})
    
    if result:
        print(f"✅ Alert disabled successfully")
        return True
    else:
        print(f"❌ Failed to disable alert")
        return False


def list_disabled_alerts() -> None:
    """List all disabled alerts"""
    alerts = list_alerts()
    
    disabled = [a for a in alerts if a.get('action') != 'on']
    
    print("=" * 70)
    print(f"DISABLED ALERTS ({len(disabled)} total)")
    print("=" * 70)
    
    if not disabled:
        print("✅ All alerts are enabled!")
        return
    
    for alert in disabled:
        alert_id = alert.get('alert_id', 'N/A')
        name = alert.get('name', alert.get('stock_name', 'Unnamed'))
        ticker = alert.get('ticker', alert.get('ticker1', 'N/A'))
        
        print(f"\n📛 {name}")
        print(f"   ID: {alert_id}")
        print(f"   Ticker: {ticker}")
        print(f"   Status: {alert.get('action', 'unknown')}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Manage alert enabled/disabled status",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check if an alert is enabled
  python manage_alert_status.py --status ALERT_ID
  
  # Enable an alert
  python manage_alert_status.py --enable ALERT_ID
  
  # Disable an alert
  python manage_alert_status.py --disable ALERT_ID
  
  # List all disabled alerts
  python manage_alert_status.py --list-disabled
        """
    )
    
    parser.add_argument(
        '--status',
        type=str,
        metavar='ALERT_ID',
        help='Check status of an alert'
    )
    
    parser.add_argument(
        '--enable',
        type=str,
        metavar='ALERT_ID',
        help='Enable an alert'
    )
    
    parser.add_argument(
        '--disable',
        type=str,
        metavar='ALERT_ID',
        help='Disable an alert'
    )
    
    parser.add_argument(
        '--list-disabled',
        action='store_true',
        help='List all disabled alerts'
    )
    
    args = parser.parse_args()
    
    if args.status:
        show_alert_status(args.status)
        return 0
    
    elif args.enable:
        success = enable_alert(args.enable)
        return 0 if success else 1
    
    elif args.disable:
        success = disable_alert(args.disable)
        return 0 if success else 1
    
    elif args.list_disabled:
        list_disabled_alerts()
        return 0
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())

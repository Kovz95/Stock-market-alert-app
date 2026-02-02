#!/usr/bin/env python3
"""
Manual test script for futures scheduler.

This script helps test the futures scheduler in a real environment
before deploying to production.

Usage:
    python scripts/analysis/test_futures_scheduler.py --help
    python scripts/analysis/test_futures_scheduler.py --check-config
    python scripts/analysis/test_futures_scheduler.py --test-job
    python scripts/analysis/test_futures_scheduler.py --test-notifications
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import argparse
import json

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

# Add src/data_access to path for legacy db_config imports
sys.path.append(str(BASE_DIR / "src" / "data_access"))

from src.services import futures_scheduler
from src.services.futures_alert_checker import FuturesAlertChecker


def check_config():
    """Check scheduler configuration."""
    print("=" * 60)
    print("CHECKING FUTURES SCHEDULER CONFIGURATION")
    print("=" * 60)

    config = futures_scheduler.load_scheduler_config()

    print("\n✅ Configuration loaded successfully")
    print("\nConfiguration:")
    print(json.dumps(config, indent=2))

    # Check required fields
    required = ["update_times", "ib_hours", "enabled"]
    missing = [field for field in required if field not in config]

    if missing:
        print(f"\n⚠️  Warning: Missing fields: {', '.join(missing)}")
    else:
        print("\n✅ All required fields present")

    # Check IB hours
    if "ib_hours" in config:
        ib_hours = config["ib_hours"]
        print(f"\n📅 IB Trading Hours: {ib_hours.get('start', 'N/A')} - {ib_hours.get('end', 'N/A')} UTC")

        # Check if currently within IB hours
        is_available = futures_scheduler.is_ib_available()
        print(f"   Current time within IB hours: {'Yes ✅' if is_available else 'No ❌'}")

    # Check update times
    if "update_times" in config:
        update_times = config["update_times"]
        print(f"\n⏰ Scheduled Update Times (UTC): {', '.join(update_times)}")

    # Check webhook configuration
    if "scheduler_webhook" in config:
        webhook = config["scheduler_webhook"]
        webhook_enabled = webhook.get("enabled", False)
        webhook_url = webhook.get("url", "")
        print(f"\n🔔 Discord Notifications: {'Enabled ✅' if webhook_enabled else 'Disabled ❌'}")
        if webhook_enabled:
            print(f"   Webhook URL: {webhook_url[:50]}..." if len(webhook_url) > 50 else f"   Webhook URL: {webhook_url}")

    return True


def test_notifications():
    """Test Discord notification sending."""
    print("=" * 60)
    print("TESTING DISCORD NOTIFICATIONS")
    print("=" * 60)

    config = futures_scheduler.load_scheduler_config()
    webhook_config = config.get("scheduler_webhook", {})

    if not webhook_config.get("enabled"):
        print("\n❌ Discord notifications are disabled in config")
        print("   Enable them by setting scheduler_webhook.enabled = true")
        return False

    if not webhook_config.get("url"):
        print("\n❌ No webhook URL configured")
        print("   Set scheduler_webhook.url in config")
        return False

    print(f"\n📡 Sending test notification to webhook...")

    test_message = f"""🧪 **Test Notification**

This is a test notification from the futures scheduler.

• Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC
• Status: Testing notification system

If you see this message, notifications are working! ✅"""

    success = futures_scheduler.send_scheduler_notification(test_message, event="test")

    if success:
        print("✅ Test notification sent successfully!")
        print("   Check your Discord channel to verify delivery")
    else:
        print("❌ Failed to send test notification")
        print("   Check your webhook URL and network connection")

    return success


def test_price_update():
    """Test futures price update."""
    print("=" * 60)
    print("TESTING FUTURES PRICE UPDATE")
    print("=" * 60)

    # Check IB availability
    if not futures_scheduler.is_ib_available():
        print("\n⚠️  Warning: Current time is outside IB hours")
        print("   Price update will be skipped")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return False

    print("\n🔄 Running futures price update...")
    print("   (This may take several minutes)")

    result = futures_scheduler.run_price_update()

    print("\n📊 Price Update Results:")
    print(f"   Updated: {result.get('updated', 0)}")
    print(f"   Failed: {result.get('failed', 0)}")

    if result.get('error'):
        print(f"   ❌ Error: {result['error']}")
        return False
    else:
        print("   ✅ Price update completed successfully")
        return True


def test_alert_checking():
    """Test futures alert checking."""
    print("=" * 60)
    print("TESTING FUTURES ALERT CHECKING")
    print("=" * 60)

    print("\n🔍 Running futures alert check...")

    result = futures_scheduler.run_alert_checks()

    print("\n📊 Alert Check Results:")
    print(f"   Total Alerts: {result.get('total', 0)}")
    print(f"   Triggered: {result.get('triggered', 0)}")
    print(f"   Errors: {result.get('errors', 0)}")
    print(f"   Skipped: {result.get('skipped', 0)}")
    print(f"   No Data: {result.get('no_data', 0)}")

    if result.get('errors', 0) > 0:
        print("   ⚠️  Some errors occurred during alert checking")
    else:
        print("   ✅ Alert check completed successfully")

    return result.get('errors', 0) == 0


def test_full_job():
    """Test complete job execution."""
    print("=" * 60)
    print("TESTING COMPLETE JOB EXECUTION")
    print("=" * 60)

    # Check IB availability
    if not futures_scheduler.is_ib_available():
        print("\n⚠️  Warning: Current time is outside IB hours")
        print("   Job will be skipped by the scheduler")
        response = input("\nContinue anyway? (y/n): ")
        if response.lower() != 'y':
            return False

    print("\n🚀 Executing complete futures job...")
    print("   (This includes price update + alert check)")
    print("   (This may take several minutes)")

    # Note: This will run in subprocess isolation
    result = futures_scheduler.execute_futures_job()

    if result is None:
        print("\n❌ Job execution failed or was skipped")
        print("   Check logs for details")
        return False

    print("\n✅ Job completed successfully!")
    print("\n📊 Results:")
    print(f"\nPrice Update:")
    price_stats = result.get('price_stats', {})
    print(f"   Updated: {price_stats.get('updated', 0)}")
    print(f"   Failed: {price_stats.get('failed', 0)}")

    print(f"\nAlert Check:")
    alert_stats = result.get('alert_stats', {})
    print(f"   Total: {alert_stats.get('total', 0)}")
    print(f"   Triggered: {alert_stats.get('triggered', 0)}")
    print(f"   Errors: {alert_stats.get('errors', 0)}")

    print(f"\nDuration: {result.get('duration', 0):.1f} seconds")

    return True


def check_status():
    """Check scheduler status."""
    print("=" * 60)
    print("CHECKING SCHEDULER STATUS")
    print("=" * 60)

    # Check if running
    is_running = futures_scheduler.is_scheduler_running()
    print(f"\nScheduler Running: {'Yes ✅' if is_running else 'No ❌'}")

    # Get status info
    status = futures_scheduler.get_scheduler_info()

    if status:
        print("\n📊 Status Information:")
        print(json.dumps(status, indent=2))
    else:
        print("\n⚠️  No status information available")
        print("   Scheduler may not be running or status file missing")

    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test futures scheduler functionality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/analysis/test_futures_scheduler.py --check-config
  python scripts/analysis/test_futures_scheduler.py --test-notifications
  python scripts/analysis/test_futures_scheduler.py --test-job
  python scripts/analysis/test_futures_scheduler.py --all
        """,
    )

    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Check scheduler configuration",
    )
    parser.add_argument(
        "--test-notifications",
        action="store_true",
        help="Test Discord notification sending",
    )
    parser.add_argument(
        "--test-price-update",
        action="store_true",
        help="Test futures price update",
    )
    parser.add_argument(
        "--test-alerts",
        action="store_true",
        help="Test futures alert checking",
    )
    parser.add_argument(
        "--test-job",
        action="store_true",
        help="Test complete job execution (price + alerts)",
    )
    parser.add_argument(
        "--check-status",
        action="store_true",
        help="Check scheduler status",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all tests (non-destructive)",
    )

    args = parser.parse_args()

    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        return 0

    results = []

    try:
        if args.all or args.check_config:
            results.append(("Configuration Check", check_config()))

        if args.all or args.check_status:
            results.append(("Status Check", check_status()))

        if args.all or args.test_notifications:
            results.append(("Notification Test", test_notifications()))

        if args.test_price_update:
            results.append(("Price Update Test", test_price_update()))

        if args.test_alerts:
            results.append(("Alert Check Test", test_alert_checking()))

        if args.test_job:
            results.append(("Full Job Test", test_full_job()))

        # Print summary
        if results:
            print("\n" + "=" * 60)
            print("TEST SUMMARY")
            print("=" * 60)

            for name, success in results:
                status = "✅ PASS" if success else "❌ FAIL"
                print(f"{name}: {status}")

            all_passed = all(success for _, success in results)
            if all_passed:
                print("\n✅ All tests passed!")
                return 0
            else:
                print("\n❌ Some tests failed")
                return 1

    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

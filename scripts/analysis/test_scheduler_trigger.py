#!/usr/bin/env python3
"""
Test Scheduler Trigger Script
Manually trigger daily/weekly scheduler jobs to test alert evaluation

This script allows you to:
1. Test daily alert checks for a specific exchange
2. Test weekly alert checks for a specific exchange
3. Run in dry-run mode to see what would happen without sending Discord notifications
4. Test with a specific alert ID
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Hashable

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from data_access.alert_repository import list_alerts, refresh_alert_cache
from data_access.metadata_repository import fetch_stock_metadata_df
from scheduled_price_updater import update_prices_for_exchanges
from stock_alert_checker import StockAlertChecker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_alerts_for_exchange(
    exchanges: List[str],
    timeframe_key: str = "daily",
    include_disabled: bool = False
) -> List[Dict[str, Any]]:
    """
    Get alerts that would be checked for the given exchanges.

    Args:
        exchanges: List of exchange names
        timeframe_key: 'daily' or 'weekly'
        include_disabled: If True, also show disabled alerts

    Returns:
        List of matching alerts
    """
    # Get metadata to filter by exchange (optional - alerts can also have exchange field directly)
    metadata_df = fetch_stock_metadata_df()
    exchange_tickers = set()

    if metadata_df is not None and not metadata_df.empty:
        # Get tickers for these exchanges from metadata
        for exchange in exchanges:
            exchange_df = metadata_df[metadata_df["exchange"] == exchange]
            if not exchange_df.empty and "symbol" in exchange_df.columns:
                exchange_tickers.update(exchange_df["symbol"].tolist())
        logger.info(f"Found {len(exchange_tickers)} tickers for {exchanges} from metadata")
    else:
        logger.warning("No metadata available - will match alerts by exchange field only")

    # Get all alerts
    all_alerts = list_alerts()
    logger.info(f"Total alerts in system: {len(all_alerts)}")

    # Show summary of all alerts for debugging
    if all_alerts:
        exchange_summary = {}
        for a in all_alerts:
            ex = a.get("exchange", "Unknown")
            exchange_summary[ex] = exchange_summary.get(ex, 0) + 1
        logger.info(f"Alerts by exchange: {exchange_summary}")
    relevant_alerts = []

    for alert in all_alerts:
        if not isinstance(alert, dict):
            continue

        alert_ticker = alert.get("ticker")
        alert_exchange = alert.get("exchange")
        alert_timeframe = alert.get("timeframe", "daily")
        alert_action = alert.get("action", "on")

        # Debug: show what we're checking
        logger.debug(f"Alert: {alert.get('name')} - exchange={alert_exchange}, timeframe={alert_timeframe}, action={alert_action}")

        # Skip disabled alerts unless include_disabled is True
        if alert_action == "off" and not include_disabled:
            logger.debug(f"  -> Skipped (disabled)")
            continue

        # Check if alert matches the exchange filter
        # Match by: 1) alert's exchange field, OR 2) ticker in metadata for the exchange
        exchange_match = alert_exchange in exchanges
        ticker_match = alert_ticker in exchange_tickers

        if exchange_match or ticker_match:
            # Check if alert matches the timeframe
            timeframe_normalized = alert_timeframe.lower() if alert_timeframe else "daily"
            if timeframe_key == "weekly" and timeframe_normalized in ("weekly", "1wk"):
                relevant_alerts.append(alert)
                logger.debug(f"  -> MATCHED (weekly)")
            elif timeframe_key == "daily" and timeframe_normalized in ("daily", "1d"):
                relevant_alerts.append(alert)
                logger.debug(f"  -> MATCHED (daily)")
            else:
                logger.debug(f"  -> Timeframe mismatch: {timeframe_normalized} vs {timeframe_key}")
        else:
            logger.debug(f"  -> Exchange mismatch: {alert_exchange} not in {exchanges}")

    return relevant_alerts


def list_exchanges_with_alerts() -> Dict[str, Dict[str, int]]:
    """List all exchanges that have alerts configured."""
    alerts = list_alerts()

    exchange_counts = {}

    for alert in alerts:
        if not isinstance(alert, dict):
            continue

        exchange = alert.get("exchange", "Unknown")
        timeframe = alert.get("timeframe", "daily").lower()
        action = alert.get("action", "on")

        if exchange not in exchange_counts:
            exchange_counts[exchange] = {"daily": 0, "weekly": 0, "disabled": 0}

        if action == "off":
            exchange_counts[exchange]["disabled"] += 1
        elif timeframe in ("weekly", "1wk"):
            exchange_counts[exchange]["weekly"] += 1
        else:
            exchange_counts[exchange]["daily"] += 1

    return exchange_counts


def run_scheduler_job(
    exchange: str,
    job_type: str = "daily",
    dry_run: bool = False,
    skip_price_update: bool = False,
    alert_id: Optional[str] = None
) -> Dict[Hashable, Any]:
    """
    Run a scheduler job for testing.

    Args:
        exchange: Exchange name (e.g., 'NYSE', 'NASDAQ')
        job_type: 'daily' or 'weekly'
        dry_run: If True, don't send Discord notifications
        skip_price_update: If True, skip price updates (faster for testing)
        alert_id: If provided, only test this specific alert

    Returns:
        Results dictionary
    """
    results : Dict[Hashable, Any] = {
        "exchange": exchange,
        "job_type": job_type,
        "dry_run": dry_run,
        "started_at": datetime.now().isoformat(),
        "price_update": None,
        "alert_check": None,
        "errors": []
    }

    logger.info("=" * 70)
    logger.info(f"RUNNING {job_type.upper()} JOB FOR {exchange}")
    logger.info(f"Dry Run: {dry_run}")
    logger.info(f"Skip Price Update: {skip_price_update}")
    if alert_id:
        logger.info(f"Testing specific alert: {alert_id}")
    logger.info("=" * 70)

    # Step 1: Update prices (unless skipped)
    if not skip_price_update:
        logger.info("\n--- STEP 1: Updating Prices ---")
        try:
            resample_weekly = job_type == "weekly"
            price_stats = update_prices_for_exchanges([exchange], resample_weekly=resample_weekly)
            results["price_update"] = price_stats
            logger.info(f"Price update complete: {price_stats}")
        except Exception as e:
            error_msg = f"Price update failed: {e}"
            logger.error(error_msg)
            results["errors"].append(error_msg)
    else:
        logger.info("\n--- STEP 1: Skipping Price Update ---")

    # Step 2: Get alerts to check
    logger.info("\n--- STEP 2: Finding Alerts to Check ---")

    if alert_id:
        # Test specific alert
        all_alerts = list_alerts()
        alerts_to_check = [a for a in all_alerts if a.get("alert_id") == alert_id]
        if not alerts_to_check:
            error_msg = f"Alert {alert_id} not found"
            logger.error(error_msg)
            results["errors"].append(error_msg)
            return results
    else:
        alerts_to_check = get_alerts_for_exchange([exchange], job_type)

    logger.info(f"Found {len(alerts_to_check)} alerts to check")

    for alert in alerts_to_check:
        ticker = alert.get("ticker", alert.get("ticker1", "N/A"))
        name = alert.get("name", alert.get("stock_name", "Unnamed"))
        action = alert.get("action", "on")
        logger.info(f"  - {ticker}: {name} (status: {action})")

    if not alerts_to_check:
        logger.warning("No alerts found for this exchange/timeframe")
        results["alert_check"] = {"total": 0, "triggered": 0, "errors": 0}
        return results

    # Step 3: Check alerts
    logger.info("\n--- STEP 3: Checking Alerts ---")

    checker = StockAlertChecker()

    if dry_run:
        # In dry-run mode, manually evaluate conditions without sending notifications
        logger.info("DRY RUN MODE - Will evaluate conditions but NOT send Discord notifications")

        alert_results : Dict[Hashable, Any] = {
            "total": len(alerts_to_check),
            "triggered": 0,
            "errors": 0,
            "skipped": 0,
            "details": []
        }

        for alert in alerts_to_check:
            alert_id_str = alert.get("alert_id", "unknown")
            ticker = alert.get("ticker", alert.get("ticker1", "N/A"))
            name = alert.get("name", "Unnamed")
            action = alert.get("action", "on")

            logger.info(f"\nChecking alert: {name} ({ticker})")

            detail = {
                "alert_id": alert_id_str,
                "ticker": ticker,
                "name": name,
            }

            try:
                # Check if disabled
                if action == "off":
                    alert_results["skipped"] += 1
                    detail["result"] = {"skipped": True, "skip_reason": "disabled"}
                    logger.info(f"  -> Skipped: disabled")
                    alert_results["details"].append(detail)
                    continue

                # Check if already triggered today
                if checker.should_skip_alert(alert):
                    alert_results["skipped"] += 1
                    detail["result"] = {"skipped": True, "skip_reason": "already_triggered_today"}
                    logger.info(f"  -> Skipped: already triggered today")
                    alert_results["details"].append(detail)
                    continue

                # Get price data
                timeframe = alert.get("timeframe", "1d")
                df = checker.get_price_data(ticker, timeframe)

                if df is None or df.empty:
                    alert_results["errors"] += 1
                    detail["result"] = {"error": f"No price data for {ticker}"}
                    logger.warning(f"  -> Error: No price data for {ticker}")
                    alert_results["details"].append(detail)
                    continue

                # Evaluate conditions (this is the core logic, doesn't send notifications)
                triggered = checker.evaluate_alert(alert, df)

                if triggered:
                    alert_results["triggered"] += 1
                    message = checker.format_alert_message(alert, df)
                    detail["result"] = {"triggered": True, "message": message}
                    logger.info(f"  -> WOULD TRIGGER!")
                    logger.info(f"     Message preview: {message[:200]}...")
                else:
                    detail["result"] = {"triggered": False}
                    logger.info(f"  -> Conditions not met")

                alert_results["details"].append(detail)

            except Exception as e:
                alert_results["errors"] += 1
                detail["result"] = {"error": str(e)}
                logger.error(f"  -> Exception: {e}")
                alert_results["details"].append(detail)

        results["alert_check"] = alert_results

    else:
        # Full run - will send notifications
        logger.info("LIVE MODE - Will send Discord notifications if alerts trigger!")

        try:
            alert_stats = checker.check_alerts(alerts_to_check, job_type)
            results["alert_check"] = alert_stats
            logger.info(f"Alert check complete: {alert_stats}")
        except Exception as e:
            error_msg = f"Alert check failed: {e}"
            logger.error(error_msg)
            results["errors"].append(error_msg)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("JOB COMPLETE")
    logger.info("=" * 70)

    if results["price_update"]:
        logger.info(f"Price Updates: {results['price_update']}")

    if results["alert_check"]:
        ac = results["alert_check"]
        logger.info(f"Alerts Checked: {ac.get('total', 0)}")
        logger.info(f"Alerts Triggered: {ac.get('triggered', 0)}")
        logger.info(f"Alerts Skipped: {ac.get('skipped', 0)}")
        logger.info(f"Errors: {ac.get('errors', 0)}")

    if results["errors"]:
        logger.warning(f"Errors encountered: {results['errors']}")

    results["completed_at"] = datetime.now().isoformat()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Test scheduler job execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List exchanges with alerts
  python test_scheduler_trigger.py --list-exchanges

  # Dry run: See what would happen for NYSE daily alerts (no notifications)
  python test_scheduler_trigger.py --exchange NYSE --type daily --dry-run

  # Dry run for weekly alerts
  python test_scheduler_trigger.py --exchange NYSE --type weekly --dry-run

  # Full run: Actually trigger alerts and send notifications
  python test_scheduler_trigger.py --exchange NYSE --type daily

  # Test a specific alert (dry run)
  python test_scheduler_trigger.py --alert-id YOUR_ALERT_ID --dry-run

  # Skip price update for faster testing
  python test_scheduler_trigger.py --exchange NYSE --dry-run --skip-price-update
        """
    )

    parser.add_argument(
        '--list-exchanges',
        action='store_true',
        help='List all exchanges that have alerts configured'
    )

    parser.add_argument(
        '--exchange',
        type=str,
        help='Exchange to test (e.g., NYSE, NASDAQ, LONDON)'
    )

    parser.add_argument(
        '--type',
        type=str,
        choices=['daily', 'weekly'],
        default='daily',
        help='Job type: daily or weekly (default: daily)'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run mode - evaluate conditions but do NOT send Discord notifications'
    )

    parser.add_argument(
        '--skip-price-update',
        action='store_true',
        help='Skip price updates (faster for testing alert logic)'
    )

    parser.add_argument(
        '--alert-id',
        type=str,
        help='Test a specific alert by ID'
    )

    parser.add_argument(
        '--show-alerts',
        action='store_true',
        help='Show alerts that would be checked for the exchange'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose debug output'
    )

    parser.add_argument(
        '--no-cache-refresh',
        action='store_true',
        help='Skip refreshing the alert cache (use cached data)'
    )

    args = parser.parse_args()

    # Set debug logging if verbose
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)

    # Refresh alert cache by default (unless --no-cache-refresh)
    if not args.no_cache_refresh:
        logger.info("Refreshing alert cache from database...")
        try:
            refresh_alert_cache()
            logger.info("Alert cache refreshed")
        except Exception as e:
            logger.warning(f"Failed to refresh cache: {e}")

    # List exchanges
    if args.list_exchanges:
        logger.info("=" * 70)
        logger.info("EXCHANGES WITH CONFIGURED ALERTS")
        logger.info("=" * 70)

        exchange_counts = list_exchanges_with_alerts()

        if not exchange_counts:
            logger.warning("No alerts found in the system")
            return 1

        logger.info(f"\n{'Exchange':<25} {'Daily':<10} {'Weekly':<10} {'Disabled':<10}")
        logger.info("-" * 55)

        for exchange in sorted(exchange_counts.keys()):
            counts = exchange_counts[exchange]
            logger.info(f"{exchange:<25} {counts['daily']:<10} {counts['weekly']:<10} {counts['disabled']:<10}")

        total_daily = sum(c['daily'] for c in exchange_counts.values())
        total_weekly = sum(c['weekly'] for c in exchange_counts.values())
        total_disabled = sum(c['disabled'] for c in exchange_counts.values())

        logger.info("-" * 55)
        logger.info(f"{'TOTAL':<25} {total_daily:<10} {total_weekly:<10} {total_disabled:<10}")

        return 0

    # Show alerts for exchange
    if args.show_alerts:
        if not args.exchange:
            logger.error("--exchange is required with --show-alerts")
            return 1

        logger.info("=" * 70)
        logger.info(f"ALERTS FOR {args.exchange} ({args.type})")
        logger.info("=" * 70)

        alerts = get_alerts_for_exchange([args.exchange], args.type, include_disabled=True)

        if not alerts:
            logger.warning(f"No {args.type} alerts found for {args.exchange}")
            return 1

        for alert in alerts:
            alert_id = alert.get("alert_id", "N/A")
            ticker = alert.get("ticker", alert.get("ticker1", "N/A"))
            name = alert.get("name", alert.get("stock_name", "Unnamed"))
            action = alert.get("action", "on")

            status_icon = "🟢" if action != "off" else "🔴"
            logger.info(f"\n{status_icon} {name}")
            logger.info(f"   ID: {alert_id}")
            logger.info(f"   Ticker: {ticker}")
            logger.info(f"   Status: {action}")

            # Show conditions preview
            conditions = alert.get("conditions", [])
            if conditions:
                cond = conditions[0] if isinstance(conditions[0], str) else conditions[0].get("conditions", "")
                logger.info(f"   Condition: {str(cond)[:80]}...")

        return 0

    # Run scheduler job
    if args.exchange or args.alert_id:
        exchange = args.exchange or "NYSE"  # Default for alert-id testing

        results = run_scheduler_job(
            exchange=exchange,
            job_type=args.type,
            dry_run=args.dry_run,
            skip_price_update=args.skip_price_update,
            alert_id=args.alert_id
        )

        # Return code based on results
        if results.get("errors"):
            return 1
        return 0

    # No action specified
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())

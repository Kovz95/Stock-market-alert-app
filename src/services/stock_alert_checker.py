"""
Stock Alert Checker - Evaluates stock alerts and sends Discord notifications

This module processes stock alerts by:
1. Loading alerts from the alert repository
2. Fetching current price data from FMP API
3. Evaluating conditions using backend.evaluate_expression()
4. Sending Discord notifications via discord_routing.send_economy_alert()
5. Logging to the audit trail
6. Updating alert status (last_triggered timestamp)
"""

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.services.backend import evaluate_expression, evaluate_expression_list
from src.services.backend_fmp import FMPDataFetcher
from src.data_access.alert_repository import list_alerts, update_alert
from src.services.discord_routing import send_economy_discord_alert
from src.services.alert_audit_logger import (
    log_alert_check_start,
    log_price_data_pulled,
    log_no_data_available,
    log_conditions_evaluated,
    log_error,
    log_completion,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockAlertChecker:
    """
    Evaluates stock alerts and sends Discord notifications when conditions are met.
    """

    def __init__(self):
        self.fmp = FMPDataFetcher()
        self.price_cache: Dict[str, pd.DataFrame] = {}

    def get_price_data(
        self,
        ticker: str,
        timeframe: str = "1d",
        days: int = 200
    ) -> Optional[pd.DataFrame]:
        """
        Get price data for a stock ticker.

        Args:
            ticker: Stock symbol (e.g., "AAPL", "MSFT")
            timeframe: Timeframe for the data ("1d", "1wk", "1h")
            days: Number of days of historical data to fetch

        Returns:
            DataFrame with OHLCV data, or None if unavailable
        """
        cache_key = f"{ticker}_{timeframe}"

        # Check cache first
        if cache_key in self.price_cache:
            logger.debug(f"Using cached data for {ticker}")
            return self.price_cache[cache_key]

        try:
            # Map timeframe to FMP period
            if timeframe in ("1wk", "weekly"):
                df = self.fmp.get_historical_data(ticker, period="1day", timeframe="1wk")
            elif timeframe in ("1h", "hourly"):
                df = self.fmp.get_historical_data(ticker, period="1hour")
            else:
                # Default to daily
                df = self.fmp.get_historical_data(ticker, period="1day")

            if df is not None and not df.empty:
                # Limit to requested number of days
                if len(df) > days:
                    df = df.tail(days)
                self.price_cache[cache_key] = df
                logger.debug(f"Fetched {len(df)} price records for {ticker}")
                return df
            else:
                logger.warning(f"No price data available for {ticker}")
                return None

        except Exception as e:
            logger.error(f"Error fetching price data for {ticker}: {e}")
            return None

    def extract_conditions(self, alert: Dict[str, Any]) -> List[str]:
        """
        Extract condition strings from an alert.

        Args:
            alert: Alert dictionary

        Returns:
            List of condition strings to evaluate
        """
        conditions = alert.get("conditions", [])
        result = []

        if isinstance(conditions, list):
            for cond in conditions:
                if isinstance(cond, dict):
                    # Handle dict format: {"conditions": "Close[-1] > 100"}
                    cond_str = cond.get("conditions", "")
                    if cond_str:
                        result.append(cond_str)
                elif isinstance(cond, str):
                    # Handle string format directly
                    result.append(cond)
                elif isinstance(cond, list) and len(cond) > 0:
                    # Handle nested list format
                    if isinstance(cond[0], str):
                        result.append(cond[0])

        return result

    def evaluate_alert(self, alert: Dict[str, Any], df: pd.DataFrame) -> bool:
        """
        Evaluate an alert's conditions against price data.

        Args:
            alert: Alert dictionary with conditions
            df: DataFrame with price data

        Returns:
            True if alert conditions are met, False otherwise
        """
        try:
            conditions = self.extract_conditions(alert)
            if not conditions:
                logger.debug(f"No conditions found for alert {alert.get('alert_id')}")
                return False

            combination = alert.get("combination_logic", "AND")
            vals = {"ticker": alert.get("ticker", "UNKNOWN")}

            # Use evaluate_expression_list for multiple conditions
            result = evaluate_expression_list(
                df, conditions, combination, vals=vals
            )
            return bool(result)

        except Exception as e:
            logger.error(f"Error evaluating alert {alert.get('alert_id')}: {e}")
            return False

    def format_alert_message(
        self,
        alert: Dict[str, Any],
        df: pd.DataFrame
    ) -> str:
        """
        Format a Discord notification message for a triggered alert.

        Args:
            alert: Alert dictionary
            df: DataFrame with price data

        Returns:
            Formatted message string
        """
        ticker = alert.get("ticker", "Unknown")
        name = alert.get("name", alert.get("stock_name", "Unnamed Alert"))
        current_price = df["Close"].iloc[-1] if not df.empty else 0
        timeframe = alert.get("timeframe", "daily")

        # Get conditions for display
        conditions = self.extract_conditions(alert)
        conditions_str = "\n".join(f"  • {c}" for c in conditions[:3])  # Show first 3
        if len(conditions) > 3:
            conditions_str += f"\n  ... and {len(conditions) - 3} more"

        message = f"""🚨 **Stock Alert Triggered**

**{name}**
• Ticker: `{ticker}`
• Price: ${current_price:.2f}
• Timeframe: {timeframe}

**Conditions Met:**
{conditions_str}

*Triggered at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"""

        return message

    def should_skip_alert(self, alert: Dict[str, Any]) -> bool:
        """
        Check if an alert should be skipped (already triggered today).

        Args:
            alert: Alert dictionary

        Returns:
            True if alert should be skipped, False otherwise
        """
        last_triggered = alert.get("last_triggered", "")
        if not last_triggered:
            return False

        try:
            # Parse the last triggered timestamp
            if isinstance(last_triggered, str):
                # Handle ISO format with or without timezone
                triggered_date = datetime.fromisoformat(
                    last_triggered.replace("Z", "+00:00")
                )
            else:
                triggered_date = last_triggered

            # Skip if already triggered today
            if triggered_date.date() == datetime.now().date():
                return True

        except (ValueError, AttributeError) as e:
            logger.debug(f"Could not parse last_triggered: {e}")

        return False

    def check_alert(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check a single alert and send notification if triggered.

        Args:
            alert: Alert dictionary

        Returns:
            Result dictionary with status information
        """
        start_time = time.time()
        alert_id = alert.get("alert_id", "unknown")
        ticker = alert.get("ticker", alert.get("ticker1", ""))

        result = {
            "alert_id": alert_id,
            "ticker": ticker,
            "triggered": False,
            "error": None,
            "skipped": False,
        }

        # Start audit logging
        audit_id = log_alert_check_start(alert, "scheduled")

        try:
            # Check if alert is disabled (only skip if explicitly set to "off")
            if alert.get("action", "on") == "off":
                result["skipped"] = True
                result["skip_reason"] = "disabled"
                return result

            # Check if already triggered today
            if self.should_skip_alert(alert):
                result["skipped"] = True
                result["skip_reason"] = "already_triggered_today"
                logger.debug(f"Alert {alert_id} already triggered today, skipping")
                return result

            # Get ticker - handle ratio alerts
            if alert.get("ratio") == "Yes" or alert.get("is_ratio"):
                ticker1 = alert.get("ticker1", "")
                ticker2 = alert.get("ticker2", "")
                if not ticker1 or not ticker2:
                    result["error"] = "Missing ticker1 or ticker2 for ratio alert"
                    if audit_id:
                        log_error(audit_id, result["error"])
                    return result
                ticker = ticker1  # Use ticker1 as primary for data fetch
            elif not ticker:
                result["error"] = "No ticker specified"
                if audit_id:
                    log_error(audit_id, result["error"])
                return result

            # Get price data
            timeframe = alert.get("timeframe", "1d")
            df = self.get_price_data(ticker, timeframe)

            if df is None or df.empty:
                result["error"] = f"No price data for {ticker}"
                if audit_id:
                    log_no_data_available(audit_id, ticker)
                return result

            if audit_id:
                log_price_data_pulled(audit_id, "FMP", cache_hit=ticker in self.price_cache)

            # Evaluate conditions
            triggered = self.evaluate_alert(alert, df)

            if audit_id:
                trigger_reason = "conditions_met" if triggered else None
                log_conditions_evaluated(audit_id, triggered, trigger_reason)

            if triggered:
                result["triggered"] = True
                logger.info(f"TRIGGERED: {ticker} - {alert.get('name', '')}")

                # Send Discord notification
                message = self.format_alert_message(alert, df)
                success = send_economy_discord_alert(alert, message)

                if success:
                    logger.info(f"Discord notification sent for {ticker}")
                else:
                    logger.warning(f"Failed to send Discord notification for {ticker}")

                # Update alert's last_triggered timestamp
                update_alert(alert_id, {"last_triggered": datetime.now().isoformat()})

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Error checking alert {alert_id}: {e}")
            if audit_id:
                log_error(audit_id, str(e))

        finally:
            # Log completion
            execution_time = int((time.time() - start_time) * 1000)
            if audit_id:
                log_completion(audit_id, execution_time)

        return result

    def check_alerts(
        self,
        alerts: List[Dict[str, Any]],
        timeframe_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check multiple alerts and return statistics.

        Args:
            alerts: List of alert dictionaries
            timeframe_filter: Optional filter for timeframe ("daily", "weekly")

        Returns:
            Statistics dictionary with check results
        """
        stats = {
            "total": 0,
            "triggered": 0,
            "errors": 0,
            "skipped": 0,
            "no_data": 0,
            "success": 0,
        }

        # Filter alerts by timeframe if specified
        filtered_alerts = []
        for alert in alerts:
            if not isinstance(alert, dict):
                continue

            alert_timeframe = alert.get("timeframe", "daily").lower()

            # Check timeframe filter
            if timeframe_filter:
                if timeframe_filter == "weekly" and alert_timeframe not in ("weekly", "1wk"):
                    continue
                elif timeframe_filter == "daily" and alert_timeframe not in ("daily", "1d"):
                    continue
                elif timeframe_filter == "hourly" and alert_timeframe not in ("hourly", "1h", "1hr"):
                    continue

            filtered_alerts.append(alert)

        stats["total"] = len(filtered_alerts)
        logger.info(f"Checking {len(filtered_alerts)} {timeframe_filter or 'all'} alerts")

        for alert in filtered_alerts:
            try:
                result = self.check_alert(alert)

                if result.get("skipped"):
                    stats["skipped"] += 1
                elif result.get("error"):
                    if "No price data" in str(result["error"]):
                        stats["no_data"] += 1
                    else:
                        stats["errors"] += 1
                elif result.get("triggered"):
                    stats["triggered"] += 1
                    stats["success"] += 1
                else:
                    stats["success"] += 1

            except Exception as e:
                logger.error(f"Unexpected error checking alert: {e}")
                stats["errors"] += 1

        logger.info(
            f"Alert check complete: {stats['triggered']} triggered, "
            f"{stats['errors']} errors, {stats['skipped']} skipped"
        )

        return stats

    def check_all_alerts(self, timeframe_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Load and check all active alerts.

        Args:
            timeframe_filter: Optional filter for timeframe ("daily", "weekly")

        Returns:
            Statistics dictionary with check results
        """
        logger.info("=" * 60)
        logger.info("STOCK ALERT CHECKER")
        logger.info("=" * 60)

        try:
            # Load all alerts from repository
            all_alerts = list_alerts()
            logger.info(f"Loaded {len(all_alerts)} total alerts")

            # Filter to active alerts only
            active_alerts = [
                a for a in all_alerts
                if isinstance(a, dict) and a.get("action", "on") == "on"
            ]
            logger.info(f"Found {len(active_alerts)} active alerts")

            # Check alerts
            stats = self.check_alerts(active_alerts, timeframe_filter)

            return stats

        except Exception as e:
            logger.error(f"Error in check_all_alerts: {e}")
            return {
                "total": 0,
                "triggered": 0,
                "errors": 1,
                "skipped": 0,
                "no_data": 0,
                "success": 0,
            }

    def check_alerts_for_exchanges(
        self,
        exchanges: List[str],
        timeframe_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check alerts for specific exchanges.

        Args:
            exchanges: List of exchange names to check
            timeframe_filter: Optional filter for timeframe ("daily", "weekly")

        Returns:
            Statistics dictionary with check results
        """
        logger.info(f"Checking alerts for exchanges: {exchanges}")

        try:
            # Load all alerts
            all_alerts = list_alerts()

            # Filter by exchange
            exchange_alerts = [
                a for a in all_alerts
                if isinstance(a, dict) and a.get("exchange") in exchanges
            ]

            logger.info(f"Found {len(exchange_alerts)} alerts for specified exchanges")

            # Check alerts
            return self.check_alerts(exchange_alerts, timeframe_filter)

        except Exception as e:
            logger.error(f"Error checking alerts for exchanges: {e}")
            return {
                "total": 0,
                "triggered": 0,
                "errors": 1,
                "skipped": 0,
                "no_data": 0,
                "success": 0,
            }


def check_stock_alerts(
    exchanges: Optional[List[str]] = None,
    timeframe_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to check stock alerts.

    Args:
        exchanges: Optional list of exchanges to filter by
        timeframe_filter: Optional timeframe filter ("daily", "weekly")

    Returns:
        Statistics dictionary
    """
    checker = StockAlertChecker()

    if exchanges:
        return checker.check_alerts_for_exchanges(exchanges, timeframe_filter)
    else:
        return checker.check_all_alerts(timeframe_filter)


def main():
    """Main function for standalone execution."""
    logger.info("Starting stock alert checker...")

    checker = StockAlertChecker()
    stats = checker.check_all_alerts()

    logger.info(f"Results: {stats}")
    return 0 if stats.get("errors", 0) == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

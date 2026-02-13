"""
Futures Alert Checker - Evaluates futures alerts and sends Discord notifications

This module processes futures alerts by:
1. Loading futures metadata and alerts from document store
2. Fetching continuous futures price data from PostgreSQL
3. Applying price adjustments (Panama or Ratio method) if specified
4. Evaluating conditions using backend.evaluate_expression_list()
5. Sending Discord notifications via discord_routing
6. Updating alert status (last_triggered timestamp)
"""
from pytz import timezone

import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data_access.db_config import db_config
from src.data_access.document_store import load_document, save_document
from src.services.backend import evaluate_expression_list
from src.services.discord_routing import send_economy_discord_alert

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FuturesAlertChecker:
    """
    Evaluates futures alerts and sends Discord notifications when conditions are met.

    Handles futures-specific features like continuous contract adjustments (Panama/Ratio methods)
    and manages database connections to the futures price database.
    """

    def __init__(self):
        """Initialize the futures alert checker."""
        self.futures_db: Dict[str, Any] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.conn = None
        self.price_cache: Dict[str, pd.DataFrame] = {}

    def load_futures_metadata(self) -> bool:
        """
        Load futures metadata from document store.

        Returns:
            True if successful, False otherwise
        """
        try:
            data = load_document(
                "futures_database",
                default={},
                fallback_path='futures_database.json',
            ) or {}

            if isinstance(data, dict):
                self.futures_db = data
                logger.info(f"Loaded {len(self.futures_db)} futures from metadata")
                return True

            logger.warning("Futures metadata is not a dictionary")
            self.futures_db = {}
            return False

        except Exception as e:
            logger.error(f"Failed to load futures metadata: {e}")
            self.futures_db = {}
            return False

    def load_futures_alerts(self) -> bool:
        """
        Load active futures alerts from document store.

        Returns:
            True if successful, False otherwise
        """
        try:
            alerts = load_document(
                "futures_alerts",
                default=[],
                fallback_path='futures_alerts.json',
            )

            if not isinstance(alerts, list):
                alerts = []

            # Filter to active alerts only
            self.alerts = [a for a in alerts if a.get('action', 'on') != 'off']
            logger.info(f"Loaded {len(self.alerts)} active futures alerts")
            return True

        except Exception as e:
            logger.error(f"Failed to load futures alerts: {e}")
            self.alerts = []
            return False

    def connect_db(self) -> bool:
        """
        Connect to futures price database.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.conn = db_config.get_connection(role="futures_prices")
            logger.info("Connected to futures price database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False

    def get_price_data(
        self,
        symbol: str,
        days: int = 200,
        adjustment_method: str = 'none'
    ) -> Optional[pd.DataFrame]:
        """
        Get continuous futures price data with optional adjustment.

        Args:
            symbol: Futures symbol (e.g., "ES", "CL", "GC")
            days: Number of days of historical data to fetch
            adjustment_method: Adjustment type ('none', 'panama', 'ratio')

        Returns:
            DataFrame with OHLCV data, or None if unavailable
        """
        cache_key = f"{symbol}_{days}_{adjustment_method}"

        # Check cache first
        if cache_key in self.price_cache:
            logger.debug(f"Using cached data for {symbol}")
            return self.price_cache[cache_key]

        try:
            query = """
                SELECT date, open, high, low, close, volume
                FROM continuous_prices
                WHERE symbol = %s
                ORDER BY date DESC
                LIMIT %s
            """

            df = pd.read_sql_query(query, self.conn, params=(symbol, days))

            if df.empty:
                logger.warning(f"No price data for {symbol}")
                return None

            # Convert date to datetime and sort
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date').reset_index(drop=True)

            # Rename columns to match backend expectations (capitalized)
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'date': 'Date'
            })

            # Apply adjustment if needed
            if adjustment_method != 'none':
                df = self.apply_adjustment(df, adjustment_method)

            # Cache the result
            self.price_cache[cache_key] = df
            logger.debug(f"Fetched {len(df)} price records for {symbol}")

            return df

        except Exception as e:
            logger.error(f"Error getting price data for {symbol}: {e}")
            return None

    def apply_adjustment(
        self,
        df: pd.DataFrame,
        method: str = 'panama'
    ) -> pd.DataFrame:
        """
        Apply Panama or Ratio adjustment to futures prices.

        This handles gaps in continuous contracts due to contract rolls,
        creating a smooth price series for technical analysis.

        Args:
            df: DataFrame with OHLCV data
            method: Adjustment method ('panama' for additive, 'ratio' for multiplicative)

        Returns:
            Adjusted DataFrame
        """
        if method == 'none' or df.empty:
            return df

        df = df.sort_values('Date').copy()

        # Calculate daily returns to detect roll gaps
        df['close_change'] = df['Close'].diff()
        df['pct_change'] = df['Close'].pct_change()

        # Identify potential roll points (gaps > 2% or 2x std deviation)
        returns_std = df['pct_change'].std()
        roll_threshold = max(0.02, 2 * returns_std)
        df['is_roll'] = abs(df['pct_change']) > roll_threshold

        if method == 'panama':
            # Panama Canal method: Additive adjustment
            cumulative_adjustment = 0
            for idx in df.index:
                if df.loc[idx, 'is_roll'] and idx > 0:
                    gap = df.loc[idx, 'close_change']
                    cumulative_adjustment -= gap

                df.loc[idx, 'Close'] = df.loc[idx, 'Close'] + cumulative_adjustment
                df.loc[idx, 'Open'] = df.loc[idx, 'Open'] + cumulative_adjustment
                df.loc[idx, 'High'] = df.loc[idx, 'High'] + cumulative_adjustment
                df.loc[idx, 'Low'] = df.loc[idx, 'Low'] + cumulative_adjustment

        elif method == 'ratio':
            # Ratio method: Multiplicative adjustment
            cumulative_factor = 1.0
            for idx in df.index:
                if df.loc[idx, 'is_roll'] and idx > 0:
                    prev_idx = df.index[df.index.get_loc(idx) - 1]
                    prev_close = df.loc[prev_idx, 'Close']

                    if prev_close != 0:
                        # Calculate the ratio between new contract and old contract
                        old_contract_implied = df.loc[idx, 'Close'] - df.loc[idx, 'close_change']
                        if old_contract_implied != 0:
                            ratio = df.loc[idx, 'Close'] / old_contract_implied
                            cumulative_factor *= ratio

                df.loc[idx, 'Close'] = df.loc[idx, 'Close'] * cumulative_factor
                df.loc[idx, 'Open'] = df.loc[idx, 'Open'] * cumulative_factor
                df.loc[idx, 'High'] = df.loc[idx, 'High'] * cumulative_factor
                df.loc[idx, 'Low'] = df.loc[idx, 'Low'] * cumulative_factor

        # Clean up temporary columns
        df = df.drop(['close_change', 'pct_change', 'is_roll'], axis=1, errors='ignore')

        return df

    def extract_conditions(self, alert: Dict[str, Any]) -> List[str]:
        """
        Extract condition strings from a futures alert.

        Handles multiple condition formats used in futures alerts.

        Args:
            alert: Alert dictionary

        Returns:
            List of condition strings to evaluate
        """
        conditions = alert.get('entry_conditions', {})
        result = []

        # Handle dictionary format
        if isinstance(conditions, dict):
            for key, value in conditions.items():
                if isinstance(value, dict) and 'conditions' in value:
                    cond_list = value['conditions']
                    if isinstance(cond_list, list):
                        result.extend([str(c) for c in cond_list])
                    else:
                        result.append(str(cond_list))

        # Handle list format
        elif isinstance(conditions, list):
            for cond in conditions:
                if isinstance(cond, dict):
                    cond_str = cond.get('conditions', '')
                    if cond_str:
                        result.append(str(cond_str))
                else:
                    result.append(str(cond))

        return result

    def evaluate_alert(self, alert: Dict[str, Any], df: pd.DataFrame) -> bool:
        """
        Evaluate a futures alert's conditions against price data.

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

            # Get combination logic (default to AND)
            combination = alert.get('entry_combination', 'AND')

            # Use backend's evaluate_expression_list
            result = evaluate_expression_list(df, conditions, combination)
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
        Format a Discord notification message for a triggered futures alert.

        Args:
            alert: Alert dictionary
            df: DataFrame with price data

        Returns:
            Formatted message string
        """
        symbol = alert.get('ticker', 'Unknown')
        name = alert.get('name', 'Unnamed Alert')
        current_price = df['Close'].iloc[-1] if not df.empty else 0
        timeframe = alert.get('timeframe', 'daily')
        adjustment = alert.get('adjustment_method', 'none')

        # Get futures info from metadata
        futures_info = self.futures_db.get(symbol, {})
        full_name = futures_info.get('name', symbol)
        category = futures_info.get('category', 'Unknown')

        # Get conditions for display
        conditions = self.extract_conditions(alert)
        conditions_str = "\n".join(f"  • {c}" for c in conditions[:3])  # Show first 3
        if len(conditions) > 3:
            conditions_str += f"\n  ... and {len(conditions) - 3} more"

        message = f"""🚨 **Futures Alert Triggered**

**{name}**
• Symbol: `{symbol}` ({full_name})
• Economy: Commodities/Futures
• Category: {category}
• Price: ${current_price:.2f}
• Timeframe: {timeframe}
• Adjustment: {adjustment}

**Conditions Met:**
{conditions_str}


*Triggered at {datetime.now(tz=timezone('UTC')).astimezone(timezone('US/Eastern')).strftime('%Y-%m-%d %I:%M:%S %p ET')}*"""


        return message

    def should_skip_alert(self, alert: Dict[str, Any]) -> bool:
        """
        Check if an alert should be skipped (already triggered today).

        Args:
            alert: Alert dictionary

        Returns:
            True if alert should be skipped, False otherwise
        """
        last_triggered = alert.get('last_triggered', '')

        if not last_triggered or last_triggered == 'Never':
            return False

        try:
            # Parse the last triggered timestamp
            triggered_date = datetime.fromisoformat(
                last_triggered.replace('Z', '+00:00')
            )

            # Skip if already triggered today
            if triggered_date.date() == datetime.now().date():
                return True

        except (ValueError, AttributeError) as e:
            logger.debug(f"Could not parse last_triggered: {e}")

        return False

    def update_alert_status(self, alert_id: str) -> bool:
        """
        Update alert's last_triggered timestamp.

        Args:
            alert_id: Alert identifier

        Returns:
            True if successful, False otherwise
        """
        try:
            all_alerts = load_document(
                "futures_alerts",
                default=[],
                fallback_path='futures_alerts.json',
            )

            if not isinstance(all_alerts, list):
                all_alerts = []

            # Find and update the alert
            for alert in all_alerts:
                if alert.get('alert_id') == alert_id:
                    alert['last_triggered'] = datetime.now().isoformat()
                    break

            save_document(
                "futures_alerts",
                all_alerts,
                fallback_path='futures_alerts.json',
            )

            return True

        except Exception as e:
            logger.error(f"Error updating alert status: {e}")
            return False

    def check_alert(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check a single futures alert and send notification if triggered.

        Args:
            alert: Alert dictionary

        Returns:
            Result dictionary with status information
        """
        start_time = time.time()
        alert_id = alert.get('alert_id', 'unknown')
        symbol = alert.get('ticker', '')

        result = {
            'alert_id': alert_id,
            'ticker': symbol,
            'triggered': False,
            'error': None,
            'skipped': False,
        }

        try:
            # Check if alert is disabled
            if alert.get('action', 'on') == 'off':
                result['skipped'] = True
                result['skip_reason'] = 'disabled'
                return result

            # Check if already triggered today
            if self.should_skip_alert(alert):
                result['skipped'] = True
                result['skip_reason'] = 'already_triggered_today'
                logger.debug(f"Alert {alert_id} already triggered today, skipping")
                return result

            if not symbol:
                result['error'] = 'No ticker specified'
                return result

            # Get price data with adjustment
            adjustment = alert.get('adjustment_method', 'none')
            df = self.get_price_data(symbol, days=200, adjustment_method=adjustment)

            if df is None or df.empty:
                result['error'] = f'No price data for {symbol}'
                return result

            # Evaluate conditions
            triggered = self.evaluate_alert(alert, df)

            if triggered:
                result['triggered'] = True
                logger.info(f"TRIGGERED: {symbol} - {alert.get('name', '')}")

                # Send Discord notification
                message = self.format_alert_message(alert, df)

                # Add ticker to alert dict for discord routing
                alert_with_ticker = alert.copy()
                alert_with_ticker['ticker'] = symbol

                success = send_economy_discord_alert(alert_with_ticker, message)

                if success:
                    logger.info(f"Discord notification sent for {symbol}")
                else:
                    logger.warning(f"Failed to send Discord notification for {symbol}")

                # Update alert's last_triggered timestamp
                self.update_alert_status(alert_id)

        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error checking alert {alert_id}: {e}")

        finally:
            execution_time = int((time.time() - start_time) * 1000)
            logger.debug(f"Alert {alert_id} check completed in {execution_time}ms")

        return result

    def check_all_alerts(self) -> Dict[str, Any]:
        """
        Load and check all active futures alerts.

        Returns:
            Statistics dictionary with check results
        """
        logger.info("=" * 60)
        logger.info("FUTURES ALERT CHECKER")
        logger.info("=" * 60)

        stats = {
            'total': 0,
            'triggered': 0,
            'errors': 0,
            'skipped': 0,
            'no_data': 0,
            'success': 0,
        }

        try:
            # Load metadata and alerts
            if not self.load_futures_metadata():
                logger.error("Failed to load futures metadata")
                stats['errors'] = 1
                return stats

            if not self.load_futures_alerts():
                logger.error("Failed to load futures alerts")
                stats['errors'] = 1
                return stats

            # Connect to database
            if not self.connect_db():
                logger.error("Failed to connect to database")
                stats['errors'] = 1
                return stats

            stats['total'] = len(self.alerts)
            logger.info(f"Checking {len(self.alerts)} active futures alerts")

            # Check each alert
            for alert in self.alerts:
                try:
                    result = self.check_alert(alert)

                    if result.get('skipped'):
                        stats['skipped'] += 1
                    elif result.get('error'):
                        if 'No price data' in str(result['error']):
                            stats['no_data'] += 1
                        else:
                            stats['errors'] += 1
                    elif result.get('triggered'):
                        stats['triggered'] += 1
                        stats['success'] += 1
                    else:
                        stats['success'] += 1

                except Exception as e:
                    logger.error(f"Unexpected error checking alert: {e}")
                    stats['errors'] += 1

            logger.info(
                f"Alert check complete: {stats['triggered']} triggered, "
                f"{stats['errors']} errors, {stats['skipped']} skipped"
            )

        except Exception as e:
            logger.error(f"Error in check_all_alerts: {e}")
            stats['errors'] += 1

        finally:
            # Close database connection
            if self.conn:
                try:
                    db_config.close_connection(self.conn)
                    logger.info("Database connection closed")
                except Exception as e:
                    logger.error(f"Error closing database connection: {e}")

        return stats


def check_futures_alerts() -> Dict[str, Any]:
    """
    Convenience function to check futures alerts.

    Returns:
        Statistics dictionary
    """
    checker = FuturesAlertChecker()
    return checker.check_all_alerts()


def main():
    """Main function for standalone execution."""
    logger.info("Starting futures alert checker...")

    checker = FuturesAlertChecker()
    stats = checker.check_all_alerts()

    logger.info(f"Results: {stats}")
    return 0 if stats.get('errors', 0) == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

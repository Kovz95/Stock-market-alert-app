"""
Futures Alert Checker - Evaluates futures alerts and sends Discord notifications
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import requests
from typing import Dict, List, Any
import os
import sys

from data_access.document_store import load_document, save_document
from db_config import db_config

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import indicators - import everything for direct access
import indicators_lib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FuturesAlertChecker:
    def __init__(self):
        self.futures_db = {}
        self.alerts = []
        self.conn = None
        self.discord_config = self.load_discord_config()

    def load_discord_config(self):
        """Load Discord webhook configuration"""
        try:
            config = load_document(
                "discord_channels_config",
                default={},
                fallback_path='discord_channels_config.json',
            ) or {}
            if 'futures' in config:
                return config['futures']
            return config.get('default', {})
        except Exception:
            logger.warning("No Discord config found")
            return {}

    def load_futures_metadata(self):
        """Load futures metadata"""
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

    def load_futures_alerts(self):
        """Load futures alerts from JSON"""
        try:
            alerts = load_document(
                "futures_alerts",
                default=[],
                fallback_path='futures_alerts.json',
            )
            if not isinstance(alerts, list):
                alerts = []
            self.alerts = [a for a in alerts if a.get('action', 'on') != 'off']
            logger.info(f"Loaded {len(self.alerts)} active futures alerts")
            return True
        except Exception as e:
            logger.error(f"Failed to load futures alerts: {e}")
            self.alerts = []
            return False

    def connect_db(self):
        """Connect to futures price database"""
        try:
            self.conn = db_config.get_connection(role="futures_prices")
            logger.info("Connected to futures price database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False

    def get_price_data(self, symbol, days=100, adjustment_method='none'):
        """Get price data for a futures symbol with adjustment"""
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

            # Convert date to datetime
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

            # Apply adjustment if needed
            if adjustment_method != 'none':
                df = self.apply_adjustment(df, adjustment_method)

            return df

        except Exception as e:
            logger.error(f"Error getting price data for {symbol}: {e}")
            return None

    def apply_adjustment(self, df, method='panama'):
        """Apply Panama or Ratio adjustment to futures prices"""
        if method == 'none' or df.empty:
            return df

        df = df.sort_values('date').copy()

        # Calculate daily returns to detect roll gaps
        df['close_change'] = df['close'].diff()
        df['pct_change'] = df['close'].pct_change()

        # Identify potential roll points (gaps > 2%)
        returns_std = df['pct_change'].std()
        roll_threshold = max(0.02, 2 * returns_std)
        df['is_roll'] = abs(df['pct_change']) > roll_threshold

        if method == 'panama':
            # Additive adjustment
            cumulative_adjustment = 0
            for idx in df.index:
                if df.loc[idx, 'is_roll'] and idx > 0:
                    gap = df.loc[idx, 'close_change']
                    cumulative_adjustment -= gap

                df.loc[idx, 'close'] = df.loc[idx, 'close'] + cumulative_adjustment
                df.loc[idx, 'open'] = df.loc[idx, 'open'] + cumulative_adjustment
                df.loc[idx, 'high'] = df.loc[idx, 'high'] + cumulative_adjustment
                df.loc[idx, 'low'] = df.loc[idx, 'low'] + cumulative_adjustment

        elif method == 'ratio':
            # Multiplicative adjustment
            cumulative_factor = 1.0
            for idx in df.index:
                if df.loc[idx, 'is_roll'] and idx > 0:
                    prev_idx = df.index[df.index.get_loc(idx) - 1]
                    if df.loc[prev_idx, 'close'] != 0:
                        ratio = df.loc[idx, 'close'] / (df.loc[prev_idx, 'close'] + df.loc[idx, 'close_change'])
                        cumulative_factor *= ratio

                df.loc[idx, 'close'] = df.loc[idx, 'close'] * cumulative_factor
                df.loc[idx, 'open'] = df.loc[idx, 'open'] * cumulative_factor
                df.loc[idx, 'high'] = df.loc[idx, 'high'] * cumulative_factor
                df.loc[idx, 'low'] = df.loc[idx, 'low'] * cumulative_factor

        # Clean up temporary columns
        df = df.drop(['close_change', 'pct_change', 'is_roll'], axis=1, errors='ignore')

        return df

    def evaluate_condition(self, condition_str, df, current_price):
        """Evaluate a single condition string"""
        try:
            # Calculate common indicators
            indicators = {}

            # Add some common indicators if they're used in the condition
            if 'SMA' in condition_str:
                indicators['SMA_20'] = indicators_lib.SMA(df, 20).iloc[-1]
                indicators['SMA_50'] = indicators_lib.SMA(df, 50).iloc[-1]
            if 'EMA' in condition_str:
                indicators['EMA_20'] = indicators_lib.EMA(df, 20).iloc[-1]
                indicators['EMA_50'] = indicators_lib.EMA(df, 50).iloc[-1]
            if 'RSI' in condition_str:
                indicators['RSI'] = indicators_lib.RSI(df, 14).iloc[-1]
            if 'MACD' in condition_str:
                macd_result = indicators_lib.MACD(df)
                indicators['MACD'] = macd_result['MACD'].iloc[-1]
                indicators['MACD_signal'] = macd_result['MACD_signal'].iloc[-1]
            if 'BB' in condition_str:
                bb_upper, bb_lower = indicators_lib.BBANDS(df)
                indicators['BB_upper'] = bb_upper.iloc[-1]
                indicators['BB_lower'] = bb_lower.iloc[-1]

            # Create evaluation context with access to full arrays for indexing
            context = {
                'price': current_price,
                'close': current_price,
                'volume': df['volume'].iloc[-1] if 'volume' in df.columns else 0,
                'high': df['high'].iloc[-1],
                'low': df['low'].iloc[-1],
                'open': df['open'].iloc[-1],
                # Add arrays for indexing like Close[-1], Close[-2] etc
                'Close': df['close'].values,
                'Open': df['open'].values,
                'High': df['high'].values,
                'Low': df['low'].values,
                'Volume': df['volume'].values if 'volume' in df.columns else None,
                **indicators
            }

            # Evaluate condition
            result = eval(condition_str, {"__builtins__": {}}, context)
            return bool(result)

        except Exception as e:
            logger.error(f"Error evaluating condition '{condition_str}': {e}")
            return False

    def check_alert(self, alert):
        """Check if an alert should trigger"""
        try:
            symbol = alert.get('ticker', '')
            adjustment = alert.get('adjustment_method', 'none')

            # Get price data
            df = self.get_price_data(symbol, days=100, adjustment_method=adjustment)

            if df is None or df.empty:
                return False

            current_price = df['close'].iloc[-1]

            # Get conditions
            conditions = alert.get('entry_conditions', {})
            combination = alert.get('entry_combination', 'AND')

            # Evaluate conditions
            results = []

            if isinstance(conditions, dict):
                for key, value in conditions.items():
                    if isinstance(value, dict) and 'conditions' in value:
                        cond_list = value['conditions']
                        if isinstance(cond_list, list):
                            for cond in cond_list:
                                result = self.evaluate_condition(cond, df, current_price)
                                results.append(result)
                        else:
                            result = self.evaluate_condition(str(cond_list), df, current_price)
                            results.append(result)
            elif isinstance(conditions, list):
                for cond in conditions:
                    if isinstance(cond, dict):
                        cond_str = cond.get('conditions', '')
                    else:
                        cond_str = str(cond)
                    result = self.evaluate_condition(cond_str, df, current_price)
                    results.append(result)

            # Combine results
            if not results:
                return False

            if combination == 'AND':
                return all(results)
            else:  # OR
                return any(results)

        except Exception as e:
            logger.error(f"Error checking alert {alert.get('alert_id', '')}: {e}")
            return False

    def send_discord_notification(self, alert, symbol_data):
        """Send Discord notification for triggered alert"""
        try:
            webhook_url = self.discord_config.get('webhook_url')
            if not webhook_url:
                logger.warning("No Discord webhook configured")
                return False

            # Get futures info
            futures_info = self.futures_db.get(alert['ticker'], {})

            # Create embed
            embed = {
                "title": f"🚨 Futures Alert Triggered: {alert['ticker']}",
                "description": alert.get('name', 'Unnamed Alert'),
                "color": 16711680,  # Red color
                "fields": [
                    {
                        "name": "Symbol",
                        "value": alert['ticker'],
                        "inline": True
                    },
                    {
                        "name": "Name",
                        "value": futures_info.get('name', alert['ticker']),
                        "inline": True
                    },
                    {
                        "name": "Category",
                        "value": futures_info.get('category', 'Unknown'),
                        "inline": True
                    },
                    {
                        "name": "Current Price",
                        "value": f"${symbol_data.get('close', 0):.2f}",
                        "inline": True
                    },
                    {
                        "name": "Adjustment",
                        "value": alert.get('adjustment_method', 'none'),
                        "inline": True
                    },
                    {
                        "name": "Timeframe",
                        "value": alert.get('timeframe', 'daily'),
                        "inline": True
                    },
                    {
                        "name": "Conditions",
                        "value": str(alert.get('entry_conditions', ''))[:1000],
                        "inline": False
                    }
                ],
                "timestamp": datetime.now().isoformat(),
                "footer": {
                    "text": "Futures Alert System"
                }
            }

            # Send to Discord
            response = requests.post(
                webhook_url,
                json={
                    "username": "Futures Alert Bot",
                    "embeds": [embed]
                }
            )

            if response.status_code == 204:
                logger.info(f"Discord notification sent for {alert['ticker']}")
                return True
            else:
                logger.error(f"Discord notification failed: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error sending Discord notification: {e}")
            return False

    def update_alert_status(self, alert_id):
        """Update alert's last_triggered timestamp"""
        try:
            all_alerts = load_document(
                "futures_alerts",
                default=[],
                fallback_path='futures_alerts.json',
            )
            if not isinstance(all_alerts, list):
                all_alerts = []

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

    def check_all_alerts(self):
        """Check all active futures alerts"""
        if not self.load_futures_metadata():
            return False

        if not self.load_futures_alerts():
            return False

        if not self.connect_db():
            return False

        logger.info(f"Checking {len(self.alerts)} active futures alerts")

        triggered_count = 0

        for alert in self.alerts:
            try:
                symbol = alert.get('ticker', '')
                alert_id = alert.get('alert_id', '')

                logger.debug(f"Checking {symbol} - {alert.get('name', '')}")

                # Check if alert triggers
                if self.check_alert(alert):
                    # Check if already triggered today
                    last_triggered = alert.get('last_triggered', 'Never')
                    if last_triggered != 'Never':
                        try:
                            triggered_date = datetime.fromisoformat(last_triggered.replace('Z', '+00:00'))
                            if triggered_date.date() == datetime.now().date():
                                logger.debug(f"Alert {alert_id} already triggered today")
                                continue
                        except:
                            pass

                    logger.info(f"TRIGGERED: {symbol} - {alert.get('name', '')}")

                    # Get current price data for notification
                    df = self.get_price_data(symbol, days=1,
                                            adjustment_method=alert.get('adjustment_method', 'none'))
                    symbol_data = {
                        'close': df['close'].iloc[-1] if df is not None else 0
                    }

                    # Send Discord notification
                    self.send_discord_notification(alert, symbol_data)

                    # Update alert status
                    self.update_alert_status(alert_id)

                    triggered_count += 1

            except Exception as e:
                logger.error(f"Error processing alert {alert.get('alert_id', '')}: {e}")

        logger.info(f"Alert check complete: {triggered_count} alerts triggered")

        # Close database
        if self.conn:
            db_config.close_connection(self.conn)

        return True


def main():
    """Main function for standalone execution"""
    logger.info("="*60)
    logger.info("FUTURES ALERT CHECKER")
    logger.info("="*60)

    checker = FuturesAlertChecker()
    success = checker.check_all_alerts()

    if success:
        logger.info("Alert check completed successfully")
        return 0
    else:
        logger.error("Alert check failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

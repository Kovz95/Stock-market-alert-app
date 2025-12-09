"""
Futures Price Updater - Fetches latest futures prices from Interactive Brokers
"""

import logging
from datetime import datetime, timedelta
from ib_insync import IB, ContFuture, util
import pandas as pd
import time
import sys
import argparse
import os
from data_access.document_store import load_document
from db_config import db_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FuturesPriceUpdater:
    def __init__(self):
        self.ib = None
        self.conn = None
        self.futures_db = {}
        self.config = self.load_config()

    def load_config(self):
        """Load IB connection config"""
        default = {
            "connection": {
                "host": "127.0.0.1",
                "port": 7496,
                "client_id": 1,
            }
        }
        try:
            config = load_document(
                "ib_futures_config",
                default=default,
                fallback_path='ib_futures_config.json',
            )
            return config if isinstance(config, dict) else default
        except Exception as e:
            logger.error(f"Failed to load IB config: {e}")
            return default

    def connect_ib(self):
        """Connect to Interactive Brokers"""
        try:
            self.ib = IB()
            conn_config = self.config.get('connection', {})
            self.ib.connect(
                conn_config.get('host', '127.0.0.1'),
                conn_config.get('port', 7496),
                clientId=conn_config.get('client_id', 1)
            )
            logger.info("Connected to Interactive Brokers")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            return False

    def connect_db(self):
        """Connect to PostgreSQL price database"""
        try:
            self.conn = db_config.get_connection(role="futures_prices")
            logger.info("Connected to price database")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            return False

    def load_futures_metadata(self):
        """Load futures metadata from JSON"""
        try:
            data = load_document(
                "futures_database",
                default={},
                fallback_path='futures_database.json',
            )
            self.futures_db = data if isinstance(data, dict) else {}
            logger.info(f"Loaded {len(self.futures_db)} futures from metadata")
            return True
        except Exception as e:
            logger.error(f"Failed to load futures metadata: {e}")
            return False

    def get_latest_price(self, symbol, futures_data):
        """Get latest price for a single futures contract"""
        try:
            # For continuous contracts, always use the base symbol
            # Don't use local_symbol which contains specific contract months
            ib_symbol = symbol

            # Currency futures mapping
            currency_map = {
                '6E': 'EUR', '6B': 'GBP', '6J': 'JPY',
                '6C': 'CAD', '6A': 'AUD', '6S': 'CHF',
                '6N': 'NZD', '6M': 'MXN'
            }

            if symbol in currency_map:
                ib_symbol = currency_map[symbol]

            # Get exchange
            exchange = futures_data.get('exchange', 'SMART')

            # Create continuous contract
            contract = ContFuture(symbol=ib_symbol, exchange=exchange)

            # Get contract details
            self.ib.qualifyContracts(contract)

            # Request latest bar (1 day)
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr='1 D',
                barSizeSetting='1 day',
                whatToShow='TRADES',
                useRTH=False,
                formatDate=1
            )

            if bars:
                latest = bars[-1]
                return {
                    'symbol': symbol,
                    'date': latest.date.strftime('%Y-%m-%d') if hasattr(latest.date, 'strftime') else str(latest.date),
                    'open': float(latest.open),
                    'high': float(latest.high),
                    'low': float(latest.low),
                    'close': float(latest.close),
                    'volume': int(latest.volume) if latest.volume else 0
                }

        except Exception as e:
            logger.warning(f"Failed to get price for {symbol}: {e}")

        return None

    def update_database(self, price_data):
        """Update database with latest price"""
        try:
            with self.conn.cursor() as cursor:
                # Check if record exists for today
                cursor.execute(
                    """
                    SELECT COUNT(*) FROM continuous_prices
                    WHERE symbol = %s AND date = %s
                    """,
                    (price_data["symbol"], price_data["date"]),
                )

                exists = cursor.fetchone()[0] > 0

                if exists:
                    cursor.execute(
                        """
                        UPDATE continuous_prices
                        SET open = %s, high = %s, low = %s, close = %s, volume = %s
                        WHERE symbol = %s AND date = %s
                        """,
                        (
                            price_data["open"],
                            price_data["high"],
                            price_data["low"],
                            price_data["close"],
                            price_data["volume"],
                            price_data["symbol"],
                            price_data["date"],
                        ),
                    )
                    logger.debug(f"Updated {price_data['symbol']} for {price_data['date']}")
                else:
                    cursor.execute(
                        """
                        INSERT INTO continuous_prices
                        (symbol, date, open, high, low, close, volume, adjustment_method, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, 'none', %s)
                        """,
                        (
                            price_data["symbol"],
                            price_data["date"],
                            price_data["open"],
                            price_data["high"],
                            price_data["low"],
                            price_data["close"],
                            price_data["volume"],
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        ),
                    )
                    logger.debug(f"Inserted {price_data['symbol']} for {price_data['date']}")

            self.conn.commit()
            return True

        except Exception as e:
            logger.error(f"Failed to update database for {price_data['symbol']}: {e}")
            return False

    def update_all_futures(self):
        """Update prices for all futures with IB metadata"""
        if not self.connect_ib():
            return False

        if not self.connect_db():
            self.disconnect()
            return False

        if not self.load_futures_metadata():
            self.disconnect()
            return False

        # Get futures with IB metadata
        futures_to_update = []
        for symbol, data in self.futures_db.items():
            if data.get('ib_metadata_available'):
                futures_to_update.append((symbol, data))

        logger.info(f"Updating {len(futures_to_update)} futures contracts")

        updated_count = 0
        failed_count = 0

        # Update in batches to respect IB rate limits
        for i, (symbol, data) in enumerate(futures_to_update):
            try:
                logger.info(f"Updating {symbol} ({i+1}/{len(futures_to_update)})")

                # Get latest price
                price_data = self.get_latest_price(symbol, data)

                if price_data:
                    # Update database
                    if self.update_database(price_data):
                        updated_count += 1
                        logger.info(f"[OK] {symbol}: {price_data['close']:.2f}")
                    else:
                        failed_count += 1
                        logger.warning(f"[DB ERROR] {symbol}")
                else:
                    failed_count += 1
                    logger.warning(f"[NO DATA] {symbol}")

                # Rate limiting - 2 seconds between requests
                if i < len(futures_to_update) - 1:
                    time.sleep(2)

            except Exception as e:
                logger.error(f"Error updating {symbol}: {e}")
                failed_count += 1

        logger.info(f"Update complete: {updated_count} succeeded, {failed_count} failed")

        self.disconnect()
        return updated_count > 0

    def update_specific_futures(self, symbols):
        """Update specific futures symbols"""
        if not self.connect_ib():
            return False

        if not self.connect_db():
            self.disconnect()
            return False

        if not self.load_futures_metadata():
            self.disconnect()
            return False

        updated_count = 0

        for symbol in symbols:
            if symbol in self.futures_db:
                data = self.futures_db[symbol]
                if data.get('ib_metadata_available'):
                    price_data = self.get_latest_price(symbol, data)
                    if price_data and self.update_database(price_data):
                        updated_count += 1
                        logger.info(f"Updated {symbol}: {price_data['close']:.2f}")

        self.disconnect()
        return updated_count > 0

    def disconnect(self):
        """Disconnect from IB and database"""
        try:
            if self.ib and self.ib.isConnected():
                self.ib.disconnect()
                logger.info("Disconnected from IB")
        except:
            pass

        try:
            if self.conn:
                db_config.close_connection(self.conn)
                self.conn = None
                logger.info("Disconnected from database")
        except:
            pass


def main():
    """Main function for standalone execution"""
    logger.info("="*60)
    logger.info("FUTURES PRICE UPDATER")
    logger.info("="*60)

    parser = argparse.ArgumentParser(description='Update futures price data')
    parser.add_argument('--symbols', help='File containing symbols to update or comma-separated list')
    parser.add_argument('symbols_list', nargs='*', help='Symbols to update')

    args = parser.parse_args()

    updater = FuturesPriceUpdater()

    symbols_to_update = []

    if args.symbols:
        # Check if it's a file or comma-separated list
        if os.path.exists(args.symbols):
            # Read symbols from file
            with open(args.symbols, 'r') as f:
                symbols_to_update = [line.strip() for line in f if line.strip()]
            logger.info(f"Read {len(symbols_to_update)} symbols from file")
        else:
            # Parse as comma-separated list
            symbols_to_update = [s.strip() for s in args.symbols.split(',')]
            logger.info(f"Parsed {len(symbols_to_update)} symbols from input")
    elif args.symbols_list:
        # Use positional arguments
        symbols_to_update = args.symbols_list
        logger.info(f"Using {len(symbols_to_update)} symbols from arguments")

    if symbols_to_update:
        # Update specific symbols
        logger.info(f"Updating specific futures: {symbols_to_update}")
        success = updater.update_specific_futures(symbols_to_update)
    else:
        # Update all futures
        logger.info("Updating all futures with IB metadata")
        success = updater.update_all_futures()

    if success:
        logger.info("Price update completed successfully")
        return 0
    else:
        logger.error("Price update failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())

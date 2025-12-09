"""
Interactive Brokers futures data backend
Supports all major futures contracts with front month rolling
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
import logging
import json
import os
from threading import Thread, Event
import queue
import time

# IB API imports (install with: pip install ib_insync)
try:
    from ib_insync import IB, Future, Contract, util
    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    print("Warning: ib_insync not installed. Run: pip install ib_insync")

logger = logging.getLogger(__name__)

class IBFuturesDataFetcher:
    """
    Interactive Brokers futures data fetcher
    Handles all futures contracts with automatic front month rolling
    """

    # Comprehensive futures contract specifications
    FUTURES_CONTRACTS = {
        # Energy
        'CL': {'exchange': 'NYMEX', 'name': 'WTI Crude Oil', 'category': 'Energy', 'multiplier': 1000, 'currency': 'USD'},
        'QM': {'exchange': 'NYMEX', 'name': 'E-mini Crude Oil', 'category': 'Energy', 'multiplier': 500, 'currency': 'USD'},
        'BZ': {'exchange': 'NYMEX', 'name': 'Brent Crude Oil', 'category': 'Energy', 'multiplier': 1000, 'currency': 'USD'},
        'NG': {'exchange': 'NYMEX', 'name': 'Natural Gas', 'category': 'Energy', 'multiplier': 10000, 'currency': 'USD'},
        'QG': {'exchange': 'NYMEX', 'name': 'E-mini Natural Gas', 'category': 'Energy', 'multiplier': 2500, 'currency': 'USD'},
        'HO': {'exchange': 'NYMEX', 'name': 'Heating Oil', 'category': 'Energy', 'multiplier': 42000, 'currency': 'USD'},
        'RB': {'exchange': 'NYMEX', 'name': 'RBOB Gasoline', 'category': 'Energy', 'multiplier': 42000, 'currency': 'USD'},

        # Precious Metals
        'GC': {'exchange': 'COMEX', 'name': 'Gold', 'category': 'Metals', 'multiplier': 100, 'currency': 'USD'},
        'MGC': {'exchange': 'COMEX', 'name': 'Micro Gold', 'category': 'Metals', 'multiplier': 10, 'currency': 'USD'},
        'QO': {'exchange': 'COMEX', 'name': 'E-mini Gold', 'category': 'Metals', 'multiplier': 50, 'currency': 'USD'},
        'SI': {'exchange': 'COMEX', 'name': 'Silver', 'category': 'Metals', 'multiplier': 5000, 'currency': 'USD'},
        'SIL': {'exchange': 'COMEX', 'name': 'Micro Silver', 'category': 'Metals', 'multiplier': 1000, 'currency': 'USD'},
        'QI': {'exchange': 'COMEX', 'name': 'E-mini Silver', 'category': 'Metals', 'multiplier': 2500, 'currency': 'USD'},
        'HG': {'exchange': 'COMEX', 'name': 'Copper', 'category': 'Metals', 'multiplier': 25000, 'currency': 'USD'},
        'QC': {'exchange': 'COMEX', 'name': 'E-mini Copper', 'category': 'Metals', 'multiplier': 12500, 'currency': 'USD'},
        'PL': {'exchange': 'NYMEX', 'name': 'Platinum', 'category': 'Metals', 'multiplier': 50, 'currency': 'USD'},
        'PA': {'exchange': 'NYMEX', 'name': 'Palladium', 'category': 'Metals', 'multiplier': 100, 'currency': 'USD'},

        # Stock Index Futures
        'ES': {'exchange': 'CME', 'name': 'E-mini S&P 500', 'category': 'Indices', 'multiplier': 50, 'currency': 'USD'},
        'MES': {'exchange': 'CME', 'name': 'Micro E-mini S&P 500', 'category': 'Indices', 'multiplier': 5, 'currency': 'USD'},
        'NQ': {'exchange': 'CME', 'name': 'E-mini Nasdaq 100', 'category': 'Indices', 'multiplier': 20, 'currency': 'USD'},
        'MNQ': {'exchange': 'CME', 'name': 'Micro E-mini Nasdaq', 'category': 'Indices', 'multiplier': 2, 'currency': 'USD'},
        'YM': {'exchange': 'CBOT', 'name': 'E-mini Dow Jones', 'category': 'Indices', 'multiplier': 5, 'currency': 'USD'},
        'MYM': {'exchange': 'CBOT', 'name': 'Micro E-mini Dow', 'category': 'Indices', 'multiplier': 0.5, 'currency': 'USD'},
        'RTY': {'exchange': 'CME', 'name': 'E-mini Russell 2000', 'category': 'Indices', 'multiplier': 50, 'currency': 'USD'},
        'M2K': {'exchange': 'CME', 'name': 'Micro E-mini Russell', 'category': 'Indices', 'multiplier': 5, 'currency': 'USD'},

        # Agricultural
        'ZC': {'exchange': 'CBOT', 'name': 'Corn', 'category': 'Agriculture', 'multiplier': 50, 'currency': 'USD'},
        'YC': {'exchange': 'CBOT', 'name': 'Mini Corn', 'category': 'Agriculture', 'multiplier': 10, 'currency': 'USD'},
        'ZW': {'exchange': 'CBOT', 'name': 'Chicago Wheat', 'category': 'Agriculture', 'multiplier': 50, 'currency': 'USD'},
        'YW': {'exchange': 'CBOT', 'name': 'Mini Wheat', 'category': 'Agriculture', 'multiplier': 10, 'currency': 'USD'},
        'ZS': {'exchange': 'CBOT', 'name': 'Soybeans', 'category': 'Agriculture', 'multiplier': 50, 'currency': 'USD'},
        'YK': {'exchange': 'CBOT', 'name': 'Mini Soybeans', 'category': 'Agriculture', 'multiplier': 10, 'currency': 'USD'},
        'ZM': {'exchange': 'CBOT', 'name': 'Soybean Meal', 'category': 'Agriculture', 'multiplier': 100, 'currency': 'USD'},
        'ZL': {'exchange': 'CBOT', 'name': 'Soybean Oil', 'category': 'Agriculture', 'multiplier': 60000, 'currency': 'USD'},
        'KE': {'exchange': 'CBOT', 'name': 'KC Wheat', 'category': 'Agriculture', 'multiplier': 50, 'currency': 'USD'},
        'CT': {'exchange': 'NYBOT', 'name': 'Cotton', 'category': 'Agriculture', 'multiplier': 50000, 'currency': 'USD'},
        'SB': {'exchange': 'NYBOT', 'name': 'Sugar #11', 'category': 'Agriculture', 'multiplier': 112000, 'currency': 'USD'},
        'KC': {'exchange': 'NYBOT', 'name': 'Coffee', 'category': 'Agriculture', 'multiplier': 37500, 'currency': 'USD'},
        'CC': {'exchange': 'NYBOT', 'name': 'Cocoa', 'category': 'Agriculture', 'multiplier': 10, 'currency': 'USD'},
        'OJ': {'exchange': 'NYBOT', 'name': 'Orange Juice', 'category': 'Agriculture', 'multiplier': 15000, 'currency': 'USD'},

        # Livestock
        'LE': {'exchange': 'CME', 'name': 'Live Cattle', 'category': 'Livestock', 'multiplier': 40000, 'currency': 'USD'},
        'GF': {'exchange': 'CME', 'name': 'Feeder Cattle', 'category': 'Livestock', 'multiplier': 50000, 'currency': 'USD'},
        'HE': {'exchange': 'CME', 'name': 'Lean Hogs', 'category': 'Livestock', 'multiplier': 40000, 'currency': 'USD'},

        # Currencies
        '6E': {'exchange': 'CME', 'name': 'Euro FX', 'category': 'Currencies', 'multiplier': 125000, 'currency': 'USD'},
        'M6E': {'exchange': 'CME', 'name': 'Micro Euro FX', 'category': 'Currencies', 'multiplier': 12500, 'currency': 'USD'},
        '6B': {'exchange': 'CME', 'name': 'British Pound', 'category': 'Currencies', 'multiplier': 62500, 'currency': 'USD'},
        'M6B': {'exchange': 'CME', 'name': 'Micro British Pound', 'category': 'Currencies', 'multiplier': 6250, 'currency': 'USD'},
        '6J': {'exchange': 'CME', 'name': 'Japanese Yen', 'category': 'Currencies', 'multiplier': 12500000, 'currency': 'USD'},
        'MJY': {'exchange': 'CME', 'name': 'Micro Japanese Yen', 'category': 'Currencies', 'multiplier': 1250000, 'currency': 'USD'},
        '6C': {'exchange': 'CME', 'name': 'Canadian Dollar', 'category': 'Currencies', 'multiplier': 100000, 'currency': 'USD'},
        'MCD': {'exchange': 'CME', 'name': 'Micro Canadian Dollar', 'category': 'Currencies', 'multiplier': 10000, 'currency': 'USD'},
        '6A': {'exchange': 'CME', 'name': 'Australian Dollar', 'category': 'Currencies', 'multiplier': 100000, 'currency': 'USD'},
        'M6A': {'exchange': 'CME', 'name': 'Micro Australian Dollar', 'category': 'Currencies', 'multiplier': 10000, 'currency': 'USD'},
        '6S': {'exchange': 'CME', 'name': 'Swiss Franc', 'category': 'Currencies', 'multiplier': 125000, 'currency': 'USD'},
        'MSF': {'exchange': 'CME', 'name': 'Micro Swiss Franc', 'category': 'Currencies', 'multiplier': 12500, 'currency': 'USD'},
        '6N': {'exchange': 'CME', 'name': 'New Zealand Dollar', 'category': 'Currencies', 'multiplier': 100000, 'currency': 'USD'},
        '6M': {'exchange': 'CME', 'name': 'Mexican Peso', 'category': 'Currencies', 'multiplier': 500000, 'currency': 'USD'},
        'DX': {'exchange': 'NYBOT', 'name': 'US Dollar Index', 'category': 'Currencies', 'multiplier': 1000, 'currency': 'USD'},

        # Interest Rates/Bonds
        'ZN': {'exchange': 'CBOT', 'name': '10-Year T-Note', 'category': 'Bonds', 'multiplier': 1000, 'currency': 'USD'},
        '10Y': {'exchange': 'CBOT', 'name': 'Micro 10-Year Note', 'category': 'Bonds', 'multiplier': 100, 'currency': 'USD'},
        'ZB': {'exchange': 'CBOT', 'name': '30-Year T-Bond', 'category': 'Bonds', 'multiplier': 1000, 'currency': 'USD'},
        'UB': {'exchange': 'CBOT', 'name': 'Ultra T-Bond', 'category': 'Bonds', 'multiplier': 1000, 'currency': 'USD'},
        'ZF': {'exchange': 'CBOT', 'name': '5-Year T-Note', 'category': 'Bonds', 'multiplier': 1000, 'currency': 'USD'},
        '5YR': {'exchange': 'CBOT', 'name': 'Micro 5-Year Note', 'category': 'Bonds', 'multiplier': 100, 'currency': 'USD'},
        'ZT': {'exchange': 'CBOT', 'name': '2-Year T-Note', 'category': 'Bonds', 'multiplier': 2000, 'currency': 'USD'},
        '2YR': {'exchange': 'CBOT', 'name': 'Micro 2-Year Note', 'category': 'Bonds', 'multiplier': 200, 'currency': 'USD'},
        'ZQ': {'exchange': 'CBOT', 'name': '30-Day Fed Funds', 'category': 'Bonds', 'multiplier': 4167, 'currency': 'USD'},
        'GE': {'exchange': 'CME', 'name': 'Eurodollar', 'category': 'Bonds', 'multiplier': 2500, 'currency': 'USD'},

        # Volatility
        'VX': {'exchange': 'CFE', 'name': 'VIX Futures', 'category': 'Volatility', 'multiplier': 1000, 'currency': 'USD'},
        'VXM': {'exchange': 'CFE', 'name': 'Mini VIX', 'category': 'Volatility', 'multiplier': 100, 'currency': 'USD'},

        # European Indices
        'FDAX': {'exchange': 'EUREX', 'name': 'DAX', 'category': 'European Indices', 'multiplier': 25, 'currency': 'EUR'},
        'FDXM': {'exchange': 'EUREX', 'name': 'Mini DAX', 'category': 'European Indices', 'multiplier': 5, 'currency': 'EUR'},
        'FESX': {'exchange': 'EUREX', 'name': 'Euro Stoxx 50', 'category': 'European Indices', 'multiplier': 10, 'currency': 'EUR'},
        'FESB': {'exchange': 'EUREX', 'name': 'Micro Euro Stoxx 50', 'category': 'European Indices', 'multiplier': 1, 'currency': 'EUR'},
        'FGBL': {'exchange': 'EUREX', 'name': 'Euro Bund', 'category': 'European Indices', 'multiplier': 1000, 'currency': 'EUR'},
        'Z': {'exchange': 'LIFFE', 'name': 'FTSE 100', 'category': 'European Indices', 'multiplier': 10, 'currency': 'GBP'},

        # Asian Indices
        'NIY': {'exchange': 'OSE', 'name': 'Nikkei 225', 'category': 'Asian Indices', 'multiplier': 500, 'currency': 'JPY'},
        'HSI': {'exchange': 'HKFE', 'name': 'Hang Seng', 'category': 'Asian Indices', 'multiplier': 50, 'currency': 'HKD'},
        'MHI': {'exchange': 'HKFE', 'name': 'Mini Hang Seng', 'category': 'Asian Indices', 'multiplier': 10, 'currency': 'HKD'},
    }

    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        """
        Initialize IB Futures Data Fetcher

        Args:
            host: TWS/Gateway host (default localhost)
            port: TWS port 7497 or Gateway port 4001
            client_id: Client ID for IB API
        """
        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = None
        self.connected = False
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 60  # 1 minute cache for real-time data

        # Load configuration
        self.config = self._load_config()

        # Connect on initialization if configured
        if self.config.get('auto_connect', False):
            self.connect()

    def _load_config(self) -> Dict:
        """Load IB configuration from file"""
        config_file = 'ib_futures_config.json'
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    # Update connection parameters if provided
                    self.host = config.get('host', self.host)
                    self.port = config.get('port', self.port)
                    self.client_id = config.get('client_id', self.client_id)
                    return config
            except Exception as e:
                logger.error(f"Error loading IB config: {e}")
        return {}

    def connect(self) -> bool:
        """
        Connect to Interactive Brokers TWS or Gateway

        Returns:
            True if connected successfully
        """
        if not IB_AVAILABLE:
            logger.error("ib_insync not installed. Run: pip install ib_insync")
            return False

        try:
            self.ib = IB()
            self.ib.connect(self.host, self.port, clientId=self.client_id)
            self.connected = True
            logger.info(f"Connected to IB at {self.host}:{self.port}")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IB: {e}")
            self.connected = False
            return False

    def disconnect(self):
        """Disconnect from IB"""
        if self.ib and self.connected:
            self.ib.disconnect()
            self.connected = False
            logger.info("Disconnected from IB")

    def get_front_month_expiry(self, symbol: str) -> str:
        """
        Calculate the front month expiry for a futures contract
        Uses proper roll dates based on contract specifications

        Args:
            symbol: Futures symbol

        Returns:
            Expiry date string in YYYYMM format
        """
        now = datetime.now()
        current_day = now.day

        # Contract-specific roll rules (days before expiry to roll)
        roll_rules = {
            # Equity indices - roll 8 days before expiry (usually 3rd Friday)
            'ES': 8, 'MES': 8, 'NQ': 8, 'MNQ': 8, 'YM': 8, 'MYM': 8, 'RTY': 8, 'M2K': 8,

            # Energy - roll 3-4 days before expiry (usually around 20th-25th)
            'CL': 4, 'QM': 4, 'NG': 3, 'QG': 3, 'BZ': 4, 'HO': 4, 'RB': 4,

            # Metals - roll 3-4 days before expiry (usually last business day)
            'GC': 4, 'MGC': 4, 'SI': 4, 'SIL': 4, 'HG': 4, 'QC': 4, 'PL': 4, 'PA': 4,

            # Agriculture - roll 5-7 days before expiry (14th for grains)
            'ZC': 7, 'YC': 7, 'ZW': 7, 'YW': 7, 'ZS': 7, 'YK': 7, 'ZM': 7, 'ZL': 7,
            'KC': 5, 'SB': 5, 'CT': 5, 'CC': 5, 'OJ': 5,

            # Currencies - roll 2 days before expiry (usually Monday before 3rd Wednesday)
            '6E': 2, 'M6E': 2, '6B': 2, 'M6B': 2, '6J': 2, 'MJY': 2,
            '6C': 2, 'MCD': 2, '6A': 2, 'M6A': 2, '6S': 2, 'MSF': 2,

            # Bonds - roll 7 days before expiry (last business day)
            'ZN': 7, '10Y': 7, 'ZB': 7, 'ZF': 7, '5YR': 7, 'ZT': 7, '2YR': 7,

            # Volatility - special handling for VIX
            'VX': 6, 'VXM': 6,
        }

        # Get roll days for this symbol (default 5 if not specified)
        roll_days = roll_rules.get(symbol, 5)

        # Determine which contract month to use based on typical expiry patterns
        if symbol in ['ES', 'MES', 'NQ', 'MNQ', 'YM', 'MYM', 'RTY', 'M2K']:
            # Equity indices expire on 3rd Friday of month
            # Calculate 3rd Friday of current month
            first_day = datetime(now.year, now.month, 1)
            first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
            third_friday = first_friday + timedelta(weeks=2)

            # Roll if we're within roll_days of expiry
            if now >= third_friday - timedelta(days=roll_days):
                # Move to next quarter month (Mar, Jun, Sep, Dec)
                next_month = now.month + 1 if now.month < 12 else 1
                next_year = now.year if now.month < 12 else now.year + 1
                # Find next quarterly month
                quarter_months = [3, 6, 9, 12]
                for qm in quarter_months:
                    if qm > now.month:
                        next_month = qm
                        break
                else:
                    next_month = 3
                    next_year = now.year + 1
                expiry = datetime(next_year, next_month, 1)
            else:
                expiry = now

        elif symbol in ['CL', 'QM', 'NG', 'QG', 'BZ', 'HO', 'RB']:
            # Energy contracts expire around 20th-25th of month
            # Roll if past the 20th minus roll_days
            if current_day >= (20 - roll_days):
                if now.month == 12:
                    expiry = datetime(now.year + 1, 1, 1)
                else:
                    expiry = datetime(now.year, now.month + 1, 1)
            else:
                expiry = now

        elif symbol in ['GC', 'MGC', 'SI', 'SIL', 'HG', 'QC']:
            # Metals expire on 3rd last business day
            # Roll if past the 25th minus roll_days
            if current_day >= (25 - roll_days):
                if now.month == 12:
                    expiry = datetime(now.year + 1, 1, 1)
                else:
                    expiry = datetime(now.year, now.month + 1, 1)
            else:
                expiry = now

        elif symbol in ['ZC', 'YC', 'ZW', 'YW', 'ZS', 'YK', 'ZM', 'ZL']:
            # Grains expire on 14th of month
            # Roll if past the 14th minus roll_days
            if current_day >= (14 - roll_days):
                if now.month == 12:
                    expiry = datetime(now.year + 1, 1, 1)
                else:
                    expiry = datetime(now.year, now.month + 1, 1)
            else:
                expiry = now

        elif symbol in ['VX', 'VXM']:
            # VIX expires 30 days before S&P options (Wed)
            # Complex calculation - simplified to roll mid-month
            if current_day >= (15 - roll_days):
                expiry = now + timedelta(days=30)
            else:
                expiry = now

        else:
            # Default: Roll if past mid-month
            if current_day >= (15 - roll_days):
                if now.month == 12:
                    expiry = datetime(now.year + 1, 1, 1)
                else:
                    expiry = datetime(now.year, now.month + 1, 1)
            else:
                expiry = now

        # Return in YYYYMM format for IB
        return expiry.strftime('%Y%m')

    def create_ib_contract(self, symbol: str) -> Optional[Contract]:
        """
        Create an IB contract object for a futures symbol

        Args:
            symbol: Futures symbol

        Returns:
            IB Contract object or None
        """
        if not IB_AVAILABLE:
            return None

        if symbol not in self.FUTURES_CONTRACTS:
            logger.error(f"Unknown futures symbol: {symbol}")
            return None

        spec = self.FUTURES_CONTRACTS[symbol]
        expiry = self.get_front_month_expiry(symbol)

        # Create futures contract
        contract = Future(
            symbol=symbol,
            exchange=spec['exchange'],
            lastTradeDateOrContractMonth=expiry,
            currency=spec['currency']
        )

        return contract

    def adjust_futures_data(self, df: pd.DataFrame, symbol: str, adjustment_method: str = "panama") -> pd.DataFrame:
        """
        Adjust historical futures data for contract rolls

        Args:
            df: DataFrame with historical data (potentially multiple contracts)
            symbol: Futures symbol
            adjustment_method: 'panama' (additive), 'ratio' (multiplicative), or 'none'

        Returns:
            Adjusted DataFrame with continuous prices
        """
        if adjustment_method == "none" or df.empty:
            return df

        # For single contract data from standard API call, no adjustment needed
        # Adjustments are primarily for continuous contract data
        if 'contract' not in df.columns:
            # Single contract data - return as is
            return df

        # Sort by date to ensure proper chronological order
        df = df.sort_index()

        # Identify roll points (where contract changes)
        df['contract_change'] = df['contract'] != df['contract'].shift(1)
        roll_dates = df[df['contract_change']].index[1:]  # Skip the first row

        if len(roll_dates) == 0:
            # No rolls to adjust
            df.drop(['contract_change'], axis=1, errors='ignore', inplace=True)
            return df

        # Create adjusted columns
        adj_cols = ['Open', 'High', 'Low', 'Close']
        for col in adj_cols:
            if col in df.columns:
                df[f'adj_{col}'] = df[col].copy()

        if adjustment_method == "panama":
            # Panama/Additive adjustment - preserves absolute price differences
            logger.info(f"Applying Panama (additive) adjustment to {symbol}")

            cumulative_adjustment = 0.0

            for roll_date in roll_dates:
                # Find the price difference at roll point
                # Get last close of old contract and first close of new contract
                idx = df.index.get_loc(roll_date)
                if idx > 0:
                    old_close = df.iloc[idx-1]['Close']
                    new_close = df.iloc[idx]['Close']
                    adjustment = old_close - new_close
                    cumulative_adjustment += adjustment

                    # Apply cumulative adjustment to all data before this roll
                    mask = df.index < roll_date
                    for col in adj_cols:
                        if col in df.columns:
                            df.loc[mask, f'adj_{col}'] += cumulative_adjustment

                    logger.debug(f"Roll on {roll_date}: adjustment={adjustment:.2f}, cumulative={cumulative_adjustment:.2f}")

        elif adjustment_method == "ratio":
            # Ratio/Multiplicative adjustment - preserves percentage returns
            logger.info(f"Applying Ratio (multiplicative) adjustment to {symbol}")

            cumulative_factor = 1.0

            for roll_date in roll_dates:
                # Find the price ratio at roll point
                idx = df.index.get_loc(roll_date)
                if idx > 0:
                    old_close = df.iloc[idx-1]['Close']
                    new_close = df.iloc[idx]['Close']
                    if new_close != 0:
                        ratio = old_close / new_close
                        cumulative_factor *= ratio

                        # Apply cumulative factor to all data before this roll
                        mask = df.index < roll_date
                        for col in adj_cols:
                            if col in df.columns:
                                df.loc[mask, f'adj_{col}'] *= cumulative_factor

                        logger.debug(f"Roll on {roll_date}: ratio={ratio:.4f}, cumulative={cumulative_factor:.4f}")

        # Replace original OHLC with adjusted values
        for col in adj_cols:
            if f'adj_{col}' in df.columns:
                df[col] = df[f'adj_{col}']
                df.drop(f'adj_{col}', axis=1, inplace=True)

        # Clean up temporary columns
        df.drop(['contract_change'], axis=1, errors='ignore', inplace=True)
        if 'contract' in df.columns:
            df.drop('contract', axis=1, inplace=True)

        return df

    def get_continuous_futures_data(self, symbol: str, period: str = "1Y",
                                   bar_size: str = "1 day",
                                   adjustment_method: str = "panama") -> Optional[pd.DataFrame]:
        """
        Get continuous adjusted futures data across multiple contracts

        Args:
            symbol: Futures symbol
            period: Time period for historical data
            bar_size: Bar size (1 day, 1 hour, etc.)
            adjustment_method: 'panama', 'ratio', or 'none'

        Returns:
            Continuous adjusted DataFrame
        """
        if not self.connected:
            logger.warning("Not connected to IB. Call connect() first.")
            return None

        try:
            # For true continuous contracts, we'd fetch multiple contracts
            # For now, we'll fetch the current front month and mark it
            # This allows the adjustment logic to work when we have multi-contract data

            contract = self.create_ib_contract(symbol)
            if not contract:
                return None

            # Request historical data for front month
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=period,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=False,
                formatDate=1
            )

            if not bars:
                logger.warning(f"No data returned for {symbol}")
                return None

            # Convert to DataFrame
            df = util.df(bars)

            # Rename columns
            df = df.rename(columns={
                'date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })

            # Set Date as index
            df.set_index('Date', inplace=True)

            # Apply adjustment method
            df = self.adjust_futures_data(df, symbol, adjustment_method)

            # Add metadata
            df.attrs['symbol'] = symbol
            df.attrs['adjustment_method'] = adjustment_method
            df.attrs['contract_expiry'] = self.get_front_month_expiry(symbol)

            return df

        except Exception as e:
            logger.error(f"Error fetching continuous data for {symbol}: {e}")
            return None

    def get_futures_data(self, symbol: str, period: str = "1D", bar_size: str = "1 hour",
                         adjustment_method: str = "panama") -> Optional[pd.DataFrame]:
        """
        Get historical futures data from Interactive Brokers

        Args:
            symbol: Futures symbol
            period: Time period (1D, 1W, 1M, 3M, 1Y)
            bar_size: Bar size (1 min, 5 mins, 1 hour, 1 day)
            adjustment_method: 'panama' (additive), 'ratio' (multiplicative), or 'none'

        Returns:
            DataFrame with OHLCV data (adjusted if specified)
        """
        if not self.connected:
            logger.warning("Not connected to IB. Call connect() first.")
            return None

        # Check cache
        cache_key = f"{symbol}_{period}_{bar_size}"
        if cache_key in self.cache:
            if time.time() < self.cache_expiry[cache_key]:
                return self.cache[cache_key]

        try:
            # Create contract
            contract = self.create_ib_contract(symbol)
            if not contract:
                return None

            # Request historical data
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=period,
                barSizeSetting=bar_size,
                whatToShow='TRADES',
                useRTH=False,
                formatDate=1
            )

            if not bars:
                logger.warning(f"No data returned for {symbol}")
                return None

            # Convert to DataFrame
            df = util.df(bars)

            # Rename columns to match expected format
            df = df.rename(columns={
                'date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })

            # Set Date as index
            df.set_index('Date', inplace=True)

            # Apply adjustment if requested
            df = self.adjust_futures_data(df, symbol, adjustment_method)

            # Cache the data
            self.cache[cache_key] = df
            self.cache_expiry[cache_key] = time.time() + self.cache_duration

            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

    def get_real_time_price(self, symbol: str) -> Optional[float]:
        """
        Get real-time price for a futures contract

        Args:
            symbol: Futures symbol

        Returns:
            Current price or None
        """
        if not self.connected:
            return None

        try:
            contract = self.create_ib_contract(symbol)
            if not contract:
                return None

            # Request market data
            ticker = self.ib.reqMktData(contract, '', False, False)

            # Wait for data (max 5 seconds)
            self.ib.sleep(2)

            if ticker.last and not np.isnan(ticker.last):
                return ticker.last
            elif ticker.close and not np.isnan(ticker.close):
                return ticker.close

            return None

        except Exception as e:
            logger.error(f"Error getting real-time price for {symbol}: {e}")
            return None

    def get_available_futures(self) -> Dict[str, List[str]]:
        """
        Get all available futures organized by category

        Returns:
            Dictionary of categories and their symbols
        """
        categories = {}
        for symbol, spec in self.FUTURES_CONTRACTS.items():
            category = spec['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(symbol)

        return categories

    def get_futures_info(self, symbol: str) -> Dict:
        """
        Get detailed information about a futures contract

        Args:
            symbol: Futures symbol

        Returns:
            Contract specifications
        """
        if symbol in self.FUTURES_CONTRACTS:
            spec = self.FUTURES_CONTRACTS[symbol].copy()
            spec['symbol'] = symbol
            spec['expiry'] = self.get_front_month_expiry(symbol)
            spec['data_source'] = 'Interactive Brokers'
            return spec

        return {
            'symbol': symbol,
            'name': f'Unknown Contract {symbol}',
            'data_source': 'Interactive Brokers',
            'error': 'Contract not found'
        }

    def is_futures_symbol(self, symbol: str) -> bool:
        """
        Check if a symbol is a recognized futures contract

        Args:
            symbol: Symbol to check

        Returns:
            True if it's a futures symbol
        """
        return symbol.upper() in self.FUTURES_CONTRACTS

    def get_contract_multiplier(self, symbol: str) -> float:
        """
        Get the contract multiplier for a futures symbol

        Args:
            symbol: Futures symbol

        Returns:
            Contract multiplier
        """
        if symbol in self.FUTURES_CONTRACTS:
            return self.FUTURES_CONTRACTS[symbol]['multiplier']
        return 1.0


# Global instance
ib_futures_fetcher = None

def initialize_ib_connection(host='127.0.0.1', port=7497, client_id=1):
    """
    Initialize the IB connection

    Args:
        host: TWS/Gateway host
        port: Connection port
        client_id: Client ID
    """
    global ib_futures_fetcher
    ib_futures_fetcher = IBFuturesDataFetcher(host, port, client_id)
    return ib_futures_fetcher.connect()

def get_ib_futures_data(symbol: str, period: str = "1D", bar_size: str = "1 hour",
                       adjustment_method: str = "panama") -> Optional[pd.DataFrame]:
    """
    Get futures data from IB with optional adjustment

    Args:
        symbol: Futures symbol
        period: Time period
        bar_size: Bar size
        adjustment_method: 'panama' (additive), 'ratio' (multiplicative), or 'none'

    Returns:
        DataFrame with OHLCV data (adjusted if specified)
    """
    global ib_futures_fetcher
    if not ib_futures_fetcher:
        logger.error("IB not initialized. Call initialize_ib_connection() first.")
        return None

    return ib_futures_fetcher.get_futures_data(symbol, period, bar_size, adjustment_method)

def is_futures_symbol(symbol: str) -> bool:
    """
    Check if a symbol is a futures contract

    Args:
        symbol: Symbol to check

    Returns:
        True if it's a futures symbol
    """
    if not ib_futures_fetcher:
        # Create temporary instance just for checking
        temp = IBFuturesDataFetcher()
        return temp.is_futures_symbol(symbol)
    return ib_futures_fetcher.is_futures_symbol(symbol)
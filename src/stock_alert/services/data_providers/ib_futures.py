"""
Interactive Brokers futures data provider.

Supports all major futures contracts with front month rolling.
Adapted from backend_futures_ib.py
"""

import json
import logging
import os
from datetime import datetime, timedelta

import pandas as pd

# IB API imports (install with: pip install ib_insync)
try:
    from ib_insync import IB, Contract, Future, util

    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    IB = None
    Future = None
    Contract = None
    util = None

logger = logging.getLogger(__name__)


class IBFuturesProvider:
    """
    Interactive Brokers futures data provider.

    Handles all major futures contracts with automatic front month rolling.
    Provides historical and real-time data for futures markets.
    """

    # Comprehensive futures contract specifications
    FUTURES_CONTRACTS = {
        # Energy
        "CL": {
            "exchange": "NYMEX",
            "name": "WTI Crude Oil",
            "category": "Energy",
            "multiplier": 1000,
            "currency": "USD",
        },
        "NG": {
            "exchange": "NYMEX",
            "name": "Natural Gas",
            "category": "Energy",
            "multiplier": 10000,
            "currency": "USD",
        },
        "HO": {
            "exchange": "NYMEX",
            "name": "Heating Oil",
            "category": "Energy",
            "multiplier": 42000,
            "currency": "USD",
        },
        "RB": {
            "exchange": "NYMEX",
            "name": "RBOB Gasoline",
            "category": "Energy",
            "multiplier": 42000,
            "currency": "USD",
        },
        # Precious Metals
        "GC": {
            "exchange": "COMEX",
            "name": "Gold",
            "category": "Metals",
            "multiplier": 100,
            "currency": "USD",
        },
        "SI": {
            "exchange": "COMEX",
            "name": "Silver",
            "category": "Metals",
            "multiplier": 5000,
            "currency": "USD",
        },
        "HG": {
            "exchange": "COMEX",
            "name": "Copper",
            "category": "Metals",
            "multiplier": 25000,
            "currency": "USD",
        },
        "PL": {
            "exchange": "NYMEX",
            "name": "Platinum",
            "category": "Metals",
            "multiplier": 50,
            "currency": "USD",
        },
        # Stock Index Futures
        "ES": {
            "exchange": "CME",
            "name": "E-mini S&P 500",
            "category": "Indices",
            "multiplier": 50,
            "currency": "USD",
        },
        "NQ": {
            "exchange": "CME",
            "name": "E-mini Nasdaq 100",
            "category": "Indices",
            "multiplier": 20,
            "currency": "USD",
        },
        "YM": {
            "exchange": "CBOT",
            "name": "E-mini Dow Jones",
            "category": "Indices",
            "multiplier": 5,
            "currency": "USD",
        },
        "RTY": {
            "exchange": "CME",
            "name": "E-mini Russell 2000",
            "category": "Indices",
            "multiplier": 50,
            "currency": "USD",
        },
        # Agricultural
        "ZC": {
            "exchange": "CBOT",
            "name": "Corn",
            "category": "Agriculture",
            "multiplier": 50,
            "currency": "USD",
        },
        "ZW": {
            "exchange": "CBOT",
            "name": "Chicago Wheat",
            "category": "Agriculture",
            "multiplier": 50,
            "currency": "USD",
        },
        "ZS": {
            "exchange": "CBOT",
            "name": "Soybeans",
            "category": "Agriculture",
            "multiplier": 50,
            "currency": "USD",
        },
        # Currencies
        "6E": {
            "exchange": "CME",
            "name": "Euro FX",
            "category": "Currencies",
            "multiplier": 125000,
            "currency": "USD",
        },
        "6B": {
            "exchange": "CME",
            "name": "British Pound",
            "category": "Currencies",
            "multiplier": 62500,
            "currency": "USD",
        },
        "6J": {
            "exchange": "CME",
            "name": "Japanese Yen",
            "category": "Currencies",
            "multiplier": 12500000,
            "currency": "USD",
        },
        # Interest Rates/Bonds
        "ZN": {
            "exchange": "CBOT",
            "name": "10-Year T-Note",
            "category": "Bonds",
            "multiplier": 1000,
            "currency": "USD",
        },
        "ZB": {
            "exchange": "CBOT",
            "name": "30-Year T-Bond",
            "category": "Bonds",
            "multiplier": 1000,
            "currency": "USD",
        },
        # Volatility
        "VX": {
            "exchange": "CFE",
            "name": "VIX Futures",
            "category": "Volatility",
            "multiplier": 1000,
            "currency": "USD",
        },
    }

    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 1):
        """
        Initialize IB Futures Provider.

        Args:
            host: TWS/Gateway host (default localhost)
            port: TWS port 7497 or Gateway port 4001
            client_id: Client ID for IB API
        """
        if not IB_AVAILABLE:
            logger.warning("ib_insync not installed. Run: pip install ib_insync")

        self.host = host
        self.port = port
        self.client_id = client_id
        self.ib = None
        self.connected = False

        # Cache for real-time data
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = 60  # 1 minute cache

        # Load configuration
        self.config = self._load_config()

    @property
    def name(self) -> str:
        """Return provider name"""
        return "Interactive Brokers Futures"

    def _load_config(self) -> dict:
        """Load IB configuration from file"""
        config_file = "ib_futures_config.json"
        if os.path.exists(config_file):
            try:
                with open(config_file) as f:
                    config = json.load(f)
                    # Update connection parameters if provided
                    self.host = config.get("host", self.host)
                    self.port = config.get("port", self.port)
                    self.client_id = config.get("client_id", self.client_id)
                    return config
            except Exception as e:
                logger.error(f"Error loading IB config: {e}")
        return {}

    def connect(self) -> bool:
        """
        Connect to Interactive Brokers TWS or Gateway.

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

    def is_futures_symbol(self, symbol: str) -> bool:
        """
        Check if symbol is a valid futures contract.

        Args:
            symbol: Futures symbol (e.g., 'ES', 'CL')

        Returns:
            True if valid futures symbol
        """
        return symbol.upper() in self.FUTURES_CONTRACTS

    def get_contract_info(self, symbol: str) -> dict | None:
        """
        Get contract information for a futures symbol.

        Args:
            symbol: Futures symbol

        Returns:
            Dictionary with contract information
        """
        return self.FUTURES_CONTRACTS.get(symbol.upper())

    def get_contract_multiplier(self, symbol: str) -> float:
        """
        Get the contract multiplier for point value calculations.

        Args:
            symbol: Futures symbol

        Returns:
            Multiplier value
        """
        info = self.get_contract_info(symbol)
        return info.get("multiplier", 1.0) if info else 1.0

    def get_available_futures(self) -> dict[str, list[str]]:
        """
        Get all available futures grouped by category.

        Returns:
            Dictionary mapping category to list of symbols
        """
        futures_by_category = {}
        for symbol, info in self.FUTURES_CONTRACTS.items():
            category = info["category"]
            if category not in futures_by_category:
                futures_by_category[category] = []
            futures_by_category[category].append(symbol)
        return futures_by_category

    def get_historical_prices(
        self,
        symbol: str,
        period: str = "1Y",
        bar_size: str = "1 day",
        adjustment_method: str = "panama",
    ) -> pd.DataFrame:
        """
        Fetch historical futures prices with continuous contract rolling.

        Args:
            symbol: Futures symbol (e.g., 'ES', 'CL')
            period: Time period (e.g., '1Y', '6M', '30D')
            bar_size: Bar size (e.g., '1 day', '1 hour')
            adjustment_method: Roll adjustment method ('panama', 'ratio', 'difference')

        Returns:
            DataFrame with OHLCV data
        """
        if not self.is_futures_symbol(symbol):
            logger.error(f"Invalid futures symbol: {symbol}")
            return pd.DataFrame()

        if not self.connected:
            if not self.connect():
                return pd.DataFrame()

        try:
            # Create continuous futures contract
            contract_info = self.get_contract_info(symbol)
            contract = Future(symbol=symbol, exchange=contract_info["exchange"])

            # Fetch historical data
            bars = self.ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr=period,
                barSizeSetting=bar_size,
                whatToShow="TRADES",
                useRTH=False,
                formatDate=1,
            )

            if not bars:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = util.df(bars)

            if df.empty:
                return df

            # Rename columns to match standard format
            df = df.rename(
                columns={
                    "date": "date",
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                }
            )

            # Ensure date is datetime
            df["date"] = pd.to_datetime(df["date"])

            # Sort by date
            df = df.sort_values("date").reset_index(drop=True)

            # Select only needed columns
            columns = ["date", "open", "high", "low", "close", "volume"]
            df = df[[col for col in columns if col in df.columns]]

            return df

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

    def get_latest_price(self, symbol: str) -> float | None:
        """
        Fetch the latest price for a futures contract.

        Args:
            symbol: Futures symbol

        Returns:
            Latest price or None
        """
        # Check cache first
        cache_key = f"{symbol}_price"
        if cache_key in self.cache:
            if datetime.now() < self.cache_expiry.get(cache_key, datetime.min):
                return self.cache[cache_key]

        if not self.connected:
            if not self.connect():
                return None

        try:
            contract_info = self.get_contract_info(symbol)
            contract = Future(symbol=symbol, exchange=contract_info["exchange"])

            # Request market data
            ticker = self.ib.reqMktData(contract)
            self.ib.sleep(2)  # Wait for data

            price = ticker.last
            if not price or price <= 0:
                price = ticker.close
            if not price or price <= 0:
                price = (ticker.bid + ticker.ask) / 2 if ticker.bid and ticker.ask else None

            # Cache the result
            if price:
                self.cache[cache_key] = price
                self.cache_expiry[cache_key] = datetime.now() + timedelta(
                    seconds=self.cache_duration
                )

            return price

        except Exception as e:
            logger.error(f"Error fetching latest price for {symbol}: {e}")
            return None

    def validate_ticker(self, symbol: str) -> bool:
        """
        Check if a futures symbol is valid.

        Args:
            symbol: Futures symbol

        Returns:
            True if valid
        """
        return self.is_futures_symbol(symbol)

    def close(self):
        """Close connection and cleanup"""
        self.disconnect()

    def __enter__(self):
        """Context manager entry"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

"""
Financial Modeling Prep (FMP) Data Fetcher
Provides historical and real-time market data for stocks and ETFs
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging
import os
import time

logger = logging.getLogger(__name__)


class FMPDataFetcher:
    """
    Data fetcher for Financial Modeling Prep API
    Supports daily, weekly, and hourly/intraday data
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FMP data fetcher
        
        Args:
            api_key: Optional FMP API key. If not provided, uses environment variable.
        """
        self.api_key = api_key or os.getenv('FMP_API_KEY', '')
        self.base_url = "https://financialmodelingprep.com/api/v3"
        
        if not self.api_key:
            logger.warning("FMP_API_KEY not set - API calls will fail")

    def get_historical_data(
        self, 
        ticker: str, 
        period: str = "1day", 
        timeframe: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical data for a ticker
        
        Args:
            ticker: Stock symbol (e.g., "AAPL", "MSFT")
            period: Data period/interval:
                - "1day" - daily data
                - "1min", "5min", "15min", "30min" - intraday
                - "1hour", "4hour" - hourly data
            timeframe: Optional timeframe for resampling:
                - "1d" - daily (default)
                - "1wk" - weekly (resamples daily to weekly)
                
        Returns:
            DataFrame with OHLCV data, indexed by date/datetime
            Returns None on error or if no data available
        """
        try:
            # Handle weekly timeframe - fetch daily and resample
            if timeframe in ["1wk", "weekly", "1week"]:
                logger.debug(f"{ticker}: Fetching daily data for weekly resampling")
                df = self._fetch_daily_data(ticker, limit=750)
                if df is not None and not df.empty:
                    df = self._resample_to_weekly(df)
                return df
            
            # Handle daily data
            if period in ["1day", "daily"]:
                return self._fetch_daily_data(ticker, limit=750)
            
            # Handle intraday/hourly data
            return self._fetch_intraday_data(ticker, period)
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None

    def _fetch_daily_data(self, ticker: str, limit: int = 750) -> Optional[pd.DataFrame]:
        """
        Fetch daily historical data from FMP API
        
        Args:
            ticker: Stock symbol
            limit: Number of days to fetch (default 750 for ~3 years)
            
        Returns:
            DataFrame with daily OHLCV data
        """
        try:
            url = f"{self.base_url}/historical-price-full/{ticker}"
            params = {
                "apikey": self.api_key,
                "limit": limit
            }
            
            logger.debug(f"Fetching daily data for {ticker} (limit: {limit})")
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if "historical" in data and data["historical"]:
                    df = pd.DataFrame(data["historical"])
                    
                    # Process the data
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                    
                    # Standardize column names (FMP uses lowercase)
                    column_mapping = {
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume',
                        'adjClose': 'Adj Close',
                        'change': 'Change',
                        'changePercent': 'Change %',
                        'vwap': 'VWAP',
                        'changeOverTime': 'Change Over Time'
                    }
                    df = df.rename(columns=column_mapping)
                    
                    # Set date as index
                    df = df.set_index('date')
                    
                    logger.debug(f"{ticker}: Fetched {len(df)} daily records")
                    return df
                else:
                    logger.warning(f"{ticker}: No historical data in API response")
                    return None
            else:
                logger.error(f"{ticker}: API error {response.status_code}")
                if response.status_code == 401:
                    logger.error("API key is invalid or missing")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching daily data for {ticker}: {e}")
            return None

    def _fetch_intraday_data(self, ticker: str, interval: str = "1hour") -> Optional[pd.DataFrame]:
        """
        Fetch intraday/hourly data from FMP API
        
        Args:
            ticker: Stock symbol
            interval: Time interval (1min, 5min, 15min, 30min, 1hour, 4hour)
            
        Returns:
            DataFrame with intraday OHLCV data
        """
        try:
            # Map common interval names to FMP format
            interval_map = {
                "1min": "1min",
                "5min": "5min",
                "15min": "15min",
                "30min": "30min",
                "1hour": "1hour",
                "1hr": "1hour",
                "hourly": "1hour",
                "4hour": "4hour"
            }
            
            fmp_interval = interval_map.get(interval, interval)
            
            url = f"{self.base_url}/historical-chart/{fmp_interval}/{ticker}"
            params = {"apikey": self.api_key}
            
            logger.debug(f"Fetching {fmp_interval} data for {ticker}")
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data and len(data) > 0:
                    df = pd.DataFrame(data)
                    
                    # Process the data
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                    
                    # Standardize column names
                    column_mapping = {
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume'
                    }
                    df = df.rename(columns=column_mapping)
                    
                    # Set date as index
                    df = df.set_index('date')
                    
                    logger.debug(f"{ticker}: Fetched {len(df)} {fmp_interval} records")
                    return df
                else:
                    logger.warning(f"{ticker}: No intraday data available")
                    return None
            else:
                logger.error(f"{ticker}: API error {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching intraday data for {ticker}: {e}")
            return None

    def get_hourly_data(
        self, 
        ticker: str, 
        from_date: Optional[str] = None, 
        to_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch hourly data for a ticker within a date range
        Uses chunking for large date ranges to handle API limitations
        
        Args:
            ticker: Stock symbol
            from_date: Start date in YYYY-MM-DD format (default: 2 years ago)
            to_date: End date in YYYY-MM-DD format (default: today)
            
        Returns:
            DataFrame with hourly OHLCV data
        """
        try:
            # Set default date range if not provided
            if not to_date:
                to_date = datetime.now().strftime('%Y-%m-%d')
            if not from_date:
                from_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
            
            # Convert to datetime for calculations
            start_dt = datetime.strptime(from_date, '%Y-%m-%d')
            end_dt = datetime.strptime(to_date, '%Y-%m-%d')
            
            # FMP API limitation: can only fetch ~90 days of hourly data at once
            # Split into 60-day chunks for safety
            chunk_days = 60
            all_data = []
            
            current_start = start_dt
            while current_start < end_dt:
                current_end = min(current_start + timedelta(days=chunk_days), end_dt)
                
                chunk_from = current_start.strftime('%Y-%m-%d')
                chunk_to = current_end.strftime('%Y-%m-%d')
                
                logger.debug(f"{ticker}: Fetching hourly chunk {chunk_from} to {chunk_to}")
                
                chunk_df = self._fetch_hourly_chunk(ticker, chunk_from, chunk_to)
                
                if chunk_df is not None and not chunk_df.empty:
                    all_data.append(chunk_df)
                
                # Move to next chunk
                current_start = current_end + timedelta(days=1)
                
                # Rate limiting - sleep briefly between requests
                time.sleep(0.2)
            
            # Combine all chunks
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=False)
                combined_df = combined_df.sort_index()
                
                # Remove duplicates (can happen at chunk boundaries)
                combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
                
                logger.info(f"{ticker}: Fetched {len(combined_df)} hourly records")
                return combined_df
            else:
                logger.warning(f"{ticker}: No hourly data available")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching hourly data for {ticker}: {e}")
            return None

    def _fetch_hourly_chunk(
        self, 
        ticker: str, 
        from_date: str, 
        to_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch a single chunk of hourly data
        Internal method used by get_hourly_data
        
        Args:
            ticker: Stock symbol
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with hourly OHLCV data for the date range
        """
        try:
            url = f"{self.base_url}/historical-chart/1hour/{ticker}"
            params = {
                "apikey": self.api_key,
                "from": from_date,
                "to": to_date
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data and len(data) > 0:
                    df = pd.DataFrame(data)
                    
                    # Process the data
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                    
                    # Standardize column names
                    column_mapping = {
                        'open': 'Open',
                        'high': 'High',
                        'low': 'Low',
                        'close': 'Close',
                        'volume': 'Volume'
                    }
                    df = df.rename(columns=column_mapping)
                    
                    # Set date as index
                    df = df.set_index('date')
                    
                    logger.debug(f"{ticker}: Fetched {len(df)} hourly records for chunk")
                    return df
                else:
                    logger.debug(f"{ticker}: No data for chunk {from_date} to {to_date}")
                    return pd.DataFrame()
            else:
                logger.error(f"{ticker}: API error {response.status_code} for chunk")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching hourly chunk for {ticker}: {e}")
            return None

    def _resample_to_weekly(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Resample daily data to weekly OHLCV
        Uses actual last trading day of each week
        
        Args:
            df: DataFrame with daily OHLCV data
            
        Returns:
            DataFrame with weekly OHLCV data
        """
        try:
            if df.empty:
                return None
            
            # Ensure we have a datetime index
            if 'date' in df.columns:
                df = df.set_index('date')
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Create a copy to avoid modifying the original
            df_copy = df.copy()
            
            # Add week identifier (year-week number using Sunday start)
            df_copy['year_week'] = df_copy.index.strftime('%Y-%U')
            
            # Group by week and aggregate
            weekly_data = []
            today = pd.Timestamp.now().normalize()
            current_week_start = today - pd.Timedelta(days=today.weekday() + 1)
            
            for year_week, week_df in df_copy.groupby('year_week'):
                if week_df.empty:
                    continue
                
                # Use the ACTUAL last trading day of the week as the date
                # This ensures we use Thursday if there's no Friday trading, etc.
                last_trading_day = week_df.index[-1]
                
                # Skip partial weeks (current week Mon-Thu only)
                # A week is "complete" if it's from a past week
                if last_trading_day >= current_week_start:
                    # This is current week data
                    if today.weekday() < 4:  # Today is Mon-Thu
                        continue  # Skip current week's partial data
                
                # Calculate weekly OHLCV using actual trading days
                weekly_record = {
                    'date': last_trading_day,  # Use actual last trading day
                    'Open': week_df['Open'].iloc[0],  # First open of the week
                    'High': week_df['High'].max(),     # Highest high of the week
                    'Low': week_df['Low'].min(),       # Lowest low of the week
                    'Close': week_df['Close'].iloc[-1], # Last close of the week
                    'Volume': week_df['Volume'].sum()   # Total volume for the week
                }
                weekly_data.append(weekly_record)
            
            if not weekly_data:
                return None
            
            # Create DataFrame from weekly data
            weekly_df = pd.DataFrame(weekly_data)
            weekly_df.set_index('date', inplace=True)
            
            logger.debug(f"Resampled to {len(weekly_df)} weekly records")
            return weekly_df
            
        except Exception as e:
            logger.error(f"Error resampling to weekly: {e}")
            return None

    def get_quote(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get real-time quote for a symbol
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with quote data including:
            - price, change, changePercent
            - open, high, low, volume
            - previousClose, timestamp
            Returns None on error
        """
        try:
            url = f"{self.base_url}/quote/{symbol}"
            params = {"apikey": self.api_key}
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    return data[0]
                else:
                    logger.warning(f"{symbol}: No quote data available")
                    return None
            else:
                logger.error(f"{symbol}: API error {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return None

    def get_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get company profile information
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Dictionary with company profile data
        """
        try:
            url = f"{self.base_url}/profile/{symbol}"
            params = {"apikey": self.api_key}
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    return data[0]
                return None
            else:
                logger.error(f"{symbol}: API error {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching profile for {symbol}: {e}")
            return None

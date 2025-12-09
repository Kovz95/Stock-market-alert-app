"""
Pivot-based Support and Resistance Line Detector
Creates horizontal support/resistance lines from pivot points and tracks them over time
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

class PivotSupportResistance:
    def __init__(self, ticker: str, left_bars: int = 5, right_bars: int = 5,
                 proximity_threshold: float = 1.0, buffer_percent: float = 0.5,
                 lookback_years: int = 3):
        """
        Initialize Pivot Support/Resistance Detector

        Args:
            ticker: Stock ticker symbol
            left_bars: Number of bars to the left of pivot for validation
            right_bars: Number of bars to the right of pivot for validation
            proximity_threshold: Percentage threshold for proximity alerts (e.g., 1.0 = 1%)
            buffer_percent: Minimum % separation between lines to avoid clustering
            lookback_years: Number of years to maintain historical lines (rolling basis)
        """
        self.ticker = ticker
        self.left_bars = left_bars
        self.right_bars = right_bars
        self.proximity_threshold = proximity_threshold
        self.buffer_percent = buffer_percent
        self.lookback_years = lookback_years

        # Create pivot_levels folder if it doesn't exist
        self.storage_dir = 'pivot_levels'
        os.makedirs(self.storage_dir, exist_ok=True)

        # Storage file for this ticker's levels
        self.storage_file = os.path.join(self.storage_dir, f'{ticker}_pivot_levels.json')

        # Load existing levels
        self.levels = self.load_levels()

    def load_levels(self) -> Dict:
        """Load existing support/resistance levels from storage"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    data = json.load(f)
                    # Convert date strings back to datetime for comparison
                    for level_type in ['support', 'resistance']:
                        if level_type in data:
                            for level in data[level_type]:
                                level['first_detected'] = pd.to_datetime(level['first_detected'])
                                level['last_touched'] = pd.to_datetime(level['last_touched'])
                    return data
            except:
                return {'support': [], 'resistance': [], 'last_update': None}
        return {'support': [], 'resistance': [], 'last_update': None}

    def save_levels(self):
        """Save support/resistance levels to storage"""
        # Convert datetime objects to strings for JSON serialization
        data_to_save = {
            'support': [],
            'resistance': [],
            'last_update': datetime.now().isoformat()
        }

        for level_type in ['support', 'resistance']:
            for level in self.levels.get(level_type, []):
                level_copy = level.copy()
                level_copy['first_detected'] = level_copy['first_detected'].isoformat()
                level_copy['last_touched'] = level_copy['last_touched'].isoformat()
                data_to_save[level_type].append(level_copy)

        with open(self.storage_file, 'w') as f:
            json.dump(data_to_save, f, indent=2)

    def find_pivot_highs(self, df: pd.DataFrame) -> List[Tuple[int, float]]:
        """
        Find pivot highs (potential resistance levels)

        Returns:
            List of tuples (index, price) for pivot highs
        """
        highs = df['High'].values
        pivot_highs = []

        for i in range(self.left_bars, len(highs) - self.right_bars):
            is_pivot = True
            pivot_value = highs[i]

            # Check left bars
            for j in range(i - self.left_bars, i):
                if highs[j] >= pivot_value:
                    is_pivot = False
                    break

            # Check right bars
            if is_pivot:
                for j in range(i + 1, i + self.right_bars + 1):
                    if highs[j] >= pivot_value:
                        is_pivot = False
                        break

            if is_pivot:
                pivot_highs.append((i, pivot_value))

        return pivot_highs

    def find_pivot_lows(self, df: pd.DataFrame) -> List[Tuple[int, float]]:
        """
        Find pivot lows (potential support levels)

        Returns:
            List of tuples (index, price) for pivot lows
        """
        lows = df['Low'].values
        pivot_lows = []

        for i in range(self.left_bars, len(lows) - self.right_bars):
            is_pivot = True
            pivot_value = lows[i]

            # Check left bars
            for j in range(i - self.left_bars, i):
                if lows[j] <= pivot_value:
                    is_pivot = False
                    break

            # Check right bars
            if is_pivot:
                for j in range(i + 1, i + self.right_bars + 1):
                    if lows[j] <= pivot_value:
                        is_pivot = False
                        break

            if is_pivot:
                pivot_lows.append((i, pivot_value))

        return pivot_lows

    def is_within_buffer(self, new_price: float, existing_levels: List[Dict]) -> bool:
        """
        Check if a new price level is within buffer zone of existing levels

        Args:
            new_price: The new price level to check
            existing_levels: List of existing level dictionaries

        Returns:
            True if too close to existing levels, False otherwise
        """
        for level in existing_levels:
            existing_price = level['price']
            percent_diff = abs((new_price - existing_price) / existing_price * 100)
            if percent_diff < self.buffer_percent:
                return True
        return False

    def update_or_add_level(self, price: float, level_type: str, date: pd.Timestamp):
        """
        Update existing level or add new one if not within buffer

        Args:
            price: Price level
            level_type: 'support' or 'resistance'
            date: Date when level was detected
        """
        levels = self.levels[level_type]

        # Check if this price is within buffer of existing levels
        for level in levels:
            existing_price = level['price']
            percent_diff = abs((price - existing_price) / existing_price * 100)
            if percent_diff < self.buffer_percent:
                # Update existing level
                level['touches'] += 1
                level['last_touched'] = date
                return

        # Add new level if not within buffer
        new_level = {
            'price': price,
            'first_detected': date,
            'last_touched': date,
            'touches': 1,
            'strength': 1  # Can be increased based on number of touches
        }
        levels.append(new_level)

    def clean_old_levels(self, current_date: pd.Timestamp, earliest_data_date: pd.Timestamp = None):
        """
        Remove levels older than lookback_years (or earliest available data)

        Args:
            current_date: Current date for comparison
            earliest_data_date: Earliest date in available data (optional)
        """
        cutoff_date = current_date - pd.Timedelta(days=self.lookback_years * 365)

        # If we know the earliest data date, don't remove levels newer than that
        if earliest_data_date and earliest_data_date > cutoff_date:
            cutoff_date = earliest_data_date

        for level_type in ['support', 'resistance']:
            self.levels[level_type] = [
                level for level in self.levels[level_type]
                if level['last_touched'] >= cutoff_date
            ]

    def initialize_levels(self, df: pd.DataFrame):
        """
        Initialize support/resistance levels from historical data (up to 3 years)
        Uses all available data if less than 3 years exists

        Args:
            df: DataFrame with OHLCV data
        """
        print(f"Initializing pivot levels for {self.ticker}...")

        # Ensure we have datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Get up to 3 years of data, or all available data if less
        cutoff_date = df.index[-1] - pd.Timedelta(days=self.lookback_years * 365)

        # Use all data if it's less than 3 years, otherwise use 3 years
        if df.index[0] > cutoff_date:
            # Stock has less than 3 years of data, use all of it
            df_lookback = df.copy()
            print(f"Using all available data ({len(df)} days) - less than {self.lookback_years} years available")
        else:
            # Use 3 years of data
            df_lookback = df[df.index >= cutoff_date].copy()
            print(f"Using {self.lookback_years} years of data ({len(df_lookback)} days)")

        if len(df_lookback) < (self.left_bars + self.right_bars + 1):
            print(f"Insufficient data for pivot calculation. Need at least {self.left_bars + self.right_bars + 1} bars")
            return

        # Clear existing levels for fresh initialization
        self.levels = {'support': [], 'resistance': []}

        # Find all pivot points
        pivot_highs = self.find_pivot_highs(df_lookback)
        pivot_lows = self.find_pivot_lows(df_lookback)

        # Process pivot highs (resistance)
        for idx, price in pivot_highs:
            date = df_lookback.index[idx]
            self.update_or_add_level(price, 'resistance', date)

        # Process pivot lows (support)
        for idx, price in pivot_lows:
            date = df_lookback.index[idx]
            self.update_or_add_level(price, 'support', date)

        # Sort levels by price
        self.levels['support'].sort(key=lambda x: x['price'])
        self.levels['resistance'].sort(key=lambda x: x['price'])

        # Save to file
        self.save_levels()

        print(f"Initialized {len(self.levels['support'])} support and {len(self.levels['resistance'])} resistance levels")

    def update_levels(self, df: pd.DataFrame):
        """
        Update levels with new data (check for new pivots)

        Args:
            df: DataFrame with recent OHLCV data
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        current_date = df.index[-1]

        # Check if we need to initialize (first run or no levels)
        if not self.levels['support'] and not self.levels['resistance']:
            self.initialize_levels(df)
            return

        # Get recent data for pivot detection (need enough bars for left and right)
        min_bars_needed = self.left_bars + self.right_bars + 1
        if len(df) >= min_bars_needed:
            # Check recent portion for new pivots
            recent_df = df.tail(min_bars_needed * 2)  # Check recent bars for new pivots

            # Find new pivots
            pivot_highs = self.find_pivot_highs(recent_df)
            pivot_lows = self.find_pivot_lows(recent_df)

            # Add any new pivots from recent data
            for idx, price in pivot_highs:
                if idx >= len(recent_df) - self.right_bars - 1:  # Only truly new pivots
                    date = recent_df.index[idx]
                    self.update_or_add_level(price, 'resistance', date)

            for idx, price in pivot_lows:
                if idx >= len(recent_df) - self.right_bars - 1:  # Only truly new pivots
                    date = recent_df.index[idx]
                    self.update_or_add_level(price, 'support', date)

        # Clean old levels (pass earliest data date to prevent removing valid levels for stocks with < 3 years)
        earliest_data_date = df.index[0] if len(df) > 0 else None
        self.clean_old_levels(current_date, earliest_data_date)

        # Sort levels
        self.levels['support'].sort(key=lambda x: x['price'])
        self.levels['resistance'].sort(key=lambda x: x['price'])

        # Save updated levels
        self.save_levels()

    def check_proximity_and_crossover(self, df: pd.DataFrame) -> Dict:
        """
        Check if current price is near or has crossed support/resistance levels

        Args:
            df: DataFrame with price data

        Returns:
            Dictionary with proximity and crossover alerts
        """
        if len(df) < 2:
            return {'proximity': [], 'crossover': []}

        current_price = df['Close'].iloc[-1]
        prev_price = df['Close'].iloc[-2]

        alerts = {
            'proximity': [],
            'crossover': []
        }

        # Check support levels
        for level in self.levels['support']:
            level_price = level['price']
            distance_percent = abs((current_price - level_price) / level_price * 100)

            # Check proximity
            if distance_percent <= self.proximity_threshold:
                alerts['proximity'].append({
                    'type': 'support',
                    'level_price': level_price,
                    'current_price': current_price,
                    'distance_percent': distance_percent,
                    'touches': level['touches'],
                    'first_detected': level['first_detected'],
                    'last_touched': level['last_touched']
                })

            # Check crossover
            if prev_price > level_price and current_price <= level_price:
                alerts['crossover'].append({
                    'type': 'support',
                    'direction': 'down',
                    'level_price': level_price,
                    'current_price': current_price,
                    'prev_price': prev_price,
                    'touches': level['touches']
                })
            elif prev_price < level_price and current_price >= level_price:
                alerts['crossover'].append({
                    'type': 'support',
                    'direction': 'up',
                    'level_price': level_price,
                    'current_price': current_price,
                    'prev_price': prev_price,
                    'touches': level['touches']
                })

        # Check resistance levels
        for level in self.levels['resistance']:
            level_price = level['price']
            distance_percent = abs((current_price - level_price) / level_price * 100)

            # Check proximity
            if distance_percent <= self.proximity_threshold:
                alerts['proximity'].append({
                    'type': 'resistance',
                    'level_price': level_price,
                    'current_price': current_price,
                    'distance_percent': distance_percent,
                    'touches': level['touches'],
                    'first_detected': level['first_detected'],
                    'last_touched': level['last_touched']
                })

            # Check crossover
            if prev_price < level_price and current_price >= level_price:
                alerts['crossover'].append({
                    'type': 'resistance',
                    'direction': 'up',
                    'level_price': level_price,
                    'current_price': current_price,
                    'prev_price': prev_price,
                    'touches': level['touches']
                })
            elif prev_price > level_price and current_price <= level_price:
                alerts['crossover'].append({
                    'type': 'resistance',
                    'direction': 'down',
                    'level_price': level_price,
                    'current_price': current_price,
                    'prev_price': prev_price,
                    'touches': level['touches']
                })

        return alerts

    def get_levels_summary(self) -> Dict:
        """Get summary of current support/resistance levels"""
        return {
            'ticker': self.ticker,
            'support_count': len(self.levels['support']),
            'resistance_count': len(self.levels['resistance']),
            'support_levels': [{'price': l['price'], 'touches': l['touches']}
                              for l in self.levels['support']],
            'resistance_levels': [{'price': l['price'], 'touches': l['touches']}
                                for l in self.levels['resistance']],
            'last_update': self.levels.get('last_update', 'Never')
        }


# Indicator functions for use in alerts
def PIVOT_SR(df: pd.DataFrame, ticker: str, left_bars: int = 5, right_bars: int = 5,
             proximity_threshold: float = 1.0, buffer_percent: float = 0.5) -> pd.Series:
    """
    Pivot Support/Resistance indicator for use in alerts

    Args:
        df: DataFrame with OHLCV data
        ticker: Stock ticker symbol
        left_bars: Bars to left of pivot
        right_bars: Bars to right of pivot
        proximity_threshold: % threshold for proximity alerts
        buffer_percent: % buffer between levels

    Returns:
        Series with signals:
        - 3: Strong bullish (crossed above strong resistance)
        - 2: Bullish crossover (crossed above support)
        - 1: Near support
        - 0: No signal
        - -1: Near resistance
        - -2: Bearish crossover (crossed below resistance)
        - -3: Strong bearish (crossed below strong support)
    """
    detector = PivotSupportResistance(
        ticker=ticker,
        left_bars=left_bars,
        right_bars=right_bars,
        proximity_threshold=proximity_threshold,
        buffer_percent=buffer_percent
    )

    # Update levels with latest data
    detector.update_levels(df)

    # Create signal series
    signals = pd.Series(0, index=df.index)

    # Check recent bars for signals
    for i in range(min(20, len(df))):
        idx = -(i + 1)
        if idx >= -len(df) and (i + 1) < len(df):
            df_slice = df.iloc[:len(df)-i] if i > 0 else df
            alerts = detector.check_proximity_and_crossover(df_slice)

            signal = 0

            # Check crossovers first (stronger signals)
            for crossover in alerts['crossover']:
                if crossover['type'] == 'support':
                    if crossover['direction'] == 'up':
                        signal = 2  # Bullish - bounced off support
                    else:
                        # Breaking below support - check if it's strong support
                        if crossover['touches'] >= 3:
                            signal = -3  # Strong bearish - broke strong support
                        else:
                            signal = -2  # Bearish - broke support
                elif crossover['type'] == 'resistance':
                    if crossover['direction'] == 'up':
                        # Breaking above resistance - check if it's strong
                        if crossover['touches'] >= 3:
                            signal = 3  # Strong bullish - broke strong resistance
                        else:
                            signal = 2  # Bullish - broke resistance
                    else:
                        signal = -2  # Bearish - rejected at resistance

            # If no crossover, check proximity
            if signal == 0:
                for proximity in alerts['proximity']:
                    if proximity['type'] == 'support':
                        signal = max(signal, 1)
                    elif proximity['type'] == 'resistance':
                        signal = min(signal, -1)

            signals.iloc[idx] = signal

    return signals


def PIVOT_SR_CROSSOVER(df: pd.DataFrame, ticker: str, left_bars: int = 5, right_bars: int = 5,
                       buffer_percent: float = 0.5) -> pd.Series:
    """
    Simplified version that only detects crossovers

    Returns:
        1 for bullish crossover, -1 for bearish crossover, 0 otherwise
    """
    detector = PivotSupportResistance(
        ticker=ticker,
        left_bars=left_bars,
        right_bars=right_bars,
        proximity_threshold=100.0,  # Set very high to ignore proximity
        buffer_percent=buffer_percent
    )

    detector.update_levels(df)
    signals = pd.Series(0, index=df.index)

    for i in range(min(5, len(df))):
        idx = -(i + 1)
        if idx >= -len(df) and (i + 1) < len(df):
            df_slice = df.iloc[:len(df)-i] if i > 0 else df
            alerts = detector.check_proximity_and_crossover(df_slice)

            signal = 0
            for crossover in alerts['crossover']:
                if crossover['direction'] == 'up' and crossover['type'] in ['support', 'resistance']:
                    signal = 1  # Bullish crossover
                elif crossover['direction'] == 'down' and crossover['type'] in ['support', 'resistance']:
                    signal = -1  # Bearish crossover

            signals.iloc[idx] = signal

    return signals


def PIVOT_SR_PROXIMITY(df: pd.DataFrame, ticker: str, left_bars: int = 5, right_bars: int = 5,
                       proximity_threshold: float = 1.0, buffer_percent: float = 0.5) -> pd.Series:
    """
    Simplified version that only detects proximity to levels

    Returns:
        1 for near support, -1 for near resistance, 0 otherwise
    """
    detector = PivotSupportResistance(
        ticker=ticker,
        left_bars=left_bars,
        right_bars=right_bars,
        proximity_threshold=proximity_threshold,
        buffer_percent=buffer_percent
    )

    detector.update_levels(df)
    signals = pd.Series(0, index=df.index)

    for i in range(min(5, len(df))):
        idx = -(i + 1)
        if idx >= -len(df) and (i + 1) < len(df):
            df_slice = df.iloc[:len(df)-i] if i > 0 else df
            alerts = detector.check_proximity_and_crossover(df_slice)

            signal = 0
            for proximity in alerts['proximity']:
                if proximity['type'] == 'support':
                    signal = 1
                elif proximity['type'] == 'resistance':
                    signal = -1

            signals.iloc[idx] = signal

    return signals
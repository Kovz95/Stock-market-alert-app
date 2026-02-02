"""
Pivot-based Support and Resistance Line Detector
Creates horizontal support/resistance lines from pivot points and tracks them over time
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Storage under project root so pivot_levels/ works regardless of cwd
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_STORAGE_DIR = _PROJECT_ROOT / "pivot_levels"


class PivotSupportResistance:
    def __init__(
        self,
        ticker: str,
        left_bars: int = 5,
        right_bars: int = 5,
        proximity_threshold: float = 1.0,
        buffer_percent: float = 0.5,
        lookback_years: int = 3,
        storage_dir: Optional[os.PathLike] = None,
    ):
        """
        Initialize Pivot Support/Resistance Detector

        Args:
            ticker: Stock ticker symbol
            left_bars: Number of bars to the left of pivot for validation
            right_bars: Number of bars to the right of pivot for validation
            proximity_threshold: Percentage threshold for proximity alerts (e.g., 1.0 = 1%)
            buffer_percent: Minimum % separation between lines to avoid clustering
            lookback_years: Number of years to maintain historical lines (rolling basis)
            storage_dir: Optional directory for JSON storage; defaults to project_root/pivot_levels
        """
        self.ticker = ticker
        self.left_bars = left_bars
        self.right_bars = right_bars
        self.proximity_threshold = proximity_threshold
        self.buffer_percent = buffer_percent
        self.lookback_years = lookback_years

        self.storage_dir = Path(storage_dir) if storage_dir else _DEFAULT_STORAGE_DIR
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.storage_file = self.storage_dir / f"{ticker}_pivot_levels.json"

        self.levels = self.load_levels()

    def load_levels(self) -> Dict:
        """Load existing support/resistance levels from storage"""
        if not self.storage_file.exists():
            return {"support": [], "resistance": [], "last_update": None}
        try:
            with open(self.storage_file, encoding="utf-8") as f:
                data = json.load(f)
            for level_type in ["support", "resistance"]:
                if level_type in data:
                    for level in data[level_type]:
                        level["first_detected"] = pd.to_datetime(level["first_detected"])
                        level["last_touched"] = pd.to_datetime(level["last_touched"])
            return data
        except (json.JSONDecodeError, KeyError, TypeError, OSError) as e:
            logger.warning(
                "Failed to load pivot levels from %s: %s",
                self.storage_file,
                e,
                exc_info=False,
            )
            return {"support": [], "resistance": [], "last_update": None}

    def save_levels(self) -> None:
        """Save support/resistance levels to storage"""
        data_to_save: Dict[str, Any] = {
            "support": [],
            "resistance": [],
            "last_update": datetime.now().isoformat(),
        }
        for level_type in ["support", "resistance"]:
            for level in self.levels.get(level_type, []):
                level_copy = level.copy()
                level_copy["first_detected"] = level_copy["first_detected"].isoformat()
                level_copy["last_touched"] = level_copy["last_touched"].isoformat()
                data_to_save[level_type].append(level_copy)
        with open(self.storage_file, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, indent=2)

    def find_pivot_highs(self, df: pd.DataFrame) -> List[Tuple[int, float]]:
        """
        Find pivot highs (potential resistance levels)

        Returns:
            List of tuples (index, price) for pivot highs
        """
        highs = df["High"].values
        pivot_highs = []
        for i in range(self.left_bars, len(highs) - self.right_bars):
            is_pivot = True
            pivot_value = highs[i]
            for j in range(i - self.left_bars, i):
                if highs[j] >= pivot_value:
                    is_pivot = False
                    break
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
        lows = df["Low"].values
        pivot_lows = []
        for i in range(self.left_bars, len(lows) - self.right_bars):
            is_pivot = True
            pivot_value = lows[i]
            for j in range(i - self.left_bars, i):
                if lows[j] <= pivot_value:
                    is_pivot = False
                    break
            if is_pivot:
                for j in range(i + 1, i + self.right_bars + 1):
                    if lows[j] <= pivot_value:
                        is_pivot = False
                        break
            if is_pivot:
                pivot_lows.append((i, pivot_value))
        return pivot_lows

    def is_within_buffer(self, new_price: float, existing_levels: List[Dict]) -> bool:
        """Check if a new price level is within buffer zone of existing levels."""
        for level in existing_levels:
            existing_price = level["price"]
            percent_diff = abs((new_price - existing_price) / existing_price * 100)
            if percent_diff < self.buffer_percent:
                return True
        return False

    def update_or_add_level(
        self, price: float, level_type: str, date: pd.Timestamp
    ) -> None:
        """Update existing level or add new one if not within buffer."""
        levels = self.levels[level_type]
        for level in levels:
            existing_price = level["price"]
            percent_diff = abs((price - existing_price) / existing_price * 100)
            if percent_diff < self.buffer_percent:
                level["touches"] += 1
                level["last_touched"] = date
                return
        new_level = {
            "price": price,
            "first_detected": date,
            "last_touched": date,
            "touches": 1,
            "strength": 1,
        }
        levels.append(new_level)

    def clean_old_levels(
        self,
        current_date: pd.Timestamp,
        earliest_data_date: Optional[pd.Timestamp] = None,
    ) -> None:
        """Remove levels older than lookback_years (or earliest available data)."""
        cutoff_date = current_date - pd.Timedelta(days=self.lookback_years * 365)
        if earliest_data_date and earliest_data_date > cutoff_date:
            cutoff_date = earliest_data_date
        for level_type in ["support", "resistance"]:
            self.levels[level_type] = [
                level
                for level in self.levels[level_type]
                if level["last_touched"] >= cutoff_date
            ]

    def initialize_levels(self, df: pd.DataFrame) -> None:
        """Initialize support/resistance levels from historical data (up to lookback_years)."""
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        cutoff_date = df.index[-1] - pd.Timedelta(days=self.lookback_years * 365)
        if df.index[0] > cutoff_date:
            df_lookback = df.copy()
        else:
            df_lookback = df[df.index >= cutoff_date].copy()
        if len(df_lookback) < (self.left_bars + self.right_bars + 1):
            return
        self.levels = {"support": [], "resistance": []}
        pivot_highs = self.find_pivot_highs(df_lookback)
        pivot_lows = self.find_pivot_lows(df_lookback)
        for idx, price in pivot_highs:
            self.update_or_add_level(price, "resistance", df_lookback.index[idx])
        for idx, price in pivot_lows:
            self.update_or_add_level(price, "support", df_lookback.index[idx])
        self.levels["support"].sort(key=lambda x: x["price"])
        self.levels["resistance"].sort(key=lambda x: x["price"])
        self.save_levels()

    def update_levels(self, df: pd.DataFrame) -> None:
        """Update levels with new data (check for new pivots)."""
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        current_date = df.index[-1]
        if not self.levels["support"] and not self.levels["resistance"]:
            self.initialize_levels(df)
            return
        min_bars_needed = self.left_bars + self.right_bars + 1
        if len(df) >= min_bars_needed:
            recent_df = df.tail(min_bars_needed * 2)
            pivot_highs = self.find_pivot_highs(recent_df)
            pivot_lows = self.find_pivot_lows(recent_df)
            for idx, price in pivot_highs:
                if idx >= len(recent_df) - self.right_bars - 1:
                    self.update_or_add_level(
                        price, "resistance", recent_df.index[idx]
                    )
            for idx, price in pivot_lows:
                if idx >= len(recent_df) - self.right_bars - 1:
                    self.update_or_add_level(price, "support", recent_df.index[idx])
        earliest_data_date = df.index[0] if len(df) > 0 else None
        self.clean_old_levels(current_date, earliest_data_date)
        self.levels["support"].sort(key=lambda x: x["price"])
        self.levels["resistance"].sort(key=lambda x: x["price"])
        self.save_levels()

    def check_proximity_and_crossover(self, df: pd.DataFrame) -> Dict:
        """Check if current price is near or has crossed support/resistance levels."""
        if len(df) < 2:
            return {"proximity": [], "crossover": []}
        current_price = df["Close"].iloc[-1]
        prev_price = df["Close"].iloc[-2]
        alerts: Dict[str, List] = {"proximity": [], "crossover": []}
        for level in self.levels["support"]:
            level_price = level["price"]
            distance_percent = abs(
                (current_price - level_price) / level_price * 100
            )
            if distance_percent <= self.proximity_threshold:
                alerts["proximity"].append(
                    {
                        "type": "support",
                        "level_price": level_price,
                        "current_price": current_price,
                        "distance_percent": distance_percent,
                        "touches": level["touches"],
                        "first_detected": level["first_detected"],
                        "last_touched": level["last_touched"],
                    }
                )
            if prev_price > level_price and current_price <= level_price:
                alerts["crossover"].append(
                    {
                        "type": "support",
                        "direction": "down",
                        "level_price": level_price,
                        "current_price": current_price,
                        "prev_price": prev_price,
                        "touches": level["touches"],
                    }
                )
            elif prev_price < level_price and current_price >= level_price:
                alerts["crossover"].append(
                    {
                        "type": "support",
                        "direction": "up",
                        "level_price": level_price,
                        "current_price": current_price,
                        "prev_price": prev_price,
                        "touches": level["touches"],
                    }
                )
        for level in self.levels["resistance"]:
            level_price = level["price"]
            distance_percent = abs(
                (current_price - level_price) / level_price * 100
            )
            if distance_percent <= self.proximity_threshold:
                alerts["proximity"].append(
                    {
                        "type": "resistance",
                        "level_price": level_price,
                        "current_price": current_price,
                        "distance_percent": distance_percent,
                        "touches": level["touches"],
                        "first_detected": level["first_detected"],
                        "last_touched": level["last_touched"],
                    }
                )
            if prev_price < level_price and current_price >= level_price:
                alerts["crossover"].append(
                    {
                        "type": "resistance",
                        "direction": "up",
                        "level_price": level_price,
                        "current_price": current_price,
                        "prev_price": prev_price,
                        "touches": level["touches"],
                    }
                )
            elif prev_price > level_price and current_price <= level_price:
                alerts["crossover"].append(
                    {
                        "type": "resistance",
                        "direction": "down",
                        "level_price": level_price,
                        "current_price": current_price,
                        "prev_price": prev_price,
                        "touches": level["touches"],
                    }
                )
        return alerts

    def get_levels_summary(self) -> Dict:
        """Get summary of current support/resistance levels."""
        return {
            "ticker": self.ticker,
            "support_count": len(self.levels["support"]),
            "resistance_count": len(self.levels["resistance"]),
            "support_levels": [
                {"price": l["price"], "touches": l["touches"]}
                for l in self.levels["support"]
            ],
            "resistance_levels": [
                {"price": l["price"], "touches": l["touches"]}
                for l in self.levels["resistance"]
            ],
            "last_update": self.levels.get("last_update", "Never"),
        }


# ---------------------------------------------------------------------------
# Indicator functions for use in alerts (ticker optional for backend compatibility)
# ---------------------------------------------------------------------------


def PIVOT_SR(
    df: pd.DataFrame,
    ticker: str = "UNKNOWN",
    left_bars: int = 5,
    right_bars: int = 5,
    proximity_threshold: float = 1.0,
    buffer_percent: float = 0.5,
) -> pd.Series:
    """
    Pivot Support/Resistance indicator for use in alerts.

    Returns:
        Series with signals: 3 strong bullish, 2 bullish, 1 near support,
        0 no signal, -1 near resistance, -2 bearish, -3 strong bearish
    """
    detector = PivotSupportResistance(
        ticker=ticker,
        left_bars=left_bars,
        right_bars=right_bars,
        proximity_threshold=proximity_threshold,
        buffer_percent=buffer_percent,
    )
    detector.update_levels(df)
    signals = pd.Series(0, index=df.index)
    for i in range(min(20, len(df))):
        idx = -(i + 1)
        if idx >= -len(df) and (i + 1) < len(df):
            df_slice = df.iloc[: len(df) - i] if i > 0 else df
            alerts = detector.check_proximity_and_crossover(df_slice)
            signal = 0
            for crossover in alerts["crossover"]:
                if crossover["type"] == "support":
                    if crossover["direction"] == "up":
                        signal = 2
                    else:
                        signal = -3 if crossover["touches"] >= 3 else -2
                elif crossover["type"] == "resistance":
                    if crossover["direction"] == "up":
                        signal = 3 if crossover["touches"] >= 3 else 2
                    else:
                        signal = -2
            if signal == 0:
                for proximity in alerts["proximity"]:
                    if proximity["type"] == "support":
                        signal = max(signal, 1)
                    elif proximity["type"] == "resistance":
                        signal = min(signal, -1)
            signals.iloc[idx] = signal
    return signals


def PIVOT_SR_CROSSOVER(
    df: pd.DataFrame,
    ticker: str = "UNKNOWN",
    left_bars: int = 5,
    right_bars: int = 5,
    buffer_percent: float = 0.5,
) -> pd.Series:
    """Pivot S/R crossover only: 1 bullish, -1 bearish, 0 otherwise."""
    detector = PivotSupportResistance(
        ticker=ticker,
        left_bars=left_bars,
        right_bars=right_bars,
        proximity_threshold=100.0,
        buffer_percent=buffer_percent,
    )
    detector.update_levels(df)
    signals = pd.Series(0, index=df.index)
    for i in range(min(5, len(df))):
        idx = -(i + 1)
        if idx >= -len(df) and (i + 1) < len(df):
            df_slice = df.iloc[: len(df) - i] if i > 0 else df
            alerts = detector.check_proximity_and_crossover(df_slice)
            signal = 0
            for crossover in alerts["crossover"]:
                if crossover["direction"] == "up" and crossover["type"] in [
                    "support",
                    "resistance",
                ]:
                    signal = 1
                elif crossover["direction"] == "down" and crossover["type"] in [
                    "support",
                    "resistance",
                ]:
                    signal = -1
            signals.iloc[idx] = signal
    return signals


def PIVOT_SR_PROXIMITY(
    df: pd.DataFrame,
    ticker: str = "UNKNOWN",
    left_bars: int = 5,
    right_bars: int = 5,
    proximity_threshold: float = 1.0,
    buffer_percent: float = 0.5,
) -> pd.Series:
    """Pivot S/R proximity only: 1 near support, -1 near resistance, 0 otherwise."""
    detector = PivotSupportResistance(
        ticker=ticker,
        left_bars=left_bars,
        right_bars=right_bars,
        proximity_threshold=proximity_threshold,
        buffer_percent=buffer_percent,
    )
    detector.update_levels(df)
    signals = pd.Series(0, index=df.index)
    for i in range(min(5, len(df))):
        idx = -(i + 1)
        if idx >= -len(df) and (i + 1) < len(df):
            df_slice = df.iloc[: len(df) - i] if i > 0 else df
            alerts = detector.check_proximity_and_crossover(df_slice)
            signal = 0
            for proximity in alerts["proximity"]:
                if proximity["type"] == "support":
                    signal = 1
                elif proximity["type"] == "resistance":
                    signal = -1
            signals.iloc[idx] = signal
    return signals

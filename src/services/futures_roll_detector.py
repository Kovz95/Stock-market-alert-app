"""
Service for detecting Interactive Brokers futures contract rolls

This service provides methods to detect when Interactive Brokers rolls the front month
in continuous futures contracts. It uses two approaches:
1. Gap detection - analyzing price/volume anomalies
2. Theoretical calculation - using known contract specifications
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List
import logging

from src.utils.reference_data.futures_specs import (
    FUTURES_GAP_THRESHOLDS,
    FUTURES_ROLL_SCHEDULES
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def detect_roll_dates_from_continuous(
    df: pd.DataFrame,
    symbol: str,
    threshold_pct: float = 0.5
) -> List:
    """
    Detect likely roll dates from continuous contract data by identifying abnormal gaps.

    This method analyzes price and volume data to identify characteristics typical of
    contract rolls:
    - Abnormal price gaps
    - Volume drops
    - Timing within typical roll windows
    - Quick reversal of gap direction

    Args:
        df: DataFrame with OHLC data from IB's ContFuture
        symbol: Futures symbol (for contract-specific thresholds)
        threshold_pct: Percentage move threshold to identify rolls (default 0.5%)

    Returns:
        List of likely roll dates

    Examples:
        >>> df = fetch_continuous_data('CL')
        >>> roll_dates = detect_roll_dates_from_continuous(df, 'CL')
        >>> print(f"Found {len(roll_dates)} roll dates")
    """
    # Calculate daily returns
    df['daily_return'] = df['Close'].pct_change()

    # Calculate gap between previous close and current open
    df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

    # Also check for volume anomalies (often volume drops during roll)
    df['volume_ma'] = df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma']

    # Get symbol-specific gap threshold
    threshold = FUTURES_GAP_THRESHOLDS.get(symbol, threshold_pct / 100)

    # Detect potential roll dates
    roll_dates = []

    for idx in df.index[1:]:
        row = df.loc[idx]

        # Check for abnormal gap
        if abs(row['gap']) > threshold:
            # Additional checks to confirm it's likely a roll

            # 1. Check if it's around typical roll dates (varies by contract)
            date = pd.to_datetime(idx)
            day_of_month = date.day

            # Most contracts roll between 5th and 25th of month
            if 5 <= day_of_month <= 25:

                # 2. Check if volume dropped (common during rolls)
                if row['volume_ratio'] < 0.8:  # Volume less than 80% of average

                    # 3. Check if the gap reverses quickly (not a trend move)
                    next_idx = df.index[df.index.get_loc(idx) + 1] if df.index.get_loc(idx) < len(df) - 1 else None
                    if next_idx:
                        next_return = df.loc[next_idx, 'daily_return']
                        # If next day moves opposite direction, likely a roll adjustment
                        if np.sign(row['gap']) != np.sign(next_return):
                            roll_dates.append(idx)
                            logger.info(
                                f"Detected likely roll on {idx}: "
                                f"gap={row['gap']:.3%}, volume_ratio={row['volume_ratio']:.2f}"
                            )

    return roll_dates


def detect_rolls_using_contract_specs(
    symbol: str,
    start_date: datetime,
    end_date: datetime
) -> List:
    """
    Calculate theoretical roll dates based on contract specifications.

    This method uses known contract roll schedules to calculate when rolls
    should theoretically occur. Note that IB may roll a few days early or late.

    Args:
        symbol: Futures symbol
        start_date: Start of period
        end_date: End of period

    Returns:
        List of theoretical roll dates

    Examples:
        >>> from datetime import datetime
        >>> start = datetime(2024, 1, 1)
        >>> end = datetime(2024, 12, 31)
        >>> rolls = detect_rolls_using_contract_specs('CL', start, end)
        >>> print(f"Expected {len(rolls)} rolls for CL in 2024")
    """
    if symbol not in FUTURES_ROLL_SCHEDULES:
        logger.warning(f"No roll schedule found for {symbol}")
        return []

    schedule = FUTURES_ROLL_SCHEDULES[symbol]
    roll_dates = []

    current_date = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    while current_date <= end:
        if 'months' in schedule:
            # Quarterly contracts (e.g., ES, NQ)
            if current_date.month in schedule['months']:
                if schedule['day'] == 'thursday_2':
                    # Find 2nd Thursday
                    first_day = current_date.replace(day=1)
                    first_thursday = first_day + pd.Timedelta(days=(3 - first_day.dayofweek) % 7)
                    second_thursday = first_thursday + pd.Timedelta(days=7)
                    roll_dates.append(second_thursday)
            current_date += pd.DateOffset(months=1)
        else:
            # Monthly contracts (e.g., CL, NG, GC)
            roll_day = schedule['day']
            try:
                roll_date = current_date.replace(day=roll_day)
                roll_dates.append(roll_date)
            except ValueError:
                # Handle months with fewer days
                pass
            current_date += pd.DateOffset(months=schedule.get('months_ahead', 1))

    return [d for d in roll_dates if start_date <= d <= end]


def mark_roll_periods_in_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Mark likely roll periods in continuous contract data.

    This method combines both detection approaches:
    1. Detects rolls from price/volume anomalies
    2. Calculates theoretical roll dates
    3. Marks dates where both methods agree as highly likely rolls

    Args:
        df: DataFrame with continuous contract data
        symbol: Futures symbol

    Returns:
        DataFrame with roll marker columns:
        - detected_roll: Rolls detected from price anomalies
        - theoretical_roll: Theoretical roll periods (±2 days)
        - likely_roll: Where both methods agree

    Examples:
        >>> df = fetch_continuous_data('GC')
        >>> df_marked = mark_roll_periods_in_data(df, 'GC')
        >>> likely_rolls = df_marked[df_marked['likely_roll']]
        >>> print(f"Found {len(likely_rolls)} high-confidence rolls")
    """
    # Method 1: Detect from price anomalies
    detected_rolls = detect_roll_dates_from_continuous(df.copy(), symbol)

    # Method 2: Calculate theoretical rolls
    theoretical_rolls = detect_rolls_using_contract_specs(
        symbol,
        df.index.min(),
        df.index.max()
    )

    # Add columns to mark rolls
    df['detected_roll'] = False
    df['theoretical_roll'] = False
    df['likely_roll'] = False

    # Mark detected rolls
    for roll_date in detected_rolls:
        if roll_date in df.index:
            df.loc[roll_date, 'detected_roll'] = True

    # Mark theoretical rolls (within 2 days)
    for roll_date in theoretical_rolls:
        # Mark within a window (IB might roll a day or two early/late)
        mask = (df.index >= roll_date - pd.Timedelta(days=2)) & \
               (df.index <= roll_date + pd.Timedelta(days=2))
        df.loc[mask, 'theoretical_roll'] = True

    # Mark likely rolls (where both methods agree)
    df['likely_roll'] = df['detected_roll'] & df['theoretical_roll']

    # Log summary
    logger.info(f"Roll detection summary for {symbol}:")
    logger.info(f"  - Detected rolls (from gaps): {df['detected_roll'].sum()}")
    logger.info(f"  - Theoretical roll periods: {df['theoretical_roll'].sum()} days")
    logger.info(f"  - Likely rolls (both agree): {df['likely_roll'].sum()}")

    return df


# Example usage
if __name__ == "__main__":
    print("""
    FUTURES ROLL DETECTION SERVICE
    ===============================

    This service provides methods to detect when Interactive Brokers rolls
    the front month in continuous futures contracts.

    METHODS:
    --------
    1. GAP DETECTION METHOD:
       - Look for abnormal price gaps between close and next open
       - Check for volume drops (common during rolls)
       - Verify it happens during typical roll windows

    2. THEORETICAL CALCULATION:
       - Use known contract specifications
       - Calculate when rolls should occur based on expiry rules
       - Account for IB potentially rolling a few days early

    3. COMBINED APPROACH:
       - Mark where both methods agree as highly likely rolls
       - Use these dates to apply Panama/Ratio adjustments

    USAGE EXAMPLE:
    --------------
    from src.services.futures_roll_detector import mark_roll_periods_in_data

    # Fetch continuous contract data
    df = fetch_continuous_data('CL')

    # Detect and mark roll periods
    df_marked = mark_roll_periods_in_data(df, 'CL')

    # Get high-confidence roll dates
    likely_rolls = df_marked[df_marked['likely_roll']].index.tolist()

    LIMITATIONS:
    ------------
    - IB's ContFuture already adjusts prices somehow
    - Detecting rolls from adjusted data is imperfect
    - Some rolls might be smooth with no detectable gap

    RECOMMENDATION:
    ---------------
    For accurate roll detection, you would need to:
    1. Query individual contracts (CLZ24, CLF25, etc.)
    2. Build your own continuous series
    3. Mark exact roll points as you switch contracts
    """)

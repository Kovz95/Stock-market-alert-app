"""
Methods to detect when Interactive Brokers rolls the front month in continuous contracts
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def detect_roll_dates_from_continuous(df, symbol, threshold_pct=0.5):
    """
    Detect likely roll dates from continuous contract data by identifying abnormal gaps

    Args:
        df: DataFrame with OHLC data from IB's ContFuture
        symbol: Futures symbol (for contract-specific thresholds)
        threshold_pct: Percentage move threshold to identify rolls (default 0.5%)

    Returns:
        List of likely roll dates
    """

    # Calculate daily returns
    df['daily_return'] = df['Close'].pct_change()

    # Calculate gap between previous close and current open
    df['gap'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)

    # Also check for volume anomalies (often volume drops during roll)
    df['volume_ma'] = df['Volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['Volume'] / df['volume_ma']

    # Symbol-specific gap thresholds (some contracts have bigger roll gaps)
    gap_thresholds = {
        'CL': 0.01,    # Oil can have 1% gaps on rolls
        'NG': 0.02,    # Natural gas can have 2% gaps
        'GC': 0.003,   # Gold typically smaller gaps
        'ES': 0.002,   # Equity indices very small gaps
        'ZC': 0.01,    # Grains can have 1% gaps
        'ZW': 0.01,
        'ZS': 0.01,
    }

    threshold = gap_thresholds.get(symbol, threshold_pct / 100)

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
                            logger.info(f"Detected likely roll on {idx}: gap={row['gap']:.3%}, volume_ratio={row['volume_ratio']:.2f}")

    return roll_dates


def detect_rolls_using_contract_specs(symbol, start_date, end_date):
    """
    Calculate theoretical roll dates based on contract specifications

    Args:
        symbol: Futures symbol
        start_date: Start of period
        end_date: End of period

    Returns:
        List of theoretical roll dates
    """

    # Contract roll schedules (simplified)
    roll_schedules = {
        # Energy - usually rolls around the 20th
        'CL': {'day': 20, 'months_ahead': 1},
        'NG': {'day': 27, 'months_ahead': 1},  # NG rolls later

        # Metals - rolls around 25th-27th
        'GC': {'day': 27, 'months_ahead': 2},  # Bi-monthly
        'SI': {'day': 27, 'months_ahead': 2},

        # Equity indices - quarterly, 2nd Thursday
        'ES': {'day': 'thursday_2', 'months': [3, 6, 9, 12]},
        'NQ': {'day': 'thursday_2', 'months': [3, 6, 9, 12]},

        # Grains - rolls around 14th
        'ZC': {'day': 14, 'months_ahead': 1},
        'ZW': {'day': 14, 'months_ahead': 1},
        'ZS': {'day': 14, 'months_ahead': 1},
    }

    if symbol not in roll_schedules:
        return []

    schedule = roll_schedules[symbol]
    roll_dates = []

    current_date = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    while current_date <= end:
        if 'months' in schedule:
            # Quarterly contracts
            if current_date.month in schedule['months']:
                if schedule['day'] == 'thursday_2':
                    # Find 2nd Thursday
                    first_day = current_date.replace(day=1)
                    first_thursday = first_day + pd.Timedelta(days=(3 - first_day.dayofweek) % 7)
                    second_thursday = first_thursday + pd.Timedelta(days=7)
                    roll_dates.append(second_thursday)
            current_date += pd.DateOffset(months=1)
        else:
            # Monthly contracts
            roll_day = schedule['day']
            try:
                roll_date = current_date.replace(day=roll_day)
                roll_dates.append(roll_date)
            except:
                # Handle months with fewer days
                pass
            current_date += pd.DateOffset(months=schedule.get('months_ahead', 1))

    return [d for d in roll_dates if start_date <= d <= end]


def mark_roll_periods_in_data(df, symbol):
    """
    Mark likely roll periods in continuous contract data

    Args:
        df: DataFrame with continuous contract data
        symbol: Futures symbol

    Returns:
        DataFrame with roll markers
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
    METHODS TO DETECT IB ROLL DATES:
    =================================

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

    LIMITATIONS:
    - IB's ContFuture already adjusts prices somehow
    - Detecting rolls from adjusted data is imperfect
    - Some rolls might be smooth with no detectable gap

    RECOMMENDATION:
    For accurate roll detection, you would need to:
    1. Query individual contracts (CLZ24, CLF25, etc.)
    2. Build your own continuous series
    3. Mark exact roll points as you switch contracts
    """)
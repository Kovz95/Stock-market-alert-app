"""
Futures contract specifications and roll schedules for Interactive Brokers

This module contains reference data for futures contracts including:
- Roll schedules (when contracts typically roll)
- Gap thresholds for detecting rolls in price data
"""

# Symbol-specific gap thresholds for roll detection
# These represent typical price gaps that occur during contract rolls
FUTURES_GAP_THRESHOLDS = {
    'CL': 0.01,    # Oil can have 1% gaps on rolls
    'NG': 0.02,    # Natural gas can have 2% gaps
    'GC': 0.003,   # Gold typically smaller gaps
    'ES': 0.002,   # Equity indices very small gaps
    'ZC': 0.01,    # Grains can have 1% gaps
    'ZW': 0.01,    # Wheat
    'ZS': 0.01,    # Soybeans
}

# Contract roll schedules
# Defines when each futures contract typically rolls to the next month
FUTURES_ROLL_SCHEDULES = {
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

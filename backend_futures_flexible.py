"""
Flexible futures backend for symbol detection and basic utilities.
Provides broker-agnostic futures symbol identification without requiring connection to any specific broker.
"""

from typing import Optional, Dict, List


# Comprehensive list of futures symbols across major categories
FUTURES_SYMBOLS = {
    # Energy
    'CL', 'QM', 'BZ', 'NG', 'QG', 'HO', 'RB',

    # Precious Metals
    'GC', 'MGC', 'QO', 'SI', 'SIL', 'QI', 'HG', 'QC', 'PL', 'PA',

    # Stock Index Futures
    'ES', 'MES', 'NQ', 'MNQ', 'YM', 'MYM', 'RTY', 'M2K',

    # Agricultural
    'ZC', 'YC', 'ZW', 'YW', 'ZS', 'YK', 'ZM', 'ZL', 'KE', 'CT', 'SB', 'KC', 'CC', 'OJ',

    # Livestock
    'LE', 'GF', 'HE',

    # Currencies
    '6E', 'M6E', '6B', 'M6B', '6J', 'MJY', '6C', 'MCD', '6A', 'M6A', '6S', 'MSF', '6N', '6M', 'DX',

    # Interest Rates/Bonds
    'ZN', '10Y', 'ZB', 'UB', 'ZF', '5YR', 'ZT', '2YR', 'ZQ', 'GE',

    # Volatility
    'VX', 'VXM',

    # European Indices
    'FDAX', 'FDXM', 'FESX', 'FESB', 'FGBL', 'Z',

    # Asian Indices
    'NIY', 'HSI', 'MHI',
}


# Futures contract details for reference
FUTURES_INFO = {
    # Energy
    'CL': {'name': 'WTI Crude Oil', 'category': 'Energy'},
    'NG': {'name': 'Natural Gas', 'category': 'Energy'},
    'BZ': {'name': 'Brent Crude Oil', 'category': 'Energy'},
    'HO': {'name': 'Heating Oil', 'category': 'Energy'},
    'RB': {'name': 'RBOB Gasoline', 'category': 'Energy'},

    # Metals
    'GC': {'name': 'Gold', 'category': 'Metals'},
    'SI': {'name': 'Silver', 'category': 'Metals'},
    'HG': {'name': 'Copper', 'category': 'Metals'},
    'PL': {'name': 'Platinum', 'category': 'Metals'},
    'PA': {'name': 'Palladium', 'category': 'Metals'},

    # Indices
    'ES': {'name': 'E-mini S&P 500', 'category': 'Indices'},
    'NQ': {'name': 'E-mini Nasdaq 100', 'category': 'Indices'},
    'YM': {'name': 'E-mini Dow Jones', 'category': 'Indices'},
    'RTY': {'name': 'E-mini Russell 2000', 'category': 'Indices'},

    # Agriculture
    'ZC': {'name': 'Corn', 'category': 'Agriculture'},
    'ZW': {'name': 'Wheat', 'category': 'Agriculture'},
    'ZS': {'name': 'Soybeans', 'category': 'Agriculture'},
    'ZM': {'name': 'Soybean Meal', 'category': 'Agriculture'},
    'ZL': {'name': 'Soybean Oil', 'category': 'Agriculture'},
    'KC': {'name': 'Coffee', 'category': 'Agriculture'},
    'SB': {'name': 'Sugar', 'category': 'Agriculture'},
    'CT': {'name': 'Cotton', 'category': 'Agriculture'},
    'CC': {'name': 'Cocoa', 'category': 'Agriculture'},

    # Currencies
    'DX': {'name': 'US Dollar Index', 'category': 'Currencies'},
    '6E': {'name': 'Euro FX', 'category': 'Currencies'},
    '6B': {'name': 'British Pound', 'category': 'Currencies'},
    '6J': {'name': 'Japanese Yen', 'category': 'Currencies'},
    '6C': {'name': 'Canadian Dollar', 'category': 'Currencies'},

    # Bonds
    'ZN': {'name': '10-Year T-Note', 'category': 'Bonds'},
    'ZB': {'name': '30-Year T-Bond', 'category': 'Bonds'},
    'ZF': {'name': '5-Year T-Note', 'category': 'Bonds'},

    # Volatility
    'VX': {'name': 'VIX Futures', 'category': 'Volatility'},
}


def is_futures_symbol(ticker: str) -> bool:
    """
    Check if a ticker symbol represents a futures contract.

    This function uses multiple detection methods:
    1. Check if symbol ends with '=F' (common futures notation)
    2. Check if symbol is in the known futures symbols list

    Args:
        ticker: Symbol to check

    Returns:
        True if the symbol is a futures contract, False otherwise
    """
    if not ticker:
        return False

    ticker_upper = ticker.upper().strip()

    # Check for common futures indicators
    # Many data providers use '=F' suffix for futures
    if ticker_upper.endswith('=F'):
        return True

    # Remove common suffixes before checking
    # Some systems append month codes or other identifiers
    base_symbol = ticker_upper.split('=')[0].split('.')[0]

    # Check if it's in our known futures symbols
    if base_symbol in FUTURES_SYMBOLS:
        return True

    return False


def get_futures_info(symbol: str) -> Optional[Dict]:
    """
    Get information about a futures contract.

    Args:
        symbol: Futures symbol

    Returns:
        Dictionary with contract information, or None if not found
    """
    symbol_upper = symbol.upper().strip()
    base_symbol = symbol_upper.split('=')[0].split('.')[0]

    if base_symbol in FUTURES_INFO:
        info = FUTURES_INFO[base_symbol].copy()
        info['symbol'] = base_symbol
        return info

    return None


def get_all_futures_symbols() -> List[str]:
    """
    Get a list of all supported futures symbols.

    Returns:
        List of futures symbols
    """
    return sorted(list(FUTURES_SYMBOLS))


def get_futures_by_category() -> Dict[str, List[str]]:
    """
    Get futures symbols organized by category.

    Returns:
        Dictionary mapping categories to lists of symbols
    """
    categories = {}

    for symbol in FUTURES_SYMBOLS:
        info = FUTURES_INFO.get(symbol)
        if info:
            category = info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(symbol)
        else:
            # Default category for symbols without detailed info
            if 'Other' not in categories:
                categories['Other'] = []
            categories['Other'].append(symbol)

    return categories


# Aliases for backward compatibility
isFuturesSymbol = is_futures_symbol
getFuturesInfo = get_futures_info

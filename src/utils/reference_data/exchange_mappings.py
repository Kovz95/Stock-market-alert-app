"""
Comprehensive exchange mapping utilities.
Contains all exchange-related lookups and conversions.

This module consolidates:
- Exchange code to display name mappings
- Exchange to country mappings
- Country to exchange mappings
- Ticker suffix to exchange mappings
"""

from typing import Optional


# Exchange code to full display name mapping
EXCHANGE_CODE_TO_NAME = {
    # Major exchanges
    "NYSE": "New York Stock Exchange",
    "NASDAQ": "NASDAQ Stock Market",
    "LSE": "London Stock Exchange",
    "TSE": "Tokyo Stock Exchange",
    "FRA": "Frankfurt Stock Exchange",
    "PAR": "Euronext Paris",
    "AMS": "Euronext Amsterdam",
    "MIL": "Borsa Italiana",
    "MAD": "Bolsa de Madrid",
    "STO": "Stockholm Stock Exchange",
    "OSL": "Oslo Stock Exchange",
    "CPH": "Copenhagen Stock Exchange",
    "HEL": "Helsinki Stock Exchange",
    "BRU": "Euronext Brussels",
    "DUB": "Irish Stock Exchange",
    "LIS": "Euronext Lisbon",
    "VIE": "Vienna Stock Exchange",
    "WAR": "Warsaw Stock Exchange",
    "ATH": "Athens Stock Exchange",
    "BUD": "Budapest Stock Exchange",
    "PRA": "Prague Stock Exchange",
    "IST": "Istanbul Stock Exchange",
    "MEX": "Mexican Stock Exchange",
    "HKG": "Hong Kong Stock Exchange",
    "SGX": "Singapore Exchange",
    "TWO": "Taiwan Stock Exchange",
    "KLSE": "Bursa Malaysia",
    "KRX": "Korea Exchange",
    "NSE": "National Stock Exchange of India",
    "BSE": "Bombay Stock Exchange",
    "SSE": "Shanghai Stock Exchange",
    "SZSE": "Shenzhen Stock Exchange",
    "SET": "Stock Exchange of Thailand",
    "IDX": "Indonesia Stock Exchange",
    "PSE": "Philippine Stock Exchange",
    "HOSE": "Ho Chi Minh Stock Exchange",
    "ASX": "ASX",
    "SIX": "Swiss Exchange",
    "TSX": "Toronto Stock Exchange",
    "TSXV": "TSX Venture Exchange",

    # Country codes that are being used as exchanges in the Stock Database
    "US": "United States Markets",
    "CA": "Canadian Markets",
    "UK": "United Kingdom Markets",
    "JP": "Japanese Markets",
    "DE": "German Markets",
    "FR": "French Markets",
    "AU": "Australian Markets",
    "CH": "Swiss Markets",
    "NL": "Dutch Markets",
    "IT": "Italian Markets",
    "ES": "Spanish Markets",
    "SE": "Swedish Markets",
    "NO": "Norwegian Markets",
    "DK": "Danish Markets",
    "FI": "Finnish Markets",
    "BE": "Belgian Markets",
    "IE": "Irish Markets",
    "PT": "Portuguese Markets",
    "AT": "Austrian Markets",
    "PL": "Polish Markets",
    "GR": "Greek Markets",
    "HU": "Hungarian Markets",
    "CZ": "Czech Markets",
    "TR": "Turkish Markets",
    "MX": "Mexican Markets",
    "HK": "Hong Kong Markets",
    "SG": "Singapore Markets",
    "TW": "Taiwan Markets",
    "MY": "Malaysian Markets",
    "KR": "South Korean Markets",
    "IN": "Indian Markets",
    "CN": "Chinese Markets",
    "TH": "Thai Markets",
    "ID": "Indonesian Markets",
    "PH": "Philippine Markets",
    "VN": "Vietnamese Markets"
}


# Exchange to country mapping
EXCHANGE_COUNTRY_MAP = {
    # Asia-Pacific
    "TOKYO": "Japan",
    "TAIWAN": "Taiwan",
    "HONG KONG": "Hong Kong",
    "SINGAPORE": "Singapore",
    "MALAYSIA": "Malaysia",
    "INDONESIA": "Indonesia",
    "THAILAND": "Thailand",
    "ASX": "Australia",
    "OMX NORDIC ICELAND": "Iceland",
    "OMX NORDIC STOCKHOLM": "Sweden",
    "OMX NORDIC HELSINKI": "Finland",
    "OMX NORDIC COPENHAGEN": "Denmark",

    # India
    "BSE INDIA": "India",
    "NSE INDIA": "India",

    # Europe
    "LONDON": "United Kingdom",
    "XETRA": "Germany",
    "FRANKFURT": "Germany",
    "EURONEXT AMSTERDAM": "Netherlands",
    "EURONEXT BRUSSELS": "Belgium",
    "EURONEXT DUBLIN": "Ireland",
    "EURONEXT LISBON": "Portugal",
    "EURONEXT PARIS": "France",
    "MILAN": "Italy",
    "MADRID": "Spain",
    "SPAIN": "Spain",
    "VIENNA": "Austria",
    "WARSAW": "Poland",
    "ATHENS": "Greece",
    "PRAGUE": "Czech Republic",
    "BUDAPEST": "Hungary",
    "OSLO": "Norway",
    "SIX SWISS": "Switzerland",

    # Americas
    "NASDAQ": "United States",
    "NYSE": "United States",
    "NYSE AMERICAN": "United States",
    "NYSE ARCA": "United States",
    "CBOE BZX": "United States",
    "TORONTO": "Canada",
    "SANTIAGO": "Chile",
    "BUENOS AIRES": "Argentina",
    "MEXICO": "Mexico",
    "COLOMBIA": "Colombia",
    "SAO PAULO": "Brazil",

    # Middle East / Africa
    "ISTANBUL": "Turkey",
    "JSE": "South Africa",
}


# Country to Exchange Name mapping
COUNTRY_TO_EXCHANGE = {
    # Americas
    "UNITED STATES": "NYSE/NASDAQ",
    "CANADA": "TSX",
    "BRAZIL": "B3",
    "MEXICO": "BMV",
    "ARGENTINA": "BYMA",
    "CHILE": "Santiago Exchange",
    "COLOMBIA": "BVC",
    "PERU": "BVL",
    "URUGUAY": "NYSE/NASDAQ",  # Usually listed in US
    "PUERTO RICO": "NYSE/NASDAQ",
    "COSTA RICA": "BNV",
    "PANAMA": "BVP",

    # Europe
    "UNITED KINGDOM": "LSE",
    "GERMANY": "XETRA",
    "FRANCE": "Euronext Paris",
    "SWITZERLAND": "SIX",
    "NETHERLANDS": "Euronext Amsterdam",
    "SPAIN": "BME",
    "ITALY": "Borsa Italiana",
    "BELGIUM": "Euronext Brussels",
    "SWEDEN": "Nasdaq Stockholm",
    "DENMARK": "Nasdaq Copenhagen",
    "NORWAY": "Oslo Børs",
    "FINLAND": "Nasdaq Helsinki",
    "AUSTRIA": "Vienna Stock Exchange",
    "PORTUGAL": "Euronext Lisbon",
    "IRELAND": "Euronext Dublin",
    "GREECE": "ATHEX",
    "POLAND": "WSE",
    "HUNGARY": "BSE",
    "CZECH REPUBLIC": "PSE",
    "LUXEMBOURG": "LuxSE",
    "LITHUANIA": "Nasdaq Vilnius",
    "MONACO": "Euronext Paris",
    "CYPRUS": "CSE",

    # Asia-Pacific
    "JAPAN": "TSE",
    "HONG KONG": "HKEX",
    "CHINA": "SSE/SZSE",
    "SINGAPORE": "SGX",
    "TAIWAN": "TWSE",
    "AUSTRALIA": "ASX",
    "NEW ZEALAND": "NZX",
    "THAILAND": "SET",
    "INDONESIA": "IDX",
    "MALAYSIA": "Bursa Malaysia",
    "PHILIPPINES": "PSE",
    "INDIA": "NSE/BSE",
    "ISRAEL": "TASE",
    "KAZAKHSTAN": "KASE",
    "MONGOLIA": "MSE",
    "MACAU": "HKEX",  # Usually listed in HK

    # Middle East & Africa
    "UNITED ARAB EMIRATES": "ADX/DFM",
    "SOUTH AFRICA": "JSE",
    "TURKEY": "BIST",

    # Other
    "ICELAND": "Nasdaq Iceland",
    "BERMUDA": "BSX",
    "BAHAMAS": "BISX",
    "BRITISH VIRGIN ISLANDS": "NYSE/NASDAQ",  # Usually US listed
    "CAYMAN ISLANDS": "NYSE/NASDAQ",  # Usually US listed
}


# Ticker suffix to exchange mapping
TICKER_SUFFIX_TO_EXCHANGE = {
    'US': 'NYSE/NASDAQ',
    'GB': 'LSE',
    'JP': 'TSE',
    'HK': 'HKEX',
    'CN': 'SSE/SZSE',
    'DE': 'XETRA',
    'FR': 'Euronext Paris',
    'CH': 'SIX',
    'NL': 'Euronext Amsterdam',
    'ES': 'BME',
    'IT': 'Borsa Italiana',
    'BE': 'Euronext Brussels',
    'SE': 'Nasdaq Stockholm',
    'DK': 'Nasdaq Copenhagen',
    'NO': 'Oslo Børs',
    'FI': 'Nasdaq Helsinki',
    'AU': 'ASX',
    'CA': 'TSX',
    'SG': 'SGX',
    'TW': 'TWSE',
    'BR': 'B3',
    'MX': 'BMV',
    'ZA': 'JSE',
    'TR': 'BIST',
    'TH': 'SET',
    'ID': 'IDX',
    'MY': 'Bursa Malaysia',
    'PH': 'PSE',
    'IN': 'NSE/BSE',
    'IS': 'Nasdaq Iceland',
}


# Public API functions

def get_exchange_display_name(exchange_code: str) -> str:
    """
    Convert exchange code to full display name.

    Args:
        exchange_code: Exchange code (e.g., "NYSE", "LSE", "US")

    Returns:
        Full exchange display name, or the original code if not found

    Examples:
        >>> get_exchange_display_name("NYSE")
        'New York Stock Exchange'
        >>> get_exchange_display_name("US")
        'United States Markets'
    """
    return EXCHANGE_CODE_TO_NAME.get(exchange_code, exchange_code)


def get_country_for_exchange(exchange: str) -> Optional[str]:
    """
    Return country name for an exchange, or None if unknown.

    Args:
        exchange: Exchange name (e.g., "NASDAQ", "TOKYO", "LONDON")

    Returns:
        Country name, or None if exchange is not found

    Examples:
        >>> get_country_for_exchange("NASDAQ")
        'United States'
        >>> get_country_for_exchange("TOKYO")
        'Japan'
        >>> get_country_for_exchange("UNKNOWN")
        None
    """
    if not exchange:
        return None
    return EXCHANGE_COUNTRY_MAP.get(exchange.upper())


def get_exchange_from_country(country: str) -> str:
    """
    Get exchange name from country name.

    Args:
        country: Country name (e.g., "UNITED STATES", "JAPAN")

    Returns:
        Exchange name, or the original country if not found

    Examples:
        >>> get_exchange_from_country("UNITED STATES")
        'NYSE/NASDAQ'
        >>> get_exchange_from_country("JAPAN")
        'TSE'
        >>> get_exchange_from_country("UNKNOWN COUNTRY")
        'UNKNOWN COUNTRY'
    """
    return COUNTRY_TO_EXCHANGE.get(country, country)


def get_exchange_from_ticker(ticker: str) -> str:
    """
    Get exchange abbreviation from ticker suffix.

    Extracts the suffix after the last '-' in the ticker and maps it
    to an exchange name.

    Args:
        ticker: Stock ticker with suffix (e.g., "AAPL-US", "TM-JP")

    Returns:
        Exchange name, or 'Unknown' if no suffix or mapping not found

    Examples:
        >>> get_exchange_from_ticker("AAPL-US")
        'NYSE/NASDAQ'
        >>> get_exchange_from_ticker("TM-JP")
        'TSE'
        >>> get_exchange_from_ticker("AAPL")
        'Unknown'
    """
    if '-' in ticker:
        suffix = ticker.split('-')[-1]
        return TICKER_SUFFIX_TO_EXCHANGE.get(suffix, suffix)
    return 'Unknown'

"""
Minimal exchange-to-country mapping used by schedulers and alert routing.
If an exchange is missing, the lookup returns None and callers should handle
the fallback gracefully.
"""

from typing import Optional

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


def get_country_for_exchange(exchange: str) -> Optional[str]:
    """Return country name for an exchange code, or None if unknown."""
    if not exchange:
        return None
    return EXCHANGE_COUNTRY_MAP.get(exchange.upper())

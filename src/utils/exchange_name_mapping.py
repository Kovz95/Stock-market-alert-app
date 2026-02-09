#!/usr/bin/env python3
"""
Maps country names to proper stock exchange names
"""

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

def get_exchange_name(country):
    """Get proper exchange name from country"""
    return COUNTRY_TO_EXCHANGE.get(country, country)

def get_exchange_from_ticker(ticker):
    """Get exchange abbreviation from ticker suffix"""
    if '-' in ticker:
        suffix = ticker.split('-')[-1]
        exchange_map = {
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
        return exchange_map.get(suffix, suffix)
    return 'Unknown'

if __name__ == "__main__":
    # Test the mapping
    import sys
    from src.data_access.alert_repository import list_alerts

    # Fix Unicode output on Windows  
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    alerts = list_alerts()
    
    # Show sample mappings
    print("Sample Country → Exchange mappings:")
    print("=" * 60)
    
    samples = {}
    for alert in alerts[:500]:
        country = alert.get('exchange', '')
        ticker = alert.get('ticker', '')
        if country and country not in samples:
            exchange = get_exchange_name(country)
            samples[country] = (ticker, exchange)
            if len(samples) >= 20:
                break
    
    for country, (ticker, exchange) in samples.items():
        print(f"{country:25} → {exchange:20} (e.g., {ticker})")
    
    print("\n" + "=" * 60)
    print("Exchange names will now show properly in Discord notifications!")
    print("E.g., 'NYSE/NASDAQ' instead of 'UNITED STATES'")

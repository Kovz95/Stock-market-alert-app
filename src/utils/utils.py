from typing import Any, Dict, Optional, cast
from urllib3 import HTTPResponse
import pandas as pd
import datetime
from datetime import timezone
import numpy as np
import os
import uuid
from src.utils.indicators import *
import requests
import time
import operator
import pytz
import threading
from functools import lru_cache
from collections import defaultdict
import concurrent.futures
import fmpsdk
from dotenv import load_dotenv
from src.data_access.document_store import load_document, save_document
from src.data_access.json_bridge import enable_json_bridge

enable_json_bridge()

from src.data_access.alert_repository import (
    create_alert as repo_create_alert,
    update_alert as repo_update_alert,
    list_alerts as repo_list_alerts,
    get_alert as repo_get_alert,
)
from src.data_access.metadata_repository import fetch_stock_metadata_map
from src.utils.discord_env import get_discord_environment_tag

from src.data_access.alert_repository import (
    create_alert as repo_create_alert,
    update_alert as repo_update_alert,
    delete_alert as repo_delete_alert,
    list_alerts as repo_list_alerts,
    get_alert as repo_get_alert,
)

# Load environment variables from .env file
load_dotenv()

# Try to import streamlit for caching, but don't fail if not available
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

# Enhanced Caching System:
# - Custom in-memory cache: 10 minutes, up to 10,000 entries (for rate limiting)
# - Streamlit cache: 5 minutes (300s) for price data, 3 minutes (180s) for stock data
# - Dual-layer caching provides both performance and persistence across sessions

MAX_DISCORD_MESSAGE_LENGTH = 2000
FMP_API_KEY = os.getenv("FMP_API_KEY")
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
WEBHOOK_URL_2 = os.getenv("WEBHOOK_URL_2")

# Webhook URLs for logging - must be set in environment
WEBHOOK_URL_LOGGING = os.getenv("WEBHOOK_URL_LOGGING")
WEBHOOK_URL_LOGGING_2 = os.getenv("WEBHOOK_URL_LOGGING_2")
LOG_BUFFER = []


# Enhanced cache with TTL and size limits
fmp_cache = {}
fmp_cache_lock = threading.Lock()
MAX_CACHE_SIZE = 10000  # Maximum number of cached entries

# FMP Ticker Mapping System
# Maps ticker formats to FMP API ticker formats
# Based on investigation results from investigate_fmp_tickers.py
_BASE_FMP_TICKER_MAPPING = {
    # Hong Kong stocks - FMP supports .HK suffix
    "0700.HK": "0700.HK",   # Tencent
    "939.HK": "939.HK",      # China Construction Bank
    "9988.HK": "9988.HK",    # Alibaba
    "1299.HK": "1299.HK",    # AIA Group

    # London stocks - FMP uses different symbols
    "HSBC.L": "HSBA.L",      # HSBC Holdings (FMP uses HSBA.L)
    "BP.L": "BP.L",          # BP
    "VOD.L": "VOD.L",        # Vodafone

    # European stocks - Some work with suffixes, some don't
    "ASML.AS": "ASML.AS",    # ASML Holding (Amsterdam) - FMP supports .AS
    "SAP.DE": "SAP.DE",      # SAP SE (Frankfurt) - FMP supports .DE
    "LVMH.PA": "LVMHF",      # LVMH (Paris) - FMP uses OTC symbol LVMHF
    "NESN.SW": "NESN.SW",    # Nestle (Switzerland)

    # US stocks (usually same)
    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "GOOGL": "GOOGL",
    "AMZN": "AMZN",

    # Special cases - May need ADR alternatives
    "BRK.A": "BRK.A",
    "BRK.B": "BRK.B",

    # ADR alternatives for international stocks
    "TCEHY": "TCEHY",       # Tencent ADR (OTC)
    "LVMHF": "LVMHF",       # LVMH OTC alternative
}

_EXTRA_FMP_MAPPING = load_document(
    "fmp_ticker_mapping",
    default={},
    fallback_path="fmp_ticker_mapping.json",
)
if not isinstance(_EXTRA_FMP_MAPPING, dict):
    _EXTRA_FMP_MAPPING = {}

FMP_TICKER_MAPPING = {**_BASE_FMP_TICKER_MAPPING, **_EXTRA_FMP_MAPPING}


@lru_cache(maxsize=1)
def _get_enhanced_fmp_mappings() -> dict:
    data = load_document(
        "enhanced_fmp_ticker_mapping",
        default={},
        fallback_path="enhanced_fmp_ticker_mapping.json",
    )
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def _get_local_exchange_mappings() -> dict:
    data = load_document(
        "local_exchange_mappings",
        default={},
        fallback_path="local_exchange_mappings.json",
    )
    return data if isinstance(data, dict) else {}

def get_fmp_ticker(ticker):
    """
    Enhanced ticker conversion for FMP API with comprehensive mapping rules
    Returns the FMP ticker if mapping exists, otherwise applies transformation rules
    """
    import re

    # First check the main FMP ticker mapping (includes HK fixes)
    if ticker in FMP_TICKER_MAPPING:
        return FMP_TICKER_MAPPING[ticker]

    enhanced_mappings = _get_enhanced_fmp_mappings()

    local_exchange_doc = _get_local_exchange_mappings()
    local_mappings = {}
    for category in local_exchange_doc.get('local_exchange_corrections', {}).values():
        if isinstance(category, dict):
            local_mappings.update(category)

    # Check local exchange mappings first (highest priority for failed tickers)
    if ticker in local_mappings:
        return local_mappings[ticker]

    # Check enhanced mappings
    if ticker in enhanced_mappings:
        return enhanced_mappings[ticker]

    # Check original mappings
    if ticker in FMP_TICKER_MAPPING:
        return FMP_TICKER_MAPPING[ticker]

    # Load exchange transforms from document store data
    exchange_transforms = {}
    for old_suffix, new_suffix in local_exchange_doc.get('exchange_suffix_transforms', {}).items():
        exchange_transforms[f'.{old_suffix}'] = f'.{new_suffix}'

    # Fallback to default transforms if file loading fails
    if not exchange_transforms:
        exchange_transforms = {
            '.JP': '.T',      # Japan: .JP -> .T
            '.UK': '.L',      # UK: .UK -> .L
            '.CA': '.TO',     # Canada: .CA -> .TO
            '.AU': '.AX',     # Australia: .AU -> .AX
            '.NL': '.AS',     # Netherlands: .NL -> .AS (Amsterdam)
            '.CH': '.SW',     # Switzerland: .CH -> .SW (SIX Swiss Exchange)
            '.FR': '.PA',     # France: .FR -> .PA (Paris)
        }

    # Special case: Hong Kong stocks misclassified as Shanghai
    # Convert short .SS codes (≤4 digits) to .HK format with 4-digit padding
    if ticker.endswith('.SS'):
        base_code = ticker.replace('.SS', '')
        if len(base_code) <= 4 and base_code.isdigit():
            # Hong Kong stocks need 4-digit padding with leading zeros
            padded_code = base_code.zfill(4)
            return f"{padded_code}.HK"

    # Apply exchange transformations
    for old_suffix, new_suffix in exchange_transforms.items():
        if ticker.endswith(old_suffix):
            base_ticker = ticker.replace(old_suffix, '')
            return f"{base_ticker}{new_suffix}"

    # Special cases for individual tickers
    special_cases = {
        'HSBC.L': 'HSBA.L',
        'HSBC.UK': 'HSBA.L',
        'LVMH.PA': 'LVMHF',
    }

    if ticker in special_cases:
        return special_cases[ticker]

    # Check for invalid ticker patterns (likely bonds/derivatives)
    invalid_patterns = [
        r'^\d{4}[A-Z]\d$',  # Pattern like 0000J0, 0008T0
        # Removed 6-digit pattern - these are international ETFs
    ]

    for pattern in invalid_patterns:
        if re.match(pattern, ticker):
            return None  # Mark as invalid - don't try to fetch

    # Handle Hong Kong stocks (both .HK and -HK formats)
    if ticker.endswith('.HK') or ticker.endswith('-HK'):
        # Hong Kong stocks need 4-digit padding with leading zeros
        base_code = ticker.replace('.HK', '').replace('-HK', '')
        if base_code.isdigit():
            padded_code = base_code.zfill(4)
            return f"{padded_code}.HK"
        return ticker.replace('-HK', '.HK')  # Non-numeric, just change format

    # Handle Malaysian stocks (both .MY and -MY formats)
    elif ticker.endswith('.MY') or ticker.endswith('-MY'):
        # Malaysian stocks need .KL suffix
        base_code = ticker.replace('.MY', '').replace('-MY', '')
        if base_code.isdigit():
            padded_code = base_code.zfill(4)
            return f"{padded_code}.KL"
        return ticker.replace('-MY', '.KL').replace('.MY', '.KL')

    # Handle special cases with dots in the base ticker (e.g., SRU.UT-CA, NOVO.B-DK)
    elif '-' in ticker:
        parts = ticker.rsplit('-', 1)  # Split from the right to handle dots in ticker
        if len(parts) == 2:
            base, suffix = parts
        else:
            base = ticker
            suffix = None

        if not suffix:
            return ticker  # Can't process without suffix

        # For US stocks with dual-class shares (.A, .B, etc.), convert period to hyphen for FMP
        if suffix == 'US' and '.' in base:
            # Check if it's a dual-class share pattern (e.g., MOG.A, BRK.B)
            base_parts = base.rsplit('.', 1)
            if len(base_parts) == 2 and len(base_parts[1]) == 1 and base_parts[1].isalpha():
                # Convert MOG.A to MOG-A for FMP
                return f"{base_parts[0]}-{base_parts[1]}"

        # Map common suffixes to FMP equivalents
        suffix_mapping = {
            'US': '',        # US stocks don't need suffix (except dual-class handled above)
            'GB': '.L',      # Great Britain -> London
            'UK': '.L',      # UK -> London
            'CA': '.TO',     # Canada -> Toronto
            'AU': '.AX',     # Australia
            'JP': '.T',      # Japan -> Tokyo
            'DE': '.DE',     # Germany
            'FR': '.PA',     # France -> Paris
            'ES': '.MC',     # Spain -> Madrid
            'IT': '.MI',     # Italy -> Milan
            'CH': '.SW',     # Switzerland
            'NL': '.AS',     # Netherlands -> Amsterdam
            'BE': '.BR',     # Belgium -> Brussels
            'SE': '.ST',     # Sweden -> Stockholm
            'NO': '.OL',     # Norway -> Oslo
            'DK': '.CO',     # Denmark -> Copenhagen
            'FI': '.HE',     # Finland -> Helsinki
            'AT': '.VI',     # Austria -> Vienna
            'PT': '.LS',     # Portugal -> Lisbon
            'IE': '.IR',     # Ireland
            'GR': '.AT',     # Greece -> Athens
            'PL': '.WA',     # Poland -> Warsaw
            'CZ': '.PR',     # Czech -> Prague
            'HU': '.BD',     # Hungary -> Budapest
            'RO': '.BU',     # Romania -> Bucharest
            'TR': '.IS',     # Turkey -> Istanbul
            'ZA': '.JO',     # South Africa -> Johannesburg
            'SG': '.SI',     # Singapore
            'IN': '.NS',     # India -> NSE
            'KR': '.KS',     # South Korea -> KOSPI
            'TW': '.TW',     # Taiwan
            'TH': '.BK',     # Thailand -> Bangkok
            'ID': '.JK',     # Indonesia -> Jakarta
            'MY': '.KL',     # Malaysia -> Kuala Lumpur
            'PH': '.PS',     # Philippines
            'VN': '.VN',     # Vietnam
            'NZ': '.NZ',     # New Zealand
            'IL': '.TA',     # Israel -> Tel Aviv
            'SA': '.SR',     # Saudi Arabia -> Riyadh
            'AE': '.DU',     # UAE -> Dubai
            'EG': '.CA',     # Egypt -> Cairo (conflicts with Canada .CA)
            'BR': '.SA',     # Brazil -> Sao Paulo
            'MX': '.MX',     # Mexico
            'AR': '.BA',     # Argentina -> Buenos Aires
            'CL': '.SN',     # Chile -> Santiago
            'CO': '.CN',     # Colombia
            'PE': '.LM',     # Peru -> Lima
        }

        if suffix in suffix_mapping:
            if suffix_mapping[suffix] == '':
                return base  # US stocks
            else:
                # Special handling for stocks with class suffixes (.A, .B, etc.)
                # FMP requires hyphen instead of period for class suffixes
                # This applies to Nordic countries AND Canada
                if suffix in ['SE', 'DK', 'NO', 'FI', 'CA'] and '.' in base:
                    # Check if it's a class suffix pattern (e.g., VOLV.B, CARL.B, BBD.B)
                    base_parts = base.rsplit('.', 1)
                    if len(base_parts) == 2 and len(base_parts[1]) in [1, 2] and base_parts[1].replace('.', '').isalpha():
                        # Convert VOLV.B to VOLV-B or BBD.B to BBD-B for FMP
                        base = f"{base_parts[0]}-{base_parts[1]}"
                    # Special handling for Canadian Unit Trusts (.UT suffix)
                    elif base.endswith('.UT'):
                        # Remove .UT suffix for FMP (e.g., BEI.UT -> BEI)
                        base = base[:-3]

                # Special handling for Turkish stocks with .E suffix
                # FMP doesn't use the .E equity class indicator
                if suffix == 'TR' and base.endswith('.E'):
                    # Remove .E suffix (e.g., BIMAS.E -> BIMAS)
                    base = base[:-2]

                # For numeric codes, pad to 4 digits for certain exchanges
                if base.isdigit() and suffix in ['MY', 'HK', 'JP', 'KR', 'TW', 'TH', 'ID', 'PH']:
                    base = base.zfill(4)
                return f"{base}{suffix_mapping[suffix]}"
        else:
            # Unknown suffix, just convert dash to dot
            return ticker.replace('-', '.')

    elif ticker.endswith('.L'):
        return ticker  # London stocks - FMP supports .L suffix
    elif ticker.endswith('.AS') or ticker.endswith('.DE') or ticker.endswith('.PA'):
        return ticker  # European stocks - FMP supports these suffixes
    elif ticker.endswith('.TW'):
        return ticker  # Taiwan stocks - FMP supports .TW suffix
    elif ticker.endswith(('.BR', '.MC', '.MI', '.OL', '.HE', '.ST')):
        return ticker  # Other European exchanges that work well

    # Brazilian stock pattern detection - add .SA suffix
    # Brazilian companies typically end with 3, 4, or 11 (share class indicators)
    if ('.' not in ticker and len(ticker) >= 4 and len(ticker) <= 7 and
        (ticker.endswith('3') or ticker.endswith('4') or ticker.endswith('11')) and
        (ticker[:-1].isalpha() or ticker[:-2].isalpha())):
        return f"{ticker}.SA"

    # Default case - return original ticker
    return ticker

def get_fmp_ticker_with_fallback(ticker):
    """
    Enhanced ticker conversion with US listing fallback support
    Based on investigation showing many international stocks available as US listings
    """
    # First try the standard mapping
    fmp_ticker = get_fmp_ticker(ticker)

    # For international stocks, also prepare US listing fallback
    # Based on investigation showing 46.4% recovery rate with US listings
    if '.' in ticker:
        base_ticker = ticker.split('.')[0]
        suffix = ticker.split('.')[-1]

        # These exchanges often have US listings available
        us_fallback_exchanges = [
            'UK', 'IE', 'NL', 'DE', 'CH', 'SI', 'HE', 'CA', 'PS',
            'MX', 'MI', 'JK', 'AU', 'TW', 'BR', 'AT', 'OL', 'VI',
            'NS', 'FR', 'CO'  # Based on investigation results
        ]

        if suffix in us_fallback_exchanges:
            return {'primary': fmp_ticker, 'fallback': base_ticker}

    # For other tickers, just return the standard mapping
    return {'primary': fmp_ticker, 'fallback': None}

def test_fmp_ticker_availability(ticker):
    """
    Test if a ticker is available in FMP API using the quote endpoint
    Returns True if ticker exists, False otherwise
    """
    try:
        if not FMP_API_KEY:
            return False

        # Convert ticker to FMP format
        fmp_ticker = get_fmp_ticker(ticker)

        # If ticker is marked as invalid, don't test
        if fmp_ticker is None:
            return False

        # Test with quote endpoint (lightweight)
        quote_data = fmpsdk.quote(
            apikey=FMP_API_KEY,
            symbol=fmp_ticker
        )

        return quote_data is not None and len(quote_data) > 0

    except Exception as e:
        print(f"[WARNING] Error testing FMP ticker availability for {ticker}: {e}")
        return False

def load_fmp_ticker_mapping():
    """
    Load FMP ticker mapping from JSON file if it exists
    """
    global FMP_TICKER_MAPPING
    try:
        custom_mapping = load_document(
            "fmp_ticker_mapping",
            default={},
            fallback_path="fmp_ticker_mapping.json",
        )
        if isinstance(custom_mapping, dict):
            FMP_TICKER_MAPPING.update(custom_mapping)
            try:
                print(f"[OK] Loaded custom FMP ticker mapping with {len(custom_mapping)} entries")
            except Exception:
                pass
    except Exception as e:
        try:
            print(f"[WARNING] Could not load custom FMP ticker mapping: {e}")
        except Exception:
            pass

# Load custom mapping on module import
load_fmp_ticker_mapping()

# Yahoo Finance Ticker Mapping
YAHOO_TICKER_MAPPING = {}

def load_yahoo_ticker_mapping():
    """
    Load Yahoo Finance ticker mapping from JSON file if it exists
    """
    global YAHOO_TICKER_MAPPING
    try:
        mapping = load_document(
            "yahoo_ticker_mapping",
            default={},
            fallback_path="yahoo_ticker_mapping.json",
        )
        if isinstance(mapping, dict):
            YAHOO_TICKER_MAPPING = mapping
            try:
                print(f"[OK] Loaded Yahoo ticker mapping with {len(YAHOO_TICKER_MAPPING)} entries")
            except Exception:
                pass
    except Exception as e:
        try:
            print(f"[WARNING] Could not load Yahoo ticker mapping: {e}")
        except Exception:
            pass

# Load Yahoo mapping on module import
load_yahoo_ticker_mapping()

def get_yahoo_ticker(ticker):
    """
    Convert ticker to Yahoo Finance format
    """
    import re

    # Remove -US suffix as Yahoo Finance doesn't use it for US stocks
    if ticker.endswith('-US'):
        ticker = ticker[:-3]

    # Check mapping first
    if ticker in YAHOO_TICKER_MAPPING:
        return YAHOO_TICKER_MAPPING[ticker]

    # Default conversions if not in mapping
    # Hong Kong stocks need 4-digit format
    if ticker.endswith('.HK'):
        match = re.match(r'^(\d+)\.HK$', ticker)
        if match:
            number = match.group(1)
            return f"{number.zfill(4)}.HK"

    # Australian stocks use .AX in Yahoo
    if ticker.endswith('.AU'):
        return ticker[:-3] + '.AX'

    # No special handling needed for stocks/ETFs

    # Default: return as is
    return ticker


def process_alerts_in_batches(alerts, process_function, max_workers=None):
    """
    Process alerts in batches to respect rate limits

    Args:
        alerts: List of alerts to process
        process_function: Function to process each alert
        max_workers: Maximum number of worker threads (default: batch_size)

    Returns:
        List of results
    """
    if max_workers is None:
        max_workers = 5  # Default value

    batch_size = 10  # Default batch size
    results = []

    for i in range(0, len(alerts), batch_size):
        batch = alerts[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(alerts) + batch_size - 1)//batch_size} ({len(batch)} alerts)")

        # Process batch with limited concurrency
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_results = list(executor.map(process_function, batch))
            results.extend(batch_results)

        # Add delay between batches to respect rate limits
        if i + batch_size < len(alerts):
            time.sleep(1)  # Small delay between batches

    return results


# FMP API client is initialized later when needed
client = None


# Path to CSV file for storing exchange and stock data
CSV_FILE_PATH = "cleaned_data.csv"

# Path to main database with all securities
MAIN_DATABASE_PATH = "main_database_with_etfs.json"

# Path to CSV file for storing stock alerts
FUTURES_ALERTS_FILE_PATH = "futures_alerts.json"

# Function to load stock exchange and ticker data from database
def load_market_data():
    metadata = fetch_stock_metadata_map()
    if metadata:
        rows = []
        for symbol, info in metadata.items():
            raw = info.get("raw_payload") if isinstance(info.get("raw_payload"), dict) else {}
            get = lambda key: info.get(key) if info.get(key) is not None else raw.get(key)

            row = {
                "Symbol": symbol,
                "Name": get("name") or "",
                "Country": get("country") or "",
                "Exchange": get("exchange") or "",
                "Asset_Type": get("asset_type") or "Stock",
                "RBICS_Economy": get("rbics_economy") or "",
                "RBICS_Sector": get("rbics_sector") or "",
                "RBICS_Subsector": get("rbics_subsector") or "",
                "RBICS_Industry_Group": get("rbics_industry_group") or "",
                "RBICS_Industry": get("rbics_industry") or "",
                "RBICS_Subindustry": get("rbics_subindustry") or "",
                "ETF_Issuer": get("etf_issuer") or "",
                "ETF_Asset_Class": get("etf_asset_class") or "",
                "ETF_Focus": get("etf_focus") or "",
                "ETF_Niche": get("etf_niche") or "",
                "Expense_Ratio": get("expense_ratio") or "",
                "AUM": get("aum") or "",
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            return df

    if os.path.exists(CSV_FILE_PATH):
        return pd.read_csv(CSV_FILE_PATH)

    return pd.DataFrame(columns=["Symbol", "Name", "Country", "Exchange", "Asset_Type"])

# Add caching only if streamlit is available
if STREAMLIT_AVAILABLE:
    load_market_data = st.cache_data(ttl=1800)(load_market_data)

# Predefined suggestions for technical indicators (Single timeframe mode)
predefined_suggestions = [
    "sma(period = )[-1]",
    "hma(period = )[-1]",
    "rsi(period = )[-1]",
    "ema(period = )[-1]",
    "slope_sma(period = )[-1]",
    "slope_ema(period = )[-1]",
    "slope_hma(period = )[-1]",
    "bb(period = , std_dev = , type = )[-1]",
    "macd(fast_period = , slow_period = , signal_period = , type = )[-1]",
    "breakout",
    "atr(period = )[-1]",
    "cci(period = )[-1]",
    "roc(period = )[-1]",
    "williamsr(period = )[-1]",
    "Close[-1]",
    "Open[-1]",
    "Low[-1]",
    "High[-1]",
    "HARSI_Flip(period = , smoothing = )[-1]",
    "SROCST(ma_type = EMA, lsma_offset = 0, smoothing_length = 12, kalman_src = Close, sharpness = 25, filter_period = 1, roc_length = 9, k_length = 14, k_smoothing = 1, d_smoothing = 3)[-1]"
]


# Predefined suggestions for multiple timeframes mode
predefined_suggestions_alt = [
    "sma(period = ,timeframe = )[-1]", "hma(period = ,timeframe = )[-1]",
    "rsi(period = ,timeframe = )[-1]", "ema(period = ,timeframe = )[-1]",
    "slope_sma(period = ,timeframe = )[-1]", "slope_ema(period = ,timeframe = )[-1]",
    "slope_hma(period = ,timeframe = )[-1]", "bb(period = , std_dev = , type = ,timeframe = )[-1]",
    "macd(fast_period = , slow_period = , signal_period = , type = ,timeframe = )[-1]", "Breakout",
    "atr(period = ,timeframe = )[-1]", "cci(period = ,timeframe = )[-1]",
    "roc(period = ,timeframe = )[-1]", "WilliamSR(period = ,timeframe = )[-1]",
    "psar(acceleration = , max_acceleration = ,timeframe = )[-1]", "Close(timeframe = )[-1]",
    "Open(timeframe = )[-1]", "Low(timeframe = )[-1]", "High(timeframe = )[-1]"
]

inverse_map = {'>': '<=', '<': '>=', '==': '!=', '!=': '==', '>=': '<', '<=': '>'}

# Function to add blank spaces for UI formatting
def bl_sp(n):
    """Returns blank spaces for UI spacing in Streamlit."""
    return '\u200e ' * (n + 1)



# Function to log messages to Discord
def log_to_discord(message: str, *args, **kwargs):
    """Log messages to Discord using async logger for better performance"""
    # Try to use async logger first (much faster)
    try:
        from src.services.discord_logger import log_to_discord_async
        log_to_discord_async(str(message))
        return
    except ImportError:
        pass

    # Fallback to buffer method if async logger not available
    global LOG_BUFFER
    LOG_BUFFER.append(str(message))

# Function to split a long message into multiple code blocks
def split_message(message, max_length):
    lines = message.split("\n")
    chunks = []
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) + 1 < max_length - 6:  # 6 for code block fences
            current_chunk += line + "\n"
        else:
            chunks.append(f"```{current_chunk.strip()}```")
            current_chunk = line + "\n"
    if current_chunk:
        chunks.append(f"```{current_chunk.strip()}```")

    return chunks

# Function to flush log buffer to Discord
def flush_logs_to_discord():
    """Flush Discord logs using async logger for better performance"""
    # Try to use async logger first (much faster)
    try:
        from src.services.discord_logger import flush_discord_logs
        flush_discord_logs()
        return
    except ImportError:
        pass

    # Fallback to original method
    global LOG_BUFFER
    if not LOG_BUFFER:
        return

    # Check if webhook URLs are configured
    if not WEBHOOK_URL_LOGGING:
        LOG_BUFFER.clear()  # Clear buffer even if we can't send
        return

    full_message = "\n".join(LOG_BUFFER)
    messages = split_message(full_message, MAX_DISCORD_MESSAGE_LENGTH)
    tag = get_discord_environment_tag()

    for msg in messages:
        payload = {"content": tag + msg}
        try:
            response = requests.post(WEBHOOK_URL_LOGGING, json=payload)
            response.raise_for_status()
            time.sleep(0.1)  # Reduced delay from 1s to 0.1s
        except requests.exceptions.RequestException as e:
            # Silently fail - can't print in Streamlit environment
            break  # Exit on failure to prevent flooding
        # Skip second webhook if it's the same as the first
        if WEBHOOK_URL_LOGGING_2 and WEBHOOK_URL_LOGGING_2 != WEBHOOK_URL_LOGGING:
            try:
                response_2 = requests.post(WEBHOOK_URL_LOGGING_2, json=payload)
                response_2.raise_for_status()
                time.sleep(0.1)  # Reduced delay from 1s to 0.1s
            except requests.exceptions.RequestException as e:
                # Silently fail - can't print in Streamlit environment
                break  # Exit on failure to prevent flooding

    LOG_BUFFER.clear()  # Clear buffer after successful sends

@st.cache_data(ttl=60) if STREAMLIT_AVAILABLE else lambda func: func
def get_asset_type(ticker):
    """
    Determine the asset type based on ticker symbol
    Returns: 'us_stock' or 'international_stock'
    """
    ticker_upper = ticker.upper()

    # International stock patterns (non-US exchanges)
    international_patterns = [
        '.HK', '.SI', '.TW', '.KL', '.KS', '.NS', '.SS', '.BK', '.JK', '.PS', '.HM',  # Asian
        '.AS', '.L', '.PA', '.DE', '.MI', '.SW', '.CH', '.NL', '.ES', '.ST', '.OL', '.CO', '.HE', '.BR', '.IE', '.LS', '.VI', '.WA', '.AT', '.BD', '.PR', '.IS'  # European
    ]

    # Check for international stocks
    for pattern in international_patterns:
        if ticker_upper.endswith(pattern):
            return 'international_stock'

    # Default to US stock
    return 'us_stock'

def is_us_stock(ticker):
    """
    Check if a ticker is a US stock (no country suffix)
    """
    ticker_upper = ticker.upper()

    # International stock patterns (non-US exchanges)
    international_patterns = [
        '.HK', '.SI', '.TW', '.KL', '.KS', '.NS', '.SS', '.BK', '.JK', '.PS', '.HM',  # Asian
        '.AS', '.L', '.PA', '.DE', '.MI', '.SW', '.CH', '.NL', '.ES', '.ST', '.OL', '.CO', '.HE', '.BR', '.IE', '.LS', '.VI', '.WA', '.AT', '.BD', '.PR', '.IS'  # European
    ]

    # Check for international patterns
    for pattern in international_patterns:
        if ticker_upper.endswith(pattern):
            return False

        if pattern in ticker_upper:
            return False

    return True

def grab_new_data_fmp(ticker, timespan="1d", period="1y"):
    """
    Fetch historical data from FMP API using backend_fmp module
    Handles daily and weekly timeframes properly
    """
    from src.services.backend_fmp import FMPDataFetcher

    if not FMP_API_KEY:
        print(f"[ERROR] FMP API key not available")
        return None

    try:
        # Use the FMPDataFetcher class which handles all the complexity
        fetcher = FMPDataFetcher(api_key=FMP_API_KEY)

        # Map timespan to proper format for backend_fmp
        if timespan in ["1wk", "weekly"]:
            # Pass timeframe parameter to get proper weekly resampling
            df = fetcher.get_historical_data(ticker, period="1day", timeframe="1wk")
        elif timespan in ["1d", "daily"]:
            df = fetcher.get_historical_data(ticker, period="1day", timeframe="1d")
        else:
            # Handle intraday timeframes
            timeframe_map = {
                "1min": "1min",
                "5min": "5min",
                "15min": "15min",
                "30min": "30min",
                "1hour": "1hour",
                "4hour": "4hour"
            }

            if timespan in timeframe_map:
                df = fetcher.get_historical_data(ticker, period=timeframe_map[timespan])
            else:
                print(f"[WARNING] Unsupported timespan: {timespan}")
                return None

        if df is not None and not df.empty:
            print(f"[INFO] Successfully fetched {len(df)} records for {ticker} (timespan: {timespan})")
            # Check if we have enough data for indicators
            if timespan in ["1wk", "weekly"] and len(df) < 200:
                print(f"[WARNING] Only {len(df)} weekly records for {ticker}, may not be enough for SMA(200)")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to fetch FMP data for {ticker}: {e}")
        return None

@st.cache_data(ttl=60) if STREAMLIT_AVAILABLE else lambda func: func
def grab_new_data_universal(ticker, timespan="1d", period="1y"):
    """
    Universal data fetching function - FMP ONLY
    - US stocks: FMP API only
    - International stocks: FMP API only
    - ETFs: FMP API only
    """
    asset_type = get_asset_type(ticker)

    print(f"[INFO] Detected asset type for {ticker}: {asset_type}")

    # Always use FMP API for everything - no Yahoo Finance fallback
    if FMP_API_KEY:
        try:
            print(f"[FMP] Using FMP API for {asset_type}: {ticker}")
            fmp_data = grab_new_data_fmp(ticker, timespan, period)
            if fmp_data is not None and not fmp_data.empty:
                return fmp_data
            else:
                print(f"[WARNING] FMP API returned no data for {ticker}")
                return None  # Return None instead of falling back to Yahoo
        except Exception as e:
            print(f"[ERROR] FMP API failed for {ticker}: {e}")
            return None  # Return None instead of falling back to Yahoo
    else:
        print(f"[ERROR] FMP API key not available")
        return None  # Return None if no FMP API key

def calculate_ratio(df1,df2):
    try:
        t1 = df1.columns.get_level_values('Ticker')[0]
        t2 = df2.columns.get_level_values('Ticker')[0]

        # pull out just the metrics (Volume, VWAP, Open, …) for each ticker
        a = df1.xs(t1, level='Ticker', axis=1)
        b = df2.xs(t2, level='Ticker', axis=1)

        # element-wise ratio
        df3 = a.div(b)
        df3.columns = df3.columns.droplevel('Ticker')
        df3.columns.name = None
        return df3
    except:
        df3 = df1.div(df2)
        df3.columns.name = None
        df3.index.name = None
        return df3

def calculate_cross_exchange_ratio(df1, df2, ticker1, ticker2):
    """
    Calculate ratio between stocks from different exchanges
    Handles different data formats and normalizes them
    """
    try:
        # Normalize dataframes to have consistent column names
        df1_normalized = normalize_dataframe(df1, ticker1)
        df2_normalized = normalize_dataframe(df2, ticker2)

        # Ensure both dataframes have the same date range
        common_dates = df1_normalized.index.intersection(df2_normalized.index)
        df1_aligned = df1_normalized.loc[common_dates]
        df2_aligned = df2_normalized.loc[common_dates]

        # Calculate ratio
        ratio_df = df1_aligned.div(df2_aligned)

        # Add metadata
        ratio_df.attrs['ticker1'] = ticker1
        ratio_df.attrs['ticker2'] = ticker2
        ratio_df.attrs['ratio_type'] = 'cross_exchange'

        return ratio_df

    except Exception as e:
        print(f"Error calculating cross-exchange ratio: {e}")
        return None

def normalize_dataframe(df, ticker):
    """
    Normalize dataframe to have consistent column names regardless of exchange
    """
    try:
        # Handle different column structures
        if isinstance(df.columns, pd.MultiIndex):
            if 'Ticker' in df.columns.names:
                normalized_df = df.xs(ticker, level='Ticker', axis=1)
            else:
                normalized_df = df
        else:
            normalized_df = df

        # Ensure we have the standard OHLCV columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = [col for col in normalized_df.columns if col in required_columns]

        if len(available_columns) < 4:  # Need at least OHLC
            raise ValueError(f"Insufficient data columns for {ticker}")

        # Select only the available OHLCV columns
        normalized_df = normalized_df[available_columns]

        # Ensure numeric data
        for col in normalized_df.columns:
            normalized_df[col] = pd.to_numeric(normalized_df[col], errors='coerce')

        # Remove any rows with NaN values
        normalized_df = normalized_df.dropna()

        return normalized_df

    except Exception as e:
        print(f"Error normalizing dataframe for {ticker}: {e}")
        return None

def validate_conditions(entry_conditions_list):
    print("Validating conditions...")

    # Handle dict format (from Add_Alert page)
    if isinstance(entry_conditions_list, dict):
        for key, value in entry_conditions_list.items():
            if isinstance(value, dict):
                conditions = value.get("conditions", "")
                # If conditions is a list of strings
                if isinstance(conditions, list):
                    for condition in conditions:
                        if not condition or (isinstance(condition, str) and condition.strip() == ""):
                            print("Empty condition found.")
                            return False
                        if isinstance(condition, str) and condition.count("[") != condition.count("]"):
                            print("Unclosed brackets found.")
                            return False
                # If conditions is a single string or other format
                elif conditions:
                    if isinstance(conditions, str):
                        if conditions.strip() == "":
                            print("Empty condition found.")
                            return False
                        if conditions.count("[") != conditions.count("]"):
                            print("Unclosed brackets found.")
                            return False
    # Handle list format
    elif isinstance(entry_conditions_list, list):
        for entry in entry_conditions_list:
            # If it's a string (from new UI)
            if isinstance(entry, str):
                if not entry or entry.strip() == "":
                    print("Empty condition found.")
                    return False
                if entry.count("[") != entry.count("]"):
                    print("Unclosed brackets found.")
                    return False
            # If it's a dict (legacy format)
            elif isinstance(entry, dict):
                condition = entry.get("conditions", "")
                if not condition or (isinstance(condition, str) and condition.strip() == ""):
                    print("Empty condition found.")
                    return False
                if isinstance(condition, str) and condition.count("[") != condition.count("]"):
                    print("Unclosed brackets found.")
                    return False
    else:
        print(f"Unknown format for entry_conditions_list: {type(entry_conditions_list)}")
        return False

    return True



#Save an alert with multiple entry conditions as a JSON object in alerts.csv
def _normalize_conditions_for_comparison(entry_conditions_list):
    if isinstance(entry_conditions_list, dict):
        conditions_for_compare = []
        for key, value in entry_conditions_list.items():
            if isinstance(value, dict) and "conditions" in value:
                for idx, condition_str in enumerate(value["conditions"], 1):
                    conditions_for_compare.append({"index": idx, "conditions": condition_str})
        return conditions_for_compare
    return entry_conditions_list


def _conditions_to_storage_format(entry_conditions_list):
    if isinstance(entry_conditions_list, dict):
        conditions_for_save = []
        for key, value in entry_conditions_list.items():
            if isinstance(value, dict) and "conditions" in value:
                for idx, condition_str in enumerate(value["conditions"], 1):
                    conditions_for_save.append({"index": idx, "conditions": condition_str})
        return conditions_for_save
    return entry_conditions_list


def _derive_country(country: Optional[str], exchange: str) -> str:
    if country:
        return country
    try:
        from src.utils.reference_data import get_country_for_exchange
        return get_country_for_exchange(exchange) or exchange
    except ImportError:
        return exchange


def _is_futures_symbol(ticker: str) -> bool:
    futures_db = load_document(
        "futures_database",
        default={},
        fallback_path="futures_database.json",
    ) or {}
    return ticker in futures_db or ticker.upper() in futures_db


def save_alert(name, entry_conditions_list, combination_logic, ticker, stock_name, exchange, timeframe, last_triggered, action, ratio, dtp_params=None, multi_timeframe_params=None, mixed_timeframe_params=None, country=None, adjustment_method=None):
    if not entry_conditions_list or not ticker or not stock_name:
        raise ValueError("Entry conditions cannot be empty.")

    if isinstance(entry_conditions_list, dict):
        has_conditions = any(
            isinstance(value, dict) and value.get("conditions")
            for value in entry_conditions_list.values()
        )
        if not has_conditions:
            raise ValueError("Entry conditions cannot be empty.")

    if validate_conditions(entry_conditions_list) is False:
        raise ValueError("Invalid conditions provided.")

    conditions_for_compare = _normalize_conditions_for_comparison(entry_conditions_list)
    conditions_for_save = _conditions_to_storage_format(entry_conditions_list)

    existing_alerts = repo_list_alerts()
    for alert in existing_alerts:
        if (
            alert.get("stock_name") == stock_name
            and alert.get("ticker") == ticker
            and alert.get("conditions") == conditions_for_compare
            and alert.get("combination_logic") == combination_logic
            and (alert.get("exchange") or alert.get("country")) == exchange
            and alert.get("timeframe") == timeframe
            and alert.get("name") == name
        ):
            raise ValueError("Alert already exists with the same name and data fields.")

    country_value = _derive_country(country, exchange)
    is_futures = _is_futures_symbol(ticker)

    payload = {
        "name": name,
        "stock_name": stock_name,
        "ticker": ticker,
        "conditions": conditions_for_save,
        "combination_logic": combination_logic,
        "last_triggered": last_triggered,
        "action": action,
        "timeframe": timeframe,
        "exchange": exchange,
        "country": country_value,
        "ratio": ratio,
        "is_ratio": False,
    }

    if dtp_params:
        payload["dtp_params"] = dtp_params
    if multi_timeframe_params:
        payload["multi_timeframe_params"] = multi_timeframe_params
    if mixed_timeframe_params:
        payload["mixed_timeframe_params"] = mixed_timeframe_params
    if adjustment_method or is_futures:
        payload["adjustment_method"] = adjustment_method

    return repo_create_alert(payload)

def save_ratio_alert(name, entry_conditions_list, combination_logic, ticker1, ticker2, stock_name, exchange, timeframe, last_triggered, action, ratio, country=None, adjustment_method=None):
    if not entry_conditions_list or not ticker1 or not ticker2 or not stock_name:
        raise ValueError("Entry conditions cannot be empty.")

    if validate_conditions(entry_conditions_list) is False:
        raise ValueError("Invalid conditions provided.")

    conditions_for_compare = _normalize_conditions_for_comparison(entry_conditions_list)
    conditions_for_save = _conditions_to_storage_format(entry_conditions_list)

    existing_alerts = repo_list_alerts()
    for alert in existing_alerts:
        if alert.get("ratio", "No") == "Yes":
            if (
                alert.get("stock_name") == stock_name
                and alert.get("ticker1") == ticker1
                and alert.get("ticker2") == ticker2
                and alert.get("conditions") == conditions_for_compare
                and alert.get("combination_logic") == combination_logic
                and alert.get("exchange") == exchange
                and alert.get("timeframe") == timeframe
                and alert.get("name") == name
            ):
                raise ValueError("Alert already exists with the same name and data fields.")

    country_value = _derive_country(country, exchange)

    payload = {
        "name": name,
        "stock_name": stock_name,
        "ticker": f"{ticker1}_{ticker2}",
        "ticker1": ticker1,
        "ticker2": ticker2,
        "conditions": conditions_for_save,
        "combination_logic": combination_logic,
        "last_triggered": last_triggered,
        "action": action,
        "timeframe": timeframe,
        "exchange": exchange,
        "country": country_value,
        "ratio": ratio or "Yes",
        "is_ratio": True,
    }

    if adjustment_method:
        payload["adjustment_method"] = adjustment_method

    return repo_create_alert(payload)


def update_alert(alert_id, name, entry_conditions_list, combination_logic, ticker, stock_name, exchange, timeframe, last_triggered, action, ratio):
    existing = repo_get_alert(alert_id)
    if not existing:
        raise ValueError(f"Alert with ID {alert_id} not found.")

    if not entry_conditions_list or not ticker or not stock_name:
        raise ValueError("Entry conditions cannot be empty.")

    if validate_conditions(entry_conditions_list) is False:
        raise ValueError("Invalid conditions provided.")

    payload = {
        "name": name,
        "stock_name": stock_name,
        "ticker": ticker,
        "conditions": _conditions_to_storage_format(entry_conditions_list),
        "combination_logic": combination_logic,
        "last_triggered": last_triggered,
        "action": action,
        "timeframe": timeframe,
        "exchange": exchange,
        "country": existing.get("country"),
        "ratio": ratio,
        "is_ratio": existing.get("is_ratio") or str(ratio).lower() in {"yes", "true", "1"},
    }

    repo_update_alert(alert_id, payload)
    return repo_get_alert(alert_id)


def update_ratio_alert(alert_id, name, entry_conditions_list, combination_logic, ticker1, ticker2, stock_name, exchange, timeframe, last_triggered, action, ratio):
    existing = repo_get_alert(alert_id)
    if not existing:
        raise ValueError(f"Alert with ID {alert_id} not found.")

    if not entry_conditions_list or not ticker1 or not ticker2 or not stock_name:
        raise ValueError("Entry conditions cannot be empty.")

    if validate_conditions(entry_conditions_list) is False:
        raise ValueError("Invalid conditions provided.")

    payload = {
        "name": name,
        "stock_name": stock_name,
        "ticker": f"{ticker1}_{ticker2}",
        "ticker1": ticker1,
        "ticker2": ticker2,
        "conditions": _conditions_to_storage_format(entry_conditions_list),
        "combination_logic": combination_logic,
        "last_triggered": last_triggered,
        "action": action,
        "timeframe": timeframe,
        "exchange": exchange,
        "country": existing.get("country"),
        "ratio": ratio or "Yes",
        "is_ratio": True,
    }

    repo_update_alert(alert_id, payload)
    return repo_get_alert(alert_id)


def get_alert_by_id(alert_id):
    """Get a specific alert by its ID"""
    return repo_get_alert(alert_id)


## FOR update_stocks.py ONLY
# Load alert data from JSON file
def load_alert_data():
    return repo_list_alerts()


# Get all unique stock tickers from alert data
def get_all_stocks(alert_data,timeframe):
    return list(set([alert['ticker'] for alert in alert_data if alert['timeframe'] == timeframe]))

#get exchange of a stock
def get_stock_exchange(alert_data, stock):
    return [alert['exchange'] for alert in alert_data if alert['ticker'] == stock][0]

# Get all alerts related to a specific stock
def get_all_alerts_for_stock(alert_data, stock):
    return [alert for alert in alert_data if alert['ticker'] == stock]

# Fetch the latest stock data
@st.cache_data(ttl=900) if STREAMLIT_AVAILABLE else lambda func: func
def get_latest_stock_data(stock, exchange, timespan):
    # Use FMP API for all data fetching
    df = grab_new_data_fmp(stock, timespan=timespan, period="1mo")
    return df


# Load or create the historical database for a stock
def check_database(stock,timeframe):
    file_path = f"data/{stock}_{timeframe}.csv"

    if not os.path.exists(file_path):
        print(f"[FETCH] No existing data for {stock}, fetching new data...")
        exchange = get_stock_exchange(load_alert_data(), stock)
        timeframe = "day" if timeframe == "daily" else "week"
        df = get_latest_stock_data(stock, exchange,timeframe)
        df.reset_index(inplace=True)  # Move Date index to a column
        df.insert(0, "index", range(1, len(df) + 1))
        df.to_csv(file_path, index=False, date_format="%Y-%m-%d")
        return df

    else:
        df = pd.read_csv(file_path)
        # Drop any extra unnamed columns that may have been added in previous runs
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        # Ensure that if an "index" column exists, it is of integer type; if not, create one.
        if 'index' in df.columns:
            df['index'] = df['index'].astype(int)
        else:
            df.insert(0, "index", range(1, len(df) + 1))
        return df



def update_stock_database(stock, new_stock_data,timeframe):
    file_path = f"data/{stock}_{timeframe}.csv"

    # Load existing data
    existing_data = check_database(stock,timeframe)

    # Ensure new_stock_data has the same structure
    new_stock_data.reset_index(inplace=True)  # Convert Date index to column
    new_stock_data = new_stock_data[~new_stock_data["Date"].isin(existing_data["Date"])]

    df_combined = pd.concat([existing_data, new_stock_data])
    df_combined.reset_index(drop=True, inplace=True)

    # Regenerate the "index" column to be consistent
    df_combined['index'] = range(1, len(df_combined) + 1)
    cols = df_combined.columns.tolist()
    if 'index' in cols:
        cols.insert(0, cols.pop(cols.index('index')))
    df_combined = df_combined[cols]

    # Save the combined data consistently without using pandas' default index
    df_combined.to_csv(file_path, index=False, date_format="%Y-%m-%d")

    return df_combined


def send_alert(stock, alert, condition_str, df):
    # Ensure the condition_str is actually a string
    if not isinstance(condition_str, str):
        print(f"[Alert Check] Provided condition is not a string: {condition_str}")
        return

    current_price = df.iloc[-1]['Close']
    # Add action to the alert
    action = alert['action']
    timeframe = alert['timeframe']

    # Get proper exchange name from country
    from src.utils.reference_data import get_exchange_from_country
    country = alert.get('exchange', 'Unknown')
    exchange = get_exchange_from_country(country)

    # Send the alert via Discord
    send_stock_alert(WEBHOOK_URL, timeframe, alert["name"], stock, condition_str, current_price, action, exchange)

    # Only send to second webhook if it's different from the first
    if WEBHOOK_URL_2 and WEBHOOK_URL_2 != WEBHOOK_URL:
        send_stock_alert(WEBHOOK_URL_2, timeframe, alert["name"], stock, condition_str, current_price, action, exchange)

    # Also send to ALL portfolio channels that contain this stock
    try:
        from src.services.portfolio_discord import portfolio_manager
        portfolios_with_stock = portfolio_manager.get_portfolios_for_stock(stock)

        for portfolio_id, portfolio in portfolios_with_stock:
            webhook_url = portfolio.get("discord_webhook", "")
            if webhook_url and portfolio.get("enabled", True):
                # Send the same alert to this portfolio channel
                portfolio_name = portfolio.get("name", "Portfolio")
                send_stock_alert(webhook_url, timeframe, f"[{portfolio_name}] {alert['name']}", stock, condition_str, current_price, action, exchange)
                log_to_discord(f"  → Also sent to portfolio: {portfolio_name}")
    except Exception as e:
        pass  # Silently skip if portfolio system not available

    # Send to custom Discord channels based on condition matching
    try:
        custom_channels = load_document(
            "custom_discord_channels",
            default={},
            fallback_path="custom_discord_channels.json",
        ) or {}

        # Check each custom channel
        for channel_name, channel_config in custom_channels.items():
            if channel_config.get('enabled', True):
                # Check if the triggered condition matches this channel's condition
                # Need to normalize the condition strings for comparison
                triggered_condition_normalized = condition_str.replace(' ', '')
                channel_condition_normalized = channel_config.get('condition', '').replace(' ', '')

                # Check if the triggered condition contains the channel's condition
                if channel_condition_normalized in triggered_condition_normalized:
                    webhook_url = channel_config.get('webhook_url')
                    if webhook_url:
                        # Send alert to this custom channel
                        send_stock_alert(webhook_url, timeframe, f"[{channel_name}] {alert['name']}", stock, condition_str, current_price, action, exchange)
                        log_to_discord(f"  → Also sent to custom channel: {channel_config.get('channel_name', channel_name)}")
    except Exception:
        pass  # Silently skip if custom channels not configured or error occurs

    log_to_discord(f"[Alert Triggered] '{alert['name']}' for {stock}: condition '{condition_str}' at {datetime.datetime.now()}.")


def send_stock_alert(webhook_url, timeframe, alert_name, ticker, triggered_condition, current_price, action, exchange='Unknown'):
    from src.utils.discord_env import is_discord_send_enabled
    if not is_discord_send_enabled():
        return
    # Check if webhook URL is valid
    if not webhook_url:
        print(f"[WARNING] No webhook URL configured for alert: {alert_name}")
        return

    # Change the color based on the action
    color = 0x00ff00 if action == "Buy" else 0xff0000

    # Format timeframe for display
    timeframe_display = {
        "1d": "1D (Daily)",
        "1wk": "1W (Weekly)",
        "1w": "1W (Weekly)",
        "1D": "1D (Daily)",
        "1W": "1W (Weekly)",
        "daily": "1D (Daily)",
        "weekly": "1W (Weekly)"
    }.get(timeframe.lower() if isinstance(timeframe, str) else timeframe, timeframe)

    tag = get_discord_environment_tag().strip()
    embed = {
        "title": f"{tag} [ALERT] {alert_name} ({ticker})",
        "description": f"The condition **{triggered_condition}** was triggered. \n Action: {action}",
        "fields": [
            {
                "name": "Timeframe",
                "value": timeframe_display,
                "inline": True
            },
            {
                "name": "Exchange",
                "value": exchange,
                "inline": True
            },
            {
                "name": "Current Price",
                "value": f"${current_price:.2f}",
                "inline": True
            }
        ],
        "color": color,  # Default green color.
        "timestamp": datetime.datetime.now(timezone.utc).isoformat()
        }

    payload = {
        "embeds": [embed]
    }

    try:
        response = requests.post(webhook_url, json=payload)
        if response.status_code == 204:
            print("Alert sent successfully!")
        else:
            print(f"Failed to send alert. HTTP Status Code: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"An error occurred: {e}")




ops = {
    '>': operator.gt,
    '<': operator.lt,
    '==': operator.eq,
    '>=': operator.ge,
    '<=': operator.le,
    '!=': operator.ne
}

supported_indicators = {
    "sma": SMA,
    "ema": EMA,
    "hma": HMA,
    "frama": FRAMA,
    "kama": KAMA,
    "ewo": EWO,
    "ma_spread_zscore": MA_SPREAD_ZSCORE,
    "slope_sma": SLOPE_SMA,
    "slope_ema": SLOPE_EMA,
    "slope_hma": SLOPE_HMA,
    "rsi": RSI,
    "atr": ATR,
    "cci": CCI,
    "bb": BBANDS,
    "roc": ROC,
    "williamsr": WILLR,
    "macd": MACD,
    "psar": SAR,
    "HARSI_Flip" : HARSI_Flip,
}

# USED IN APPLY FUNCTION ONLY
period_and_input = ['sma','ema','rsi','hma','slope_sma','slope_ema','slope_hma','roc']

period_only = ['sma','ema','rsi','hma','slope_sma','slope_ema','slope_hma','roc', 'atr', 'cci', 'willr', 'bbands']

def get_dst_adjusted_time(local_time_str, timezone_str, target_timezone="America/New_York"):
    """
    Convert local market time to target timezone with DST awareness

    Args:
        local_time_str: Time string in local market time (e.g., "4:00 PM")
        timezone_str: Market timezone (e.g., "Asia/Hong_Kong")
        target_timezone: Target timezone for conversion (default: EST/EDT)

    Returns:
        tuple: (hour, minute) in target timezone
    """
    try:
        # Parse the local time
        if "PM" in local_time_str.upper():
            time_part = local_time_str.replace(" PM", "").replace(" pm", "")
            hour, minute = map(int, time_part.split(":"))
            if hour != 12:
                hour += 12
        elif "AM" in local_time_str.upper():
            time_part = local_time_str.replace(" AM", "").replace(" am", "")
            hour, minute = map(int, time_part.split(":"))
            if hour == 12:
                hour = 0
        else:
            # Handle 24-hour format
            hour, minute = map(int, local_time_str.split(":"))

        # Create a datetime object in the market's timezone
        market_tz = pytz.timezone(timezone_str)
        target_tz = pytz.timezone(target_timezone)

        # Use today's date for the conversion (DST will be automatically handled)
        today = datetime.datetime.now().date()
        local_dt = datetime.datetime.combine(today, datetime.time(hour, minute))

        # Localize to market timezone
        local_dt = market_tz.localize(local_dt)

        # Convert to target timezone
        target_dt = local_dt.astimezone(target_tz)

        return target_dt.hour, target_dt.minute

    except Exception as e:
        print(f"Error converting time {local_time_str} from {timezone_str}: {e}")
        # Fallback to manual conversion
        return convert_time_manual(local_time_str, timezone_str, target_timezone)

def convert_time_manual(local_time_str, timezone_str, target_timezone="America/New_York"):
    """
    Manual time conversion with DST handling for common markets
    """
    # Parse local time
    if "PM" in local_time_str.upper():
        time_part = local_time_str.replace(" PM", "").replace(" pm", "")
        hour, minute = map(int, time_part.split(":"))
        if hour != 12:
            hour += 12
    elif "AM" in local_time_str.upper():
        time_part = local_time_str.replace(" AM", "").replace(" am", "")
        hour, minute = map(int, time_part.split(":"))
        if hour == 12:
            hour = 0
    else:
        hour, minute = map(int, local_time_str.split(":"))

    # DST-aware conversion table
    # Format: (market_tz, local_hour, local_minute, est_hour_dst, est_minute_dst, est_hour_standard, est_minute_standard)
    conversion_table = {
        "Asia/Hong_Kong": (hour, minute, hour - 13, minute, hour - 12, minute),  # HK: UTC+8
        "Asia/Singapore": (hour, minute, hour - 13, minute, hour - 12, minute),  # SG: UTC+8
        "Asia/Taipei": (hour, minute, hour - 13, minute, hour - 12, minute),     # TW: UTC+8
        "Asia/Kuala_Lumpur": (hour, minute, hour - 13, minute, hour - 12, minute), # MY: UTC+8
        "Asia/Tokyo": (hour, minute, hour - 14, minute, hour - 13, minute),      # JP: UTC+9
        "Europe/London": (hour, minute, hour - 5, minute, hour - 6, minute),     # UK: UTC+0/+1
        "Europe/Paris": (hour, minute, hour - 6, minute, hour - 7, minute),      # FR: UTC+1/+2
        "Europe/Berlin": (hour, minute, hour - 6, minute, hour - 7, minute),     # DE: UTC+1/+2
        "America/Toronto": (hour, minute, hour + 0, minute, hour + 0, minute),   # CA: Same as US
    }

    # Check if current time is in DST
    now = datetime.datetime.now(pytz.timezone(target_timezone))
    is_dst = now.dst() != datetime.timedelta(0)

    if timezone_str in conversion_table:
        _, _, _, dst_hour, dst_minute, std_hour, std_minute = conversion_table[timezone_str]

        if is_dst:
            return dst_hour, dst_minute
        else:
            return std_hour, std_minute

    # Default fallback
    return hour - 5, minute  # Assume UTC-5 for unknown timezones

def get_market_timezone(exchange_name):
    """
    Get the timezone for a given exchange
    """
    timezone_map = {
        "Hong Kong": "Asia/Hong_Kong",
        "Singapore": "Asia/Singapore",
        "Taiwan": "Asia/Taipei",
        "Malaysia": "Asia/Kuala_Lumpur",
        "Tokyo": "Asia/Tokyo",
        "London": "Europe/London",
        "Euronext Paris": "Europe/Paris",
        "Xetra": "Europe/Berlin",
        "Toronto": "America/Toronto",
        "Nasdaq": "America/New_York",
        "NYSE": "America/New_York",
        "NYSE American": "America/New_York"
    }

    return timezone_map.get(exchange_name, "UTC")

def is_dst_active():
    """
    Check if Daylight Saving Time is currently active in EST/EDT
    """
    now = datetime.datetime.now(pytz.timezone("America/New_York"))
    return now.dst() != datetime.timedelta(0)

def get_dst_status():
    """
    Get current DST status and next transition dates
    """
    ny_tz = pytz.timezone("America/New_York")
    now = datetime.datetime.now(ny_tz)

    # Get next DST transitions
    transitions = ny_tz._utc_transition_times
    current_year = now.year

    # Find next spring forward (DST starts)
    spring_forward = None
    fall_back = None

    for transition in transitions:
        if transition.year >= current_year:
            transition_dt = datetime.datetime.fromtimestamp(transition.timestamp(), ny_tz)
            if transition_dt > now:
                if spring_forward is None:
                    spring_forward = transition_dt
                elif fall_back is None:
                    fall_back = transition_dt
                    break

    return {
        "is_dst": now.dst() != datetime.timedelta(0),
        "current_offset": now.utcoffset(),
        "spring_forward": spring_forward,
        "fall_back": fall_back,
        "current_time": now
    }


def calculate_cache_hit_rate():
    """
    Calculate cache hit rate (simplified - you'd need to track hits/misses)
    """
    # This is a simplified calculation - in practice you'd track actual hits/misses
    if cache_size > 0:
        # Estimate based on cache size and age
        return min(85, cache_size / 100)  # Rough estimate
    return 0

def estimate_daily_requests():
    """
    Estimate daily request volume based on current patterns
    """
    current_time = time.time()

    # Count requests in last hour
    requests_last_hour = 100  # Default estimate

    # Estimate daily requests (assuming consistent pattern)
    estimated_daily = requests_last_hour * 24

    return estimated_daily

def get_rate_limit_recommendations(minute_utilization, hour_utilization):
    """
    Get recommendations based on current utilization
    """
    recommendations = []

    if minute_utilization > 80:
        recommendations.append("[CRITICAL] Minute limit nearly reached - consider pausing processing")
    elif minute_utilization > 60:
        recommendations.append("[WARNING] High minute utilization - monitor closely")

    if hour_utilization > 80:
        recommendations.append("[CRITICAL] Hour limit nearly reached - implement aggressive caching")
    elif hour_utilization > 60:
        recommendations.append("[WARNING] High hour utilization - consider increasing cache duration")

    if minute_utilization < 30 and hour_utilization < 30:
        recommendations.append("[GOOD] Rate limits well within safe range")

        recommendations.append("[SUGGESTION] Low cache usage - consider reducing cache duration")
        recommendations.append("[SUGGESTION] High cache usage - consider increasing cache cleanup")

    return recommendations

    print(f"{'='*50}")
    print(f"[TIME] Requests (last minute): {status['requests_last_minute']}/{status['limit_per_minute']} ({status['minute_utilization_percent']}%)")
    print(f"[TIME] Requests (last hour): {status['requests_last_hour']}/{status['limit_per_hour']} ({status['hour_utilization_percent']}%)")
    print(f"[STATUS] Can make request: {'YES' if status['can_make_request'] else 'NO'}")
    print(f"[CACHE] Cache size: {status['cache_size']} entries")
    print(f"[CACHE] Cache hit rate: ~{status['cache_hit_rate']}%")
    print(f"[STATS] Estimated daily requests: {status['estimated_requests_per_day']:,}")
    print(f"[INTERVAL] Min interval: {status['min_interval']} seconds")

    if status['recommendations']:
        print(f"\n[RECOMMENDATIONS]:")
        for rec in status['recommendations']:
            print(f"   {rec}")

    print(f"{'='*50}\n")

def emergency_rate_limit_pause():
    """
    Emergency function to pause processing when rate limits are exceeded
    """
    print("[EMERGENCY] Rate limits exceeded - pausing processing for 5 minutes")
    time.sleep(300)  # Wait 5 minutes
    print("[RESUMED] Processing resumed")

def adjust_rate_limits_for_high_volume():
    """
    Automatically adjust rate limits for high volume scenarios
    """
    # This function is deprecated
    pass

def check_similar_alerts(stock_name, ticker, entry_conditions_list, combination_logic, exchange, timeframe):
    """
    Check if an alert with similar conditions already exists for the same stock.
    Returns a list of similar alerts that could be updated instead of creating duplicates.
    """
    alerts = repo_list_alerts()

    similar_alerts = []
    for alert in alerts:
        # Check if this is the same stock and ticker
        if (alert["stock_name"] == stock_name and
            alert["ticker"] == ticker and
            alert["exchange"] == exchange and
            alert["timeframe"] == timeframe):

            # Check if conditions are similar (same structure but potentially different values)
            if (alert["conditions"] == entry_conditions_list and
                alert["combination_logic"] == combination_logic):
                similar_alerts.append(alert)

    return similar_alerts

def check_similar_ratio_alerts(stock_name, ticker1, ticker2, entry_conditions_list, combination_logic, exchange, timeframe):
    """
    Check if a ratio alert with similar conditions already exists.
    Returns a list of similar alerts that could be updated instead of creating duplicates.
    """
    alerts = repo_list_alerts()

    similar_alerts = []
    for alert in alerts:
        if (alert.get("ratio") == "Yes") or alert.get("is_ratio"):
            # Check if this is the same stock and tickers
            if (alert["stock_name"] == stock_name and
                alert.get("ticker1") == ticker1 and
                alert.get("ticker2") == ticker2 and
                alert["exchange"] == exchange and
                alert["timeframe"] == timeframe):

                # Check if conditions are similar
                if (alert["conditions"] == entry_conditions_list and
                    alert["combination_logic"] == combination_logic):
                    similar_alerts.append(alert)

    return similar_alerts

def suggest_alert_update(stock_name, ticker, entry_conditions_list, combination_logic, exchange, timeframe):
    """
    Suggest updating an existing alert instead of creating a duplicate.
    Returns a helpful message with suggestions.
    """
    similar_alerts = check_similar_alerts(stock_name, ticker, entry_conditions_list, combination_logic, exchange, timeframe)

    if not similar_alerts:
        return None

    suggestions = []
    for alert in similar_alerts:
        suggestion = {
            "alert_id": alert["alert_id"],
            "name": alert["name"],
            "action": alert.get("action", "Unknown"),
            "last_triggered": alert.get("last_triggered", "Never"),
            "message": f"Alert '{alert['name']}' already exists with the same conditions. Consider updating it instead of creating a duplicate."
        }
        suggestions.append(suggestion)

    return suggestions

def suggest_ratio_alert_update(stock_name, ticker1, ticker2, entry_conditions_list, combination_logic, exchange, timeframe):
    """
    Suggest updating an existing ratio alert instead of creating a duplicate.
    Returns a helpful message with suggestions.
    """
    similar_alerts = check_similar_ratio_alerts(stock_name, ticker1, ticker2, entry_conditions_list, combination_logic, exchange, timeframe)

    if not similar_alerts:
        return None

    suggestions = []
    for alert in similar_alerts:
        suggestion = {
            "alert_id": alert["alert_id"],
            "name": alert["name"],
            "action": alert.get("action", "Unknown"),
            "last_triggered": alert.get("last_triggered", "Never"),
            "message": f"Ratio alert '{alert['name']}' already exists with the same conditions. Consider updating it instead of creating a duplicate."
        }
        suggestions.append(suggestion)

    return suggestions

def get_stock_alerts_summary(stock_name, ticker):
    """
    Get a summary of all existing alerts for a specific stock.
    Returns a formatted summary for display.
    """
    stock_alerts = []
    for alert in repo_list_alerts():
        if (alert["stock_name"] == stock_name and
            alert["ticker"] == ticker):
            summary = {
                "alert_id": alert["alert_id"],
                "name": alert["name"],
                "conditions": alert["conditions"],
                "combination_logic": alert["combination_logic"],
                "timeframe": alert["timeframe"],
                "exchange": alert["exchange"],
                "action": alert.get("action", "Unknown"),
                "last_triggered": alert.get("last_triggered", "Never"),
                "ratio": alert.get("ratio", "No")
            }
            stock_alerts.append(summary)

    return stock_alerts

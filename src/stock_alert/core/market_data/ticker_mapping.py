"""Ticker mapping and conversion utilities for different exchanges and data providers."""

import re
from functools import lru_cache

import fmpsdk

from src.stock_alert.data_access.document_store import load_document

# FMP Ticker Mapping System
# Maps ticker formats to FMP API ticker formats
# Based on investigation results from investigate_fmp_tickers.py
_BASE_FMP_TICKER_MAPPING = {
    # Hong Kong stocks - FMP supports .HK suffix
    "0700.HK": "0700.HK",  # Tencent
    "939.HK": "939.HK",  # China Construction Bank
    "9988.HK": "9988.HK",  # Alibaba
    "1299.HK": "1299.HK",  # AIA Group
    # London stocks - FMP uses different symbols
    "HSBC.L": "HSBA.L",  # HSBC Holdings (FMP uses HSBA.L)
    "BP.L": "BP.L",  # BP
    "VOD.L": "VOD.L",  # Vodafone
    # European stocks - Some work with suffixes, some don't
    "ASML.AS": "ASML.AS",  # ASML Holding (Amsterdam) - FMP supports .AS
    "SAP.DE": "SAP.DE",  # SAP SE (Frankfurt) - FMP supports .DE
    "LVMH.PA": "LVMHF",  # LVMH (Paris) - FMP uses OTC symbol LVMHF
    "NESN.SW": "NESN.SW",  # Nestle (Switzerland)
    # US stocks (usually same)
    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "GOOGL": "GOOGL",
    "AMZN": "AMZN",
    # Special cases - May need ADR alternatives
    "BRK.A": "BRK.A",
    "BRK.B": "BRK.B",
    # ADR alternatives for international stocks
    "TCEHY": "TCEHY",  # Tencent ADR (OTC)
    "LVMHF": "LVMHF",  # LVMH OTC alternative
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
    """Get enhanced FMP ticker mappings from document store."""
    data = load_document(
        "enhanced_fmp_ticker_mapping",
        default={},
        fallback_path="enhanced_fmp_ticker_mapping.json",
    )
    return data if isinstance(data, dict) else {}


@lru_cache(maxsize=1)
def _get_local_exchange_mappings() -> dict:
    """Get local exchange mappings from document store."""
    data = load_document(
        "local_exchange_mappings",
        default={},
        fallback_path="local_exchange_mappings.json",
    )
    return data if isinstance(data, dict) else {}


def get_fmp_ticker(ticker):
    """
    Enhanced ticker conversion for FMP API with comprehensive mapping rules.
    Returns the FMP ticker if mapping exists, otherwise applies transformation rules.

    Args:
        ticker: Original ticker symbol

    Returns:
        Converted FMP ticker symbol or None if invalid
    """
    # First check the main FMP ticker mapping (includes HK fixes)
    if ticker in FMP_TICKER_MAPPING:
        return FMP_TICKER_MAPPING[ticker]

    enhanced_mappings = _get_enhanced_fmp_mappings()

    local_exchange_doc = _get_local_exchange_mappings()
    local_mappings = {}
    for category in local_exchange_doc.get("local_exchange_corrections", {}).values():
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
    for old_suffix, new_suffix in local_exchange_doc.get("exchange_suffix_transforms", {}).items():
        exchange_transforms[f".{old_suffix}"] = f".{new_suffix}"

    # Fallback to default transforms if file loading fails
    if not exchange_transforms:
        exchange_transforms = {
            ".JP": ".T",  # Japan: .JP -> .T
            ".UK": ".L",  # UK: .UK -> .L
            ".CA": ".TO",  # Canada: .CA -> .TO
            ".AU": ".AX",  # Australia: .AU -> .AX
            ".NL": ".AS",  # Netherlands: .NL -> .AS (Amsterdam)
            ".CH": ".SW",  # Switzerland: .CH -> .SW (SIX Swiss Exchange)
            ".FR": ".PA",  # France: .FR -> .PA (Paris)
        }

    # Special case: Hong Kong stocks misclassified as Shanghai
    # Convert short .SS codes (≤4 digits) to .HK format with 4-digit padding
    if ticker.endswith(".SS"):
        base_code = ticker.replace(".SS", "")
        if len(base_code) <= 4 and base_code.isdigit():
            # Hong Kong stocks need 4-digit padding with leading zeros
            padded_code = base_code.zfill(4)
            return f"{padded_code}.HK"

    # Apply exchange transformations
    for old_suffix, new_suffix in exchange_transforms.items():
        if ticker.endswith(old_suffix):
            base_ticker = ticker.replace(old_suffix, "")
            return f"{base_ticker}{new_suffix}"

    # Special cases for individual tickers
    special_cases = {
        "HSBC.L": "HSBA.L",
        "HSBC.UK": "HSBA.L",
        "LVMH.PA": "LVMHF",
    }

    if ticker in special_cases:
        return special_cases[ticker]

    # Check for invalid ticker patterns (likely bonds/derivatives)
    invalid_patterns = [
        r"^\d{4}[A-Z]\d$",  # Pattern like 0000J0, 0008T0
    ]

    for pattern in invalid_patterns:
        if re.match(pattern, ticker):
            return None  # Mark as invalid - don't try to fetch

    # Handle Hong Kong stocks (both .HK and -HK formats)
    if ticker.endswith(".HK") or ticker.endswith("-HK"):
        # Hong Kong stocks need 4-digit padding with leading zeros
        base_code = ticker.replace(".HK", "").replace("-HK", "")
        if base_code.isdigit():
            padded_code = base_code.zfill(4)
            return f"{padded_code}.HK"
        return ticker.replace("-HK", ".HK")  # Non-numeric, just change format

    # Handle Malaysian stocks (both .MY and -MY formats)
    elif ticker.endswith(".MY") or ticker.endswith("-MY"):
        # Malaysian stocks need .KL suffix
        base_code = ticker.replace(".MY", "").replace("-MY", "")
        if base_code.isdigit():
            padded_code = base_code.zfill(4)
            return f"{padded_code}.KL"
        return ticker.replace("-MY", ".KL").replace(".MY", ".KL")

    # Handle special cases with dots in the base ticker (e.g., SRU.UT-CA, NOVO.B-DK)
    elif "-" in ticker:
        parts = ticker.rsplit("-", 1)  # Split from the right to handle dots in ticker
        if len(parts) == 2:
            base, suffix = parts
        else:
            base = ticker
            suffix = None

        if not suffix:
            return ticker  # Can't process without suffix

        # For US stocks with dual-class shares (.A, .B, etc.), convert period to hyphen for FMP
        if suffix == "US" and "." in base:
            # Check if it's a dual-class share pattern (e.g., MOG.A, BRK.B)
            base_parts = base.rsplit(".", 1)
            if len(base_parts) == 2 and len(base_parts[1]) == 1 and base_parts[1].isalpha():
                # Convert MOG.A to MOG-A for FMP
                return f"{base_parts[0]}-{base_parts[1]}"

        # Map common suffixes to FMP equivalents
        suffix_mapping = {
            "US": "",  # US stocks don't need suffix (except dual-class handled above)
            "GB": ".L",  # Great Britain -> London
            "UK": ".L",  # UK -> London
            "CA": ".TO",  # Canada -> Toronto
            "AU": ".AX",  # Australia
            "JP": ".T",  # Japan -> Tokyo
            "DE": ".DE",  # Germany
            "FR": ".PA",  # France -> Paris
            "ES": ".MC",  # Spain -> Madrid
            "IT": ".MI",  # Italy -> Milan
            "CH": ".SW",  # Switzerland
            "NL": ".AS",  # Netherlands -> Amsterdam
            "BE": ".BR",  # Belgium -> Brussels
            "SE": ".ST",  # Sweden -> Stockholm
            "NO": ".OL",  # Norway -> Oslo
            "DK": ".CO",  # Denmark -> Copenhagen
            "FI": ".HE",  # Finland -> Helsinki
            "AT": ".VI",  # Austria -> Vienna
            "PT": ".LS",  # Portugal -> Lisbon
            "IE": ".IR",  # Ireland
            "GR": ".AT",  # Greece -> Athens
            "PL": ".WA",  # Poland -> Warsaw
            "CZ": ".PR",  # Czech -> Prague
            "HU": ".BD",  # Hungary -> Budapest
            "RO": ".BU",  # Romania -> Bucharest
            "TR": ".IS",  # Turkey -> Istanbul
            "ZA": ".JO",  # South Africa -> Johannesburg
            "SG": ".SI",  # Singapore
            "IN": ".NS",  # India -> NSE
            "KR": ".KS",  # South Korea -> KOSPI
            "TW": ".TW",  # Taiwan
            "TH": ".BK",  # Thailand -> Bangkok
            "ID": ".JK",  # Indonesia -> Jakarta
            "MY": ".KL",  # Malaysia -> Kuala Lumpur
            "PH": ".PS",  # Philippines
            "VN": ".VN",  # Vietnam
            "NZ": ".NZ",  # New Zealand
            "IL": ".TA",  # Israel -> Tel Aviv
            "SA": ".SR",  # Saudi Arabia -> Riyadh
            "AE": ".DU",  # UAE -> Dubai
            "EG": ".CA",  # Egypt -> Cairo (conflicts with Canada .CA)
            "BR": ".SA",  # Brazil -> Sao Paulo
            "MX": ".MX",  # Mexico
            "AR": ".BA",  # Argentina -> Buenos Aires
            "CL": ".SN",  # Chile -> Santiago
            "CO": ".CN",  # Colombia
            "PE": ".LM",  # Peru -> Lima
        }

        if suffix in suffix_mapping:
            if suffix_mapping[suffix] == "":
                return base  # US stocks
            else:
                # Special handling for stocks with class suffixes (.A, .B, etc.)
                # FMP requires hyphen instead of period for class suffixes
                # This applies to Nordic countries AND Canada
                if suffix in ["SE", "DK", "NO", "FI", "CA"] and "." in base:
                    # Check if it's a class suffix pattern (e.g., VOLV.B, CARL.B, BBD.B)
                    base_parts = base.rsplit(".", 1)
                    if (
                        len(base_parts) == 2
                        and len(base_parts[1]) in [1, 2]
                        and base_parts[1].replace(".", "").isalpha()
                    ):
                        # Convert VOLV.B to VOLV-B or BBD.B to BBD-B for FMP
                        base = f"{base_parts[0]}-{base_parts[1]}"
                    # Special handling for Canadian Unit Trusts (.UT suffix)
                    elif base.endswith(".UT"):
                        # Remove .UT suffix for FMP (e.g., BEI.UT -> BEI)
                        base = base[:-3]

                # Special handling for Turkish stocks with .E suffix
                # FMP doesn't use the .E equity class indicator
                if suffix == "TR" and base.endswith(".E"):
                    # Remove .E suffix (e.g., BIMAS.E -> BIMAS)
                    base = base[:-2]

                # For numeric codes, pad to 4 digits for certain exchanges
                if base.isdigit() and suffix in ["MY", "HK", "JP", "KR", "TW", "TH", "ID", "PH"]:
                    base = base.zfill(4)
                return f"{base}{suffix_mapping[suffix]}"
        else:
            # Unknown suffix, just convert dash to dot
            return ticker.replace("-", ".")

    elif ticker.endswith(".L"):
        return ticker  # London stocks - FMP supports .L suffix
    elif ticker.endswith(".AS") or ticker.endswith(".DE") or ticker.endswith(".PA"):
        return ticker  # European stocks - FMP supports these suffixes
    elif ticker.endswith(".TW"):
        return ticker  # Taiwan stocks - FMP supports .TW suffix
    elif ticker.endswith((".BR", ".MC", ".MI", ".OL", ".HE", ".ST")):
        return ticker  # Other European exchanges that work well

    # Brazilian stock pattern detection - add .SA suffix
    # Brazilian companies typically end with 3, 4, or 11 (share class indicators)
    if (
        "." not in ticker
        and len(ticker) >= 4
        and len(ticker) <= 7
        and (ticker.endswith("3") or ticker.endswith("4") or ticker.endswith("11"))
        and (ticker[:-1].isalpha() or ticker[:-2].isalpha())
    ):
        return f"{ticker}.SA"

    # Default case - return original ticker
    return ticker


def get_fmp_ticker_with_fallback(ticker):
    """
    Enhanced ticker conversion with US listing fallback support.
    Based on investigation showing many international stocks available as US listings.

    Args:
        ticker: Original ticker symbol

    Returns:
        dict with 'primary' and 'fallback' ticker options
    """
    # First try the standard mapping
    fmp_ticker = get_fmp_ticker(ticker)

    # For international stocks, also prepare US listing fallback
    # Based on investigation showing 46.4% recovery rate with US listings
    if "." in ticker:
        base_ticker = ticker.split(".")[0]
        suffix = ticker.split(".")[-1]

        # These exchanges often have US listings available
        us_fallback_exchanges = [
            "UK",
            "IE",
            "NL",
            "DE",
            "CH",
            "SI",
            "HE",
            "CA",
            "PS",
            "MX",
            "MI",
            "JK",
            "AU",
            "TW",
            "BR",
            "AT",
            "OL",
            "VI",
            "NS",
            "FR",
            "CO",
        ]

        if suffix in us_fallback_exchanges:
            return {"primary": fmp_ticker, "fallback": base_ticker}

    # For other tickers, just return the standard mapping
    return {"primary": fmp_ticker, "fallback": None}


def test_fmp_ticker_availability(ticker, fmp_api_key):
    """
    Test if a ticker is available in FMP API using the quote endpoint.

    Args:
        ticker: Ticker symbol to test
        fmp_api_key: FMP API key

    Returns:
        True if ticker exists, False otherwise
    """
    try:
        if not fmp_api_key:
            return False

        # Convert ticker to FMP format
        fmp_ticker = get_fmp_ticker(ticker)

        # If ticker is marked as invalid, don't test
        if fmp_ticker is None:
            return False

        # Test with quote endpoint (lightweight)
        quote_data = fmpsdk.quote(apikey=fmp_api_key, symbol=fmp_ticker)

        return quote_data is not None and len(quote_data) > 0

    except Exception as e:
        print(f"[WARNING] Error testing FMP ticker availability for {ticker}: {e}")
        return False


def load_fmp_ticker_mapping():
    """Load FMP ticker mapping from JSON file if it exists."""
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


# Yahoo Finance Ticker Mapping
YAHOO_TICKER_MAPPING = {}


def load_yahoo_ticker_mapping():
    """Load Yahoo Finance ticker mapping from JSON file if it exists."""
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


def get_yahoo_ticker(ticker):
    """
    Convert ticker to Yahoo Finance format.

    Args:
        ticker: Original ticker symbol

    Returns:
        Yahoo Finance formatted ticker
    """
    # Remove -US suffix as Yahoo Finance doesn't use it for US stocks
    if ticker.endswith("-US"):
        ticker = ticker[:-3]

    # Check mapping first
    if ticker in YAHOO_TICKER_MAPPING:
        return YAHOO_TICKER_MAPPING[ticker]

    # Default conversions if not in mapping
    # Hong Kong stocks need 4-digit format
    if ticker.endswith(".HK"):
        match = re.match(r"^(\d+)\.HK$", ticker)
        if match:
            number = match.group(1)
            return f"{number.zfill(4)}.HK"

    # Australian stocks use .AX in Yahoo
    if ticker.endswith(".AU"):
        return ticker[:-3] + ".AX"

    # Default: return as is
    return ticker


# Load custom mappings on module import
load_fmp_ticker_mapping()
load_yahoo_ticker_mapping()

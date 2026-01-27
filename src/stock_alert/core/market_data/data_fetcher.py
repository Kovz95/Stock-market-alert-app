"""Market data fetching utilities."""

import concurrent.futures
import os
import time

import pandas as pd

# Try to import streamlit for caching, but don't fail if not available
try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from src.stock_alert.data_access.metadata_repository import fetch_stock_metadata_map

# Path to CSV file for storing exchange and stock data
CSV_FILE_PATH = "cleaned_data.csv"


def load_market_data():
    """
    Load stock exchange and ticker data from database.

    Returns:
        DataFrame with stock metadata including Symbol, Name, Country, Exchange, etc.
    """
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


def grab_new_data_fmp(ticker, timespan="1d", period="1y", fmp_api_key=None):
    """
    Fetch historical data from FMP API using backend_fmp module.
    Handles daily and weekly timeframes properly.

    Args:
        ticker: Stock ticker symbol
        timespan: Timeframe (1d, 1wk, 1min, etc.)
        period: Period to fetch (1y, etc.)
        fmp_api_key: FMP API key

    Returns:
        DataFrame with OHLCV data or None if error
    """
    from backend_fmp import FMPDataFetcher

    if not fmp_api_key:
        print("[ERROR] FMP API key not available")
        return None

    try:
        # Use the FMPDataFetcher class which handles all the complexity
        fetcher = FMPDataFetcher(api_key=fmp_api_key)

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
                "4hour": "4hour",
            }

            if timespan in timeframe_map:
                df = fetcher.get_historical_data(ticker, period=timeframe_map[timespan])
            else:
                print(f"[WARNING] Unsupported timespan: {timespan}")
                return None

        if df is not None and not df.empty:
            print(
                f"[INFO] Successfully fetched {len(df)} records for {ticker} (timespan: {timespan})"
            )
            # Check if we have enough data for indicators
            if timespan in ["1wk", "weekly"] and len(df) < 200:
                print(
                    f"[WARNING] Only {len(df)} weekly records for {ticker}, may not be enough for SMA(200)"
                )
        return df
    except Exception as e:
        print(f"[ERROR] Failed to fetch FMP data for {ticker}: {e}")
        return None


@st.cache_data(ttl=60) if STREAMLIT_AVAILABLE else lambda func: func
def grab_new_data_universal(ticker, timespan="1d", period="1y", fmp_api_key=None):
    """
    Universal data fetching function - FMP ONLY.
    - US stocks: FMP API only
    - International stocks: FMP API only
    - ETFs: FMP API only

    Args:
        ticker: Stock ticker symbol
        timespan: Timeframe (1d, 1wk, etc.)
        period: Period to fetch
        fmp_api_key: FMP API key

    Returns:
        DataFrame with OHLCV data or None if error
    """
    from src.stock_alert.utils.formatting import get_asset_type

    asset_type = get_asset_type(ticker)

    print(f"[INFO] Detected asset type for {ticker}: {asset_type}")

    # Always use FMP API for everything - no Yahoo Finance fallback
    if fmp_api_key:
        try:
            print(f"[FMP] Using FMP API for {asset_type}: {ticker}")
            fmp_data = grab_new_data_fmp(ticker, timespan, period, fmp_api_key)
            if fmp_data is not None and not fmp_data.empty:
                return fmp_data
            else:
                print(f"[WARNING] FMP API returned no data for {ticker}")
                return None
        except Exception as e:
            print(f"[ERROR] FMP API failed for {ticker}: {e}")
            return None
    else:
        print("[ERROR] FMP API key not available")
        return None


def process_alerts_in_batches(alerts, process_function, max_workers=None):
    """
    Process alerts in batches to respect rate limits.

    Args:
        alerts: List of alerts to process
        process_function: Function to process each alert
        max_workers: Maximum number of worker threads (default: 5)

    Returns:
        List of results
    """
    if max_workers is None:
        max_workers = 5  # Default value

    batch_size = 10  # Default batch size
    results = []

    for i in range(0, len(alerts), batch_size):
        batch = alerts[i : i + batch_size]
        print(
            f"Processing batch {i//batch_size + 1}/{(len(alerts) + batch_size - 1)//batch_size} ({len(batch)} alerts)"
        )

        # Process batch with limited concurrency
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            batch_results = list(executor.map(process_function, batch))
            results.extend(batch_results)

        # Add delay between batches to respect rate limits
        if i + batch_size < len(alerts):
            time.sleep(1)  # Small delay between batches

    return results


@st.cache_data(ttl=900) if STREAMLIT_AVAILABLE else lambda func: func
def get_latest_stock_data(stock, exchange, timespan, fmp_api_key):
    """
    Fetch the latest stock data.

    Args:
        stock: Stock ticker
        exchange: Exchange name
        timespan: Timeframe
        fmp_api_key: FMP API key

    Returns:
        DataFrame with latest stock data
    """
    # Use FMP API for all data fetching
    df = grab_new_data_fmp(stock, timespan=timespan, period="1mo", fmp_api_key=fmp_api_key)
    return df

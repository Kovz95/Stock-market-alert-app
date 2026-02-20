import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import concurrent.futures
from src.services.backend import evaluate_expression_list, simplify_conditions, indicator_calculation
from src.utils.utils import supported_indicators
from src.data_access.metadata_repository import fetch_stock_metadata_map, fetch_portfolios
from src.data_access.db_config import db_config
from src.services.smart_price_fetcher import SmartPriceFetcher
from src.utils.cache_helpers import format_age, get_freshness_icon
import threading

# MUST be the first Streamlit command after imports
st.set_page_config(
    page_title="Scanner",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Market Scanner")
st.write("Scan all stocks/ETFs for symbols that meet your technical conditions")

# Initialize session state for scanner (MUST be before sidebar)
if 'scanner_conditions' not in st.session_state:
    st.session_state.scanner_conditions = []
if 'scanner_logic' not in st.session_state:
    st.session_state.scanner_logic = "AND"
if 'scanner_timeframe' not in st.session_state:
    st.session_state.scanner_timeframe = "1d"
if 'data_freshness' not in st.session_state:
    st.session_state.data_freshness = {}

# Initialize filter session states
if 'filter_portfolio' not in st.session_state:
    st.session_state.filter_portfolio = "All"
if 'filter_asset_types' not in st.session_state:
    st.session_state.filter_asset_types = ["Stock", "ETF"]
if 'filter_countries' not in st.session_state:
    st.session_state.filter_countries = []
if 'filter_exchanges' not in st.session_state:
    st.session_state.filter_exchanges = []
if 'filter_economies' not in st.session_state:
    st.session_state.filter_economies = []
if 'filter_sectors' not in st.session_state:
    st.session_state.filter_sectors = []
if 'filter_subsectors' not in st.session_state:
    st.session_state.filter_subsectors = []
if 'filter_industry_groups' not in st.session_state:
    st.session_state.filter_industry_groups = []
if 'filter_industries' not in st.session_state:
    st.session_state.filter_industries = []
if 'filter_subindustries' not in st.session_state:
    st.session_state.filter_subindustries = []

# Initialize pair scanning session state
if 'scan_mode' not in st.session_state:
    st.session_state.scan_mode = "Single Symbol"
if 'pair_symbols' not in st.session_state:
    st.session_state.pair_symbols = []

# Load stock database
@st.cache_data(ttl=300)
def load_stock_database():
    try:
        return fetch_stock_metadata_map()
    except Exception:
        st.error("Stock database not found!")
        return {}

# Load portfolios
@st.cache_data(ttl=300)
def load_portfolios():
    try:
        return fetch_portfolios()
    except Exception:
        return {}

# Create smart price fetcher instance (reused across scans)
@st.cache_resource
def get_smart_fetcher():
    """Get cached smart price fetcher instance"""
    return SmartPriceFetcher()

# Load price database with smart fetching
def get_price_data_smart(ticker, timeframe='1d', lookback_days=250, force_refresh=False):
    """
    Get price data with intelligent caching.

    Uses 3-tier fetching: Cache -> Database -> API
    Tracks freshness for UI display.

    Default lookback: 250 days (optimized for performance while maintaining indicator accuracy)
    """
    try:
        fetcher = get_smart_fetcher()
        result = fetcher.get_price_data(
            ticker=ticker,
            timeframe=timeframe,
            lookback_days=lookback_days,
            force_refresh=force_refresh
        )

        # Track freshness for UI display
        if 'data_freshness' not in st.session_state:
            st.session_state.data_freshness = {}
        st.session_state.data_freshness[ticker] = result

        return result.df
    except Exception as e:
        st.warning(f"Error fetching data for {ticker}: {e}")
        return None

# Legacy function for backward compatibility (direct database access)
def get_price_data_legacy(ticker, timeframe='1d', lookback_days=250):
    """Get price data from PostgreSQL database (legacy method)"""
    try:
        end_ts = datetime.now()
        start_ts = end_ts - timedelta(days=lookback_days)

        if timeframe == '1h':
            query = """
                SELECT datetime AS date, open, high, low, close, volume
                FROM hourly_prices
                WHERE ticker = %s AND datetime BETWEEN %s AND %s
                ORDER BY datetime ASC
            """
            params = (ticker, start_ts, end_ts)
        elif timeframe == '1d':
            query = """
                SELECT date, open, high, low, close, volume
                FROM daily_prices
                WHERE ticker = %s AND date BETWEEN %s AND %s
                ORDER BY date ASC
            """
            params = (ticker, start_ts.date(), end_ts.date())
        else:
            query = """
                SELECT week_ending AS date, open, high, low, close, volume
                FROM weekly_prices
                WHERE ticker = %s AND week_ending BETWEEN %s AND %s
                ORDER BY week_ending ASC
            """
            params = (ticker, start_ts.date(), end_ts.date())

        with db_config.connection(role="prices") as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            return None

        df = df.rename(
            columns={
                'date': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
            }
        )

        return df
    except Exception:
        return None

# Use smart fetcher by default
get_price_data = get_price_data_smart


def _count_filtered_symbols(
    stock_db,
    *,
    selected_portfolio,
    portfolio_symbols,
    asset_type_filter,
    selected_countries,
    selected_exchanges,
    selected_economies,
    selected_sectors,
    selected_subsectors,
    selected_industry_groups,
    selected_industries,
    selected_subindustries,
):
    """Count how many symbols match the current sidebar filters."""
    count = 0
    for ticker, info in stock_db.items():
        if selected_portfolio != "All" and ticker not in portfolio_symbols:
            continue
        if asset_type_filter and info.get('asset_type') not in asset_type_filter:
            continue
        if selected_countries and info.get('country') not in selected_countries:
            continue
        if selected_exchanges and info.get('exchange') not in selected_exchanges:
            continue

        if info.get('asset_type') == 'Stock':
            if selected_economies and info.get('rbics_economy') not in selected_economies:
                continue
            if selected_sectors and info.get('rbics_sector') not in selected_sectors:
                continue
            if selected_subsectors and info.get('rbics_subsector') not in selected_subsectors:
                continue
            if selected_industry_groups and info.get('rbics_industry_group') not in selected_industry_groups:
                continue
            if selected_industries and info.get('rbics_industry') not in selected_industries:
                continue
            if selected_subindustries and info.get('rbics_subindustry') not in selected_subindustries:
                continue

        count += 1
    return count

# Scan function
def _normalize_indicator_dict(ind: dict | None) -> dict | None:
    """Normalize indicator dict to be compatible with legacy apply_function."""
    if not ind:
        return None
    normed = dict(ind)
    # Alias timeperiod -> period for TA-Lib calls
    if "timeperiod" in normed and "period" not in normed:
        normed["period"] = normed["timeperiod"]
    return normed


def _extract_condition_values(df: pd.DataFrame, condition_list):
    """Return latest evaluated values for each condition for display."""
    values = []
    for cond in condition_list:
        try:
            parsed = simplify_conditions(cond)
            if not parsed:
                values.append(None)
                continue
            ind1 = _normalize_indicator_dict(parsed.get("ind1"))
            ind2 = _normalize_indicator_dict(parsed.get("ind2"))

            def _last(val):
                if val is None:
                    return None
                if hasattr(val, "iloc"):
                    try:
                        val = val.iloc[-1]
                    except Exception:
                        return None
                return val

            val1 = _last(indicator_calculation(df, ind1, None, False) if ind1 else None)
            val2 = _last(indicator_calculation(df, ind2, None, False) if ind2 else None)

            values.append(
                {
                    "expr": cond,
                    "left": val1,
                    "right": val2,
                    "op": parsed.get("comparison"),
                }
            )
        except Exception:
            values.append(None)
    return values


def apply_zscore_indicator(indicator_expr: str, use_zscore: bool, lookback: int) -> str:
    """Optionally wrap a numeric indicator expression in a z-score call."""
    if not use_zscore or not indicator_expr:
        return indicator_expr
    if any(op in indicator_expr for op in [">", "<", "=", " and ", " or ", ":"]):
        return indicator_expr
    import re
    base = indicator_expr.strip()
    match = re.match(r"(.+)\[(-?\d+)\]$", base)
    if match:
        base = match.group(1)
    try:
        lb = int(lookback)
    except Exception:
        lb = 20
    return f"zscore({base}, lookback={lb})[-1]"


def _format_condition_values(values):
    """Format condition values for table display."""
    if not values:
        return ""

    def _fmt(v):
        try:
            if v is None or pd.isna(v):
                return None
        except Exception:
            if v is None:
                return None
        try:
            v = float(v)
        except Exception:
            return str(v)
        # Compact formatting
        if abs(v) >= 1000 or (abs(v) > 0 and abs(v) < 0.01):
            return f"{v:.4g}"
        return f"{v:.4f}"

    parts = []
    for idx, item in enumerate(values, 1):
        if not item:
            continue
        left = _fmt(item.get("left"))
        right = _fmt(item.get("right"))
        op = item.get("op")
        if left is None:
            continue
        if op and right is not None:
            parts.append(f"{idx}. {left} {op} {right}")
        else:
            parts.append(f"{idx}. {left}")
    return " | ".join(parts)


def scan_symbol(ticker, conditions, combination_logic, timeframe, stock_info):
    """Scan a single symbol for condition match"""
    try:
        # Get price data
        df = get_price_data(ticker, timeframe)

        if df is None or df.empty:
            return None

        # Ensure we have enough data for indicators
        if len(df) < 50:
            return None

        # Evaluate conditions
        condition_list = [cond.strip() for cond in conditions if cond.strip()]

        if not condition_list:
            return None

        result = evaluate_expression_list(
            df=df,
            exps=condition_list,
            combination=combination_logic if combination_logic else '1'
        )

        if result:
            # Get latest price
            latest_price = df.iloc[-1]['Close']
            cond_values = _extract_condition_values(df, condition_list)

            return {
                'ticker': ticker,
                'name': stock_info.get('name', ticker),
                'isin': stock_info.get('isin', ''),
                'exchange': stock_info.get('exchange', ''),
                'country': stock_info.get('country', ''),
                'price': latest_price,
                'asset_type': stock_info.get('asset_type', 'Stock'),
                # RBICS 6-level classification (for stocks)
                'rbics_economy': stock_info.get('rbics_economy', ''),
                'rbics_sector': stock_info.get('rbics_sector', ''),
                'rbics_subsector': stock_info.get('rbics_subsector', ''),
                'rbics_industry_group': stock_info.get('rbics_industry_group', ''),
                'rbics_industry': stock_info.get('rbics_industry', ''),
                'rbics_subindustry': stock_info.get('rbics_subindustry', ''),
                # ETF fields
                'etf_issuer': stock_info.get('etf_issuer', ''),
                'etf_asset_class': stock_info.get('asset_class', ''),
                'etf_focus': stock_info.get('etf_focus', ''),
                'etf_niche': stock_info.get('etf_niche', ''),
                'expense_ratio': stock_info.get('expense_ratio', ''),
                'aum': stock_info.get('aum', ''),
                'condition_values': cond_values,
            }

        return None
    except Exception as e:
        # Silently skip errors
        return None

def scan_pair(symbol1, symbol2, conditions, combination_logic, timeframe, stock_info1, stock_info2):
    """Scan a pair of symbols for condition match"""
    try:
        # Get price data for both symbols atomically
        fetcher = get_smart_fetcher()
        result1, result2 = fetcher.get_pair_price_data(symbol1, symbol2, timeframe)

        # Track freshness for both symbols
        if 'data_freshness' not in st.session_state:
            st.session_state.data_freshness = {}
        st.session_state.data_freshness[symbol1] = result1
        st.session_state.data_freshness[symbol2] = result2

        df1 = result1.df
        df2 = result2.df

        if df1 is None or df1.empty or df2 is None or df2.empty:
            return None

        # Ensure we have enough data
        if len(df1) < 50 or len(df2) < 50:
            return None

        # Align dataframes by date
        df1 = df1.set_index('Date')
        df2 = df2.set_index('Date')

        # Find common dates
        common_dates = df1.index.intersection(df2.index)
        if len(common_dates) < 50:
            return None

        df1 = df1.loc[common_dates].reset_index()
        df2 = df2.loc[common_dates].reset_index()

        # Create combined dataframe with _2 suffix for second symbol
        df_combined = df1.copy()
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df_combined[f'{col}_2'] = df2[col].values

        # Create ratio columns (symbol1/symbol2) as the primary OHLC
        # This way conditions on Close, Open, etc. operate on the ratio
        df_combined['Open'] = df_combined['Open'] / df_combined['Open_2']
        df_combined['High'] = df_combined['High'] / df_combined['High_2']
        df_combined['Low'] = df_combined['Low'] / df_combined['Low_2']
        df_combined['Close'] = df_combined['Close'] / df_combined['Close_2']
        # Volume doesn't make sense as ratio, keep as sum
        df_combined['Volume'] = df_combined['Volume'] + df_combined['Volume_2']

        # Evaluate conditions on combined dataframe (now using ratios)
        condition_list = [cond.strip() for cond in conditions if cond.strip()]

        if not condition_list:
            return None

        result = evaluate_expression_list(
            df=df_combined,
            exps=condition_list,
            combination=combination_logic if combination_logic else '1'
        )

        if result:
            # Get latest prices (from original data before ratio calculation)
            latest_price1 = df1.iloc[-1]['Close']
            latest_price2 = df2.iloc[-1]['Close']
            ratio = latest_price1 / latest_price2
            cond_values = _extract_condition_values(df_combined, condition_list)

            return {
                'pair': f"{symbol1}/{symbol2}",
                'symbol1': symbol1,
                'symbol2': symbol2,
                'name1': stock_info1.get('name', symbol1),
                'name2': stock_info2.get('name', symbol2),
                'price1': latest_price1,
                'price2': latest_price2,
                'ratio': ratio,
                'exchange1': stock_info1.get('exchange', ''),
                'exchange2': stock_info2.get('exchange', ''),
                'country1': stock_info1.get('country', ''),
                'country2': stock_info2.get('country', ''),
                'asset_type1': stock_info1.get('asset_type', 'Stock'),
                'asset_type2': stock_info2.get('asset_type', 'Stock'),
                'condition_values': cond_values,
            }

        return None
    except Exception as e:
        # Silently skip errors
        return None

# Main scanner interface
stock_db = load_stock_database()
portfolios = load_portfolios()

if not stock_db:
    st.error("Unable to load stock database. Scanner cannot proceed.")
    st.stop()

# Sidebar - Filters
with st.sidebar:
    st.header("🔧 Scanner Configuration")

    st.divider()

    # Portfolio filter
    st.subheader("📁 Portfolio Filter")
    portfolio_options = ["All"] + [p['name'] for p in portfolios.values()]
    # Find index for default
    try:
        portfolio_index = portfolio_options.index(st.session_state.filter_portfolio)
    except ValueError:
        portfolio_index = 0
    selected_portfolio = st.selectbox(
        "Filter by Portfolio",
        portfolio_options,
        index=portfolio_index,
        help="Scan only stocks in a specific portfolio"
    )
    st.session_state.filter_portfolio = selected_portfolio

    # Get portfolio symbols if a portfolio is selected
    portfolio_symbols = set()
    if selected_portfolio != "All":
        for portfolio in portfolios.values():
            if portfolio['name'] == selected_portfolio:
                portfolio_symbols = set([stock['symbol'] for stock in portfolio['stocks']])
                break

    st.divider()

    # Asset type filter
    st.subheader("📊 Asset Type Filter")
    asset_type_filter = st.multiselect(
        "Filter by Asset Type",
        ["Stock", "ETF"],
        default=st.session_state.filter_asset_types,
        help="Select which asset types to scan"
    )
    st.session_state.filter_asset_types = asset_type_filter

    # Country filter
    st.subheader("🌍 Geographic Filter")
    all_countries = sorted(set([info.get('country', '') for info in stock_db.values() if info.get('country')]))
    selected_countries = st.multiselect(
        "Filter by Country",
        all_countries,
        default=st.session_state.filter_countries,
        help="Leave empty to scan all countries"
    )
    st.session_state.filter_countries = selected_countries

    # Exchange filter
    all_exchanges = sorted({
        (info.get('exchange') or '').strip()
        for info in stock_db.values()
        if info.get('exchange')
    })
    if selected_countries:
        filtered_exchanges = sorted(set([
            info.get('exchange', '')
            for info in stock_db.values()
            if info.get('country') in selected_countries and info.get('exchange')
        ]))
        all_exchanges = filtered_exchanges

    # Keep only valid, unique defaults to avoid Streamlit invalid-tag issues
    default_exchanges = [
        ex for ex in st.session_state.filter_exchanges
        if ex in all_exchanges
    ]

    selected_exchanges = st.multiselect(
        "Filter by Exchange",
        options=all_exchanges,
        default=default_exchanges,
        key="filter_exchanges_widget",
        help="Leave empty to scan all exchanges"
    )
    st.session_state.filter_exchanges = list(dict.fromkeys(selected_exchanges))

    # Stock Industry Filters (6-level RBICS classification)
    st.divider()
    st.subheader("🏭 Stock Industry Filters")

    # Only show stocks (not ETFs) for industry filters
    stocks_only_db = {
        ticker: info for ticker, info in stock_db.items()
        if info.get('asset_type') == 'Stock'
    }

    # Apply country and exchange filters if selected
    if selected_countries:
        stocks_only_db = {
            ticker: info for ticker, info in stocks_only_db.items()
            if info.get('country') in selected_countries
        }
    if selected_exchanges:
        stocks_only_db = {
            ticker: info for ticker, info in stocks_only_db.items()
            if info.get('exchange') in selected_exchanges
        }

    # Economy filter (Level 1)
    available_economies = sorted(set([
        info.get('rbics_economy', '')
        for info in stocks_only_db.values()
        if info.get('rbics_economy')
    ]))
    selected_economies = st.multiselect(
        "Filter by Economy:",
        available_economies,
        default=st.session_state.filter_economies,
        help="Select economies to filter stocks"
    )
    st.session_state.filter_economies = selected_economies

    # Sector filter (Level 2) - cascading from economy
    if selected_economies:
        available_sectors = sorted(set([
            info.get('rbics_sector', '')
            for info in stocks_only_db.values()
            if info.get('rbics_economy') in selected_economies and info.get('rbics_sector')
        ]))
    else:
        available_sectors = sorted(set([
            info.get('rbics_sector', '')
            for info in stocks_only_db.values()
            if info.get('rbics_sector')
        ]))

    selected_sectors = st.multiselect(
        "Filter by Sector:",
        available_sectors,
        default=st.session_state.filter_sectors,
        help="Select sectors to filter stocks"
    )
    st.session_state.filter_sectors = selected_sectors

    # Subsector filter (Level 3) - cascading from sector
    if selected_sectors:
        available_subsectors = sorted(set([
            info.get('rbics_subsector', '')
            for info in stocks_only_db.values()
            if info.get('rbics_sector') in selected_sectors and info.get('rbics_subsector')
        ]))
    else:
        available_subsectors = sorted(set([
            info.get('rbics_subsector', '')
            for info in stocks_only_db.values()
            if info.get('rbics_subsector')
        ]))

    selected_subsectors = st.multiselect(
        "Filter by Subsector:",
        available_subsectors,
        default=st.session_state.filter_subsectors,
        help="Select subsectors to filter stocks"
    )
    st.session_state.filter_subsectors = selected_subsectors

    # Industry Group filter (Level 4) - cascading from subsector
    if selected_subsectors:
        available_industry_groups = sorted(set([
            info.get('rbics_industry_group', '')
            for info in stocks_only_db.values()
            if info.get('rbics_subsector') in selected_subsectors and info.get('rbics_industry_group')
        ]))
    else:
        available_industry_groups = sorted(set([
            info.get('rbics_industry_group', '')
            for info in stocks_only_db.values()
            if info.get('rbics_industry_group')
        ]))

    selected_industry_groups = st.multiselect(
        "Filter by Industry Group:",
        available_industry_groups,
        default=st.session_state.filter_industry_groups,
        help="Select industry groups to filter stocks"
    )
    st.session_state.filter_industry_groups = selected_industry_groups

    # Industry filter (Level 5) - cascading from industry group
    if selected_industry_groups:
        available_industries = sorted(set([
            info.get('rbics_industry', '')
            for info in stocks_only_db.values()
            if info.get('rbics_industry_group') in selected_industry_groups and info.get('rbics_industry')
        ]))
    else:
        available_industries = sorted(set([
            info.get('rbics_industry', '')
            for info in stocks_only_db.values()
            if info.get('rbics_industry')
        ]))

    selected_industries = st.multiselect(
        "Filter by Industry:",
        available_industries,
        default=st.session_state.filter_industries,
        help="Select industries to filter stocks"
    )
    st.session_state.filter_industries = selected_industries

    # Subindustry filter (Level 6) - cascading from industry
    if selected_industries:
        available_subindustries = sorted(set([
            info.get('rbics_subindustry', '')
            for info in stocks_only_db.values()
            if info.get('rbics_industry') in selected_industries and info.get('rbics_subindustry')
        ]))
    else:
        available_subindustries = sorted(set([
            info.get('rbics_subindustry', '')
            for info in stocks_only_db.values()
            if info.get('rbics_subindustry')
        ]))

    selected_subindustries = st.multiselect(
        "Filter by Subindustry:",
        available_subindustries,
        default=st.session_state.filter_subindustries,
        help="Select subindustries to filter stocks"
    )
    st.session_state.filter_subindustries = selected_subindustries

    # Show how many symbols match current filters (helpful before running scan)
    filtered_symbol_count = _count_filtered_symbols(
        stock_db,
        selected_portfolio=selected_portfolio,
        portfolio_symbols=portfolio_symbols,
        asset_type_filter=asset_type_filter,
        selected_countries=selected_countries,
        selected_exchanges=selected_exchanges,
        selected_economies=selected_economies,
        selected_sectors=selected_sectors,
        selected_subsectors=selected_subsectors,
        selected_industry_groups=selected_industry_groups,
        selected_industries=selected_industries,
        selected_subindustries=selected_subindustries,
    )
    st.caption(f"🔢 Symbols matching filters: {filtered_symbol_count:,}")
    if st.session_state.scan_mode == "Pair Trading":
        st.caption(f"🔗 Pairs queued: {len(st.session_state.pair_symbols):,} (pair mode uses the list below)")

# Main content - Condition builder with dropdowns
st.header("📝 Build Scan Conditions")

# Timeframe Selection
st.markdown("### 📅 Timeframe Configuration")

timeframe = st.selectbox(
    "Select the required Timeframe",
    ["1h", "1d", "1wk"],
    index=0 if st.session_state.scanner_timeframe == "1h" else (1 if st.session_state.scanner_timeframe == "1d" else 2),
    key="timeframe_select",
    help="Hourly (1h), Daily (1d), or Weekly (1wk) timeframe for analysis"
)

# Update session state
st.session_state.scanner_timeframe = timeframe

# Display timeframe info
timeframe_display = "Hourly" if timeframe == "1h" else ("Daily" if timeframe == "1d" else "Weekly")
st.info(f"📊 **Selected Timeframe:** {timeframe_display}")

# Data Freshness Status UI
if st.session_state.get('data_freshness') and len(st.session_state.data_freshness) > 0:
    with st.expander("📊 Data Freshness Status", expanded=False):
        freshness_data = []
        for ticker, result in st.session_state.data_freshness.items():
            freshness_data.append({
                'Ticker': ticker,
                'Source': result.source.upper(),
                'Status': get_freshness_icon(result.freshness),
                'Age': format_age(result.last_update),
            })

        if freshness_data:
            # Display as dataframe
            freshness_df = pd.DataFrame(freshness_data)
            st.dataframe(freshness_df, use_container_width=True, hide_index=True)

            # Summary statistics
            total = len(freshness_data)
            from_cache = sum(1 for d in freshness_data if d['Source'] == 'CACHE')
            from_db = sum(1 for d in freshness_data if d['Source'] == 'DATABASE')
            from_api = sum(1 for d in freshness_data if d['Source'] == 'API')

            st.caption(
                f"**Summary:** {total} symbols | "
                f"Cache: {from_cache} | Database: {from_db} | API: {from_api}"
            )

    # Add force refresh button
    if st.button("🔄 Force Refresh All Data"):
        st.session_state.data_freshness = {}
        st.rerun()

st.divider()

# Pair Scanning Configuration
st.markdown("### 🔗 Pair Scanning (Optional)")

scan_mode = st.radio(
    "Scan Mode:",
    ["Single Symbol", "Pair Trading"],
    horizontal=True,
    key="scan_mode_radio"
)

st.session_state.scan_mode = scan_mode

if scan_mode == "Pair Trading":
    st.info("💡 Pair trading scans compare two symbols. Conditions like `Close[-1]` will use Symbol 1, and `Close_2[-1]` will use Symbol 2.")

    # Display loaded pairs from saved scan (if any)
    if st.session_state.pair_symbols and len(st.session_state.pair_symbols) > 0:
        st.success(f"✅ {len(st.session_state.pair_symbols):,} pairs loaded and ready to scan")
        with st.expander("View Loaded Pairs"):
            sample_pairs = st.session_state.pair_symbols[:20]
            for i, (s1, s2) in enumerate(sample_pairs, 1):
                st.write(f"{i}. {s1} / {s2}")
            if len(st.session_state.pair_symbols) > 20:
                st.write(f"... and {len(st.session_state.pair_symbols) - 20:,} more pairs")

    pair_mode_col1, pair_mode_col2 = st.columns(2)

    with pair_mode_col1:
        pair_selection_mode = st.radio(
            "How to create pairs:",
            ["Manual Selection", "Auto-Generate from Filters", "One Symbol vs Group"],
            key="pair_selection_mode"
        )

    if pair_selection_mode == "Manual Selection":
        st.markdown("#### Select Two Symbols for Pair")

        # Get all tickers for selection
        all_tickers = sorted(stock_db.keys())

        pair_col1, pair_col2 = st.columns(2)

        with pair_col1:
            symbol1 = st.selectbox(
                "Symbol 1:",
                [""] + all_tickers,
                key="pair_symbol1"
            )

        with pair_col2:
            symbol2 = st.selectbox(
                "Symbol 2:",
                [""] + all_tickers,
                key="pair_symbol2"
            )

        if symbol1 and symbol2:
            if symbol1 == symbol2:
                st.error("Please select two different symbols")
                st.session_state.pair_symbols = []
            else:
                st.session_state.pair_symbols = [(symbol1, symbol2)]
                st.success(f"✅ Pair selected: {symbol1} / {symbol2}")
        else:
            st.session_state.pair_symbols = []

    elif pair_selection_mode == "Auto-Generate from Filters":
        st.markdown("#### Auto-Generate Pairs")
        st.info("Apply filters in the sidebar, then click 'Generate Pairs' to create all unique combinations.")

        if st.button("🔄 Generate Pairs from Current Filters", key="generate_pairs"):
            # Get filtered symbols based on current sidebar filters
            filtered_for_pairs = []

            for ticker, info in stock_db.items():
                # Apply all sidebar filters
                if selected_portfolio != "All" and ticker not in portfolio_symbols:
                    continue
                if asset_type_filter and info.get('asset_type') not in asset_type_filter:
                    continue
                if selected_countries and info.get('country') not in selected_countries:
                    continue
                if selected_exchanges and info.get('exchange') not in selected_exchanges:
                    continue
                if info.get('asset_type') == 'Stock':
                    if selected_economies and info.get('rbics_economy') not in selected_economies:
                        continue
                    if selected_sectors and info.get('rbics_sector') not in selected_sectors:
                        continue
                    if selected_subsectors and info.get('rbics_subsector') not in selected_subsectors:
                        continue
                    if selected_industry_groups and info.get('rbics_industry_group') not in selected_industry_groups:
                        continue
                    if selected_industries and info.get('rbics_industry') not in selected_industries:
                        continue
                    if selected_subindustries and info.get('rbics_subindustry') not in selected_subindustries:
                        continue

                filtered_for_pairs.append(ticker)

            # Generate unique pairs (order doesn't matter)
            from itertools import combinations
            pairs = list(combinations(sorted(filtered_for_pairs), 2))

            st.session_state.pair_symbols = pairs
            st.success(f"✅ Generated {len(pairs):,} unique pairs from {len(filtered_for_pairs)} filtered symbols")

        if st.session_state.pair_symbols:
            st.info(f"📊 {len(st.session_state.pair_symbols):,} pairs ready to scan")

            # Show sample pairs
            with st.expander("View Sample Pairs"):
                sample_pairs = st.session_state.pair_symbols[:20]
                for i, (s1, s2) in enumerate(sample_pairs, 1):
                    st.write(f"{i}. {s1} / {s2}")
                if len(st.session_state.pair_symbols) > 20:
                    st.write(f"... and {len(st.session_state.pair_symbols) - 20:,} more pairs")

    else:  # One Symbol vs Group
        st.markdown("#### One Symbol vs Filtered Group")
        st.info("Select one symbol, then use sidebar filters to create pairs with that symbol vs all filtered symbols.")

        # Get all tickers for selection
        all_tickers = sorted(stock_db.keys())

        base_symbol = st.selectbox(
            "Select Base Symbol:",
            [""] + all_tickers,
            key="base_symbol_pair"
        )

        if base_symbol:
            if st.button("🔄 Generate Pairs with Filtered Group", key="generate_group_pairs"):
                # Get filtered symbols based on current sidebar filters
                filtered_group = []

                for ticker, info in stock_db.items():
                    # Skip the base symbol itself
                    if ticker == base_symbol:
                        continue

                    # Apply all sidebar filters
                    if selected_portfolio != "All" and ticker not in portfolio_symbols:
                        continue
                    if asset_type_filter and info.get('asset_type') not in asset_type_filter:
                        continue
                    if selected_countries and info.get('country') not in selected_countries:
                        continue
                    if selected_exchanges and info.get('exchange') not in selected_exchanges:
                        continue
                    if info.get('asset_type') == 'Stock':
                        if selected_economies and info.get('rbics_economy') not in selected_economies:
                            continue
                        if selected_sectors and info.get('rbics_sector') not in selected_sectors:
                            continue
                        if selected_subsectors and info.get('rbics_subsector') not in selected_subsectors:
                            continue
                        if selected_industry_groups and info.get('rbics_industry_group') not in selected_industry_groups:
                            continue
                        if selected_industries and info.get('rbics_industry') not in selected_industries:
                            continue
                        if selected_subindustries and info.get('rbics_subindustry') not in selected_subindustries:
                            continue

                    filtered_group.append(ticker)

                # Create pairs with base symbol vs each filtered symbol
                pairs = [(base_symbol, ticker) for ticker in sorted(filtered_group)]

                st.session_state.pair_symbols = pairs
                st.success(f"✅ Generated {len(pairs):,} pairs: {base_symbol} vs {len(filtered_group)} filtered symbols")

            if st.session_state.pair_symbols:
                st.info(f"📊 {len(st.session_state.pair_symbols):,} pairs ready to scan")

                # Show sample pairs
                with st.expander("View Sample Pairs"):
                    sample_pairs = st.session_state.pair_symbols[:20]
                    for i, (s1, s2) in enumerate(sample_pairs, 1):
                        st.write(f"{i}. {s1} / {s2}")
                    if len(st.session_state.pair_symbols) > 20:
                        st.write(f"... and {len(st.session_state.pair_symbols) - 20:,} more pairs")

st.divider()

# Condition Builder Interface
st.markdown("### Build Conditions Using Dropdown Menus")

col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    # Indicator category selection
    indicator_category = st.selectbox(
        "Select Indicator Category:",
        ["", "Price Data", "Moving Averages", "RSI", "MACD", "Bollinger Bands",
         "Volume", "ATR", "CCI", "Williams %R", "ROC", "EWO", "MA Z-Score", "HARSI", "OBV MACD",
         "SAR", "SuperTrend", "Trend Magic", "Ichimoku Cloud", "Kalman ROC Stoch", "Pivot S/R", "Donchian Channels", "Custom"],
        key="indicator_category"
    )

    # Indicator selection based on category
    indicator = ""
    if indicator_category == "Price Data":
        price_type = st.selectbox(
            "Select Price Type:",
            ["", "Price Comparison", "Price Data Points"],
            key="price_type"
        )
        if price_type == "Price Comparison":
            price_condition = st.selectbox(
                "Select Price Condition:",
                ["", "price_above", "price_below", "price_equals"],
                key="price_condition_type"
            )
            if price_condition:
                price_value = st.number_input(
                    "Price Value:",
                    min_value=0.01, max_value=100000.0, value=100.0, step=0.01,
                    key="price_value"
                )
                indicator = f"{price_condition}: {price_value}"
        elif price_type == "Price Data Points":
            indicator = st.selectbox(
                "Select Price Data:",
                ["", "Close[-1]", "Open[-1]", "High[-1]", "Low[-1]",
                 "Close[-2]", "Open[-2]", "High[-2]", "Low[-2]",
                 "Close[0]", "Open[0]", "High[0]", "Low[0]"],
                key="price_indicator"
            )
    elif indicator_category == "Moving Averages":
        ma_condition_type = st.selectbox(
            "Select MA Condition Type:",
            ["", "Price vs MA", "MA Crossover", "MA Value"],
            key="ma_condition_type"
        )
        if ma_condition_type == "Price vs MA":
            ma_type = st.selectbox(
                "Select MA Type:",
                ["SMA", "EMA", "HMA", "FRAMA", "KAMA"],
                key="ma_type"
            )
            ma_period = st.number_input(
                "Period:",
                min_value=1, max_value=500, value=20,
                key="ma_period"
            )
            price_ma_condition = st.selectbox(
                "Condition:",
                ["", "price_above_ma", "price_below_ma"],
                key="price_ma_condition"
            )
            if price_ma_condition:
                # Convert shorthand to actual condition
                operator = ">" if price_ma_condition == "price_above_ma" else "<"

                # Generate proper condition based on MA type
                if ma_type in ["SMA", "EMA", "HMA"]:
                    if ma_type == "HMA":
                        indicator = f"Close[-1] {operator} HMA(period={ma_period})[-1]"
                    else:
                        indicator = f"Close[-1] {operator} {ma_type}(period={ma_period}, input=Close)[-1]"
                elif ma_type == "FRAMA":
                    indicator = f"Close[-1] {operator} FRAMA(len=16, FC=1, SC=198)[-1]"
                elif ma_type == "KAMA":
                    indicator = f"Close[-1] {operator} KAMA(period={ma_period})[-1]"
        elif ma_condition_type == "MA Crossover":
            fast_period = st.number_input(
                "Fast MA Period:",
                min_value=1, max_value=200, value=10,
                key="fast_ma_period"
            )
            slow_period = st.number_input(
                "Slow MA Period:",
                min_value=2, max_value=500, value=20,
                key="slow_ma_period"
            )
            ma_type = st.selectbox(
                "MA Type:",
                ["SMA", "EMA", "HMA", "FRAMA", "KAMA"],
                key="ma_crossover_type"
            )
            crossover_direction = st.selectbox(
                "Crossover Direction:",
                ["", "ma_crossover", "ma_crossunder"],
                key="crossover_direction"
            )
            if crossover_direction:
                indicator = f"{crossover_direction}: {fast_period} > {slow_period} ({ma_type})"
        elif ma_condition_type == "MA Value":
            ma_type = st.selectbox(
                "Select MA Type:",
                ["SMA", "EMA", "HMA", "FRAMA", "KAMA"],
                key="ma_type_value"
            )
            if ma_type == "FRAMA":
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    frama_len = st.number_input(
                        "Length:",
                        min_value=1, max_value=200, value=16,
                        key="frama_len"
                    )
                with col_b:
                    frama_fc = st.number_input(
                        "Fast Constant:",
                        min_value=1, max_value=300, value=1,
                        key="frama_fc"
                    )
                with col_c:
                    frama_sc = st.number_input(
                        "Slow Constant:",
                        min_value=1, max_value=300, value=198,
                        key="frama_sc"
                    )
                indicator = f"FRAMA(df, length={frama_len}, FC={frama_fc}, SC={frama_sc})[-1]"
            elif ma_type == "KAMA":
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    kama_len = st.number_input(
                        "Length:",
                        min_value=2, max_value=200, value=21,
                        key="kama_len"
                    )
                with col_b:
                    kama_fast = st.number_input(
                        "Fast End:",
                        min_value=0.01, max_value=1.0, value=0.666, step=0.001,
                        key="kama_fast"
                    )
                with col_c:
                    kama_slow = st.number_input(
                        "Slow End:",
                        min_value=0.01, max_value=1.0, value=0.0645, step=0.0001,
                        key="kama_slow"
                    )
                indicator = f"KAMA(df, length={kama_len}, fast_end={kama_fast}, slow_end={kama_slow})[-1]"
            else:
                ma_period = st.number_input(
                    "Period:",
                    min_value=1, max_value=500, value=20,
                    key="ma_period_value"
                )
                ma_input_source = st.selectbox(
                    "Input Source:",
                    [
                        "Close (default)",
                        "Open",
                        "High",
                        "Low",
                        "EWO (Elliott Wave Oscillator)",
                        "RSI",
                        "MACD (Line)",
                        "MACD (Signal)",
                        "MACD (Histogram)"
                    ],
                    key="ma_input_source"
                )

                ma_input_expr = ""
                if ma_input_source == "Close (default)":
                    ma_input_expr = ""
                elif ma_input_source in ["Open", "High", "Low", "Close"]:
                    ma_input_expr = f"'{ma_input_source}'"
                elif ma_input_source == "EWO (Elliott Wave Oscillator)":
                    ewo_sma1 = st.number_input(
                        "EWO Fast SMA Period:",
                        min_value=1, max_value=100, value=5,
                        key="ma_input_ewo_sma1"
                    )
                    ewo_sma2 = st.number_input(
                        "EWO Slow SMA Period:",
                        min_value=1, max_value=200, value=35,
                        key="ma_input_ewo_sma2"
                    )
                    ewo_source = st.selectbox(
                        "EWO Source:",
                        ["Close", "Open", "High", "Low"],
                        index=0,
                        key="ma_input_ewo_source"
                    )
                    ewo_use_percent = st.checkbox(
                        "EWO as % of price",
                        value=True,
                        key="ma_input_ewo_use_percent"
                    )
                    ma_input_expr = f"EWO(sma1_length={ewo_sma1}, sma2_length={ewo_sma2}, source='{ewo_source}', use_percent={ewo_use_percent})"
                elif ma_input_source == "RSI":
                    ma_rsi_period = st.number_input(
                        "RSI Period (for input):",
                        min_value=2, max_value=100, value=14,
                        key="ma_input_rsi_period"
                    )
                    ma_input_expr = f"rsi({ma_rsi_period})"
                elif ma_input_source.startswith("MACD"):
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        ma_macd_fast = st.number_input("MACD Fast:", min_value=5, max_value=50, value=12, key="ma_input_macd_fast")
                    with col_m2:
                        ma_macd_slow = st.number_input("MACD Slow:", min_value=10, max_value=100, value=26, key="ma_input_macd_slow")
                    with col_m3:
                        ma_macd_signal = st.number_input("MACD Signal:", min_value=5, max_value=50, value=9, key="ma_input_macd_signal")

                    macd_type = "line"
                    if "Signal" in ma_input_source:
                        macd_type = "signal"
                    elif "Histogram" in ma_input_source:
                        macd_type = "histogram"
                    ma_input_expr = f"macd(fast_period={ma_macd_fast}, slow_period={ma_macd_slow}, signal_period={ma_macd_signal}, type='{macd_type}')"

                if ma_input_expr:
                    indicator = f"{ma_type.lower()}(period={ma_period}, input={ma_input_expr})[-1]"
                else:
                    indicator = f"{ma_type.lower()}({ma_period})[-1]"
    elif indicator_category == "MA Z-Score":
        st.markdown("**Price vs MA Z-Score**")

        ma_type = st.selectbox(
            "MA Type:",
            ["SMA", "EMA", "HMA"],
            index=0,
            key="ma_zs_ma_type",
            help="Choose SMA or EMA as the baseline moving average"
        )

        ma_period = st.number_input(
            "MA Period:",
            min_value=1, max_value=500, value=20,
            key="ma_zs_period"
        )

        spread_mean_window = st.number_input(
            "Spread Mean Window (lookback for average spread):",
            min_value=1, max_value=500, value=ma_period,
            key="ma_zs_mean_window"
        )

        spread_std_window = st.number_input(
            "Spread Std Dev Window:",
            min_value=1, max_value=500, value=spread_mean_window,
            key="ma_zs_std_window"
        )

        price_col = st.selectbox(
            "Price Column:",
            ["Close", "Open", "High", "Low"],
            index=0,
            key="ma_zs_price_col"
        )

        use_percent = st.checkbox(
            "Use percent spread ((Price - MA)/MA * 100)",
            value=True,
            key="ma_zs_use_percent",
            help="If unchecked, uses absolute price minus moving average"
        )

        operator = st.selectbox(
            "Z-Score Condition:",
            [">", ">=", "<", "<="],
            index=0,
            key="ma_zs_operator"
        )

        threshold = st.number_input(
            "Z-Score Threshold:",
            value=2.0,
            step=0.1,
            format="%.2f",
            key="ma_zs_threshold"
        )

        zs_expr = (
            f"MA_SPREAD_ZSCORE(price_col='{price_col}', ma_length={ma_period}, "
            f"spread_mean_window={spread_mean_window}, spread_std_window={spread_std_window}, "
            f"ma_type='{ma_type}', use_percent={use_percent}, output='zscore')[-1]"
        )
        indicator = f"{zs_expr} {operator} {threshold}"
    elif indicator_category == "RSI":
        rsi_condition_type = st.selectbox(
            "Select RSI Condition:",
            ["", "RSI Levels", "RSI Value"],
            key="rsi_condition_type"
        )
        if rsi_condition_type == "RSI Levels":
            rsi_period = st.number_input(
                "RSI Period:",
                min_value=2, max_value=100, value=14,
                key="rsi_period_levels"
            )
            rsi_level = st.selectbox(
                "RSI Condition:",
                ["", "rsi_oversold", "rsi_overbought", "rsi_neutral"],
                key="rsi_level"
            )
            if rsi_level == "rsi_oversold":
                oversold_level = st.number_input(
                    "Oversold Level:",
                    min_value=10, max_value=40, value=30,
                    key="scanner_rsi_oversold_level",
                    help="RSI below this level is oversold. Change this value, then click 'Add Condition' so the scan uses the new level."
                )
                indicator = f"rsi({rsi_period})[-1] < {oversold_level}"
            elif rsi_level == "rsi_overbought":
                overbought_level = st.number_input(
                    "Overbought Level:",
                    min_value=60, max_value=90, value=70,
                    key="scanner_rsi_overbought_level",
                    help="RSI above this level is overbought. Change this value, then click 'Add Condition' so the scan uses the new level."
                )
                indicator = f"rsi({rsi_period})[-1] > {overbought_level}"
            elif rsi_level == "rsi_neutral":
                indicator = f"rsi({rsi_period})[-1] > 30 and rsi({rsi_period})[-1] < 70"
        elif rsi_condition_type == "RSI Value":
            rsi_period = st.number_input(
                "RSI Period:",
                min_value=2, max_value=100, value=14,
                key="rsi_period"
            )
            indicator = f"rsi({rsi_period})[-1]"
    elif indicator_category == "MACD":
        macd_condition_type = st.selectbox(
            "Select MACD Condition:",
            ["", "MACD Crossovers", "MACD Values"],
            key="macd_condition_type"
        )
        if macd_condition_type == "MACD Crossovers":
            macd_crossover = st.selectbox(
                "Select Crossover:",
                ["", "macd_bullish_crossover", "macd_bearish_crossover"],
                key="macd_crossover"
            )
            if macd_crossover:
                col1, col2, col3 = st.columns(3)
                with col1:
                    fast_period = st.number_input("Fast Period:", min_value=5, max_value=50, value=12, key="macd_fast")
                with col2:
                    slow_period = st.number_input("Slow Period:", min_value=10, max_value=100, value=26, key="macd_slow")
                with col3:
                    signal_period = st.number_input("Signal Period:", min_value=5, max_value=50, value=9, key="macd_signal")
                if macd_crossover == "macd_bullish_crossover":
                    indicator = f"macd(fast_period = {fast_period}, slow_period = {slow_period}, signal_period = {signal_period}, type = 'line')[-1] > macd(fast_period = {fast_period}, slow_period = {slow_period}, signal_period = {signal_period}, type = 'signal')[-1]"
                else:
                    indicator = f"macd(fast_period = {fast_period}, slow_period = {slow_period}, signal_period = {signal_period}, type = 'line')[-1] < macd(fast_period = {fast_period}, slow_period = {slow_period}, signal_period = {signal_period}, type = 'signal')[-1]"
        elif macd_condition_type == "MACD Values":
            indicator = st.selectbox(
                "Select MACD Component:",
                ["", "macd[-1]", "macd_signal[-1]", "macd_histogram[-1]",
                 "macd[0]", "macd_signal[0]", "macd_histogram[0]"],
                key="macd_indicator"
            )
    elif indicator_category == "Bollinger Bands":
        bb_period = st.number_input(
            "BB Period:",
            min_value=5, max_value=100, value=20,
            key="bb_period"
        )
        bb_std = st.number_input(
            "Standard Deviations:",
            min_value=0.5, max_value=5.0, value=2.0, step=0.5,
            key="bb_std"
        )
        bb_component = st.selectbox(
            "Select Band:",
            ["upper", "middle", "lower", "width"],
            key="bb_component"
        )
        indicator = f"bb_{bb_component}({bb_period},{bb_std})[-1]"
    elif indicator_category == "Volume":
        volume_type = st.selectbox(
            "Select Volume Type:",
            ["", "Volume Conditions", "Volume Data"],
            key="volume_type"
        )
        if volume_type == "Volume Conditions":
            volume_condition = st.selectbox(
                "Select Volume Condition:",
                ["", "volume_above_average", "volume_spike", "volume_below_average"],
                key="volume_condition_type"
            )
            if volume_condition:
                if volume_condition in ["volume_above_average", "volume_spike"]:
                    volume_multiplier = st.number_input(
                        "Volume Multiplier:",
                        min_value=1.1, max_value=10.0, value=1.5, step=0.1,
                        key="volume_multiplier",
                        help="Volume must be X times the average"
                    )
                    indicator = f"{volume_condition}: {volume_multiplier}x"
                elif volume_condition == "volume_below_average":
                    volume_fraction = st.number_input(
                        "Volume Fraction:",
                        min_value=0.1, max_value=0.9, value=0.5, step=0.1,
                        key="volume_fraction",
                        help="Volume must be less than X times the average"
                    )
                    indicator = f"{volume_condition}: {volume_fraction}x"
        elif volume_type == "Volume Data":
            indicator = st.selectbox(
                "Select Volume Data:",
                ["", "volume[-1]", "volume[0]", "volume_avg(20)[-1]",
                 "volume[-1] / volume_avg(20)[-1]"],
                key="volume_indicator"
            )
    elif indicator_category == "ATR":
        atr_period = st.number_input(
            "ATR Period:",
            min_value=1, max_value=100, value=14,
            key="atr_period"
        )
        indicator = f"atr({atr_period})[-1]"
    elif indicator_category == "CCI":
        cci_period = st.number_input(
            "CCI Period:",
            min_value=5, max_value=100, value=20,
            key="cci_period"
        )
        indicator = f"cci({cci_period})[-1]"
    elif indicator_category == "Williams %R":
        willr_period = st.number_input(
            "Williams %R Period:",
            min_value=5, max_value=100, value=14,
            key="willr_period"
        )
        indicator = f"willr({willr_period})[-1]"
    elif indicator_category == "ROC":
        roc_period = st.number_input(
            "ROC Period:",
            min_value=1, max_value=100, value=12,
            key="roc_period"
        )
        indicator = f"roc({roc_period})[-1]"

    elif indicator_category == "EWO":
        st.markdown("**Elliott Wave Oscillator (EWO)**")

        ewo_sma1 = st.number_input(
            "Fast SMA Period:",
            min_value=1, max_value=100, value=5,
            key="ewo_sma1",
            help="Fast SMA period (default 5)"
        )

        ewo_sma2 = st.number_input(
            "Slow SMA Period:",
            min_value=1, max_value=200, value=35,
            key="ewo_sma2",
            help="Slow SMA period (default 35)"
        )

        ewo_source = st.selectbox(
            "Source:",
            ["Close", "Open", "High", "Low"],
            index=0,
            key="ewo_source"
        )

        ewo_use_percent = st.checkbox(
            "Show as percentage of current price",
            value=True,
            key="ewo_use_percent",
            help="If checked, shows difference as percent; otherwise absolute value"
        )

        ewo_condition_type = st.selectbox(
            "Condition Type:",
            ["", "EWO Levels", "EWO Value"],
            key="ewo_condition_type"
        )

        if ewo_condition_type == "EWO Levels":
            ewo_level_condition = st.selectbox(
                "EWO Condition:",
                ["Above Zero", "Below Zero", "Crossover Above Zero", "Crossover Below Zero"],
                key="ewo_level_condition"
            )

            if ewo_level_condition == "Above Zero":
                indicator = f"EWO(sma1_length={ewo_sma1}, sma2_length={ewo_sma2}, source='{ewo_source}', use_percent={ewo_use_percent})[-1] > 0"
            elif ewo_level_condition == "Below Zero":
                indicator = f"EWO(sma1_length={ewo_sma1}, sma2_length={ewo_sma2}, source='{ewo_source}', use_percent={ewo_use_percent})[-1] < 0"
            elif ewo_level_condition == "Crossover Above Zero":
                indicator = f"(EWO(sma1_length={ewo_sma1}, sma2_length={ewo_sma2}, source='{ewo_source}', use_percent={ewo_use_percent})[-1] > 0) & (EWO(sma1_length={ewo_sma1}, sma2_length={ewo_sma2}, source='{ewo_source}', use_percent={ewo_use_percent})[-2] <= 0)"
            elif ewo_level_condition == "Crossover Below Zero":
                indicator = f"(EWO(sma1_length={ewo_sma1}, sma2_length={ewo_sma2}, source='{ewo_source}', use_percent={ewo_use_percent})[-1] < 0) & (EWO(sma1_length={ewo_sma1}, sma2_length={ewo_sma2}, source='{ewo_source}', use_percent={ewo_use_percent})[-2] >= 0)"

        elif ewo_condition_type == "EWO Value":
            ewo_operator = st.selectbox(
                "Operator:",
                [">", "<", ">=", "<=", "=="],
                key="ewo_operator"
            )

            ewo_value = st.number_input(
                "Value:",
                value=0.0,
                step=0.1,
                format="%.2f",
                key="ewo_value"
            )

            indicator = f"EWO(sma1_length={ewo_sma1}, sma2_length={ewo_sma2}, source='{ewo_source}', use_percent={ewo_use_percent})[-1] {ewo_operator} {ewo_value}"

    elif indicator_category == "HARSI":
        harsi_type = st.selectbox(
            "Select HARSI Type:",
            ["", "HARSI_FLIP", "HARSI"],
            key="harsi_type",
            help="HARSI_FLIP returns flip signals (0=no change, 1=red to green, 2=green to red)"
        )
        if harsi_type:
            harsi_period = st.number_input(
                "HARSI Period:",
                min_value=5, max_value=100, value=14,
                key="harsi_period"
            )
            harsi_smoothing = st.number_input(
                "HARSI Smoothing:",
                min_value=1, max_value=20, value=1,
                key="harsi_smoothing"
            )
            if harsi_type == "HARSI_FLIP":
                # HARSI_FLIP returns transition codes: 0=no change, 1=green to red, 2=red to green
                indicator_options = st.selectbox(
                    "Select HARSI_FLIP Condition:",
                    ["",
                     f"HARSI_Flip(period = {harsi_period}, smoothing = {harsi_smoothing})[-1] == 1",  # Green to Red (Sell signal)
                     f"HARSI_Flip(period = {harsi_period}, smoothing = {harsi_smoothing})[-1] == 2",  # Red to Green (Buy signal)
                     f"HARSI_Flip(period = {harsi_period}, smoothing = {harsi_smoothing})[-1] > 0"],  # Any flip
                    key="harsi_flip_option",
                    help="1 = Green to Red (bearish), 2 = Red to Green (bullish), >0 = Any flip"
                )
                indicator = indicator_options if indicator_options else f"HARSI_Flip(period = {harsi_period}, smoothing = {harsi_smoothing})[-1]"
            else:
                indicator = f"HARSI(period = {harsi_period}, smoothing = {harsi_smoothing})[-1]"

    elif indicator_category == "OBV MACD":
        st.info("OBV MACD: MACD based on OBV with various moving average types")

        obv_macd_type = st.selectbox(
            "Select OBV MACD Type:",
            ["", "OBV_MACD (Value)", "OBV_MACD_SIGNAL (Direction)"],
            key="obv_macd_type",
            help="OBV_MACD returns the indicator value, OBV_MACD_SIGNAL returns trend direction (1=bullish, -1=bearish)"
        )

        if obv_macd_type:
            col1, col2 = st.columns(2)

            with col1:
                window_len = st.number_input("Window Length:", min_value=5, max_value=100, value=28, key="obv_window")
                v_len = st.number_input("Volume Smoothing:", min_value=5, max_value=50, value=14, key="obv_v_len")
                obv_len = st.number_input("OBV EMA Length:", min_value=1, max_value=50, value=1, key="obv_len")
                ma_type = st.selectbox("MA Type:",
                    ['DEMA', 'EMA', 'TEMA', 'TDEMA', 'TTEMA', 'AVG', 'THMA', 'ZLEMA', 'ZLDEMA', 'ZLTEMA', 'DZLEMA', 'TZLEMA', 'LLEMA', 'NMA'],
                    index=0, key="obv_ma_type")

            with col2:
                ma_len = st.number_input("MA Length:", min_value=1, max_value=100, value=9, key="obv_ma_len")
                slow_len = st.number_input("MACD Slow Length:", min_value=10, max_value=100, value=26, key="obv_slow")
                slope_len = st.number_input("Slope Length:", min_value=1, max_value=20, value=2, key="obv_slope")

                if "SIGNAL" in obv_macd_type:
                    p_val = st.number_input("Channel Sensitivity (p):", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="obv_p")

            # Build indicator string
            if obv_macd_type == "OBV_MACD (Value)":
                indicator = f"OBV_MACD(window_len={window_len}, v_len={v_len}, obv_len={obv_len}, ma_type='{ma_type}', ma_len={ma_len}, slow_len={slow_len}, slope_len={slope_len})[-1]"
            else:  # OBV_MACD_SIGNAL
                indicator = f"OBV_MACD_SIGNAL(window_len={window_len}, v_len={v_len}, obv_len={obv_len}, ma_type='{ma_type}', ma_len={ma_len}, slow_len={slow_len}, slope_len={slope_len}, p={p_val})[-1]"

    elif indicator_category == "SAR":
        sar_accel = st.number_input(
            "SAR Acceleration:",
            min_value=0.01, max_value=0.2, value=0.02, step=0.01,
            key="sar_accel"
        )
        sar_max = st.number_input(
            "SAR Max Acceleration:",
            min_value=0.1, max_value=0.5, value=0.2, step=0.05,
            key="sar_max"
        )
        indicator = f"sar({sar_accel},{sar_max})[-1]"
    elif indicator_category == "SuperTrend":
        st.markdown("**SuperTrend Indicator**")
        supertrend_type = st.selectbox(
            "Select Condition Type:",
            ["", "Trend Direction", "Price vs SuperTrend", "Trend Change"],
            key="supertrend_type"
        )

        col_a, col_b = st.columns(2)
        with col_a:
            st_period = st.number_input(
                "ATR Period:",
                min_value=1, max_value=100, value=10,
                key="st_period"
            )
        with col_b:
            st_mult = st.number_input(
                "ATR Multiplier:",
                min_value=0.1, max_value=10.0, value=3.0, step=0.1,
                key="st_mult"
            )

        col_c, col_d = st.columns(2)
        with col_c:
            st_use_hl2 = st.checkbox(
                "Use (H+L)/2 as Source",
                value=True,
                key="st_use_hl2",
                help="If unchecked, uses Close price"
            )
        with col_d:
            st_use_atr = st.checkbox(
                "Use Built-in ATR",
                value=True,
                key="st_use_atr",
                help="Alternative ATR calculation method"
            )

        if supertrend_type == "Trend Direction":
            trend_condition = st.selectbox(
                "Trend Condition:",
                ["", "Uptrend", "Downtrend"],
                key="st_trend_condition"
            )
            if trend_condition == "Uptrend":
                indicator = f"SUPERTREND(df, period={st_period}, multiplier={st_mult}, use_hl2={st_use_hl2}, use_builtin_atr={st_use_atr})[-1] == 1"
            elif trend_condition == "Downtrend":
                indicator = f"SUPERTREND(df, period={st_period}, multiplier={st_mult}, use_hl2={st_use_hl2}, use_builtin_atr={st_use_atr})[-1] == -1"

        elif supertrend_type == "Price vs SuperTrend":
            price_condition = st.selectbox(
                "Price Condition:",
                ["", "Price above SuperTrend", "Price below SuperTrend"],
                key="st_price_condition"
            )
            if price_condition == "Price above SuperTrend":
                indicator = f"Close[-1] > SUPERTREND_UPPER(df, period={st_period}, multiplier={st_mult}, use_hl2={st_use_hl2}, use_builtin_atr={st_use_atr})[-1]"
            elif price_condition == "Price below SuperTrend":
                indicator = f"Close[-1] < SUPERTREND_LOWER(df, period={st_period}, multiplier={st_mult}, use_hl2={st_use_hl2}, use_builtin_atr={st_use_atr})[-1]"

        elif supertrend_type == "Trend Change":
            change_condition = st.selectbox(
                "Change Direction:",
                ["", "Changed to Uptrend (Buy Signal)", "Changed to Downtrend (Sell Signal)", "Any Change"],
                key="st_change_condition"
            )
            if change_condition == "Changed to Uptrend (Buy Signal)":
                indicator = f"SUPERTREND(df, period={st_period}, multiplier={st_mult}, use_hl2={st_use_hl2}, use_builtin_atr={st_use_atr})[-1] == 1 and SUPERTREND(df, period={st_period}, multiplier={st_mult}, use_hl2={st_use_hl2}, use_builtin_atr={st_use_atr})[-2] == -1"
            elif change_condition == "Changed to Downtrend (Sell Signal)":
                indicator = f"SUPERTREND(df, period={st_period}, multiplier={st_mult}, use_hl2={st_use_hl2}, use_builtin_atr={st_use_atr})[-1] == -1 and SUPERTREND(df, period={st_period}, multiplier={st_mult}, use_hl2={st_use_hl2}, use_builtin_atr={st_use_atr})[-2] == 1"
            elif change_condition == "Any Change":
                indicator = f"SUPERTREND(df, period={st_period}, multiplier={st_mult}, use_hl2={st_use_hl2}, use_builtin_atr={st_use_atr})[-1] != SUPERTREND(df, period={st_period}, multiplier={st_mult}, use_hl2={st_use_hl2}, use_builtin_atr={st_use_atr})[-2]"
    elif indicator_category == "Trend Magic":
        st.markdown("**Trend Magic Indicator**")
        tm_type = st.selectbox(
            "Select Condition Type:",
            ["", "Trend Direction", "Price vs Trend Magic", "Trend Crossover"],
            key="tm_type"
        )

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            tm_cci_period = st.number_input(
                "CCI Period:",
                min_value=5, max_value=100, value=20,
                key="tm_cci_period"
            )
        with col_b:
            tm_atr_mult = st.number_input(
                "ATR Multiplier:",
                min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                key="tm_atr_mult"
            )
        with col_c:
            tm_atr_period = st.number_input(
                "ATR Period:",
                min_value=1, max_value=50, value=5,
                key="tm_atr_period"
            )

        if tm_type == "Trend Direction":
            trend_condition = st.selectbox(
                "Trend Condition:",
                ["", "Bullish (CCI >= 0)", "Bearish (CCI < 0)"],
                key="tm_trend_condition"
            )
            if trend_condition == "Bullish (CCI >= 0)":
                indicator = f"TREND_MAGIC_SIGNAL(df, cci_period={tm_cci_period}, atr_multiplier={tm_atr_mult}, atr_period={tm_atr_period})[-1] == 1"
            elif trend_condition == "Bearish (CCI < 0)":
                indicator = f"TREND_MAGIC_SIGNAL(df, cci_period={tm_cci_period}, atr_multiplier={tm_atr_mult}, atr_period={tm_atr_period})[-1] == -1"

        elif tm_type == "Price vs Trend Magic":
            price_condition = st.selectbox(
                "Price Condition:",
                ["", "Price above Trend Magic", "Price below Trend Magic", "Price crossed Trend Magic"],
                key="tm_price_condition"
            )
            if price_condition == "Price above Trend Magic":
                indicator = f"Close[-1] > TREND_MAGIC(df, cci_period={tm_cci_period}, atr_multiplier={tm_atr_mult}, atr_period={tm_atr_period})[-1]"
            elif price_condition == "Price below Trend Magic":
                indicator = f"Close[-1] < TREND_MAGIC(df, cci_period={tm_cci_period}, atr_multiplier={tm_atr_mult}, atr_period={tm_atr_period})[-1]"
            elif price_condition == "Price crossed Trend Magic":
                indicator = f"(Close[-1] > TREND_MAGIC(df, cci_period={tm_cci_period}, atr_multiplier={tm_atr_mult}, atr_period={tm_atr_period})[-1] and Close[-2] <= TREND_MAGIC(df, cci_period={tm_cci_period}, atr_multiplier={tm_atr_mult}, atr_period={tm_atr_period})[-2]) or (Close[-1] < TREND_MAGIC(df, cci_period={tm_cci_period}, atr_multiplier={tm_atr_mult}, atr_period={tm_atr_period})[-1] and Close[-2] >= TREND_MAGIC(df, cci_period={tm_cci_period}, atr_multiplier={tm_atr_mult}, atr_period={tm_atr_period})[-2])"

        elif tm_type == "Trend Crossover":
            cross_condition = st.selectbox(
                "Crossover Type:",
                ["", "Buy Signal (Low crosses above)", "Sell Signal (High crosses below)", "Any Cross"],
                key="tm_cross_condition"
            )
            if cross_condition == "Buy Signal (Low crosses above)":
                indicator = f"Low[-1] > TREND_MAGIC(df, cci_period={tm_cci_period}, atr_multiplier={tm_atr_mult}, atr_period={tm_atr_period})[-1] and Low[-2] <= TREND_MAGIC(df, cci_period={tm_cci_period}, atr_multiplier={tm_atr_mult}, atr_period={tm_atr_period})[-2]"
            elif cross_condition == "Sell Signal (High crosses below)":
                indicator = f"High[-1] < TREND_MAGIC(df, cci_period={tm_cci_period}, atr_multiplier={tm_atr_mult}, atr_period={tm_atr_period})[-1] and High[-2] >= TREND_MAGIC(df, cci_period={tm_cci_period}, atr_multiplier={tm_atr_mult}, atr_period={tm_atr_period})[-2]"
            elif cross_condition == "Any Cross":
                indicator = f"((Close[-1] > TREND_MAGIC(df, cci_period={tm_cci_period}, atr_multiplier={tm_atr_mult}, atr_period={tm_atr_period})[-1]) != (Close[-2] > TREND_MAGIC(df, cci_period={tm_cci_period}, atr_multiplier={tm_atr_mult}, atr_period={tm_atr_period})[-2]))"
    elif indicator_category == "Ichimoku Cloud":
        st.markdown("**Ichimoku Cloud Indicator**")
        ichimoku_type = st.selectbox(
            "Select Condition Type:",
            ["", "Price vs Cloud", "Line Crossovers", "Cloud Color", "Individual Lines", "Lagging Span"],
            key="ichimoku_type"
        )

        # Parameters
        col_a, col_b = st.columns(2)
        with col_a:
            ich_conversion = st.number_input(
                "Conversion Line Period:",
                min_value=1, max_value=100, value=9,
                key="ich_conversion"
            )
            ich_base = st.number_input(
                "Base Line Period:",
                min_value=1, max_value=100, value=26,
                key="ich_base"
            )
        with col_b:
            ich_span_b = st.number_input(
                "Span B Period:",
                min_value=1, max_value=200, value=52,
                key="ich_span_b"
            )
            ich_displacement = st.number_input(
                "Displacement:",
                min_value=1, max_value=100, value=26,
                key="ich_displacement"
            )

        if ichimoku_type == "Price vs Cloud":
            price_cloud = st.selectbox(
                "Condition:",
                ["", "Price above cloud", "Price below cloud", "Price in cloud",
                 "Price entered cloud (from above)", "Price entered cloud (from below)", "Price entered cloud (any direction)",
                 "Price crossed above cloud", "Price crossed below cloud"],
                key="ich_price_cloud"
            )
            if price_cloud == "Price above cloud":
                indicator = f"Close[-1] > ICHIMOKU_CLOUD_TOP(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1]"
            elif price_cloud == "Price below cloud":
                indicator = f"Close[-1] < ICHIMOKU_CLOUD_BOTTOM(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1]"
            elif price_cloud == "Price in cloud":
                indicator = f"(Close[-1] <= ICHIMOKU_CLOUD_TOP(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1]) and (Close[-1] >= ICHIMOKU_CLOUD_BOTTOM(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1])"
            elif price_cloud == "Price entered cloud (from above)":
                # Was above cloud, now inside cloud
                indicator = f"(Close[-2] > ICHIMOKU_CLOUD_TOP(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-2]) and (Close[-1] <= ICHIMOKU_CLOUD_TOP(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1]) and (Close[-1] >= ICHIMOKU_CLOUD_BOTTOM(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1])"
            elif price_cloud == "Price entered cloud (from below)":
                # Was below cloud, now inside cloud
                indicator = f"(Close[-2] < ICHIMOKU_CLOUD_BOTTOM(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-2]) and (Close[-1] >= ICHIMOKU_CLOUD_BOTTOM(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1]) and (Close[-1] <= ICHIMOKU_CLOUD_TOP(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1])"
            elif price_cloud == "Price entered cloud (any direction)":
                # Was outside cloud (either above or below), now inside cloud
                indicator = f"((Close[-2] > ICHIMOKU_CLOUD_TOP(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-2]) or (Close[-2] < ICHIMOKU_CLOUD_BOTTOM(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-2])) and (Close[-1] <= ICHIMOKU_CLOUD_TOP(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1]) and (Close[-1] >= ICHIMOKU_CLOUD_BOTTOM(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1])"
            elif price_cloud == "Price crossed above cloud":
                indicator = f"(Close[-1] > ICHIMOKU_CLOUD_TOP(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1]) and (Close[-2] <= ICHIMOKU_CLOUD_TOP(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-2])"
            elif price_cloud == "Price crossed below cloud":
                indicator = f"(Close[-1] < ICHIMOKU_CLOUD_BOTTOM(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1]) and (Close[-2] >= ICHIMOKU_CLOUD_BOTTOM(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-2])"

        elif ichimoku_type == "Line Crossovers":
            line_cross = st.selectbox(
                "Crossover Type:",
                ["", "Conversion crosses above Base (TK Cross Bull)", "Conversion crosses below Base (TK Cross Bear)",
                 "Price crosses above Conversion", "Price crosses below Conversion",
                 "Price crosses above Base", "Price crosses below Base"],
                key="ich_line_cross"
            )
            if line_cross == "Conversion crosses above Base (TK Cross Bull)":
                indicator = f"(ICHIMOKU_CONVERSION(periods={ich_conversion})[-1] > ICHIMOKU_BASE(periods={ich_base})[-1]) and (ICHIMOKU_CONVERSION(periods={ich_conversion})[-2] <= ICHIMOKU_BASE(periods={ich_base})[-2])"
            elif line_cross == "Conversion crosses below Base (TK Cross Bear)":
                indicator = f"(ICHIMOKU_CONVERSION(periods={ich_conversion})[-1] < ICHIMOKU_BASE(periods={ich_base})[-1]) and (ICHIMOKU_CONVERSION(periods={ich_conversion})[-2] >= ICHIMOKU_BASE(periods={ich_base})[-2])"
            elif line_cross == "Price crosses above Conversion":
                indicator = f"(Close[-1] > ICHIMOKU_CONVERSION(periods={ich_conversion})[-1]) and (Close[-2] <= ICHIMOKU_CONVERSION(periods={ich_conversion})[-2])"
            elif line_cross == "Price crosses below Conversion":
                indicator = f"(Close[-1] < ICHIMOKU_CONVERSION(periods={ich_conversion})[-1]) and (Close[-2] >= ICHIMOKU_CONVERSION(periods={ich_conversion})[-2])"
            elif line_cross == "Price crosses above Base":
                indicator = f"(Close[-1] > ICHIMOKU_BASE(periods={ich_base})[-1]) and (Close[-2] <= ICHIMOKU_BASE(periods={ich_base})[-2])"
            elif line_cross == "Price crosses below Base":
                indicator = f"(Close[-1] < ICHIMOKU_BASE(periods={ich_base})[-1]) and (Close[-2] >= ICHIMOKU_BASE(periods={ich_base})[-2])"

        elif ichimoku_type == "Cloud Color":
            cloud_color = st.selectbox(
                "Cloud Condition:",
                ["", "Bullish cloud (green)", "Bearish cloud (red)", "Cloud color changed to bullish", "Cloud color changed to bearish"],
                key="ich_cloud_color"
            )
            if cloud_color == "Bullish cloud (green)":
                indicator = f"ICHIMOKU_CLOUD_SIGNAL(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1] == 1"
            elif cloud_color == "Bearish cloud (red)":
                indicator = f"ICHIMOKU_CLOUD_SIGNAL(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1] == -1"
            elif cloud_color == "Cloud color changed to bullish":
                indicator = f"(ICHIMOKU_CLOUD_SIGNAL(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1] == 1) and (ICHIMOKU_CLOUD_SIGNAL(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-2] != 1)"
            elif cloud_color == "Cloud color changed to bearish":
                indicator = f"(ICHIMOKU_CLOUD_SIGNAL(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-1] == -1) and (ICHIMOKU_CLOUD_SIGNAL(conversion_periods={ich_conversion}, base_periods={ich_base}, span_b_periods={ich_span_b}, displacement={ich_displacement}, visual=True)[-2] != -1)"

        elif ichimoku_type == "Individual Lines":
            line_condition = st.selectbox(
                "Line Condition:",
                ["", "Price above Conversion Line", "Price below Conversion Line",
                 "Price above Base Line", "Price below Base Line",
                 "Conversion Line above Base Line", "Conversion Line below Base Line"],
                key="ich_line_condition"
            )
            if line_condition == "Price above Conversion Line":
                indicator = f"Close[-1] > ICHIMOKU_CONVERSION(periods={ich_conversion})[-1]"
            elif line_condition == "Price below Conversion Line":
                indicator = f"Close[-1] < ICHIMOKU_CONVERSION(periods={ich_conversion})[-1]"
            elif line_condition == "Price above Base Line":
                indicator = f"Close[-1] > ICHIMOKU_BASE(periods={ich_base})[-1]"
            elif line_condition == "Price below Base Line":
                indicator = f"Close[-1] < ICHIMOKU_BASE(periods={ich_base})[-1]"
            elif line_condition == "Conversion Line above Base Line":
                indicator = f"ICHIMOKU_CONVERSION(periods={ich_conversion})[-1] > ICHIMOKU_BASE(periods={ich_base})[-1]"
            elif line_condition == "Conversion Line below Base Line":
                indicator = f"ICHIMOKU_CONVERSION(periods={ich_conversion})[-1] < ICHIMOKU_BASE(periods={ich_base})[-1]"

        elif ichimoku_type == "Lagging Span":
            lagging_condition = st.selectbox(
                "Lagging Span Condition:",
                ["", "Lagging Span above price (26 periods ago)", "Lagging Span below price (26 periods ago)",
                 "Lagging Span crossed above price", "Lagging Span crossed below price"],
                key="ich_lagging"
            )
            if lagging_condition == "Lagging Span above price (26 periods ago)":
                indicator = f"ICHIMOKU_LAGGING(displacement={ich_displacement}, visual=True)[-1] > Close[-{ich_displacement}-1]"
            elif lagging_condition == "Lagging Span below price (26 periods ago)":
                indicator = f"ICHIMOKU_LAGGING(displacement={ich_displacement}, visual=True)[-1] < Close[-{ich_displacement}-1]"
            elif lagging_condition == "Lagging Span crossed above price":
                indicator = f"(ICHIMOKU_LAGGING(displacement={ich_displacement}, visual=True)[-1] > Close[-{ich_displacement}-1]) and (ICHIMOKU_LAGGING(displacement={ich_displacement}, visual=True)[-2] <= Close[-{ich_displacement}-2])"
            elif lagging_condition == "Lagging Span crossed below price":
                indicator = f"(ICHIMOKU_LAGGING(displacement={ich_displacement}, visual=True)[-1] < Close[-{ich_displacement}-1]) and (ICHIMOKU_LAGGING(displacement={ich_displacement}, visual=True)[-2] >= Close[-{ich_displacement}-2])"
    elif indicator_category == "Kalman ROC Stoch":
        st.markdown("**Kalman Smoothed ROC & Stochastic Indicator**")
        st.info("Advanced momentum oscillator combining Kalman-filtered ROC with Stochastic, smoothed by various MA types")

        krs_type = st.selectbox(
            "Select Condition Type:",
            ["", "Direction", "Crossovers", "Levels", "Value"],
            key="krs_type"
        )

        # Parameters in expandable section
        with st.expander("Parameters", expanded=True):
            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.markdown("**Smoothing**")
                krs_ma_type = st.selectbox(
                    "MA Type:",
                    ['TEMA', 'EMA', 'DEMA', 'WMA', 'VWMA', 'SMA', 'SMMA', 'HMA', 'LSMA', 'PEMA'],
                    key="krs_ma_type"
                )
                krs_smooth_len = st.number_input(
                    "Smoothing Length:",
                    min_value=1, max_value=50, value=12,
                    key="krs_smooth_len"
                )
                if krs_ma_type == 'LSMA':
                    krs_lsma_off = st.number_input(
                        "LSMA Offset:",
                        min_value=0, max_value=10, value=0,
                        key="krs_lsma_off"
                    )
                else:
                    krs_lsma_off = 0

            with col_b:
                st.markdown("**Kalman Filter**")
                krs_kal_src = st.selectbox(
                    "Source:",
                    ['Close', 'Open', 'High', 'Low', 'HL2', 'HLC3', 'OHLC4'],
                    key="krs_kal_src"
                )
                krs_sharp = st.number_input(
                    "Sharpness:",
                    min_value=1.0, max_value=100.0, value=25.0, step=1.0,
                    key="krs_sharp"
                )
                krs_k_period = st.number_input(
                    "Filter Period:",
                    min_value=0.1, max_value=10.0, value=1.0, step=0.1,
                    key="krs_k_period"
                )

            with col_c:
                st.markdown("**ROC & Stochastic**")
                krs_roc_len = st.number_input(
                    "ROC Length:",
                    min_value=1, max_value=50, value=9,
                    key="krs_roc_len"
                )
                krs_stoch_len = st.number_input(
                    "Stoch %K Length:",
                    min_value=1, max_value=50, value=14,
                    key="krs_stoch_len"
                )
                krs_smooth_k = st.number_input(
                    "%K Smooth:",
                    min_value=1, max_value=20, value=1,
                    key="krs_smooth_k"
                )
                krs_smooth_d = st.number_input(
                    "%D Smooth:",
                    min_value=1, max_value=20, value=3,
                    key="krs_smooth_d"
                )

        # Build parameter string for function calls
        krs_params = f"ma_type='{krs_ma_type}', lsma_off={krs_lsma_off}, smooth_len={krs_smooth_len}, kal_src='{krs_kal_src}', sharp={krs_sharp}, k_period={krs_k_period}, roc_len={krs_roc_len}, stoch_len={krs_stoch_len}, smooth_k={krs_smooth_k}, smooth_d={krs_smooth_d}"

        if krs_type == "Direction":
            direction = st.selectbox(
                "Direction:",
                ["", "Uptrend (White)", "Downtrend (Blue)"],
                key="krs_direction"
            )
            if direction == "Uptrend (White)":
                indicator = f"KALMAN_ROC_STOCH_SIGNAL(df, {krs_params})[-1] == 1"
            elif direction == "Downtrend (Blue)":
                indicator = f"KALMAN_ROC_STOCH_SIGNAL(df, {krs_params})[-1] == -1"

        elif krs_type == "Crossovers":
            crossover = st.selectbox(
                "Crossover Type:",
                ["", "Bullish Crossover (Buy)", "Bearish Crossunder (Sell)", "Any Cross"],
                key="krs_crossover"
            )
            if crossover == "Bullish Crossover (Buy)":
                indicator = f"KALMAN_ROC_STOCH_CROSSOVER(df, {krs_params})[-1] == 1"
            elif crossover == "Bearish Crossunder (Sell)":
                indicator = f"KALMAN_ROC_STOCH_CROSSOVER(df, {krs_params})[-1] == -1"
            elif crossover == "Any Cross":
                indicator = f"KALMAN_ROC_STOCH_CROSSOVER(df, {krs_params})[-1] != 0"

        elif krs_type == "Levels":
            level_condition = st.selectbox(
                "Level Condition:",
                ["", "Above 60 (Overbought)", "Below 10 (Oversold)",
                 "Above 50", "Below 50", "Between 10 and 60"],
                key="krs_level"
            )
            if level_condition == "Above 60 (Overbought)":
                indicator = f"KALMAN_ROC_STOCH(df, {krs_params})[-1] > 60"
            elif level_condition == "Below 10 (Oversold)":
                indicator = f"KALMAN_ROC_STOCH(df, {krs_params})[-1] < 10"
            elif level_condition == "Above 50":
                indicator = f"KALMAN_ROC_STOCH(df, {krs_params})[-1] > 50"
            elif level_condition == "Below 50":
                indicator = f"KALMAN_ROC_STOCH(df, {krs_params})[-1] < 50"
            elif level_condition == "Between 10 and 60":
                indicator = f"(KALMAN_ROC_STOCH(df, {krs_params})[-1] > 10) and (KALMAN_ROC_STOCH(df, {krs_params})[-1] < 60)"

        elif krs_type == "Value":
            st.info("Get the raw indicator value to compare with custom thresholds")
            indicator = f"KALMAN_ROC_STOCH(df, {krs_params})[-1]"
    elif indicator_category == "Pivot S/R":
        st.info("Pivot Support/Resistance detector finds horizontal levels from pivot highs/lows.\n" +
               "It checks for both proximity and crossovers automatically.\n" +
               "Levels are stored and maintained on a rolling 3-year basis.")

        pivot_signal = st.selectbox(
            "Alert When:",
            ["",
             "Any Signal (Proximity or Crossover)",
             "Near Support",
             "Near Resistance",
             "Near Any Level",
             "Bullish Crossover",
             "Bearish Crossover",
             "Any Crossover",
             "Broke Strong Support (3+ touches)",
             "Broke Strong Resistance (3+ touches)",
             "Custom Signal Value"],
            key="pivot_signal",
            help="Signal values: 3=Broke strong resist, 2=Bullish cross, 1=Near support, -1=Near resist, -2=Bearish cross, -3=Broke strong support"
        )

        # Parameters
        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Pivot Detection Parameters:**")
            left_bars = st.number_input(
                "Left Bars:",
                min_value=2, max_value=120, value=5,
                key="pivot_left_bars",
                help="Number of bars to the left of pivot for validation"
            )
            right_bars = st.number_input(
                "Right Bars:",
                min_value=2, max_value=120, value=5,
                key="pivot_right_bars",
                help="Number of bars to the right of pivot for validation"
            )

        with col_b:
            st.write("**Alert & Level Management:**")
            proximity_threshold = st.number_input(
                "Proximity Threshold (%):",
                min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                key="pivot_proximity",
                help="Alert when price is within X% of a level"
            )
            buffer_percent = st.number_input(
                "Buffer Between Levels (%):",
                min_value=0.1, max_value=5.0, value=0.5, step=0.1,
                key="pivot_buffer",
                help="Minimum % separation between levels to avoid clustering"
            )

        # Generate condition based on selection
        detector_func = f"PIVOT_SR(df, ticker, left_bars={left_bars}, right_bars={right_bars}, proximity_threshold={proximity_threshold}, buffer_percent={buffer_percent})"

        if pivot_signal == "Any Signal (Proximity or Crossover)":
            indicator = f"{detector_func} != 0"
        elif pivot_signal == "Near Support":
            indicator = f"{detector_func} == 1"
        elif pivot_signal == "Near Resistance":
            indicator = f"{detector_func} == -1"
        elif pivot_signal == "Near Any Level":
            indicator = f"abs({detector_func}) == 1"
        elif pivot_signal == "Bullish Crossover":
            indicator = f"{detector_func} == 2"
        elif pivot_signal == "Bearish Crossover":
            indicator = f"{detector_func} == -2"
        elif pivot_signal == "Any Crossover":
            indicator = f"abs({detector_func}) == 2"
        elif pivot_signal == "Broke Strong Support (3+ touches)":
            indicator = f"{detector_func} == -3"
        elif pivot_signal == "Broke Strong Resistance (3+ touches)":
            indicator = f"{detector_func} == 3"
        elif pivot_signal == "Custom Signal Value":
            custom_value = st.number_input(
                "Signal Value to Match:",
                min_value=-3, max_value=3, value=0,
                key="pivot_custom_value",
                help="-3=Broke strong support, -2=Bearish cross, -1=Near resist, 0=None, 1=Near support, 2=Bullish cross, 3=Broke strong resist"
            )
            if custom_value != 0:
                indicator = f"{detector_func} == {custom_value}"
            else:
                indicator = ""
        else:
            indicator = ""
    elif indicator_category == "Donchian Channels":
        donchian_type = st.selectbox(
            "Select Donchian Type:",
            ["", "Channel Lines", "Channel Breakout", "Channel Position", "Channel Width"],
            key="donchian_type"
        )

        if donchian_type:
            donchian_length = st.number_input(
                "Donchian Period:",
                min_value=5, max_value=200, value=20,
                key="donchian_length",
                help="Number of periods for highest high and lowest low"
            )
            donchian_offset = st.number_input(
                "Offset:",
                min_value=-50, max_value=50, value=0,
                key="donchian_offset",
                help="Positive offset looks back, negative offset looks forward"
            )

            if donchian_type == "Channel Lines":
                line_type = st.selectbox(
                    "Select Line:",
                    ["", "Upper Band", "Lower Band", "Basis (Middle)", "Price vs Upper", "Price vs Lower", "Price vs Basis"],
                    key="donchian_line"
                )
                if line_type == "Upper Band":
                    indicator = f"DONCHIAN_UPPER(df, {donchian_length}, {donchian_offset})[-1]"
                elif line_type == "Lower Band":
                    indicator = f"DONCHIAN_LOWER(df, {donchian_length}, {donchian_offset})[-1]"
                elif line_type == "Basis (Middle)":
                    indicator = f"DONCHIAN_BASIS(df, {donchian_length}, {donchian_offset})[-1]"
                elif line_type == "Price vs Upper":
                    indicator = f"Close[-1] > DONCHIAN_UPPER(df, {donchian_length}, {donchian_offset})[-1]"
                elif line_type == "Price vs Lower":
                    indicator = f"Close[-1] < DONCHIAN_LOWER(df, {donchian_length}, {donchian_offset})[-1]"
                elif line_type == "Price vs Basis":
                    indicator = f"Close[-1] > DONCHIAN_BASIS(df, {donchian_length}, {donchian_offset})[-1]"

            elif donchian_type == "Channel Breakout":
                breakout_type = st.selectbox(
                    "Select Breakout:",
                    ["", "Upper Band Breakout", "Lower Band Breakout", "Basis Cross Up", "Basis Cross Down"],
                    key="donchian_breakout"
                )
                if breakout_type == "Upper Band Breakout":
                    indicator = f"(Close[-1] > DONCHIAN_UPPER(df, {donchian_length}, {donchian_offset})[-1]) and (Close[-2] <= DONCHIAN_UPPER(df, {donchian_length}, {donchian_offset})[-2])"
                elif breakout_type == "Lower Band Breakout":
                    indicator = f"(Close[-1] < DONCHIAN_LOWER(df, {donchian_length}, {donchian_offset})[-1]) and (Close[-2] >= DONCHIAN_LOWER(df, {donchian_length}, {donchian_offset})[-2])"
                elif breakout_type == "Basis Cross Up":
                    indicator = f"(Close[-1] > DONCHIAN_BASIS(df, {donchian_length}, {donchian_offset})[-1]) and (Close[-2] <= DONCHIAN_BASIS(df, {donchian_length}, {donchian_offset})[-2])"
                elif breakout_type == "Basis Cross Down":
                    indicator = f"(Close[-1] < DONCHIAN_BASIS(df, {donchian_length}, {donchian_offset})[-1]) and (Close[-2] >= DONCHIAN_BASIS(df, {donchian_length}, {donchian_offset})[-2])"

            elif donchian_type == "Channel Position":
                position_type = st.selectbox(
                    "Select Position Check:",
                    ["", "Position Value", "Near Upper Band", "Near Lower Band", "Near Middle"],
                    key="donchian_position"
                )
                if position_type == "Position Value":
                    indicator = f"DONCHIAN_POSITION(df, {donchian_length}, {donchian_offset})[-1]"
                    st.info("Position value: 0 = at lower band, 0.5 = at middle, 1 = at upper band")
                elif position_type == "Near Upper Band":
                    indicator = f"DONCHIAN_POSITION(df, {donchian_length}, {donchian_offset})[-1] > 0.8"
                elif position_type == "Near Lower Band":
                    indicator = f"DONCHIAN_POSITION(df, {donchian_length}, {donchian_offset})[-1] < 0.2"
                elif position_type == "Near Middle":
                    indicator = f"(DONCHIAN_POSITION(df, {donchian_length}, {donchian_offset})[-1] > 0.4) and (DONCHIAN_POSITION(df, {donchian_length}, {donchian_offset})[-1] < 0.6)"

            elif donchian_type == "Channel Width":
                width_type = st.selectbox(
                    "Select Width Check:",
                    ["", "Width Value", "Width Expanding", "Width Contracting"],
                    key="donchian_width"
                )
                if width_type == "Width Value":
                    indicator = f"DONCHIAN_WIDTH(df, {donchian_length}, {donchian_offset})[-1]"
                elif width_type == "Width Expanding":
                    indicator = f"DONCHIAN_WIDTH(df, {donchian_length}, {donchian_offset})[-1] > DONCHIAN_WIDTH(df, {donchian_length}, {donchian_offset})[-2]"
                elif width_type == "Width Contracting":
                    indicator = f"DONCHIAN_WIDTH(df, {donchian_length}, {donchian_offset})[-1] < DONCHIAN_WIDTH(df, {donchian_length}, {donchian_offset})[-2]"
    elif indicator_category == "Custom":
        st.info("Enter any custom condition in the text field below. Examples:\n" +
               "• Close[-1] > 150.00\n" +
               "• rsi(14)[-1] < 30\n" +
               "• sma(20)[-1] > sma(50)[-1]\n" +
               "• HARSI_Flip(period = 14, smoothing = 1)[-1] == 2")
        indicator = ""

    # Optional z-score transform for numeric indicator values
    if indicator:
        use_zscore = st.checkbox(
            "Transform to Z-score (rolling)",
            value=False,
            key="zscore_option_scanner",
            help="Use the z-score of this indicator over a rolling lookback window"
        )
        if use_zscore:
            zscore_lookback = st.number_input(
                "Z-score Lookback:",
                min_value=5, max_value=500, value=20,
                key="zscore_lookback_scanner"
            )
        else:
            zscore_lookback = 20
        indicator = apply_zscore_indicator(indicator, use_zscore, zscore_lookback)

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        if st.button("➕ Add Condition", key="add_condition"):
            if indicator and indicator.strip():
                st.session_state.scanner_conditions.append(indicator.strip())
            st.success(f"✅ Added: {indicator.strip()}")
            st.rerun()

with col3:
    st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
    if st.button("🗑️ Clear All", key="clear_all"):
        st.session_state.scanner_conditions = []
        st.rerun()

# Save/Load Scan Configuration
st.divider()
st.markdown("### 💾 Save/Load Scan Configuration")

col_save1, col_save2, col_save3 = st.columns([2, 1, 1])

with col_save1:
    # Load saved scans
    try:
        with open('saved_scans.json', 'r') as f:
            saved_scans = json.load(f)
    except:
        saved_scans = {}

    if saved_scans:
        selected_scan = st.selectbox(
            "Load Saved Scan:",
            [""] + list(saved_scans.keys()),
            key="load_scan_selector"
        )

        if selected_scan and st.button("📂 Load", key="load_scan_btn"):
            scan_config = saved_scans[selected_scan]
            st.session_state.scanner_conditions = scan_config.get('conditions', [])
            st.session_state.scanner_logic = scan_config.get('logic', 'AND')
            st.session_state.scanner_timeframe = scan_config.get('timeframe', '1d')
            st.session_state.scan_mode = scan_config.get('scan_mode', 'Single Symbol')

            # Load filters into session state
            if 'filters' in scan_config:
                filters = scan_config['filters']
                st.session_state.filter_portfolio = filters.get('portfolio', 'All')
                st.session_state.filter_asset_types = filters.get('asset_types', ['Stock', 'ETF'])
                st.session_state.filter_countries = filters.get('countries', [])
                st.session_state.filter_exchanges = filters.get('exchanges', [])
                st.session_state.filter_economies = filters.get('economies', [])
                st.session_state.filter_sectors = filters.get('sectors', [])
                st.session_state.filter_subsectors = filters.get('subsectors', [])
                st.session_state.filter_industry_groups = filters.get('industry_groups', [])
                st.session_state.filter_industries = filters.get('industries', [])
                st.session_state.filter_subindustries = filters.get('subindustries', [])

            # Load pair trading settings if available
            if 'pair_trading' in scan_config:
                pair_config = scan_config['pair_trading']
                st.session_state.pair_symbols = pair_config.get('pair_symbols', [])

            st.success(f"✅ Loaded scan: {selected_scan}")
            st.rerun()
    else:
        st.info("No saved scans yet")

with col_save2:
    scan_name = st.text_input("Scan Name:", key="save_scan_name", placeholder="Enter name...")

with col_save3:
    st.markdown("<br>", unsafe_allow_html=True)  # Spacing
    if st.button("💾 Save", key="save_scan_btn"):
        if not scan_name:
            st.error("Please enter a scan name")
        elif not st.session_state.scanner_conditions:
            st.error("No conditions to save")
        else:
            # Prepare scan configuration
            scan_config = {
                'conditions': st.session_state.scanner_conditions,
                'logic': st.session_state.scanner_logic,
                'timeframe': st.session_state.scanner_timeframe,
                'scan_mode': st.session_state.scan_mode,
                'filters': {
                    'portfolio': selected_portfolio,
                    'asset_types': asset_type_filter,
                    'countries': selected_countries,
                    'exchanges': selected_exchanges,
                    'economies': selected_economies if 'selected_economies' in locals() else [],
                    'sectors': selected_sectors if 'selected_sectors' in locals() else [],
                    'subsectors': selected_subsectors if 'selected_subsectors' in locals() else [],
                    'industry_groups': selected_industry_groups if 'selected_industry_groups' in locals() else [],
                    'industries': selected_industries if 'selected_industries' in locals() else [],
                    'subindustries': selected_subindustries if 'selected_subindustries' in locals() else []
                },
                'saved_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            # Save pair trading settings if in pair mode
            if st.session_state.scan_mode == "Pair Trading":
                scan_config['pair_trading'] = {
                    'pair_symbols': st.session_state.pair_symbols,
                    'num_pairs': len(st.session_state.pair_symbols)
                }

            # Save to file
            saved_scans[scan_name] = scan_config
            with open('saved_scans.json', 'w') as f:
                json.dump(saved_scans, f, indent=2)

            st.success(f"✅ Saved scan: {scan_name}")
            st.rerun()

# Manage saved scans
if saved_scans:
    with st.expander("🗂️ Manage Saved Scans"):
        for scan_name, scan_config in saved_scans.items():
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{scan_name}**")
                st.caption(f"Saved: {scan_config.get('saved_at', 'Unknown')} | "
                          f"{len(scan_config.get('conditions', []))} conditions | "
                          f"Timeframe: {scan_config.get('timeframe', 'N/A')}")
            with col2:
                if st.button("👁️ View", key=f"view_{scan_name}"):
                    st.info(f"**Conditions:**\n" + "\n".join([f"{i+1}. {c}" for i, c in enumerate(scan_config.get('conditions', []))]))
                    st.info(f"**Logic:** {scan_config.get('logic', 'N/A')}")
            with col3:
                if st.button("🗑️ Delete", key=f"delete_{scan_name}"):
                    del saved_scans[scan_name]
                    with open('saved_scans.json', 'w') as f:
                        json.dump(saved_scans, f, indent=2)
                    st.success(f"Deleted: {scan_name}")
                    st.rerun()

# Display current conditions
st.divider()
st.markdown("### 📋 Current Scan Conditions")

if st.session_state.scanner_conditions:
    for i, cond in enumerate(st.session_state.scanner_conditions):
        col1, col2 = st.columns([5, 1])
        with col1:
            st.code(f"{i+1}. {cond}", language="python")
        with col2:
            if st.button("✖", key=f"remove_{i}", help="Remove this condition"):
                st.session_state.scanner_conditions.pop(i)
                st.rerun()

    # Combination logic for multiple conditions
    if len(st.session_state.scanner_conditions) > 1:
        st.markdown("**Combination Logic**")
        logic_mode = st.selectbox(
            "How to combine conditions?",
            ["All (AND)", "Any (OR)", "Custom expression", "Pick two"],
            index=0 if st.session_state.scanner_logic.upper() == "AND" else 1
            if st.session_state.scanner_logic.upper() == "OR"
            else 3 if st.session_state.scanner_logic.strip().upper() not in {"AND", "OR", "1"}
            else 0,
        )

        total_conds = len(st.session_state.scanner_conditions)
        cond_options = [str(i + 1) for i in range(total_conds)]

        if logic_mode == "All (AND)":
            st.session_state.scanner_logic = "AND"
            st.caption("All conditions must be true.")
        elif logic_mode == "Any (OR)":
            st.session_state.scanner_logic = "OR"
            st.caption("Any condition can be true.")
        elif logic_mode == "Pick two":
            c1 = st.selectbox("First condition", cond_options, key="logic_c1")
            c2 = st.selectbox("Second condition", cond_options, key="logic_c2")
            op = st.selectbox("Operator", ["AND", "OR"], key="logic_op")
            # Simple builder: (1 AND 2)
            st.session_state.scanner_logic = f"({c1} {op} {c2})"
            st.caption(f"Using: {st.session_state.scanner_logic}")
        else:
            # Custom free-form for advanced expressions
            st.session_state.scanner_logic = st.text_input(
                "Custom logic (e.g. '(1 AND 2) OR 3')",
                value=st.session_state.scanner_logic,
                help="Use condition numbers, AND/OR, and parentheses",
            )
    else:
        st.session_state.scanner_logic = "1"
else:
    st.info("No conditions added yet. Use the dropdowns above to build conditions.")

# Use conditions for scanning
conditions = st.session_state.scanner_conditions
combination_logic = st.session_state.scanner_logic

# Scan button
st.divider()

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    scan_button = st.button("🚀 Run Scan", type="primary", use_container_width=True)

if scan_button:
    if not conditions:
        st.error("Please add at least one condition!")
    elif st.session_state.scan_mode == "Pair Trading" and not st.session_state.pair_symbols:
        st.error("Please select or generate pairs first!")
    else:
        # Get timeframe from session state
        scan_timeframe = st.session_state.scanner_timeframe
        scan_timeframe_display = "Daily" if scan_timeframe == "1d" else "Weekly"

        # Check if we're doing pair scanning
        if st.session_state.scan_mode == "Pair Trading":
            # Pair scanning
            pairs_to_scan = st.session_state.pair_symbols
            total_symbols = len(pairs_to_scan)

            st.info(f"🔍 Scanning {total_symbols:,} pairs on {scan_timeframe_display} timeframe...")

            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()

            matches = []
            processed = 0
            lock = threading.Lock()

            def update_progress():
                with lock:
                    progress = processed / total_symbols
                    progress_bar.progress(progress)
                    status_text.text(f"Scanned {processed:,}/{total_symbols:,} ({progress*100:.1f}%) - Found {len(matches)} matches")

            # Parallel scanning
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                future_to_pair = {
                    executor.submit(
                        scan_pair,
                        s1, s2,
                        conditions,
                        combination_logic,
                        scan_timeframe,
                        stock_db.get(s1, {}),
                        stock_db.get(s2, {})
                    ): (s1, s2)
                    for s1, s2 in pairs_to_scan
                }

                for future in concurrent.futures.as_completed(future_to_pair):
                    result = future.result()

                    with lock:
                        processed += 1
                        if result:
                            matches.append(result)

                    # Update progress every 50 symbols (or on completion) to reduce Streamlit reruns
                    if processed % 50 == 0 or processed == total_symbols:
                        update_progress()

            # Final update
            progress_bar.progress(1.0)
            status_text.text(f"✅ Scan complete! Found {len(matches)} matches")

            # Store matches in session state
            st.session_state.scan_matches = matches
            st.session_state.scan_type = "pair"

        else:
            # Single symbol scanning
            symbols_to_scan = []

            for ticker, info in stock_db.items():
                # Portfolio filter
                if selected_portfolio != "All" and ticker not in portfolio_symbols:
                    continue

                # Asset type filter
                if asset_type_filter and info.get('asset_type') not in asset_type_filter:
                    continue

                # Geographic filters
                if selected_countries and info.get('country') not in selected_countries:
                    continue
                if selected_exchanges and info.get('exchange') not in selected_exchanges:
                    continue

                # RBICS industry filters (6 levels) - only apply to stocks
                if info.get('asset_type') == 'Stock':
                    if selected_economies and info.get('rbics_economy') not in selected_economies:
                        continue
                    if selected_sectors and info.get('rbics_sector') not in selected_sectors:
                        continue
                    if selected_subsectors and info.get('rbics_subsector') not in selected_subsectors:
                        continue
                    if selected_industry_groups and info.get('rbics_industry_group') not in selected_industry_groups:
                        continue
                    if selected_industries and info.get('rbics_industry') not in selected_industries:
                        continue
                    if selected_subindustries and info.get('rbics_subindustry') not in selected_subindustries:
                        continue

                symbols_to_scan.append((ticker, info))

            total_symbols = len(symbols_to_scan)

            if total_symbols == 0:
                st.warning("No symbols match your filter criteria!")
            else:
                st.info(f"🔍 Scanning {total_symbols:,} symbols on {scan_timeframe_display} timeframe...")

                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()

                matches = []
                processed = 0
                lock = threading.Lock()

                def update_progress():
                    with lock:
                        progress = processed / total_symbols
                        progress_bar.progress(progress)
                        status_text.text(f"Scanned {processed:,}/{total_symbols:,} ({progress*100:.1f}%) - Found {len(matches)} matches")

                # Parallel scanning
                with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                    future_to_symbol = {
                        executor.submit(scan_symbol, ticker, conditions, combination_logic, scan_timeframe, info): (ticker, info)
                        for ticker, info in symbols_to_scan
                    }

                    for future in concurrent.futures.as_completed(future_to_symbol):
                        result = future.result()

                        with lock:
                            processed += 1
                            if result:
                                matches.append(result)

                        # Update progress every 50 symbols (or on completion) to reduce Streamlit reruns
                        if processed % 50 == 0 or processed == total_symbols:
                            update_progress()

                # Final update
                progress_bar.progress(1.0)
                status_text.text(f"✅ Scan complete! Found {len(matches)} matches")

                # Store matches in session state for persistence across reruns
                st.session_state.scan_matches = matches
                st.session_state.scan_type = "single"

                # Display results
                st.divider()
                st.header("📊 Scan Results")

# Display results section - OUTSIDE the scan button block
if 'scan_matches' in st.session_state and st.session_state.scan_matches:
    matches = st.session_state.scan_matches
    scan_type = st.session_state.get('scan_type', 'single')

    if scan_type == "pair":
        st.success(f"Found {len(matches)} pairs matching your conditions!")
    else:
        st.success(f"Found {len(matches)} symbols matching your conditions!")

    results_df = pd.DataFrame(matches)

    # Sort based on scan type
    if scan_type == "pair":
        results_df = results_df.sort_values(['pair'])
    else:
        results_df = results_df.sort_values(['asset_type', 'ticker'])

    # Add formatted condition values column for display
    if 'condition_values' in results_df.columns:
        results_df['Condition Values'] = results_df['condition_values'].apply(_format_condition_values)

    # Search and filter controls (only for single symbol scans)
    if scan_type == "single":
        st.markdown("### 🔍 Filter Results")

        # Search box
        search_term = st.text_input(
            "Search by Ticker or Name:",
            key="results_search",
            placeholder="Enter ticker or company name..."
        )

        # Filter controls in columns
        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:
            # Filter by asset type
            asset_types_in_results = results_df['asset_type'].unique().tolist()
            selected_asset_filter = st.multiselect(
                "Filter by Type:",
                asset_types_in_results,
                default=asset_types_in_results,
                key="results_asset_filter"
            )

        with filter_col2:
            # Filter by country
            countries_in_results = sorted(results_df['country'].dropna().unique().tolist())
            if countries_in_results:
                selected_country_filter = st.multiselect(
                    "Filter by Country:",
                    countries_in_results,
                    key="results_country_filter"
                )
            else:
                selected_country_filter = []

        with filter_col3:
            # Filter by exchange
            exchanges_in_results = sorted(results_df['exchange'].dropna().unique().tolist())
            if exchanges_in_results:
                selected_exchange_filter = st.multiselect(
                    "Filter by Exchange:",
                    exchanges_in_results,
                    key="results_exchange_filter"
                )
            else:
                selected_exchange_filter = []

        # RBICS Classification Filters (6 levels) - only show if stocks present
        if (results_df['asset_type'] == 'Stock').any():
            st.markdown("#### 🏭 RBICS Industry Classification Filters")
            rbics_col1, rbics_col2, rbics_col3 = st.columns(3)

            with rbics_col1:
                # Economy filter
                economies = sorted(results_df['rbics_economy'].dropna().unique().tolist())
                if economies:
                    selected_economies_filter = st.multiselect(
                        "Economy:",
                        economies,
                        key="results_economy_filter"
                    )
                else:
                    selected_economies_filter = []

                # Subsector filter
                subsectors = sorted(results_df['rbics_subsector'].dropna().unique().tolist())
                if subsectors:
                    selected_subsectors_filter = st.multiselect(
                        "Subsector:",
                        subsectors,
                        key="results_subsector_filter"
                    )
                else:
                    selected_subsectors_filter = []

            with rbics_col2:
                # Sector filter
                sectors = sorted(results_df['rbics_sector'].dropna().unique().tolist())
                if sectors:
                    selected_sectors_filter = st.multiselect(
                        "Sector:",
                        sectors,
                        key="results_sector_filter"
                    )
                else:
                    selected_sectors_filter = []

                # Industry Group filter
                industry_groups = sorted(results_df['rbics_industry_group'].dropna().unique().tolist())
                if industry_groups:
                    selected_industry_groups_filter = st.multiselect(
                        "Industry Group:",
                        industry_groups,
                        key="results_industry_group_filter"
                    )
                else:
                    selected_industry_groups_filter = []

            with rbics_col3:
                # Industry filter
                industries = sorted(results_df['rbics_industry'].dropna().unique().tolist())
                if industries:
                    selected_industries_filter = st.multiselect(
                        "Industry:",
                        industries,
                        key="results_industry_filter"
                    )
                else:
                    selected_industries_filter = []

                # Subindustry filter
                subindustries = sorted(results_df['rbics_subindustry'].dropna().unique().tolist())
                if subindustries:
                    selected_subindustries_filter = st.multiselect(
                        "Subindustry:",
                        subindustries,
                        key="results_subindustry_filter"
                    )
                else:
                    selected_subindustries_filter = []
        else:
            selected_economies_filter = []
            selected_sectors_filter = []
            selected_subsectors_filter = []
            selected_industry_groups_filter = []
            selected_industries_filter = []
            selected_subindustries_filter = []

        # Apply all filters
        filtered_results = results_df.copy()

        # Search filter
        if search_term:
            mask = (
                filtered_results['ticker'].str.contains(search_term, case=False, na=False) |
                filtered_results['name'].str.contains(search_term, case=False, na=False)
            )
            filtered_results = filtered_results[mask]

        # Asset type filter
        if selected_asset_filter:
            filtered_results = filtered_results[filtered_results['asset_type'].isin(selected_asset_filter)]

        # Country filter
        if selected_country_filter:
            filtered_results = filtered_results[filtered_results['country'].isin(selected_country_filter)]

        # Exchange filter
        if selected_exchange_filter:
            filtered_results = filtered_results[filtered_results['exchange'].isin(selected_exchange_filter)]

        # RBICS filters
        if selected_economies_filter:
            filtered_results = filtered_results[filtered_results['rbics_economy'].isin(selected_economies_filter)]
        if selected_sectors_filter:
            filtered_results = filtered_results[filtered_results['rbics_sector'].isin(selected_sectors_filter)]
        if selected_subsectors_filter:
            filtered_results = filtered_results[filtered_results['rbics_subsector'].isin(selected_subsectors_filter)]
        if selected_industry_groups_filter:
            filtered_results = filtered_results[filtered_results['rbics_industry_group'].isin(selected_industry_groups_filter)]
        if selected_industries_filter:
            filtered_results = filtered_results[filtered_results['rbics_industry'].isin(selected_industries_filter)]
        if selected_subindustries_filter:
            filtered_results = filtered_results[filtered_results['rbics_subindustry'].isin(selected_subindustries_filter)]

    else:
        # Pair scan results - simple search only
        st.markdown("### 🔍 Filter Pairs")
        search_term = st.text_input(
            "Search by Pair (e.g., NNN/VNQ):",
            key="results_pair_search",
            placeholder="Enter symbols..."
        )

        filtered_results = results_df.copy()
        if search_term:
            mask = filtered_results['pair'].str.contains(search_term, case=False, na=False)
            filtered_results = filtered_results[mask]

    # Show filtered count
    st.info(f"📊 Showing **{len(filtered_results)}** of **{len(results_df)}** total results")

    st.markdown("---")

    # Display table
    try:
        st.dataframe(
            filtered_results,
            use_container_width=True,
            hide_index=True,
            key=f"results_table_{len(filtered_results)}",  # Unique key
            column_config={
            'ticker': st.column_config.TextColumn('Ticker', width='small'),
            'name': st.column_config.TextColumn('Name', width='medium'),
            'isin': st.column_config.TextColumn('ISIN', width='small'),
            'asset_type': st.column_config.TextColumn('Type', width='small'),
            'exchange': st.column_config.TextColumn('Exchange', width='small'),
            'country': st.column_config.TextColumn('Country', width='small'),
            'price': st.column_config.NumberColumn('Price', format='$%.2f', width='small'),
            # RBICS columns
            'rbics_economy': st.column_config.TextColumn('Economy', width='medium'),
            'rbics_sector': st.column_config.TextColumn('Sector', width='medium'),
            'rbics_subsector': st.column_config.TextColumn('Subsector', width='medium'),
            'rbics_industry_group': st.column_config.TextColumn('Industry Group', width='medium'),
            'rbics_industry': st.column_config.TextColumn('Industry', width='medium'),
            'rbics_subindustry': st.column_config.TextColumn('Subindustry', width='medium'),
            # ETF columns
            'etf_issuer': st.column_config.TextColumn('ETF Issuer', width='small'),
            'etf_asset_class': st.column_config.TextColumn('Asset Class', width='medium'),
            'etf_focus': st.column_config.TextColumn('ETF Focus', width='medium'),
            'etf_niche': st.column_config.TextColumn('ETF Niche', width='medium'),
            'expense_ratio': st.column_config.NumberColumn('Expense Ratio', format='%.2f%%', width='small'),
            'aum': st.column_config.NumberColumn('AUM', format='$%.0f', width='small'),
            'Condition Values': st.column_config.TextColumn('Condition Values', width='large'),
        }
        )
    except Exception as e:
        st.error(f"Error displaying table: {str(e)}")
        st.write("Filtered results preview:")
        st.write(filtered_results.head())

    # Download button
    st.markdown("")
    csv = filtered_results.to_csv(index=False)
    st.download_button(
        label="📥 Download Results as CSV",
        data=csv,
        file_name=f"scan_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

    # Statistics
    st.divider()
    st.subheader("📈 Scan Statistics")

    if scan_type == "single":
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Matches", len(results_df))

        with col2:
            stock_count = len(results_df[results_df['asset_type'] == 'Stock'])
            st.metric("Stocks", stock_count)

        with col3:
            etf_count = len(results_df[results_df['asset_type'] == 'ETF'])
            st.metric("ETFs", etf_count)

        with col4:
            unique_countries = results_df['country'].nunique()
            st.metric("Countries", unique_countries)
    else:
        # Pair scan statistics
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Pairs", len(results_df))

        with col2:
            # Count unique symbols across all pairs
            unique_symbols = set()
            for pair in results_df['pair']:
                sym1, sym2 = pair.split('/')
                unique_symbols.add(sym1)
                unique_symbols.add(sym2)
            st.metric("Unique Symbols", len(unique_symbols))
else:
    st.warning("No symbols matched your conditions. Try adjusting your criteria or filters.")

# Footer
st.divider()
st.caption("💡 **Tip:** Use the Scanner to find trading opportunities across your entire universe of stocks and ETFs!")
st.caption("⚡ **Performance:** Scans utilize parallel processing for fast results across thousands of symbols")

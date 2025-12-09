import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utils for market data
from utils import load_market_data
from data_access.alert_repository import (
    delete_alert as repo_delete_alert,
    list_alerts as repo_list_alerts,
    refresh_alert_cache,
)

# Load alerts
alert_data = repo_list_alerts()

# Cache market data loading for better performance
@st.cache_data(ttl=60)  # Cache for 1 minute to ensure fresh data
def load_cached_market_data():
    return load_market_data()

# Load market data with caching
market_data = load_cached_market_data()

# Initialize session state for pagination and selections
if 'current_page' not in st.session_state:
    st.session_state.current_page = 1

if 'selected_alert_ids' not in st.session_state:
    st.session_state.selected_alert_ids = []

# Define pagination constant
ITEMS_PER_PAGE = 10

st.title("🗑️ Batch Delete Stock Alerts")

st.markdown("---")

if not alert_data:
    st.warning("No active alerts to delete.")
    st.stop()

# Convert alerts to DataFrame for easier filtering
df_alerts = pd.DataFrame(alert_data)

# Data structure info (minimal)
if len(df_alerts) > 0:
    st.sidebar.markdown(f"**📊 Data:** {len(df_alerts)} alerts loaded")
else:
    st.sidebar.error("⚠️ No alerts loaded")

# Add helper columns for filtering
try:
    def extract_condition_text(conditions):
        if isinstance(conditions, list):
            # Extract the 'conditions' string from each dict in the list
            condition_strings = []
            for c in conditions:
                if isinstance(c, dict) and 'conditions' in c:
                    condition_strings.append(c['conditions'])
            return ' | '.join(condition_strings)
        return ''

    df_alerts['condition_text'] = df_alerts['conditions'].apply(extract_condition_text)
except Exception as e:
    st.sidebar.error(f"Error processing conditions: {e}")
    df_alerts['condition_text'] = ''

try:
    df_alerts['last_triggered_date'] = pd.to_datetime(df_alerts['last_triggered'], errors='coerce')
except Exception as e:
    st.sidebar.error(f"Error processing last_triggered: {e}")
    df_alerts['last_triggered_date'] = pd.NaT

# Add asset class column using the same logic as Home.py
def determine_asset_class_for_delete_page(ticker, ratio=None):
    """Determine asset class using the same logic as Home.py"""
    if ratio == 'Yes':
        return "Cross-Industry Ratio"
    elif 'etf' in ticker.lower():
        return "ETFs"
    # Only stocks and ETFs are supported
    else:
        return "Stocks"

try:
    df_alerts['asset_class'] = df_alerts.apply(
        lambda row: determine_asset_class_for_delete_page(
            row.get('ticker', ''), 
            row.get('ratio', None)
        ), axis=1
    )
except Exception as e:
    st.sidebar.error(f"Error processing asset class: {e}")
    df_alerts['asset_class'] = 'Unknown'

# Initialize filter variables
selected_asset_type = "All"
selected_economies = []
selected_sectors = []
selected_subsectors = []
selected_industry_groups = []
selected_industries = []
selected_subindustries = []
selected_countries = []
selected_exchanges = []
selected_timeframes = []
selected_issuers = []
selected_asset_classes = []
selected_focuses = []
selected_niches = []

# Sidebar for filtering
st.sidebar.header("🔍 Filter Options")

# Asset Type Selection
st.sidebar.subheader("📈 Asset Type")
selected_asset_type = st.sidebar.selectbox(
    "Filter by asset type:",
    ["All", "Stocks", "ETFs", "Futures"],
    help="Filter alerts by asset type"
)

# Country Filter - available for all asset types
st.sidebar.subheader("🌍 Location Filters")
if 'Country' in market_data.columns:
    available_countries = sorted(market_data['Country'].dropna().unique())
    selected_countries = st.sidebar.multiselect(
        "Filter by Country:",
        available_countries,
        default=[],
        help="Select countries to filter alerts"
    )

# Exchange Filter - cascading from country
if 'Exchange' in market_data.columns:
    if selected_countries:
        available_exchanges = sorted(market_data[market_data['Country'].isin(selected_countries)]['Exchange'].dropna().unique())
    else:
        available_exchanges = sorted(market_data['Exchange'].dropna().unique())
    
    selected_exchanges = st.sidebar.multiselect(
        "Filter by Exchange:",
        available_exchanges,
        default=[],
        help="Select exchanges to filter alerts"
    )

# Timeframe Filter
st.sidebar.subheader("⏱️ Timeframe")
timeframe_options = ["All", "Weekly", "Daily", "Hourly", "15 minutes", "5 minutes", "1 minute"]
selected_timeframes = st.sidebar.multiselect(
    "Filter by Timeframe:",
    timeframe_options[1:],  # Exclude "All" from multiselect
    default=[],
    help="Select timeframes to filter alerts"
)

# Filter by company/ticker
company_filter = st.sidebar.text_input("Search by Company/Ticker:", "").strip().lower()

# Stock Industry Filters
if selected_asset_type in ["All", "Stocks"] and 'RBICS_Sector' in market_data.columns:
    st.sidebar.subheader("🏭 Industry Filters")
    
    # Only show stocks (not ETFs) for stock filters - use Asset_Type field
    if 'Asset_Type' in market_data.columns:
        stocks_only_data = market_data[market_data['Asset_Type'] == 'Stock']
    else:
        # Fallback to old method if Asset_Type not available
        stocks_only_data = market_data[~market_data['ETF_Issuer'].notna()] if 'ETF_Issuer' in market_data.columns else market_data
    
    # Apply country and exchange filters if selected
    if selected_countries and 'Country' in stocks_only_data.columns:
        stocks_only_data = stocks_only_data[stocks_only_data['Country'].isin(selected_countries)]
    if selected_exchanges and 'Exchange' in stocks_only_data.columns:
        stocks_only_data = stocks_only_data[stocks_only_data['Exchange'].isin(selected_exchanges)]
    
    # Economy filter
    available_economies = stocks_only_data['RBICS_Economy'].dropna().unique() if 'RBICS_Economy' in stocks_only_data.columns else []
    selected_economies = st.sidebar.multiselect(
        "Filter by Economy:",
        sorted(available_economies),
        default=[],
        help="Select economies to filter alerts"
    )
    
    # Sector filter - cascading from economy
    if selected_economies and 'RBICS_Economy' in stocks_only_data.columns:
        available_sectors = stocks_only_data[stocks_only_data['RBICS_Economy'].isin(selected_economies)]['RBICS_Sector'].dropna().unique()
    else:
        available_sectors = stocks_only_data['RBICS_Sector'].dropna().unique() if 'RBICS_Sector' in stocks_only_data.columns else []
    
    selected_sectors = st.sidebar.multiselect(
        "Filter by Sector:",
        sorted(available_sectors),
        default=[],
        help="Select sectors to filter alerts"
    )
    
    # Subsector filter - cascading from sector
    if selected_sectors and 'RBICS_Sector' in stocks_only_data.columns:
        available_subsectors = stocks_only_data[stocks_only_data['RBICS_Sector'].isin(selected_sectors)]['RBICS_Subsector'].dropna().unique() if 'RBICS_Subsector' in stocks_only_data.columns else []
    else:
        available_subsectors = stocks_only_data['RBICS_Subsector'].dropna().unique() if 'RBICS_Subsector' in stocks_only_data.columns else []
    
    selected_subsectors = st.sidebar.multiselect(
        "Filter by Subsector:",
        sorted(available_subsectors),
        default=[],
        help="Select subsectors to filter alerts"
    )
    
    # Industry Group filter - cascading from subsector
    if selected_subsectors and 'RBICS_Subsector' in stocks_only_data.columns:
        available_industry_groups = stocks_only_data[stocks_only_data['RBICS_Subsector'].isin(selected_subsectors)]['RBICS_Industry_Group'].dropna().unique() if 'RBICS_Industry_Group' in stocks_only_data.columns else []
    else:
        available_industry_groups = stocks_only_data['RBICS_Industry_Group'].dropna().unique() if 'RBICS_Industry_Group' in stocks_only_data.columns else []
    
    selected_industry_groups = st.sidebar.multiselect(
        "Filter by Industry Group:",
        sorted(available_industry_groups),
        default=[],
        help="Select industry groups to filter alerts"
    )
    
    # Industry filter - cascading from industry group
    if selected_industry_groups and 'RBICS_Industry_Group' in stocks_only_data.columns:
        available_industries = stocks_only_data[stocks_only_data['RBICS_Industry_Group'].isin(selected_industry_groups)]['RBICS_Industry'].dropna().unique() if 'RBICS_Industry' in stocks_only_data.columns else []
    else:
        available_industries = stocks_only_data['RBICS_Industry'].dropna().unique() if 'RBICS_Industry' in stocks_only_data.columns else []
    
    selected_industries = st.sidebar.multiselect(
        "Filter by Industry:",
        sorted(available_industries),
        default=[],
        help="Select industries to filter alerts"
    )
    
    # Subindustry filter - cascading from industry
    if selected_industries and 'RBICS_Industry' in stocks_only_data.columns:
        available_subindustries = stocks_only_data[stocks_only_data['RBICS_Industry'].isin(selected_industries)]['RBICS_Subindustry'].dropna().unique() if 'RBICS_Subindustry' in stocks_only_data.columns else []
    else:
        available_subindustries = stocks_only_data['RBICS_Subindustry'].dropna().unique() if 'RBICS_Subindustry' in stocks_only_data.columns else []
    
    selected_subindustries = st.sidebar.multiselect(
        "Filter by Subindustry:",
        sorted(available_subindustries),
        default=[],
        help="Select subindustries to filter alerts"
    )

# ETF Filters
if selected_asset_type in ["All", "ETFs"] and 'ETF_Issuer' in market_data.columns:
    st.sidebar.subheader("📊 ETF Filters")
    
    # Only show ETFs for ETF filters - use Asset_Type field
    if 'Asset_Type' in market_data.columns:
        etfs_only_data = market_data[market_data['Asset_Type'] == 'ETF']
    else:
        # Fallback to old method if Asset_Type not available
        etfs_only_data = market_data[market_data['ETF_Issuer'].notna()]
    
    # Apply country and exchange filters if selected
    if selected_countries and 'Country' in etfs_only_data.columns:
        etfs_only_data = etfs_only_data[etfs_only_data['Country'].isin(selected_countries)]
    if selected_exchanges and 'Exchange' in etfs_only_data.columns:
        etfs_only_data = etfs_only_data[etfs_only_data['Exchange'].isin(selected_exchanges)]
    
    # ETF Issuer filter
    available_issuers = sorted(etfs_only_data['ETF_Issuer'].dropna().unique())
    selected_issuers = st.sidebar.multiselect(
        "Filter by ETF Issuer:",
        available_issuers,
        default=[],
        help="Select ETF issuers to filter alerts"
    )
    
    # Asset Class filter - cascading from issuer
    if 'Asset_Class' in etfs_only_data.columns:
        if selected_issuers:
            available_asset_classes = sorted(etfs_only_data[etfs_only_data['ETF_Issuer'].isin(selected_issuers)]['Asset_Class'].dropna().unique())
        else:
            available_asset_classes = sorted(etfs_only_data['Asset_Class'].dropna().unique())
        
        selected_asset_classes = st.sidebar.multiselect(
            "Filter by Asset Class:",
            available_asset_classes,
            default=[],
            help="Select asset classes to filter alerts"
        )
    
    # ETF Focus filter - cascading from asset class
    if 'ETF_Focus' in etfs_only_data.columns:
        if selected_asset_classes and 'Asset_Class' in etfs_only_data.columns:
            available_focuses = sorted(etfs_only_data[etfs_only_data['Asset_Class'].isin(selected_asset_classes)]['ETF_Focus'].dropna().unique())
        else:
            available_focuses = sorted(etfs_only_data['ETF_Focus'].dropna().unique())
        
        selected_focuses = st.sidebar.multiselect(
            "Filter by ETF Focus:",
            available_focuses,
            default=[],
            help="Select ETF focus areas to filter alerts"
        )
    
    # ETF Niche filter - cascading from focus
    if 'ETF_Niche' in etfs_only_data.columns:
        if selected_focuses and 'ETF_Focus' in etfs_only_data.columns:
            available_niches = sorted(etfs_only_data[etfs_only_data['ETF_Focus'].isin(selected_focuses)]['ETF_Niche'].dropna().unique())
        else:
            available_niches = sorted(etfs_only_data['ETF_Niche'].dropna().unique())
        
        selected_niches = st.sidebar.multiselect(
            "Filter by ETF Niche:",
            available_niches,
            default=[],
            help="Select ETF niche categories to filter alerts"
        )

# Filter by condition type with dropdown
condition_options = [
    "All Conditions",
    "RSI",
    "MACD", 
    "SMA",
    "EMA",
    "HMA",
    "BB (Bollinger Bands)",
    "ATR",
    "CCI",
    "ROC",
    "Williams %R",
    "Close",
    "Open",
    "High",
    "Low",
    "HARSI",
    "SROCST",
    "Breakout",
    "Slope",
    "Price"
]

selected_condition = st.sidebar.selectbox("Filter by Condition Type:", condition_options)

# Convert dropdown selection to search term
if selected_condition == "All Conditions":
    condition_filter = ""
else:
    condition_filter = selected_condition.lower()

# Additional custom condition search
custom_condition_filter = st.sidebar.text_input("Custom Condition Search (e.g., 'HARSI', 'RSI', 'SMA'):", "").strip().lower()

# Combine both filters
if custom_condition_filter:
    condition_filter = custom_condition_filter

# Help section for condition syntax
with st.sidebar.expander("ℹ️ Condition Search Help"):
    st.markdown("""
    **Quick Search by Condition Type:**
    - `rsi` - Find all RSI conditions (any period)
    - `macd` - Find all MACD conditions
    - `sma` - Find all Simple Moving Average conditions
    - `ema` - Find all Exponential Moving Average conditions
    - `hma` - Find all Hull Moving Average conditions
    - `bb` - Find all Bollinger Bands conditions
    - `atr` - Find all ATR conditions
    - `cci` - Find all CCI conditions
    - `roc` - Find all Rate of Change conditions
    - `williams` - Find all Williams %R conditions
    - `harsi` - Find all HARSI conditions
    - `srocst` - Find all SROCST conditions
    - `breakout` - Find all breakout conditions
    - `slope` - Find all slope conditions
    - `close`, `open`, `high`, `low` - Find price conditions
    
    **Examples:**
    - Type `rsi` to find all alerts with RSI conditions
    - Type `sma` to find all alerts with SMA conditions
    - Type `harsi` to find all alerts with HARSI conditions
    """)

# Filter by last triggered
triggered_filter = st.sidebar.selectbox(
    "Filter by Last Triggered:",
    ["All", "Never Triggered", "Triggered Today", "Triggered This Week", "Triggered This Month", "Triggered This Year"]
)

# Filter by alert name
name_filter = st.sidebar.text_input("Search by Alert Name:", "").strip().lower()

st.sidebar.markdown("---")

# Create a cache for market data lookups
@st.cache_data(ttl=60)
def create_ticker_lookup_cache(market_data_df):
    """Create a dictionary for fast ticker lookups"""
    ticker_cache = {}
    for idx, row in market_data_df.iterrows():
        symbol = row['Symbol'].upper() if pd.notna(row['Symbol']) else ''
        if symbol:
            ticker_cache[symbol] = row.to_dict()
            # Also add without suffix
            base_symbol = symbol.split('-')[0]
            if base_symbol not in ticker_cache:
                ticker_cache[base_symbol] = row.to_dict()
    return ticker_cache

# Create the cache
ticker_cache = create_ticker_lookup_cache(market_data) if not market_data.empty else {}

# Apply filters function
def apply_filters(alert):
    """Apply all selected filters to an alert"""
    # Get ticker for this alert (handle ratio alerts)
    ticker = alert.get('ticker', '')
    if not ticker and 'ticker1' in alert:
        ticker = alert.get('ticker1', '')

    # Look up stock/ETF data for this ticker using cache
    stock_data = None
    if ticker:
        ticker_upper = ticker.upper()
        # Try exact match first
        if ticker_upper in ticker_cache:
            stock_data = ticker_cache[ticker_upper]
        elif f"{ticker_upper}-US" in ticker_cache:
            stock_data = ticker_cache[f"{ticker_upper}-US"]
        else:
            # Try base ticker
            base_ticker = ticker_upper.split('-')[0]
            if base_ticker in ticker_cache:
                stock_data = ticker_cache[base_ticker]
    
    # Asset type filter
    if selected_asset_type != "All":
        # Check if this is a futures alert
        from backend_futures_flexible import is_futures_symbol
        ticker = alert.get('ticker', '')

        if selected_asset_type == "Futures":
            # Only show futures alerts
            if not is_futures_symbol(ticker):
                return False
        else:
            # For Stocks and ETFs, exclude futures
            if is_futures_symbol(ticker):
                return False

            alert_type = "Stocks"  # Default
            if stock_data is not None:
                if stock_data.get('ETF_Issuer'):
                    alert_type = "ETFs"

            if alert_type != selected_asset_type:
                return False

    # Country filter
    if selected_countries and stock_data is not None:
        country = stock_data.get('Country')
        if not country or country not in selected_countries:
            return False
    
    # Exchange filter
    if selected_exchanges:
        if alert.get('exchange', '') not in selected_exchanges:
            return False
    
    # Timeframe filter
    if selected_timeframes:
        alert_timeframe = alert.get('timeframe', 'Daily')
        if alert_timeframe not in selected_timeframes:
            return False
    
    # Industry filters for stocks
    if stock_data is not None and not stock_data.get('ETF_Issuer'):
        # Economy filter
        if selected_economies:
            economy = stock_data.get('RBICS_Economy')
            if not economy or economy not in selected_economies:
                return False
        
        # Sector filter
        if selected_sectors:
            sector = stock_data.get('RBICS_Sector')
            if not sector or sector not in selected_sectors:
                return False
        
        # Subsector filter
        if selected_subsectors:
            subsector = stock_data.get('RBICS_Subsector')
            if not subsector or subsector not in selected_subsectors:
                return False
        
        # Industry Group filter
        if selected_industry_groups:
            industry_group = stock_data.get('RBICS_Industry_Group')
            if not industry_group or industry_group not in selected_industry_groups:
                return False
        
        # Industry filter
        if selected_industries:
            industry = stock_data.get('RBICS_Industry')
            if not industry or industry not in selected_industries:
                return False
        
        # Subindustry filter
        if selected_subindustries:
            subindustry = stock_data.get('RBICS_Subindustry')
            if not subindustry or subindustry not in selected_subindustries:
                return False
    
    # ETF filters
    if stock_data is not None and stock_data.get('ETF_Issuer'):
        # ETF Issuer filter
        if selected_issuers:
            issuer = stock_data.get('ETF_Issuer')
            if not issuer or issuer not in selected_issuers:
                return False
        
        # Asset Class filter
        if selected_asset_classes:
            asset_class = stock_data.get('Asset_Class')
            if not asset_class or asset_class not in selected_asset_classes:
                return False
        
        # ETF Focus filter
        if selected_focuses:
            focus = stock_data.get('ETF_Focus')
            if not focus or focus not in selected_focuses:
                return False
        
        # ETF Niche filter
        if selected_niches:
            niche = stock_data.get('ETF_Niche')
            if not niche or niche not in selected_niches:
                return False
    
    return True

# Apply filters
filtered_df = df_alerts.copy()

# Apply text search filters
if company_filter:
    try:
        filtered_df = filtered_df[
            filtered_df['stock_name'].str.lower().str.contains(company_filter, na=False) |
            filtered_df['ticker'].str.lower().str.contains(company_filter, na=False)
        ]
    except Exception as e:
        st.sidebar.error(f"Error filtering by company: {e}")

if condition_filter:
    try:
        filtered_df = filtered_df[filtered_df['condition_text'].str.lower().str.contains(condition_filter, na=False)]
    except Exception as e:
        st.sidebar.error(f"Error filtering by condition: {e}")
        # Continue without this filter

if name_filter:
    try:
        filtered_df = filtered_df[filtered_df['name'].str.lower().str.contains(name_filter, na=False)]
    except Exception as e:
        st.sidebar.error(f"Error filtering by name: {e}")
        # Continue without this filter

# Apply the main filters - only if filters are actually active
if any([selected_asset_type != "All", selected_countries, selected_exchanges, selected_timeframes,
        selected_economies, selected_sectors, selected_subsectors, selected_industry_groups,
        selected_industries, selected_subindustries, selected_issuers, selected_asset_classes,
        selected_focuses, selected_niches]):
    filtered_list = []
    for idx, row in filtered_df.iterrows():
        if apply_filters(row.to_dict()):
            filtered_list.append(row)

    if filtered_list:
        filtered_df = pd.DataFrame(filtered_list)
    else:
        filtered_df = pd.DataFrame()  # Empty dataframe if no matches
# If no filters active, keep filtered_df as is

# Apply triggered filter
if triggered_filter != "All":
    today = pd.Timestamp.now().normalize()
    if triggered_filter == "Never Triggered":
        filtered_df = filtered_df[filtered_df['last_triggered'].isna()]
    elif triggered_filter == "Triggered Today":
        filtered_df = filtered_df[
            (filtered_df['last_triggered_date'].dt.date == today.date()) &
            (filtered_df['last_triggered_date'].notna())
        ]
    elif triggered_filter == "Triggered This Week":
        week_ago = today - pd.Timedelta(days=7)
        filtered_df = filtered_df[
            (filtered_df['last_triggered_date'] >= week_ago) &
            (filtered_df['last_triggered_date'].notna())
        ]
    elif triggered_filter == "Triggered This Month":
        month_ago = today - pd.Timedelta(days=30)
        filtered_df = filtered_df[
            (filtered_df['last_triggered_date'] >= month_ago) &
            (filtered_df['last_triggered_date'].notna())
        ]
    elif triggered_filter == "Triggered This Year":
        year_ago = today - pd.Timedelta(days=365)
        filtered_df = filtered_df[
            (filtered_df['last_triggered_date'] >= year_ago) &
            (filtered_df['last_triggered_date'].notna())
        ]



# Display filter summary
st.sidebar.markdown("---")
st.sidebar.markdown(f"**📊 Filter Results:**")
st.sidebar.markdown(f"Total Alerts: {len(df_alerts)}")
st.sidebar.markdown(f"Filtered Alerts: {len(filtered_df)}")

# Show active filters count in sidebar
active_filters = []
if selected_asset_type != "All":
    active_filters.append(f"Asset: {selected_asset_type}")
if selected_countries:
    active_filters.append(f"{len(selected_countries)} countries")
if selected_exchanges:
    active_filters.append(f"{len(selected_exchanges)} exchanges")
if selected_timeframes:
    active_filters.append(f"{len(selected_timeframes)} timeframes")
if selected_economies:
    active_filters.append(f"{len(selected_economies)} economies")
if selected_sectors:
    active_filters.append(f"{len(selected_sectors)} sectors")
if selected_subsectors:
    active_filters.append(f"{len(selected_subsectors)} subsectors")
if selected_industry_groups:
    active_filters.append(f"{len(selected_industry_groups)} groups")
if selected_industries:
    active_filters.append(f"{len(selected_industries)} industries")
if selected_subindustries:
    active_filters.append(f"{len(selected_subindustries)} subindustries")
if selected_issuers:
    active_filters.append(f"{len(selected_issuers)} issuers")
if selected_asset_classes:
    active_filters.append(f"{len(selected_asset_classes)} asset classes")
if selected_focuses:
    active_filters.append(f"{len(selected_focuses)} focuses")
if selected_niches:
    active_filters.append(f"{len(selected_niches)} niches")

if active_filters:
    st.sidebar.info("🔍 **Active Filters:** " + ", ".join(active_filters[:3]) + ("..." if len(active_filters) > 3 else ""))



# Calculate total pages for pagination
total_pages = (len(filtered_df) + ITEMS_PER_PAGE - 1) // ITEMS_PER_PAGE



# Display filter info and clear button
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    total_original_alerts = len(df_alerts)
    if len(filtered_df) != total_original_alerts:
        st.info(f"📊 Showing {len(filtered_df)} of {total_original_alerts} alerts (filtered)")
    else:
        st.info(f"📊 Showing all {len(filtered_df)} alerts")

with col2:
    if active_filters:
        st.info("🔍 **Active:** " + ", ".join(active_filters[:3]) + ("..." if len(active_filters) > 3 else ""))
    else:
        st.info("🔍 **No filters applied** - showing all alerts")

with col3:
    # Clear filters button
    if any([selected_asset_type != "All", selected_countries, selected_exchanges, selected_timeframes,
            selected_economies, selected_sectors, selected_subsectors, selected_industry_groups,
            selected_industries, selected_subindustries, selected_issuers, selected_asset_classes,
            selected_focuses, selected_niches, company_filter, condition_filter, name_filter]):
        if st.button("🗑️ Clear Filters", use_container_width=True):
            st.rerun()

# SECTION 1: FILTERED ALERTS DISPLAY (like home page)
st.subheader("📋 Filtered Alerts")

if len(filtered_df) == 0:
    st.info("No alerts match your current filters. Try adjusting the filter criteria.")
else:
    # Pagination controls
    if total_pages > 1:
        col_page1, col_page2, col_page3 = st.columns([1, 2, 1])
        
        with col_page1:
            if st.button("⬅️ Previous", disabled=(st.session_state.get('current_page', 1) <= 1)):
                st.session_state.current_page = max(1, st.session_state.get('current_page', 1) - 1)
                st.rerun()
        
        with col_page2:
            current_page = st.selectbox(
                "Page:",
                range(1, total_pages + 1),
                index=st.session_state.get('current_page', 1) - 1,
                key="page_selector"
            )
            if current_page != st.session_state.get('current_page', 1):
                st.session_state.current_page = current_page
                st.rerun()
        
        with col_page3:
            if st.button("Next ➡️", disabled=(st.session_state.get('current_page', 1) >= total_pages)):
                st.session_state.current_page = min(total_pages, st.session_state.get('current_page', 1) + 1)
                st.rerun()
        
        # Update current page in session state
        st.session_state.current_page = current_page
        
        # Calculate start and end indices for current page
        start_idx = (current_page - 1) * ITEMS_PER_PAGE
        end_idx = min(start_idx + ITEMS_PER_PAGE, len(filtered_df))
        
        # Get current page data
        current_page_df = filtered_df.iloc[start_idx:end_idx]
        
        st.info(f"Showing alerts {start_idx + 1}-{end_idx} of {len(filtered_df)} (Page {current_page} of {total_pages})")
    else:
        current_page_df = filtered_df
        st.session_state.current_page = 1
    
    # Display alerts in expandable sections (like home page)
    for idx, alert in current_page_df.iterrows():
        # Get the ID column name
        id_column = 'alert_id' if 'alert_id' in alert else 'id'
        
        with st.expander(f"📊 {alert.get('name', 'N/A')} - {alert.get('stock_name', 'N/A')} ({alert.get('ticker', 'N/A')})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Company:** {alert.get('stock_name', 'N/A')}")
                st.write(f"**Ticker:** {alert.get('ticker', 'N/A')}")
                st.write(f"**Exchange:** {alert.get('exchange', 'N/A')}")
                st.write(f"**Timeframe:** {alert.get('timeframe', 'N/A')}")
                st.write(f"**Action:** {alert.get('action', 'N/A')}")
                
                # Format conditions
                try:
                    if 'conditions' in alert and alert['conditions']:
                        if isinstance(alert['conditions'], list):
                            condition_strings = []
                            for c in alert['conditions']:
                                if isinstance(c, dict) and 'conditions' in c:
                                    condition_strings.append(c['conditions'])
                            condition_text = ' | '.join(condition_strings)
                        else:
                            condition_text = str(alert['conditions'])
                        st.write(f"**Conditions:** {condition_text}")
                    else:
                        st.write("**Conditions:** No conditions specified")
                except:
                    st.write("**Conditions:** Error reading conditions")
                
                # Show last triggered
                if 'last_triggered' in alert and alert['last_triggered']:
                    st.write(f"**Last Triggered:** {alert['last_triggered']}")
                else:
                    st.write("**Last Triggered:** Never")
            
            with col2:
                # Selection checkbox - check if already selected
                is_selected = alert[id_column] in st.session_state.selected_alert_ids
                if st.checkbox(f"Select for deletion", value=is_selected, key=f"select_{alert[id_column]}"):
                    if alert[id_column] not in st.session_state.selected_alert_ids:
                        st.session_state.selected_alert_ids.append(alert[id_column])
                else:
                    if alert[id_column] in st.session_state.selected_alert_ids:
                        st.session_state.selected_alert_ids.remove(alert[id_column])
    
    # Select all options
    st.markdown("---")
    
    # Show current selection count
    if st.session_state.selected_alert_ids:
        st.info(f"📋 Currently selected: {len(st.session_state.selected_alert_ids)} alert(s)")
    
    col_select1, col_select2 = st.columns(2)
    
    with col_select1:
        if st.button("✅ Select All on This Page"):
            for idx, alert in current_page_df.iterrows():
                id_column = 'alert_id' if 'alert_id' in alert else 'id'
                if alert[id_column] not in st.session_state.selected_alert_ids:
                    st.session_state.selected_alert_ids.append(alert[id_column])
            st.rerun()
    
    with col_select2:
        if total_pages > 1:
            if st.button("✅ Select All Filtered Alerts"):
                for idx, alert in filtered_df.iterrows():
                    id_column = 'alert_id' if 'alert_id' in alert else 'id'
                    if alert[id_column] not in st.session_state.selected_alert_ids:
                        st.session_state.selected_alert_ids.append(alert[id_column])
                st.rerun()
    
    if st.session_state.selected_alert_ids:
        if st.button("🗑️ Clear All Selections"):
            st.session_state.selected_alert_ids.clear()
            st.rerun()

# DELETE BUTTON - ALWAYS VISIBLE RIGHT AFTER SELECTION
if st.session_state.selected_alert_ids:
    st.markdown("---")
    st.markdown("### 🗑️ **DELETE BUTTON - ALWAYS VISIBLE**")
    
    # Get selected alerts count
    id_column = 'alert_id' if 'alert_id' in filtered_df.columns else 'id'
    selected_alerts = filtered_df[filtered_df[id_column].isin(st.session_state.selected_alert_ids)]
    
    # Make the delete button very prominent
    st.markdown("🚨 **WARNING: This action cannot be undone!** 🚨")
    
    col_delete1, col_delete2, col_delete3 = st.columns([1, 2, 1])
    with col_delete2:
        if st.button(f"🗑️ DELETE ALL {len(selected_alerts)} SELECTED ALERTS", type="primary", use_container_width=True):
            for alert_id in st.session_state.selected_alert_ids:
                repo_delete_alert(alert_id)
            refresh_alert_cache()
            st.success(f"Successfully deleted {len(st.session_state.selected_alert_ids)} alerts!")
            # Clear selections after successful deletion
            st.session_state.selected_alert_ids.clear()
            st.rerun()
    
    st.markdown("---")
else:
    st.info("No alerts selected. Use the checkboxes above to select alerts for deletion.")
    st.markdown("---")

# Display statistics
st.markdown("---")
st.subheader("📈 Alert Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Alerts", len(df_alerts))
    
with col2:
    st.metric("Filtered Alerts", len(filtered_df))
    
with col3:
    never_triggered = len(df_alerts[df_alerts['last_triggered'].isna()])
    st.metric("Never Triggered", never_triggered)
    
with col4:
    triggered_today = len(df_alerts[
        (df_alerts['last_triggered_date'].dt.date == pd.Timestamp.now().date()) &
        (df_alerts['last_triggered_date'].notna())
    ])
    st.metric("Triggered Today", triggered_today)

# Top companies by alert count
st.markdown("### 🏢 Top Companies by Alert Count")
company_counts = df_alerts['stock_name'].value_counts().head(10)
st.bar_chart(company_counts)

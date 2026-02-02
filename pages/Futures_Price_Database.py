"""
Streamlit page for viewing and analyzing futures price data
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import numpy as np

from src.data_access.db_config import db_config

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent))

# Page config
st.set_page_config(
    page_title="Futures Price Database",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Futures Price Database")
st.markdown("View and analyze continuous futures price data with adjustment methods")

# Initialize session state
if 'selected_future' not in st.session_state:
    st.session_state.selected_future = None
if 'adjustment_method' not in st.session_state:
    st.session_state.adjustment_method = 'none'

@st.cache_resource
def load_futures_metadata():
    """Load futures metadata from JSON"""
    try:
        with open('futures_database.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return {}

def get_database_connection():
    """Create database connection - no caching to avoid closed connection issues"""
    return db_config.get_connection(role="futures")

def get_unique_categories(futures_db):
    """Get unique categories from futures database"""
    categories = set()
    for symbol, data in futures_db.items():
        category = data.get('category', '')
        if category:
            categories.add(category)
    return sorted(list(categories))

def get_futures_by_category(futures_db, selected_categories):
    """Get futures filtered by category"""
    futures = []
    for symbol, data in futures_db.items():
        if not selected_categories or data.get('category', '') in selected_categories:
            futures.append(symbol)
    return sorted(futures)

def load_futures_price_data(conn, symbols=None, start_date=None, end_date=None):
    """Load futures price data with filters"""
    query = "SELECT * FROM continuous_prices WHERE 1=1"
    params = []

    if symbols:
        placeholders = ','.join(['%s' for _ in symbols])
        query += f" AND symbol IN ({placeholders})"
        params.extend(symbols)

    if start_date:
        query += " AND date >= %s"
        params.append(start_date)

    if end_date:
        query += " AND date <= %s"
        params.append(end_date)

    query += " ORDER BY symbol, date DESC"

    try:
        df = pd.read_sql_query(query, conn, params=params)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def apply_adjustment(df, method='panama'):
    """Apply Panama or Ratio adjustment to futures prices"""
    if method == 'none' or df.empty:
        return df

    # Sort by date ascending for proper adjustment
    df = df.sort_values('date').copy()

    # Calculate daily returns to detect roll gaps
    df['close_change'] = df['close'].diff()
    df['pct_change'] = df['close'].pct_change()

    # Identify potential roll points (gaps > 2% or > 2 standard deviations)
    returns_std = df['pct_change'].std()
    roll_threshold = max(0.02, 2 * returns_std)  # 2% or 2 std, whichever is larger

    df['is_roll'] = abs(df['pct_change']) > roll_threshold

    # Apply adjustment
    if method == 'panama':
        # Additive adjustment - preserves dollar moves
        df['adj_close'] = df['close'].copy()
        cumulative_adjustment = 0

        for idx in df.index:
            if df.loc[idx, 'is_roll'] and idx > 0:
                # Calculate the gap
                gap = df.loc[idx, 'close_change']
                # Adjust by removing the gap
                cumulative_adjustment -= gap

            df.loc[idx, 'adj_close'] = df.loc[idx, 'close'] + cumulative_adjustment
            df.loc[idx, 'adj_open'] = df.loc[idx, 'open'] + cumulative_adjustment
            df.loc[idx, 'adj_high'] = df.loc[idx, 'high'] + cumulative_adjustment
            df.loc[idx, 'adj_low'] = df.loc[idx, 'low'] + cumulative_adjustment

    elif method == 'ratio':
        # Multiplicative adjustment - preserves percentage returns
        df['adj_close'] = df['close'].copy()
        cumulative_factor = 1.0

        for idx in df.index:
            if df.loc[idx, 'is_roll'] and idx > 0:
                # Calculate the ratio
                prev_idx = df.index[df.index.get_loc(idx) - 1]
                if df.loc[prev_idx, 'close'] != 0:
                    ratio = df.loc[idx, 'close'] / (df.loc[prev_idx, 'close'] + df.loc[idx, 'close_change'])
                    cumulative_factor *= ratio

            df.loc[idx, 'adj_close'] = df.loc[idx, 'close'] * cumulative_factor
            df.loc[idx, 'adj_open'] = df.loc[idx, 'open'] * cumulative_factor
            df.loc[idx, 'adj_high'] = df.loc[idx, 'high'] * cumulative_factor
            df.loc[idx, 'adj_low'] = df.loc[idx, 'low'] * cumulative_factor

    # Clean up temporary columns
    df = df.drop(['close_change', 'pct_change', 'is_roll'], axis=1)

    return df

def get_database_stats(conn):
    """Get database statistics"""
    stats = {}

    # Get total symbols
    query = "SELECT COUNT(DISTINCT symbol) as total FROM continuous_prices"
    result = pd.read_sql_query(query, conn)
    stats['total_symbols'] = result['total'][0] if not result.empty else 0

    # Get date range
    query = "SELECT MIN(date) as min_date, MAX(date) as max_date FROM continuous_prices"
    result = pd.read_sql_query(query, conn)
    if not result.empty:
        stats['min_date'] = result['min_date'][0]
        stats['max_date'] = result['max_date'][0]

    # Get total records
    query = "SELECT COUNT(*) as total FROM continuous_prices"
    result = pd.read_sql_query(query, conn)
    stats['total_records'] = result['total'][0] if not result.empty else 0

    return stats

# Main app
def main():
    # Load data
    futures_db = load_futures_metadata()
    conn = get_database_connection()

    # Get database statistics first to get min_date
    stats = get_database_stats(conn)

    # Sidebar filters
    with st.sidebar:
        st.header("🔍 Filters")

        # Category filter
        categories = get_unique_categories(futures_db)
        selected_categories = st.multiselect(
            "Categories",
            categories,
            help="Filter by futures category"
        )

        # Futures selection
        available_futures = get_futures_by_category(futures_db, selected_categories)
        selected_futures = st.multiselect(
            "Select Futures",
            available_futures,
            help="Choose specific futures to view"
        )

        # Date range
        st.subheader("📅 Date Range")

        # Get the earliest date from database or default to 365 days ago
        if stats.get('min_date'):
            try:
                min_date_db = pd.to_datetime(stats['min_date']).date()
                default_start_date = min_date_db
            except:
                default_start_date = datetime.now().date() - timedelta(days=365)
        else:
            default_start_date = datetime.now().date() - timedelta(days=365)

        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=default_start_date,
                max_value=datetime.now().date()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now().date(),
                max_value=datetime.now().date()
            )

        # Adjustment method
        st.subheader("📊 Adjustment Method")
        adjustment_method = st.radio(
            "Select adjustment for continuous contracts:",
            ["none", "panama", "ratio"],
            format_func=lambda x: {
                "none": "None (Raw Data)",
                "panama": "Panama (Additive)",
                "ratio": "Ratio (Multiplicative)"
            }[x],
            help="""
            - None: Raw continuous contract data with roll gaps
            - Panama: Additive adjustment - preserves dollar moves
            - Ratio: Multiplicative adjustment - preserves % returns
            """
        )
        st.session_state.adjustment_method = adjustment_method

        # Load data button
        if st.button("📥 Load Data", type="primary", use_container_width=True):
            st.session_state.selected_future = selected_futures[0] if selected_futures else None

    # Main content area
    # Display statistics (already calculated above)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Futures", f"{stats.get('total_symbols', 0):,}")
    with col2:
        st.metric("Total Records", f"{stats.get('total_records', 0):,}")
    with col3:
        min_date_val = stats.get('min_date')
        if min_date_val is not None:
            if isinstance(min_date_val, (pd.Timestamp, datetime, date)):
                min_display = pd.to_datetime(min_date_val).strftime("%Y-%m-%d")
            else:
                min_display = str(min_date_val)
            st.metric("First Date", min_display)
    with col4:
        max_date_val = stats.get('max_date')
        if max_date_val is not None:
            if isinstance(max_date_val, (pd.Timestamp, datetime, date)):
                max_display = pd.to_datetime(max_date_val).strftime("%Y-%m-%d")
            else:
                max_display = str(max_date_val)
            st.metric("Last Date", max_display)

    st.divider()

    # Load and display price data
    if selected_futures:
        # Load data
        df = load_futures_price_data(conn, selected_futures, start_date, end_date)

        if not df.empty:
            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Price Charts", "📋 Data Table", "📈 Statistics", "🔄 Roll Analysis"])

            with tab1:
                # Price chart for each selected future
                for symbol in selected_futures:
                    symbol_df = df[df['symbol'] == symbol].copy()

                    if not symbol_df.empty:
                        # Apply adjustment
                        symbol_df = apply_adjustment(symbol_df, adjustment_method)

                        # Get futures info
                        futures_info = futures_db.get(symbol, {})
                        name = futures_info.get('name', symbol)
                        category = futures_info.get('category', 'Unknown')

                        st.subheader(f"{symbol} - {name}")
                        st.caption(f"Category: {category} | Records: {len(symbol_df):,}")

                        # Create candlestick chart
                        fig = go.Figure()

                        # Use adjusted prices if available
                        if adjustment_method != 'none' and 'adj_close' in symbol_df.columns:
                            fig.add_trace(go.Candlestick(
                                x=symbol_df['date'],
                                open=symbol_df['adj_open'],
                                high=symbol_df['adj_high'],
                                low=symbol_df['adj_low'],
                                close=symbol_df['adj_close'],
                                name=f"{symbol} (Adjusted)"
                            ))

                            # Add volume
                            fig.add_trace(go.Bar(
                                x=symbol_df['date'],
                                y=symbol_df['volume'],
                                name='Volume',
                                yaxis='y2',
                                opacity=0.3
                            ))
                        else:
                            fig.add_trace(go.Candlestick(
                                x=symbol_df['date'],
                                open=symbol_df['open'],
                                high=symbol_df['high'],
                                low=symbol_df['low'],
                                close=symbol_df['close'],
                                name=symbol
                            ))

                            # Add volume
                            fig.add_trace(go.Bar(
                                x=symbol_df['date'],
                                y=symbol_df['volume'],
                                name='Volume',
                                yaxis='y2',
                                opacity=0.3
                            ))

                        # Update layout
                        fig.update_layout(
                            title=f"{symbol} - {adjustment_method.capitalize()} Adjustment" if adjustment_method != 'none' else symbol,
                            yaxis_title="Price",
                            yaxis2=dict(
                                title="Volume",
                                overlaying='y',
                                side='right'
                            ),
                            xaxis_rangeslider_visible=False,
                            height=500
                        )

                        st.plotly_chart(fig, use_container_width=True)

            with tab2:
                # Data table
                st.subheader("📋 Price Data Table")

                # Display options
                col1, col2, col3 = st.columns(3)
                with col1:
                    show_adjusted = st.checkbox("Show Adjusted Prices", value=True)
                with col2:
                    records_per_page = st.selectbox("Records per page", [50, 100, 200, 500], index=0)
                with col3:
                    export_format = st.selectbox("Export format", ["CSV", "Excel"])

                # Prepare display dataframe
                display_df = df.copy()

                # Apply adjustment if selected
                adjusted_dfs = []
                for symbol in selected_futures:
                    symbol_df = display_df[display_df['symbol'] == symbol].copy()
                    if not symbol_df.empty:
                        symbol_df = apply_adjustment(symbol_df, adjustment_method)
                        adjusted_dfs.append(symbol_df)

                if adjusted_dfs:
                    display_df = pd.concat(adjusted_dfs, ignore_index=True)

                # Select columns to display
                if show_adjusted and adjustment_method != 'none' and 'adj_close' in display_df.columns:
                    display_columns = ['symbol', 'date', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'volume']
                    display_df = display_df[display_columns].rename(columns={
                        'adj_open': 'open',
                        'adj_high': 'high',
                        'adj_low': 'low',
                        'adj_close': 'close'
                    })
                else:
                    display_columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
                    display_df = display_df[display_columns]

                # Sort by date descending
                display_df = display_df.sort_values(['symbol', 'date'], ascending=[True, False])

                # Display with pagination
                total_records = len(display_df)
                total_pages = (total_records // records_per_page) + (1 if total_records % records_per_page else 0)

                page = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
                start_idx = (page - 1) * records_per_page
                end_idx = min(start_idx + records_per_page, total_records)

                st.dataframe(
                    display_df.iloc[start_idx:end_idx],
                    use_container_width=True,
                    height=400
                )

                st.caption(f"Showing records {start_idx+1}-{end_idx} of {total_records}")

                # Export buttons
                col1, col2 = st.columns(2)
                with col1:
                    if export_format == "CSV":
                        csv = display_df.to_csv(index=False)
                        st.download_button(
                            "📥 Download CSV",
                            csv,
                            f"futures_prices_{datetime.now().strftime('%Y%m%d')}.csv",
                            "text/csv"
                        )
                with col2:
                    if export_format == "Excel":
                        # Create Excel file in memory
                        from io import BytesIO
                        buffer = BytesIO()
                        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                            display_df.to_excel(writer, index=False, sheet_name='Futures Prices')
                        buffer.seek(0)
                        st.download_button(
                            "📥 Download Excel",
                            buffer,
                            f"futures_prices_{datetime.now().strftime('%Y%m%d')}.xlsx",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

            with tab3:
                # Statistics
                st.subheader("📈 Price Statistics")

                for symbol in selected_futures:
                    symbol_df = df[df['symbol'] == symbol].copy()

                    if not symbol_df.empty:
                        # Apply adjustment
                        symbol_df = apply_adjustment(symbol_df, adjustment_method)

                        # Get futures info
                        futures_info = futures_db.get(symbol, {})
                        name = futures_info.get('name', symbol)

                        st.write(f"**{symbol} - {name}**")

                        # Use adjusted prices if available
                        price_col = 'adj_close' if adjustment_method != 'none' and 'adj_close' in symbol_df.columns else 'close'

                        # Calculate statistics
                        col1, col2, col3, col4, col5 = st.columns(5)

                        with col1:
                            current_price = symbol_df[price_col].iloc[0] if not symbol_df.empty else 0
                            st.metric("Current Price", f"{current_price:.2f}")

                        with col2:
                            avg_price = symbol_df[price_col].mean()
                            st.metric("Average", f"{avg_price:.2f}")

                        with col3:
                            min_price = symbol_df[price_col].min()
                            st.metric("Min", f"{min_price:.2f}")

                        with col4:
                            max_price = symbol_df[price_col].max()
                            st.metric("Max", f"{max_price:.2f}")

                        with col5:
                            volatility = symbol_df[price_col].pct_change().std() * np.sqrt(252) * 100
                            st.metric("Volatility", f"{volatility:.1f}%")

                        # Returns analysis
                        col1, col2, col3, col4 = st.columns(4)

                        symbol_df = symbol_df.sort_values('date')

                        with col1:
                            # 1 month return
                            if len(symbol_df) >= 20:
                                ret_1m = (symbol_df[price_col].iloc[-1] / symbol_df[price_col].iloc[-20] - 1) * 100
                                st.metric("1M Return", f"{ret_1m:.2f}%")

                        with col2:
                            # 3 month return
                            if len(symbol_df) >= 60:
                                ret_3m = (symbol_df[price_col].iloc[-1] / symbol_df[price_col].iloc[-60] - 1) * 100
                                st.metric("3M Return", f"{ret_3m:.2f}%")

                        with col3:
                            # 6 month return
                            if len(symbol_df) >= 120:
                                ret_6m = (symbol_df[price_col].iloc[-1] / symbol_df[price_col].iloc[-120] - 1) * 100
                                st.metric("6M Return", f"{ret_6m:.2f}%")

                        with col4:
                            # 1 year return
                            if len(symbol_df) >= 252:
                                ret_1y = (symbol_df[price_col].iloc[-1] / symbol_df[price_col].iloc[-252] - 1) * 100
                                st.metric("1Y Return", f"{ret_1y:.2f}%")

                        st.divider()

            with tab4:
                # Roll Analysis
                st.subheader("🔄 Roll Gap Analysis")

                st.info("""
                This analysis identifies potential contract roll points by detecting large price gaps.
                Roll gaps are typically removed by adjustment methods (Panama or Ratio).
                """)

                for symbol in selected_futures:
                    symbol_df = df[df['symbol'] == symbol].copy()

                    if not symbol_df.empty:
                        # Sort by date
                        symbol_df = symbol_df.sort_values('date')

                        # Calculate gaps
                        symbol_df['close_change'] = symbol_df['close'].diff()
                        symbol_df['pct_change'] = symbol_df['close'].pct_change() * 100

                        # Identify large gaps (potential rolls)
                        threshold = 2.0  # 2% threshold
                        roll_gaps = symbol_df[abs(symbol_df['pct_change']) > threshold].copy()

                        # Get futures info
                        futures_info = futures_db.get(symbol, {})
                        name = futures_info.get('name', symbol)

                        st.write(f"**{symbol} - {name}**")

                        if not roll_gaps.empty:
                            st.write(f"Found {len(roll_gaps)} potential roll gaps (>{threshold}% change)")

                            # Display roll gaps
                            roll_display = roll_gaps[['date', 'close', 'close_change', 'pct_change']].copy()
                            roll_display['date'] = roll_display['date'].dt.strftime('%Y-%m-%d')
                            roll_display.columns = ['Date', 'Close', 'Price Gap', '% Change']

                            st.dataframe(
                                roll_display.style.format({
                                    'Close': '{:.2f}',
                                    'Price Gap': '{:+.2f}',
                                    '% Change': '{:+.2f}%'
                                }),
                                use_container_width=True,
                                height=min(400, len(roll_display) * 35 + 38)
                            )

                            # Quarterly vs Monthly analysis
                            quarterly_symbols = ['ES', 'NQ', 'YM', 'RTY', '6E', '6B', '6J', 'ZN', 'ZB']
                            roll_freq = "Quarterly (Mar, Jun, Sep, Dec)" if symbol in quarterly_symbols else "Monthly"

                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Roll Frequency", roll_freq)
                            with col2:
                                avg_gap = roll_gaps['close_change'].abs().mean()
                                st.metric("Avg Gap Size", f"{avg_gap:.2f} pts")
                            with col3:
                                max_gap = roll_gaps['close_change'].abs().max()
                                st.metric("Max Gap Size", f"{max_gap:.2f} pts")
                        else:
                            st.success(f"No significant roll gaps detected (>{threshold}% change)")

                        st.divider()
        else:
            st.warning("No data found for selected futures and date range")
    else:
        # Show instructions
        st.info("👈 Select futures from the sidebar to view price data")

        # Show available futures summary
        st.subheader("Available Futures Contracts")

        # Group by category
        category_summary = {}
        for symbol, data in futures_db.items():
            category = data.get('category', 'Unknown')
            if category not in category_summary:
                category_summary[category] = []
            category_summary[category].append(symbol)

        # Display in columns
        cols = st.columns(3)
        for idx, (category, symbols) in enumerate(sorted(category_summary.items())):
            with cols[idx % 3]:
                st.write(f"**{category}** ({len(symbols)})")
                st.write(", ".join(symbols[:10]))
                if len(symbols) > 10:
                    st.caption(f"...and {len(symbols) - 10} more")

    # Stale Data Check Section
    st.divider()
    st.subheader("⚠️ Contracts with Stale Data")
    st.caption(f"Contracts not updated as of today ({datetime.now().strftime('%Y-%m-%d')})")

    # Query for contracts with stale data
    today = datetime.now().strftime('%Y-%m-%d')

    stale_query = """
    SELECT
        symbol,
        MAX(date) as last_update,
        COUNT(*) as total_records,
        CAST((%s::date - MAX(date)) AS INTEGER) as days_stale
    FROM continuous_prices
    GROUP BY symbol
    HAVING MAX(date) < %s::date
    ORDER BY days_stale DESC
    """

    try:
        stale_df = pd.read_sql_query(stale_query, conn, params=[today, today])

        if not stale_df.empty:
            # Add futures metadata
            stale_df['name'] = stale_df['symbol'].apply(lambda x: futures_db.get(x, {}).get('name', 'Unknown'))
            stale_df['category'] = stale_df['symbol'].apply(lambda x: futures_db.get(x, {}).get('category', 'Unknown'))
            stale_df['exchange'] = stale_df['symbol'].apply(lambda x: futures_db.get(x, {}).get('exchange', 'Unknown'))
            stale_df['ib_metadata'] = stale_df['symbol'].apply(lambda x: 'Yes' if futures_db.get(x, {}).get('ib_metadata_available', False) else 'No')

            # Format days stale
            stale_df['days_stale'] = stale_df['days_stale'].round(0).astype(int)

            # Reorder columns
            display_df = stale_df[['symbol', 'name', 'category', 'exchange', 'last_update', 'days_stale', 'total_records', 'ib_metadata']]

            # Display summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Stale Contracts", len(stale_df))
            with col2:
                avg_days = stale_df['days_stale'].mean()
                st.metric("Avg Days Stale", f"{avg_days:.1f}")
            with col3:
                max_days = stale_df['days_stale'].max()
                st.metric("Most Stale", f"{max_days} days")
            with col4:
                no_ib = len(stale_df[stale_df['ib_metadata'] == 'No'])
                st.metric("No IB Metadata", no_ib)

            # Display table
            st.dataframe(
                display_df,
                use_container_width=True,
                column_config={
                    "symbol": st.column_config.TextColumn("Symbol", width="small"),
                    "name": st.column_config.TextColumn("Name", width="medium"),
                    "category": st.column_config.TextColumn("Category", width="small"),
                    "exchange": st.column_config.TextColumn("Exchange", width="small"),
                    "last_update": st.column_config.DateColumn("Last Update", format="YYYY-MM-DD"),
                    "days_stale": st.column_config.NumberColumn("Days Stale", format="%d days"),
                    "total_records": st.column_config.NumberColumn("Records", format="%d"),
                    "ib_metadata": st.column_config.TextColumn("IB Data", width="small")
                }
            )

            # Warning for contracts without IB metadata
            no_ib_contracts = stale_df[stale_df['ib_metadata'] == 'No']['symbol'].tolist()
            if no_ib_contracts:
                st.warning(f"⚠️ The following contracts have no IB metadata and cannot be updated: {', '.join(no_ib_contracts)}")

            # Update buttons
            st.subheader("🔄 Update Price Data")

            # Add IB connection status check
            st.info("📌 **Requirements**: Interactive Brokers TWS or Gateway must be running on port 7496")

            col1, col2, col3 = st.columns(3)

            with col1:
                if st.button("📊 Update All Stale Contracts", type="primary", help="Update price data for all contracts with stale data"):
                    with st.spinner(f"Updating {len(stale_df)} contracts... This may take several minutes."):
                        try:
                            import subprocess
                            result = subprocess.run(
                                [sys.executable, "futures_price_updater.py"],
                                capture_output=True,
                                text=True
                            )
                            if result.returncode == 0:
                                st.success("✅ Price update completed successfully!")
                                st.rerun()
                            else:
                                error_msg = result.stderr if result.stderr else result.stdout
                                st.error("❌ Update failed")
                                with st.expander("View error details"):
                                    st.code(error_msg[:2000])
                                st.info("💡 Make sure Interactive Brokers TWS or Gateway is running and connected")
                        except Exception as e:
                            st.error(f"❌ Update failed: {str(e)}")

            with col2:
                # Select specific contracts to update
                selected_to_update = st.multiselect(
                    "Select specific contracts to update:",
                    stale_df['symbol'].tolist(),
                    help="Choose which contracts to update"
                )

            with col3:
                if selected_to_update:
                    if st.button(f"📊 Update Selected ({len(selected_to_update)})", type="secondary"):
                        with st.spinner(f"Updating {len(selected_to_update)} selected contracts..."):
                            try:
                                import subprocess
                                # Save selected symbols to temp file
                                import tempfile
                                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                                    for symbol in selected_to_update:
                                        f.write(f"{symbol}\n")
                                    temp_file = f.name

                                result = subprocess.run(
                                    [sys.executable, "futures_price_updater.py", "--symbols", temp_file],
                                    capture_output=True,
                                    text=True,
                                    timeout=300
                                )

                                # Clean up temp file
                                import os
                                try:
                                    os.remove(temp_file)
                                except:
                                    pass

                                if result.returncode == 0:
                                    st.success(f"✅ Updated {len(selected_to_update)} contracts successfully!")
                                    st.rerun()
                                else:
                                    error_msg = result.stderr if result.stderr else result.stdout
                                    st.error("❌ Update failed")
                                    with st.expander("View error details"):
                                        st.code(error_msg[:2000])
                                    st.info("💡 Make sure Interactive Brokers TWS or Gateway is running and connected")
                            except subprocess.TimeoutExpired:
                                st.error("❌ Update timed out after 5 minutes")
                            except Exception as e:
                                st.error(f"❌ Update failed: {str(e)}")
        else:
            st.success("✅ All contracts have been updated today!")

    except Exception as e:
        st.error(f"Error checking stale data: {e}")

    # Close database connection at the very end
    try:
        db_config.close_connection(conn)
    except Exception:
        pass  # Connection may already be closed

if __name__ == "__main__":
    main()

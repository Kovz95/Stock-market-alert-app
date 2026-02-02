import streamlit as st
import pandas as pd
import os
import sys
import json
from datetime import datetime
from src.data_access.db_config import db_config

# Add the parent directory to the path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

st.set_page_config(
    page_title="Futures Database",
    page_icon="📈",
    layout="wide",
)

# Load futures database
@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_futures_database():
    """Load and prepare futures database for display"""
    try:
        # Load the futures database JSON
        with open('futures_database.json', 'r') as f:
            futures_data = json.load(f)

        # Convert to DataFrame
        df_data = []
        for symbol, data in futures_data.items():
            row = {'Symbol': symbol}
            row.update(data)
            df_data.append(row)

        df = pd.DataFrame(df_data)

        return df
    except Exception as e:
        st.error(f"Error loading futures database: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=600)
def load_futures_price_stats():
    """Load futures price statistics from PostgreSQL"""
    try:
        with db_config.connection(role="futures_prices") as conn:
            query = """
                SELECT
                    symbol,
                    COUNT(*) as days,
                    MIN(date) as first_date,
                    MAX(date) as last_date,
                    AVG(volume) as avg_volume
                FROM continuous_prices
                GROUP BY symbol
            """
            return pd.read_sql_query(query, conn)
    except Exception:
        return pd.DataFrame()

def main():
    st.title("📈 Futures Database")
    st.markdown("Comprehensive database of futures contracts with Interactive Brokers integration")

    # Load data
    df = load_futures_database()
    price_stats = load_futures_price_stats()

    if df.empty:
        st.error("No futures data available. Please check if 'futures_database.json' exists.")
        return

    # Merge price statistics if available
    if not price_stats.empty:
        df = df.merge(price_stats, left_on='symbol', right_on='symbol', how='left')

    # Database Statistics Section
    st.markdown("---")
    st.subheader("📊 Database Overview Statistics")

    # Calculate statistics
    total_futures = len(df)
    with_ib_metadata = df['ib_metadata_available'].sum() if 'ib_metadata_available' in df.columns else 0
    with_price_data = (~df['days'].isna()).sum() if 'days' in df.columns else 0
    unique_categories = df['category'].nunique() if 'category' in df.columns else 0
    unique_exchanges = df['exchange'].nunique() if 'exchange' in df.columns else 0

    # Display top-level metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Futures", f"{total_futures:,}")
    with col2:
        st.metric("With IB Metadata", f"{with_ib_metadata:,}")
    with col3:
        st.metric("With Price Data", f"{with_price_data:,}")
    with col4:
        st.metric("Categories", f"{unique_categories:,}")
    with col5:
        st.metric("Exchanges", f"{unique_exchanges:,}")

    # Category Breakdown
    if 'category' in df.columns:
        st.markdown("### 🏭 Futures by Category")

        category_counts = df['category'].value_counts()

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Major Categories:**")
            for category, count in category_counts.items():
                pct = (count / total_futures * 100)
                # Show IB metadata status for each category
                cat_df = df[df['category'] == category]
                with_ib = cat_df['ib_metadata_available'].sum() if 'ib_metadata_available' in cat_df.columns else 0
                with_prices = (~cat_df['days'].isna()).sum() if 'days' in cat_df.columns else 0
                st.write(f"• **{category}**: {count} contracts ({pct:.1f}%)")
                st.write(f"  → IB: {with_ib}/{count}, Prices: {with_prices}/{count}")

        with col2:
            # Exchange breakdown
            if 'exchange' in df.columns:
                st.write("**Top Exchanges:**")
                exchange_counts = df['exchange'].value_counts().head(10)
                for exchange, count in exchange_counts.items():
                    pct = (count / total_futures * 100)
                    st.write(f"• **{exchange}**: {count} contracts ({pct:.1f}%)")

    # IB Metadata Status
    st.markdown("### 🔗 Interactive Brokers Integration Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        if 'ib_metadata_available' in df.columns:
            ib_yes = df['ib_metadata_available'].sum()
            ib_no = len(df) - ib_yes
            st.metric("Connected to IB", f"{ib_yes} contracts")
            st.metric("Not Available in IB", f"{ib_no} contracts")
            st.progress(ib_yes / total_futures)

    with col2:
        if 'contract_id' in df.columns:
            has_contract_id = df['contract_id'].notna().sum()
            st.metric("With Contract ID", f"{has_contract_id} contracts")
            if 'min_tick' in df.columns:
                has_min_tick = df['min_tick'].notna().sum()
                st.metric("With Min Tick", f"{has_min_tick} contracts")

    with col3:
        if 'days' in df.columns:
            has_prices = (~df['days'].isna()).sum()
            total_days = df['days'].sum()
            avg_days = int(total_days / has_prices) if has_prices > 0 else 0
            st.metric("With Historical Prices", f"{has_prices} contracts")
            st.metric("Avg Days of History", f"{avg_days:,}")

    st.markdown("---")

    # Create sidebar for filters
    with st.sidebar:
        st.header("🔍 Filters")

        # Category Filter
        st.subheader("📊 Category")
        if 'category' in df.columns:
            categories = sorted(df['category'].dropna().unique())
            selected_categories = st.multiselect(
                "Select categories:",
                categories,
                help="Filter by futures category"
            )
        else:
            selected_categories = []

        # Exchange Filter
        st.subheader("🏛️ Exchange")
        if 'exchange' in df.columns:
            # Filter exchanges based on selected categories if applicable
            if selected_categories:
                available_exchanges = sorted(df[df['category'].isin(selected_categories)]['exchange'].dropna().unique())
            else:
                available_exchanges = sorted(df['exchange'].dropna().unique())
            selected_exchanges = st.multiselect(
                "Select exchanges:",
                available_exchanges,
                help="Filter by exchange"
            )
        else:
            selected_exchanges = []

        # IB Metadata Filter
        st.subheader("🔗 IB Status")
        ib_filter = st.radio(
            "IB Metadata:",
            ["All", "With IB Metadata", "Without IB Metadata"],
            help="Filter by IB metadata availability"
        )

        # Price Data Filter
        st.subheader("💹 Price Data")
        price_filter = st.radio(
            "Historical Prices:",
            ["All", "With Prices", "Without Prices"],
            help="Filter by price data availability"
        )

        # Roll Frequency Filter
        st.subheader("📅 Roll Frequency")
        roll_frequencies = ["All", "Monthly", "Quarterly"]
        selected_roll_freq = st.radio(
            "Roll frequency:",
            roll_frequencies,
            help="Filter by contract roll frequency"
        )

        # Clear filters button
        if st.button("🗑️ Clear All Filters"):
            st.rerun()

    # Apply filters
    filtered_df = df.copy()

    # Apply category filter
    if selected_categories and 'category' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['category'].isin(selected_categories)]

    # Apply exchange filter
    if selected_exchanges and 'exchange' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['exchange'].isin(selected_exchanges)]

    # Apply IB metadata filter
    if ib_filter == "With IB Metadata" and 'ib_metadata_available' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['ib_metadata_available'] == True]
    elif ib_filter == "Without IB Metadata" and 'ib_metadata_available' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['ib_metadata_available'] != True]

    # Apply price data filter
    if price_filter == "With Prices" and 'days' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['days'].notna()]
    elif price_filter == "Without Prices" and 'days' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['days'].isna()]

    # Apply roll frequency filter
    monthly_symbols = ['CL', 'NG', 'RB', 'HO', 'GC', 'SI', 'HG', 'PL', 'PA',
                      'ZC', 'ZS', 'ZW', 'ZL', 'ZM', 'KE', 'LE', 'GF', 'HE',
                      'CC', 'KC', 'SB', 'CT', 'OJ', 'LBS', 'DX']

    if selected_roll_freq == "Monthly":
        filtered_df = filtered_df[filtered_df['symbol'].isin(monthly_symbols)]
    elif selected_roll_freq == "Quarterly":
        filtered_df = filtered_df[~filtered_df['symbol'].isin(monthly_symbols)]

    # Display filtered results
    st.subheader(f"📋 Futures Contract List ({len(filtered_df):,} contracts)")

    # Search functionality
    search_term = st.text_input(
        "🔍 Search futures:",
        placeholder="Enter symbol, name, or any text...",
        help="Search across all columns"
    )

    if search_term:
        # Create a mask for search across all string columns
        search_mask = pd.DataFrame([filtered_df[col].astype(str).str.contains(search_term, case=False, na=False)
                                  for col in filtered_df.columns]).any()
        filtered_df = filtered_df[search_mask]

    # Prepare display columns
    display_columns = ['Symbol', 'name', 'category', 'exchange']

    # Add IB metadata columns if present
    ib_columns = ['ib_metadata_available', 'local_symbol', 'multiplier', 'min_tick', 'contract_id']
    for col in ib_columns:
        if col in filtered_df.columns:
            display_columns.append(col)

    # Add price data columns if present
    if 'days' in filtered_df.columns:
        display_columns.extend(['days', 'first_date', 'last_date', 'avg_volume'])

    # Add trading hours if present
    if 'trading_hours' in filtered_df.columns:
        display_columns.append('trading_hours')

    # Filter to only existing columns
    display_columns = [col for col in display_columns if col in filtered_df.columns]
    display_df = filtered_df[display_columns]

    # Display results
    if not display_df.empty:
        # Show download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download as CSV",
            data=csv,
            file_name="futures_database_export.csv",
            mime="text/csv"
        )

        # Configure column display names and formatting
        column_config = {
            "Symbol": st.column_config.TextColumn("Symbol", width="small"),
            "name": st.column_config.TextColumn("Name", width="large"),
            "category": st.column_config.TextColumn("Category", width="medium"),
            "exchange": st.column_config.TextColumn("Exchange", width="small"),
            "ib_metadata_available": st.column_config.CheckboxColumn("IB", width="small"),
            "local_symbol": st.column_config.TextColumn("IB Symbol", width="small"),
            "multiplier": st.column_config.NumberColumn("Multiplier", width="small", format="%.0f"),
            "min_tick": st.column_config.NumberColumn("Min Tick", width="small", format="%.4f"),
            "contract_id": st.column_config.NumberColumn("Contract ID", width="medium"),
            "days": st.column_config.NumberColumn("Days", width="small", format="%d"),
            "first_date": st.column_config.TextColumn("First Date", width="small"),
            "last_date": st.column_config.TextColumn("Last Date", width="small"),
            "avg_volume": st.column_config.NumberColumn("Avg Volume", width="medium", format="%.0f"),
            "trading_hours": st.column_config.TextColumn("Trading Hours", width="large")
        }

        # Display the table
        st.dataframe(
            display_df,
            use_container_width=True,
            height=600,
            column_config=column_config
        )

        # Show detailed statistics
        st.divider()
        st.subheader("📈 Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Category Distribution:**")
            if 'category' in filtered_df.columns:
                category_counts = filtered_df['category'].value_counts().head(10)
                for category, count in category_counts.items():
                    st.write(f"• {category}: {count:,}")

        with col2:
            st.write("**IB Integration Status:**")
            if 'ib_metadata_available' in filtered_df.columns:
                with_ib = filtered_df['ib_metadata_available'].sum()
                without_ib = len(filtered_df) - with_ib
                st.write(f"• With IB Metadata: {with_ib:,}")
                st.write(f"• Without IB Metadata: {without_ib:,}")
                if with_ib > 0:
                    st.write(f"• Coverage: {(with_ib/len(filtered_df)*100):.1f}%")

        with col3:
            st.write("**Price Data Status:**")
            if 'days' in filtered_df.columns:
                with_prices = (~filtered_df['days'].isna()).sum()
                without_prices = len(filtered_df) - with_prices
                st.write(f"• With Price Data: {with_prices:,}")
                st.write(f"• Without Price Data: {without_prices:,}")
                if with_prices > 0:
                    avg_days = int(filtered_df['days'].mean())
                    st.write(f"• Avg History: {avg_days:,} days")
    else:
        st.warning("No futures match the selected filters. Try adjusting your search criteria.")

    # Roll Schedule Information
    st.divider()
    st.subheader("📅 Contract Roll Schedule")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Quarterly Roll Contracts (Mar, Jun, Sep, Dec):**")
        quarterly = ['ES', 'NQ', 'YM', 'RTY', '6E', '6B', '6J', '6A', '6C',
                    'ZN', 'ZB', 'ZF', 'UB', 'SR3', 'ER3']
        for symbol in quarterly[:10]:
            if symbol in df['symbol'].values:
                name = df[df['symbol'] == symbol]['name'].iloc[0] if not df[df['symbol'] == symbol].empty else symbol
                st.write(f"• {symbol}: {name}")

    with col2:
        st.write("**Monthly Roll Contracts:**")
        monthly = ['CL', 'NG', 'GC', 'SI', 'ZC', 'ZS', 'ZW', 'HG', 'PL', 'PA']
        for symbol in monthly:
            if symbol in df['symbol'].values:
                name = df[df['symbol'] == symbol]['name'].iloc[0] if not df[df['symbol'] == symbol].empty else symbol
                st.write(f"• {symbol}: {name}")

    # IB Connection Status
    st.divider()
    st.subheader("🔗 Interactive Brokers Connection")

    # Check if IB is configured
    try:
        with open('ib_futures_config.json', 'r') as f:
            ib_config = json.load(f)
            host = ib_config.get('connection', {}).get('host', '127.0.0.1')
            port = ib_config.get('connection', {}).get('port', 7496)
            st.success(f"✅ IB Configuration: {host}:{port}")
            st.info("Ensure TWS or IB Gateway is running on this port for live data updates")
    except:
        st.warning("⚠️ IB configuration not found. Create ib_futures_config.json to enable IB integration")

    # Adjustment Methods Info
    st.divider()
    st.subheader("📊 Continuous Contract Adjustments")

    st.info("""
    **Why Adjustments Matter:**
    Continuous contracts from IB contain price gaps at roll points that can trigger false alerts.

    **Available Adjustment Methods:**
    - **Panama (Additive)**: Adds/subtracts roll gaps - preserves absolute price levels (good for stop-loss alerts)
    - **Ratio (Multiplicative)**: Scales by roll ratios - preserves percentage returns (good for % change alerts)

    **When to Use Each:**
    - Use Panama for price level alerts (e.g., "Alert when ES > 4500")
    - Use Ratio for percentage alerts (e.g., "Alert when CL up 5%")
    """)

if __name__ == "__main__":
    main()

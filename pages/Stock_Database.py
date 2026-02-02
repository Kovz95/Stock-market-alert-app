import streamlit as st
import pandas as pd
import os
import sys
import json

# Add the parent directory to the path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.utils import load_market_data
from src.utils.reference_data import get_country_display_name, get_exchange_display_name
from data_access.metadata_repository import fetch_stock_metadata_df

st.set_page_config(
    page_title="Stock Database",
    page_icon="📊",
    layout="wide",
)

# Load database filters
@st.cache_data(ttl=600)
def load_database_filters():
    """Load database filter options"""
    try:
        with open('database_filters.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading database filters: {e}")
        return {}

# Load stock database
@st.cache_data(ttl=600)  # Cache for 10 minutes
def load_stock_database():
    """Load and prepare stock database for display"""
    try:
        df = fetch_stock_metadata_df().copy()
        if df.empty:
            return df

        df.rename(columns={"symbol": "Symbol"}, inplace=True)

        if 'country' in df.columns:
            df['Country_Display'] = df['country'].apply(get_country_display_name)

        return df
    except Exception as e:
        st.error(f"Error loading stock database: {e}")
        return pd.DataFrame()

def main():
    st.title("📊 Stock Database")

    # Load data
    df = load_stock_database()
    filters = load_database_filters()

    if df.empty:
        st.error("No stock data available. Ensure the stock_metadata table is populated.")
        return

    st.markdown(f"Complete database of {len(df):,} symbols with industry classifications (auto-updated)")

    # Add Database Statistics Section
    st.markdown("---")
    st.subheader("📈 Database Overview Statistics")

    # Calculate statistics
    total_symbols = len(df)
    unique_exchanges = df['exchange'].nunique() if 'exchange' in df.columns else 0
    unique_countries = df['country'].nunique() if 'country' in df.columns else 0

    # Separate stocks and ETFs
    stocks_df = df[df['rbics_economy'].notna() & df['etf_issuer'].isna()] if 'rbics_economy' in df.columns and 'etf_issuer' in df.columns else pd.DataFrame()
    etfs_df = df[df['etf_issuer'].notna()] if 'etf_issuer' in df.columns else pd.DataFrame()

    # Display top-level metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Symbols", f"{total_symbols:,}")
    with col2:
        st.metric("Exchanges", f"{unique_exchanges:,}")
    with col3:
        st.metric("Countries", f"{unique_countries:,}")
    with col4:
        st.metric("Stocks", f"{len(stocks_df):,}")
    with col5:
        st.metric("ETFs", f"{len(etfs_df):,}")

    # RBICS Industry Breakdown
    if not stocks_df.empty and 'rbics_economy' in stocks_df.columns:
        st.markdown("### 🏭 RBICS Industry Breakdown (All Stocks)")

        # Count stocks by RBICS Economy - get ALL of them
        rbics_counts = stocks_df['rbics_economy'].value_counts()

        # Display in columns - split the list in half for two columns
        col1, col2 = st.columns(2)

        # Calculate midpoint for splitting
        midpoint = (len(rbics_counts) + 1) // 2

        with col1:
            st.write("**RBICS Economy Groups (Part 1):**")
            # First half of RBICS economies
            for economy, count in rbics_counts.iloc[:midpoint].items():
                pct = (count / len(stocks_df) * 100)
                st.write(f"• **{economy}**: {count:,} stocks ({pct:.1f}%)")

        with col2:
            st.write("**RBICS Economy Groups (Part 2):**")
            # Second half of RBICS economies
            for economy, count in rbics_counts.iloc[midpoint:].items():
                pct = (count / len(stocks_df) * 100)
                st.write(f"• **{economy}**: {count:,} stocks ({pct:.1f}%)")

        # Add summary metrics for RBICS
        st.write("")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total RBICS Categories", len(rbics_counts))
        with col2:
            st.metric("Largest Category", f"{rbics_counts.index[0]} ({rbics_counts.values[0]:,})")
        with col3:
            st.metric("Smallest Category", f"{rbics_counts.index[-1]} ({rbics_counts.values[-1]:,})")

    # ETF Breakdown
    if not etfs_df.empty:
        st.markdown("### 💼 ETF Analysis")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Asset Class breakdown
            if 'etf_asset_class' in etfs_df.columns:
                st.write("**Asset Classes:**")
                asset_classes = etfs_df['etf_asset_class'].value_counts().head(8)
                for asset_class, count in asset_classes.items():
                    if pd.notna(asset_class):
                        pct = (count / len(etfs_df) * 100)
                        st.write(f"• {asset_class}: {count:,} ({pct:.1f}%)")

        with col2:
            # ETF Focus breakdown
            if 'etf_focus' in etfs_df.columns:
                st.write("**ETF Focus:**")
                focuses = etfs_df['etf_focus'].value_counts().head(8)
                for focus, count in focuses.items():
                    if pd.notna(focus):
                        pct = (count / len(etfs_df) * 100)
                        st.write(f"• {focus}: {count:,} ({pct:.1f}%)")

        with col3:
            # ETF Niche breakdown
            if 'etf_niche' in etfs_df.columns:
                st.write("**ETF Niches:**")
                niches = etfs_df['etf_niche'].value_counts().head(8)
                for niche, count in niches.items():
                    if pd.notna(niche):
                        pct = (count / len(etfs_df) * 100)
                        st.write(f"• {niche}: {count:,} ({pct:.1f}%)")

    # Top Exchanges
    st.markdown("### 🏛️ Top Exchanges by Symbol Count")
    col1, col2 = st.columns(2)

    with col1:
        if 'exchange' in df.columns:
            top_exchanges = df['exchange'].value_counts().head(10)
            for exchange, count in top_exchanges.items():
                pct = (count / total_symbols * 100)
                st.write(f"• **{exchange}**: {count:,} symbols ({pct:.1f}%)")

    with col2:
        if 'country' in df.columns:
            st.write("**Top Countries:**")
            top_countries = df['country'].value_counts().head(10)
            for country, count in top_countries.items():
                pct = (count / total_symbols * 100)
                st.write(f"• {country}: {count:,} symbols ({pct:.1f}%)")

    st.markdown("---")

    # Create sidebar for filters
    with st.sidebar:
        st.header("🔍 Filters")

        # Country Filter
        st.subheader("🌍 Country")
        if 'country' in df.columns:
            countries = sorted(df['country'].dropna().unique())
            selected_countries = st.multiselect(
                "Select countries:",
                countries,
                help="Filter by one or more countries"
            )
        else:
            selected_countries = []

        # Exchange Filter
        st.subheader("🏛️ Exchange")
        if 'exchange' in df.columns:
            # Filter exchanges based on selected countries if applicable
            if selected_countries:
                available_exchanges = sorted(df[df['country'].isin(selected_countries)]['exchange'].dropna().unique())
            else:
                available_exchanges = sorted(df['exchange'].dropna().unique())
            selected_exchanges = st.multiselect(
                "Select exchanges:",
                available_exchanges,
                help="Filter by one or more exchanges"
            )
        else:
            selected_exchanges = []

        # Asset Type Filter
        st.subheader("📈 Asset Type")
        asset_types = ["All", "Stocks", "ETFs"]
        selected_asset_type = st.radio(
            "Select asset type:",
            asset_types,
            help="Filter by stocks or ETFs"
        )

        # Industry filters for stocks
        if selected_asset_type in ["All", "Stocks"]:
            st.subheader("🏭 Stock Industry Filters")

            # Economy filter
            if 'rbics_economy' in df.columns:
                economies = df[(df['rbics_economy'].notna()) & (~df['etf_issuer'].notna() if 'etf_issuer' in df.columns else True)]['rbics_economy'].dropna().unique()
                selected_economies = st.multiselect(
                    "Economy:",
                    sorted(economies),
                    help="Select one or more economies to filter stocks"
                )
            else:
                selected_economies = []

            # Sector filter
            if 'rbics_sector' in df.columns:
                # Filter sectors based on selected economies if possible
                if selected_economies:
                    available_sectors = df[(df['rbics_economy'].isin(selected_economies)) & (~df['etf_issuer'].notna() if 'etf_issuer' in df.columns else True)]['rbics_sector'].dropna().unique()
                else:
                    available_sectors = df[(df['rbics_economy'].notna()) & (~df['etf_issuer'].notna() if 'etf_issuer' in df.columns else True)]['rbics_sector'].dropna().unique()

                selected_sectors = st.multiselect(
                    "Sector:",
                    sorted(available_sectors),
                    help="Select one or more sectors to filter stocks"
                )
            else:
                selected_sectors = []

            # Subsector filter
            if 'rbics_subsector' in df.columns:
                # Filter subsectors based on selected sectors if possible
                if selected_sectors:
                    available_subsectors = df[(df['rbics_sector'].isin(selected_sectors)) & (~df['etf_issuer'].notna() if 'etf_issuer' in df.columns else True)]['rbics_subsector'].dropna().unique()
                else:
                    available_subsectors = df[(df['rbics_economy'].notna()) & (~df['etf_issuer'].notna() if 'etf_issuer' in df.columns else True)]['rbics_subsector'].dropna().unique()

                selected_subsectors = st.multiselect(
                    "Subsector:",
                    sorted(available_subsectors),
                    help="Select one or more subsectors to filter stocks"
                )
            else:
                selected_subsectors = []

            # Industry Group filter
            if 'rbics_industry_group' in df.columns:
                # Filter industry groups based on selected subsectors if possible
                if selected_subsectors:
                    available_groups = df[(df['rbics_subsector'].isin(selected_subsectors)) & (~df['etf_issuer'].notna() if 'etf_issuer' in df.columns else True)]['rbics_industry_group'].dropna().unique()
                else:
                    available_groups = df[(df['rbics_economy'].notna()) & (~df['etf_issuer'].notna() if 'etf_issuer' in df.columns else True)]['rbics_industry_group'].dropna().unique()

                selected_industry_groups = st.multiselect(
                    "Industry Group:",
                    sorted(available_groups),
                    help="Select one or more industry groups to filter stocks"
                )
            else:
                selected_industry_groups = []

            # Industry filter
            if 'rbics_industry' in df.columns:
                # Filter industries based on selected industry groups if possible
                if selected_industry_groups:
                    available_industries = df[(df['rbics_industry_group'].isin(selected_industry_groups)) & (~df['etf_issuer'].notna() if 'etf_issuer' in df.columns else True)]['rbics_industry'].dropna().unique()
                else:
                    available_industries = df[(df['rbics_economy'].notna()) & (~df['etf_issuer'].notna() if 'etf_issuer' in df.columns else True)]['rbics_industry'].dropna().unique()

                selected_industries = st.multiselect(
                    "Industry:",
                    sorted(available_industries),
                    help="Select one or more industries to filter stocks"
                )
            else:
                selected_industries = []

            # Subindustry filter
            if 'rbics_subindustry' in df.columns:
                # Filter subindustries based on selected industries if possible
                if selected_industries:
                    available_subindustries = df[(df['rbics_industry'].isin(selected_industries)) & (~df['etf_issuer'].notna() if 'etf_issuer' in df.columns else True)]['rbics_subindustry'].dropna().unique()
                else:
                    available_subindustries = df[(df['rbics_economy'].notna()) & (~df['etf_issuer'].notna() if 'etf_issuer' in df.columns else True)]['rbics_subindustry'].dropna().unique()

                selected_subindustries = st.multiselect(
                    "Subindustry:",
                    sorted(available_subindustries),
                    help="Select one or more subindustries to filter stocks"
                )
            else:
                selected_subindustries = []

        # Asset class filter for ETFs
        if selected_asset_type in ["All", "ETFs"]:
            st.subheader("💼 ETF Filters")

            # ETF Issuer filter (first as it's a primary filter)
            if 'etf_issuer' in df.columns:
                # Filter based on selected asset class if needed
                etf_df = df[df['etf_issuer'].notna()]
                issuers = etf_df['etf_issuer'].dropna().unique()
                selected_issuers = st.multiselect(
                    "ETF Issuer/Provider:",
                    sorted([i for i in issuers if i]),
                    help="Select one or more ETF issuers/providers (e.g., Vanguard, BlackRock)"
                )
            else:
                selected_issuers = []

            # Asset Class filter
            if 'asset_class' in df.columns:
                # Filter based on selected issuers
                if selected_issuers:
                    asset_classes = df[(df['etf_issuer'].isin(selected_issuers))]['asset_class'].dropna().unique()
                else:
                    asset_classes = df[df['etf_issuer'].notna()]['asset_class'].dropna().unique() if 'etf_issuer' in df.columns else []
                selected_asset_classes = st.multiselect(
                    "Asset Class:",
                    sorted([a for a in asset_classes if a]),
                    help="Select one or more asset classes (e.g., Equity, Fixed Income, Commodity)"
                )
            else:
                selected_asset_classes = []

            # ETF Focus filter
            if 'etf_focus' in df.columns:
                # Filter based on selected asset classes
                if selected_asset_classes:
                    focuses = df[(df['asset_class'].isin(selected_asset_classes)) & (df['etf_issuer'].notna())]['etf_focus'].dropna().unique()
                elif selected_issuers:
                    focuses = df[(df['etf_issuer'].isin(selected_issuers))]['etf_focus'].dropna().unique()
                else:
                    focuses = df[df['etf_issuer'].notna()]['etf_focus'].dropna().unique()
                selected_focuses = st.multiselect(
                    "ETF Focus:",
                    sorted([f for f in focuses if f]),
                    help="Select one or more ETF focus areas (e.g., Large Cap, Small Cap, Emerging Markets)"
                )
            else:
                selected_focuses = []

            # ETF Niche filter
            if 'etf_niche' in df.columns:
                # Filter based on selected focuses
                if selected_focuses:
                    niches = df[(df['etf_focus'].isin(selected_focuses)) & (df['etf_issuer'].notna())]['etf_niche'].dropna().unique()
                elif selected_asset_classes:
                    niches = df[(df['asset_class'].isin(selected_asset_classes)) & (df['etf_issuer'].notna())]['etf_niche'].dropna().unique()
                elif selected_issuers:
                    niches = df[(df['etf_issuer'].isin(selected_issuers))]['etf_niche'].dropna().unique()
                else:
                    niches = df[df['etf_issuer'].notna()]['etf_niche'].dropna().unique()
                selected_niches = st.multiselect(
                    "ETF Niche:",
                    sorted([n for n in niches if n]),
                    help="Select one or more ETF niches/specialties"
                )
            else:
                selected_niches = []

        # Clear filters button
        if st.button("🗑️ Clear All Filters"):
            st.rerun()

    # Display summary statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_symbols = filters.get('total_symbols', len(df))
        st.metric("Total Symbols", f"{total_symbols:,}")

    with col2:
        # Count stocks (have rbics_economy field but not etf_issuer)
        total_stocks = filters.get('total_stocks', ((df['rbics_economy'].notna()) & (~df['etf_issuer'].notna() if 'etf_issuer' in df.columns else True)).sum())
        st.metric("Stocks", f"{total_stocks:,}")

    with col3:
        # Count ETFs (have etf_issuer field)
        total_etfs = filters.get('total_etfs', df['etf_issuer'].notna().sum() if 'etf_issuer' in df.columns else 0)
        st.metric("ETFs", f"{total_etfs:,}")

    with col4:
        if selected_asset_type == "Stocks":
            sectors = len(filters.get('sectors', []))
            st.metric("Sectors", f"{sectors}")
        elif selected_asset_type == "ETFs":
            asset_classes = len(filters.get('asset_classes', []))
            st.metric("Asset Classes", f"{asset_classes}")
        else:
            st.metric("Categories", f"{len(filters.get('sectors', [])) + len(filters.get('asset_classes', []))}")

    st.divider()

    # Apply filters
    filtered_df = df.copy()

    # Apply country filter
    if selected_countries and 'country' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['country'].isin(selected_countries)]

    # Apply exchange filter
    if selected_exchanges and 'exchange' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['exchange'].isin(selected_exchanges)]

    # Apply asset type filter
    if selected_asset_type == "Stocks":
        # Filter by asset_type field
        filtered_df = filtered_df[filtered_df['asset_type'] == 'Stock']
    elif selected_asset_type == "ETFs":
        # Filter by asset_type field
        filtered_df = filtered_df[filtered_df['asset_type'] == 'ETF']

    # Apply stock filters
    if selected_asset_type in ["All", "Stocks"]:
        if 'selected_economies' in locals() and selected_economies and 'rbics_economy' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['rbics_economy'].isin(selected_economies)]

        if 'selected_sectors' in locals() and selected_sectors and 'rbics_sector' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['rbics_sector'].isin(selected_sectors)]

        if 'selected_subsectors' in locals() and selected_subsectors and 'rbics_subsector' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['rbics_subsector'].isin(selected_subsectors)]

        if 'selected_industry_groups' in locals() and selected_industry_groups and 'rbics_industry_group' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['rbics_industry_group'].isin(selected_industry_groups)]

        if 'selected_industries' in locals() and selected_industries and 'rbics_industry' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['rbics_industry'].isin(selected_industries)]

        if 'selected_subindustries' in locals() and selected_subindustries and 'rbics_subindustry' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['rbics_subindustry'].isin(selected_subindustries)]

    # Apply ETF filters
    if selected_asset_type in ["All", "ETFs"]:
        if 'selected_issuers' in locals() and selected_issuers and 'etf_issuer' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['etf_issuer'].isin(selected_issuers)]

        if 'selected_asset_classes' in locals() and selected_asset_classes and 'asset_class' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['asset_class'].isin(selected_asset_classes)]

        if 'selected_focuses' in locals() and selected_focuses and 'etf_focus' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['etf_focus'].isin(selected_focuses)]

        if 'selected_niches' in locals() and selected_niches and 'etf_niche' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['etf_niche'].isin(selected_niches)]

    # Display filtered results
    st.subheader(f"📋 Symbol List ({len(filtered_df):,} symbols)")

    # Search functionality
    search_term = st.text_input(
        "🔍 Search symbols:",
        placeholder="Enter ticker symbol, company name, or any text...",
        help="Search across all columns"
    )

    if search_term:
        # Create a mask for search across all string columns
        search_mask = pd.DataFrame([filtered_df[col].astype(str).str.contains(search_term, case=False, na=False)
                                  for col in filtered_df.columns]).any()
        filtered_df = filtered_df[search_mask]

    # Prepare display columns based on asset type
    display_columns = ['Symbol', 'name', 'asset_type']

    if 'exchange' in filtered_df.columns:
        display_columns.append('exchange')
    if 'country' in filtered_df.columns:
        display_columns.append('country')
    if 'isin' in filtered_df.columns:
        display_columns.append('isin')

    # Add relevant columns based on what's being displayed
    if selected_asset_type in ["All", "Stocks"]:
        for col in ['rbics_economy', 'rbics_sector', 'rbics_subsector', 'rbics_industry_group', 'rbics_industry', 'rbics_subindustry']:
            if col in filtered_df.columns:
                display_columns.append(col)

    if selected_asset_type in ["All", "ETFs"]:
        for col in ['etf_issuer', 'asset_class', 'etf_focus', 'etf_niche', 'expense_ratio', 'aum']:
            if col in filtered_df.columns:
                display_columns.append(col)

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
            file_name="stock_database_export.csv",
            mime="text/csv"
        )

        # Configure column display names
        column_config = {
            "Symbol": st.column_config.TextColumn("Symbol", width="small"),
            "name": st.column_config.TextColumn("Name", width="medium"),
            "asset_type": st.column_config.TextColumn("Type", width="small"),
            "exchange": st.column_config.TextColumn("Exchange", width="small"),
            "country": st.column_config.TextColumn("Country", width="small"),
            "isin": st.column_config.TextColumn("ISIN", width="small"),
            "rbics_economy": st.column_config.TextColumn("Economy", width="medium"),
            "rbics_sector": st.column_config.TextColumn("Sector", width="medium"),
            "rbics_subsector": st.column_config.TextColumn("Subsector", width="medium"),
            "rbics_industry_group": st.column_config.TextColumn("Industry Group", width="medium"),
            "rbics_industry": st.column_config.TextColumn("Industry", width="medium"),
            "rbics_subindustry": st.column_config.TextColumn("Subindustry", width="medium"),
            "etf_issuer": st.column_config.TextColumn("ETF Issuer", width="small"),
            "asset_class": st.column_config.TextColumn("Asset Class", width="medium"),
            "etf_focus": st.column_config.TextColumn("ETF Focus", width="medium"),
            "etf_niche": st.column_config.TextColumn("ETF Niche", width="medium"),
            "expense_ratio": st.column_config.NumberColumn("Expense Ratio", width="small", format="%.3f"),
            "aum": st.column_config.NumberColumn("AUM", width="small", format="$%.0f")
        }

        # Display the table with more rows visible
        st.dataframe(
            display_df,
            use_container_width=True,
            height=2400,  # Increased height to show ~100 rows
            column_config=column_config
        )

        # Show detailed statistics
        st.divider()
        st.subheader("📈 Statistics")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Asset Type Distribution:**")
            # Count stocks vs ETFs based on classification fields
            if 'etf_issuer' in filtered_df.columns:
                etf_count = filtered_df['etf_issuer'].notna().sum()
                stock_count = len(filtered_df) - etf_count
            else:
                stock_count = filtered_df['rbics_economy'].notna().sum()
                etf_count = 0
            st.write(f"• Stocks: {stock_count:,}")
            st.write(f"• ETFs: {etf_count:,}")

        with col2:
            if selected_asset_type in ["All", "Stocks"] and 'rbics_sector' in filtered_df.columns:
                st.write("**Top Stock Sectors:**")
                stock_df = filtered_df[(filtered_df['rbics_economy'].notna()) & (~filtered_df['etf_issuer'].notna() if 'etf_issuer' in filtered_df.columns else True)]
                if not stock_df.empty:
                    sector_counts = stock_df['rbics_sector'].value_counts().head(5)
                    for sector, count in sector_counts.items():
                        if sector != 'Other':
                            st.write(f"• {sector}: {count:,}")
            elif selected_asset_type == "ETFs" and 'asset_class' in filtered_df.columns:
                st.write("**ETF Asset Classes:**")
                etf_df = filtered_df[filtered_df['etf_issuer'].notna() if 'etf_issuer' in filtered_df.columns else pd.Series([False]*len(filtered_df))]
                if not etf_df.empty:
                    asset_counts = etf_df['asset_class'].value_counts()
                    for asset_class, count in asset_counts.items():
                        if asset_class != 'Other':
                            st.write(f"• {asset_class}: {count:,}")

        with col3:
            if selected_asset_type in ["All", "Stocks"] and 'rbics_economy' in filtered_df.columns:
                st.write("**Top Economies:**")
                stock_df = filtered_df[(filtered_df['rbics_economy'].notna()) & (~filtered_df['etf_issuer'].notna() if 'etf_issuer' in filtered_df.columns else True)]
                if not stock_df.empty:
                    economy_counts = stock_df['rbics_economy'].value_counts().head(5)
                    for economy, count in economy_counts.items():
                        if economy != 'Other':
                            st.write(f"• {economy}: {count:,}")
            elif selected_asset_type == "ETFs" and 'etf_issuer' in filtered_df.columns:
                st.write("**Top ETF Issuers:**")
                etf_df = filtered_df[filtered_df['etf_issuer'].notna()]
                if not etf_df.empty:
                    provider_counts = etf_df['etf_issuer'].value_counts().head(5)
                    for provider, count in provider_counts.items():
                        if provider:
                            st.write(f"• {provider}: {count:,}")

    else:
        st.warning("No symbols match the selected filters. Try adjusting your search criteria.")

    # Show exchange breakdown
    st.divider()
    st.subheader("🏛️ Exchange Breakdown")
    st.write("Combined stock and ETF counts by exchange:")

    # Get counts by exchange
    exchange_counts = df['exchange'].value_counts().sort_index()

    # Display in columns for better layout
    col1, col2, col3 = st.columns(3)

    exchanges_list = exchange_counts.index.tolist()
    third = len(exchanges_list) // 3

    with col1:
        for exchange in exchanges_list[:third+1]:
            count = exchange_counts[exchange]
            st.write(f"• **{exchange}**: {count:,}")

    with col2:
        for exchange in exchanges_list[third+1:2*third+1]:
            count = exchange_counts[exchange]
            st.write(f"• **{exchange}**: {count:,}")

    with col3:
        for exchange in exchanges_list[2*third+1:]:
            count = exchange_counts[exchange]
            st.write(f"• **{exchange}**: {count:,}")

    st.write(f"\n**Total Exchanges: {len(exchange_counts)}**")

if __name__ == "__main__":
    main()

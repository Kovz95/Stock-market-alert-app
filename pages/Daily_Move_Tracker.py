import math

import numpy as np
import pandas as pd
import streamlit as st

from utils import load_market_data
from db_config import db_config

st.set_page_config(
    page_title="Daily Move Tracker",
    page_icon="📊",
    layout="wide",
)

TABLE_NAME = "daily_move_stats"
LOOKBACK_DAYS = 30
PAGE_SIZE = 50


@st.cache_data(ttl=300)
def load_stats_and_metadata():
    try:
        with db_config.connection(role="prices") as conn:
            stats = pd.read_sql_query(
                f"SELECT * FROM {TABLE_NAME}",
                conn,
                parse_dates=["date"],
            )
    except Exception:
        stats = pd.DataFrame()

    try:
        market = load_market_data()
    except Exception:
        market = pd.DataFrame()

    if market is None:
        market = pd.DataFrame()
    else:
        market = market.copy()
        if "Symbol" in market.columns:
            market.rename(columns={"Symbol": "ticker"}, inplace=True)
        market = market.loc[:, ~market.columns.duplicated()]
        if "ticker" not in market.columns:
            market["ticker"] = market.index.astype(str)
        market.drop_duplicates(subset="ticker", inplace=True)

        if "ETF_Asset_Class" in market.columns and "Asset_Class" not in market.columns:
            market["Asset_Class"] = market["ETF_Asset_Class"]
        if "Asset_Type" not in market.columns:
            market["Asset_Type"] = "Stock"

        display_map = {
            "Name": "Name",
            "Asset_Type": "Asset Type",
            "Country": "Country",
            "Exchange": "Exchange",
            "RBICS_Economy": "Economy",
            "RBICS_Sector": "Sector",
            "RBICS_Subsector": "Subsector",
            "RBICS_Industry_Group": "Industry Group",
            "RBICS_Industry": "Industry",
            "RBICS_Subindustry": "Subindustry",
            "ETF_Issuer": "ETF Issuer",
            "Asset_Class": "Asset Class",
            "ETF_Focus": "ETF Focus",
            "ETF_Niche": "ETF Niche",
        }
        for source, target in display_map.items():
            if source in market.columns:
                market[target] = market[source]
            else:
                market[target] = ""

    return stats, market


def build_highlighter(sigma_lookup, direction_lookup, date_columns):
    def highlight(row):
        ticker = row.name
        styles = []
        for date in date_columns:
            sigma = sigma_lookup.get((ticker, date), np.nan)
            direction = direction_lookup.get((ticker, date), None)
            value = row.get(date)

            if pd.isna(value) or pd.isna(sigma):
                styles.append("")
            elif sigma >= 2:
                styles.append("background-color: #ccffcc;" if direction == "up" else "background-color: #ffcccc;")
            elif sigma == 1:
                styles.append("background-color: #e6ffe6;" if direction == "up" else "background-color: #ffe6e6;")
            else:
                styles.append("")
        return styles

    return highlight


def main():
    st.title("📊 Daily Move Tracker")
    st.caption(
        "Track daily percent changes versus long-term volatility. Cells are highlighted when moves exceed ±1σ (light) or ±2σ (bold)."
    )

    stats_df, market_df = load_stats_and_metadata()
    if stats_df.empty:
        st.info(
            "Daily move statistics are not available yet. They will populate after the next daily scheduler run."
        )
        return

    if market_df.empty:
        st.warning("Market metadata is unavailable; filters cannot be applied.")
        return

    stats_df["date"] = pd.to_datetime(stats_df["date"])
    combined_df = stats_df.merge(market_df, on="ticker", how="left")

    default_fill = {
        "Name": "",
        "Asset_Type": "Stock",
        "Asset Type": "Stock",
        "Country": "Unknown",
        "Exchange": "Unknown",
        "RBICS_Economy": "",
        "RBICS_Sector": "",
        "RBICS_Subsector": "",
        "RBICS_Industry_Group": "",
        "RBICS_Industry": "",
        "RBICS_Subindustry": "",
        "ETF_Issuer": "",
        "Asset_Class": "",
        "Asset Class": "",
        "ETF_Focus": "",
        "ETF_Niche": "",
        "direction": "",
    }
    for col, default in default_fill.items():
        if col not in combined_df.columns:
            combined_df[col] = default
        else:
            combined_df[col] = combined_df[col].fillna(default)

    sidebar = st.sidebar
    sidebar.header("Filters")

    if sidebar.button("Refresh data", help="Clear cached data and reload the latest daily moves"):
        load_stats_and_metadata.clear()
        st.rerun()

    asset_type_options = ["All", "Stocks", "ETFs", "Pairs/Ratios"]
    selected_asset_type = sidebar.selectbox("Asset Type", asset_type_options, index=0)

    filter_base = market_df.copy()
    if selected_asset_type == "Stocks":
        filter_base = filter_base[filter_base["Asset_Type"] == "Stock"]
    elif selected_asset_type == "ETFs":
        filter_base = filter_base[filter_base["Asset_Type"] == "ETF"]
    elif selected_asset_type == "Pairs/Ratios":
        filter_base = filter_base[filter_base["Asset_Type"] == "Pair"]

    available_countries = sorted(filter_base["Country"].dropna().unique())
    selected_countries = sidebar.multiselect("Country", available_countries, default=[])

    country_filtered = filter_base
    if selected_countries:
        country_filtered = country_filtered[country_filtered["Country"].isin(selected_countries)]

    available_exchanges = sorted(country_filtered["Exchange"].dropna().unique())
    selected_exchanges = sidebar.multiselect("Exchange", available_exchanges, default=[])

    exchange_filtered = country_filtered
    if selected_exchanges:
        exchange_filtered = exchange_filtered[exchange_filtered["Exchange"].isin(selected_exchanges)]

    timeframe_mapping = {"Daily": "1d", "Weekly": "1wk"}
    timeframe_display = list(timeframe_mapping.keys())
    selected_timeframe_display = sidebar.multiselect(
        "Timeframe",
        timeframe_display,
        default=["Daily"],
    )
    selected_timeframes = [timeframe_mapping[tf] for tf in selected_timeframe_display]
    if selected_timeframes and "1d" not in selected_timeframes:
        sidebar.info("The daily move tracker currently supports Daily data only.")
        return

    selected_economies = []
    selected_sectors = []
    selected_subsectors = []
    selected_industry_groups = []
    selected_industries = []
    selected_subindustries = []

    if selected_asset_type in ["All", "Stocks"]:
        stocks_only = exchange_filtered[exchange_filtered["Asset_Type"] == "Stock"]
        if not stocks_only.empty:
            available_economies = sorted(stocks_only["RBICS_Economy"].dropna().unique())
            selected_economies = sidebar.multiselect("Economy", available_economies, default=[])
            if selected_economies:
                stocks_only = stocks_only[stocks_only["RBICS_Economy"].isin(selected_economies)]

            available_sectors = sorted(stocks_only["RBICS_Sector"].dropna().unique())
            selected_sectors = sidebar.multiselect("Sector", available_sectors, default=[])
            if selected_sectors:
                stocks_only = stocks_only[stocks_only["RBICS_Sector"].isin(selected_sectors)]

            available_subsectors = sorted(stocks_only["RBICS_Subsector"].dropna().unique())
            selected_subsectors = sidebar.multiselect("Subsector", available_subsectors, default=[])
            if selected_subsectors:
                stocks_only = stocks_only[stocks_only["RBICS_Subsector"].isin(selected_subsectors)]

            available_industry_groups = sorted(stocks_only["RBICS_Industry_Group"].dropna().unique())
            selected_industry_groups = sidebar.multiselect("Industry Group", available_industry_groups, default=[])
            if selected_industry_groups:
                stocks_only = stocks_only[stocks_only["RBICS_Industry_Group"].isin(selected_industry_groups)]

            available_industries = sorted(stocks_only["RBICS_Industry"].dropna().unique())
            selected_industries = sidebar.multiselect("Industry", available_industries, default=[])
            if selected_industries:
                stocks_only = stocks_only[stocks_only["RBICS_Industry"].isin(selected_industries)]

            available_subindustries = sorted(stocks_only["RBICS_Subindustry"].dropna().unique())
            selected_subindustries = sidebar.multiselect("Subindustry", available_subindustries, default=[])
        else:
            if selected_asset_type == "Stocks":
                sidebar.info("No stock symbols match the current asset type/location filters.")

    selected_issuers = []
    selected_asset_classes = []
    selected_focuses = []
    selected_niches = []

    if selected_asset_type in ["All", "ETFs"]:
        etfs_only = exchange_filtered[exchange_filtered["Asset_Type"] == "ETF"].copy()
        if not etfs_only.empty:
            if "Asset_Class" not in etfs_only.columns and "ETF_Asset_Class" in etfs_only.columns:
                etfs_only["Asset_Class"] = etfs_only["ETF_Asset_Class"]

            available_issuers = sorted(etfs_only["ETF_Issuer"].dropna().unique())
            selected_issuers = sidebar.multiselect("ETF Issuer", available_issuers, default=[])
            if selected_issuers:
                etfs_only = etfs_only[etfs_only["ETF_Issuer"].isin(selected_issuers)]

            available_asset_classes = sorted(etfs_only["Asset_Class"].dropna().unique()) if "Asset_Class" in etfs_only.columns else []
            selected_asset_classes = sidebar.multiselect("Asset Class", available_asset_classes, default=[])
            if selected_asset_classes and "Asset_Class" in etfs_only.columns:
                etfs_only = etfs_only[etfs_only["Asset_Class"].isin(selected_asset_classes)]

            available_focuses = sorted(etfs_only["ETF_Focus"].dropna().unique()) if "ETF_Focus" in etfs_only.columns else []
            selected_focuses = sidebar.multiselect("ETF Focus", available_focuses, default=[])
            if selected_focuses and "ETF_Focus" in etfs_only.columns:
                etfs_only = etfs_only[etfs_only["ETF_Focus"].isin(selected_focuses)]

            available_niches = sorted(etfs_only["ETF_Niche"].dropna().unique()) if "ETF_Niche" in etfs_only.columns else []
            selected_niches = sidebar.multiselect("ETF Niche", available_niches, default=[])
        else:
            if selected_asset_type == "ETFs":
                sidebar.info("No ETF symbols match the current asset type/location filters.")

    ticker_search = sidebar.text_input("Search ticker", "", placeholder="e.g., AAPL").strip().upper()
    name_search = sidebar.text_input("Search company name", "", placeholder="e.g., NVIDIA").strip().lower()

    mask = pd.Series(True, index=combined_df.index)
    if selected_asset_type == "Stocks":
        mask &= combined_df["Asset_Type"].eq("Stock")
    elif selected_asset_type == "ETFs":
        mask &= combined_df["Asset_Type"].eq("ETF")
    elif selected_asset_type == "Pairs/Ratios":
        mask &= combined_df["Asset_Type"].eq("Pair")

    if selected_countries:
        mask &= combined_df["Country"].isin(selected_countries)
    if selected_exchanges:
        mask &= combined_df["Exchange"].isin(selected_exchanges)

    stock_mask = combined_df["Asset_Type"].eq("Stock")
    if selected_economies:
        mask &= (~stock_mask) | combined_df["RBICS_Economy"].isin(selected_economies)
    if selected_sectors:
        mask &= (~stock_mask) | combined_df["RBICS_Sector"].isin(selected_sectors)
    if selected_subsectors:
        mask &= (~stock_mask) | combined_df["RBICS_Subsector"].isin(selected_subsectors)
    if selected_industry_groups:
        mask &= (~stock_mask) | combined_df["RBICS_Industry_Group"].isin(selected_industry_groups)
    if selected_industries:
        mask &= (~stock_mask) | combined_df["RBICS_Industry"].isin(selected_industries)
    if selected_subindustries:
        mask &= (~stock_mask) | combined_df["RBICS_Subindustry"].isin(selected_subindustries)

    etf_mask = combined_df["Asset_Type"].eq("ETF")
    if selected_issuers:
        mask &= (~etf_mask) | combined_df["ETF_Issuer"].isin(selected_issuers)
    if selected_asset_classes:
        mask &= (~etf_mask) | combined_df["Asset_Class"].isin(selected_asset_classes)
    if selected_focuses:
        mask &= (~etf_mask) | combined_df["ETF_Focus"].isin(selected_focuses)
    if selected_niches:
        mask &= (~etf_mask) | combined_df["ETF_Niche"].isin(selected_niches)

    filtered = combined_df[mask].copy()

    if ticker_search:
        filtered = filtered[filtered["ticker"].str.contains(ticker_search, na=False)]
    if name_search:
        filtered = filtered[filtered["Name"].str.lower().str.contains(name_search, na=False)]

    if filtered.empty:
        st.warning("No rows match the current filters.")
        return

    recent_dates = sorted(filtered["date"].unique())[-LOOKBACK_DAYS:]
    if not recent_dates:
        st.warning("Not enough historical data to display.")
        return
    date_labels = [d.strftime("%Y-%m-%d") for d in reversed(recent_dates)]

    filtered = filtered[filtered["date"].isin(recent_dates)].copy()
    filtered["date_str"] = filtered["date"].dt.strftime("%Y-%m-%d")

    pct_pivot = filtered.pivot_table(index="ticker", columns="date_str", values="pct_change", aggfunc="last")
    sigma_pivot = filtered.pivot_table(index="ticker", columns="date_str", values="sigma_level", aggfunc="last")
    direction_pivot = filtered.pivot_table(index="ticker", columns="date_str", values="direction", aggfunc="last")
    zscore_pivot = filtered.pivot_table(index="ticker", columns="date_str", values="zscore", aggfunc="last")

    pct_pivot = pct_pivot.reindex(columns=date_labels)
    sigma_pivot = sigma_pivot.reindex(columns=date_labels)
    direction_pivot = direction_pivot.reindex(columns=date_labels)
    zscore_pivot = zscore_pivot.reindex(columns=date_labels)

    meta_columns = [col for col in ["Name", "Asset Type", "Economy", "Exchange", "Country", "Asset Class"] if col in filtered.columns]
    meta_fields = filtered[["ticker"] + meta_columns].drop_duplicates()
    meta_fields.set_index("ticker", inplace=True)

    latest_date_label = date_labels[0]
    pct_pivot = pct_pivot.reindex(meta_fields.index)
    sigma_pivot = sigma_pivot.reindex(meta_fields.index)
    direction_pivot = direction_pivot.reindex(meta_fields.index)
    zscore_pivot = zscore_pivot.reindex(meta_fields.index)

    pct_pivot.sort_values(
        by=latest_date_label,
        key=lambda col: np.abs(col.fillna(0)),
        ascending=False,
        inplace=True,
    )

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    with summary_col1:
        st.metric("Symbols ≥ 2σ", int((filtered["sigma_level"] >= 2).sum()))
    with summary_col2:
        st.metric("Symbols ≥ 1σ", int((filtered["sigma_level"] >= 1).sum()))
    with summary_col3:
        st.metric("Up Moves", int((filtered["direction"] == "up").sum()))
    with summary_col4:
        st.metric("Down Moves", int((filtered["direction"] == "down").sum()))

    display_df = meta_fields.join(pct_pivot)
    display_df.reset_index(inplace=True)
    if "index" in display_df.columns:
        display_df.rename(columns={"index": "Ticker"}, inplace=True)
    elif "ticker" in display_df.columns:
        display_df.rename(columns={"ticker": "Ticker"}, inplace=True)

    total_rows = display_df.shape[0]
    total_pages = max(1, math.ceil(total_rows / PAGE_SIZE))
    col_page, col_total = st.columns([1, 3])
    with col_page:
        page_number = st.number_input(
            "Row page",
            min_value=1,
            max_value=total_pages,
            value=1,
            step=1,
            help="50 rows per page",
        )
    with col_total:
        st.write(f"{total_rows} tickers match the filters.")

    start_idx = (page_number - 1) * PAGE_SIZE
    end_idx = start_idx + PAGE_SIZE
    page_df = display_df.iloc[start_idx:end_idx].copy()
    page_df.set_index("Ticker", inplace=True)

    sigma_lookup = sigma_pivot.stack().to_dict()
    direction_lookup = direction_pivot.stack().to_dict()

    style_subset = [col for col in page_df.columns if col in date_labels]
    formatter = {col: "{:+.2f}%" for col in style_subset}

    if not style_subset:
        st.info("No recent observations to display for the selected filters.")
        return

    highlighter = build_highlighter(sigma_lookup, direction_lookup, style_subset)
    styled = page_df.style.apply(highlighter, axis=1, subset=style_subset).format(formatter)

    st.dataframe(styled, use_container_width=True)

    show_zscores = sidebar.checkbox("Show Z-score table", value=True)
    if show_zscores:
        zscore_page = zscore_pivot.loc[page_df.index].copy()
        zscore_formatter = {col: "{:+.2f}" for col in zscore_page.columns}
        st.markdown("**Per-day Z-Scores**")
        st.dataframe(
            zscore_page.style.apply(highlighter, axis=1, subset=style_subset).format(zscore_formatter),
            use_container_width=True,
        )

    csv_data = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data (CSV)",
        data=csv_data,
        file_name="daily_move_tracker_filtered.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()

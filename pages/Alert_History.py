"""
Alert History Lookup Page
Allows users to search for a company by name or ticker and view alert trigger history
"""

import streamlit as st
import json
from datetime import datetime, timedelta

import pandas as pd

from src.data_access.db_config import db_config
from src.data_access.metadata_repository import (
    fetch_alerts_list,
    fetch_portfolios,
    fetch_stock_metadata_map,
)

st.set_page_config(page_title="Alert History", page_icon="🔍", layout="wide")

st.title("🔍 Alert History Lookup")
st.markdown("Search for alerts with advanced filters or look up specific companies")

# Load necessary data
@st.cache_data
def load_stock_database():
    """Load the main stock database"""
    try:
        return fetch_stock_metadata_map()
    except:
        return {}

@st.cache_data
def load_alerts():
    """Load all alerts"""
    try:
        return fetch_alerts_list()
    except:
        return []

@st.cache_data
def load_portfolios():
    """Load all portfolios"""
    try:
        return fetch_portfolios()
    except:
        return {}

def get_alert_history_from_audit_log(ticker):
    """Get alert trigger history from audit log"""
    try:
        conn = db_config.get_connection(role="alerts")

        # Query for triggered alerts
        query = """
        SELECT
            alert_id,
            timestamp,
            alert_triggered as condition_met,
            trigger_reason as condition_details,
            execution_time_ms,
            evaluation_type as trigger_source
        FROM alert_audits
        WHERE ticker = %s
        AND alert_triggered = TRUE
        ORDER BY timestamp DESC
        LIMIT 50
        """

        df = pd.read_sql_query(query, conn, params=(ticker,))
        db_config.close_connection(conn)

        if not df.empty:
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            df['time'] = df['timestamp'].dt.strftime('%H:%M:%S')

            # Map alert_id to alert_name
            alerts = load_alerts()
            alert_id_to_name = {a.get('alert_id'): a.get('name', 'Unknown Alert') for a in alerts}
            df['alert_name'] = df['alert_id'].map(lambda x: alert_id_to_name.get(x, f'Alert {x[:8] if len(x) > 8 else x}...'))

        return df
    except Exception as e:
        st.error(f"Error accessing audit log: {e}")
        return pd.DataFrame()

def get_all_triggered_history(ticker):
    """Get all evaluation history, not just triggers"""
    try:
        conn = db_config.get_connection(role="alerts")

        # Query for all evaluations
        query = """
        SELECT
            alert_id,
            timestamp,
            alert_triggered as condition_met,
            trigger_reason as condition_details,
            execution_time_ms,
            evaluation_type as trigger_source
        FROM alert_audits
        WHERE ticker = %s
        ORDER BY timestamp DESC
        LIMIT 100
        """

        df = pd.read_sql_query(query, conn, params=(ticker,))
        db_config.close_connection(conn)

        if not df.empty:
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            df['time'] = df['timestamp'].dt.strftime('%H:%M:%S')

            # Map alert_id to alert_name
            alerts = load_alerts()
            alert_id_to_name = {a.get('alert_id'): a.get('name', 'Unknown Alert') for a in alerts}
            df['alert_name'] = df['alert_id'].map(lambda x: alert_id_to_name.get(x, f'Alert {x[:8] if len(x) > 8 else x}...'))

        return df
    except Exception as e:
        return pd.DataFrame()

def search_companies(search_term, stock_db):
    """Search for companies by name or ticker"""
    search_term = search_term.upper()
    results = []

    for ticker, data in stock_db.items():
        name = data.get('name', '').upper()
        if search_term in ticker.upper() or search_term in name:
            results.append({
                'ticker': ticker,
                'name': data.get('name', 'Unknown'),
                'exchange': data.get('exchange', 'Unknown'),
                'type': data.get('type', 'Stock')
            })

    # Sort by relevance (exact ticker match first, then ticker contains, then name contains)
    results.sort(key=lambda x: (
        x['ticker'].upper() != search_term,  # Exact match gets priority
        search_term not in x['ticker'].upper(),  # Ticker contains
        len(x['ticker'])  # Shorter tickers first
    ))

    return results[:20]  # Limit to 20 results

def get_portfolio_tickers(portfolios):
    """Get all tickers from all portfolios"""
    all_tickers = set()
    portfolio_map = {}  # Map ticker to list of portfolio names

    for portfolio_id, portfolio_data in portfolios.items():
        portfolio_name = portfolio_data.get('name', '')
        for stock in portfolio_data.get('stocks', []):
            ticker = stock.get('symbol')
            if ticker:
                all_tickers.add(ticker)
                if ticker not in portfolio_map:
                    portfolio_map[ticker] = []
                portfolio_map[ticker].append(portfolio_name)

    return all_tickers, portfolio_map

def format_condition(condition_details):
    """Format condition details for display"""
    if not condition_details:
        return "N/A"

    # Try to parse JSON if it's a string
    if isinstance(condition_details, str):
        try:
            details = json.loads(condition_details)
            if isinstance(details, dict):
                return '\n'.join([f"• {k}: {v}" for k, v in details.items()])
            else:
                return condition_details
        except:
            return condition_details
    return str(condition_details)

def filter_alerts(alerts, filters):
    """Filter alerts based on selected criteria"""
    filtered = alerts

    # Filter by exchange
    if filters.get('exchanges') and 'All' not in filters['exchanges']:
        filtered = [a for a in filtered if a.get('exchange', '') in filters['exchanges']]

    # Filter by condition
    if filters.get('conditions') and 'All' not in filters['conditions']:
        filtered = [a for a in filtered if any(
            cond.get('conditions', '') in filters['conditions']
            for cond in a.get('conditions', [])
        )]

    # Filter by economy (sector)
    if filters.get('economies') and 'All' not in filters['economies']:
        filtered = [a for a in filtered if a.get('rbics_economy', 'Unknown') in filters['economies']]

    # Filter by action (buy/sell)
    if filters.get('actions') and 'All' not in filters['actions']:
        filtered = [a for a in filtered if a.get('action', '') in filters['actions']]

    # Filter by portfolio membership
    if filters.get('in_any_portfolio'):
        portfolio_tickers = filters.get('portfolio_tickers', set())
        filtered = [a for a in filtered if a.get('ticker', '') in portfolio_tickers]

    # Filter by specific portfolios
    if filters.get('specific_portfolios'):
        portfolio_map = filters.get('portfolio_map', {})
        filtered = [a for a in filtered if any(
            portfolio in portfolio_map.get(a.get('ticker', ''), [])
            for portfolio in filters['specific_portfolios']
        )]

    return filtered

# Main app
stock_db = load_stock_database()
alerts = load_alerts()
portfolios = load_portfolios()
portfolio_tickers, portfolio_map = get_portfolio_tickers(portfolios)

# Enrich alerts with economy data from stock database
for alert in alerts:
    ticker = alert.get('ticker')
    if ticker and ticker in stock_db:
        alert['rbics_economy'] = stock_db[ticker].get('rbics_economy', 'Unknown')
    else:
        alert['rbics_economy'] = 'Unknown'

# Get unique values for filters
all_exchanges = sorted(list(set(a.get('exchange', '') for a in alerts if a.get('exchange'))))
all_economies = sorted(list(set(a.get('rbics_economy', 'Unknown') for a in alerts if a.get('rbics_economy'))))
all_actions = sorted(list(set(a.get('action', '') for a in alerts if a.get('action'))))

# Get all unique conditions from alerts
all_conditions_set = set()
for alert in alerts:
    for cond in alert.get('conditions', []):
        condition_text = cond.get('conditions', '')
        if condition_text:
            all_conditions_set.add(condition_text)
all_conditions = sorted(list(all_conditions_set))

# Get portfolio names
portfolio_names = [p.get('name', '') for p in portfolios.values() if p.get('name')]

# Add tabs for different search modes
# Check if we should show a message about switching tabs
if st.session_state.get('switch_to_search_tab', False):
    st.info("⬆️ Click on the **'🔍 Search by Ticker/Company'** tab above to view the full history")
    st.session_state['switch_to_search_tab'] = False

# Create tabs
tab1, tab2 = st.tabs(["🔍 Search by Ticker/Company", "📊 Browse with Filters"])

with tab2:
    st.markdown("### Filter Alerts")

    # Create filter columns
    col1, col2, col3 = st.columns(3)

    with col1:
        selected_exchanges = st.multiselect(
            "Exchange",
            options=['All'] + all_exchanges,
            default=['All'],
            help="Filter by exchange"
        )

        selected_economies = st.multiselect(
            "Economy",
            options=['All'] + all_economies,
            default=['All'],
            help="Filter by economy/sector (Technology, Healthcare, etc.)"
        )

    with col2:
        selected_actions = st.multiselect(
            "Action",
            options=['All'] + all_actions,
            default=['All'],
            help="Filter by buy/sell action"
        )

        in_any_portfolio = st.checkbox(
            "In Any Portfolio",
            value=False,
            help="Show only alerts for stocks in any portfolio"
        )

    with col3:
        selected_portfolios = st.multiselect(
            "Specific Portfolios",
            options=portfolio_names,
            default=[],
            help="Filter by specific portfolios (e.g., CBRE, Kovich Capital)"
        )

        triggered_filter = st.selectbox(
            "Trigger Status",
            options=['All', 'Triggered Only', 'Not Triggered'],
            help="Filter by whether alerts have been triggered"
        )

    # Condition filter (full width)
    selected_conditions = st.multiselect(
        "Conditions",
        options=['All'] + all_conditions[:50],  # Limit to first 50 for performance
        default=['All'],
        help="Filter by specific indicator conditions"
    )

    # Date range filter
    st.markdown("### Date Range Filter")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=30),
            help="Filter alerts triggered on or after this date"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            help="Filter alerts triggered on or before this date"
        )

    # Apply filters button
    apply_filters = st.button("📊 Apply Filters", type="primary", use_container_width=True)

    if apply_filters or selected_portfolios or in_any_portfolio or (selected_exchanges and 'All' not in selected_exchanges):
        # Convert dates to date strings for SQL query (using date() function for comparison)
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')

        # Query audit log for triggered alerts within date range
        conn = db_config.get_connection(role="alerts")

        if triggered_filter == 'Triggered Only':
            # Get all triggered alerts with their details
            query = """
            SELECT
                alert_id,
                ticker,
                timestamp,
                trigger_reason
            FROM alert_audits
            WHERE alert_triggered = TRUE
            AND timestamp BETWEEN %s AND %s
            ORDER BY timestamp DESC
            """
            triggered_events_df = pd.read_sql_query(query, conn, params=(start_date, end_date))

            # Get unique alert_ids that triggered
            triggered_alert_ids = set(triggered_events_df['alert_id'].unique())

            # Filter alerts to only those that triggered
            filtered_alerts = [a for a in alerts if a.get('alert_id') in triggered_alert_ids]

        elif triggered_filter == 'Not Triggered':
            # Get alerts that triggered within date range
            query = """
            SELECT DISTINCT alert_id
            FROM alert_audits
            WHERE alert_triggered = TRUE
            AND timestamp BETWEEN %s AND %s
            """
            triggered_df = pd.read_sql_query(query, conn, params=(start_date, end_date))
            triggered_alert_ids = set(triggered_df['alert_id'].unique())

            # Show all alerts that did NOT trigger
            filtered_alerts = [a for a in alerts if a.get('alert_id') not in triggered_alert_ids]
        else:
            # Show all alerts
            filtered_alerts = alerts.copy()

        db_config.close_connection(conn)

        # Now apply other filters
        filters = {
            'exchanges': selected_exchanges,
            'conditions': selected_conditions,
            'economies': selected_economies,
            'actions': selected_actions,
            'in_any_portfolio': in_any_portfolio,
            'specific_portfolios': selected_portfolios,
            'portfolio_tickers': portfolio_tickers,
            'portfolio_map': portfolio_map
        }
        filtered_alerts = filter_alerts(filtered_alerts, filters)

        # Display results
        st.markdown("---")
        st.markdown(f"### Results: {len(filtered_alerts)} alerts")

        if triggered_filter == 'Triggered Only':
            st.caption(f"Showing alerts that triggered between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}")
        elif triggered_filter == 'Not Triggered':
            st.caption(f"Showing alerts that did NOT trigger between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}")
        else:
            st.caption(f"Showing all alerts (filtered by date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})")

        if filtered_alerts:
            # Get trigger info for each alert within date range
            conn = db_config.get_connection(role="alerts")
            query = """
            SELECT
                alert_id,
                MAX(timestamp) as last_triggered,
                COUNT(*) as trigger_count
            FROM alert_audits
            WHERE alert_triggered = TRUE
            AND timestamp BETWEEN %s AND %s
            GROUP BY alert_id
            """
            trigger_dates_df = pd.read_sql_query(query, conn, params=(start_date, end_date))
            db_config.close_connection(conn)

            # Create a dict for quick lookup
            trigger_info = {}
            for _, row in trigger_dates_df.iterrows():
                trigger_info[row['alert_id']] = {
                    'last_triggered': row['last_triggered'],
                    'trigger_count': row['trigger_count']
                }

            # Display alerts individually (not grouped)
            for alert in filtered_alerts:
                alert_id = alert.get('alert_id')
                ticker = alert.get('ticker', 'Unknown')
                stock_name = alert.get('stock_name', ticker)
                exchange = alert.get('exchange', 'Unknown')
                economy = alert.get('rbics_economy', 'Unknown')

                # Get trigger info
                trigger_count = trigger_info.get(alert_id, {}).get('trigger_count', 0)
                last_triggered = trigger_info.get(alert_id, {}).get('last_triggered')

                # Build expander title
                trigger_status = f"({trigger_count} trigger{'s' if trigger_count != 1 else ''})" if trigger_count > 0 else "(Not triggered in range)"
                expander_title = f"**{ticker}** - {alert.get('name', 'Unknown')} {trigger_status}"

                with st.expander(expander_title):
                    # Stock info row
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"**Stock:** {stock_name}")
                        st.write(f"**Ticker:** {ticker}")
                    with col2:
                        st.write(f"**Exchange:** {exchange}")
                        st.write(f"**Economy:** {economy}")
                    with col3:
                        in_portfolios = portfolio_map.get(ticker, [])
                        if in_portfolios:
                            st.write(f"**Portfolio:** {', '.join(in_portfolios)}")
                        else:
                            st.write("**Portfolio:** None")

                    st.markdown("---")

                    # Alert details row
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.write(f"**Action:** {alert.get('action', 'N/A')}")
                    with col2:
                        st.write(f"**Timeframe:** {alert.get('timeframe', 'N/A')}")
                    with col3:
                        st.write(f"**Triggers in Range:** {trigger_count}")
                    with col4:
                        if last_triggered:
                            last_triggered_dt = pd.to_datetime(last_triggered)
                            st.write(f"**Last Triggered:**")
                            st.caption(last_triggered_dt.strftime('%Y-%m-%d %H:%M:%S'))

                    # Show conditions
                    conditions = alert.get('conditions', [])
                    if conditions:
                        st.write(f"**Conditions ({len(conditions)}):**")
                        for cond in conditions:
                            st.code(cond.get('conditions', 'N/A'), language=None)

                    # Button to view full history
                    if st.button(f"View Full History", key=f"view_{alert_id}"):
                        st.session_state['selected_ticker_from_filter'] = ticker
                        st.session_state['switch_to_search_tab'] = True
                        st.session_state['show_full_history'] = True
                        st.rerun()
        else:
            st.info("No alerts match the selected filters")

with tab1:
    # Check if we came from the filter view
    default_search = st.session_state.get('selected_ticker_from_filter', '')
    auto_search = False
    if default_search:
        auto_search = True
        if 'ticker_already_loaded' not in st.session_state or st.session_state.get('ticker_already_loaded') != default_search:
            st.session_state['ticker_already_loaded'] = default_search
        else:
            # Already loaded this ticker, don't keep resetting
            default_search = st.session_state.get('ticker_already_loaded', '')
            auto_search = False

    # Search interface
    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input(
            "Search by ticker or company name",
            value=default_search,
            placeholder="Enter ticker (e.g., AAPL) or company name (e.g., Apple)",
            help="Search is case-insensitive"
        )

    with col2:
        search_button = st.button("🔍 Search", type="primary", use_container_width=True)

    # Clear the session state after we've used it for the input
    if default_search and auto_search:
        del st.session_state['selected_ticker_from_filter']

    if search_term and (search_button or auto_search):
        # Search for matching companies
        results = search_companies(search_term, stock_db)

        if results:
            st.markdown("### Search Results")

            # Let user select from results if multiple matches
            if len(results) > 1:
                # Create selection options
                options = [f"{r['ticker']} - {r['name']} ({r['exchange']})" for r in results]
                selected_option = st.selectbox("Select a company:", options)

                # Extract selected ticker
                selected_ticker = selected_option.split(" - ")[0]
                selected_result = next(r for r in results if r['ticker'] == selected_ticker)
            else:
                selected_result = results[0]
                selected_ticker = selected_result['ticker']

            # Display company info
            st.markdown("---")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Ticker", selected_ticker)
            with col2:
                st.metric("Company", selected_result['name'][:30] + "..." if len(selected_result['name']) > 30 else selected_result['name'])
            with col3:
                st.metric("Exchange", selected_result['exchange'])
            with col4:
                st.metric("Type", selected_result['type'])

            # Get active alerts for this ticker
            ticker_alerts = [a for a in alerts if a.get('ticker') == selected_ticker]

            if ticker_alerts:
                st.markdown(f"### 📋 Active Alerts ({len(ticker_alerts)})")

                # Display active alerts
                for alert in ticker_alerts:
                    with st.expander(f"**{alert.get('name', 'Unnamed Alert')}**"):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.write(f"**Timeframe:** {alert.get('timeframe', 'N/A')}")
                            st.write(f"**Action:** {alert.get('action', 'N/A')}")
                        with col2:
                            st.write(f"**Entry Conditions:** {len(alert.get('entry_conditions', []))}")
                            st.write(f"**Exit Conditions:** {len(alert.get('exit_conditions', []))}")
                        with col3:
                            st.write(f"**Created:** {alert.get('created_date', 'Unknown')}")
                            st.write(f"**ID:** {alert.get('id', 'N/A')[:8]}...")
            else:
                st.info(f"No active alerts found for {selected_ticker}")

            # Get trigger history
            st.markdown("### 🔔 Alert Trigger History")

            # Add filter options
            col1, col2 = st.columns(2)
            with col1:
                show_all = st.checkbox("Show all evaluations (not just triggers)", value=False)
            with col2:
                # If came from filter view, default to 365 days to see full history
                show_full = st.session_state.get('show_full_history', False)
                default_days = 365 if show_full else 30
                days_back = st.number_input("Days of history", min_value=1, max_value=365, value=default_days)
                # Reset flag after using it once
                if show_full:
                    st.session_state['show_full_history'] = False

            if show_all:
                history_df = get_all_triggered_history(selected_ticker)
            else:
                history_df = get_alert_history_from_audit_log(selected_ticker)

            if not history_df.empty:
                # Filter by date range - handle timezone-aware timestamps
                cutoff_date = datetime.now() - timedelta(days=days_back)
                # Make cutoff_date timezone-aware or convert timestamps to naive
                history_df['timestamp'] = pd.to_datetime(history_df['timestamp']).dt.tz_localize(None)
                history_df = history_df[history_df['timestamp'] >= cutoff_date]

                if not history_df.empty:
                    # Summary stats
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Triggers", len(history_df[history_df['condition_met'] == 1]))
                    with col2:
                        st.metric("Total Evaluations", len(history_df))
                    with col3:
                        unique_alerts = history_df['alert_name'].nunique()
                        st.metric("Unique Alerts", unique_alerts)
                    with col4:
                        if not history_df.empty:
                            last_trigger = history_df[history_df['condition_met'] == 1]['timestamp'].max()
                            if pd.notna(last_trigger):
                                days_since = (datetime.now() - last_trigger).days
                                st.metric("Days Since Last Trigger", days_since)

                    st.markdown("---")

                    # Show recent triggers
                    if show_all:
                        st.markdown("#### Recent Evaluations")
                        display_df = history_df.head(50)
                    else:
                        st.markdown("#### Recent Triggers")
                        display_df = history_df[history_df['condition_met'] == 1].head(20)

                    if not display_df.empty:
                        for idx, row in display_df.iterrows():
                            # Create an expander for each trigger
                            icon = "✅" if row['condition_met'] == 1 else "❌"
                            trigger_date = row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else 'Unknown'
                            trigger_time = row['time'] if pd.notna(row['time']) else ''

                            with st.expander(f"{icon} **{row['alert_name']}** - {trigger_date} {trigger_time}"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write(f"**Triggered:** {'Yes ✅' if row['condition_met'] == 1 else 'No ❌'}")
                                    st.write(f"**Source:** {row['trigger_source']}")
                                    st.write(f"**Execution Time:** {row['execution_time_ms']:.2f} ms" if pd.notna(row['execution_time_ms']) else "N/A")
                                with col2:
                                    st.write("**Condition Details:**")
                                    st.code(format_condition(row['condition_details']), language=None)

                        # Group by alert name to show statistics
                        st.markdown("#### Alert Statistics")
                        alert_stats = display_df.groupby('alert_name').agg({
                            'condition_met': ['count', 'sum'],
                            'timestamp': ['min', 'max']
                        }).round(2)

                        alert_stats.columns = ['Total Checks', 'Triggers', 'First Check', 'Last Check']
                        alert_stats['Trigger Rate'] = (alert_stats['Triggers'] / alert_stats['Total Checks'] * 100).round(1)
                        alert_stats['Trigger Rate'] = alert_stats['Trigger Rate'].astype(str) + '%'

                        st.dataframe(alert_stats, use_container_width=True)
                    else:
                        if show_all:
                            st.info(f"No evaluations found for {selected_ticker} in the last {days_back} days")
                        else:
                            st.info(f"No triggers found for {selected_ticker} in the last {days_back} days")
                else:
                    st.info(f"No data found for {selected_ticker} in the last {days_back} days")
            else:
                if show_all:
                    st.info(f"No evaluation history found for {selected_ticker}")
                else:
                    st.info(f"No trigger history found for {selected_ticker}")

                    # Suggest checking all evaluations
                    st.markdown("💡 **Tip:** Check 'Show all evaluations' to see when alerts were checked but didn't trigger.")
        else:
            st.warning(f"No companies found matching '{search_term}'")
    else:
        # Show instructions
        st.markdown("""
        ### How to use:
        1. Enter a ticker symbol (e.g., AAPL, MSFT) or company name (e.g., Apple, Microsoft)
        2. Click Search or press Enter
        3. Select the company if multiple matches are found
        4. View active alerts and trigger history

        ### Features:
        - 📋 View all active alerts for a company
        - 🔔 See when alerts were triggered
        - 📊 View trigger statistics and patterns
        - ⏱️ Filter by date range
        - ✅ Option to show all evaluations vs only triggers
        """)

        # Show some popular searches
        st.markdown("### Popular Tickers")
        col1, col2, col3, col4, col5 = st.columns(5)

        popular_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        for col, ticker in zip([col1, col2, col3, col4, col5], popular_tickers):
            with col:
                if st.button(ticker, use_container_width=True):
                    st.rerun()

# Add footer
st.markdown("---")
st.caption("Data sourced from alert audit logs and configured alerts")

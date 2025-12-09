import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add the parent directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the new functions
from alert_audit_logger import (
    get_audit_summary, get_alert_history, get_performance_metrics,
    get_daily_evaluation_stats, get_evaluation_coverage, get_expected_daily_evaluations
)
from db_config import db_config


@st.cache_data(ttl=30, show_spinner=False)
def cached_summary(days: int) -> pd.DataFrame:
    return get_audit_summary(days)


@st.cache_data(ttl=30, show_spinner=False)
def cached_metrics(days: int) -> dict:
    return get_performance_metrics(days)


@st.cache_data(ttl=30, show_spinner=False)
def cached_daily_stats(days: int) -> pd.DataFrame:
    return get_daily_evaluation_stats(days)


@st.cache_data(ttl=30, show_spinner=False)
def cached_coverage(days: int) -> pd.DataFrame:
    return get_evaluation_coverage(days)


st.set_page_config(
    page_title="Alert Audit Logs",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Alert Audit Logs & Analytics")
st.markdown("Comprehensive tracking of all alert evaluations, performance metrics, and system health")

# Sidebar filters
st.sidebar.header("🔍 Filters & Options")

# Date range filter
days_back = st.sidebar.slider(
    "Days to analyze", 
    min_value=1, 
    max_value=90, 
    value=7,
    help="How many days back to analyze"
)

# Alert ID filter
alert_id_filter = st.sidebar.text_input(
    "Filter by Alert ID",
    placeholder="Enter alert ID to filter",
    help="Leave empty to see all alerts"
)

# Ticker filter
ticker_filter = st.sidebar.text_input(
    "Filter by Ticker",
    placeholder="Enter ticker symbol",
    help="Leave empty to see all tickers"
)

# Evaluation type filter
evaluation_types = ["All", "scheduled", "manual", "test"]
selected_evaluation_type = st.sidebar.selectbox(
    "Evaluation Type",
    evaluation_types,
    help="Filter by how the alert was evaluated"
)

# Status filter
status_filter = st.sidebar.selectbox(
    "Status Filter",
    ["All", "Success", "Error", "Triggered", "Not Triggered"],
    help="Filter by evaluation status"
)

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Performance Overview")
    
    # Get performance metrics
    metrics = cached_metrics(days_back)
    
    if metrics:
        # Display key metrics
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.metric(
                "Total Checks", 
                f"{metrics['total_checks']:,}",
                help="Total alert evaluations in the period"
            )
        
        with metric_cols[1]:
            st.metric(
                "Success Rate", 
                f"{metrics['success_rate']:.1f}%",
                help="Percentage of successful price data pulls"
            )
        
        with metric_cols[2]:
            st.metric(
                "Cache Hit Rate", 
                f"{metrics['cache_hit_rate']:.1f}%",
                help="Percentage of requests served from cache"
            )
        
        with metric_cols[3]:
            st.metric(
                "Avg Execution", 
                f"{metrics['avg_execution_time_ms']:.0f}ms",
                help="Average time to evaluate an alert"
            )
        
        # Error rate warning
        if metrics['error_rate'] > 5:
            st.warning(f"⚠️ High error rate: {metrics['error_rate']:.1f}% of evaluations failed")
        elif metrics['error_rate'] > 1:
            st.info(f"ℹ️ Error rate: {metrics['error_rate']:.1f}% of evaluations failed")
        else:
            st.success(f"✅ Low error rate: {metrics['error_rate']:.1f}% of evaluations failed")
    
    else:
        st.info("No audit data available for the selected period")

with col2:
    st.subheader("⚡ Quick Actions")
    
    if st.button("🔄 Refresh Data", help="Refresh all audit data"):
        st.rerun()
    
    if st.button("🧹 Clear All Audit Data", help="Clear all audit logs and start fresh"):
        with st.spinner("Clearing all audit records..."):
            from alert_audit_logger import audit_logger
            # Clear all records by setting days_to_keep to 0
            deleted_count = audit_logger.cleanup_old_records(0)
            if deleted_count > 0:
                st.success(f"✅ Successfully cleared {deleted_count:,} audit records!")
            else:
                st.info("No audit records to clear.")
        st.rerun()
    
    # Export options
    st.markdown("**📤 Export Options**")
    
    if st.button("📊 Export Summary CSV"):
        summary_df = get_audit_summary(days_back)
        if not summary_df.empty:
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="💾 Download Summary CSV",
                data=csv,
                file_name=f"alert_audit_summary_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

# Main audit summary table
st.subheader("📋 Alert Evaluation Summary")
st.markdown(f"Showing data for the last **{days_back} days**")

max_summary_rows = st.sidebar.number_input(
    "Max rows to display",
    min_value=100,
    max_value=5000,
    value=500,
    step=100,
    help="Limit the number of summary rows rendered in the table.",
)

# Get audit summary
summary_df = cached_summary(days_back)

if not summary_df.empty:
    if len(summary_df) > max_summary_rows:
        st.caption(f"Displaying the most recent {max_summary_rows:,} alerts. Increase 'Max rows to display' to see more.")
        summary_df = summary_df.head(max_summary_rows)

    # Apply filters
    filtered_df = summary_df.copy()
    
    if alert_id_filter:
        filtered_df = filtered_df[filtered_df['alert_id'].str.contains(alert_id_filter, case=False, na=False)]
    
    if ticker_filter:
        filtered_df = filtered_df[filtered_df['ticker'].str.contains(ticker_filter.upper(), case=False, na=False)]
    
    if selected_evaluation_type != "All":
        filtered_df = filtered_df[filtered_df['evaluation_type'] == selected_evaluation_type]
    
    if status_filter != "All":
        if status_filter == "Success":
            filtered_df = filtered_df[filtered_df['successful_evaluations'] > 0]
        elif status_filter == "Error":
            filtered_df = filtered_df[filtered_df['total_checks'] > filtered_df['successful_evaluations']]
        elif status_filter == "Triggered":
            filtered_df = filtered_df[filtered_df['total_triggers'] > 0]
        elif status_filter == "Not Triggered":
            filtered_df = filtered_df[filtered_df['total_triggers'] == 0]
    
    # Display filtered results
    if not filtered_df.empty:
        st.markdown(f"**Found {len(filtered_df)} alerts matching your filters**")
        
        # Format the dataframe for display
        display_df = filtered_df.copy()
        # Convert UTC timestamps to local timezone (EST/EDT)
        import pytz
        eastern = pytz.timezone('America/New_York')
        display_df['last_check'] = pd.to_datetime(display_df['last_check'], utc=True).dt.tz_convert(eastern).dt.strftime('%Y-%m-%d %H:%M')
        display_df['first_check'] = pd.to_datetime(display_df['first_check'], utc=True).dt.tz_convert(eastern).dt.strftime('%Y-%m-%d %H:%M')
        display_df['avg_execution_time_ms'] = display_df['avg_execution_time_ms'].round(1)
        
        # Rename columns for better display
        display_df = display_df.rename(columns={
            'alert_id': 'Alert ID',
            'ticker': 'Ticker',
            'stock_name': 'Stock Name',
            'exchange': 'Exchange',
            'timeframe': 'Timeframe',
            'action': 'Action',
            'evaluation_type': 'Type',
            'total_checks': 'Checks',
            'successful_price_pulls': 'Price Pulls',
            'successful_evaluations': 'Evaluations',
            'total_triggers': 'Triggers',
            'avg_execution_time_ms': 'Avg Time (ms)',
            'last_check': 'Last Check',
            'first_check': 'First Check'
        })
        
        # Display the table
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        # Add search functionality
        search_term = st.text_input(
            "🔍 Search in results",
            placeholder="Search by alert ID, ticker, or stock name...",
            help="Search within the filtered results"
        )
        
        if search_term:
            search_results = display_df[
                display_df.apply(
                    lambda x: search_term.lower() in str(x).lower(), 
                    axis=1
                )
            ]
            st.markdown(f"**Search results: {len(search_results)} matches**")
            st.dataframe(search_results, use_container_width=True, hide_index=True)
    
    else:
        st.warning("No alerts match your current filters. Try adjusting the filter criteria.")
        
else:
    st.info("No audit data available. Alerts will start appearing here once they are evaluated.")

# Detailed view section
st.subheader("🔍 Detailed Alert History")

if alert_id_filter:
    st.markdown(f"**Detailed history for Alert ID: {alert_id_filter}**")
    
    # Get detailed history
    history_df = get_alert_history(alert_id_filter, 100)
    
    if not history_df.empty:
        # Format for display
        history_display = history_df.copy()
        # Convert UTC timestamps to local timezone (EST/EDT)
        import pytz
        eastern = pytz.timezone('America/New_York')
        history_display['timestamp'] = pd.to_datetime(history_display['timestamp'], utc=True).dt.tz_convert(eastern).dt.strftime('%Y-%m-%d %H:%M:%S')
        history_display['execution_time_ms'] = history_display['execution_time_ms'].fillna('N/A')
        
        # Select columns to display
        display_columns = [
            'timestamp', 'ticker', 'evaluation_type', 'price_data_pulled',
            'price_data_source', 'cache_hit', 'conditions_evaluated',
            'alert_triggered', 'trigger_reason', 'execution_time_ms', 'error_message'
        ]
        
        history_display = history_display[display_columns].rename(columns={
            'timestamp': 'Timestamp',
            'ticker': 'Ticker',
            'evaluation_type': 'Type',
            'price_data_pulled': 'Price Data',
            'price_data_source': 'Source',
            'cache_hit': 'Cache Hit',
            'conditions_evaluated': 'Evaluated',
            'alert_triggered': 'Triggered',
            'trigger_reason': 'Trigger Reason',
            'execution_time_ms': 'Time (ms)',
            'error_message': 'Error'
        })
        
        st.dataframe(
            history_display,
            use_container_width=True,
            hide_index=True
        )
        
        # Performance trend chart
        if len(history_df) > 1:
            st.subheader("📈 Performance Trend")
            
            # Convert timestamp to datetime for plotting
            plot_df = history_df.copy()
            plot_df['timestamp'] = pd.to_datetime(plot_df['timestamp'])
            plot_df = plot_df.sort_values('timestamp')
            
            # Execution time trend
            fig = px.line(
                plot_df[plot_df['execution_time_ms'].notna()],
                x='timestamp',
                y='execution_time_ms',
                title='Execution Time Trend',
                labels={'execution_time_ms': 'Execution Time (ms)', 'timestamp': 'Time'}
            )
            fig.update_layout(xaxis_title="Time", yaxis_title="Execution Time (ms)")
            st.plotly_chart(fig, use_container_width=True)
            
    else:
        st.info(f"No history found for Alert ID: {alert_id_filter}")
        
else:
    st.info("Enter an Alert ID above to view detailed history")

# Charts and analytics
if not summary_df.empty:
    st.subheader("📊 Analytics & Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Success rate by timeframe
        if 'timeframe' in summary_df.columns:
            timeframe_success = summary_df.groupby('timeframe').agg({
                'total_checks': 'sum',
                'successful_evaluations': 'sum'
            }).reset_index()
            timeframe_success['success_rate'] = (
                timeframe_success['successful_evaluations'] / timeframe_success['total_checks'] * 100
            )
            
            fig = px.bar(
                timeframe_success,
                x='timeframe',
                y='success_rate',
                title='Success Rate by Timeframe',
                labels={'success_rate': 'Success Rate (%)', 'timeframe': 'Timeframe'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Cache hit rate over time
        if 'cache_hit' in summary_df.columns:
            cache_data = summary_df.groupby('ticker').agg({
                'total_checks': 'sum',
                'cache_hit': 'sum'
            }).reset_index()
            cache_data['cache_hit_rate'] = (
                cache_data['cache_hit'] / cache_data['total_checks'] * 100
            )
            
            fig = px.scatter(
                cache_data.head(20),  # Top 20 tickers
                x='total_checks',
                y='cache_hit_rate',
                size='total_checks',
                hover_data=['ticker'],
                title='Cache Hit Rate vs Check Frequency',
                labels={'cache_hit_rate': 'Cache Hit Rate (%)', 'total_checks': 'Total Checks'}
            )
            st.plotly_chart(fig, use_container_width=True)

# Failed Price Data Analysis Section
st.markdown("---")
st.subheader("🚨 Failed Price Data Analysis")

# Get failed price data alerts
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(f"**Alerts that failed to retrieve price data from FMP API (last {days_back} days)**")
with col2:
    if st.button("🔄 Refresh Failed Data", help="Refresh failed alerts analysis"):
        st.rerun()

# Query for failed price data
try:
    conn = db_config.get_connection(role="alert_audit_logs")
    interval_clause = f"timestamp >= NOW() - INTERVAL '{int(days_back)} days'"
    failed_query = f"""
        SELECT 
            alert_id,
            ticker,
            stock_name,
            exchange,
            timeframe,
            COUNT(*) as failure_count,
            MAX(timestamp) as last_failure,
            MIN(timestamp) as first_failure,
            AVG(execution_time_ms) as avg_execution_time
        FROM alert_audits
        WHERE 
            (price_data_pulled = FALSE
            OR error_message ILIKE %s
            OR error_message ILIKE %s)
            AND {interval_clause}
        GROUP BY alert_id, ticker, stock_name, exchange, timeframe
        ORDER BY failure_count DESC
    """
    failed_params = ('%No data available%', '%FMP API%')

    # Get recent failed price data alerts (filtered by days_back from sidebar)
    failed_df = pd.read_sql_query(failed_query, conn, params=failed_params)
    
    # Add asset type by checking market data
    if not failed_df.empty:
        # Load market data to determine asset type
        from utils import load_market_data
        market_data = load_market_data()
        
        # Function to determine asset type
        def get_asset_type(ticker):
            if market_data.empty:
                return "Unknown"
            
            # Normalize ticker for comparison
            ticker_upper = str(ticker).upper() if ticker else ""
            
            # Try exact match first
            stock_matches = market_data[market_data['Symbol'].str.upper() == ticker_upper]
            if stock_matches.empty:
                # Try with -US suffix for US stocks/ETFs
                stock_matches = market_data[market_data['Symbol'].str.upper() == f"{ticker_upper}-US"]
            if stock_matches.empty:
                # Try removing any suffix from the ticker
                base_ticker = ticker_upper.split('-')[0]
                stock_matches = market_data[market_data['Symbol'].str.upper().str.startswith(base_ticker)]
            
            if not stock_matches.empty:
                stock_data = stock_matches.iloc[0]
                # Check if it's an ETF
                if pd.notna(stock_data.get('ETF_Issuer')):
                    return "ETF"
                else:
                    return "Stock"
            
            return "Unknown"
        
        # Add asset type column
        failed_df['asset_type'] = failed_df['ticker'].apply(get_asset_type)
    
    if not failed_df.empty:
        # Calculate statistics
        total_failed_alerts = len(failed_df)
        total_failures = failed_df['failure_count'].sum()
        
        # Display metrics
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric(
                "Failed Alerts",
                f"{total_failed_alerts:,}",
                help="Unique alerts that failed to get price data"
            )
        with metric_cols[1]:
            st.metric(
                "Total Failures", 
                f"{total_failures:,}",
                help="Total number of failed price data attempts"
            )
        with metric_cols[2]:
            avg_failures = total_failures / max(total_failed_alerts, 1)
            st.metric(
                "Avg Failures/Alert",
                f"{avg_failures:.1f}",
                help="Average failures per alert"
            )
        with metric_cols[3]:
            # Calculate failure rate using the same days_back filter
            total_checks_query = f"""
                SELECT COUNT(*) 
                FROM alert_audits 
                WHERE timestamp >= NOW() - INTERVAL '{int(days_back)} days'
            """
            total_checks = pd.read_sql_query(total_checks_query, conn).iloc[0, 0]
            failure_rate = (total_failures / max(total_checks, 1)) * 100
            st.metric(
                "Failure Rate",
                f"{failure_rate:.1f}%",
                help=f"Percentage of checks that failed to get price data in last {days_back} days"
            )
        
        # Asset type breakdown
        st.markdown("#### 📊 Failures by Asset Type")
        asset_type_stats = failed_df.groupby('asset_type').agg({
            'alert_id': 'count',
            'failure_count': 'sum'
        }).rename(columns={'alert_id': 'failed_alerts'})
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Asset Type Breakdown:**")
            for asset_type, row in asset_type_stats.iterrows():
                percentage = (row['failed_alerts'] / total_failed_alerts) * 100
                st.caption(f"• **{asset_type}**: {row['failed_alerts']} alerts ({percentage:.1f}%), {row['failure_count']} total failures")
        
        with col2:
            # Create pie chart of asset types
            fig_asset = px.pie(
                asset_type_stats.reset_index(),
                values='failed_alerts',
                names='asset_type',
                title='Failed Alerts by Asset Type'
            )
            st.plotly_chart(fig_asset, use_container_width=True)
        
        # Group by exchange
        st.markdown("#### 📊 Failures by Exchange")
        exchange_failures = failed_df.groupby('exchange').agg({
            'alert_id': 'count',
            'failure_count': 'sum'
        }).rename(columns={'alert_id': 'failed_alerts'}).sort_values('failure_count', ascending=False)
        
        if not exchange_failures.empty:
            # Show top exchanges with failures
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Top Exchanges with Failed Alerts:**")
                top_exchanges = exchange_failures.head(10)
                for exchange, row in top_exchanges.iterrows():
                    st.caption(f"• **{exchange}**: {row['failed_alerts']} alerts, {row['failure_count']} failures")
            
            with col2:
                # Create bar chart of failures by exchange
                fig_exchange = px.bar(
                    exchange_failures.head(15).reset_index(),
                    x='exchange',
                    y='failure_count',
                    title='Top 15 Exchanges by Failure Count',
                    labels={'failure_count': 'Number of Failures', 'exchange': 'Exchange'}
                )
                fig_exchange.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_exchange, use_container_width=True)
        
        # Group by ticker patterns
        st.markdown("#### 🔍 Most Problematic Tickers")
        
        # Format the dataframe for display
        display_failed_df = failed_df.copy()
        # Convert UTC timestamps to local timezone (EST/EDT)
        import pytz
        eastern = pytz.timezone('America/New_York')
        display_failed_df['last_failure'] = pd.to_datetime(display_failed_df['last_failure'], utc=True).dt.tz_convert(eastern).dt.strftime('%Y-%m-%d %H:%M')
        display_failed_df['first_failure'] = pd.to_datetime(display_failed_df['first_failure'], utc=True).dt.tz_convert(eastern).dt.strftime('%Y-%m-%d %H:%M')
        display_failed_df = display_failed_df.rename(columns={
            'alert_id': 'Alert ID',
            'ticker': 'Ticker',
            'asset_type': 'Asset Type',
            'stock_name': 'Stock Name',
            'exchange': 'Exchange',
            'timeframe': 'Timeframe',
            'failure_count': 'Failures',
            'last_failure': 'Last Failure',
            'first_failure': 'First Failure',
            'avg_execution_time': 'Avg Time (ms)'
        })
        
        # Show top 20 most problematic alerts
        st.markdown("**Top 20 Alerts with Most Failures:**")
        st.dataframe(
            display_failed_df.head(20),
            use_container_width=True,
            hide_index=True
        )
        
        # Export functionality
        st.markdown("#### 📤 Export Failed Alerts Data")
        col1, col2 = st.columns(2)
        
        with col1:
            # Prepare CSV for download
            csv_failed = display_failed_df.to_csv(index=False)
            st.download_button(
                label="💾 Download Failed Alerts CSV",
                data=csv_failed,
                file_name=f"failed_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                help="Download all failed alerts data as CSV"
            )
        
        with col2:
            # Create summary report
            summary_text = f"""Failed Price Data Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY STATISTICS:
- Total Failed Alerts: {total_failed_alerts:,}
- Total Failure Events: {total_failures:,}
- Average Failures per Alert: {avg_failures:.1f}
- Overall Failure Rate: {failure_rate:.1f}%

TOP 5 EXCHANGES WITH FAILURES:
"""
            for i, (exchange, row) in enumerate(exchange_failures.head(5).iterrows(), 1):
                summary_text += f"{i}. {exchange}: {row['failed_alerts']} alerts, {row['failure_count']} failures\n"
            
            # Add asset type breakdown to summary
            summary_text += "\n\nASSET TYPE BREAKDOWN:\n"
            for asset_type, row in asset_type_stats.iterrows():
                percentage = (row['failed_alerts'] / total_failed_alerts) * 100
                summary_text += f"- {asset_type}: {row['failed_alerts']} alerts ({percentage:.1f}%), {row['failure_count']} failures\n"
            
            summary_text += "\nTOP 10 PROBLEMATIC TICKERS:\n"
            for i, row in enumerate(failed_df.head(10).itertuples(), 1):
                summary_text += f"{i}. {row.ticker} ({row.asset_type}, {row.exchange}): {row.failure_count} failures\n"
            
            st.download_button(
                label="📄 Download Summary Report",
                data=summary_text,
                file_name=f"failed_alerts_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                help="Download summary report as text file"
            )
        
        # Analysis insights
        st.markdown("#### 💡 Insights & Recommendations")
        
        # Check for pattern insights
        if failure_rate > 10:
            st.error(f"⚠️ **High Failure Rate**: {failure_rate:.1f}% of price data requests are failing. This requires immediate attention.")
        elif failure_rate > 5:
            st.warning(f"⚠️ **Moderate Failure Rate**: {failure_rate:.1f}% failure rate. Consider investigating the most problematic exchanges.")
        else:
            st.info(f"✅ **Acceptable Failure Rate**: {failure_rate:.1f}% failure rate is within normal limits.")
        
        # Exchange-specific insights
        if not exchange_failures.empty:
            worst_exchange = exchange_failures.index[0]
            worst_count = exchange_failures.iloc[0]['failure_count']
            st.info(f"📊 **{worst_exchange}** has the most failures ({worst_count:,}). Consider checking FMP API coverage for this exchange.")
        
        # Ticker pattern insights
        failed_tickers = failed_df['ticker'].tolist()
        if any('-' in str(ticker) for ticker in failed_tickers[:20]):
            st.info("🔍 Some failed tickers contain special characters (e.g., '-'). These may require special handling in the API.")
        
        if any(str(ticker).endswith(('.TO', '.L', '.HK', '.AS')) for ticker in failed_tickers[:20]):
            st.info("🌍 Many failed tickers have international exchange suffixes. Ensure proper exchange code handling.")
            
    else:
        st.success("✅ No failed price data retrievals found! All alerts are getting price data successfully.")
    
    db_config.close_connection(conn)
    
except Exception as e:
    st.error(f"Error analyzing failed price data: {e}")
    st.info("Make sure alert evaluations have been running to see failed data analysis.")

# Footer
st.markdown("---")
st.markdown(
    "**💡 Tips:** Use the filters in the sidebar to focus on specific alerts, timeframes, or statuses. "
    "Export data to CSV for external analysis. The system automatically tracks all alert evaluations "
    "to help you monitor performance and identify issues."
)
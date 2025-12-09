import os
import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import subprocess
import psutil
import time
from data_access.document_store import load_document

# Add parent directory for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Page configuration
st.set_page_config(
    page_title="Futures Alerts Dashboard",
    page_icon="📈",
    layout="wide"
)

# Import utils
from utils import load_market_data

FUTURES_ALERTS_DOCUMENT = "futures_alerts"
FUTURES_DATABASE_DOCUMENT = "futures_database"
FUTURES_STATUS_DOCUMENT = "futures_scheduler_status"
FUTURES_CONFIG_DOCUMENT = "futures_scheduler_config"

def load_futures_alerts():
    """Load futures alerts from the document store."""
    data = load_document(
        FUTURES_ALERTS_DOCUMENT,
        default=[],
        fallback_path="futures_alerts.json",
    )
    return data if isinstance(data, list) else []

def load_futures_metadata():
    """Load futures metadata"""
    data = load_document(
        FUTURES_DATABASE_DOCUMENT,
        default={},
        fallback_path="futures_database.json",
    )
    return data if isinstance(data, dict) else {}

def get_alert_status(alert):
    """Determine alert status (Active/Triggered/Inactive)"""
    # Check if alert has been triggered
    last_triggered = alert.get("last_triggered", "Never")
    if last_triggered != "Never":
        try:
            # Check if triggered today
            triggered_date = datetime.fromisoformat(last_triggered.replace('Z', '+00:00'))
            if triggered_date.date() == datetime.now().date():
                return "🔴 Triggered Today"
            else:
                return "🟡 Previously Triggered"
        except:
            pass

    # For futures alerts, action is Buy/Sell, not on/off
    # All futures alerts are considered active
    return "🟢 Active"

def is_futures_scheduler_running():
    """Check if the futures scheduler is running"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline')
            if cmdline and any('futures_auto_scheduler.py' in arg for arg in cmdline):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Also check for lock file
    if os.path.exists('futures_scheduler.lock'):
        try:
            with open('futures_scheduler.lock', 'r') as f:
                pid = int(f.read())
            # Check if process is actually running
            try:
                psutil.Process(pid)
                return True
            except psutil.NoSuchProcess:
                # Stale lock file
                os.remove('futures_scheduler.lock')
                return False
        except:
            return False
    return False

def get_futures_scheduler_status():
    """Get detailed futures scheduler status"""
    status = load_document(
        FUTURES_STATUS_DOCUMENT,
        default={},
        fallback_path='futures_scheduler_status.json',
    )
    return status if isinstance(status, dict) else None

def load_futures_scheduler_config() -> dict:
    """Return the scheduler configuration document."""
    config = load_document(
        FUTURES_CONFIG_DOCUMENT,
        default={},
        fallback_path='futures_scheduler_config.json',
    )
    return config if isinstance(config, dict) else {}

def start_futures_scheduler():
    """Start the futures scheduler as a background process"""
    try:
        python_path = sys.executable
        if not python_path:
            python_path = r"C:\Users\NickK\AppData\Local\Programs\Python\Python313\python.exe"

        # Clean up any stale lock files
        if os.path.exists('futures_scheduler.lock'):
            try:
                os.remove('futures_scheduler.lock')
            except:
                pass

        # Start the scheduler
        env = os.environ.copy()
        process = subprocess.Popen(
            [python_path, "futures_auto_scheduler.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        )

        # Give it time to start
        time.sleep(3)

        # Check if it started
        return is_futures_scheduler_running()
    except Exception as e:
        st.error(f"Failed to start futures scheduler: {e}")
        return False

def stop_futures_scheduler():
    """Stop the futures scheduler"""
    try:
        # Try to stop gracefully using the lock file
        if os.path.exists('futures_scheduler.lock'):
            with open('futures_scheduler.lock', 'r') as f:
                pid = int(f.read())

            try:
                # Terminate the process
                proc = psutil.Process(pid)
                proc.terminate()
                time.sleep(2)

                # Force kill if still running
                if proc.is_running():
                    proc.kill()
            except psutil.NoSuchProcess:
                pass

            # Remove lock file
            if os.path.exists('futures_scheduler.lock'):
                os.remove('futures_scheduler.lock')

        # Also try to find and stop by name
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline')
                if cmdline and any('futures_auto_scheduler.py' in arg for arg in cmdline):
                    proc.terminate()
                    time.sleep(1)
                    if proc.is_running():
                        proc.kill()
            except:
                continue

        return True
    except Exception as e:
        st.error(f"Failed to stop futures scheduler: {e}")
        return False

def format_conditions(conditions):
    """Format alert conditions for display"""
    if isinstance(conditions, dict):
        # New format from Add_Alert page
        formatted = []
        for key, value in conditions.items():
            if isinstance(value, dict) and value.get("conditions"):
                conds = value["conditions"]
                if isinstance(conds, list):
                    formatted.extend(conds)
                else:
                    formatted.append(str(conds))
        return " | ".join(formatted) if formatted else "No conditions"
    elif isinstance(conditions, list):
        # Old format
        if len(conditions) > 0:
            if isinstance(conditions[0], dict):
                return " | ".join([str(c.get("conditions", c)) for c in conditions])
            else:
                return " | ".join([str(c) for c in conditions])
    return str(conditions)

def main():
    st.title("📈 Futures Alerts Dashboard")
    st.markdown("Monitor and manage your futures contract alerts")

    # Scheduler Control Section
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 3])

    with col1:
        st.subheader("⚙️ Futures Scheduler Control")

    with col2:
        # Check current scheduler status
        scheduler_running = is_futures_scheduler_running()

        # Toggle switch for scheduler
        scheduler_enabled = st.toggle(
            "Enable Scheduler",
            value=scheduler_running,
            help="Start/stop the automatic futures price updater and alert checker"
        )

        # Handle toggle changes
        if scheduler_enabled != scheduler_running:
            if scheduler_enabled:
                with st.spinner("Starting futures scheduler..."):
                    if start_futures_scheduler():
                        st.success("✅ Scheduler started!")
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("Failed to start scheduler")
            else:
                with st.spinner("Stopping futures scheduler..."):
                    if stop_futures_scheduler():
                        st.success("⏹️ Scheduler stopped!")
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("Failed to stop scheduler")

    with col3:
        # Display scheduler status
        if scheduler_running:
            status_data = get_futures_scheduler_status()
            if status_data:
                st.success(f"🟢 **Status:** {status_data.get('status', 'Running')}")
                if 'last_update' in status_data:
                    last_update = status_data['last_update']
                    try:
                        dt = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                        st.caption(f"Last price update: {dt.strftime('%I:%M:%S %p')}")
                    except:
                        st.caption(f"Last update: {last_update}")
                if 'last_check' in status_data:
                    last_check = status_data['last_check']
                    try:
                        dt = datetime.fromisoformat(last_check.replace('Z', '+00:00'))
                        st.caption(f"Last alert check: {dt.strftime('%I:%M:%S %p')}")
                    except:
                        st.caption(f"Last check: {last_check}")
            else:
                st.success("🟢 Scheduler is running")
                st.caption("Updates every 15 min | Prices 6x daily")

            # Load and display scheduler times
            try:
                config = load_futures_scheduler_config()
                update_times = config.get('update_times', [])
                if update_times:
                    # Convert 24hr to 12hr format
                    formatted_times = []
                    for time_str in update_times:
                        try:
                            hour, minute = map(int, time_str.split(':'))
                            period = 'AM' if hour < 12 else 'PM'
                            hour_12 = hour % 12
                            if hour_12 == 0:
                                hour_12 = 12
                            formatted_times.append(f"{hour_12}:{minute:02d} {period}")
                        except:
                            formatted_times.append(time_str)
                    times_str = ", ".join(formatted_times)
                    st.caption(f"📅 Scheduled runs: {times_str}")
            except Exception:
                pass
        else:
            st.warning("⚫ Scheduler is not running")
            st.caption("Enable to start automatic monitoring")

            # Still show scheduled times even when not running
            try:
                config = load_futures_scheduler_config()
                update_times = config.get('update_times', [])
                if update_times:
                    # Convert 24hr to 12hr format
                    formatted_times = []
                    for time_str in update_times:
                        try:
                            hour, minute = map(int, time_str.split(':'))
                            period = 'AM' if hour < 12 else 'PM'
                            hour_12 = hour % 12
                            if hour_12 == 0:
                                hour_12 = 12
                            formatted_times.append(f"{hour_12}:{minute:02d} {period}")
                        except:
                            formatted_times.append(time_str)
                    times_str = ", ".join(formatted_times)
                    st.caption(f"📅 Will run at: {times_str}")
            except Exception:
                pass

    st.markdown("---")

    # Load data
    alerts = load_futures_alerts()
    futures_db = load_futures_metadata()

    # Sidebar statistics
    with st.sidebar:
        st.header("📊 Statistics")

        if alerts:
            total_alerts = len(alerts)
            # For futures alerts, action is Buy/Sell, not on/off - all are considered active
            active_alerts = len(alerts)
            triggered_today = sum(1 for a in alerts if a.get("last_triggered", "Never") != "Never"
                                 and a["last_triggered"].startswith(datetime.now().strftime("%Y-%m-%d")))

            st.metric("Total Alerts", total_alerts)
            st.metric("Active Alerts", active_alerts)
            st.metric("Triggered Today", triggered_today)

            # Alerts by category
            st.subheader("By Category")
            categories = {}
            for alert in alerts:
                ticker = alert.get("ticker", "")
                if ticker in futures_db:
                    category = futures_db[ticker].get("category", "Unknown")
                    categories[category] = categories.get(category, 0) + 1

            for cat, count in sorted(categories.items()):
                st.write(f"• {cat}: {count}")

            # Adjustment methods
            st.subheader("Adjustment Methods")
            adjustments = {}
            for alert in alerts:
                method = alert.get("adjustment_method", "none")
                adjustments[method] = adjustments.get(method, 0) + 1

            for method, count in adjustments.items():
                method_name = {
                    "none": "None (Raw)",
                    "panama": "Panama (Additive)",
                    "ratio": "Ratio (Multiplicative)"
                }.get(method, method)
                st.write(f"• {method_name}: {count}")
        else:
            st.info("No futures alerts created yet")

    # Main content area
    if not alerts:
        st.info("👋 No futures alerts found. Create your first futures alert using the Add Alert page!")

        # Quick start guide
        st.markdown("### 🚀 Quick Start")
        st.markdown("""
        1. **Add Alert**: Go to the Add Alert page and enter a futures symbol (e.g., ES, CL, GC)
        2. **Set Conditions**: Define price levels, indicators, or volume conditions
        3. **Choose Adjustment**: Select Panama (for price alerts) or Ratio (for % alerts)
        4. **Monitor Here**: Your futures alerts will appear on this dashboard
        """)

        # Available futures
        st.markdown("### 📊 Available Futures Contracts")
        if futures_db:
            categories = {}
            for symbol, data in futures_db.items():
                category = data.get("category", "Unknown")
                if category not in categories:
                    categories[category] = []
                categories[category].append(f"{symbol} - {data.get('name', symbol)}")

            cols = st.columns(3)
            for idx, (category, symbols) in enumerate(sorted(categories.items())):
                with cols[idx % 3]:
                    st.write(f"**{category}**")
                    for symbol in symbols[:5]:
                        st.write(f"• {symbol}")
                    if len(symbols) > 5:
                        st.caption(f"...and {len(symbols) - 5} more")
    else:
        # Display alerts
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📋 All Alerts", "🟢 Active", "🔴 Triggered", "📊 Analytics"])

        with tab1:
            # All alerts view
            st.subheader(f"All Futures Alerts ({len(alerts)})")

            # Create DataFrame for display
            alert_data = []
            for alert in alerts:
                ticker = alert.get("ticker", "")
                futures_info = futures_db.get(ticker, {})

                alert_data.append({
                    "Status": get_alert_status(alert),
                    "Symbol": ticker,
                    "Name": futures_info.get("name", alert.get("stock_name", "")),
                    "Category": futures_info.get("category", ""),
                    "Alert Name": alert.get("name", "Unnamed Alert"),
                    "Conditions": format_conditions(alert.get("conditions", alert.get("entry_conditions", {}))),
                    "Timeframe": alert.get("timeframe", "daily"),
                    "Adjustment": alert.get("adjustment_method", "none"),
                    "Last Triggered": alert.get("last_triggered", "Never"),
                    "ID": alert.get("alert_id", "")
                })

            df = pd.DataFrame(alert_data)

            # Display with interactive table
            st.dataframe(
                df,
                use_container_width=True,
                height=400,
                column_config={
                    "Status": st.column_config.TextColumn("Status", width="small"),
                    "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                    "Name": st.column_config.TextColumn("Name", width="medium"),
                    "Category": st.column_config.TextColumn("Category", width="small"),
                    "Alert Name": st.column_config.TextColumn("Alert", width="medium"),
                    "Conditions": st.column_config.TextColumn("Conditions", width="large"),
                    "Timeframe": st.column_config.TextColumn("TF", width="small"),
                    "Adjustment": st.column_config.TextColumn("Adj", width="small"),
                    "Last Triggered": st.column_config.TextColumn("Triggered", width="medium"),
                    "ID": None  # Hide ID column
                }
            )

        with tab2:
            # Active alerts only - for futures, all alerts are active
            active = alerts
            st.subheader(f"Active Futures Alerts ({len(active)})")

            if active:
                for alert in active:
                    ticker = alert.get("ticker", "")
                    futures_info = futures_db.get(ticker, {})

                    col1, col2, col3, col4 = st.columns([2, 3, 2, 2])
                    with col1:
                        st.write(f"**{ticker}**")
                        st.caption(futures_info.get("category", ""))
                    with col2:
                        st.write(alert.get("name", "Unnamed Alert"))
                        st.caption(format_conditions(alert.get("conditions", alert.get("entry_conditions", {})))[:100] + "...")
                    with col3:
                        st.write(f"Adj: {alert.get('adjustment_method', 'none')}")
                        st.caption(f"TF: {alert.get('timeframe', 'daily')}")
                    with col4:
                        if st.button("View", key=f"view_{alert.get('alert_id', '')}"):
                            st.session_state.selected_alert = alert
                    st.divider()
            else:
                st.info("No active futures alerts")

        with tab3:
            # Triggered alerts
            triggered = [a for a in alerts if a.get("last_triggered", "Never") != "Never"]
            st.subheader(f"Triggered Futures Alerts ({len(triggered)})")

            if triggered:
                # Sort by trigger date (most recent first)
                triggered.sort(key=lambda x: x.get("last_triggered", ""), reverse=True)

                for alert in triggered[:20]:  # Show last 20
                    ticker = alert.get("ticker", "")
                    futures_info = futures_db.get(ticker, {})

                    col1, col2, col3, col4 = st.columns([2, 3, 2, 2])
                    with col1:
                        st.write(f"**{ticker}**")
                        st.caption(futures_info.get("category", ""))
                    with col2:
                        st.write(alert.get("name", "Unnamed Alert"))
                        st.caption(format_conditions(alert.get("conditions", alert.get("entry_conditions", {})))[:100] + "...")
                    with col3:
                        triggered_time = alert.get("last_triggered", "Never")
                        try:
                            dt = datetime.fromisoformat(triggered_time.replace('Z', '+00:00'))
                            st.write(dt.strftime("%Y-%m-%d"))
                            st.caption(dt.strftime("%H:%M:%S"))
                        except:
                            st.write(triggered_time)
                    with col4:
                        # For futures alerts, all are considered active (action is Buy/Sell, not on/off)
                        status = "Active"
                        st.write(f"Status: {status}")
                    st.divider()
            else:
                st.success("No triggered futures alerts")

        with tab4:
            # Analytics
            st.subheader("📊 Futures Alerts Analytics")

            col1, col2 = st.columns(2)

            with col1:
                # Most alerted futures
                st.write("**Most Alerted Futures**")
                ticker_counts = {}
                for alert in alerts:
                    ticker = alert.get("ticker", "")
                    ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1

                sorted_tickers = sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True)[:10]
                for ticker, count in sorted_tickers:
                    name = futures_db.get(ticker, {}).get("name", ticker)
                    st.write(f"• {ticker} ({name}): {count} alerts")

            with col2:
                # Trigger frequency
                st.write("**Trigger Frequency**")

                # Count triggers by date
                trigger_dates = {}
                for alert in alerts:
                    triggered = alert.get("last_triggered", "Never")
                    if triggered != "Never":
                        try:
                            dt = datetime.fromisoformat(triggered.replace('Z', '+00:00'))
                            date_str = dt.strftime("%Y-%m-%d")
                            trigger_dates[date_str] = trigger_dates.get(date_str, 0) + 1
                        except:
                            pass

                if trigger_dates:
                    # Show last 7 days
                    sorted_dates = sorted(trigger_dates.items(), reverse=True)[:7]
                    for date, count in sorted_dates:
                        st.write(f"• {date}: {count} triggers")
                else:
                    st.info("No triggers recorded yet")

            st.divider()

            # Category distribution chart
            if len(alerts) > 0:
                st.write("**Alerts by Category**")
                categories = {}
                for alert in alerts:
                    ticker = alert.get("ticker", "")
                    if ticker in futures_db:
                        category = futures_db[ticker].get("category", "Unknown")
                        categories[category] = categories.get(category, 0) + 1

                if categories:
                    import plotly.express as px
                    fig = px.pie(
                        values=list(categories.values()),
                        names=list(categories.keys()),
                        title="Futures Alerts Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # Quick Actions
    st.divider()
    st.subheader("⚡ Quick Actions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("➕ Add Futures Alert", use_container_width=True):
            st.switch_page("pages/Add_Alert.py")

    with col2:
        if st.button("📝 Edit Alerts", use_container_width=True):
            st.switch_page("pages/Edit_Alert.py")

    with col3:
        if st.button("🗑️ Delete Alerts", use_container_width=True):
            st.switch_page("pages/Delete_Alert.py")

    with col4:
        if st.button("📊 View Prices", use_container_width=True):
            st.switch_page("pages/Futures_Price_Database.py")

    # Info box
    st.info("""
    💡 **Tips for Futures Alerts:**
    - Use **Panama adjustment** for price level alerts (e.g., "Alert when ES > 4500")
    - Use **Ratio adjustment** for percentage-based alerts (e.g., "Alert when CL up 5%")
    - Monthly contracts (CL, GC) roll more frequently than quarterly (ES, NQ)
    - Check the Futures Price Database to see historical data with adjustments applied
    """)

if __name__ == "__main__":
    main()

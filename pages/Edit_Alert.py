import os
import time
import streamlit as st
import traceback

st.set_page_config(
    page_title="Edit Alert",
    page_icon="✏️",
    layout="wide",
)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import uuid
from streamlit_tags import st_tags

# Local imports
from src.utils.utils import (
    load_market_data,
    bl_sp,
    predefined_suggestions,
    # All data fetching now uses FMP only
    update_alert,
    update_ratio_alert,
    get_alert_by_id,
    calculate_ratio
)

# Load market data
market_data = load_market_data()

st.header("Edit Stock Alert")

# Get alert ID from session state
alert_id = st.session_state.get("edit_alert_id")

if not alert_id:
    st.error("No alert ID provided. Please go back to the Home page and click 'Edit' on an alert.")
    st.stop()

# Load the alert data
alert = get_alert_by_id(alert_id)

if not alert:
    st.error(f"Alert with ID {alert_id} not found.")
    st.stop()

# Initialize session state if not exists
if "entry_conditions" not in st.session_state:
    st.session_state.entry_conditions = {}
if "entry_combination" not in st.session_state:
    st.session_state.entry_combination = ""
if "indicator_text" not in st.session_state:
    st.session_state.indicator_text = ""
if "parsed_indicators" not in st.session_state:
    st.session_state.parsed_indicators = []

# Check if this is a new alert being loaded
if "current_edit_alert_id" not in st.session_state or st.session_state.current_edit_alert_id != alert_id:
    # Clear session state for new alert
    st.session_state.entry_conditions = {}
    st.session_state.entry_combination = ""
    st.session_state.indicator_text = ""
    st.session_state.parsed_indicators = []
    st.session_state.current_edit_alert_id = alert_id
    st.session_state.last_loaded_alert_id = alert_id

# Check if it's a ratio alert
is_ratio_alert = 'ticker1' in alert and 'ticker2' in alert

st.write(f"**Editing Alert:** {alert['name']}")
st.write(f"**Alert ID:** {alert_id}")
st.write(f"**Stock:** {alert['stock_name']} ({alert['ticker']})")
st.write(f"**Exchange:** {alert['exchange']}")

# Debug: Show the full alert data
with st.expander("Debug: Alert Data"):
    st.json(alert)
    st.write(f"Session State edit_alert_id: {st.session_state.get('edit_alert_id')}")
    st.write(f"Session State current_edit_alert_id: {st.session_state.get('current_edit_alert_id')}")
    st.write(f"Session State entry_conditions count: {len(st.session_state.get('entry_conditions', {}))}")
    st.write(f"Session State entry_combination: {st.session_state.get('entry_combination', '')}")
    st.write(f"Session State indicator_text: {st.session_state.get('indicator_text', '')}")

# Alert name
alert_name = st.text_input("Alert Name", value=alert['name'], key=f"alert_name_{alert_id}")

# Exchange selection
exchange_info = pd.read_csv("market_data.csv")

# Prepare mapping from exchange country name to country code used in cleaned_data
country_to_code = {
    "USA": "US", "Australia": "AUS", "Switzerland": "SW",
    "Italy": "MI", "United Kingdom": "UK", "UK": "UK",
    "Canada": "CA", "Japan": "JP", "Germany": "DE",
    "France": "FR", "Spain": "ES", "Netherlands": "NL",
    "Belgium": "BE", "Ireland": "IE", "Portugal": "PT",
    "Denmark": "DK", "Finland": "FI", "Sweden": "SE",
    "Norway": "NO", "Austria": "AT", "Poland": "PL",
    "Hungary": "HU", "Greece": "GR", "Turkey": "TR",
    "Mexico": "MX", "Czech Republic": "CZ"
}

# Exchange selection by name
exchange_names = exchange_info["Exchange Name"].tolist()
selected_exchange_name = st.selectbox("Select Market Exchange:", exchange_names, index=exchange_names.index(alert['exchange']) if alert['exchange'] in exchange_names else 0, key=f"exchange_{alert_id}")

# Map the selected exchange's country to the proper code
country_name = exchange_info.loc[exchange_info["Exchange Name"] == selected_exchange_name, "Country"].iloc[0]
country_code = country_to_code.get(country_name, country_name)

# Filter stocks from cleaned_data by this country code
filtered_stocks_data = market_data[market_data["Country"] == country_code]
filtered_stocks_display = [f"{row['Name']} ({row['Symbol']})" for _, row in filtered_stocks_data.iterrows()]

if is_ratio_alert:
    # For ratio alerts, we need to handle two stocks
    st.write("**Ratio Alert - Select Two Stocks:**")

    # Find the current stocks in the filtered list
    current_stock1_display = f"{alert['stock_name']} ({alert['ticker1']})"
    current_stock2_display = f"{alert['stock_name']} ({alert['ticker2']})"

    # Default to current values if they exist in the filtered list
    default_index1 = 0
    default_index2 = 0
    if current_stock1_display in filtered_stocks_display:
        default_index1 = filtered_stocks_display.index(current_stock1_display)
    if current_stock2_display in filtered_stocks_display:
        default_index2 = filtered_stocks_display.index(current_stock2_display)

    selected_stock1_display = st.selectbox("Select First Stock:", filtered_stocks_display, index=default_index1, key=f"stock1_{alert_id}")
    selected_stock2_display = st.selectbox("Select Second Stock:", filtered_stocks_display, index=default_index2, key=f"stock2_{alert_id}")

    # Extract stock names and tickers
    stock1_name = selected_stock1_display.split(" (")[0]
    stock2_name = selected_stock2_display.split(" (")[0]
    ticker1 = selected_stock1_display.split("(")[1].split(")")[0]
    ticker2 = selected_stock2_display.split("(")[1].split(")")[0]

else:
    # For regular alerts
    # Find the current stock in the filtered list
    current_stock_display = f"{alert['stock_name']} ({alert['ticker']})"
    default_index = 0
    if current_stock_display in filtered_stocks_display:
        default_index = filtered_stocks_display.index(current_stock_display)

    selected_stocks_display = st.multiselect("Select Stock(s):", filtered_stocks_display, default=[filtered_stocks_display[default_index]] if default_index < len(filtered_stocks_display) else [], key=f"stocks_{alert_id}")

    # Convert display names back to stock names for processing
    selected_stocks = []
    for display_name in selected_stocks_display:
        stock_name = display_name.split(" (")[0]
        selected_stocks.append(stock_name)

# Timeframe selection
timeframe_options = ["1d", "1h", "15m", "5m", "1m"]
selected_timeframe = st.selectbox("Select Timeframe:", timeframe_options, index=timeframe_options.index(alert['timeframe']) if alert['timeframe'] in timeframe_options else 0, key=f"timeframe_{alert_id}")

# Action selection
action_options = ["Buy", "Sell"]
selected_action = st.selectbox("Select Action:", action_options, index=action_options.index(alert['action']) if alert['action'] in action_options else 0, key=f"action_{alert_id}")

# Initialize conditions from existing alert (only if conditions are empty)
if len(st.session_state.entry_conditions) == 0:
    # Pre-populate with existing conditions
    for i, condition in enumerate(alert['conditions']):
        condition_uuid = str(uuid.uuid4())
        st.session_state.entry_conditions[condition_uuid] = condition['conditions']

# Pre-populate combination logic (only if empty)
if st.session_state.entry_combination == "":
    st.session_state.entry_combination = alert.get('combination_logic', '')

# Conditions section
st.write("### Alert Conditions")
st.write("Enter your technical analysis conditions below:")

# Manual condition input
condition_input = st.text_input("Type your condition:", value=st.session_state.indicator_text, key=f"manual_condition_{alert_id}")

if condition_input:
    st.session_state.indicator_text = condition_input

# Add condition button
if st.button("Add Condition", key=f"add_condition_{alert_id}"):
    if st.session_state.indicator_text:
        condition_uuid = str(uuid.uuid4())
        st.session_state.entry_conditions[condition_uuid] = st.session_state.indicator_text
        st.session_state.indicator_text = ""
        st.rerun()

# Display current conditions
st.write("**Current Conditions:**")
for condition_uuid, condition_text in st.session_state.entry_conditions.items():
    col1, col2 = st.columns([3, 1])
    with col1:
        st.write(f"• {condition_text}")
    with col2:
        if st.button("Remove", key=f"remove_{condition_uuid}_{alert_id}"):
            del st.session_state.entry_conditions[condition_uuid]
            st.rerun()

# Combination logic
st.write("### Combination Logic")
st.write("Enter how conditions should be combined (e.g., 'AND', 'OR', 'AND NOT'):")
combination_logic = st.text_input("Combination Logic:", value=st.session_state.entry_combination, key=f"combination_{alert_id}")

# Update session state
st.session_state.entry_combination = combination_logic

# Convert conditions to the expected format
entry_conditions_list = []
for i, (condition_uuid, condition_text) in enumerate(st.session_state.entry_conditions.items(), 1):
    entry_conditions_list.append({
        "index": i,
        "conditions": condition_text
    })

# Submit button
if st.button("Update Alert", key=f"update_alert_{alert_id}"):
    if not alert_name:
        st.error("Please enter an alert name.")
    elif not entry_conditions_list:
        st.error("Please add at least one condition.")
    else:
        try:
            if is_ratio_alert:
                # Update ratio alert
                update_ratio_alert(
                    alert_id=alert_id,
                    name=alert_name,
                    entry_conditions_list=entry_conditions_list,
                    combination_logic=combination_logic,
                    ticker1=ticker1,
                    ticker2=ticker2,
                    stock_name=stock1_name,  # Use first stock name
                    exchange=selected_exchange_name,
                    timeframe=selected_timeframe,
                    last_triggered=alert.get('last_triggered'),
                    action=selected_action,
                    ratio="Yes"
                )
            else:
                # Update regular alert
                if not selected_stocks:
                    st.error("Please select at least one stock.")
                    st.stop()

                failures = []
                successes = []

                for stock_name in selected_stocks:
                    try:
                        # Map to ticker - handle duplicate names with better logic
                        matching_stocks = market_data.loc[market_data["Name"] == stock_name]
                        if len(matching_stocks) == 0:
                            failures.append(f"{stock_name}: Stock not found in database")
                            continue
                        elif len(matching_stocks) > 1:
                            # If multiple stocks with same name, try to find the best match
                            # First, try to find a US stock if we're in US market
                            us_stocks = matching_stocks[matching_stocks["Country"] == "US"]
                            if len(us_stocks) > 0 and selected_exchange_name.upper() == "US":
                                ticker = us_stocks.iloc[0]["Symbol"]
                                st.info(f"Multiple stocks found for '{stock_name}'. Selected US ticker '{ticker}' for US market.")
                            else:
                                # Use the first match but provide more detailed info
                                ticker = matching_stocks.iloc[0]["Symbol"]
                                other_tickers = matching_stocks["Symbol"].tolist()
                                other_tickers.remove(ticker)
                                st.warning(f"Multiple stocks found for '{stock_name}'. Using '{ticker}'. Other options: {', '.join(other_tickers)}")
                        else:
                            ticker = matching_stocks.iloc[0]["Symbol"]

                        update_alert(
                            alert_id=alert_id,
                            name=alert_name,
                            entry_conditions_list=entry_conditions_list,
                            combination_logic=combination_logic,
                            ticker=ticker,
                            stock_name=stock_name,
                            exchange=selected_exchange_name,
                            timeframe=selected_timeframe,
                            last_triggered=alert.get('last_triggered'),
                            action=selected_action,
                            ratio="No"
                        )
                        successes.append(f"{stock_name} ({ticker})")

                    except Exception as e:
                        failures.append(f"{stock_name}: {str(e)}")

                if successes:
                    st.success(f"✅ Alert updated successfully for: {', '.join(successes)}")
                    if failures:
                        st.error(f"❌ Failed for: {', '.join(failures)}")
                else:
                    st.error(f"❌ Failed for all stocks: {', '.join(failures)}")

            # Clear session state and redirect
            st.session_state.entry_conditions = {}
            st.session_state.entry_combination = ""
            st.session_state.indicator_text = ""
            st.success("Alert updated successfully! Redirecting to Home page...")
            time.sleep(2)
            st.switch_page("Home.py")

        except Exception as e:
            st.error(f"Error updating alert: {str(e)}")
            st.error(f"Traceback: {traceback.format_exc()}")

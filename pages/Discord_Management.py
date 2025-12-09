import streamlit as st
import json
import os
import sys
from datetime import datetime

# Add the parent directory to the path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from discord_routing import DiscordEconomyRouter

st.set_page_config(
    page_title="Discord Management",
    page_icon="🔧",
    layout="wide",
)

def main():
    st.title("🔧 Discord Channel Management")
    st.markdown("Configure industry-based Discord channel routing for alerts")
    
    # Initialize router
    router = DiscordEconomyRouter()
    
    # Load current configuration
    config = router.config
    
    st.divider()
    
    # Configuration Status
    st.subheader("📊 Configuration Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        routing_enabled = config.get('enable_industry_routing', False)
        st.metric("Industry Routing", "✅ Enabled" if routing_enabled else "❌ Disabled")
    
    with col2:
        channels = router.get_available_channels()
        configured_channels = sum(1 for ch in channels if ch['configured'])
        st.metric("Configured Channels", f"{configured_channels}/{len(channels)}")
    
    with col3:
        log_enabled = config.get('log_routing_decisions', True)
        st.metric("Routing Logs", "✅ Enabled" if log_enabled else "❌ Disabled")
    
    st.divider()
    
    # Global Settings
    st.subheader("⚙️ Global Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Enable/Disable Industry Routing
        enable_routing = st.checkbox(
            "Enable Industry-Based Routing",
            value=config.get('enable_industry_routing', False),
            help="When enabled, alerts will be sent to industry-specific Discord channels"
        )
    
    with col2:
        # Enable/Disable Routing Logs
        enable_logs = st.checkbox(
            "Log Routing Decisions",
            value=config.get('log_routing_decisions', True),
            help="Log which channel each alert is routed to"
        )
    
    # Save global settings
    if st.button("💾 Save Global Settings"):
        config['enable_industry_routing'] = enable_routing
        config['log_routing_decisions'] = enable_logs
        
        with open('discord_channels_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        st.success("✅ Global settings saved successfully!")
        st.rerun()
    
    st.divider()
    
    # Main tabs for Alert Channels, Custom Channels, and Logging Channels
    main_tab1, main_tab2, main_tab3 = st.tabs(["📢 Alert Channels", "🎯 Custom Channels", "📝 Logging Channels"])
    
    with main_tab1:
        # Channel Configuration
        st.subheader("📢 Discord Alert Channel Configuration")
        st.markdown("Configure webhook URLs for each industry channel")
        
        # Get current channels
        channels = router.get_available_channels()
        
        # Create tabs for different channel categories
        tab1, tab2 = st.tabs(["🏭 Economy Channels", "📊 Special Channels"])
        
        with tab1:
            st.markdown("**Configure economy-based channels:**")
            
            # Economy channels
            economy_channels = [ch for ch in channels if ch['name'] not in ['ETFs', 'Pairs', 'General', 'Failed_Price_Data']]
            
            for channel in economy_channels:
                with st.expander(f"🏭 {channel['name']} - {channel['channel_name']}"):
                    st.markdown(f"**Description:** {channel['description']}")
                    
                    # Get current webhook URL
                    current_webhook = config['channel_mappings'][channel['name']]['webhook_url']
                    
                    # Webhook input
                    webhook_url = st.text_input(
                        f"Discord Webhook URL for {channel['name']}",
                        value=current_webhook if not current_webhook.startswith('YOUR_') else "",
                        type="password",
                        help=f"Enter the Discord webhook URL for {channel['channel_name']}",
                        key=f"webhook_{channel['name']}"
                    )
                    
                    if st.button(f"💾 Save {channel['name']}", key=f"save_{channel['name']}"):
                        if webhook_url:
                            success = router.update_channel_config(channel['name'], webhook_url)
                            if success:
                                st.success(f"✅ {channel['name']} webhook updated!")
                            else:
                                st.error(f"❌ Failed to update {channel['name']}")
                        else:
                            st.warning("⚠️ Please enter a webhook URL")
        
        with tab2:
            st.markdown("**Configure special channels:**")
            
            # Special channels (excluding Failed_Price_Data which moves to logging)
            special_channels = [ch for ch in channels if ch['name'] in ['ETFs', 'Pairs', 'General']]
            
            for channel in special_channels:
                with st.expander(f"🔧 {channel['name']} - {channel['channel_name']}"):
                    st.markdown(f"**Description:** {channel['description']}")
                    
                    # Get current webhook URL
                    current_webhook = config['channel_mappings'][channel['name']]['webhook_url']
                    
                    # Webhook input
                    webhook_url = st.text_input(
                        f"Discord Webhook URL for {channel['name']}",
                        value=current_webhook if not current_webhook.startswith('YOUR_') else "",
                        type="password",
                        help=f"Enter the Discord webhook URL for {channel['channel_name']}",
                        key=f"webhook_special_{channel['name']}"
                    )
                    
                    if st.button(f"💾 Save {channel['name']}", key=f"save_special_{channel['name']}"):
                        if webhook_url:
                            success = router.update_channel_config(channel['name'], webhook_url)
                            if success:
                                st.success(f"✅ {channel['name']} webhook updated!")
                            else:
                                st.error(f"❌ Failed to update {channel['name']}")
                        else:
                            st.warning("⚠️ Please enter a webhook URL")
        
        st.divider()
        
        # Testing Section
        st.subheader("🧪 Test Economy-Based Routing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_ticker = st.text_input(
                "Test Ticker Symbol",
                placeholder="e.g., AAPL, MSFT, XOM",
                help="Enter a ticker symbol to test economy detection",
                key="test_ticker_input"
            )
            
            if test_ticker and st.button("🔍 Test Economy Detection"):
                from discord_routing import get_stock_economy_classification
                
                economy = get_stock_economy_classification(test_ticker)
                if economy:
                    st.success(f"✅ **{test_ticker}** detected as **{economy}**")
                    
                    # Check if channel is configured
                    if economy in config['channel_mappings']:
                        webhook = config['channel_mappings'][economy]['webhook_url']
                        if not webhook.startswith('YOUR_'):
                            st.success(f"✅ Channel configured: {config['channel_mappings'][economy]['channel_name']}")
                        else:
                            st.warning(f"⚠️ Channel not configured: {config['channel_mappings'][economy]['channel_name']}")
                    else:
                        st.warning(f"⚠️ No channel mapping for {economy}")
                else:
                    st.error(f"❌ Could not determine economy for {test_ticker}")
        
        with col2:
            st.markdown("**Test Alert Message:**")
            
            test_alert = {
                'name': 'Test Alert',
                'ticker': test_ticker if test_ticker else 'AAPL',
                'action': 'Buy',
                'ratio': 'No'
            }
            
            test_message = f"📈 **Test Alert Triggered: Test Alert ({test_alert['ticker']})**\n"
            test_message += f"The condition **test_condition** was triggered.\n"
            test_message += f"**Action:** {test_alert['action']}\n"
            test_message += f"**Current Price:** $150.00\n"
            test_message += f"**Time:** Test Time"
            
            if st.button("📤 Send Test Alert"):
                if test_ticker:
                    try:
                        # Direct test without using the router's send_economy_alert
                        # to avoid any async/thread issues in Streamlit
                        import requests

                        # Get the economy for debugging
                        economy = router.get_stock_economy(test_ticker)
                        if not economy:
                            st.warning(f"⚠️ Could not determine economy for {test_ticker}")
                            economy = "General"  # Fallback

                        # Get the channel info for debugging
                        channel_name, webhook_url = router.determine_alert_channel(test_alert)

                        if not webhook_url or webhook_url.startswith('YOUR_'):
                            st.error(f"❌ Webhook not configured for {channel_name}")
                        else:
                            # Send directly to avoid thread issues
                            formatted_message = f"**{channel_name}**\n{test_message}"

                            payload = {
                                "content": formatted_message,
                                "username": "Stock Alert Bot Test"
                            }

                            # Send the webhook request directly
                            response = requests.post(webhook_url, json=payload, timeout=10)

                            if response.status_code == 204:
                                st.success(f"✅ Test alert sent successfully to {channel_name}!")
                            else:
                                st.error(f"❌ Failed to send test alert to {channel_name}. Status: {response.status_code}")
                                if response.text:
                                    st.error(f"Response: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Error sending test alert: {str(e)}")
                else:
                    st.warning("⚠️ Please enter a ticker symbol first")
        
        st.divider()

        # Individual Channel Testing Section
        st.subheader("🧪 Test Individual Channels")
        st.markdown("Test each Discord channel directly to verify webhook configuration:")

        # Create a simple test function that doesn't rely on router
        def send_test_to_channel(channel_name, webhook_url):
            """Send a test message directly to a specific channel"""
            import requests
            from datetime import datetime

            test_msg = f"""**Test Message for {channel_name}**
📊 This is a test message to verify the webhook is working.
⏰ Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
✅ If you see this message, the channel is configured correctly!"""

            payload = {
                "content": test_msg,
                "username": "Channel Test Bot"
            }

            try:
                response = requests.post(webhook_url, json=payload, timeout=10)
                return response.status_code == 204
            except Exception as e:
                return False

        # Get all channels
        channels = config.get('channel_mappings', {})

        # Create columns for channel test buttons
        num_channels = len(channels)
        cols_per_row = 3

        for i in range(0, num_channels, cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                if i + j < num_channels:
                    channel_name = list(channels.keys())[i + j]
                    channel_info = channels[channel_name]
                    webhook = channel_info.get('webhook_url', '')

                    with cols[j]:
                        # Check if webhook is configured
                        if webhook and not webhook.startswith('YOUR_'):
                            if st.button(f"Test {channel_name}", key=f"test_channel_{channel_name}"):
                                with st.spinner(f"Testing {channel_name}..."):
                                    if send_test_to_channel(channel_name, webhook):
                                        st.success(f"✅ {channel_name} working!")
                                    else:
                                        st.error(f"❌ {channel_name} failed!")
                        else:
                            st.button(f"❌ {channel_name}", key=f"test_channel_{channel_name}", disabled=True,
                                    help="Webhook not configured")

        st.divider()

        # Channel Examples Section
        st.subheader("📋 Channel Examples")
        st.markdown("**Examples of assets that will be routed to each channel:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🏭 Economy Channels:**")
            st.markdown("""
            - **Technology:** AAPL, MSFT, GOOGL
            - **Energy:** XOM, CVX, COP
            - **Financials:** JPM, BAC, WFC
            - **Healthcare:** JNJ, PFE, UNH
            - **Consumer Cyclical:** AMZN, TSLA, HD
            - **Consumer Non-Cyclical:** PG, KO, PEP
            - **Industrials:** BA, CAT, HON
            - **Utilities:** NEE, DUK, SO
            - **Materials:** LIN, APD, ECL
            - **Real Estate:** AMT, PLD, CCI
            """)
        
        with col2:
            st.markdown("**📊 Special Channels:**")
            st.markdown("""
            - **ETFs:** SPY, QQQ, VTI, ARKK
            - **Pairs:** All ratio/pair trading alerts
            """)
        
        with col3:
            st.markdown("**🔧 General Channel:**")
            st.markdown("""
            - Unclassified assets
            - Fallback for unknown tickers
            - General market alerts
            - System notifications
            """)
        
        st.divider()
        
        # Information Section
        st.subheader("ℹ️ How It Works")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **🎯 Asset Detection:**
            - Stocks are matched to economy classifications using RBICS data
            - ETFs are identified by asset_type field and routed to ETF channel
            - All ratio/pair alerts go to Pairs channel
            
            **📊 Routing Logic:**
            1. Check if industry routing is enabled
            2. Determine asset type (ETF or Stock)
            3. For stocks, determine economy classification (Technology, Finance, Healthcare, etc.)
            4. Route to appropriate Discord channel based on economy
            5. Fallback to general channel if no match
            """)
        
        with col2:
            st.markdown("""
            **🔧 Configuration:**
            - Update webhook URLs for each economy-based channel
            - Enable/disable industry routing globally
            - Monitor routing decisions in logs
            
            **📈 Benefits:**
            - Organized alerts by economy classification and asset type
            - Reduced notification noise
            - Economy-specific analysis
            - Easy to follow specific economic sectors
            - ETF alerts separated from individual stocks
            """)
    
    with main_tab2:
        # Custom Channels Tab
        st.subheader("🎯 Custom Discord Channels")
        st.markdown("Create custom Discord channels with specific conditions for alert routing")
        
        # Load custom channels configuration
        custom_channels_file = 'custom_discord_channels.json'
        
        def load_custom_channels():
            try:
                with open(custom_channels_file, 'r') as f:
                    return json.load(f)
            except FileNotFoundError:
                return {}
        
        def save_custom_channels(channels):
            with open(custom_channels_file, 'w') as f:
                json.dump(channels, f, indent=2)
        
        custom_channels = load_custom_channels()
        
        # Add New Custom Channel Section
        st.markdown("### ➕ Add New Custom Channel for Alert Conditions")
        st.info("Create Discord channels that receive alerts when specific conditions trigger")
        
        with st.form("add_custom_channel"):
            col1, col2 = st.columns(2)
            
            with col1:
                channel_name = st.text_input(
                    "Channel Name",
                    placeholder="e.g., HARSI Flip Alerts",
                    help="A unique name for this custom channel"
                )
                
                webhook_url = st.text_input(
                    "Discord Webhook URL",
                    type="password",
                    placeholder="https://discord.com/api/webhooks/...",
                    help="The Discord webhook URL for this channel"
                )
                
                description = st.text_area(
                    "Description",
                    placeholder="Describe what alerts this channel will receive",
                    help="Optional description of this channel's purpose",
                    height=100
                )
            
            with col2:
                st.markdown("**Alert Condition to Match**")

                # Condition type selection
                condition_type = st.selectbox(
                    "Condition Type",
                    options=[
                        "Custom (enter manually)",
                        "Price Level (any price comparison)",
                        "RSI - Oversold",
                        "RSI - Overbought",
                        "MACD Crossover",
                        "Moving Average Cross",
                        "Bollinger Bands",
                        "Volume Breakout",
                        "HARSI Flip"
                    ],
                    help="Select a condition type or choose 'Custom' to enter your own"
                )

                # Show appropriate input based on selection
                if condition_type == "Custom (enter manually)":
                    condition_string = st.text_area(
                        "Condition String",
                        placeholder="e.g., HARSI_Flip(period=14, smoothing=3)[-1] > 0",
                        help="Enter the exact condition that should trigger alerts to this channel",
                        height=80
                    )
                elif condition_type == "Price Level (any price comparison)":
                    st.info("This will match ANY condition where price (Close, Open, High, Low) is compared to a number")
                    condition_string = "price_level"
                    st.code("Matches: Close[-1] < 26, Open > 100, High[-1] >= 50, etc.", language="text")
                elif condition_type == "RSI - Oversold":
                    rsi_period = st.number_input("RSI Period", min_value=1, max_value=200, value=14, key="rsi_oversold_period")
                    rsi_level = st.number_input("Oversold Level", min_value=1, max_value=50, value=30, key="rsi_oversold_level")
                    condition_string = f"RSI({rsi_period})[-1] < {rsi_level}"
                    st.code(condition_string, language="python")
                elif condition_type == "RSI - Overbought":
                    rsi_period = st.number_input("RSI Period", min_value=1, max_value=200, value=14, key="rsi_overbought_period")
                    rsi_level = st.number_input("Overbought Level", min_value=50, max_value=100, value=70, key="rsi_overbought_level")
                    condition_string = f"RSI({rsi_period})[-1] > {rsi_level}"
                    st.code(condition_string, language="python")
                elif condition_type == "MACD Crossover":
                    crossover_direction = st.radio("Direction", ["Bullish (Signal > MACD)", "Bearish (Signal < MACD)"], key="macd_direction")
                    if "Bullish" in crossover_direction:
                        condition_string = "MACD_Signal[-1] > MACD[-1]"
                    else:
                        condition_string = "MACD_Signal[-1] < MACD[-1]"
                    st.code(condition_string, language="python")
                elif condition_type == "Moving Average Cross":
                    ma_type = st.selectbox("MA Type", ["EMA", "SMA", "HMA", "KAMA"], key="ma_type")
                    fast_period = st.number_input("Fast Period", min_value=1, max_value=200, value=20, key="ma_fast")
                    slow_period = st.number_input("Slow Period", min_value=1, max_value=200, value=50, key="ma_slow")
                    cross_direction = st.radio("Cross Direction", ["Bullish (Fast > Slow)", "Bearish (Fast < Slow)"], key="ma_direction")
                    operator = ">" if "Bullish" in cross_direction else "<"
                    if ma_type == "EMA":
                        condition_string = f"EMA(timeperiod={fast_period}, input=Close)[-1] {operator} EMA(timeperiod={slow_period}, input=Close)[-1]"
                    elif ma_type == "SMA":
                        condition_string = f"SMA(timeperiod={fast_period}, input=Close)[-1] {operator} SMA(timeperiod={slow_period}, input=Close)[-1]"
                    elif ma_type == "HMA":
                        condition_string = f"HMA(timeperiod={fast_period}, input=Close)[-1] {operator} HMA(timeperiod={slow_period}, input=Close)[-1]"
                    else:  # KAMA
                        condition_string = f"KAMA(length={fast_period})[-1] {operator} KAMA(length={slow_period})[-1]"
                    st.code(condition_string, language="python")
                elif condition_type == "Bollinger Bands":
                    bb_type = st.radio("Breakout Type", ["Upper Band Breakout", "Lower Band Breakdown"], key="bb_type")
                    bb_period = st.number_input("BB Period", min_value=1, max_value=200, value=20, key="bb_period")
                    if "Upper" in bb_type:
                        condition_string = f"Close[-1] > BB_Upper(timeperiod={bb_period})[-1]"
                    else:
                        condition_string = f"Close[-1] < BB_Lower(timeperiod={bb_period})[-1]"
                    st.code(condition_string, language="python")
                elif condition_type == "Volume Breakout":
                    volume_threshold = st.number_input("Volume Threshold", min_value=1000, max_value=100000000, value=1000000, step=100000, key="volume_threshold")
                    condition_string = f"Volume[-1] > {volume_threshold}"
                    st.code(condition_string, language="python")
                elif condition_type == "HARSI Flip":
                    harsi_period = st.number_input("HARSI Period", min_value=1, max_value=200, value=14, key="harsi_period")
                    harsi_smoothing = st.number_input("Smoothing", min_value=1, max_value=10, value=3, key="harsi_smoothing")
                    condition_string = f"HARSI_Flip(period={harsi_period}, smoothing={harsi_smoothing})[-1] > 0"
                    st.code(condition_string, language="python")

                enabled = st.checkbox(
                    "Enable this channel",
                    value=True,
                    help="Enable or disable this custom channel"
                )
            
            submitted = st.form_submit_button("➕ Add Custom Channel", type="primary")
            
            if submitted:
                if not channel_name:
                    st.error("Please enter a channel name")
                elif not webhook_url:
                    st.error("Please enter a webhook URL")
                elif not condition_string:
                    st.error("Please specify a condition string")
                elif channel_name in custom_channels:
                    st.error(f"A channel named '{channel_name}' already exists")
                else:
                    # Add the new custom channel with condition string
                    custom_channels[channel_name] = {
                        'webhook_url': webhook_url,
                        'description': description,
                        'condition': condition_string.strip(),  # Store the exact condition string
                        'enabled': enabled,
                        'created': datetime.now().isoformat(),
                        'channel_name': f"#{channel_name.lower().replace(' ', '-')}"
                    }
                    
                    save_custom_channels(custom_channels)
                    st.success(f"✅ Custom channel '{channel_name}' added successfully!")
                    st.rerun()
        
        st.divider()
        
        # Manage Existing Custom Channels
        st.markdown("### 📋 Existing Custom Condition Channels")
        
        if custom_channels:
            for name, channel_info in custom_channels.items():
                with st.expander(f"🎯 {name} - {channel_info.get('channel_name', 'N/A')}"):
                    col1, col2, col3 = st.columns([2, 2, 1])
                    
                    with col1:
                        st.markdown(f"**Description:** {channel_info.get('description', 'No description')}")
                        
                        # Display the condition string
                        condition = channel_info.get('condition', channel_info.get('condition_value', ''))
                        st.markdown(f"**Condition:**")
                        st.code(condition, language='python')
                        
                        st.markdown(f"**Status:** {'✅ Enabled' if channel_info.get('enabled', False) else '❌ Disabled'}")
                    
                    with col2:
                        # Test button
                        if st.button(f"🧪 Test", key=f"test_custom_{name}"):
                            import requests
                            test_condition = channel_info.get('condition', 'Unknown')
                            test_message = {
                                "embeds": [{
                                    "title": f"🧪 Test Alert - {name}",
                                    "description": "This is a test message for your custom condition channel",
                                    "color": 3066993,
                                    "fields": [
                                        {"name": "Channel", "value": name, "inline": True},
                                        {"name": "Condition", "value": test_condition[:100] + "..." if len(test_condition) > 100 else test_condition, "inline": False},
                                        {"name": "Status", "value": "✅ Working", "inline": True}
                                    ],
                                    "footer": {"text": f"Condition Channel Test - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"}
                                }]
                            }
                            
                            try:
                                response = requests.post(channel_info['webhook_url'], json=test_message, timeout=5)
                                if response.status_code == 204:
                                    st.success("✅ Test message sent successfully!")
                                else:
                                    st.error(f"Failed: {response.status_code}")
                            except Exception as e:
                                st.error(f"Error: {e}")
                        
                        # Enable/Disable toggle
                        new_enabled = st.checkbox(
                            "Enabled",
                            value=channel_info.get('enabled', False),
                            key=f"enable_custom_{name}"
                        )
                        
                        if new_enabled != channel_info.get('enabled', False):
                            custom_channels[name]['enabled'] = new_enabled
                            save_custom_channels(custom_channels)
                            st.success(f"✅ Channel {'enabled' if new_enabled else 'disabled'}")
                            st.rerun()
                    
                    with col3:
                        # Delete button
                        if st.button(f"🗑️ Delete", key=f"delete_custom_{name}", type="secondary"):
                            del custom_channels[name]
                            save_custom_channels(custom_channels)
                            st.success(f"✅ Channel '{name}' deleted")
                            st.rerun()
        else:
            st.info("No custom channels configured yet. Add one above to get started!")
        
        st.divider()
        
        # Instructions
        st.markdown("### 📖 How Condition-Based Channels Work")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Setting Up Condition Channels:**
            1. **Create a Discord Webhook** in your desired channel
            2. **Enter the exact condition string** that triggers alerts
            3. **Test the channel** to ensure webhook is working
            4. **Enable the channel** to start receiving alerts
            
            **How Matching Works:**
            - When an alert triggers, its condition is compared to your custom channels
            - If the condition matches exactly, the alert is sent to that channel
            - The same alert can go to multiple channels if conditions match
            - All matching alerts are sent, regardless of ticker or exchange
            """)
        
        with col2:
            st.markdown("""
            **Example Conditions:**
            - `HARSI_Flip(period=14, smoothing=3)[-1] > 0`
            - `RSI(14) < 30` - Oversold RSI alerts
            - `MACD_Signal > MACD` - MACD crossovers
            - `BB_Upper < close` - Bollinger Band breakouts
            - `Volume > 1000000` - High volume alerts
            
            **Use Cases:**
            - 📊 Group all HARSI flip alerts in one channel
            - 🎯 Separate different RSI levels into channels
            - 📈 Create channels for specific indicator setups
            - 🔔 Monitor particular conditions across all stocks
            """)
    
    with main_tab3:
        # Logging Channels Tab
        st.subheader("📝 Discord Logging Channel Configuration")
        st.markdown("Configure webhook URLs for logging channels that monitor alert processing")
        
        # Create tabs for logging configuration
        log_tab_failed, log_tab_scheduler = st.tabs(['Failed Price Data', 'Scheduler Settings'])


        with log_tab_failed:
            st.markdown("### ⚠️ Failed Price Data Channel")
            st.info("💡 This channel receives notifications when alerts fail to fetch price data from the FMP API")
            
            # Get Failed_Price_Data channel from main config
            failed_channel_exists = 'Failed_Price_Updates' in config.get('channel_mappings', {})

            if failed_channel_exists:
                current_webhook = config['channel_mappings']['Failed_Price_Updates']['webhook_url']
                
                webhook_url = st.text_input(
                    "Discord Webhook URL for Failed Price Data",
                    value=current_webhook if not current_webhook.startswith('YOUR_') else "",
                    type="password",
                    help="Enter the Discord webhook URL for Failed Price Data notifications",
                    key="webhook_failed_price"
                )
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("💾 Save Failed Price Data Channel", key="save_failed"):
                        if webhook_url:
                            success = router.update_channel_config('Failed_Price_Data', webhook_url)
                            if success:
                                st.success("✅ Failed Price Data webhook updated!")
                            else:
                                st.error("❌ Failed to update")
                        else:
                            st.warning("⚠️ Please enter a webhook URL")
                
                with col2:
                    if st.button("🧪 Test Failed Alert", key="test_failed"):
                        if webhook_url and not webhook_url.startswith('YOUR_'):
                            import requests
                            test_message = {
                                "embeds": [{
                                    "title": "🧪 Test Failed Price Alert",
                                    "description": "This is a test message for the Failed Price Data channel",
                                    "color": 15158332,
                                    "fields": [
                                        {"name": "Alert Name", "value": "Test Alert", "inline": True},
                                        {"name": "Ticker", "value": "TEST-US", "inline": True},
                                        {"name": "Error", "value": "Test error message", "inline": True},
                                        {"name": "Timestamp", "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "inline": False}
                                    ],
                                    "footer": {"text": "This is a test message"}
                                }]
                            }
                            
                            try:
                                response = requests.post(webhook_url, json=test_message, timeout=5)
                                if response.status_code == 204:
                                    st.success("✅ Test message sent successfully!")
                                else:
                                    st.error(f"Failed: {response.status_code}")
                            except Exception as e:
                                st.error(f"Error: {e}")
                        else:
                            st.warning("Please configure the webhook URL first")
        
        with log_tab_scheduler:
            st.markdown("### 🔔 Scheduler Settings")
            st.info("""
            Configure a dedicated Discord channel for scheduler status notifications.
            This channel will receive:
            - 🚀 Market check **start** notifications  
            - ⏳ Progress updates during processing
            - 🏁 Market check **completion** notifications with summary
            """)
            
            # Load scheduler configuration
            def load_scheduler_config():
                try:
                    with open('scheduler_config.json', 'r') as f:
                        return json.load(f)
                except:
                    return {
                        "scheduler_webhook": {
                            "url": "",
                            "name": "Scheduler Status Channel",
                            "enabled": False
                        },
                        "notification_settings": {
                            "send_start_notification": True,
                            "send_progress_updates": True,
                            "send_completion_notification": True,
                            "progress_update_interval": 500,
                            "include_summary_stats": True
                        }
                    }
            
            def save_scheduler_config(scheduler_config):
                with open('scheduler_config.json', 'w') as f:
                    json.dump(scheduler_config, f, indent=4)
            
            scheduler_config = load_scheduler_config()
            
            # Webhook configuration
            st.markdown("#### 📡 Scheduler Discord Webhook")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                scheduler_webhook_url = st.text_input(
                    "Discord Webhook URL for Scheduler Status",
                    value=scheduler_config['scheduler_webhook']['url'] if scheduler_config['scheduler_webhook']['url'] else "",
                    placeholder="https://discord.com/api/webhooks/...",
                    help="Create a webhook in your Discord channel: Server Settings → Integrations → Webhooks → New Webhook",
                    key="scheduler_webhook_url"
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🧪 Test Scheduler Webhook", type="secondary", key="test_scheduler_webhook"):
                    if scheduler_webhook_url:
                        import requests
                        test_message = {
                            "embeds": [{
                                "title": "🧪 Test Scheduler Notification",
                                "description": "This is a test message from the scheduler",
                                "color": 3066993,
                                "fields": [
                                    {"name": "Type", "value": "Test Message", "inline": True},
                                    {"name": "Status", "value": "✅ Working", "inline": True},
                                    {"name": "Time", "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "inline": False}
                                ],
                                "footer": {"text": "Scheduler Status Channel Test"}
                            }]
                        }
                        
                        try:
                            response = requests.post(scheduler_webhook_url, json=test_message, timeout=5)
                            if response.status_code == 204:
                                st.success("✅ Test message sent successfully!")
                            else:
                                st.error(f"Failed to send: {response.status_code}")
                        except Exception as e:
                            st.error(f"Error: {e}")
                    else:
                        st.warning("Please enter a webhook URL first")
            
            # Enable/Disable scheduler notifications
            scheduler_enabled = st.checkbox(
                "Enable Scheduler Notifications",
                value=scheduler_config['scheduler_webhook'].get('enabled', False),
                help="Enable or disable scheduler status notifications to Discord",
                key="scheduler_enabled"
            )
            
            # Notification Settings
            st.markdown("#### 📋 Notification Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                send_start = st.checkbox(
                    "🚀 Send Start Notifications",
                    value=scheduler_config['notification_settings']['send_start_notification'],
                    help="Send a notification when a market check begins",
                    key="send_start"
                )
                
                send_progress = st.checkbox(
                    "⏳ Send Progress Updates",
                    value=scheduler_config['notification_settings']['send_progress_updates'],
                    help="Send periodic updates during alert processing",
                    key="send_progress"
                )
                
                if send_progress:
                    progress_interval = st.number_input(
                        "Progress Update Interval (alerts)",
                        min_value=100,
                        max_value=2000,
                        value=scheduler_config['notification_settings']['progress_update_interval'],
                        step=100,
                        help="Send a progress update every N alerts processed",
                        key="progress_interval"
                    )
                else:
                    progress_interval = scheduler_config['notification_settings']['progress_update_interval']
            
            with col2:
                send_completion = st.checkbox(
                    "🏁 Send Completion Notifications",
                    value=scheduler_config['notification_settings']['send_completion_notification'],
                    help="Send a notification when a market check completes",
                    key="send_completion"
                )
                
                include_stats = st.checkbox(
                    "📊 Include Summary Statistics",
                    value=scheduler_config['notification_settings']['include_summary_stats'],
                    help="Include detailed statistics in completion notifications",
                    key="include_stats"
                )
            
            # Save scheduler settings button
            if st.button("💾 Save Scheduler Settings", type="primary", key="save_scheduler_settings"):
                # Update configuration
                scheduler_config['scheduler_webhook']['url'] = scheduler_webhook_url
                scheduler_config['scheduler_webhook']['enabled'] = scheduler_enabled and bool(scheduler_webhook_url)
                scheduler_config['notification_settings']['send_start_notification'] = send_start
                scheduler_config['notification_settings']['send_progress_updates'] = send_progress
                scheduler_config['notification_settings']['send_completion_notification'] = send_completion
                scheduler_config['notification_settings']['progress_update_interval'] = progress_interval
                scheduler_config['notification_settings']['include_summary_stats'] = include_stats
                
                # Save to file
                save_scheduler_config(scheduler_config)
                st.success("✅ Scheduler settings saved successfully!")
                
                if scheduler_enabled and scheduler_webhook_url:
                    st.info("🔔 Scheduler notifications are now active. You'll receive notifications in your Discord channel when market checks run.")
                elif not scheduler_webhook_url:
                    st.warning("⚠️ Please configure a webhook URL to enable notifications")
                else:
                    st.info("🔕 Scheduler notifications are disabled")
            
            st.divider()

            # Hourly Scheduler Status Channel Configuration
            st.markdown("### 📊 Hourly Scheduler Status")
            st.info("""
            Configure a dedicated Discord channel for **hourly data scheduler** status notifications.
            This channel will receive:
            - 🔄 Update **start** notifications with exchanges and candle types
            - ✅ Update **completion** notifications with stats (symbols updated, exchanges, duration)
            - ⏭️ Update **skipped** notifications when markets are closed
            - ❌ **Error** notifications if updates fail
            """)

            # Load hourly scheduler configuration from discord_channels_config.json
            hourly_config = config.get('logging_channels', {}).get('Hourly_Scheduler_Status', {})

            # Webhook configuration
            st.markdown("#### 📡 Hourly Scheduler Discord Webhook")

            col1, col2 = st.columns([3, 1])

            with col1:
                hourly_webhook_url = st.text_input(
                    "Discord Webhook URL for Hourly Scheduler Status",
                    value=hourly_config.get('webhook_url', 'YOUR_WEBHOOK_URL_HERE') if hourly_config.get('webhook_url') != 'YOUR_WEBHOOK_URL_HERE' else "",
                    placeholder="https://discord.com/api/webhooks/...",
                    help="Create a webhook in your Discord channel: Server Settings → Integrations → Webhooks → New Webhook",
                    key="hourly_webhook_url"
                )

            with col2:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🧪 Test Hourly Webhook", type="secondary", key="test_hourly_webhook"):
                    if hourly_webhook_url:
                        import requests
                        test_message = {
                            "embeds": [{
                                "title": "📊 Hourly Data Scheduler",
                                "description": "🧪 Test notification from hourly scheduler",
                                "color": 52479,  # Blue
                                "fields": [
                                    {"name": "Type", "value": "Test Message", "inline": True},
                                    {"name": "Status", "value": "✅ Working", "inline": True},
                                    {"name": "Time", "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), "inline": False}
                                ],
                                "footer": {"text": "Hourly Scheduler Status Test"}
                            }]
                        }

                        try:
                            response = requests.post(hourly_webhook_url, json=test_message, timeout=5)
                            if response.status_code == 204:
                                st.success("✅ Test message sent successfully!")
                            else:
                                st.error(f"Failed to send: {response.status_code}")
                        except Exception as e:
                            st.error(f"Error: {e}")
                    else:
                        st.warning("Please enter a webhook URL first")

            # Save hourly scheduler webhook button
            if st.button("💾 Save Hourly Scheduler Webhook", type="primary", key="save_hourly_webhook"):
                # Update configuration
                if 'logging_channels' not in config:
                    config['logging_channels'] = {}
                if 'Hourly_Scheduler_Status' not in config['logging_channels']:
                    config['logging_channels']['Hourly_Scheduler_Status'] = {}

                config['logging_channels']['Hourly_Scheduler_Status']['webhook_url'] = hourly_webhook_url if hourly_webhook_url else "YOUR_WEBHOOK_URL_HERE"
                config['logging_channels']['Hourly_Scheduler_Status']['channel_name'] = "#hourly-scheduler-status"
                config['logging_channels']['Hourly_Scheduler_Status']['description'] = "Hourly data scheduler status updates and monitoring"

                # Save to discord_channels_config.json
                with open('discord_channels_config.json', 'w') as f:
                    json.dump(config, f, indent=2)

                st.success("✅ Hourly scheduler webhook saved successfully!")

                if hourly_webhook_url:
                    st.info("🔔 Hourly scheduler notifications are now active. You'll receive notifications when hourly updates run.")
                else:
                    st.warning("⚠️ No webhook URL configured. Notifications are disabled.")

            # Status indicator
            if hourly_config.get('webhook_url') and hourly_config.get('webhook_url') != 'YOUR_WEBHOOK_URL_HERE':
                st.success("✅ Hourly scheduler notifications are configured")
                st.caption(f"**Channel:** {hourly_config.get('channel_name', '#hourly-scheduler-status')}")
            else:
                st.warning("⚠️ Hourly scheduler notifications are not configured. Add a webhook URL above to receive notifications.")

            st.divider()

            # Show current schedule
            st.markdown("#### 📅 Current Schedule")
            
            schedule_info = """
            **Daily Market Checks (Monday-Friday):**
            - 🌏 **12:15 AM EST** - Australia/NZ Markets (~590 alerts, ~5 min)
            - 🌏 **2:15 AM EST** - Asia Main Markets (~5,580 alerts, ~47 min)  
            - 🌏 **3:00 AM EST** - Other Asia Markets (~512 alerts, ~4 min)
            - 🌍 **11:45 AM EST** - European Markets (~3,528 alerts, ~29 min)
            - 🌎 **4:15 PM EST** - US Markets (~9,192 alerts, ~77 min)
            - 🌎 **4:30 PM EST** - Canada & Other Markets (~1,277 alerts, ~11 min)
            - 🌎 **5:00 PM EST** - Latin America Markets (~512 alerts, ~4 min)
            
            **Weekly:**
            - 📊 **Sunday 8:00 PM EST** - Comprehensive check of all alerts
            """
            
            st.markdown(schedule_info)
            
            # Status indicator
            if scheduler_config['scheduler_webhook'].get('enabled') and scheduler_config['scheduler_webhook'].get('url'):
                st.success("✅ Scheduler notifications are configured and active")
            else:
                st.warning("⚠️ Scheduler notifications are not configured. Configure a webhook URL above to receive notifications.")

if __name__ == "__main__":
    main()
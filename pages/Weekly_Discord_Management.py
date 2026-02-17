import streamlit as st
import os
import sys

# Ensure project root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_access.document_store import clear_cache, save_document  # noqa: E402
from src.services.discord_routing import DiscordEconomyRouter, get_stock_economy_classification  # noqa: E402

st.set_page_config(
    page_title="Weekly Discord Management",
    page_icon="📆",
    layout="wide",
)


def main():
    st.title("📆 Weekly Discord Channel Management")
    st.markdown("Configure dedicated Discord webhooks for **weekly timeframe** alerts.")

    router = DiscordEconomyRouter()
    config = router.config

    weekly_channels = router.get_available_channels(timeframe='weekly')
    total_channels = len(weekly_channels)
    configured_channels = sum(1 for ch in weekly_channels if ch['configured'])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Weekly Routing", "Enabled" if config.get('enable_industry_routing', False) else "Disabled")
    with col2:
        st.metric("Configured Channels", f"{configured_channels}/{total_channels}")
    with col3:
        st.metric("Routing Logs", "Enabled" if config.get('log_routing_decisions', True) else "Disabled")

    st.divider()

    if st.button("Copy Base Webhooks → Weekly", help="Populate weekly channels using the base (default) webhook URLs"):
        base_mapping = config.get('channel_mappings', {})
        weekly_mapping = config.get('channel_mappings_weekly', {})
        for name, base_cfg in base_mapping.items():
            if name in weekly_mapping and base_cfg.get('webhook_url'):
                weekly_mapping[name]['webhook_url'] = base_cfg['webhook_url']
        config['channel_mappings_weekly'] = weekly_mapping
        save_document(
            "discord_channels_config",
            config,
            fallback_path="discord_channels_config.json",
        )
        clear_cache("discord_channels_config")
        st.success("Copied base webhooks into weekly configuration.")
        st.rerun()

    st.subheader("🔁 Configure Weekly Channels")
    st.markdown("Update the webhook URL for each weekly economy or special channel.")

    special_channels = {'ETFs', 'Pairs', 'General', 'Futures', 'Failed_Price_Updates'}
    economy_channels = [ch for ch in weekly_channels if ch['name'] not in special_channels]
    special_channel_configs = [ch for ch in weekly_channels if ch['name'] in special_channels]

    tab_economy, tab_special = st.tabs(["🌍 Economy Channels", "⭐ Special Channels"])

    with tab_economy:
        for channel in economy_channels:
            cfg = config['channel_mappings_weekly'][channel['name']]
            current_url = cfg.get('webhook_url', '')
            with st.expander(f"🌍 {channel['name']} — {cfg.get('channel_name', channel['name'])}"):
                st.markdown(f"**Description:** {cfg.get('description', 'No description provided.')}")
                webhook = st.text_input(
                    f"Webhook for {channel['name']}",
                    value=current_url if current_url else "",
                    type="password",
                    key=f"weekly_webhook_{channel['name']}"
                )
                if st.button(f"💾 Save {channel['name']}", key=f"save_weekly_{channel['name']}"):
                    if webhook:
                        success = router.update_channel_config(channel['name'], webhook, timeframe='weekly')
                        if success:
                            st.success(f"Updated weekly webhook for {channel['name']}")
                        else:
                            st.error(f"Failed to update {channel['name']}")
                    else:
                        st.warning("Please enter a webhook URL before saving.")

    with tab_special:
        for channel in special_channel_configs:
            cfg = config['channel_mappings_weekly'][channel['name']]
            current_url = cfg.get('webhook_url', '')
            with st.expander(f"⭐ {channel['name']} — {cfg.get('channel_name', channel['name'])}"):
                st.markdown(f"**Description:** {cfg.get('description', 'No description provided.')}")
                webhook = st.text_input(
                    f"Webhook for {channel['name']}",
                    value=current_url if current_url else "",
                    type="password",
                    key=f"weekly_webhook_special_{channel['name']}"
                )
                if st.button(f"💾 Save {channel['name']}", key=f"save_weekly_special_{channel['name']}"):
                    if webhook:
                        success = router.update_channel_config(channel['name'], webhook, timeframe='weekly')
                        if success:
                            st.success(f"Updated weekly webhook for {channel['name']}")
                        else:
                            st.error(f"Failed to update {channel['name']}")
                    else:
                        st.warning("Please enter a webhook URL before saving.")

    st.divider()

    st.subheader("🧪 Test Weekly Routing")
    st.caption(
        "Enter a ticker to see which weekly channel it would use. Use **Send test message** to post a test to that channel (e.g. tech-weekly-alerts)."
    )
    test_ticker = st.text_input(
        "Ticker Symbol",
        placeholder="e.g., AAPL, MSFT, XOM",
        help="Enter a ticker to see which weekly channel it would use.",
        key="weekly_test_ticker",
    )

    if test_ticker and st.button("🔍 Test Weekly Channel", key="test_weekly_btn"):
        economy = get_stock_economy_classification(test_ticker)
        if economy:
            st.success(f"**{test_ticker.upper()}** detected as **{economy}**")
            weekly_cfg = config['channel_mappings_weekly'].get(economy)
            if weekly_cfg and weekly_cfg.get('webhook_url'):
                display_name = weekly_cfg.get('channel_name', economy)
                st.info(f"Weekly channel: **{display_name}**")
                st.session_state['_weekly_test_economy'] = economy
                st.session_state['_weekly_test_channel_name'] = display_name
                st.session_state['_weekly_test_webhook'] = weekly_cfg.get('webhook_url')
            else:
                st.warning("Weekly channel for this economy is not configured (no webhook URL).")
                for k in ['_weekly_test_economy', '_weekly_test_channel_name', '_weekly_test_webhook']:
                    st.session_state.pop(k, None)
        else:
            st.error("Could not determine the economy for that ticker.")
            for k in ['_weekly_test_economy', '_weekly_test_channel_name', '_weekly_test_webhook']:
                st.session_state.pop(k, None)

    if st.session_state.get('_weekly_test_webhook') and st.button("📤 Send test message to this channel", key="send_weekly_test"):
        import requests
        from datetime import datetime
        channel_name = st.session_state.get('_weekly_test_channel_name', 'weekly')
        webhook_url = st.session_state['_weekly_test_webhook']
        test_msg = (
            f"**Test message – Weekly routing**\n"
            f"📊 This is a test from **Weekly Discord Management** to verify the webhook for **{channel_name}**.\n"
            f"⏰ Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"✅ If you see this, the channel is configured correctly."
        )
        payload = {"content": test_msg, "username": "Weekly Alert Test"}
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            if response.status_code == 204:
                st.success(f"✅ Test message sent to **{channel_name}**.")
            else:
                st.error(f"Failed to send: HTTP {response.status_code}. Check webhook URL.")
        except Exception as e:
            st.error(f"Error sending test message: {e}")


# Run when loaded as Streamlit page or as script
main()

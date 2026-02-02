import streamlit as st
import json
import os
import sys

# Ensure project root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.discord_routing import DiscordEconomyRouter, get_stock_economy_classification  # noqa: E402

st.set_page_config(
    page_title="Hourly Discord Management",
    page_icon="⏰",
    layout="wide",
)


def main():
    st.title("⏰ Hourly Discord Channel Management")
    st.markdown("Configure dedicated Discord webhooks for **hourly timeframe** alerts.")

    router = DiscordEconomyRouter()
    config = router.config

    hourly_channels = router.get_available_channels(timeframe='hourly')
    total_channels = len(hourly_channels)
    configured_channels = sum(1 for ch in hourly_channels if ch['configured'])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Hourly Routing", "Enabled" if config.get('enable_industry_routing', False) else "Disabled")
    with col2:
        st.metric("Configured Channels", f"{configured_channels}/{total_channels}")
    with col3:
        st.metric("Routing Logs", "Enabled" if config.get('log_routing_decisions', True) else "Disabled")

    st.divider()

    if st.button("Copy Daily Webhooks → Hourly", help="Populate hourly channels using the existing daily webhook URLs"):
        daily_mapping = config.get('channel_mappings', {})
        hourly_mapping = config.get('channel_mappings_hourly', {})
        for name, daily_cfg in daily_mapping.items():
            if name in hourly_mapping and daily_cfg.get('webhook_url'):
                hourly_mapping[name]['webhook_url'] = daily_cfg['webhook_url']
        config['channel_mappings_hourly'] = hourly_mapping
        with open('discord_channels_config.json', 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        st.success("Copied daily webhooks into hourly configuration.")
        st.rerun()

    st.subheader("🔁 Configure Hourly Channels")
    st.markdown("Update the webhook URL for each hourly economy or special channel.")

    special_channels = {'ETFs', 'Pairs', 'General', 'Futures', 'Failed_Price_Updates'}
    economy_channels = [ch for ch in hourly_channels if ch['name'] not in special_channels]
    special_channel_configs = [ch for ch in hourly_channels if ch['name'] in special_channels]

    tab_economy, tab_special = st.tabs(["🌍 Economy Channels", "⭐ Special Channels"])

    with tab_economy:
        for channel in economy_channels:
            cfg = config['channel_mappings_hourly'][channel['name']]
            current_url = cfg.get('webhook_url', '')
            with st.expander(f"🌍 {channel['name']} — {cfg.get('channel_name', channel['name'])}"):
                st.markdown(f"**Description:** {cfg.get('description', 'No description provided.')}")
                webhook = st.text_input(
                    f"Webhook for {channel['name']}",
                    value=current_url if current_url else "",
                    type="password",
                    key=f"hourly_webhook_{channel['name']}"
                )
                if st.button(f"💾 Save {channel['name']}", key=f"save_hourly_{channel['name']}"):
                    if webhook:
                        success = router.update_channel_config(channel['name'], webhook, timeframe='hourly')
                        if success:
                            st.success(f"Updated hourly webhook for {channel['name']}")
                        else:
                            st.error(f"Failed to update {channel['name']}")
                    else:
                        st.warning("Please enter a webhook URL before saving.")

    with tab_special:
        for channel in special_channel_configs:
            cfg = config['channel_mappings_hourly'][channel['name']]
            current_url = cfg.get('webhook_url', '')
            with st.expander(f"⭐ {channel['name']} — {cfg.get('channel_name', channel['name'])}"):
                st.markdown(f"**Description:** {cfg.get('description', 'No description provided.')}")
                webhook = st.text_input(
                    f"Webhook for {channel['name']}",
                    value=current_url if current_url else "",
                    type="password",
                    key=f"hourly_webhook_special_{channel['name']}"
                )
                if st.button(f"💾 Save {channel['name']}", key=f"save_hourly_special_{channel['name']}"):
                    if webhook:
                        success = router.update_channel_config(channel['name'], webhook, timeframe='hourly')
                        if success:
                            st.success(f"Updated hourly webhook for {channel['name']}")
                        else:
                            st.error(f"Failed to update {channel['name']}")
                    else:
                        st.warning("Please enter a webhook URL before saving.")

    st.divider()

    st.subheader("🧪 Test Hourly Routing")
    st.caption(
        "Enter a ticker to see which hourly channel it would use. Use **Send test message** to post a test to that channel (e.g. tech-hourly-alerts)."
    )
    test_ticker = st.text_input(
        "Ticker Symbol",
        placeholder="e.g., AAPL, MSFT, XOM",
        help="Enter a ticker to see which hourly channel it would use."
    )

    if test_ticker and st.button("🔍 Test Hourly Channel"):
        economy = get_stock_economy_classification(test_ticker)
        if economy:
            st.success(f"**{test_ticker.upper()}** detected as **{economy}**")
            hourly_cfg = config['channel_mappings_hourly'].get(economy)
            if hourly_cfg and hourly_cfg.get('webhook_url'):
                display_name = hourly_cfg.get('channel_name', economy)
                st.info(f"Hourly channel: **{display_name}**")
                st.session_state['_hourly_test_economy'] = economy
                st.session_state['_hourly_test_channel_name'] = display_name
                st.session_state['_hourly_test_webhook'] = hourly_cfg.get('webhook_url')
            else:
                st.warning("Hourly channel for this economy is not configured (no webhook URL).")
                for k in ['_hourly_test_economy', '_hourly_test_channel_name', '_hourly_test_webhook']:
                    st.session_state.pop(k, None)
        else:
            st.error("Could not determine the economy for that ticker.")
            for k in ['_hourly_test_economy', '_hourly_test_channel_name', '_hourly_test_webhook']:
                st.session_state.pop(k, None)

    # Send test message to the channel resolved above (e.g. tech-hourly-alerts)
    if st.session_state.get('_hourly_test_webhook') and st.button("📤 Send test message to this channel"):
        import requests
        from datetime import datetime
        channel_name = st.session_state.get('_hourly_test_channel_name', 'hourly')
        webhook_url = st.session_state['_hourly_test_webhook']
        test_msg = (
            f"**Test message – Hourly routing**\n"
            f"📊 This is a test from **Hourly Discord Management** to verify the webhook for **{channel_name}**.\n"
            f"⏰ Sent at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"✅ If you see this, the channel is configured correctly."
        )
        payload = {"content": test_msg, "username": "Hourly Alert Test"}
        try:
            response = requests.post(webhook_url, json=payload, timeout=10)
            if response.status_code == 204:
                st.success(f"✅ Test message sent to **{channel_name}**.")
            else:
                st.error(f"Failed to send: HTTP {response.status_code}. Check webhook URL.")
        except Exception as e:
            st.error(f"Error sending test message: {e}")


if __name__ == "__main__":
    main()

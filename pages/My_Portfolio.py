"""
My Portfolio - Manage multiple portfolios with individual Discord webhooks
"""

import streamlit as st
import json
from datetime import datetime
import pandas as pd
import uuid

st.set_page_config(page_title="My Portfolios", page_icon="💼", layout="wide")

from src.utils.utils import load_market_data
from src.data_access.portfolio_repository import (
    list_portfolios as repo_list_portfolios,
    save_portfolio as repo_save_portfolio,
    delete_portfolio as repo_delete_portfolio,
)


def load_portfolios():
    return repo_list_portfolios()

def create_portfolio(name, webhook_url=""):
    """Create a new portfolio"""
    portfolio_id = str(uuid.uuid4())[:8]  # Short unique ID
    return {
        "id": portfolio_id,
        "name": name,
        "stocks": [],
        "discord_webhook": webhook_url,
        "enabled": True,
        "created_date": datetime.now().isoformat(),
        "last_updated": datetime.now().isoformat()
    }

def load_available_symbols():
    """Load all available symbols with company names from metadata."""
    try:
        market_df = load_market_data()
        if isinstance(market_df, pd.DataFrame) and not market_df.empty:
            return {
                str(row.Symbol): row.Name if isinstance(row.Name, str) else str(row.Symbol)
                for row in market_df.itertuples()
            }
    except Exception:
        pass
    return {}

# Header
st.title("💼 My Portfolios")
st.markdown("Manage multiple portfolios with individual Discord alert channels")
st.markdown("---")

# Load portfolios
portfolios = load_portfolios()

# Initialize session state for selected portfolio
if "selected_portfolio_id" not in st.session_state:
    st.session_state.selected_portfolio_id = None

# Sidebar - Portfolio Management
with st.sidebar:
    st.header("📁 Portfolio Management")

    # Create new portfolio section
    with st.expander("➕ Create New Portfolio", expanded=not portfolios):
        new_portfolio_name = st.text_input(
            "Portfolio Name",
            placeholder="e.g., Tech Stocks, Dividend Portfolio",
            key="new_portfolio_name"
        )

        new_webhook_url = st.text_input(
            "Discord Webhook URL",
            type="password",
            placeholder="https://discord.com/api/webhooks/...",
            key="new_webhook_url",
            help="Each portfolio can have its own Discord channel"
        )

        if st.button("Create Portfolio", type="primary", use_container_width=True):
            if new_portfolio_name:
                # Check for duplicate names
                existing_names = [p["name"] for p in portfolios.values()]
                if new_portfolio_name not in existing_names:
                    new_portfolio = create_portfolio(new_portfolio_name, new_webhook_url)
                    saved_portfolio = repo_save_portfolio(new_portfolio)
                    portfolios = load_portfolios()
                    st.session_state.selected_portfolio_id = saved_portfolio["id"]
                    st.success(f"✅ Created portfolio: {new_portfolio_name}")
                    st.rerun()
                else:
                    st.error("A portfolio with this name already exists")
            else:
                st.error("Please enter a portfolio name")

    st.markdown("---")

    # List existing portfolios
    if portfolios:
        st.subheader("📋 Your Portfolios")

        # Portfolio selector
        portfolio_options = {pid: p["name"] for pid, p in portfolios.items()}

        selected_id = st.selectbox(
            "Select Portfolio",
            options=list(portfolio_options.keys()),
            format_func=lambda x: portfolio_options[x],
            key="portfolio_selector"
        )

        if selected_id:
            st.session_state.selected_portfolio_id = selected_id
            current_portfolio = portfolios[selected_id]

            # Show portfolio info
            st.info(f"""
            **Created:** {current_portfolio['created_date'][:10]}
            **Stocks:** {len(current_portfolio['stocks'])}
            **Status:** {'✅ Enabled' if current_portfolio['enabled'] else '❌ Disabled'}
            """)

            # Edit portfolio settings
            with st.expander("⚙️ Portfolio Settings"):
                # Edit name
                edited_name = st.text_input(
                    "Portfolio Name",
                    value=current_portfolio["name"],
                    key=f"edit_name_{selected_id}"
                )

                # Edit webhook
                edited_webhook = st.text_input(
                    "Discord Webhook URL",
                    value=current_portfolio.get("discord_webhook", ""),
                    type="password",
                    key=f"edit_webhook_{selected_id}",
                    help="Leave blank to disable Discord alerts for this portfolio"
                )

                # Enable/disable
                edited_enabled = st.checkbox(
                    "Enable Portfolio Alerts",
                    value=current_portfolio.get("enabled", True),
                    key=f"edit_enabled_{selected_id}"
                )

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("💾 Save Changes", use_container_width=True):
                        updated_portfolio = dict(current_portfolio)
                        updated_portfolio["name"] = edited_name
                        updated_portfolio["discord_webhook"] = edited_webhook
                        updated_portfolio["enabled"] = edited_enabled
                        updated_portfolio["last_updated"] = datetime.now().isoformat()
                        repo_save_portfolio(updated_portfolio)
                        portfolios = load_portfolios()
                        st.session_state.selected_portfolio_id = updated_portfolio["id"]
                        st.success("Settings saved!")
                        st.rerun()

                with col2:
                    if st.button("🗑️ Delete Portfolio", type="secondary", use_container_width=True):
                        if len(current_portfolio["stocks"]) > 0:
                            st.error("Remove all stocks before deleting")
                        else:
                            repo_delete_portfolio(selected_id)
                            st.session_state.selected_portfolio_id = None
                            st.success("Portfolio deleted")
                            st.rerun()
    else:
        st.info("No portfolios yet. Create one above to get started!")

# Main content area
if st.session_state.selected_portfolio_id and st.session_state.selected_portfolio_id in portfolios:
    current_portfolio = portfolios[st.session_state.selected_portfolio_id]

    # Portfolio header
    st.header(f"📊 {current_portfolio['name']}")

    if current_portfolio.get("discord_webhook"):
        st.success("✅ Discord webhook configured for this portfolio")
    else:
        st.warning("⚠️ No Discord webhook configured - alerts won't be sent")

    # Two column layout
    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("➕ Add Stocks")

        # Load available symbols
        symbols_dict = load_available_symbols()

        if symbols_dict:
            # Exclude symbols already in this portfolio
            existing_symbols = [s["symbol"] for s in current_portfolio["stocks"]]
            available_options = [
                f"{symbol} - {name[:50]}"
                for symbol, name in sorted(symbols_dict.items())
                if symbol not in existing_symbols
            ]

            # Multi-select for adding multiple stocks at once
            selected_stocks = st.multiselect(
                "Select Stocks to Add",
                options=available_options,
                help="You can select multiple stocks at once",
                key=f"add_stocks_{st.session_state.selected_portfolio_id}"
            )

            # Add button
            if st.button("➕ Add to Portfolio", type="primary", use_container_width=True):
                if selected_stocks:
                    added_count = 0
                    for selection in selected_stocks:
                        symbol = selection.split(" - ")[0]
                        new_stock = {
                            "symbol": symbol,
                            "name": symbols_dict.get(symbol, symbol),
                            "added_date": datetime.now().isoformat()
                        }
                        current_portfolio["stocks"].append(new_stock)
                        added_count += 1

                    current_portfolio["last_updated"] = datetime.now().isoformat()
                    repo_save_portfolio(current_portfolio)
                    st.success(f"✅ Added {added_count} stock(s) to portfolio!")
                    st.rerun()
                else:
                    st.error("Please select at least one stock")

            # Quick add from text input
            st.markdown("---")
            st.subheader("⚡ Quick Add")
            quick_add = st.text_input(
                "Enter symbols (comma-separated)",
                placeholder="AAPL, MSFT, GOOGL",
                key=f"quick_add_{st.session_state.selected_portfolio_id}"
            )

            if st.button("Quick Add", use_container_width=True):
                if quick_add:
                    symbols_to_add = [s.strip().upper() for s in quick_add.split(",")]
                    added = []
                    skipped = []

                    for symbol in symbols_to_add:
                        if symbol in symbols_dict and symbol not in existing_symbols:
                            new_stock = {
                                "symbol": symbol,
                                "name": symbols_dict.get(symbol, symbol),
                                "added_date": datetime.now().isoformat()
                            }
                            current_portfolio["stocks"].append(new_stock)
                            added.append(symbol)
                        else:
                            skipped.append(symbol)

                    if added:
                        current_portfolio["last_updated"] = datetime.now().isoformat()
                        repo_save_portfolio(current_portfolio)
                        st.success(f"Added: {', '.join(added)}")
                    if skipped:
                        st.warning(f"Skipped (invalid or duplicate): {', '.join(skipped)}")

                    if added:
                        st.rerun()
        else:
            st.error("Could not load available symbols")

    with col2:
        st.subheader(f"📋 Portfolio Holdings ({len(current_portfolio['stocks'])} stocks)")

        if current_portfolio["stocks"]:
            # Display as dataframe
            df_data = []
            for stock in current_portfolio["stocks"]:
                df_data.append({
                    "Symbol": stock["symbol"],
                    "Company": stock.get("name", stock["symbol"])[:50],
                    "Date Added": stock.get("added_date", "")[:10]
                })

            df = pd.DataFrame(df_data)

            # Show the dataframe
            st.dataframe(df, use_container_width=True, height=400)

            # Remove stocks section
            st.markdown("---")
            st.subheader("🗑️ Remove Stocks")

            stocks_to_remove = st.multiselect(
                "Select stocks to remove",
                options=[s["symbol"] for s in current_portfolio["stocks"]],
                key=f"remove_stocks_{st.session_state.selected_portfolio_id}"
            )

            if st.button("🗑️ Remove Selected", type="secondary"):
                if stocks_to_remove:
                    current_portfolio["stocks"] = [
                        s for s in current_portfolio["stocks"]
                        if s["symbol"] not in stocks_to_remove
                    ]
                    current_portfolio["last_updated"] = datetime.now().isoformat()
                    repo_save_portfolio(current_portfolio)
                    st.success(f"Removed {len(stocks_to_remove)} stock(s)")
                    st.rerun()
                else:
                    st.error("Please select stocks to remove")

            # Export/Import options
            st.markdown("---")
            col_exp1, col_exp2 = st.columns(2)

            with col_exp1:
                # Export portfolio
                portfolio_export = {
                    "name": current_portfolio["name"],
                    "stocks": [s["symbol"] for s in current_portfolio["stocks"]],
                    "exported_date": datetime.now().isoformat()
                }

                st.download_button(
                    label="📥 Export Portfolio",
                    data=json.dumps(portfolio_export, indent=2),
                    file_name=f"portfolio_{current_portfolio['name'].replace(' ', '_')}.json",
                    mime="application/json",
                    use_container_width=True
                )

            with col_exp2:
                # Clear all stocks
                if st.button("🗑️ Clear All Stocks", use_container_width=True):
                    if st.session_state.get(f"confirm_clear_{st.session_state.selected_portfolio_id}"):
                        current_portfolio["stocks"] = []
                        current_portfolio["last_updated"] = datetime.now().isoformat()
                        repo_save_portfolio(current_portfolio)
                        st.session_state[f"confirm_clear_{st.session_state.selected_portfolio_id}"] = False
                        st.success("All stocks removed")
                        st.rerun()
                    else:
                        st.session_state[f"confirm_clear_{st.session_state.selected_portfolio_id}"] = True
                        st.warning("Click again to confirm")
        else:
            st.info("📭 This portfolio is empty. Add stocks on the left to get started!")

else:
    # No portfolio selected
    st.info("""
    👈 **Get Started:**
    1. Create a new portfolio in the sidebar
    2. Add a Discord webhook URL for alerts
    3. Add stocks to track
    4. Alerts will be duplicated to your portfolio's Discord channel
    """)

# Footer with instructions
st.markdown("---")
st.info("""
💡 **How Portfolio Alerts Work:**
- Each portfolio can have its own Discord webhook URL (different channel)
- When an alert triggers for a stock in a portfolio, it's sent to that portfolio's Discord channel
- You can have the same stock in multiple portfolios with different webhooks
- Portfolio alerts have a [PORTFOLIO: name] prefix to identify which portfolio triggered
- The scheduler automatically checks all portfolios when processing alerts
""")

# Show summary of all portfolios
if portfolios:
    st.markdown("---")
    st.subheader("📊 All Portfolios Summary")

    summary_data = []
    for pid, portfolio in portfolios.items():
        summary_data.append({
            "Name": portfolio["name"],
            "Stocks": len(portfolio["stocks"]),
            "Discord": "✅" if portfolio.get("discord_webhook") else "❌",
            "Status": "Enabled" if portfolio.get("enabled", True) else "Disabled",
            "Last Updated": portfolio.get("last_updated", "")[:10]
        })

    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

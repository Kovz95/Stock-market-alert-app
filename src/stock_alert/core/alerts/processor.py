"""Alert processing, saving, updating, and notification utilities."""

import datetime
import os
from typing import Any

import pandas as pd
import requests

# Try to import streamlit for caching, but don't fail if not available
try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from src.stock_alert.core.alerts.validator import (
    _conditions_to_storage_format,
    _normalize_conditions_for_comparison,
    validate_conditions,
)
from src.stock_alert.data_access.alert_repository import (
    create_alert as repo_create_alert,
)
from src.stock_alert.data_access.alert_repository import (
    get_alert as repo_get_alert,
)
from src.stock_alert.data_access.alert_repository import (
    list_alerts as repo_list_alerts,
)
from src.stock_alert.data_access.alert_repository import (
    update_alert as repo_update_alert,
)
from src.stock_alert.data_access.document_store import load_document


def _derive_country(country: str | None, exchange: str) -> str:
    """
    Derive country from exchange if not provided.

    Args:
        country: Country name (may be None)
        exchange: Exchange name

    Returns:
        Country name or exchange name
    """
    if country:
        return country
    try:
        from exchange_country_mapping import get_country_for_exchange

        return get_country_for_exchange(exchange) or exchange
    except ImportError:
        return exchange


def _is_futures_symbol(ticker: str) -> bool:
    """
    Check if ticker is a futures symbol.

    Args:
        ticker: Ticker symbol

    Returns:
        True if futures symbol, False otherwise
    """
    futures_db = (
        load_document(
            "futures_database",
            default={},
            fallback_path="futures_database.json",
        )
        or {}
    )
    return ticker in futures_db or ticker.upper() in futures_db


def save_alert(
    name: str,
    entry_conditions_list: list[dict[str, Any]],
    combination_logic: str,
    ticker: str,
    stock_name: str,
    exchange: str,
    timeframe: str,
    last_triggered: str | None,
    action: str,
    ratio: str,
    dtp_params: dict | None = None,
    multi_timeframe_params: dict | None = None,
    mixed_timeframe_params: dict | None = None,
    country: str | None = None,
    adjustment_method: str | None = None,
) -> dict[str, Any]:
    """
    Save a new alert.

    Args:
        name: Alert name
        entry_conditions_list: List of alert conditions
        combination_logic: AND/OR logic
        ticker: Stock ticker
        stock_name: Stock name
        exchange: Exchange name
        timeframe: Timeframe
        last_triggered: Last triggered timestamp
        action: Buy/Sell action
        ratio: Ratio indicator
        dtp_params: Dynamic time period parameters
        multi_timeframe_params: Multi-timeframe parameters
        mixed_timeframe_params: Mixed timeframe parameters
        country: Country name
        adjustment_method: Price adjustment method

    Returns:
        Created alert dict

    Raises:
        ValueError: If validation fails or alert already exists
    """
    if not entry_conditions_list or not ticker or not stock_name:
        raise ValueError("Entry conditions cannot be empty.")

    if isinstance(entry_conditions_list, dict):
        has_conditions = any(
            isinstance(value, dict) and value.get("conditions")
            for value in entry_conditions_list.values()
        )
        if not has_conditions:
            raise ValueError("Entry conditions cannot be empty.")

    if validate_conditions(entry_conditions_list) is False:
        raise ValueError("Invalid conditions provided.")

    conditions_for_compare = _normalize_conditions_for_comparison(entry_conditions_list)
    conditions_for_save = _conditions_to_storage_format(entry_conditions_list)

    existing_alerts = repo_list_alerts()
    for alert in existing_alerts:
        if (
            alert.get("stock_name") == stock_name
            and alert.get("ticker") == ticker
            and alert.get("conditions") == conditions_for_compare
            and alert.get("combination_logic") == combination_logic
            and (alert.get("exchange") or alert.get("country")) == exchange
            and alert.get("timeframe") == timeframe
            and alert.get("name") == name
        ):
            raise ValueError("Alert already exists with the same name and data fields.")

    country_value = _derive_country(country, exchange)
    is_futures = _is_futures_symbol(ticker)

    payload = {
        "name": name,
        "stock_name": stock_name,
        "ticker": ticker,
        "conditions": conditions_for_save,
        "combination_logic": combination_logic,
        "last_triggered": last_triggered,
        "action": action,
        "timeframe": timeframe,
        "exchange": exchange,
        "country": country_value,
        "ratio": ratio,
        "is_ratio": False,
    }

    if dtp_params:
        payload["dtp_params"] = dtp_params
    if multi_timeframe_params:
        payload["multi_timeframe_params"] = multi_timeframe_params
    if mixed_timeframe_params:
        payload["mixed_timeframe_params"] = mixed_timeframe_params
    if adjustment_method or is_futures:
        payload["adjustment_method"] = adjustment_method

    return repo_create_alert(payload)


def save_ratio_alert(
    name: str,
    entry_conditions_list: list[dict[str, Any]],
    combination_logic: str,
    ticker1: str,
    ticker2: str,
    stock_name: str,
    exchange: str,
    timeframe: str,
    last_triggered: str | None,
    action: str,
    ratio: str,
    country: str | None = None,
    adjustment_method: str | None = None,
) -> dict[str, Any]:
    """
    Save a new ratio alert.

    Args:
        name: Alert name
        entry_conditions_list: List of alert conditions
        combination_logic: AND/OR logic
        ticker1: First ticker
        ticker2: Second ticker
        stock_name: Stock name
        exchange: Exchange name
        timeframe: Timeframe
        last_triggered: Last triggered timestamp
        action: Buy/Sell action
        ratio: Ratio indicator
        country: Country name
        adjustment_method: Price adjustment method

    Returns:
        Created alert dict

    Raises:
        ValueError: If validation fails or alert already exists
    """
    if not entry_conditions_list or not ticker1 or not ticker2 or not stock_name:
        raise ValueError("Entry conditions cannot be empty.")

    if validate_conditions(entry_conditions_list) is False:
        raise ValueError("Invalid conditions provided.")

    conditions_for_compare = _normalize_conditions_for_comparison(entry_conditions_list)
    conditions_for_save = _conditions_to_storage_format(entry_conditions_list)

    existing_alerts = repo_list_alerts()
    for alert in existing_alerts:
        if alert.get("ratio", "No") == "Yes":
            if (
                alert.get("stock_name") == stock_name
                and alert.get("ticker1") == ticker1
                and alert.get("ticker2") == ticker2
                and alert.get("conditions") == conditions_for_compare
                and alert.get("combination_logic") == combination_logic
                and alert.get("exchange") == exchange
                and alert.get("timeframe") == timeframe
                and alert.get("name") == name
            ):
                raise ValueError("Alert already exists with the same name and data fields.")

    country_value = _derive_country(country, exchange)

    payload = {
        "name": name,
        "stock_name": stock_name,
        "ticker": f"{ticker1}_{ticker2}",
        "ticker1": ticker1,
        "ticker2": ticker2,
        "conditions": conditions_for_save,
        "combination_logic": combination_logic,
        "last_triggered": last_triggered,
        "action": action,
        "timeframe": timeframe,
        "exchange": exchange,
        "country": country_value,
        "ratio": ratio or "Yes",
        "is_ratio": True,
    }

    if adjustment_method:
        payload["adjustment_method"] = adjustment_method

    return repo_create_alert(payload)


def update_alert(
    alert_id: str,
    name: str,
    entry_conditions_list: list[dict[str, Any]],
    combination_logic: str,
    ticker: str,
    stock_name: str,
    exchange: str,
    timeframe: str,
    last_triggered: str | None,
    action: str,
    ratio: str,
) -> dict[str, Any]:
    """
    Update an existing alert.

    Args:
        alert_id: Alert ID
        name: Alert name
        entry_conditions_list: List of alert conditions
        combination_logic: AND/OR logic
        ticker: Stock ticker
        stock_name: Stock name
        exchange: Exchange name
        timeframe: Timeframe
        last_triggered: Last triggered timestamp
        action: Buy/Sell action
        ratio: Ratio indicator

    Returns:
        Updated alert dict

    Raises:
        ValueError: If alert not found or validation fails
    """
    existing = repo_get_alert(alert_id)
    if not existing:
        raise ValueError(f"Alert with ID {alert_id} not found.")

    if not entry_conditions_list or not ticker or not stock_name:
        raise ValueError("Entry conditions cannot be empty.")

    if validate_conditions(entry_conditions_list) is False:
        raise ValueError("Invalid conditions provided.")

    payload = {
        "name": name,
        "stock_name": stock_name,
        "ticker": ticker,
        "conditions": _conditions_to_storage_format(entry_conditions_list),
        "combination_logic": combination_logic,
        "last_triggered": last_triggered,
        "action": action,
        "timeframe": timeframe,
        "exchange": exchange,
        "country": existing.get("country"),
        "ratio": ratio,
        "is_ratio": existing.get("is_ratio") or str(ratio).lower() in {"yes", "true", "1"},
    }

    repo_update_alert(alert_id, payload)
    return repo_get_alert(alert_id)


def update_ratio_alert(
    alert_id: str,
    name: str,
    entry_conditions_list: list[dict[str, Any]],
    combination_logic: str,
    ticker1: str,
    ticker2: str,
    stock_name: str,
    exchange: str,
    timeframe: str,
    last_triggered: str | None,
    action: str,
    ratio: str,
) -> dict[str, Any]:
    """
    Update an existing ratio alert.

    Args:
        alert_id: Alert ID
        name: Alert name
        entry_conditions_list: List of alert conditions
        combination_logic: AND/OR logic
        ticker1: First ticker
        ticker2: Second ticker
        stock_name: Stock name
        exchange: Exchange name
        timeframe: Timeframe
        last_triggered: Last triggered timestamp
        action: Buy/Sell action
        ratio: Ratio indicator

    Returns:
        Updated alert dict

    Raises:
        ValueError: If alert not found or validation fails
    """
    existing = repo_get_alert(alert_id)
    if not existing:
        raise ValueError(f"Alert with ID {alert_id} not found.")

    if not entry_conditions_list or not ticker1 or not ticker2 or not stock_name:
        raise ValueError("Entry conditions cannot be empty.")

    if validate_conditions(entry_conditions_list) is False:
        raise ValueError("Invalid conditions provided.")

    payload = {
        "name": name,
        "stock_name": stock_name,
        "ticker": f"{ticker1}_{ticker2}",
        "ticker1": ticker1,
        "ticker2": ticker2,
        "conditions": _conditions_to_storage_format(entry_conditions_list),
        "combination_logic": combination_logic,
        "last_triggered": last_triggered,
        "action": action,
        "timeframe": timeframe,
        "exchange": exchange,
        "country": existing.get("country"),
        "ratio": ratio or "Yes",
        "is_ratio": True,
    }

    repo_update_alert(alert_id, payload)
    return repo_get_alert(alert_id)


def get_alert_by_id(alert_id: str) -> dict[str, Any] | None:
    """
    Get a specific alert by its ID.

    Args:
        alert_id: Alert ID

    Returns:
        Alert dict or None if not found
    """
    return repo_get_alert(alert_id)


def load_alert_data() -> list[dict[str, Any]]:
    """
    Load alert data from repository.
    FOR update_stocks.py ONLY.

    Returns:
        List of all alerts
    """
    return repo_list_alerts()


def get_all_stocks(alert_data: list[dict[str, Any]], timeframe: str) -> list[str]:
    """
    Get all unique stock tickers from alert data.

    Args:
        alert_data: List of alerts
        timeframe: Timeframe to filter by

    Returns:
        List of unique tickers
    """
    return list(set([alert["ticker"] for alert in alert_data if alert["timeframe"] == timeframe]))


def get_stock_exchange(alert_data: list[dict[str, Any]], stock: str) -> str:
    """
    Get exchange of a stock.

    Args:
        alert_data: List of alerts
        stock: Stock ticker

    Returns:
        Exchange name
    """
    return [alert["exchange"] for alert in alert_data if alert["ticker"] == stock][0]


def get_all_alerts_for_stock(alert_data: list[dict[str, Any]], stock: str) -> list[dict[str, Any]]:
    """
    Get all alerts related to a specific stock.

    Args:
        alert_data: List of alerts
        stock: Stock ticker

    Returns:
        List of alerts for the stock
    """
    return [alert for alert in alert_data if alert["ticker"] == stock]


def check_database(stock: str, timeframe: str) -> pd.DataFrame:
    """
    Load or create the historical database for a stock.

    Args:
        stock: Stock ticker
        timeframe: Timeframe (daily/weekly)

    Returns:
        DataFrame with historical data
    """
    from src.stock_alert.config.settings import get_settings
    from src.stock_alert.core.market_data.data_fetcher import get_latest_stock_data

    file_path = f"data/{stock}_{timeframe}.csv"

    if not os.path.exists(file_path):
        print(f"[FETCH] No existing data for {stock}, fetching new data...")
        exchange = get_stock_exchange(load_alert_data(), stock)
        timeframe_mapped = "day" if timeframe == "daily" else "week"
        settings = get_settings()
        df = get_latest_stock_data(stock, exchange, timeframe_mapped, settings.fmp_api_key)
        df.reset_index(inplace=True)  # Move Date index to a column
        df.insert(0, "index", range(1, len(df) + 1))
        df.to_csv(file_path, index=False, date_format="%Y-%m-%d")
        return df

    else:
        df = pd.read_csv(file_path)
        # Drop any extra unnamed columns that may have been added in previous runs
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        # Ensure that if an "index" column exists, it is of integer type; if not, create one.
        if "index" in df.columns:
            df["index"] = df["index"].astype(int)
        else:
            df.insert(0, "index", range(1, len(df) + 1))
        return df


def update_stock_database(stock: str, new_stock_data: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Update stock database with new data.

    Args:
        stock: Stock ticker
        new_stock_data: New data to add
        timeframe: Timeframe (daily/weekly)

    Returns:
        Updated DataFrame
    """
    file_path = f"data/{stock}_{timeframe}.csv"

    # Load existing data
    existing_data = check_database(stock, timeframe)

    # Ensure new_stock_data has the same structure
    new_stock_data.reset_index(inplace=True)  # Convert Date index to column
    new_stock_data = new_stock_data[~new_stock_data["Date"].isin(existing_data["Date"])]

    df_combined = pd.concat([existing_data, new_stock_data])
    df_combined.reset_index(drop=True, inplace=True)

    # Regenerate the "index" column to be consistent
    df_combined["index"] = range(1, len(df_combined) + 1)
    cols = df_combined.columns.tolist()
    if "index" in cols:
        cols.insert(0, cols.pop(cols.index("index")))
    df_combined = df_combined[cols]

    # Save the combined data consistently without using pandas' default index
    df_combined.to_csv(file_path, index=False, date_format="%Y-%m-%d")

    return df_combined


def send_alert(stock: str, alert: dict[str, Any], condition_str: str, df: pd.DataFrame):
    """
    Send an alert via Discord.

    Args:
        stock: Stock ticker
        alert: Alert dict
        condition_str: Triggered condition string
        df: Stock dataframe with price data
    """
    from src.stock_alert.config.settings import get_settings
    from src.stock_alert.services.discord.client import log_to_discord

    # Ensure the condition_str is actually a string
    if not isinstance(condition_str, str):
        print(f"[Alert Check] Provided condition is not a string: {condition_str}")
        return

    current_price = df.iloc[-1]["Close"]
    # Add action to the alert
    action = alert["action"]
    timeframe = alert["timeframe"]

    # Get proper exchange name from country
    from exchange_name_mapping import get_exchange_name

    country = alert.get("exchange", "Unknown")
    exchange = get_exchange_name(country)

    settings = get_settings()

    # Send the alert via Discord
    send_stock_alert(
        settings.webhook_url,
        timeframe,
        alert["name"],
        stock,
        condition_str,
        current_price,
        action,
        exchange,
    )

    # Only send to second webhook if it's different from the first
    if settings.webhook_url_2 and settings.webhook_url_2 != settings.webhook_url:
        send_stock_alert(
            settings.webhook_url_2,
            timeframe,
            alert["name"],
            stock,
            condition_str,
            current_price,
            action,
            exchange,
        )

    # Also send to ALL portfolio channels that contain this stock
    try:
        from portfolio_discord import portfolio_manager

        portfolios_with_stock = portfolio_manager.get_portfolios_for_stock(stock)

        for portfolio_id, portfolio in portfolios_with_stock:
            webhook_url = portfolio.get("discord_webhook", "")
            if webhook_url and portfolio.get("enabled", True):
                # Send the same alert to this portfolio channel
                portfolio_name = portfolio.get("name", "Portfolio")
                send_stock_alert(
                    webhook_url,
                    timeframe,
                    f"[{portfolio_name}] {alert['name']}",
                    stock,
                    condition_str,
                    current_price,
                    action,
                    exchange,
                )
                log_to_discord(f"  → Also sent to portfolio: {portfolio_name}")
    except Exception:
        pass  # Silently skip if portfolio system not available

    # Send to custom Discord channels based on condition matching
    try:
        custom_channels = (
            load_document(
                "custom_discord_channels",
                default={},
                fallback_path="custom_discord_channels.json",
            )
            or {}
        )

        # Check each custom channel
        for channel_name, channel_config in custom_channels.items():
            if channel_config.get("enabled", True):
                # Check if the triggered condition matches this channel's condition
                # Need to normalize the condition strings for comparison
                triggered_condition_normalized = condition_str.replace(" ", "")
                channel_condition_normalized = channel_config.get("condition", "").replace(" ", "")

                # Check if the triggered condition contains the channel's condition
                if channel_condition_normalized in triggered_condition_normalized:
                    webhook_url = channel_config.get("webhook_url")
                    if webhook_url:
                        # Send alert to this custom channel
                        send_stock_alert(
                            webhook_url,
                            timeframe,
                            f"[{channel_name}] {alert['name']}",
                            stock,
                            condition_str,
                            current_price,
                            action,
                            exchange,
                        )
                        log_to_discord(
                            f"  → Also sent to custom channel: {channel_config.get('channel_name', channel_name)}"
                        )
    except Exception:
        pass  # Silently skip if custom channels not configured or error occurs

    log_to_discord(
        f"[Alert Triggered] '{alert['name']}' for {stock}: condition '{condition_str}' at {datetime.datetime.now()}."
    )


def send_stock_alert(
    webhook_url: str,
    timeframe: str,
    alert_name: str,
    ticker: str,
    triggered_condition: str,
    current_price: float,
    action: str,
    exchange: str = "Unknown",
):
    """
    Send a stock alert to Discord via webhook.

    Args:
        webhook_url: Discord webhook URL
        timeframe: Timeframe
        alert_name: Alert name
        ticker: Stock ticker
        triggered_condition: Triggered condition string
        current_price: Current stock price
        action: Buy/Sell action
        exchange: Exchange name
    """
    # Check if webhook URL is valid
    if not webhook_url:
        print(f"[WARNING] No webhook URL configured for alert: {alert_name}")
        return

    # Change the color based on the action
    color = 0x00FF00 if action == "Buy" else 0xFF0000

    # Format timeframe for display
    timeframe_display = {
        "1d": "1D (Daily)",
        "1wk": "1W (Weekly)",
        "1w": "1W (Weekly)",
        "1D": "1D (Daily)",
        "1W": "1W (Weekly)",
        "daily": "1D (Daily)",
        "weekly": "1W (Weekly)",
    }.get(timeframe.lower() if isinstance(timeframe, str) else timeframe, timeframe)

    embed = {
        "title": f"[ALERT] {alert_name} ({ticker})",
        "description": f"The condition **{triggered_condition}** was triggered. \n Action: {action}",
        "fields": [
            {"name": "Timeframe", "value": timeframe_display, "inline": True},
            {"name": "Exchange", "value": exchange, "inline": True},
            {"name": "Current Price", "value": f"${current_price:.2f}", "inline": True},
        ],
        "color": color,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }

    payload = {"embeds": [embed]}

    try:
        response = requests.post(webhook_url, json=payload)
        if response.status_code == 204:
            print("Alert sent successfully!")
        else:
            print(f"Failed to send alert. HTTP Status Code: {response.status_code}")
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"An error occurred: {e}")

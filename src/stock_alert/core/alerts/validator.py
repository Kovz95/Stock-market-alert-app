"""Alert validation and duplicate checking utilities."""

from typing import Any

from src.stock_alert.data_access.alert_repository import list_alerts as repo_list_alerts


def validate_conditions(entry_conditions_list):
    """
    Validate alert conditions for proper format and syntax.

    Args:
        entry_conditions_list: List or dict of alert conditions

    Returns:
        bool: True if valid, False otherwise
    """
    print("Validating conditions...")

    # Handle dict format (from Add_Alert page)
    if isinstance(entry_conditions_list, dict):
        for key, value in entry_conditions_list.items():
            if isinstance(value, dict):
                conditions = value.get("conditions", "")
                # If conditions is a list of strings
                if isinstance(conditions, list):
                    for condition in conditions:
                        if not condition or (
                            isinstance(condition, str) and condition.strip() == ""
                        ):
                            print("Empty condition found.")
                            return False
                        if isinstance(condition, str) and condition.count("[") != condition.count(
                            "]"
                        ):
                            print("Unclosed brackets found.")
                            return False
                # If conditions is a single string or other format
                elif conditions:
                    if isinstance(conditions, str):
                        if conditions.strip() == "":
                            print("Empty condition found.")
                            return False
                        if conditions.count("[") != conditions.count("]"):
                            print("Unclosed brackets found.")
                            return False
    # Handle list format
    elif isinstance(entry_conditions_list, list):
        for entry in entry_conditions_list:
            # If it's a string (from new UI)
            if isinstance(entry, str):
                if not entry or entry.strip() == "":
                    print("Empty condition found.")
                    return False
                if entry.count("[") != entry.count("]"):
                    print("Unclosed brackets found.")
                    return False
            # If it's a dict (legacy format)
            elif isinstance(entry, dict):
                condition = entry.get("conditions", "")
                if not condition or (isinstance(condition, str) and condition.strip() == ""):
                    print("Empty condition found.")
                    return False
                if isinstance(condition, str) and condition.count("[") != condition.count("]"):
                    print("Unclosed brackets found.")
                    return False
    else:
        print(f"Unknown format for entry_conditions_list: {type(entry_conditions_list)}")
        return False

    return True


def _normalize_conditions_for_comparison(entry_conditions_list):
    """
    Normalize conditions for comparison.

    Args:
        entry_conditions_list: List or dict of conditions

    Returns:
        Normalized list of conditions
    """
    if isinstance(entry_conditions_list, dict):
        conditions_for_compare = []
        for key, value in entry_conditions_list.items():
            if isinstance(value, dict) and "conditions" in value:
                for idx, condition_str in enumerate(value["conditions"], 1):
                    conditions_for_compare.append({"index": idx, "conditions": condition_str})
        return conditions_for_compare
    return entry_conditions_list


def _conditions_to_storage_format(entry_conditions_list):
    """
    Convert conditions to storage format.

    Args:
        entry_conditions_list: List or dict of conditions

    Returns:
        List of conditions in storage format
    """
    if isinstance(entry_conditions_list, dict):
        conditions_for_save = []
        for key, value in entry_conditions_list.items():
            if isinstance(value, dict) and "conditions" in value:
                for idx, condition_str in enumerate(value["conditions"], 1):
                    conditions_for_save.append({"index": idx, "conditions": condition_str})
        return conditions_for_save
    return entry_conditions_list


def check_similar_alerts(
    stock_name: str,
    ticker: str,
    entry_conditions_list: list[dict[str, Any]],
    combination_logic: str,
    exchange: str,
    timeframe: str,
) -> list[dict[str, Any]]:
    """
    Check if an alert with similar conditions already exists for the same stock.

    Args:
        stock_name: Stock name
        ticker: Stock ticker
        entry_conditions_list: Alert conditions
        combination_logic: AND/OR logic
        exchange: Exchange name
        timeframe: Timeframe

    Returns:
        List of similar alerts that could be updated instead of creating duplicates
    """
    alerts = repo_list_alerts()

    similar_alerts = []
    for alert in alerts:
        # Check if this is the same stock and ticker
        if (
            alert["stock_name"] == stock_name
            and alert["ticker"] == ticker
            and alert["exchange"] == exchange
            and alert["timeframe"] == timeframe
        ):
            # Check if conditions are similar (same structure but potentially different values)
            if (
                alert["conditions"] == entry_conditions_list
                and alert["combination_logic"] == combination_logic
            ):
                similar_alerts.append(alert)

    return similar_alerts


def check_similar_ratio_alerts(
    stock_name: str,
    ticker1: str,
    ticker2: str,
    entry_conditions_list: list[dict[str, Any]],
    combination_logic: str,
    exchange: str,
    timeframe: str,
) -> list[dict[str, Any]]:
    """
    Check if a ratio alert with similar conditions already exists.

    Args:
        stock_name: Stock name
        ticker1: First ticker
        ticker2: Second ticker
        entry_conditions_list: Alert conditions
        combination_logic: AND/OR logic
        exchange: Exchange name
        timeframe: Timeframe

    Returns:
        List of similar alerts that could be updated instead of creating duplicates
    """
    alerts = repo_list_alerts()

    similar_alerts = []
    for alert in alerts:
        if (alert.get("ratio") == "Yes") or alert.get("is_ratio"):
            # Check if this is the same stock and tickers
            if (
                alert["stock_name"] == stock_name
                and alert.get("ticker1") == ticker1
                and alert.get("ticker2") == ticker2
                and alert["exchange"] == exchange
                and alert["timeframe"] == timeframe
            ):
                # Check if conditions are similar
                if (
                    alert["conditions"] == entry_conditions_list
                    and alert["combination_logic"] == combination_logic
                ):
                    similar_alerts.append(alert)

    return similar_alerts


def suggest_alert_update(
    stock_name: str,
    ticker: str,
    entry_conditions_list: list[dict[str, Any]],
    combination_logic: str,
    exchange: str,
    timeframe: str,
) -> list[dict[str, Any]] | None:
    """
    Suggest updating an existing alert instead of creating a duplicate.

    Args:
        stock_name: Stock name
        ticker: Stock ticker
        entry_conditions_list: Alert conditions
        combination_logic: AND/OR logic
        exchange: Exchange name
        timeframe: Timeframe

    Returns:
        List of suggestions or None if no similar alerts found
    """
    similar_alerts = check_similar_alerts(
        stock_name, ticker, entry_conditions_list, combination_logic, exchange, timeframe
    )

    if not similar_alerts:
        return None

    suggestions = []
    for alert in similar_alerts:
        suggestion = {
            "alert_id": alert["alert_id"],
            "name": alert["name"],
            "action": alert.get("action", "Unknown"),
            "last_triggered": alert.get("last_triggered", "Never"),
            "message": f"Alert '{alert['name']}' already exists with the same conditions. Consider updating it instead of creating a duplicate.",
        }
        suggestions.append(suggestion)

    return suggestions


def suggest_ratio_alert_update(
    stock_name: str,
    ticker1: str,
    ticker2: str,
    entry_conditions_list: list[dict[str, Any]],
    combination_logic: str,
    exchange: str,
    timeframe: str,
) -> list[dict[str, Any]] | None:
    """
    Suggest updating an existing ratio alert instead of creating a duplicate.

    Args:
        stock_name: Stock name
        ticker1: First ticker
        ticker2: Second ticker
        entry_conditions_list: Alert conditions
        combination_logic: AND/OR logic
        exchange: Exchange name
        timeframe: Timeframe

    Returns:
        List of suggestions or None if no similar alerts found
    """
    similar_alerts = check_similar_ratio_alerts(
        stock_name, ticker1, ticker2, entry_conditions_list, combination_logic, exchange, timeframe
    )

    if not similar_alerts:
        return None

    suggestions = []
    for alert in similar_alerts:
        suggestion = {
            "alert_id": alert["alert_id"],
            "name": alert["name"],
            "action": alert.get("action", "Unknown"),
            "last_triggered": alert.get("last_triggered", "Never"),
            "message": f"Ratio alert '{alert['name']}' already exists with the same conditions. Consider updating it instead of creating a duplicate.",
        }
        suggestions.append(suggestion)

    return suggestions


def get_stock_alerts_summary(stock_name: str, ticker: str) -> list[dict[str, Any]]:
    """
    Get a summary of all existing alerts for a specific stock.

    Args:
        stock_name: Stock name
        ticker: Stock ticker

    Returns:
        List of alert summaries
    """
    stock_alerts = []
    for alert in repo_list_alerts():
        if alert["stock_name"] == stock_name and alert["ticker"] == ticker:
            summary = {
                "alert_id": alert["alert_id"],
                "name": alert["name"],
                "conditions": alert["conditions"],
                "combination_logic": alert["combination_logic"],
                "timeframe": alert["timeframe"],
                "exchange": alert["exchange"],
                "action": alert.get("action", "Unknown"),
                "last_triggered": alert.get("last_triggered", "Never"),
                "ratio": alert.get("ratio", "No"),
            }
            stock_alerts.append(summary)

    return stock_alerts

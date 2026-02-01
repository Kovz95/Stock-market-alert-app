"""
Industry-specific logging module for stock alert checks.

This module provides logging functions for tracking data quality checks,
industry-specific validations, and other alert-related diagnostic information.
"""

import logging
from datetime import datetime
from typing import Optional

# Set up logging
logger = logging.getLogger(__name__)


def log_alert_check(ticker: str, check_type: str, message: str, level: str = "info") -> None:
    """
    Log an alert check event with ticker, check type, and custom message.

    This function is used to log data quality checks, validation warnings,
    and other diagnostic information during alert processing.

    Args:
        ticker: The stock ticker being checked (e.g., "AAPL", "TSLA")
        check_type: Type of check being performed (e.g., "Data Check", "Validation")
        message: Detailed message about the check result
        level: Log level - "info", "warning", "error", or "debug" (default: "info")

    Examples:
        >>> log_alert_check("AAPL", "Data Check", "Data is 2 days old")
        >>> log_alert_check("TSLA", "Validation", "Missing volume data", level="warning")
    """
    # Format the log message
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_message = f"[{timestamp}] [{ticker}] [{check_type}] {message}"

    # Log to standard Python logger based on level
    log_level = level.lower()
    if log_level == "error":
        logger.error(formatted_message)
    elif log_level == "warning":
        logger.warning(formatted_message)
    elif log_level == "debug":
        logger.debug(formatted_message)
    else:
        logger.info(formatted_message)

    # Optionally log to Discord for important checks
    # Import here to avoid circular dependencies
    try:
        if log_level in ["warning", "error"]:
            from src.services.discord_logger import log_to_discord_async
            discord_message = f"**{check_type}** - `{ticker}`\n{message}"
            log_to_discord_async(discord_message)
    except Exception as e:
        # Silently fail if Discord logging is unavailable
        logger.debug(f"Discord logging unavailable: {e}")


def log_data_quality_issue(
    ticker: str,
    issue_type: str,
    details: str,
    severity: str = "warning"
) -> None:
    """
    Log a data quality issue for a specific ticker.

    Args:
        ticker: The stock ticker
        issue_type: Type of data quality issue (e.g., "Stale Data", "Missing Data")
        details: Detailed description of the issue
        severity: Severity level - "info", "warning", or "error" (default: "warning")

    Examples:
        >>> log_data_quality_issue("AAPL", "Stale Data", "Last update was 3 days ago")
        >>> log_data_quality_issue("TSLA", "Missing Data", "No volume data available", "error")
    """
    log_alert_check(ticker, f"Data Quality - {issue_type}", details, level=severity)


def log_industry_validation(
    ticker: str,
    industry: str,
    validation_result: str,
    passed: bool = True
) -> None:
    """
    Log industry-specific validation results.

    Args:
        ticker: The stock ticker
        industry: Industry sector (e.g., "Technology", "Finance")
        validation_result: Description of validation result
        passed: Whether validation passed (default: True)

    Examples:
        >>> log_industry_validation("AAPL", "Technology", "Sector metadata validated")
        >>> log_industry_validation("XYZ", "Unknown", "Industry not found", passed=False)
    """
    level = "info" if passed else "warning"
    status = "✓" if passed else "✗"
    message = f"{status} [{industry}] {validation_result}"
    log_alert_check(ticker, "Industry Validation", message, level=level)

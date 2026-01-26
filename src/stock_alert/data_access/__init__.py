"""
Data access layer for Stock Market Alert App.

This module provides repository classes and database connection management
for accessing PostgreSQL database.

For better test isolation and to avoid circular imports, specific functions
should be imported directly from their respective modules:

    from src.stock_alert.data_access.database import db_config
    from src.stock_alert.data_access.alert_repository import list_alerts
    from src.stock_alert.data_access.portfolio_repository import list_portfolios
    from src.stock_alert.data_access.metadata_repository import fetch_stock_metadata_df
"""

# Lazy imports to avoid bootstrap issues during testing
def __getattr__(name):
    """Lazy import for backward compatibility"""
    if name == "DatabaseConfig" or name == "db_config":
        from .database import DatabaseConfig, db_config
        return db_config if name == "db_config" else DatabaseConfig

    if name in ("list_alerts", "get_alert", "create_alert", "update_alert", "delete_alert", "refresh_alert_cache"):
        from . import alert_repository
        return getattr(alert_repository, name)

    if name in ("list_portfolios", "get_portfolio", "create_portfolio", "update_portfolio", "delete_portfolio"):
        from . import portfolio_repository
        return getattr(portfolio_repository, name)

    if name in ("fetch_stock_metadata_df", "fetch_stock_metadata_map", "fetch_portfolios", "fetch_alerts", "clear_all_caches"):
        from . import metadata_repository
        return getattr(metadata_repository, name)

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    # Database
    "DatabaseConfig",
    "db_config",
    # Alert repository
    "list_alerts",
    "get_alert",
    "create_alert",
    "update_alert",
    "delete_alert",
    "refresh_alert_cache",
    # Portfolio repository
    "list_portfolios",
    "get_portfolio",
    "create_portfolio",
    "update_portfolio",
    "delete_portfolio",
    # Metadata repository
    "fetch_stock_metadata_df",
    "fetch_stock_metadata_map",
    "fetch_portfolios",
    "fetch_alerts",
    "clear_all_caches",
]

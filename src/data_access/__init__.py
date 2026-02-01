"""Data access layer for database operations."""

from src.data_access.alert_repository import list_alerts, refresh_alert_cache
from src.data_access.daily_move_stats_repository import (
    ensure_table as ensure_daily_move_stats_table,
    fetch_daily_prices,
    fetch_stats_by_date,
    fetch_stats_by_sigma,
    delete_stats_for_tickers,
    delete_stats_before_date,
    upsert_stats,
)
from src.data_access.document_store import (
    load_document,
    save_document,
    delete_document,
    document_exists,
    clear_cache,
)
from src.data_access.metadata_repository import (
    fetch_stock_metadata_df,
    fetch_stock_metadata_map,
    fetch_alerts_list,
    fetch_alerts_df,
    fetch_portfolios,
    refresh_caches,
)
from src.data_access.portfolio_repository import list_portfolios

__all__ = [
    # alert_repository
    "list_alerts",
    "refresh_alert_cache",
    # daily_move_stats_repository
    "ensure_daily_move_stats_table",
    "fetch_daily_prices",
    "fetch_stats_by_date",
    "fetch_stats_by_sigma",
    "delete_stats_for_tickers",
    "delete_stats_before_date",
    "upsert_stats",
    # document_store
    "load_document",
    "save_document",
    "delete_document",
    "document_exists",
    "clear_cache",
    # metadata_repository
    "fetch_stock_metadata_df",
    "fetch_stock_metadata_map",
    "fetch_alerts_list",
    "fetch_alerts_df",
    "fetch_portfolios",
    "refresh_caches",
    # portfolio_repository
    "list_portfolios",
]

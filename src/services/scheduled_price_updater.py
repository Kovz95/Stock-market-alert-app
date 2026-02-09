"""
Scheduled Price Updater

Handles scheduled price updates for stocks grouped by exchange/country.
Integrates with the alert system and Discord notifications for failed updates.
"""

from __future__ import annotations

import logging
import os
import time as time_module
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

from src.data_access.metadata_repository import fetch_stock_metadata_map
from src.services.backend_fmp_optimized import OptimizedDailyPriceCollector
from src.services.price_update_monitor import PriceUpdateMonitor
from src.utils.reference_data import EXCHANGE_COUNTRY_MAP, get_country_for_exchange

logger = logging.getLogger(__name__)

# Default number of parallel workers for price updates (env override: SCHEDULER_PRICE_UPDATE_WORKERS)
DEFAULT_PRICE_UPDATE_WORKERS = int(os.getenv("SCHEDULER_PRICE_UPDATE_WORKERS", "20"))


def _update_one_ticker(
    ticker: str,
    resample_weekly: bool,
) -> Tuple[str, bool, Dict[str, Any], Optional[str]]:
    """
    Update a single ticker in isolation (one collector per call for thread safety).
    Used by ThreadPoolExecutor in update_exchange_prices.
    """
    collector = OptimizedDailyPriceCollector()
    try:
        result = collector.update_ticker(ticker, resample_weekly=resample_weekly)
        stats = collector.get_statistics()
        return (ticker, result, stats, None)
    except Exception as e:
        logger.exception("Error updating %s: %s", ticker, e)
        return (ticker, False, {}, str(e))
    finally:
        collector.close()

# Re-export for callers that import from this module
__all__ = [
    "ScheduledPriceUpdater",
    "update_prices_for_exchanges",
    "run_alert_checks",
    "get_country_for_exchange",
]


class ScheduledPriceUpdater:
    """
    Scheduled price updater that processes stocks by exchange/country.

    Integrates with:
    - OptimizedDailyPriceCollector for efficient price fetching (uses DailyPriceRepository)
    - PriceUpdateMonitor for Discord notifications of failures
    """

    def __init__(self) -> None:
        self.collector = OptimizedDailyPriceCollector()
        self.monitor = PriceUpdateMonitor()
        self.exchange_mapping = EXCHANGE_COUNTRY_MAP
        self.metadata = fetch_stock_metadata_map()

    def get_tickers_for_exchange(self, exchange: str) -> List[str]:
        """
        Get all tickers for a specific exchange.

        Args:
            exchange: The exchange name (e.g., 'NASDAQ', 'NYSE')

        Returns:
            List of ticker symbols
        """
        tickers = []
        for symbol, info in self.metadata.items():
            if isinstance(info, dict) and info.get("exchange") == exchange:
                tickers.append(symbol)
        return sorted(tickers)

    def get_tickers_for_country(self, country: str) -> List[str]:
        """
        Get all tickers for a specific country.

        Args:
            country: The country name (e.g., 'United States', 'Japan')

        Returns:
            List of ticker symbols
        """
        country_exchanges = {
            ex for ex, c in self.exchange_mapping.items()
            if c == country
        }

        tickers = []
        for symbol, info in self.metadata.items():
            if isinstance(info, dict):
                exchange = info.get("exchange")
                if exchange and exchange.upper() in {e.upper() for e in country_exchanges}:
                    tickers.append(symbol)
        return sorted(tickers)

    def get_exchanges_for_country(self, country: str) -> List[str]:
        """Get all exchanges for a given country."""
        return [
            ex for ex, c in self.exchange_mapping.items()
            if c == country
        ]

    def update_exchange_prices(
        self,
        tickers: List[str],
        *,
        resample_weekly: bool = False,
        batch_size: int = 50,
        rate_limit_delay: float = 0.2,
        max_workers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Update prices for a list of tickers (optionally in parallel).

        Args:
            tickers: List of ticker symbols to update
            resample_weekly: Whether to also update weekly data
            batch_size: Number of tickers per batch (used when max_workers <= 1)
            rate_limit_delay: Delay between API calls in seconds (used when max_workers <= 1)
            max_workers: Number of parallel workers (default from SCHEDULER_PRICE_UPDATE_WORKERS or 5).
                Use 1 for sequential (original behavior).

        Returns:
            Statistics dictionary with update counts
        """
        workers = max_workers if max_workers is not None else DEFAULT_PRICE_UPDATE_WORKERS
        stats: Dict[str, Any] = {
            "total": len(tickers),
            "updated": 0,
            "skipped": 0,
            "skipped_tickers": [],
            "failed": 0,
            "failed_tickers": [],
            "new": 0,
            "stale": 0,
            "records_fetched": 0,
        }

        if not tickers:
            return stats

        start_time = time_module.time()

        if workers <= 1:
            # Sequential (original behavior)
            for i in range(0, len(tickers), batch_size):
                batch = tickers[i : i + batch_size]
                for ticker in batch:
                    try:
                        result = self.collector.update_ticker(
                            ticker,
                            resample_weekly=resample_weekly,
                        )
                        if result:
                            collector_stats = self.collector.get_statistics()
                            if ticker in collector_stats.get("skipped_tickers", []):
                                stats["skipped"] += 1
                                stats["skipped_tickers"].append(ticker)
                            else:
                                stats["updated"] += 1
                        else:
                            stats["failed"] += 1
                            stats["failed_tickers"].append({
                                "ticker": ticker,
                                "error": "Update returned False",
                            })
                        time_module.sleep(rate_limit_delay)
                    except Exception as e:
                        logger.error("Error updating %s: %s", ticker, e)
                        stats["failed"] += 1
                        stats["failed_tickers"].append({"ticker": ticker, "error": str(e)})
        else:
            # Parallel: one collector per ticker (each has its own DB connection)
            logger.info("Running price update with %d workers", workers)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_update_one_ticker, ticker, resample_weekly): ticker
                    for ticker in tickers
                }
                for future in as_completed(futures):
                    try:
                        ticker, result, coll_stats, err = future.result()
                        if err:
                            stats["failed"] += 1
                            stats["failed_tickers"].append({"ticker": ticker, "error": err})
                        elif result:
                            stats["skipped"] += coll_stats.get("skipped", 0)
                            stats["skipped_tickers"].extend(coll_stats.get("skipped_tickers", []))
                            stats["updated"] += coll_stats.get("updated", 0)
                            stats["new"] += coll_stats.get("new", 0)
                            stats["records_fetched"] += coll_stats.get("records_fetched", 0)
                        else:
                            stats["failed"] += 1
                            stats["failed_tickers"].append({
                                "ticker": ticker,
                                "error": "Update returned False",
                            })
                    except Exception as e:
                        ticker = futures.get(future, "?")
                        logger.exception("Worker error for %s: %s", ticker, e)
                        stats["failed"] += 1
                        stats["failed_tickers"].append({"ticker": str(ticker), "error": str(e)})

        stats["duration_seconds"] = time_module.time() - start_time

        if stats["failed_tickers"]:
            self.monitor.report_failed_updates(stats["failed_tickers"])

        return stats

    def run_scheduled_update(
        self,
        exchanges: Optional[List[str]] = None,
        countries: Optional[List[str]] = None,
        *,
        resample_weekly: bool = False,
        max_workers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run a scheduled price update for the given exchanges or countries.

        Args:
            exchanges: List of exchange names to update
            countries: List of country names to update
            resample_weekly: Whether to also update weekly data
            max_workers: Number of parallel workers (default from env or 5). Use 1 for sequential.

        Returns:
            Statistics dictionary with update counts
        """
        tickers: Set[str] = set()
        exchange_names: List[str] = []

        if exchanges:
            for exchange in exchanges:
                exchange_names.append(exchange)
                tickers.update(self.get_tickers_for_exchange(exchange))

        if countries:
            for country in countries:
                country_exchanges = self.get_exchanges_for_country(country)
                exchange_names.extend(country_exchanges)
                tickers.update(self.get_tickers_for_country(country))

        ticker_list = sorted(tickers)

        logger.info(
            "Starting scheduled update for %d tickers from %s",
            len(ticker_list),
            ", ".join(exchange_names) if exchange_names else "all exchanges",
        )

        stats = self.update_exchange_prices(
            ticker_list,
            resample_weekly=resample_weekly,
            max_workers=max_workers,
        )

        stats["exchanges"] = exchange_names

        summary = {
            "exchange": ", ".join(exchange_names) if exchange_names else "Unknown",
            "total": stats["total"],
            "successful": stats["updated"] + stats.get("new", 0),
            "failed": stats["failed"],
            "skipped": stats["skipped"],
            "skipped_tickers": stats.get("skipped_tickers", []),
            "duration_seconds": stats.get("duration_seconds", 0),
        }
        self.monitor.report_update_summary(summary)

        return stats

    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self.collector, "close"):
            self.collector.close()


def update_prices_for_exchanges(
    exchanges: List[str],
    *,
    resample_weekly: bool = False,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Update prices for the given exchanges.

    Convenience function that creates an updater, runs the update, and returns
    the statistics. Used by auto_scheduler_v2 and maintenance scripts.

    Args:
        exchanges: List of exchange names to update
        resample_weekly: Whether to also update weekly data
        max_workers: Number of parallel workers (default from env or 5). Use 1 for sequential.

    Returns:
        Statistics dictionary
    """
    updater = ScheduledPriceUpdater()
    try:
        return updater.run_scheduled_update(
            exchanges=exchanges,
            resample_weekly=resample_weekly,
            max_workers=max_workers,
        )
    finally:
        updater.close()


def run_alert_checks(
    exchanges: List[str],
    timeframe_key: str = "daily",
) -> Dict[str, Any]:
    """
    Run alert checks for the given exchanges.

    Args:
        exchanges: List of exchange names to check alerts for
        timeframe_key: The timeframe to check ('daily' or 'weekly')

    Returns:
        Statistics dictionary with alert check results
    """
    stats = {
        "total": 0,
        "success": 0,
        "triggered": 0,
        "errors": 0,
        "no_data": 0,
        "stale_data": 0,
    }

    try:
        from src.data_access.alert_repository import list_alerts

        metadata = fetch_stock_metadata_map()

        exchange_tickers = set()
        for symbol, info in metadata.items():
            if isinstance(info, dict) and info.get("exchange") in exchanges:
                exchange_tickers.add(symbol)

        all_alerts = list_alerts()
        relevant_alerts = []

        for alert in all_alerts:
            alert_ticker = alert.get("ticker")
            alert_exchange = alert.get("exchange")
            alert_timeframe = alert.get("timeframe", "daily")

            if alert_exchange in exchanges or alert_ticker in exchange_tickers:
                if timeframe_key == "weekly" and alert_timeframe.lower() in ("weekly", "1wk"):
                    relevant_alerts.append(alert)
                elif timeframe_key == "daily" and alert_timeframe.lower() in ("daily", "1d"):
                    relevant_alerts.append(alert)
                elif timeframe_key == "hourly" and alert_timeframe.lower() in ("hourly", "1h", "1hr"):
                    relevant_alerts.append(alert)

        stats["total"] = len(relevant_alerts)

        logger.info(
            "Found %d alerts for exchanges %s (timeframe: %s)",
            len(relevant_alerts),
            exchanges,
            timeframe_key,
        )

        # Actually run the alert checker (previously this only counted alerts)
        if relevant_alerts:
            from src.services.stock_alert_checker import StockAlertChecker

            checker = StockAlertChecker()
            check_stats = checker.check_alerts(relevant_alerts, timeframe_key)
            stats["success"] = check_stats.get("success", 0)
            stats["triggered"] = check_stats.get("triggered", 0)
            stats["errors"] = check_stats.get("errors", 0)
            stats["no_data"] = check_stats.get("no_data", 0)
            stats["stale_data"] = check_stats.get("stale_data", 0)

    except Exception as e:
        logger.error("Error running alert checks: %s", e)
        stats["errors"] = 1

    return stats

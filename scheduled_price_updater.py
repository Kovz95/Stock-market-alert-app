"""
Scheduled Price Updater

Handles scheduled price updates for stocks grouped by exchange/country.
Integrates with the alert system and Discord notifications for failed updates.
"""

from __future__ import annotations

import logging
import time as time_module
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from backend_fmp_optimized import OptimizedDailyPriceCollector
from daily_price_collector import DailyPriceDatabase
from data_access.metadata_repository import fetch_stock_metadata_map
from price_update_monitor import PriceUpdateMonitor

try:
    from exchange_country_mapping import EXCHANGE_COUNTRY_MAP as EXCHANGE_TO_COUNTRY
except ImportError:
    EXCHANGE_TO_COUNTRY = {}

logger = logging.getLogger(__name__)

# Fallback exchange to country mapping if the import fails
FALLBACK_EXCHANGE_COUNTRY_MAP = {
    # Asia-Pacific
    "TOKYO": "Japan",
    "TAIWAN": "Taiwan",
    "HONG KONG": "Hong Kong",
    "SINGAPORE": "Singapore",
    "MALAYSIA": "Malaysia",
    "INDONESIA": "Indonesia",
    "THAILAND": "Thailand",
    "ASX": "Australia",
    "OMX NORDIC ICELAND": "Iceland",
    "OMX NORDIC STOCKHOLM": "Sweden",
    "OMX NORDIC HELSINKI": "Finland",
    "OMX NORDIC COPENHAGEN": "Denmark",

    # India
    "BSE INDIA": "India",
    "NSE INDIA": "India",

    # Europe
    "LONDON": "United Kingdom",
    "XETRA": "Germany",
    "FRANKFURT": "Germany",
    "EURONEXT AMSTERDAM": "Netherlands",
    "EURONEXT BRUSSELS": "Belgium",
    "EURONEXT DUBLIN": "Ireland",
    "EURONEXT LISBON": "Portugal",
    "EURONEXT PARIS": "France",
    "MILAN": "Italy",
    "MADRID": "Spain",
    "SPAIN": "Spain",
    "VIENNA": "Austria",
    "WARSAW": "Poland",
    "ATHENS": "Greece",
    "PRAGUE": "Czech Republic",
    "BUDAPEST": "Hungary",
    "OSLO": "Norway",
    "SIX SWISS": "Switzerland",

    # Americas
    "NASDAQ": "United States",
    "NYSE": "United States",
    "NYSE AMERICAN": "United States",
    "NYSE ARCA": "United States",
    "CBOE BZX": "United States",
    "TORONTO": "Canada",
    "SANTIAGO": "Chile",
    "BUENOS AIRES": "Argentina",
    "MEXICO": "Mexico",
    "COLOMBIA": "Colombia",
    "SAO PAULO": "Brazil",

    # Middle East / Africa
    "ISTANBUL": "Turkey",
    "JSE": "South Africa",
}


def _get_exchange_mapping() -> Dict[str, str]:
    """
    Return the exchange->country map.
    Falls back to local mapping if the module import failed.
    """
    if EXCHANGE_TO_COUNTRY:
        return EXCHANGE_TO_COUNTRY
    return FALLBACK_EXCHANGE_COUNTRY_MAP


class ScheduledPriceUpdater:
    """
    Scheduled price updater that processes stocks by exchange/country.

    Integrates with:
    - OptimizedDailyPriceCollector for efficient price fetching
    - DailyPriceDatabase for storage
    - PriceUpdateMonitor for Discord notifications of failures
    """

    def __init__(self):
        self.collector = OptimizedDailyPriceCollector()
        self.db = DailyPriceDatabase()
        self.monitor = PriceUpdateMonitor()
        self.exchange_mapping = _get_exchange_mapping()
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
        # Get all exchanges for this country
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
    ) -> Dict[str, Any]:
        """
        Update prices for a list of tickers.

        Args:
            tickers: List of ticker symbols to update
            resample_weekly: Whether to also update weekly data
            batch_size: Number of tickers per batch
            rate_limit_delay: Delay between API calls in seconds

        Returns:
            Statistics dictionary with update counts
        """
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

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]

            for ticker in batch:
                try:
                    result = self.collector.update_ticker(
                        ticker,
                        resample_weekly=resample_weekly
                    )

                    if result:
                        # Check collector stats to see if it was skipped or updated
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
                            "error": "Update returned False"
                        })

                    # Rate limiting
                    time_module.sleep(rate_limit_delay)

                except Exception as e:
                    logger.error(f"Error updating {ticker}: {e}")
                    stats["failed"] += 1
                    stats["failed_tickers"].append({
                        "ticker": ticker,
                        "error": str(e)
                    })

        stats["duration_seconds"] = time_module.time() - start_time

        # Report failures to Discord
        if stats["failed_tickers"]:
            self.monitor.report_failed_updates(stats["failed_tickers"])

        return stats

    def run_scheduled_update(
        self,
        exchanges: Optional[List[str]] = None,
        countries: Optional[List[str]] = None,
        *,
        resample_weekly: bool = False,
    ) -> Dict[str, Any]:
        """
        Run a scheduled price update for the given exchanges or countries.

        Args:
            exchanges: List of exchange names to update
            countries: List of country names to update
            resample_weekly: Whether to also update weekly data

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
            ", ".join(exchange_names) if exchange_names else "all exchanges"
        )

        stats = self.update_exchange_prices(
            ticker_list,
            resample_weekly=resample_weekly
        )

        # Add exchange info to stats
        stats["exchanges"] = exchange_names

        # Report summary
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

    def close(self):
        """Clean up resources."""
        if hasattr(self.collector, "close"):
            self.collector.close()
        if hasattr(self.db, "close"):
            self.db.close()


# Module-level convenience functions for use by auto_scheduler_v2

def update_prices_for_exchanges(
    exchanges: List[str],
    *,
    resample_weekly: bool = False,
) -> Dict[str, Any]:
    """
    Update prices for the given exchanges.

    This is a convenience function that creates an updater, runs the update,
    and returns the statistics.

    Args:
        exchanges: List of exchange names to update
        resample_weekly: Whether to also update weekly data

    Returns:
        Statistics dictionary
    """
    updater = ScheduledPriceUpdater()
    try:
        return updater.run_scheduled_update(
            exchanges=exchanges,
            resample_weekly=resample_weekly
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
        # Import alert checking functionality
        from data_access.alert_repository import list_alerts

        # Get metadata to filter by exchange
        metadata = fetch_stock_metadata_map()

        # Get tickers for these exchanges
        exchange_tickers = set()
        for symbol, info in metadata.items():
            if isinstance(info, dict) and info.get("exchange") in exchanges:
                exchange_tickers.add(symbol)

        # Get alerts that match these exchanges
        all_alerts = list_alerts()
        relevant_alerts = []

        for alert in all_alerts:
            alert_ticker = alert.get("ticker")
            alert_exchange = alert.get("exchange")
            alert_timeframe = alert.get("timeframe", "daily")

            # Check if alert matches the exchange filter
            if alert_exchange in exchanges or alert_ticker in exchange_tickers:
                # Check if alert matches the timeframe
                if timeframe_key == "weekly" and alert_timeframe.lower() == "weekly":
                    relevant_alerts.append(alert)
                elif timeframe_key == "daily" and alert_timeframe.lower() in ("daily", "1d"):
                    relevant_alerts.append(alert)

        stats["total"] = len(relevant_alerts)

        # Note: Actual alert evaluation would be done by the alert processor
        # This function just returns the count of relevant alerts
        logger.info(
            "Found %d alerts for exchanges %s (timeframe: %s)",
            len(relevant_alerts),
            exchanges,
            timeframe_key
        )

    except Exception as e:
        logger.error(f"Error running alert checks: {e}")
        stats["errors"] = 1

    return stats


def get_country_for_exchange(exchange: str) -> Optional[str]:
    """
    Get the country for a given exchange.

    Args:
        exchange: The exchange name

    Returns:
        Country name or None if not found
    """
    mapping = _get_exchange_mapping()
    if not exchange:
        return None
    return mapping.get(exchange.upper())


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if len(sys.argv) > 1:
        # Update specific exchange
        exchange = sys.argv[1]
        resample_weekly = "--weekly" in sys.argv

        print(f"Updating prices for {exchange}...")
        stats = update_prices_for_exchanges([exchange], resample_weekly=resample_weekly)
        print(f"Results: {stats}")
    else:
        # Print usage
        print("Usage: python scheduled_price_updater.py <EXCHANGE> [--weekly]")
        print("Example: python scheduled_price_updater.py NASDAQ")
        print("         python scheduled_price_updater.py NYSE --weekly")

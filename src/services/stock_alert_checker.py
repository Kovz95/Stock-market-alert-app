"""
Stock Alert Checker - Evaluates stock alerts and sends Discord notifications

This module processes stock alerts by:
1. Loading alerts from the alert repository
2. Fetching current price data from FMP API
3. Evaluating conditions using backend.evaluate_expression()
4. Sending Discord notifications via discord_routing.send_economy_alert()
5. Logging to the audit trail
6. Updating alert status (last_triggered timestamp)
"""
from pytz import timezone

import logging
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.services.backend import evaluate_expression, evaluate_expression_list
from src.services.backend_fmp import FMPDataFetcher
from src.data_access.alert_repository import list_alerts, update_alert, bulk_update_last_triggered
from src.data_access.metadata_repository import fetch_stock_metadata_map
from src.services.discord_routing import (
    send_economy_discord_alert,
    resolve_alert_webhook_url,
    resolve_alert_custom_webhook_urls,
    format_alert_as_embed,
)
from src.utils.async_discord_queue import queue_discord_notification, get_discord_queue
from src.utils.discord_message_accumulator import DiscordMessageAccumulator
from src.utils.discord_rate_limiter import get_rate_limiter
from src.data_access.daily_price_repository import DailyPriceRepository
from src.services.portfolio_discord import portfolio_manager
from src.services.alert_audit_logger import (
    DeferredAuditRecord,
    audit_logger,
    log_alert_check_start,
    log_price_data_pulled,
    log_no_data_available,
    log_conditions_evaluated,
    log_error,
    log_completion,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default number of parallel workers for alert checks (env override: SCHEDULER_ALERT_CHECK_WORKERS)
DEFAULT_ALERT_CHECK_WORKERS = int(os.getenv("SCHEDULER_ALERT_CHECK_WORKERS", "5"))

# Use async Discord notifications by default (env override: ASYNC_DISCORD_NOTIFICATIONS)
USE_ASYNC_DISCORD = os.getenv("ASYNC_DISCORD_NOTIFICATIONS", "true").lower() in ("true", "1", "yes")


class StockAlertChecker:
    """
    Evaluates stock alerts and sends Discord notifications when conditions are met.
    Thread-safe for parallel check_alerts when using a shared price cache.
    """

    def __init__(self, async_discord: bool = USE_ASYNC_DISCORD):
        """
        Initialize the alert checker.

        Args:
            async_discord: If True, queue Discord notifications for background sending.
                          This prevents rate limits from blocking the main job.
        """
        self.fmp = FMPDataFetcher()
        self._price_repo = DailyPriceRepository()
        self._thread_local = threading.local()
        self.price_cache: Dict[str, pd.DataFrame] = {}
        self._cache_lock = threading.Lock()
        self.async_discord = async_discord

        # Thread-safe collections for deferred batch operations
        self._deferred_lock = threading.Lock()
        self._deferred_last_triggered: List[tuple[str, str]] = []
        self._deferred_audit_records: List[DeferredAuditRecord] = []

        if self.async_discord:
            logger.info("Using async Discord notifications (background queue)")

    def get_price_data(
        self,
        ticker: str,
        timeframe: str = "1d",
        days: int = 200
    ) -> Optional[pd.DataFrame]:
        """
        Get price data for a stock ticker (cache access is thread-safe).

        Lookup order:
        1. In-memory cache (shared across workers)
        2. PostgreSQL/SQLite DB (populated by scheduler price pre-fetch)
        3. FMP API fallback (for tickers not yet in DB)

        The source is tracked in self._thread_local.last_price_source for audit logging.

        Args:
            ticker: Stock symbol (e.g., "AAPL", "MSFT")
            timeframe: Timeframe for the data ("1d", "1wk", "1h")
            days: Number of days of historical data to fetch

        Returns:
            DataFrame with OHLCV data, or None if unavailable
        """
        cache_key = f"{ticker}_{timeframe}"
        self._thread_local.last_price_source = "unknown"

        # 1. Check in-memory cache
        with self._cache_lock:
            if cache_key in self.price_cache:
                logger.debug(f"Using cached data for {ticker}")
                self._thread_local.last_price_source = "cache"
                return self.price_cache[cache_key].copy()

        # 2. Try reading from DB (populated by scheduler's update_prices_for_exchanges)
        try:
            df = None
            if timeframe in ("1wk", "weekly"):
                df = self._price_repo.get_weekly_prices(ticker, limit=days)
                db_source = "DB-weekly"
            elif timeframe in ("1h", "hourly"):
                df = self._price_repo.get_hourly_prices(ticker, limit=days)
                db_source = "DB-hourly"
            else:
                df = self._price_repo.get_daily_prices(ticker, limit=days)
                db_source = "DB-daily"

            if df is not None and not df.empty:
                with self._cache_lock:
                    self.price_cache[cache_key] = df
                self._thread_local.last_price_source = db_source
                logger.debug(f"Loaded {len(df)} price records for {ticker} from {db_source}")
                return df.copy()

            logger.debug(f"No DB data for {ticker} ({timeframe}), falling back to FMP")
        except Exception as e:
            logger.warning(f"DB read failed for {ticker}, falling back to FMP: {e}")

        # 3. Fall back to FMP API (handles new tickers or DB gaps)
        try:
            if timeframe in ("1wk", "weekly"):
                df = self.fmp.get_historical_data(ticker, period="1day", timeframe="1wk")
            elif timeframe in ("1h", "hourly"):
                df = self.fmp.get_historical_data(ticker, period="1hour")
            else:
                df = self.fmp.get_historical_data(ticker, period="1day")

            if df is not None and not df.empty:
                if len(df) > days:
                    df = df.tail(days)
                with self._cache_lock:
                    self.price_cache[cache_key] = df
                self._thread_local.last_price_source = "FMP-fallback"
                logger.debug(f"Fetched {len(df)} price records for {ticker} from FMP (fallback)")
                return df
            else:
                logger.warning(f"No price data available for {ticker}")
                return None

        except Exception as e:
            logger.error(f"Error fetching price data for {ticker}: {e}")
            return None

    def extract_conditions(self, alert: Dict[str, Any]) -> List[str]:
        """
        Extract condition strings from an alert.

        Args:
            alert: Alert dictionary

        Returns:
            List of condition strings to evaluate
        """
        conditions = alert.get("conditions", [])
        result = []

        if isinstance(conditions, list):
            for cond in conditions:
                if isinstance(cond, dict):
                    # Handle dict format: {"conditions": "Close[-1] > 100"}
                    cond_str = cond.get("conditions", "")
                    if cond_str:
                        result.append(cond_str)
                elif isinstance(cond, str):
                    # Handle string format directly
                    result.append(cond)
                elif isinstance(cond, list) and len(cond) > 0:
                    # Handle nested list format
                    if isinstance(cond[0], str):
                        result.append(cond[0])

        return result

    def evaluate_alert(self, alert: Dict[str, Any], df: pd.DataFrame) -> bool:
        """
        Evaluate an alert's conditions against price data.

        Args:
            alert: Alert dictionary with conditions
            df: DataFrame with price data

        Returns:
            True if alert conditions are met, False otherwise
        """
        try:
            conditions = self.extract_conditions(alert)
            if not conditions:
                logger.debug(f"No conditions found for alert {alert.get('alert_id')}")
                return False

            combination = alert.get("combination_logic", "AND")
            vals = {"ticker": alert.get("ticker", "UNKNOWN")}

            # Use evaluate_expression_list for multiple conditions
            result = evaluate_expression_list(
                df, conditions, combination, vals=vals
            )
            return bool(result)

        except Exception as e:
            logger.error(f"Error evaluating alert {alert.get('alert_id')}: {e}")
            return False

    def format_alert_message(
        self,
        alert: Dict[str, Any],
        df: pd.DataFrame
    ) -> str:
        """
        Format a Discord notification message for a triggered alert.

        Args:
            alert: Alert dictionary
            df: DataFrame with price data

        Returns:
            Formatted message string
        """
        ticker = alert.get("ticker", "Unknown")
        name = alert.get("name", alert.get("stock_name", "Unnamed Alert"))
        current_price = df["Close"].iloc[-1] if not df.empty else 0
        timeframe = alert.get("timeframe", "daily")

        # Get stock metadata for economy classification and ISIN
        stock_metadata = fetch_stock_metadata_map()
        stock_info = stock_metadata.get(ticker, {})
        economy = stock_info.get("rbics_economy", "Unknown")
        isin = stock_info.get("isin", "N/A")

        # Get conditions for display
        conditions = self.extract_conditions(alert)
        conditions_str = "\n".join(f"  • {c}" for c in conditions[:3])  # Show first 3
        if len(conditions) > 3:
            conditions_str += f"\n  ... and {len(conditions) - 3} more"

        exchange = alert.get("exchange", "Unknown")

        message = f"""🚨 **Stock Alert Triggered**

**{name}**
• Ticker: `{ticker}`
• Exchange: {exchange}
• Economy: {economy}
• ISIN: {isin}
• Price: ${current_price:.2f}
• Timeframe: {timeframe}

**Conditions Met:**
{conditions_str}

*Triggered at {datetime.now(tz=timezone('UTC')).astimezone(timezone('US/Eastern')).strftime('%Y-%m-%d %I:%M:%S %p ET')}*"""

        return message

    def should_skip_alert(self, alert: Dict[str, Any]) -> bool:
        """
        Check if an alert should be skipped (already triggered today).

        Args:
            alert: Alert dictionary

        Returns:
            True if alert should be skipped, False otherwise
        """
        last_triggered = alert.get("last_triggered", "")
        if not last_triggered:
            return False

        try:
            # Parse the last triggered timestamp
            if isinstance(last_triggered, str):
                # Handle ISO format with or without timezone
                triggered_date = datetime.fromisoformat(
                    last_triggered.replace("Z", "+00:00")
                )
            else:
                triggered_date = last_triggered

            # Skip if already triggered today
            if triggered_date.date() == datetime.now().date():
                return True

        except (ValueError, AttributeError) as e:
            logger.debug(f"Could not parse last_triggered: {e}")

        return False

    def check_alert(
        self,
        alert: Dict[str, Any],
        accumulator: Optional[DiscordMessageAccumulator] = None,
    ) -> Dict[str, Any]:
        """
        Check a single alert and send notification if triggered.

        Args:
            alert: Alert dictionary
            accumulator: Optional DiscordMessageAccumulator for batched sending.
                         When provided, embeds are added to the accumulator instead
                         of being queued via the async queue.

        Returns:
            Result dictionary with status information
        """
        start_time = time.time()
        alert_id = alert.get("alert_id", "unknown")
        ticker = alert.get("ticker", alert.get("ticker1", ""))

        result = {
            "alert_id": alert_id,
            "ticker": ticker,
            "triggered": False,
            "error": None,
            "skipped": False,
        }

        # Use deferred audit record: collects all fields in-memory,
        # writes a single INSERT on flush (reduces 5 DB round-trips to 1).
        audit_record = DeferredAuditRecord(alert, "scheduled")

        try:
            # Check if alert is disabled (only skip if explicitly set to "off")
            if alert.get("action", "on") == "off":
                result["skipped"] = True
                result["skip_reason"] = "disabled"
                return result

            # Check if already triggered today
            if self.should_skip_alert(alert):
                result["skipped"] = True
                result["skip_reason"] = "already_triggered_today"
                logger.debug(f"Alert {alert_id} already triggered today, skipping")
                return result

            # Get ticker - handle ratio alerts
            if alert.get("ratio") == "Yes" or alert.get("is_ratio"):
                ticker1 = alert.get("ticker1", "")
                ticker2 = alert.get("ticker2", "")
                if not ticker1 or not ticker2:
                    result["error"] = "Missing ticker1 or ticker2 for ratio alert"
                    audit_record.set_error(result["error"])
                    return result
                ticker = ticker1  # Use ticker1 as primary for data fetch
            elif not ticker:
                result["error"] = "No ticker specified"
                audit_record.set_error(result["error"])
                return result

            # Get price data
            timeframe = alert.get("timeframe", "1d")
            df = self.get_price_data(ticker, timeframe)

            if df is None or df.empty:
                result["error"] = f"No price data for {ticker}"
                audit_record.set_no_data(ticker)
                return result

            # Record price source from thread-local (set by get_price_data)
            price_source = getattr(self._thread_local, "last_price_source", "unknown")
            cache_key = f"{ticker}_{timeframe}"
            audit_record.set_price_pulled(price_source, cache_hit=(price_source == "cache"))

            # Evaluate conditions
            triggered = self.evaluate_alert(alert, df)

            trigger_reason = "conditions_met" if triggered else None
            audit_record.set_conditions_evaluated(triggered, trigger_reason)

            if triggered:
                result["triggered"] = True
                logger.info(f"TRIGGERED: {ticker} - {alert.get('name', '')}")

                # Format the Discord notification message
                message = self.format_alert_message(alert, df)

                # Send Discord notification
                if accumulator is not None:
                    # Accumulator path: collect embeds for batched sending
                    webhook_url = resolve_alert_webhook_url(alert)
                    embed = format_alert_as_embed(alert, message)

                    if webhook_url and embed:
                        accumulator.add(webhook_url, embed)

                    # Also accumulate for custom channels
                    custom_urls = resolve_alert_custom_webhook_urls(alert)
                    for custom_url in custom_urls:
                        accumulator.add(custom_url, embed)

                elif self.async_discord:
                    # Async queue path (legacy fallback)
                    webhook_url = resolve_alert_webhook_url(alert)
                    embed = format_alert_as_embed(alert, message)

                    queued = queue_discord_notification(
                        alert, message, send_economy_discord_alert,
                        webhook_url=webhook_url,
                        embed=embed,
                    )
                    if queued:
                        logger.debug(f"Queued Discord notification for {ticker}")
                    else:
                        logger.warning(f"Failed to queue Discord notification for {ticker}")

                    custom_urls = resolve_alert_custom_webhook_urls(alert)
                    for custom_url in custom_urls:
                        queue_discord_notification(
                            alert, message, send_economy_discord_alert,
                            webhook_url=custom_url,
                            embed=embed,
                        )
                else:
                    # Synchronous sending - blocks until complete
                    success = send_economy_discord_alert(alert, message)
                    if success:
                        logger.info(f"Discord notification sent for {ticker}")
                    else:
                        logger.warning(f"Failed to send Discord notification for {ticker}")

                # Send to portfolio channels
                try:
                    matching_portfolios = portfolio_manager.get_portfolios_for_stock(ticker)
                    for portfolio_id, portfolio_data in matching_portfolios:
                        portfolio_webhook = portfolio_data.get('discord_webhook')
                        if portfolio_webhook:
                            if accumulator is not None:
                                if embed is None:
                                    embed = format_alert_as_embed(alert, message)
                                accumulator.add(portfolio_webhook, embed)
                            elif self.async_discord:
                                if embed is None:
                                    embed = format_alert_as_embed(alert, message)
                                queue_discord_notification(
                                    alert, message, send_economy_discord_alert,
                                    webhook_url=portfolio_webhook,
                                    embed=embed,
                                )
                            else:
                                portfolio_manager.send_portfolio_alert(
                                    {'message': message, 'ticker': ticker},
                                    portfolio_id, portfolio_data,
                                )
                except Exception as e:
                    logger.warning(f"Error sending portfolio alerts for {ticker}: {e}")

                # Defer last_triggered update for batch processing
                triggered_ts = datetime.now().isoformat()
                with self._deferred_lock:
                    self._deferred_last_triggered.append((alert_id, triggered_ts))

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Error checking alert {alert_id}: {e}")
            audit_record.set_error(str(e))

        finally:
            # Record completion time but defer the DB write
            execution_time = int((time.time() - start_time) * 1000)
            audit_record.set_completion(execution_time)
            with self._deferred_lock:
                self._deferred_audit_records.append(audit_record)

        return result

    def check_alerts(
        self,
        alerts: List[Dict[str, Any]],
        timeframe_filter: Optional[str] = None,
        max_workers: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Check multiple alerts and return statistics (optionally in parallel).

        Args:
            alerts: List of alert dictionaries
            timeframe_filter: Optional filter for timeframe ("daily", "weekly")
            max_workers: Number of parallel workers (default from env or 5). Use 1 for sequential.

        Returns:
            Statistics dictionary with check results
        """
        stats = {
            "total": 0,
            "triggered": 0,
            "errors": 0,
            "skipped": 0,
            "no_data": 0,
            "success": 0,
        }

        # Reload portfolio data so it's fresh for this run
        try:
            portfolio_manager.load_portfolios()
        except Exception as e:
            logger.warning(f"Error reloading portfolios: {e}")

        # Filter alerts by timeframe if specified
        filtered_alerts = []
        for alert in alerts:
            if not isinstance(alert, dict):
                continue

            alert_timeframe = alert.get("timeframe", "daily").lower()

            # Check timeframe filter
            if timeframe_filter:
                if timeframe_filter == "weekly" and alert_timeframe not in ("weekly", "1wk"):
                    continue
                elif timeframe_filter == "daily" and alert_timeframe not in ("daily", "1d"):
                    continue
                elif timeframe_filter == "hourly" and alert_timeframe not in ("hourly", "1h", "1hr"):
                    continue

            filtered_alerts.append(alert)

        stats["total"] = len(filtered_alerts)
        workers = max_workers if max_workers is not None else DEFAULT_ALERT_CHECK_WORKERS
        logger.info(
            "Checking %d %s alerts (workers=%d)",
            len(filtered_alerts),
            timeframe_filter or "all",
            workers,
        )

        # Pre-warm price cache: batch-load from DB by timeframe so workers
        # only hit the in-memory cache. This avoids lock contention and
        # reduces ~1,800 sequential DB queries to a handful of batch queries.
        unique_tickers = set()
        for a in filtered_alerts:
            t = a.get("ticker") or a.get("ticker1", "")
            tf = a.get("timeframe", "1d")
            if t:
                unique_tickers.add((t, tf))
        if unique_tickers:
            logger.info(
                "Pre-warming price cache for %d unique tickers (daily=%s, weekly=%s, hourly=%s)",
                len(unique_tickers),
                sum(1 for _, tf in unique_tickers if tf not in ("1wk", "weekly", "1h", "hourly")),
                sum(1 for _, tf in unique_tickers if tf in ("1wk", "weekly")),
                sum(1 for _, tf in unique_tickers if tf in ("1h", "hourly")),
            )

            # Group tickers by timeframe
            daily_tickers: list[str] = []
            weekly_tickers: list[str] = []
            hourly_tickers: list[str] = []
            for ticker, tf in unique_tickers:
                if tf in ("1wk", "weekly"):
                    weekly_tickers.append(ticker)
                elif tf in ("1h", "hourly"):
                    hourly_tickers.append(ticker)
                else:
                    daily_tickers.append(ticker)

            # Batch-load from DB
            batch_loaded = 0
            if daily_tickers:
                logger.info("Loading daily prices for %d tickers from DB...", len(daily_tickers))
                daily_results = self._price_repo.get_daily_prices_batch(daily_tickers, limit=200)
                for tk, df in daily_results.items():
                    self.price_cache[f"{tk}_1d"] = df
                    batch_loaded += 1
                logger.info(
                    "Loaded %d/%d daily tickers from DB (sample: %s)",
                    len(daily_results),
                    len(daily_tickers),
                    list(daily_results.keys())[:5],
                )
            if weekly_tickers:
                logger.info("Loading weekly prices for %d tickers from DB...", len(weekly_tickers))
                weekly_results = self._price_repo.get_weekly_prices_batch(weekly_tickers, limit=200)
                for tk, df in weekly_results.items():
                    self.price_cache[f"{tk}_1wk"] = df
                    batch_loaded += 1
                logger.info("Loaded %d/%d weekly tickers from DB", len(weekly_results), len(weekly_tickers))
            if hourly_tickers:
                logger.info("Loading hourly prices for %d tickers from DB...", len(hourly_tickers))
                hourly_results = self._price_repo.get_hourly_prices_batch(hourly_tickers, limit=200)
                for tk, df in hourly_results.items():
                    self.price_cache[f"{tk}_1h"] = df
                    batch_loaded += 1
                logger.info("Loaded %d/%d hourly tickers from DB", len(hourly_results), len(hourly_tickers))

            logger.info("Batch-loaded %d tickers from DB into price cache", batch_loaded)

            # Fall back to individual fetch (FMP API) for any tickers
            # missing from the batch DB results
            missing = 0
            for ticker, tf in unique_tickers:
                cache_key = f"{ticker}_{tf}"
                if cache_key not in self.price_cache:
                    missing += 1
                    self.get_price_data(ticker, tf)
            if missing:
                logger.info("Individually fetched %d tickers missing from DB via FMP API", missing)
            else:
                logger.info("All %d tickers found in DB — no FMP API fallback needed", len(unique_tickers))

        # Create accumulator for batched Discord sends (auto_flush=False
        # defers all HTTP sends until flush_all after all alerts finish)
        accumulator = DiscordMessageAccumulator(
            rate_limiter=get_rate_limiter(), auto_flush=False
        )

        # Reset deferred collections for this run
        with self._deferred_lock:
            self._deferred_last_triggered.clear()
            self._deferred_audit_records.clear()

        processed_count = 0
        log_interval = max(len(filtered_alerts) // 10, 100)  # Log every ~10% or 100

        def _aggregate_result(result: Dict[str, Any]) -> None:
            nonlocal processed_count
            processed_count += 1

            if result.get("skipped"):
                stats["skipped"] += 1
            elif result.get("error"):
                if "No price data" in str(result["error"]):
                    stats["no_data"] += 1
                else:
                    stats["errors"] += 1
            elif result.get("triggered"):
                stats["triggered"] += 1
                stats["success"] += 1
            else:
                stats["success"] += 1

            # Periodic progress logging
            if processed_count % log_interval == 0 or processed_count == len(filtered_alerts):
                logger.info(
                    "Alert evaluation progress: %d/%d checked "
                    "(triggered=%d, errors=%d, no_data=%d, skipped=%d)",
                    processed_count,
                    len(filtered_alerts),
                    stats["triggered"],
                    stats["errors"],
                    stats["no_data"],
                    stats["skipped"],
                )

        logger.info("Starting alert evaluation for %d alerts...", len(filtered_alerts))

        try:
            if workers <= 1:
                for alert in filtered_alerts:
                    try:
                        result = self.check_alert(alert, accumulator=accumulator)
                        _aggregate_result(result)
                    except Exception as e:
                        logger.error(f"Unexpected error checking alert: {e}")
                        stats["errors"] += 1
            else:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    future_to_alert = {
                        executor.submit(self.check_alert, alert, accumulator): alert
                        for alert in filtered_alerts
                    }
                    for future in as_completed(future_to_alert):
                        try:
                            result = future.result()
                            _aggregate_result(result)
                        except Exception as e:
                            logger.exception("Unexpected error checking alert: %s", e)
                            stats["errors"] += 1

            # --- Batch post-processing (all I/O deferred from hot path) ---

            # 1. Flush all Discord embeds
            acc_stats = accumulator.get_stats()
            logger.info(
                "Alert evaluation complete. Flushing %d Discord embeds to %d webhooks...",
                acc_stats["added"],
                len(accumulator._buckets),
            )
            accumulator.flush_all()
            acc_stats = accumulator.get_stats()
            logger.info(
                "Discord send complete: added=%d, sent=%d, failed=%d, flushes=%d",
                acc_stats["added"], acc_stats["sent"],
                acc_stats["failed"], acc_stats["flushes"],
            )

            # 2. Bulk-update last_triggered timestamps
            with self._deferred_lock:
                trigger_updates = list(self._deferred_last_triggered)
                self._deferred_last_triggered.clear()
            if trigger_updates:
                try:
                    bulk_update_last_triggered(trigger_updates)
                    logger.info("Bulk-updated last_triggered for %d alerts", len(trigger_updates))
                except Exception as e:
                    logger.error("Error bulk-updating last_triggered: %s", e)

            # 3. Bulk-flush audit records
            with self._deferred_lock:
                audit_records = list(self._deferred_audit_records)
                self._deferred_audit_records.clear()
            if audit_records:
                try:
                    audit_logger.bulk_flush(audit_records)
                except Exception as e:
                    logger.error("Error bulk-flushing audit records: %s", e)

        finally:
            # Clean up the DB connection used for price reads
            try:
                self._price_repo.close()
            except Exception as e:
                logger.warning("Error closing price repo: %s", e)

        logger.info(
            f"Alert check complete: {stats['triggered']} triggered, "
            f"{stats['errors']} errors, {stats['skipped']} skipped"
        )

        return stats

    def check_all_alerts(self, timeframe_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Load and check all active alerts.

        Args:
            timeframe_filter: Optional filter for timeframe ("daily", "weekly")

        Returns:
            Statistics dictionary with check results
        """
        logger.info("=" * 60)
        logger.info("STOCK ALERT CHECKER")
        logger.info("=" * 60)

        try:
            # Load all alerts from repository
            all_alerts = list_alerts()
            logger.info(f"Loaded {len(all_alerts)} total alerts")

            # Filter to active alerts only
            active_alerts = [
                a for a in all_alerts
                if isinstance(a, dict) and a.get("action", "on") == "on"
            ]
            logger.info(f"Found {len(active_alerts)} active alerts")

            # Check alerts
            stats = self.check_alerts(active_alerts, timeframe_filter)

            return stats

        except Exception as e:
            logger.error(f"Error in check_all_alerts: {e}")
            return {
                "total": 0,
                "triggered": 0,
                "errors": 1,
                "skipped": 0,
                "no_data": 0,
                "success": 0,
            }

    def check_alerts_for_exchanges(
        self,
        exchanges: List[str],
        timeframe_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check alerts for specific exchanges.

        Args:
            exchanges: List of exchange names to check
            timeframe_filter: Optional filter for timeframe ("daily", "weekly")

        Returns:
            Statistics dictionary with check results
        """
        logger.info(f"Checking alerts for exchanges: {exchanges}")

        try:
            # Load all alerts
            logger.info("Loading alerts from repository...")
            all_alerts = list_alerts()
            logger.info("Loaded %d total alerts from repository", len(all_alerts))

            # Filter by exchange
            exchange_alerts = [
                a for a in all_alerts
                if isinstance(a, dict) and a.get("exchange") in exchanges
            ]

            logger.info(
                "Filtered to %d alerts for exchanges %s (from %d total)",
                len(exchange_alerts),
                exchanges,
                len(all_alerts),
            )

            # Check alerts
            return self.check_alerts(exchange_alerts, timeframe_filter)

        except Exception as e:
            logger.error(f"Error checking alerts for exchanges: {e}")
            return {
                "total": 0,
                "triggered": 0,
                "errors": 1,
                "skipped": 0,
                "no_data": 0,
                "success": 0,
            }


def check_stock_alerts(
    exchanges: Optional[List[str]] = None,
    timeframe_filter: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to check stock alerts.

    Args:
        exchanges: Optional list of exchanges to filter by
        timeframe_filter: Optional timeframe filter ("daily", "weekly")

    Returns:
        Statistics dictionary
    """
    checker = StockAlertChecker()

    if exchanges:
        return checker.check_alerts_for_exchanges(exchanges, timeframe_filter)
    else:
        return checker.check_all_alerts(timeframe_filter)


def main():
    """Main function for standalone execution."""
    logger.info("Starting stock alert checker...")

    checker = StockAlertChecker()
    stats = checker.check_all_alerts()

    logger.info(f"Results: {stats}")
    return 0 if stats.get("errors", 0) == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

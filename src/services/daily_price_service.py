"""
Daily price collection service: orchestrates fetching and storing daily/weekly
prices for all alert tickers. Entry point for full daily update (script/scheduler).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, time as dt_time
from typing import List, Optional

import pandas as pd

from src.data_access.alert_repository import list_alerts
from src.data_access.daily_price_repository import DailyPriceRepository
from src.data_access.metadata_repository import fetch_stock_metadata_map
from src.services.backend_fmp import FMPDataFetcher

logger = logging.getLogger(__name__)


class DailyPriceCollector:
    """
    Collects daily price data for all tickers from metadata and alerts.
    Uses FMP backend and DailyPriceRepository for storage.
    """

    def __init__(self) -> None:
        self.db = DailyPriceRepository()
        self.fetcher = FMPDataFetcher()
        self.stats = {
            "updated": 0,
            "skipped": 0,
            "failed": 0,
            "new": 0,
            "weekly_updated": 0,
        }

    def get_all_tickers(self) -> List[str]:
        """Get unique tickers from metadata and alert repositories."""
        tickers = set(fetch_stock_metadata_map().keys())
        if tickers:
            logger.info("Found %s tickers from stock metadata", len(tickers))
        try:
            alerts = list_alerts()
        except Exception as exc:
            logger.error("Error loading alerts from repository: %s", exc)
            alerts = []
        for alert in alerts:
            ticker = alert.get("ticker")
            if ticker:
                tickers.add(ticker)
            if (alert.get("ratio") == "Yes") or alert.get("is_ratio"):
                ticker1 = alert.get("ticker1")
                ticker2 = alert.get("ticker2")
                if ticker1:
                    tickers.add(ticker1)
                if ticker2:
                    tickers.add(ticker2)
        return sorted(tickers)

    def update_ticker(self, ticker: str) -> bool:
        """Update daily prices for a single ticker."""
        try:
            needs_update, last_update = self.db.needs_update(ticker)
            if not needs_update:
                logger.debug("%s: Up to date (last: %s)", ticker, last_update)
                self.stats["skipped"] += 1
                return True
            logger.info("Updating %s...", ticker)
            df = self.fetcher.get_historical_data(ticker, period="1day", timeframe="1d")
            if df is None or df.empty:
                logger.warning("%s: No data received", ticker)
                self.stats["failed"] += 1
                return False
            records = self.db.store_daily_prices(ticker, df)
            weekly_df = self._resample_to_weekly(df, ticker)
            if weekly_df is not None:
                weekly_records = self.db.store_weekly_prices(ticker, weekly_df)
                if weekly_records > 0:
                    self.stats["weekly_updated"] += 1
            if last_update:
                self.stats["updated"] += 1
            else:
                self.stats["new"] += 1
            logger.info("%s: Stored %s daily records", ticker, records)
            return True
        except Exception as e:
            logger.error("Error updating %s: %s", ticker, e)
            self.stats["failed"] += 1
            return False

    def _resample_to_weekly(self, df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        """Resample daily data to weekly using actual last trading day."""
        try:
            if df.empty:
                return None
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df_copy = df.copy()
            df_copy["year_week"] = df_copy.index.strftime("%Y-%U")
            weekly_data = []
            for year_week, week_df in df_copy.groupby("year_week"):
                if week_df.empty:
                    continue
                last_trading_day = week_df.index[-1]
                weekly_data.append({
                    "date": last_trading_day,
                    "Open": week_df["Open"].iloc[0],
                    "High": week_df["High"].max(),
                    "Low": week_df["Low"].min(),
                    "Close": week_df["Close"].iloc[-1],
                    "Volume": week_df["Volume"].sum(),
                })
            if not weekly_data:
                return None
            weekly_df = pd.DataFrame(weekly_data)
            weekly_df.set_index("date", inplace=True)
            logger.debug("%s: Resampled to %s weekly records", ticker, len(weekly_df))
            return weekly_df
        except Exception as e:
            logger.error("Error resampling %s: %s", ticker, e)
            return None

    def close(self) -> None:
        """Clean up."""
        self.db.close()


def run_full_daily_update() -> None:
    """
    Update price data for ALL tickers in the database.
    Best run at night (e.g., 11 PM) after all markets have closed.
    """
    logger.info("=" * 70)
    logger.info("FULL DAILY PRICE UPDATE - %s", datetime.now())
    logger.info("=" * 70)

    collector = DailyPriceCollector()

    try:
        tickers = collector.get_all_tickers()
        logger.info("Found %s tickers to update", len(tickers))

        if len(tickers) == 0:
            logger.error("No tickers found")
            return

        stats = {"updated": 0, "skipped": 0, "failed": 0, "already_current": 0}
        start_time = time.time()
        batch_size = 50

        for i in range(0, len(tickers), batch_size):
            batch = tickers[i : i + batch_size]
            batch_end = min(i + batch_size, len(tickers))
            logger.info(
                "Processing batch %s (%s-%s/%s)",
                i // batch_size + 1,
                i + 1,
                batch_end,
                len(tickers),
            )

            for ticker in batch:
                try:
                    db = DailyPriceRepository()
                    needs_update, last_update = db.needs_update(ticker)
                    db.close()

                    if not needs_update:
                        stats["already_current"] += 1
                        continue

                    if collector.update_ticker(ticker):
                        stats["updated"] += 1
                    else:
                        stats["failed"] += 1

                    time.sleep(0.2)
                except Exception as e:
                    logger.error("Error updating %s: %s", ticker, e)
                    stats["failed"] += 1

            if (i + len(batch)) % 500 == 0:
                elapsed = time.time() - start_time
                rate = (i + len(batch)) / elapsed if elapsed > 0 else 0
                remaining = len(tickers) - (i + len(batch))
                eta = remaining / rate if rate > 0 else 0
                logger.info("Progress: %s/%s tickers", i + len(batch), len(tickers))
                logger.info(
                    "Stats: Updated=%s, Current=%s, Failed=%s",
                    stats["updated"],
                    stats["already_current"],
                    stats["failed"],
                )
                logger.info("ETA: %s minutes", eta / 60)

        elapsed = time.time() - start_time
        logger.info("=" * 70)
        logger.info("UPDATE COMPLETE")
        logger.info("=" * 70)
        logger.info("Total time: %s minutes", elapsed / 60)
        logger.info("Updated: %s tickers", stats["updated"])
        logger.info("Already current: %s tickers", stats["already_current"])
        logger.info("Failed: %s tickers", stats["failed"])

        db = DailyPriceRepository()
        try:
            db_stats = db.get_statistics()
            logger.info(
                "Database: %s tickers, %s daily records, %s weekly records, %s MB",
                db_stats["total_tickers"],
                f"{db_stats['total_daily_records']:,}",
                f"{db_stats['total_weekly_records']:,}",
                f"{db_stats.get('db_size_mb') or 0:.1f}",
            )
        finally:
            db.close()

    except Exception as e:
        logger.error("Error during update: %s", e)
    finally:
        collector.close()


def run_daily_update() -> None:
    """
    Run the daily update process (with market-close check).
    Should be scheduled to run after market close.
    """
    logger.info("=" * 70)
    logger.info("Daily Price Update - %s", datetime.now())
    logger.info("=" * 70)
    now = datetime.now()
    market_close = dt_time(16, 30)
    if now.time() < market_close and now.weekday() < 5:
        logger.warning("Market is still open, skipping update")
        return
    run_full_daily_update()

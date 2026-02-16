"""
Template Method pattern for exchange job execution.

Defines BaseExchangeJobHandler with a template ``execute()`` method that
orchestrates the full job lifecycle: acquire lock -> notify start -> run
worker -> notify complete/error -> release lock.

Concrete subclasses (DailyJobHandler, WeeklyJobHandler) override three
properties to customise behaviour per timeframe.

Usage:
    from src.services.scheduler_services import SchedulerServices
    from src.services.scheduler_job_handler import DailyJobHandler, WeeklyJobHandler

    services = SchedulerServices()
    daily = DailyJobHandler(services)
    result = daily.execute("NASDAQ")
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.services.scheduler_services import (
    JOB_TIMEOUT_SECONDS,
    SchedulerServices,
)

logger = logging.getLogger("auto_scheduler_v2")


# ---------------------------------------------------------------------------
# Worker helpers (module-level for pickle-ability with ThreadPoolExecutor)
# ---------------------------------------------------------------------------

def _exchange_worker(exchange_name: str, resample_weekly: bool):
    """Execute price update + alert checks for a single exchange."""
    from src.services.scheduled_price_updater import update_prices_for_exchanges
    from src.services.auto_scheduler_v2 import run_alert_checks

    price_stats = update_prices_for_exchanges([exchange_name], resample_weekly=resample_weekly)
    timeframe_key = "weekly" if resample_weekly else "daily"
    alert_stats = run_alert_checks([exchange_name], timeframe_key)
    return price_stats, alert_stats


def _run_exchange_job(exchange_name: str, resample_weekly: bool, timeout_seconds: int):
    """
    Run a single exchange job on a thread with a soft timeout.

    Returns (price_stats, alert_stats, error_message).
    """
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_exchange_worker, exchange_name, resample_weekly)
        try:
            price_stats, alert_stats = future.result(timeout=timeout_seconds)
            return price_stats, alert_stats, None
        except TimeoutError:
            logger.warning(
                "Job for %s timed out after %ds (worker thread will finish in background)",
                exchange_name,
                timeout_seconds,
            )
            return None, None, f"timeout after {timeout_seconds}s"
        except Exception as exc:
            return None, None, str(exc)


# ---------------------------------------------------------------------------
# Base handler (Template Method)
# ---------------------------------------------------------------------------

class BaseExchangeJobHandler(ABC):
    """Template Method pattern for exchange job lifecycle.

    The ``execute()`` method defines the invariant job lifecycle:
    acquire lock -> notify start -> run worker -> notify complete/error ->
    release lock.

    Subclasses customise behaviour by overriding three abstract properties.
    """

    def __init__(self, services: SchedulerServices) -> None:
        self.services = services

    @property
    @abstractmethod
    def job_type(self) -> str:
        """Job type identifier, e.g. ``'daily'`` or ``'weekly'``."""

    @property
    @abstractmethod
    def resample_weekly(self) -> bool:
        """Whether to resample price data to weekly candles."""

    @property
    @abstractmethod
    def timeframe_key(self) -> str:
        """Timeframe key for alert filtering, e.g. ``'daily'`` or ``'weekly'``."""

    def execute(self, exchange_name: str) -> Optional[Dict[str, Any]]:
        """Template method: full job lifecycle for a single exchange.

        Args:
            exchange_name: Name of the exchange to process.

        Returns:
            Dictionary with ``price_stats``, ``alert_stats``, and ``duration``
            on success, or ``None`` on failure.
        """
        from src.services.scheduler_discord import create_scheduler_discord

        job_id = self.services.sanitize_job_id(self.job_type, exchange_name)
        if not self.services.acquire_job_lock(job_id):
            logger.warning("Job %s is already running; skipping duplicate execution.", job_id)
            return None

        notify_settings = (self.services.load_config() or {}).get("notification_settings", {})
        send_start = notify_settings.get("send_start_notification", True)
        send_complete = notify_settings.get("send_completion_notification", True)
        job_timeout = max(int(notify_settings.get("job_timeout_seconds", JOB_TIMEOUT_SECONDS)), 60)

        run_time_utc = datetime.now(tz=timezone.utc)
        discord_notifier = create_scheduler_discord(self.job_type)

        start_time = time.time()
        if send_start:
            discord_notifier.notify_start(run_time_utc, exchange_name)

        try:
            self.services.update_status(
                status="running",
                current_job={
                    "id": job_id,
                    "exchange": exchange_name,
                    "job_type": self.job_type,
                    "started": run_time_utc.isoformat(),
                },
            )

            price_stats, alert_stats, worker_err = _run_exchange_job(
                exchange_name,
                resample_weekly=self.resample_weekly,
                timeout_seconds=job_timeout,
            )

            if worker_err:
                raise TimeoutError(worker_err) if "timeout" in worker_err else RuntimeError(worker_err)

            duration = round(time.time() - start_time, 2)

            self.services.update_status(
                status="running",
                current_job=None,
                last_run={
                    "job_id": job_id,
                    "exchange": exchange_name,
                    "job_type": self.job_type,
                    "completed_at": datetime.now(tz=timezone.utc).isoformat(),
                    "duration_seconds": duration,
                },
                last_result={
                    "price_stats": price_stats,
                    "alert_stats": alert_stats,
                },
            )

            logger.info(
                "%s job finished for %s (price updates: %s, alerts: %s)",
                self.job_type.upper(),
                exchange_name,
                price_stats.get("updated", 0) if isinstance(price_stats, dict) else price_stats,
                alert_stats.get("total", 0) if isinstance(alert_stats, dict) else alert_stats,
            )

            if send_complete:
                first_failure_reason = (
                    price_stats.get("first_failure_reason")
                    if isinstance(price_stats, dict)
                    else None
                )
                discord_notifier.notify_complete(
                    run_time_utc,
                    duration,
                    price_stats if isinstance(price_stats, dict) else {},
                    alert_stats if isinstance(alert_stats, dict) else {},
                    exchange_name,
                    first_failure_reason=first_failure_reason,
                )

            return {
                "price_stats": price_stats,
                "alert_stats": alert_stats,
                "duration": duration,
            }

        except Exception as exc:
            err_msg = (
                f"Timeout after {job_timeout}s"
                if isinstance(exc, TimeoutError)
                else str(exc)
            )
            logger.exception("Error running %s job for %s: %s", self.job_type, exchange_name, err_msg)
            self.services.update_status(
                status="error",
                current_job=None,
                last_error={
                    "time": datetime.now(tz=timezone.utc).isoformat(),
                    "job_id": job_id,
                    "exchange": exchange_name,
                    "job_type": self.job_type,
                    "message": err_msg,
                },
            )
            discord_notifier.notify_error(run_time_utc, err_msg)
            return None
        finally:
            self.services.release_job_lock(job_id)


# ---------------------------------------------------------------------------
# Concrete handlers
# ---------------------------------------------------------------------------

class DailyJobHandler(BaseExchangeJobHandler):
    """Handler for daily exchange jobs."""

    @property
    def job_type(self) -> str:
        return "daily"

    @property
    def resample_weekly(self) -> bool:
        return False

    @property
    def timeframe_key(self) -> str:
        return "daily"


class WeeklyJobHandler(BaseExchangeJobHandler):
    """Handler for weekly exchange jobs."""

    @property
    def job_type(self) -> str:
        return "weekly"

    @property
    def resample_weekly(self) -> bool:
        return True

    @property
    def timeframe_key(self) -> str:
        return "weekly"

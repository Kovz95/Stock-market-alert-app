"""
Daily and Weekly scheduler Discord notifications using the Template Method pattern.

BaseSchedulerDiscord holds all notification logic. Subclasses override three
abstract properties (job_label, timeframe_key, config_key) to produce
job-type-specific messages.

Usage:
    from src.services.scheduler_discord import create_scheduler_discord

    notifier = create_scheduler_discord("daily")
    notifier.notify_start(run_time_utc, exchange_name)
"""

from __future__ import annotations
from zoneinfo import ZoneInfo

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Iterable, Optional

import requests
from src.data_access.document_store import load_document

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (mirrored from hourly_scheduler_discord.py)
# ---------------------------------------------------------------------------

def _utc_str(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def _est_str(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %I:%M:%S %p")


def _format_duration(seconds: float) -> str:
    seconds = int(max(0, round(seconds)))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseSchedulerDiscord(ABC):
    """
    Abstract base class for per-job-type Discord notifiers.

    Subclasses must override three properties:
        job_label     – human-readable label, e.g. "Daily"
        timeframe_key – timeframe string, e.g. "1d"
        config_key    – key in discord_channels_config.logging_channels
    """

    def __init__(self) -> None:
        self.config = self._load_config()
        self.fallback_webhook = self._load_scheduler_webhook()

    # -- abstract interface --------------------------------------------------

    @property
    @abstractmethod
    def job_label(self) -> str:
        """Human-readable job label, e.g. 'Daily'."""

    @property
    @abstractmethod
    def timeframe_key(self) -> str:
        """Timeframe string sent in messages, e.g. '1d'."""

    @property
    @abstractmethod
    def config_key(self) -> str:
        """Key in discord_channels_config.logging_channels."""

    # -- config loading ------------------------------------------------------

    def _load_config(self) -> dict:
        try:
            config = load_document(
                "discord_channels_config",
                default={},
                fallback_path="discord_channels_config.json",
            ) or {}
            return config.get("logging_channels", {}).get(self.config_key, {})
        except Exception as exc:
            logger.error("Error loading Discord config for %s: %s", self.config_key, exc)
            return {}

    @staticmethod
    def _load_scheduler_webhook() -> str | None:
        """
        Use the scheduler status webhook as a fallback when the job-specific
        webhook is missing, preventing silent notification loss.
        """
        try:
            cfg = load_document(
                "scheduler_config",
                default={},
                fallback_path="scheduler_config.json",
            ) or {}
            webhook_cfg = cfg.get("scheduler_webhook", {})
            if webhook_cfg.get("enabled") and webhook_cfg.get("url"):
                return webhook_cfg["url"]
        except Exception as exc:
            logger.error("Error loading scheduler webhook fallback: %s", exc)
        return None

    # -- posting -------------------------------------------------------------

    def _post(self, message: str) -> bool:
        from src.utils.discord_env import is_discord_send_enabled
        if not is_discord_send_enabled():
            return False
        webhook_url = self.config.get("webhook_url") or self.fallback_webhook
        if not webhook_url or webhook_url == "YOUR_WEBHOOK_URL_HERE":
            logger.warning(
                "%s scheduler webhook not configured; message skipped.", self.job_label
            )
            return False
        try:
            from src.utils.discord_env import get_discord_environment_tag
            response = requests.post(
                webhook_url,
                json={"content": get_discord_environment_tag() + message},
                timeout=10,
            )
            if response.status_code in (200, 204):
                return True
            logger.error(
                "Failed to send %s scheduler message: HTTP %s %s",
                self.job_label,
                response.status_code,
                response.text,
            )
        except Exception as exc:
            logger.error("Error sending %s scheduler message: %s", self.job_label, exc)
        return False

    # -- public notification API --------------------------------------------

    def notify_start(
        self,
        run_time_utc: datetime,
        exchange: Optional[str] = None,
        candle_type: Optional[str] = None,
        exchanges: Optional[Iterable[str]] = None,
        symbol_count: Optional[int] = None,
        close_info: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Send start notification.

        This base implementation handles daily/weekly schedulers. Subclasses can
        override this method to provide custom formatting (e.g., hourly scheduler).

        Args:
            run_time_utc: UTC timestamp of the run
            exchange: Single exchange name (for daily/weekly schedulers)
            candle_type: Type of candle window (for hourly scheduler)
            exchanges: List of exchanges (for hourly scheduler)
            symbol_count: Number of symbols queued (for hourly scheduler)
            close_info: Optional close time information (for hourly scheduler)
            **kwargs: Additional keyword arguments for future extensibility
        """
        # Default implementation for daily/weekly schedulers
        if exchange is not None:
            message = "\n".join(
                [
                    f"✅ **{self.job_label} Alert Check Started**",
                    f"• Run Time (EST): {_est_str(run_time_utc)}",
                    f"• Timeframe: {self.timeframe_key}",
                    f"• Exchange: {exchange}",
                ]
            )
            self._post(message)

    def notify_skipped(self, run_time_utc: datetime, reason: str) -> None:
        message = "\n".join(
            [
                f"⚪ **{self.job_label} Alert Check Skipped**",
                f"• Run Time (EST): {_est_str(run_time_utc)}",
                f"• Reason: {reason}",
            ]
        )
        self._post(message)

    def notify_complete(
        self,
        run_time_utc: datetime,
        duration_seconds: float,
        price_stats: Optional[dict] = None,
        alert_stats: Optional[dict] = None,
        exchange: Optional[str] = None,
        first_failure_reason: Optional[str] = None,
        stats: Optional[dict] = None,
        exchanges: Optional[Iterable[str]] = None,
        **kwargs,
    ) -> None:
        """
        Send completion notification.

        This base implementation handles daily/weekly schedulers. Subclasses can
        override this method to provide custom formatting (e.g., hourly scheduler).

        Args:
            run_time_utc: UTC timestamp of the run
            duration_seconds: Duration of the job in seconds
            price_stats: Price update statistics (for daily/weekly schedulers)
            alert_stats: Alert check statistics
            exchange: Single exchange name (for daily/weekly schedulers)
            first_failure_reason: Optional first failure error message
            stats: Price update statistics (for hourly scheduler)
            exchanges: List of exchanges (for hourly scheduler)
            **kwargs: Additional keyword arguments for future extensibility
        """
        # Default implementation for daily/weekly schedulers
        if exchange is not None:
            price_stats = price_stats or {}
            alert_stats = alert_stats or {}

            price_line = (
                f"• Price Update: {price_stats.get('updated', 0):,}/{price_stats.get('total', price_stats.get('updated', 0) + price_stats.get('failed', 0) + price_stats.get('skipped', 0)):,} updated"
                f" | failed {price_stats.get('failed', 0):,}"
                f" | skipped {price_stats.get('skipped', 0):,}"
            )
            alert_line = (
                f"• Alerts: total {alert_stats.get('total', 0):,}"
                f" | triggered {alert_stats.get('triggered', alert_stats.get('success', 0)):,}"
                f" | not triggered {alert_stats.get('not_triggered', 0):,}"
                f" | no data {alert_stats.get('no_data', 0):,}"
                f" | stale {alert_stats.get('stale_data', 0):,}"
                f" | errors {alert_stats.get('errors', 0):,}"
            )

            lines = [
                f"✅ **{self.job_label} Alert Check Complete**",
                f"• Run Time (EST): {_est_str(run_time_utc)}",
                f"• Duration: {_format_duration(duration_seconds)}",
                f"• Timeframe: {self.timeframe_key}",
                f"• Exchange: {exchange}",
                price_line,
                alert_line,
            ]

            if price_stats.get("failed", 0) > 0 and first_failure_reason:
                reason = (
                    first_failure_reason[:200] + "..."
                    if len(first_failure_reason) > 200
                    else first_failure_reason
                )
                lines.append(f"• First failure: {reason}")

            self._post("\n".join(lines))

    def notify_error(self, run_time_utc: datetime, error: str) -> None:
        message = "\n".join(
            [
                f"❌ **{self.job_label} Scheduler Error**",
                f"• Run Time (EST): {_est_str(run_time_utc)}",
                f"• Error: {error}",
            ]
        )
        self._post(message)

    def notify_scheduler_start(self, schedule_info: str) -> None:
        message = "\n".join(
            [
                f"✅ **{self.job_label} Scheduler Online**",
                f"• Schedules: {schedule_info}",
                f"• Timestamp (EST): {_est_str(datetime.now(tz=timezone.utc))}",
            ]
        )
        self._post(message)

    def notify_scheduler_stop(self) -> None:
        message = "\n".join(
            [
                f"⏹️ **{self.job_label} Scheduler Stopped**",
                f"• Timestamp (EST): {_est_str(datetime.now(tz=timezone.utc))}",
            ]
        )
        self._post(message)


# ---------------------------------------------------------------------------
# Concrete subclasses
# ---------------------------------------------------------------------------

class DailySchedulerDiscord(BaseSchedulerDiscord):
    """Discord notifier for daily exchange jobs."""

    @property
    def job_label(self) -> str:
        return "Daily"

    @property
    def timeframe_key(self) -> str:
        return "1d"

    @property
    def config_key(self) -> str:
        return "Daily_Scheduler_Status"


class WeeklySchedulerDiscord(BaseSchedulerDiscord):
    """Discord notifier for weekly exchange jobs."""

    @property
    def job_label(self) -> str:
        return "Weekly"

    @property
    def timeframe_key(self) -> str:
        return "1wk"

    @property
    def config_key(self) -> str:
        return "Weekly_Scheduler_Status"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_NOTIFIER_MAP: dict[str, type[BaseSchedulerDiscord]] = {
    "daily": DailySchedulerDiscord,
    "weekly": WeeklySchedulerDiscord,
}


def create_scheduler_discord(job_type: str) -> BaseSchedulerDiscord:
    """
    Factory function — returns the correct notifier for *job_type*.

    Args:
        job_type: Either ``"daily"`` or ``"weekly"``.

    Returns:
        A configured :class:`BaseSchedulerDiscord` subclass instance.

    Raises:
        ValueError: If *job_type* is not recognised.
    """
    cls = _NOTIFIER_MAP.get(job_type)
    if cls is None:
        raise ValueError(
            f"Unknown job_type: {job_type!r}. Expected one of {list(_NOTIFIER_MAP)}"
        )
    return cls()

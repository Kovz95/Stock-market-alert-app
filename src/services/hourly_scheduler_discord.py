"""
Hourly scheduler Discord notifications using the BaseSchedulerDiscord pattern.

Extends BaseSchedulerDiscord with hourly-specific notification methods while
reusing the common configuration loading and posting logic.

Usage:
    from src.services.hourly_scheduler_discord import HourlySchedulerDiscord

    notifier = HourlySchedulerDiscord()
    notifier.notify_start(run_time_utc, "current", exchanges, symbol_count)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable, Optional

from src.services.scheduler_discord import BaseSchedulerDiscord, _utc_str, _format_duration, _est_str

logger = logging.getLogger(__name__)


def _format_list(items: Iterable[str], limit: int = 10) -> str:
    values = [item for item in items if item]
    if not values:
        return "None"
    if len(values) <= limit:
        return ", ".join(values)
    return f"{', '.join(values[:limit])} ... (+{len(values) - limit} more)"


class HourlySchedulerDiscord(BaseSchedulerDiscord):
    """Discord notifier for hourly exchange jobs."""

    @property
    def job_label(self) -> str:
        return "Hourly"

    @property
    def timeframe_key(self) -> str:
        return "1h"

    @property
    def config_key(self) -> str:
        return "Hourly_Scheduler_Status"

    # -- Hourly-specific notification methods -------------------------------

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
        Send hourly-specific start notification. Supports both single-exchange
        (scheduler_job_handler HourlyJobHandler) and multi-exchange (legacy
        hourly_data_scheduler) call patterns. Messages post to the
        hourly-scheduler channel (Hourly_Scheduler_Status webhook).

        Args:
            run_time_utc: UTC timestamp of the run
            exchange: Single exchange name (when called from HourlyJobHandler)
            candle_type: Type of candle window (e.g., "current", "previous")
            exchanges: List of exchanges (legacy multi-exchange run)
            symbol_count: Number of symbols queued (legacy)
            close_info: Optional close time information (legacy)
            **kwargs: Additional keyword arguments for compatibility
        """
        if exchanges is None:
            exchanges = []
        exchanges = list(exchanges)
        # Single-exchange run from scheduler_job_handler (e.g. HourlyJobHandler)
        if exchange and not exchanges:
            exchanges = [exchange]

        lines = [
            f"✅ **{self.job_label} Alert Check Started**",
            f"• Run Time (EST): {_est_str(run_time_utc)}",
            f"• Candle Window: {candle_type or 'current'}",
            f"• Alert Timeframe: {self.timeframe_key}",
            f"• Exchange(s): {_format_list(exchanges)}",
        ]
        if symbol_count is not None and (symbol_count > 0 or len(exchanges) != 1):
            lines.append(f"• Symbols Queued: {symbol_count:,}")
        if close_info:
            lines.append(f"• Close Times (UTC): {close_info}")
        self._post("\n".join(lines))

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
        Send hourly-specific completion notification. Supports both single-exchange
        (scheduler_job_handler HourlyJobHandler) and multi-exchange (legacy
        hourly_data_scheduler) call patterns. Messages post to the
        hourly-scheduler channel (Hourly_Scheduler_Status webhook).

        Args:
            run_time_utc: UTC timestamp of the run
            duration_seconds: Duration of the job in seconds
            price_stats: Price stats from handler (updated/total/failed/skipped)
            alert_stats: Alert check statistics
            exchange: Single exchange name (when called from HourlyJobHandler)
            first_failure_reason: Optional first failure error message
            stats: Legacy price stats (success/total/failed/skipped_closed)
            exchanges: List of exchanges (legacy)
            **kwargs: Additional keyword arguments for compatibility
        """
        if exchanges is None:
            exchanges = []
        exchanges = list(exchanges)
        if exchange and not exchanges:
            exchanges = [exchange]

        # Single-exchange run uses price_stats (handler shape); legacy uses stats
        use_stats = price_stats if (price_stats and "updated" in price_stats) else (stats or {})
        updated = use_stats.get("updated", use_stats.get("success", 0))
        failed = use_stats.get("failed", 0)
        skipped = use_stats.get("skipped", use_stats.get("skipped_closed", 0))
        total = use_stats.get("total")
        if total is None or total == 0:
            total = updated + failed + skipped or 1

        price_line = (
            f"• Price Update: {updated:,}/{total:,} updated "
            f"| failed {failed:,} | skipped {skipped:,}"
        )
        if alert_stats:
            triggered = alert_stats.get("triggered", alert_stats.get("success", 0))
            alert_line = (
                f"• Alerts: total {alert_stats.get('total', 0):,} | "
                f"triggered {triggered:,} | "
                f"not triggered {alert_stats.get('not_triggered', 0):,} | "
                f"no data {alert_stats.get('no_data', 0):,} | "
                f"stale {alert_stats.get('stale_data', 0):,} | "
                f"errors {alert_stats.get('errors', 0):,}"
            )
        else:
            alert_line = "• Alerts: no hourly alerts processed"

        lines = [
            f"✅ **{self.job_label} Alert Check Complete**",
            f"• Run Time (EST): {_est_str(run_time_utc)}",
            f"• Duration: {_format_duration(duration_seconds)}",
            f"• Alert Timeframe: {self.timeframe_key}",
            f"• Exchange(s): {_format_list(exchanges)}",
            price_line,
            alert_line,
        ]
        if failed > 0 and first_failure_reason:
            reason = first_failure_reason[:200] + "..." if len(first_failure_reason) > 200 else first_failure_reason
            lines.append(f"• First failure: {reason}")
        self._post("\n".join(lines))

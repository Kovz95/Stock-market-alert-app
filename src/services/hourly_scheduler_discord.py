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
        Send hourly-specific start notification with candle type and exchange info.

        Args:
            run_time_utc: UTC timestamp of the run
            candle_type: Type of candle window (e.g., "current", "previous")
            exchanges: List of exchanges being processed
            symbol_count: Number of symbols queued for processing
            close_info: Optional close time information
            exchange: Unused parameter for compatibility with base class
            **kwargs: Additional keyword arguments for compatibility
        """
        # If called with positional args, they should be non-None
        if exchanges is None:
            exchanges = []
        exchanges = list(exchanges)

        lines = [
            f"✅ **{self.job_label} Alert Check Started**",
            f"• Run Time (EST): {_est_str(run_time_utc)}",
            f"• Candle Window: {candle_type or 'unknown'}",
            f"• Alert Timeframe: {self.timeframe_key}",
            f"• Exchanges ({len(exchanges)}): {_format_list(exchanges)}",
            f"• Symbols Queued: {symbol_count or 0:,}",
        ]
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
        Send hourly-specific completion notification with exchange list.

        Args:
            run_time_utc: UTC timestamp of the run
            duration_seconds: Duration of the job in seconds
            stats: Price update statistics dictionary
            alert_stats: Alert check statistics dictionary
            exchanges: List of exchanges processed
            first_failure_reason: Optional first failure error message
            price_stats: Unused parameter for compatibility with base class
            exchange: Unused parameter for compatibility with base class
            **kwargs: Additional keyword arguments for compatibility
        """
        # If called with positional args, they should be non-None
        if exchanges is None:
            exchanges = []
        if stats is None:
            stats = {}

        exchanges = list(exchanges)
        price_line = (
            f"• Price Update: {stats.get('success', 0):,}/{stats.get('total', 0):,} updated "
            f"| failed {stats.get('failed', 0):,} | skipped {stats.get('skipped_closed', 0):,}"
        )
        if alert_stats:
            alert_line = (
                f"• Alerts: total {alert_stats.get('total', 0):,} | "
                f"triggered {alert_stats.get('triggered', 0):,} | "
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
            f"• Exchanges ({len(exchanges)}): {_format_list(exchanges)}",
            price_line,
            alert_line,
        ]
        failed = stats.get("failed", 0)
        if failed > 0 and first_failure_reason:
            # Truncate long error messages for Discord
            reason = first_failure_reason[:200] + "..." if len(first_failure_reason) > 200 else first_failure_reason
            lines.append(f"• First failure: {reason}")
        message = "\n".join(lines)
        self._post(message)

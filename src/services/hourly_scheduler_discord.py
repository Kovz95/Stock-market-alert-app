"""
Send hourly scheduler status updates to Discord in the same style as the
daily/weekly scheduler notifications.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Iterable, Optional

import requests
from src.data_access.document_store import load_document

logger = logging.getLogger(__name__)


def _utc_str(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def _format_list(items: Iterable[str], limit: int = 10) -> str:
    values = [item for item in items if item]
    if not values:
        return "None"
    if len(values) <= limit:
        return ", ".join(values)
    return f"{', '.join(values[:limit])} ... (+{len(values) - limit} more)"


def _format_duration(seconds: float) -> str:
    seconds = int(max(0, round(seconds)))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


class HourlySchedulerDiscord:
    def __init__(self) -> None:
        self.config = self._load_config()
        self.fallback_webhook = self._load_scheduler_webhook()

    @staticmethod
    def _load_config() -> dict:
        try:
            config = load_document(
                "discord_channels_config",
                default={},
                fallback_path="discord_channels_config.json",
            ) or {}
            return config.get("logging_channels", {}).get("Hourly_Scheduler_Status", {})
        except Exception as exc:
            logger.error("Error loading Discord config: %s", exc)
            return {}

    @staticmethod
    def _load_scheduler_webhook() -> str | None:
        """
        Use the scheduler status webhook as a fallback if the hourly-specific
        webhook is missing. This keeps start/complete notifications visible even
        when the hourly channel is not configured.
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

    def _post(self, message: str) -> bool:
        webhook_url = self.config.get("webhook_url") or self.fallback_webhook
        if not webhook_url or webhook_url == "YOUR_WEBHOOK_URL_HERE":
            logger.warning("Hourly scheduler webhook not configured; message skipped.")
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
                "Failed to send hourly scheduler message: HTTP %s %s",
                response.status_code,
                response.text,
            )
        except Exception as exc:
            logger.error("Error sending hourly scheduler message: %s", exc)
        return False

    def notify_start(
        self,
        run_time_utc: datetime,
        candle_type: str,
        exchanges: Iterable[str],
        symbol_count: int,
        close_info: Optional[str] = None,
    ) -> None:
        exchanges = list(exchanges)
        lines = [
            "✅ **Hourly Alert Check Started**",
            f"• Run Time (UTC): {_utc_str(run_time_utc)}",
            f"• Candle Window: {candle_type}",
            "• Alert Timeframe: 1h",
            f"• Exchanges ({len(exchanges)}): {_format_list(exchanges)}",
            f"• Symbols Queued: {symbol_count:,}",
        ]
        if close_info:
            lines.append(f"• Close Times (UTC): {close_info}")
        self._post("\n".join(lines))

    def notify_skipped(self, run_time_utc: datetime, reason: str) -> None:
        message = "\n".join(
            [
                "⚪ **Hourly Alert Check Skipped**",
                f"• Run Time (UTC): {_utc_str(run_time_utc)}",
                f"• Reason: {reason}",
            ]
        )
        self._post(message)

    def notify_complete(
        self,
        run_time_utc: datetime,
        duration_seconds: float,
        stats: dict,
        alert_stats: Optional[dict],
        exchanges: Iterable[str],
        first_failure_reason: Optional[str] = None,
    ) -> None:
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
            "✅ **Hourly Alert Check Complete**",
            f"• Run Time (UTC): {_utc_str(run_time_utc)}",
            f"• Duration: {_format_duration(duration_seconds)}",
            "• Alert Timeframe: 1h",
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

    def notify_error(self, run_time_utc: datetime, error: str) -> None:
        message = "\n".join(
            [
                "❌ **Hourly Scheduler Error**",
                f"• Run Time (UTC): {_utc_str(run_time_utc)}",
                f"• Error: {error}",
            ]
        )
        self._post(message)

    def notify_scheduler_start(self, schedule_info: str) -> None:
        message = "\n".join(
            [
                "✅ **Hourly Scheduler Online**",
                f"• Schedules: {schedule_info}",
                f"• Timestamp (UTC): {_utc_str(datetime.utcnow())}",
            ]
        )
        self._post(message)

    def notify_scheduler_stop(self) -> None:
        message = "\n".join(
            [
                "⏹️ **Hourly Scheduler Stopped**",
                f"• Timestamp (UTC): {_utc_str(datetime.utcnow())}",
            ]
        )
        self._post(message)

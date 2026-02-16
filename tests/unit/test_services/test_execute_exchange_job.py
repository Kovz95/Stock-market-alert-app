"""
Unit tests for execute_exchange_job() in src/services/auto_scheduler_v2.py

These tests verify that the function uses the create_scheduler_discord factory
and calls the correct notifier methods for each job lifecycle event, replacing
the old send_scheduler_notification() per-job calls.

Patches applied to keep tests fast and isolated:
  - create_scheduler_discord  → returns a Mock notifier (in scheduler_job_handler)
  - _run_exchange_job          → returns canned stats (in scheduler_job_handler)
  - load_document / save_document → mocked (in scheduler_services)
  - time.time (controls duration calculation)
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, Mock, call, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure heavy module dependencies don't cause issues on import
# ---------------------------------------------------------------------------
# auto_scheduler_v2 imports several services that may have side effects;
# the existing test_services conftest already mocks psycopg2, db_config, and
# stock_alert_checker.  The modules below are mocked here for safety in case
# they aren't already present from the session-scoped fixture.
_SAFE_MOCKS = [
    "src.services.scheduled_price_updater",
    "src.services.daily_price_service",
]
for _mod in _SAFE_MOCKS:
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from src.services.auto_scheduler_v2 import execute_exchange_job  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_DEFAULT_NOTIFY_SETTINGS = {
    "send_start_notification": True,
    "send_completion_notification": True,
    "job_timeout_seconds": 300,
}

_DEFAULT_PRICE_STATS = {"updated": 40, "failed": 0, "skipped": 2}
_DEFAULT_ALERT_STATS = {
    "total": 80,
    "success": 5,
    "triggered": 5,
    "errors": 0,
    "no_data": 1,
    "stale_data": 0,
}


def _run_job(
    exchange: str = "NYSE",
    job_type: str = "daily",
    notify_settings: dict | None = None,
    price_stats: dict | None = None,
    alert_stats: dict | None = None,
    worker_error: str | None = None,
    mock_notifier: Mock | None = None,
    time_side_effect: list | None = None,
):
    """
    Helper that runs execute_exchange_job with all external calls mocked.

    Returns (result, mock_notifier, mock_factory, mock_send_notif).
    """
    settings = notify_settings if notify_settings is not None else _DEFAULT_NOTIFY_SETTINGS
    ps = price_stats if price_stats is not None else _DEFAULT_PRICE_STATS
    als = alert_stats if alert_stats is not None else _DEFAULT_ALERT_STATS
    notifier = mock_notifier or Mock()
    time_vals = time_side_effect or [0.0, 1.5]  # duration = 1.5s

    # Config: load_document returns the config dict used by services.load_config()
    config_doc = {
        "notification_settings": settings,
        "scheduler_webhook": {"enabled": False, "url": ""},
    }

    with patch(
        "src.services.scheduler_discord.create_scheduler_discord",
        return_value=notifier,
    ) as mock_factory, \
         patch(
             "src.services.scheduler_services.load_document",
             return_value=config_doc,
         ), \
         patch("src.services.scheduler_services.save_document"), \
         patch(
             "src.services.scheduler_job_handler._run_exchange_job",
             return_value=(ps, als, worker_error),
         ), \
         patch("src.services.auto_scheduler_v2.send_scheduler_notification") as mock_send_notif, \
         patch("time.time", side_effect=time_vals):

        result = execute_exchange_job(exchange, job_type)

    return result, notifier, mock_factory, mock_send_notif


# ---------------------------------------------------------------------------
# Factory is called with the correct job_type
# ---------------------------------------------------------------------------

class TestFactoryCall:
    """Verifies create_scheduler_discord is invoked with the right job_type."""

    def test_factory_called_with_daily(self):
        _, _, mock_factory, _ = _run_job(job_type="daily")
        mock_factory.assert_called_once_with("daily")

    def test_factory_called_with_weekly(self):
        _, _, mock_factory, _ = _run_job(job_type="weekly")
        mock_factory.assert_called_once_with("weekly")

    def test_factory_called_exactly_once_per_job(self):
        _, _, mock_factory, _ = _run_job()
        assert mock_factory.call_count == 1


# ---------------------------------------------------------------------------
# notify_start
# ---------------------------------------------------------------------------

class TestNotifyStart:
    """Tests for the notify_start call inside execute_exchange_job."""

    def test_notify_start_called_when_enabled(self):
        settings = {**_DEFAULT_NOTIFY_SETTINGS, "send_start_notification": True}
        _, notifier, _, _ = _run_job(notify_settings=settings)
        notifier.notify_start.assert_called_once()

    def test_notify_start_not_called_when_disabled(self):
        settings = {**_DEFAULT_NOTIFY_SETTINGS, "send_start_notification": False}
        _, notifier, _, _ = _run_job(notify_settings=settings)
        notifier.notify_start.assert_not_called()

    def test_notify_start_receives_exchange_name(self):
        settings = {**_DEFAULT_NOTIFY_SETTINGS, "send_start_notification": True}
        _, notifier, _, _ = _run_job(exchange="NASDAQ", notify_settings=settings)
        _, kwargs = notifier.notify_start.call_args
        # exchange is the second positional arg
        positional = notifier.notify_start.call_args[0]
        assert "NASDAQ" in positional

    def test_notify_start_receives_run_time_utc(self):
        settings = {**_DEFAULT_NOTIFY_SETTINGS, "send_start_notification": True}
        _, notifier, _, _ = _run_job(notify_settings=settings)
        args = notifier.notify_start.call_args[0]
        from datetime import timezone
        # First arg is run_time_utc — must be timezone-aware
        assert args[0].tzinfo is not None


# ---------------------------------------------------------------------------
# notify_complete (success path)
# ---------------------------------------------------------------------------

class TestNotifyComplete:
    """Tests for the notify_complete call on the success path."""

    def test_notify_complete_called_on_success(self):
        settings = {**_DEFAULT_NOTIFY_SETTINGS, "send_completion_notification": True}
        _, notifier, _, _ = _run_job(notify_settings=settings)
        notifier.notify_complete.assert_called_once()

    def test_notify_complete_not_called_when_disabled(self):
        settings = {**_DEFAULT_NOTIFY_SETTINGS, "send_completion_notification": False}
        _, notifier, _, _ = _run_job(notify_settings=settings)
        notifier.notify_complete.assert_not_called()

    def test_notify_complete_receives_exchange_name(self):
        _, notifier, _, _ = _run_job(exchange="TSE")
        args = notifier.notify_complete.call_args[0]
        # exchange is the 5th positional arg: (run_time, duration, price, alerts, exchange)
        assert args[4] == "TSE"

    def test_notify_complete_receives_price_stats(self):
        ps = {"updated": 10, "failed": 1, "skipped": 0}
        _, notifier, _, _ = _run_job(price_stats=ps)
        args = notifier.notify_complete.call_args[0]
        assert args[2] == ps

    def test_notify_complete_receives_alert_stats(self):
        als = {"total": 20, "triggered": 3, "errors": 0, "no_data": 0, "stale_data": 0}
        _, notifier, _, _ = _run_job(alert_stats=als)
        args = notifier.notify_complete.call_args[0]
        assert args[3] == als

    def test_notify_complete_receives_duration(self):
        # time.time side_effect: start=100.0, end=115.5 → duration=15.5
        _, notifier, _, _ = _run_job(time_side_effect=[100.0, 115.5])
        args = notifier.notify_complete.call_args[0]
        assert args[1] == pytest.approx(15.5)

    def test_notify_complete_not_called_on_worker_error(self):
        """When the worker returns an error string, notify_error fires instead."""
        _, notifier, _, _ = _run_job(worker_error="timeout after 300s")
        notifier.notify_complete.assert_not_called()

    def test_notify_complete_passes_first_failure_reason_from_price_stats(self):
        ps = {**_DEFAULT_PRICE_STATS, "failed": 1, "first_failure_reason": "read timeout"}
        _, notifier, _, _ = _run_job(price_stats=ps)
        kwargs = notifier.notify_complete.call_args[1]
        assert kwargs.get("first_failure_reason") == "read timeout"

    def test_notify_complete_passes_none_when_no_first_failure_key(self):
        ps = {**_DEFAULT_PRICE_STATS}  # no first_failure_reason key
        _, notifier, _, _ = _run_job(price_stats=ps)
        kwargs = notifier.notify_complete.call_args[1]
        assert kwargs.get("first_failure_reason") is None


# ---------------------------------------------------------------------------
# notify_error (failure path)
# ---------------------------------------------------------------------------

class TestNotifyError:
    """Tests for the notify_error call on the failure path."""

    def test_notify_error_called_on_worker_error(self):
        _, notifier, _, _ = _run_job(worker_error="no result from worker")
        notifier.notify_error.assert_called_once()

    def test_notify_error_called_on_timeout_worker_error(self):
        _, notifier, _, _ = _run_job(worker_error="timeout after 300s")
        notifier.notify_error.assert_called_once()

    def test_notify_error_receives_run_time_utc(self):
        _, notifier, _, _ = _run_job(worker_error="crash")
        args = notifier.notify_error.call_args[0]
        from datetime import timezone
        assert args[0].tzinfo is not None

    def test_notify_error_receives_error_message(self):
        _, notifier, _, _ = _run_job(worker_error="crash")
        args = notifier.notify_error.call_args[0]
        # error message is second positional arg
        assert isinstance(args[1], str)
        assert len(args[1]) > 0

    def test_timeout_error_message_mentions_timeout(self):
        _, notifier, _, _ = _run_job(worker_error="timeout after 300s")
        err_msg = notifier.notify_error.call_args[0][1]
        assert "Timeout" in err_msg or "timeout" in err_msg.lower()

    def test_notify_start_not_called_on_error(self):
        """notify_start fires before the worker, so it fires even on error."""
        # But notify_complete must NOT fire on error
        settings = {**_DEFAULT_NOTIFY_SETTINGS}
        _, notifier, _, _ = _run_job(notify_settings=settings, worker_error="crash")
        notifier.notify_complete.assert_not_called()

    def test_returns_none_on_worker_error(self):
        result, _, _, _ = _run_job(worker_error="crash")
        assert result is None


# ---------------------------------------------------------------------------
# send_scheduler_notification NOT called per-job
# ---------------------------------------------------------------------------

class TestSendSchedulerNotificationNotUsed:
    """
    Verifies the old send_scheduler_notification() is not called for per-job
    start/complete/error events — those are now handled by the notifier.
    """

    def test_send_scheduler_notification_not_called_on_success(self):
        _, _, _, mock_send = _run_job()
        mock_send.assert_not_called()

    def test_send_scheduler_notification_not_called_on_worker_error(self):
        _, _, _, mock_send = _run_job(worker_error="crash")
        mock_send.assert_not_called()

    def test_send_scheduler_notification_not_called_when_start_disabled(self):
        settings = {**_DEFAULT_NOTIFY_SETTINGS, "send_start_notification": False}
        _, _, _, mock_send = _run_job(notify_settings=settings)
        mock_send.assert_not_called()


# ---------------------------------------------------------------------------
# Job locking
# ---------------------------------------------------------------------------

class TestJobLocking:
    """Verifies the job lock is acquired and released correctly."""

    def test_returns_none_when_lock_already_held(self):
        """Should return None when the job lock cannot be acquired."""
        from src.services.auto_scheduler_v2 import _services

        # Pre-acquire the lock so the handler can't get it
        _services.acquire_job_lock("daily_nyse")
        try:
            with patch("src.services.scheduler_job_handler._run_exchange_job") as mock_run:
                result = execute_exchange_job("NYSE", "daily")
            assert result is None
            mock_run.assert_not_called()
        finally:
            _services.release_job_lock("daily_nyse")

    def test_lock_released_after_success(self):
        """Lock should be released after a successful job."""
        from src.services.auto_scheduler_v2 import _services

        config_doc = {
            "notification_settings": _DEFAULT_NOTIFY_SETTINGS,
            "scheduler_webhook": {"enabled": False, "url": ""},
        }
        with patch("src.services.scheduler_discord.create_scheduler_discord", return_value=Mock()), \
             patch("src.services.scheduler_services.load_document", return_value=config_doc), \
             patch("src.services.scheduler_services.save_document"), \
             patch("src.services.scheduler_job_handler._run_exchange_job",
                   return_value=(_DEFAULT_PRICE_STATS, _DEFAULT_ALERT_STATS, None)), \
             patch("time.time", side_effect=[0.0, 1.0]):
            execute_exchange_job("NYSE", "daily")

        # Lock should be released — acquiring should succeed
        assert _services.acquire_job_lock("daily_nyse") is True
        _services.release_job_lock("daily_nyse")

    def test_lock_released_after_worker_error(self):
        """Lock should be released even when the worker errors."""
        from src.services.auto_scheduler_v2 import _services

        config_doc = {
            "notification_settings": _DEFAULT_NOTIFY_SETTINGS,
            "scheduler_webhook": {"enabled": False, "url": ""},
        }
        with patch("src.services.scheduler_discord.create_scheduler_discord", return_value=Mock()), \
             patch("src.services.scheduler_services.load_document", return_value=config_doc), \
             patch("src.services.scheduler_services.save_document"), \
             patch("src.services.scheduler_job_handler._run_exchange_job",
                   return_value=(None, None, "crash")), \
             patch("time.time", side_effect=[0.0, 1.0]):
            execute_exchange_job("NYSE", "daily")

        # Lock should be released — acquiring should succeed
        assert _services.acquire_job_lock("daily_nyse") is True
        _services.release_job_lock("daily_nyse")


# ---------------------------------------------------------------------------
# Return value
# ---------------------------------------------------------------------------

class TestReturnValue:
    """Verifies the shape of the return value on the success path."""

    def test_returns_dict_with_price_stats(self):
        result, _, _, _ = _run_job()
        assert isinstance(result, dict)
        assert "price_stats" in result

    def test_returns_dict_with_alert_stats(self):
        result, _, _, _ = _run_job()
        assert "alert_stats" in result

    def test_returns_dict_with_duration(self):
        result, _, _, _ = _run_job(time_side_effect=[0.0, 5.0])
        assert "duration" in result
        assert result["duration"] == pytest.approx(5.0)

    def test_returns_none_on_error(self):
        result, _, _, _ = _run_job(worker_error="boom")
        assert result is None

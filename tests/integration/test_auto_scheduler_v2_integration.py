"""Integration tests for auto_scheduler_v2 (calendar-aware stock alert scheduler)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pandas as pd
import pytest

from src.services import auto_scheduler_v2
from src.services import scheduler_services


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_lock_file(tmp_path):
    """Provide a temporary lock file path for scheduler tests."""
    lock = tmp_path / "scheduler_v2.lock"
    # Patch LOCK_FILE in scheduler_services (where the logic lives)
    # and in auto_scheduler_v2 (the re-exported reference)
    with patch.object(scheduler_services, "LOCK_FILE", lock), \
         patch.object(auto_scheduler_v2, "LOCK_FILE", lock):
        yield lock


@pytest.fixture
def reset_job_locks():
    """Clear in-memory job locks before and after test."""
    auto_scheduler_v2._job_locks.clear()
    yield
    auto_scheduler_v2._job_locks.clear()


@pytest.fixture
def sample_metadata_df():
    """Minimal metadata DataFrame for exchange/ticker filtering."""
    return pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "exchange": ["NASDAQ", "NASDAQ", "NASDAQ"],
        "name": ["Apple", "Microsoft", "Alphabet"],
    })


@pytest.fixture
def sample_alerts():
    """Minimal list of alert dicts for run_alert_checks."""
    return [
        {
            "alert_id": "a1",
            "ticker": "AAPL",
            "exchange": "NASDAQ",
            "timeframe": "daily",
            "action": "on",
            "conditions": [["Close[-1] > 100"]],
            "combination_logic": "AND",
        },
        {
            "alert_id": "a2",
            "ticker": "MSFT",
            "exchange": "NASDAQ",
            "timeframe": "weekly",
            "action": "on",
            "conditions": [["Close[-1] > 200"]],
            "combination_logic": "AND",
        },
    ]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class TestLoadSchedulerConfig:
    """Tests for load_scheduler_config."""

    @patch("src.services.scheduler_services.load_document")
    def test_returns_default_when_document_store_empty(self, mock_load):
        """Should return default config when document store returns None or non-dict."""
        mock_load.return_value = None
        config = auto_scheduler_v2.load_scheduler_config()
        assert isinstance(config, dict)
        assert "scheduler_webhook" in config
        assert config["scheduler_webhook"]["enabled"] is False
        assert "notification_settings" in config
        assert config["notification_settings"]["send_start_notification"] is True
        assert "job_timeout_seconds" in config["notification_settings"]

    @patch("src.services.scheduler_services.load_document")
    def test_returns_persisted_config_when_available(self, mock_load):
        """Should return config from document store when present."""
        persisted = {
            "scheduler_webhook": {"url": "https://discord.com/wh", "enabled": True},
            "notification_settings": {"job_timeout_seconds": 600},
        }
        mock_load.return_value = persisted
        config = auto_scheduler_v2.load_scheduler_config()
        assert config["scheduler_webhook"]["enabled"] is True
        assert config["notification_settings"]["job_timeout_seconds"] == 600

    @patch("src.services.scheduler_services.load_document")
    def test_returns_default_on_exception(self, mock_load):
        """Should return default config when load_document raises."""
        mock_load.side_effect = RuntimeError("store unavailable")
        config = auto_scheduler_v2.load_scheduler_config()
        assert config["scheduler_webhook"]["enabled"] is False


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


class TestFormatStatsForMessage:
    """Tests for _format_stats_for_message."""

    def test_formats_price_and_alert_stats(self):
        """Should produce compact one-line summary (trig uses success or triggered)."""
        price_stats = {"updated": 100, "failed": 2, "skipped": 5}
        # Format uses alert_stats.get('success', alert_stats.get('triggered', 0)) for trig
        alert_stats = {"total": 50, "triggered": 3, "errors": 1}
        msg = auto_scheduler_v2._format_stats_for_message(price_stats, alert_stats)
        assert "upd 100" in msg
        assert "fail 2" in msg
        assert "skip 5" in msg
        assert "total 50" in msg
        assert "trig 3" in msg
        assert "err 1" in msg

    def test_handles_missing_keys(self):
        """Should handle partial dicts without KeyError."""
        msg = auto_scheduler_v2._format_stats_for_message({}, {})
        assert "upd 0" in msg
        assert "total 0" in msg

    def test_handles_non_dict_stats(self):
        """Should stringify non-dict stats."""
        msg = auto_scheduler_v2._format_stats_for_message("ok", "n/a")
        assert "ok" in msg
        assert "n/a" in msg


class TestFormatDuration:
    """Tests for _format_duration."""

    def test_seconds_only(self):
        """Should format seconds only."""
        assert auto_scheduler_v2._format_duration(45) == "45s"

    def test_minutes_and_seconds(self):
        """Should format minutes and seconds."""
        assert auto_scheduler_v2._format_duration(125) == "2m 5s"

    def test_hours_minutes_seconds(self):
        """Should format hours, minutes, seconds."""
        assert auto_scheduler_v2._format_duration(3665) == "1h 1m 5s"

    def test_negative_rounded_to_zero(self):
        """Should treat negative as zero."""
        assert auto_scheduler_v2._format_duration(-10) == "0s"


# ---------------------------------------------------------------------------
# Discord notifications
# ---------------------------------------------------------------------------


class TestSendSchedulerNotification:
    """Tests for send_scheduler_notification."""

    @patch("src.services.scheduler_services.requests.post")
    @patch("src.services.scheduler_services.load_document")
    def test_returns_false_when_webhook_disabled(self, mock_load, mock_post):
        """Should return False when webhook is disabled or missing."""
        mock_load.return_value = {
            "scheduler_webhook": {"enabled": False, "url": ""},
            "notification_settings": {},
        }
        assert auto_scheduler_v2.send_scheduler_notification("test") is False
        mock_post.assert_not_called()

    @patch("src.services.scheduler_services.requests.post")
    @patch("src.services.scheduler_services.load_document")
    def test_sends_post_when_webhook_enabled(self, mock_load, mock_post):
        """Should POST to webhook URL when enabled."""
        mock_load.return_value = {
            "scheduler_webhook": {
                "enabled": True,
                "url": "https://discord.com/api/webhooks/123",
                "name": "Scheduler",
            },
            "notification_settings": {},
        }
        mock_post.return_value = Mock(status_code=204)
        result = auto_scheduler_v2.send_scheduler_notification("Hello", event="info")
        assert result is True
        mock_post.assert_called_once()
        call_kw = mock_post.call_args[1]
        assert call_kw["json"]["content"] == "Hello"
        assert call_kw["json"]["username"] == "Scheduler"
        assert call_kw["timeout"] == 10

    @patch("src.services.scheduler_services.requests.post")
    @patch("src.services.scheduler_services.load_document")
    def test_returns_false_on_http_error(self, mock_load, mock_post):
        """Should return False when response is not 200/204."""
        mock_load.return_value = {
            "scheduler_webhook": {"enabled": True, "url": "https://discord.com/wh", "name": "S"},
            "notification_settings": {},
        }
        mock_post.return_value = Mock(status_code=500, text="Server Error")
        assert auto_scheduler_v2.send_scheduler_notification("test") is False


# ---------------------------------------------------------------------------
# Job locking
# ---------------------------------------------------------------------------


class TestJobLocking:
    """Tests for job lock acquire/release and sanitize_job_id."""

    def test_sanitize_job_id(self):
        """Should produce consistent lowercase job IDs."""
        assert auto_scheduler_v2.sanitize_job_id("daily", "NASDAQ") == "daily_nasdaq"
        assert auto_scheduler_v2.sanitize_job_id("Weekly", "New York") == "weekly_new_york"

    def test_acquire_release_cycle(self, reset_job_locks):
        """Should acquire then release and allow re-acquire."""
        job_id = "daily_nasdaq"
        assert auto_scheduler_v2.acquire_job_lock(job_id) is True
        assert auto_scheduler_v2.acquire_job_lock(job_id) is False
        auto_scheduler_v2.release_job_lock(job_id)
        assert auto_scheduler_v2.acquire_job_lock(job_id) is True
        auto_scheduler_v2.release_job_lock(job_id)

    def test_different_jobs_independent(self, reset_job_locks):
        """Different job IDs should not block each other."""
        assert auto_scheduler_v2.acquire_job_lock("daily_nasdaq") is True
        assert auto_scheduler_v2.acquire_job_lock("weekly_nyse") is True
        assert auto_scheduler_v2.acquire_job_lock("daily_nasdaq") is False
        auto_scheduler_v2.release_job_lock("daily_nasdaq")
        auto_scheduler_v2.release_job_lock("weekly_nyse")


# ---------------------------------------------------------------------------
# Status management
# ---------------------------------------------------------------------------


class TestStatusManagement:
    """Tests for update_scheduler_status and get_scheduler_info."""

    @patch("src.services.scheduler_services.save_document")
    @patch("src.services.scheduler_services.load_document")
    def test_update_scheduler_status_merges_fields(self, mock_load, mock_save):
        """Should merge status, heartbeat, and optional current_job/last_run/last_error."""
        mock_load.return_value = {}
        auto_scheduler_v2.update_scheduler_status(
            status="running",
            current_job={"id": "daily_nasdaq", "exchange": "NASDAQ", "job_type": "daily"},
        )
        assert mock_save.called
        saved = mock_save.call_args[0][1]
        assert saved["status"] == "running"
        assert "heartbeat" in saved
        assert saved["current_job"]["id"] == "daily_nasdaq"

    @patch("src.services.scheduler_services.save_document")
    @patch("src.services.scheduler_services.load_document")
    def test_update_scheduler_status_last_error(self, mock_load, mock_save):
        """Should persist last_error when provided."""
        mock_load.return_value = {}
        auto_scheduler_v2.update_scheduler_status(
            status="error",
            current_job=None,
            last_error={"job_id": "daily_nasdaq", "message": "timeout"},
        )
        saved = mock_save.call_args[0][1]
        assert saved["status"] == "error"
        assert saved["last_error"]["message"] == "timeout"

    @patch("src.services.auto_scheduler_v2.scheduler", None)
    @patch("src.services.scheduler_services.load_document")
    def test_get_scheduler_info_no_scheduler(self, mock_load):
        """Should return status dict without job counts when scheduler is None."""
        mock_load.return_value = {"status": "stopped", "heartbeat": "2024-01-01T00:00:00"}
        info = auto_scheduler_v2.get_scheduler_info()
        assert info is not None
        assert info["status"] == "stopped"
        assert "total_daily_jobs" not in info

    @patch("src.services.scheduler_services.load_document")
    def test_get_scheduler_info_with_jobs(self, mock_load):
        """Should include daily/weekly job counts and next_run when scheduler has jobs."""
        mock_load.return_value = {"status": "running"}
        mock_scheduler = MagicMock()
        job1 = Mock(id="daily_nasdaq", next_run_time=None)
        job2 = Mock(id="weekly_nyse", next_run_time=None)
        mock_scheduler.get_jobs.return_value = [job1, job2]
        with patch.object(auto_scheduler_v2, "scheduler", mock_scheduler):
            info = auto_scheduler_v2.get_scheduler_info()
        assert info is not None
        assert info.get("total_daily_jobs") == 1
        assert info.get("total_weekly_jobs") == 1


# ---------------------------------------------------------------------------
# run_alert_checks (integration with mocked repos)
# ---------------------------------------------------------------------------


class TestRunAlertChecks:
    """Integration tests for run_alert_checks with mocked repositories."""

    @patch("src.services.auto_scheduler_v2.StockAlertChecker")
    @patch("src.services.auto_scheduler_v2.list_alerts")
    @patch("src.services.auto_scheduler_v2.fetch_stock_metadata_df")
    def test_filters_alerts_by_exchange_and_timeframe_daily(
        self, mock_metadata, mock_list_alerts, mock_checker_class, sample_metadata_df, sample_alerts
    ):
        """Should filter alerts by exchange and daily timeframe and call checker."""
        mock_metadata.return_value = sample_metadata_df
        mock_list_alerts.return_value = sample_alerts
        mock_checker = Mock()
        mock_checker.check_alerts.return_value = {
            "total": 1,
            "success": 1,
            "triggered": 0,
            "errors": 0,
            "no_data": 0,
        }
        mock_checker_class.return_value = mock_checker

        stats = auto_scheduler_v2.run_alert_checks(["NASDAQ"], timeframe_key="daily")

        assert stats["total"] >= 1
        mock_checker.check_alerts.assert_called_once()
        call_args = mock_checker.check_alerts.call_args[0]
        assert call_args[1] == "daily"
        alerts_passed = call_args[0]
        assert all(a.get("timeframe", "").lower() in ("daily", "1d") for a in alerts_passed)

    @patch("src.services.auto_scheduler_v2.StockAlertChecker")
    @patch("src.services.auto_scheduler_v2.list_alerts")
    @patch("src.services.auto_scheduler_v2.fetch_stock_metadata_df")
    def test_skips_alerts_with_action_off(
        self, mock_metadata, mock_list_alerts, mock_checker_class, sample_metadata_df
    ):
        """Should not include alerts with action 'off' in relevant_alerts."""
        mock_metadata.return_value = sample_metadata_df
        mock_list_alerts.return_value = [
            {"alert_id": "a1", "ticker": "AAPL", "exchange": "NASDAQ", "timeframe": "daily", "action": "on"},
            {"alert_id": "a2", "ticker": "MSFT", "exchange": "NASDAQ", "timeframe": "daily", "action": "off"},
        ]
        mock_checker = Mock()
        mock_checker.check_alerts.return_value = {"total": 0, "success": 0, "triggered": 0, "errors": 0, "no_data": 0}
        mock_checker_class.return_value = mock_checker

        auto_scheduler_v2.run_alert_checks(["NASDAQ"], timeframe_key="daily")

        alerts_passed = mock_checker.check_alerts.call_args[0][0]
        assert len(alerts_passed) == 1
        assert alerts_passed[0]["alert_id"] == "a1"

    @patch("src.services.auto_scheduler_v2.fetch_stock_metadata_df")
    def test_returns_empty_stats_when_no_metadata(self, mock_metadata):
        """Should return zero stats when metadata is None or empty."""
        mock_metadata.return_value = None
        stats = auto_scheduler_v2.run_alert_checks(["NASDAQ"], timeframe_key="daily")
        assert stats["total"] == 0
        assert stats["success"] == 0
        assert stats["errors"] == 0

    @patch("src.services.auto_scheduler_v2.StockAlertChecker")
    @patch("src.services.auto_scheduler_v2.list_alerts")
    @patch("src.services.auto_scheduler_v2.fetch_stock_metadata_df")
    def test_returns_merged_check_stats(
        self, mock_metadata, mock_list_alerts, mock_checker_class, sample_metadata_df
    ):
        """Should merge checker stats (triggered, errors, no_data) into return value."""
        # Use two daily alerts so relevant_alerts has 2
        daily_alerts = [
            {"alert_id": "a1", "ticker": "AAPL", "exchange": "NASDAQ", "timeframe": "daily", "action": "on"},
            {"alert_id": "a2", "ticker": "MSFT", "exchange": "NASDAQ", "timeframe": "1d", "action": "on"},
        ]
        mock_metadata.return_value = sample_metadata_df
        mock_list_alerts.return_value = daily_alerts
        mock_checker = Mock()
        mock_checker.check_alerts.return_value = {
            "total": 2,
            "success": 2,
            "triggered": 1,
            "errors": 0,
            "no_data": 1,
        }
        mock_checker_class.return_value = mock_checker

        stats = auto_scheduler_v2.run_alert_checks(["NASDAQ"], timeframe_key="daily")

        assert stats["total"] == 2
        assert stats["triggered"] == 1
        assert stats["no_data"] == 1
        assert stats["success"] == 2

    @patch("src.services.auto_scheduler_v2.fetch_stock_metadata_df")
    def test_handles_exception_with_errors_increment(self, mock_metadata):
        """Should set stats['errors'] and return on exception."""
        mock_metadata.side_effect = RuntimeError("DB unavailable")
        stats = auto_scheduler_v2.run_alert_checks(["NASDAQ"], timeframe_key="daily")
        assert stats["errors"] == 1
        assert stats["total"] == 0


# ---------------------------------------------------------------------------
# execute_exchange_job (mocked worker and notifications)
# ---------------------------------------------------------------------------


class TestExecuteExchangeJob:
    """Integration tests for execute_exchange_job with mocked worker and Discord."""

    @patch("src.services.scheduler_discord.create_scheduler_discord")
    @patch("src.services.scheduler_services.save_document")
    @patch("src.services.scheduler_services.load_document")
    @patch("src.services.scheduler_job_handler._run_exchange_job")
    def test_successful_job_returns_stats_and_updates_status(
        self, mock_run_job, mock_load_doc, mock_save_doc, mock_discord, reset_job_locks
    ):
        """Should return price_stats and alert_stats and update status on success."""
        mock_load_doc.return_value = {
            "notification_settings": {
                "send_start_notification": False,
                "send_completion_notification": False,
                "job_timeout_seconds": 900,
            },
        }
        mock_discord.return_value = Mock()
        mock_run_job.return_value = (
            {"updated": 50, "failed": 0, "skipped": 2},
            {"total": 10, "triggered": 1, "errors": 0, "success": 10},
            None,
        )

        result = auto_scheduler_v2.execute_exchange_job("NASDAQ", "daily")

        assert result is not None
        assert result["price_stats"]["updated"] == 50
        assert result["alert_stats"]["triggered"] == 1
        assert "duration" in result
        mock_run_job.assert_called_once_with("NASDAQ", resample_weekly=False, timeout_seconds=900)

    @patch("src.services.scheduler_discord.create_scheduler_discord")
    @patch("src.services.scheduler_services.save_document")
    @patch("src.services.scheduler_services.load_document")
    @patch("src.services.scheduler_job_handler._run_exchange_job")
    def test_job_updates_status_on_success(
        self, mock_run_job, mock_load_doc, mock_save_doc, mock_discord, reset_job_locks
    ):
        """Should call update_status with current_job then last_run/last_result."""
        mock_load_doc.return_value = {
            "notification_settings": {
                "send_start_notification": False,
                "send_completion_notification": False,
                "job_timeout_seconds": 900,
            },
        }
        mock_discord.return_value = Mock()
        mock_run_job.return_value = (
            {"updated": 0},
            {"total": 0},
            None,
        )

        auto_scheduler_v2.execute_exchange_job("NYSE", "weekly")

        # save_document is called each time update_status runs; verify at least 2 calls
        assert mock_save_doc.call_count >= 2
        # Check that last_run was saved with the correct job_id
        saved_calls = [c[0][1] for c in mock_save_doc.call_args_list if isinstance(c[0][1], dict)]
        last_run_calls = [s for s in saved_calls if s.get("last_run")]
        assert len(last_run_calls) >= 1
        assert last_run_calls[0]["last_run"]["job_id"] == "weekly_nyse"

    @patch("src.services.scheduler_discord.create_scheduler_discord")
    @patch("src.services.scheduler_services.save_document")
    @patch("src.services.scheduler_services.load_document")
    @patch("src.services.scheduler_job_handler._run_exchange_job")
    def test_job_handles_worker_error_and_updates_last_error(
        self, mock_run_job, mock_load_doc, mock_save_doc, mock_discord, reset_job_locks
    ):
        """Should update last_error and send failure notification when worker fails."""
        mock_load_doc.return_value = {
            "notification_settings": {
                "send_start_notification": False,
                "send_completion_notification": False,
                "job_timeout_seconds": 900,
            },
        }
        mock_notifier = Mock()
        mock_discord.return_value = mock_notifier
        mock_run_job.return_value = (None, None, "timeout after 900s")

        result = auto_scheduler_v2.execute_exchange_job("NASDAQ", "daily")

        assert result is None
        # Should have saved last_error
        saved_calls = [c[0][1] for c in mock_save_doc.call_args_list if isinstance(c[0][1], dict)]
        error_calls = [s for s in saved_calls if s.get("last_error")]
        assert len(error_calls) >= 1
        err_msg = error_calls[0]["last_error"]["message"]
        assert "timeout" in err_msg.lower() or "Timeout" in err_msg
        mock_notifier.notify_error.assert_called()

    def test_duplicate_job_returns_none_when_lock_held(self, reset_job_locks):
        """Should return None when job lock is already held."""
        auto_scheduler_v2.acquire_job_lock("daily_nasdaq")
        try:
            with patch("src.services.scheduler_job_handler._run_exchange_job") as mock_run:
                result = auto_scheduler_v2.execute_exchange_job("NASDAQ", "daily")
            assert result is None
            mock_run.assert_not_called()
        finally:
            auto_scheduler_v2.release_job_lock("daily_nasdaq")


# ---------------------------------------------------------------------------
# Scheduler setup (_schedule_exchange_jobs)
# ---------------------------------------------------------------------------


class TestScheduleExchangeJobs:
    """Tests for _schedule_exchange_jobs and job registration."""

    @patch("src.services.auto_scheduler_v2.get_exchanges_by_closing_time")
    def test_schedules_daily_and_weekly_per_exchange(self, mock_get_times):
        """Should add daily and weekly job per exchange from get_exchanges_by_closing_time."""
        mock_get_times.return_value = {
            "16:40": [
                {"exchange": "NASDAQ", "name": "NASDAQ Stock Market"},
            ],
            "17:40": [
                {"exchange": "NYSE", "name": "New York Stock Exchange"},
            ],
        }
        sched = MagicMock()

        with patch.object(auto_scheduler_v2, "scheduler", sched):
            with patch("src.services.auto_scheduler_v2.EXCHANGE_SCHEDULES", {"NASDAQ": {}, "NYSE": {}}):
                with patch("src.services.auto_scheduler_v2.get_market_days_for_exchange") as mock_days:
                    mock_days.return_value = {"daily": "mon-fri", "weekly": "fri"}
                    auto_scheduler_v2._schedule_exchange_jobs()

        # 2 exchanges × 2 (daily + weekly) = 4 jobs
        assert sched.add_job.call_count == 4
        job_ids = [c[1]["id"] for c in sched.add_job.call_args_list]
        assert "daily_nasdaq" in job_ids
        assert "weekly_nasdaq" in job_ids
        assert "daily_nyse" in job_ids
        assert "weekly_nyse" in job_ids

    @patch("src.services.auto_scheduler_v2.get_exchanges_by_closing_time")
    def test_uses_cron_trigger_with_utc(self, mock_get_times):
        """Should add jobs with CronTrigger and timezone UTC."""
        mock_get_times.return_value = {
            "16:40": [{"exchange": "NASDAQ", "name": "NASDAQ"}],
        }
        sched = MagicMock()
        with patch.object(auto_scheduler_v2, "scheduler", sched):
            with patch("src.services.auto_scheduler_v2.EXCHANGE_SCHEDULES", {"NASDAQ": {}}):
                with patch("src.services.auto_scheduler_v2.get_market_days_for_exchange") as mock_days:
                    mock_days.return_value = {"daily": "mon-fri", "weekly": "fri"}
                    auto_scheduler_v2._schedule_exchange_jobs()

        first_call = sched.add_job.call_args_list[0]
        trigger = first_call[0][1]  # second positional is CronTrigger
        # APScheduler may use datetime.timezone.utc (no .key); check UTC is used
        assert "UTC" in str(trigger.timezone) or getattr(trigger.timezone, "key", None) == "UTC"


# ---------------------------------------------------------------------------
# Process lock and start/stop lifecycle
# ---------------------------------------------------------------------------


class TestProcessLock:
    """Tests for _acquire_process_lock and _release_process_lock."""

    def test_acquire_creates_lock_file(self, tmp_lock_file):
        """Should create lock file with pid and type when no existing lock."""
        assert not tmp_lock_file.exists()
        result = auto_scheduler_v2._acquire_process_lock()
        assert result is True
        assert tmp_lock_file.exists()
        data = json.loads(tmp_lock_file.read_text())
        assert data["type"] == "auto_scheduler_v2"
        assert "pid" in data
        assert "heartbeat" in data

    def test_acquire_fails_when_other_process_holds_lock(self, tmp_lock_file):
        """Should return False when lock exists and PID matches our process type."""
        tmp_lock_file.write_text(json.dumps({"pid": 99999, "type": "auto_scheduler_v2"}))
        with patch.object(scheduler_services.SchedulerServices, "process_matches", return_value=True):
            result = auto_scheduler_v2._acquire_process_lock()
        assert result is False

    def test_release_removes_lock_file(self, tmp_lock_file):
        """Should remove lock file on release."""
        tmp_lock_file.write_text("{}")
        auto_scheduler_v2._release_process_lock()
        assert not tmp_lock_file.exists()


class TestSchedulerLifecycle:
    """Integration tests for start_auto_scheduler and stop_auto_scheduler."""

    @patch("src.services.auto_scheduler_v2.send_scheduler_notification")
    @patch("src.services.auto_scheduler_v2.update_scheduler_status")
    @patch("src.services.auto_scheduler_v2.is_scheduler_running")
    @patch("src.services.auto_scheduler_v2._acquire_process_lock")
    @patch("src.services.auto_scheduler_v2.os.getppid")
    @patch("src.services.auto_scheduler_v2.os.getpid")
    def test_start_schedules_jobs_and_heartbeat(
        self, mock_getpid, mock_getppid, mock_acquire, mock_running, mock_status, mock_notify
    ):
        """Should create scheduler, schedule exchange jobs, heartbeat, and full daily update."""
        # Same pid/ppid so start runs in-process instead of spawning subprocess
        mock_getpid.return_value = 12345
        mock_getppid.return_value = 12345
        mock_running.return_value = False
        mock_acquire.return_value = True
        mock_scheduler = MagicMock()

        with patch("src.services.auto_scheduler_v2.BackgroundScheduler") as mock_sched_class:
            mock_sched_class.return_value = mock_scheduler
            with patch("src.services.auto_scheduler_v2._schedule_exchange_jobs"):
                result = auto_scheduler_v2.start_auto_scheduler()

        assert result is True
        mock_scheduler.start.assert_called_once()
        # heartbeat + daily_full_update + exchange jobs
        assert mock_scheduler.add_job.call_count >= 2
        mock_status.assert_called()
        mock_notify.assert_called()
        # Reset global so other tests don't see it
        auto_scheduler_v2.scheduler = None

    @patch("src.services.auto_scheduler_v2.is_scheduler_running")
    def test_start_returns_true_when_already_running(self, mock_running):
        """Should return True without starting again when is_scheduler_running."""
        mock_running.return_value = True
        result = auto_scheduler_v2.start_auto_scheduler()
        assert result is True

    @patch("src.services.auto_scheduler_v2.send_scheduler_notification")
    @patch("src.services.auto_scheduler_v2.update_scheduler_status")
    @patch("src.services.auto_scheduler_v2._release_process_lock")
    def test_stop_shuts_down_scheduler_and_releases_lock(
        self, mock_release, mock_status, mock_notify
    ):
        """Should shutdown scheduler and release process lock."""
        mock_scheduler = MagicMock()
        mock_scheduler.running = True

        with patch.object(auto_scheduler_v2, "scheduler", mock_scheduler):
            result = auto_scheduler_v2.stop_auto_scheduler()

        assert result is True
        mock_scheduler.shutdown.assert_called_once_with(wait=True)
        mock_release.assert_called_once()
        mock_status.assert_called()


# ---------------------------------------------------------------------------
# run_daily_job / run_weekly_job wrappers
# ---------------------------------------------------------------------------


class TestRunDailyWeeklyWrappers:
    """Tests for run_daily_job and run_weekly_job."""

    @patch("src.services.auto_scheduler_v2.execute_exchange_job")
    def test_run_daily_job_calls_execute_with_daily(self, mock_execute):
        """run_daily_job should call execute_exchange_job(exchange, 'daily')."""
        mock_execute.return_value = {"price_stats": {}, "alert_stats": {}}
        auto_scheduler_v2.run_daily_job("NASDAQ")
        mock_execute.assert_called_once_with("NASDAQ", "daily")

    @patch("src.services.auto_scheduler_v2.execute_exchange_job")
    def test_run_weekly_job_calls_execute_with_weekly(self, mock_execute):
        """run_weekly_job should call execute_exchange_job(exchange, 'weekly')."""
        mock_execute.return_value = {"price_stats": {}, "alert_stats": {}}
        auto_scheduler_v2.run_weekly_job("NYSE")
        mock_execute.assert_called_once_with("NYSE", "weekly")

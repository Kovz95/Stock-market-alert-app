"""Unit tests for futures_scheduler module."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import pytest
from freezegun import freeze_time

# Import the module we're testing
from src.services import futures_scheduler


class TestConfiguration:
    """Tests for configuration loading and saving."""

    @patch("src.services.futures_scheduler.load_document")
    def test_load_scheduler_config_returns_defaults_when_no_config(self, mock_load):
        """Should return default config when no config exists."""
        mock_load.return_value = {}

        config = futures_scheduler.load_scheduler_config()

        assert config["enabled"] is True
        assert config["update_on_start"] is True
        assert "update_times" in config
        assert "ib_hours" in config

    @patch("src.services.futures_scheduler.load_document")
    def test_load_scheduler_config_merges_with_defaults(self, mock_load):
        """Should merge loaded config with defaults."""
        mock_load.return_value = {
            "enabled": False,
            "custom_setting": "test",
        }

        config = futures_scheduler.load_scheduler_config()

        assert config["enabled"] is False
        assert config["custom_setting"] == "test"
        assert "update_times" in config  # From defaults
        assert "ib_hours" in config  # From defaults

    @patch("src.services.futures_scheduler.load_document")
    def test_load_scheduler_config_handles_invalid_type(self, mock_load):
        """Should return defaults when loaded config is not a dict."""
        mock_load.return_value = "invalid"

        config = futures_scheduler.load_scheduler_config()

        assert isinstance(config, dict)
        assert config["enabled"] is True

    @patch("src.services.futures_scheduler.load_document")
    def test_load_scheduler_config_handles_exception(self, mock_load):
        """Should return defaults when loading raises exception."""
        mock_load.side_effect = Exception("Load failed")

        config = futures_scheduler.load_scheduler_config()

        assert isinstance(config, dict)
        assert config["enabled"] is True

    @patch("src.services.futures_scheduler.save_document")
    def test_save_scheduler_config_saves_to_document_store(self, mock_save):
        """Should save config to document store."""
        mock_save.return_value = None

        test_config = {"enabled": True, "test": "value"}
        result = futures_scheduler.save_scheduler_config(test_config)

        assert result is True
        mock_save.assert_called_once()
        assert mock_save.call_args[0][1] == test_config

    @patch("src.services.futures_scheduler.save_document")
    def test_save_scheduler_config_handles_error(self, mock_save):
        """Should return False when save fails."""
        mock_save.side_effect = Exception("Save failed")

        result = futures_scheduler.save_scheduler_config({"test": "data"})

        assert result is False


class TestIBHoursCheck:
    """Tests for IB trading hours checking."""

    @patch("src.services.futures_scheduler.load_scheduler_config")
    @freeze_time("2024-01-15 10:00:00")
    def test_is_ib_available_within_hours(self, mock_config):
        """Should return True when within IB hours."""
        mock_config.return_value = {
            "ib_hours": {"start": "05:00", "end": "23:00"}
        }

        result = futures_scheduler.is_ib_available()

        assert result is True

    @patch("src.services.futures_scheduler.load_scheduler_config")
    @freeze_time("2024-01-15 02:00:00")
    def test_is_ib_available_outside_hours(self, mock_config):
        """Should return False when outside IB hours."""
        mock_config.return_value = {
            "ib_hours": {"start": "05:00", "end": "23:00"}
        }

        result = futures_scheduler.is_ib_available()

        assert result is False

    @patch("src.services.futures_scheduler.load_scheduler_config")
    @freeze_time("2024-01-15 05:00:00")
    def test_is_ib_available_at_start_boundary(self, mock_config):
        """Should return True at start boundary."""
        mock_config.return_value = {
            "ib_hours": {"start": "05:00", "end": "23:00"}
        }

        result = futures_scheduler.is_ib_available()

        assert result is True

    @patch("src.services.futures_scheduler.load_scheduler_config")
    @freeze_time("2024-01-15 23:00:00")
    def test_is_ib_available_at_end_boundary(self, mock_config):
        """Should return True at end boundary."""
        mock_config.return_value = {
            "ib_hours": {"start": "05:00", "end": "23:00"}
        }

        result = futures_scheduler.is_ib_available()

        assert result is True

    @patch("src.services.futures_scheduler.load_scheduler_config")
    def test_is_ib_available_handles_exception(self, mock_config):
        """Should return True (fail open) when check fails."""
        mock_config.side_effect = Exception("Config error")

        result = futures_scheduler.is_ib_available()

        assert result is True


class TestJobLocking:
    """Tests for job locking mechanism."""

    def setup_method(self):
        """Reset job locks before each test."""
        futures_scheduler._job_locks.clear()

    def test_acquire_job_lock_succeeds_first_time(self):
        """Should acquire lock successfully first time."""
        result = futures_scheduler.acquire_job_lock("test_job")

        assert result is True
        assert futures_scheduler._job_locks["test_job"] is True

    def test_acquire_job_lock_fails_when_locked(self):
        """Should fail to acquire already locked job."""
        futures_scheduler.acquire_job_lock("test_job")

        result = futures_scheduler.acquire_job_lock("test_job")

        assert result is False

    def test_release_job_lock_removes_lock(self):
        """Should remove lock when released."""
        futures_scheduler.acquire_job_lock("test_job")

        futures_scheduler.release_job_lock("test_job")

        assert "test_job" not in futures_scheduler._job_locks

    def test_release_job_lock_handles_non_existent_lock(self):
        """Should handle releasing non-existent lock gracefully."""
        # Should not raise exception
        futures_scheduler.release_job_lock("non_existent")

        assert "non_existent" not in futures_scheduler._job_locks

    def test_multiple_different_locks_can_coexist(self):
        """Should allow multiple different job locks."""
        result1 = futures_scheduler.acquire_job_lock("job_1")
        result2 = futures_scheduler.acquire_job_lock("job_2")

        assert result1 is True
        assert result2 is True
        assert len(futures_scheduler._job_locks) == 2


class TestStatusManagement:
    """Tests for status tracking and updates."""

    @patch("src.services.futures_scheduler.save_document")
    @patch("src.services.futures_scheduler.load_document")
    @patch("src.services.futures_scheduler.LOCK_FILE")
    def test_update_scheduler_status_creates_status(
        self, mock_lock_file, mock_load, mock_save
    ):
        """Should create status with all required fields."""
        mock_load.return_value = {}
        mock_lock_file.exists.return_value = False

        futures_scheduler.update_scheduler_status(status="running")

        mock_save.assert_called_once()
        status_data = mock_save.call_args[0][1]
        assert status_data["status"] == "running"
        assert "heartbeat" in status_data
        assert "pid" in status_data
        assert status_data["type"] == "futures_scheduler"

    @patch("src.services.futures_scheduler.save_document")
    @patch("src.services.futures_scheduler.load_document")
    @patch("src.services.futures_scheduler.LOCK_FILE")
    def test_update_scheduler_status_updates_current_job(
        self, mock_lock_file, mock_load, mock_save
    ):
        """Should update current job information."""
        mock_load.return_value = {}
        mock_lock_file.exists.return_value = False

        job_info = {
            "id": "test_job",
            "started": "2024-01-15T10:00:00",
        }
        futures_scheduler.update_scheduler_status(
            status="running", current_job=job_info
        )

        status_data = mock_save.call_args[0][1]
        assert status_data["current_job"] == job_info

    @patch("src.services.futures_scheduler.save_document")
    @patch("src.services.futures_scheduler.load_document")
    @patch("src.services.futures_scheduler.LOCK_FILE")
    def test_update_scheduler_status_records_last_run(
        self, mock_lock_file, mock_load, mock_save
    ):
        """Should record last run information."""
        mock_load.return_value = {}
        mock_lock_file.exists.return_value = False

        last_run = {
            "job_id": "test_job",
            "completed_at": "2024-01-15T10:05:00",
            "duration_seconds": 45.2,
        }
        futures_scheduler.update_scheduler_status(status="running", last_run=last_run)

        status_data = mock_save.call_args[0][1]
        assert status_data["last_run"] == last_run

    @patch("src.services.futures_scheduler.save_document")
    @patch("src.services.futures_scheduler.load_document")
    @patch("src.services.futures_scheduler.LOCK_FILE")
    def test_update_scheduler_status_updates_lock_file_heartbeat(
        self, mock_lock_file, mock_load, mock_save
    ):
        """Should update heartbeat in lock file if it exists."""
        mock_load.return_value = {}
        mock_lock_file.exists.return_value = True
        mock_lock_file.read_text.return_value = '{"pid": 12345}'

        futures_scheduler.update_scheduler_status(status="running")

        mock_lock_file.write_text.assert_called_once()
        written_data = json.loads(mock_lock_file.write_text.call_args[0][0])
        assert "heartbeat" in written_data

    @patch("src.services.futures_scheduler.load_document")
    def test_get_scheduler_info_returns_status(self, mock_load):
        """Should return scheduler status information."""
        expected_status = {
            "status": "running",
            "heartbeat": "2024-01-15T10:00:00",
            "pid": 12345,
        }
        mock_load.return_value = expected_status

        result = futures_scheduler.get_scheduler_info()

        assert result["status"] == "running"
        assert result["pid"] == 12345

    @patch("src.services.futures_scheduler.load_document")
    def test_get_scheduler_info_returns_none_for_invalid_status(self, mock_load):
        """Should return None when status is not a dict."""
        mock_load.return_value = "invalid"

        result = futures_scheduler.get_scheduler_info()

        assert result is None

    @patch("src.services.futures_scheduler.load_document")
    def test_get_scheduler_info_handles_exception(self, mock_load):
        """Should return None when loading fails."""
        mock_load.side_effect = Exception("Load failed")

        result = futures_scheduler.get_scheduler_info()

        assert result is None


class TestProcessManagement:
    """Tests for process locking and management."""

    @patch("src.services.futures_scheduler.psutil.Process")
    def test_process_matches_returns_true_for_matching_process(self, mock_process):
        """Should return True when process matches scheduler."""
        mock_proc = Mock()
        mock_proc.cmdline.return_value = ["python", "src/services/futures_scheduler.py"]
        mock_process.return_value = mock_proc

        result = futures_scheduler._process_matches(12345)

        assert result is True

    @patch("src.services.futures_scheduler.psutil.Process")
    def test_process_matches_returns_false_for_different_process(self, mock_process):
        """Should return False when process doesn't match scheduler."""
        mock_proc = Mock()
        mock_proc.cmdline.return_value = ["python", "other_script.py"]
        mock_process.return_value = mock_proc

        result = futures_scheduler._process_matches(12345)

        assert result is False

    @patch("src.services.futures_scheduler.psutil.Process")
    def test_process_matches_handles_no_such_process(self, mock_process):
        """Should return False when process doesn't exist."""
        mock_process.side_effect = futures_scheduler.psutil.NoSuchProcess(12345)

        result = futures_scheduler._process_matches(12345)

        assert result is False

    @patch("src.services.futures_scheduler._process_matches")
    @patch("src.services.futures_scheduler.LOCK_FILE")
    def test_acquire_process_lock_succeeds_when_no_lock(
        self, mock_lock_file, mock_matches
    ):
        """Should acquire lock when no existing lock file."""
        mock_lock_file.exists.return_value = False

        result = futures_scheduler._acquire_process_lock()

        assert result is True
        mock_lock_file.write_text.assert_called_once()

    @patch("src.services.futures_scheduler._process_matches")
    @patch("src.services.futures_scheduler.LOCK_FILE")
    def test_acquire_process_lock_fails_when_process_running(
        self, mock_lock_file, mock_matches
    ):
        """Should fail when another scheduler is running."""
        mock_lock_file.exists.return_value = True
        mock_lock_file.read_text.return_value = '{"pid": 12345}'
        mock_matches.return_value = True

        result = futures_scheduler._acquire_process_lock()

        assert result is False

    @patch("src.services.futures_scheduler._process_matches")
    @patch("src.services.futures_scheduler.LOCK_FILE")
    def test_acquire_process_lock_removes_stale_lock(
        self, mock_lock_file, mock_matches
    ):
        """Should remove stale lock and acquire new one."""
        mock_lock_file.exists.return_value = True
        mock_lock_file.read_text.return_value = '{"pid": 99999}'
        mock_matches.return_value = False

        result = futures_scheduler._acquire_process_lock()

        assert result is True
        mock_lock_file.unlink.assert_called_once()

    @patch("src.services.futures_scheduler.LOCK_FILE")
    def test_release_process_lock_removes_file(self, mock_lock_file):
        """Should remove lock file when releasing."""
        futures_scheduler._release_process_lock()

        mock_lock_file.unlink.assert_called_once()

    @patch("src.services.futures_scheduler.psutil.process_iter")
    def test_is_scheduler_running_returns_true_when_running(self, mock_iter):
        """Should return True when scheduler process is found."""
        mock_proc = {"cmdline": ["python", "src/services/futures_scheduler.py"]}
        mock_iter.return_value = [MagicMock(info=mock_proc)]

        result = futures_scheduler.is_scheduler_running()

        assert result is True

    @patch("src.services.futures_scheduler.psutil.process_iter")
    def test_is_scheduler_running_returns_false_when_not_running(self, mock_iter):
        """Should return False when no scheduler process found."""
        mock_proc = {"cmdline": ["python", "other_script.py"]}
        mock_iter.return_value = [MagicMock(info=mock_proc)]

        result = futures_scheduler.is_scheduler_running()

        assert result is False


class TestPriceUpdate:
    """Tests for futures price update execution."""

    @patch("src.services.futures_scheduler.is_ib_available")
    def test_run_price_update_skips_when_outside_ib_hours(self, mock_ib):
        """Should skip update when outside IB hours."""
        mock_ib.return_value = False

        result = futures_scheduler.run_price_update()

        assert result["error"] == "Outside IB hours"
        assert result["updated"] == 0

    @patch("src.services.futures_scheduler.subprocess.run")
    @patch("src.services.futures_scheduler.is_ib_available")
    def test_run_price_update_executes_subprocess(self, mock_ib, mock_run):
        """Should execute price updater subprocess."""
        mock_ib.return_value = True
        mock_run.return_value = Mock(
            returncode=0, stdout="Update complete: 60 succeeded, 2 failed", stderr=""
        )

        result = futures_scheduler.run_price_update()

        mock_run.assert_called_once()
        assert "src" in str(mock_run.call_args) and "services" in str(mock_run.call_args) and "futures_price_updater.py" in str(mock_run.call_args)

    @patch("src.services.futures_scheduler.subprocess.run")
    @patch("src.services.futures_scheduler.is_ib_available")
    def test_run_price_update_parses_success_counts(self, mock_ib, mock_run):
        """Should parse update counts from output."""
        mock_ib.return_value = True
        mock_run.return_value = Mock(
            returncode=0, stdout="Update complete: 60 succeeded, 2 failed", stderr=""
        )

        result = futures_scheduler.run_price_update()

        assert result["updated"] == 60
        assert result["failed"] == 2
        assert result["error"] is None

    @patch("src.services.futures_scheduler.subprocess.run")
    @patch("src.services.futures_scheduler.is_ib_available")
    def test_run_price_update_handles_subprocess_error(self, mock_ib, mock_run):
        """Should handle subprocess execution errors."""
        mock_ib.return_value = True
        mock_run.return_value = Mock(returncode=1, stdout="", stderr="Update failed")

        result = futures_scheduler.run_price_update()

        assert result["error"] == "Update failed"
        assert result["updated"] == 0

    @patch("src.services.futures_scheduler.subprocess.run")
    @patch("src.services.futures_scheduler.is_ib_available")
    def test_run_price_update_handles_timeout(self, mock_ib, mock_run):
        """Should handle subprocess timeout."""
        mock_ib.return_value = True
        mock_run.side_effect = subprocess.TimeoutExpired("cmd", 600)

        result = futures_scheduler.run_price_update()

        assert "Timeout" in result["error"]


class TestAlertChecking:
    """Tests for futures alert checking."""

    @patch("src.services.futures_scheduler.FuturesAlertChecker")
    def test_run_alert_checks_executes_checker(self, mock_checker_class):
        """Should execute alert checker."""
        mock_checker = Mock()
        mock_checker.check_all_alerts.return_value = {
            "total": 10,
            "triggered": 2,
            "errors": 0,
        }
        mock_checker_class.return_value = mock_checker

        result = futures_scheduler.run_alert_checks()

        mock_checker.check_all_alerts.assert_called_once()
        assert result["total"] == 10
        assert result["triggered"] == 2

    @patch("src.services.futures_scheduler.FuturesAlertChecker")
    def test_run_alert_checks_handles_exception(self, mock_checker_class):
        """Should handle checker exceptions."""
        mock_checker = Mock()
        mock_checker.check_all_alerts.side_effect = Exception("Check failed")
        mock_checker_class.return_value = mock_checker

        result = futures_scheduler.run_alert_checks()

        assert result["errors"] == 1


class TestDiscordNotifications:
    """Tests for Discord notification sending."""

    @patch("src.services.futures_scheduler.requests.post")
    @patch("src.services.futures_scheduler.load_scheduler_config")
    def test_send_scheduler_notification_sends_when_enabled(self, mock_config, mock_post):
        """Should send notification when webhook is configured and enabled."""
        mock_config.return_value = {
            "scheduler_webhook": {
                "url": "https://discord.com/webhook",
                "enabled": True,
                "name": "Test Scheduler",
            }
        }
        mock_post.return_value = Mock(status_code=200)

        result = futures_scheduler.send_scheduler_notification("Test message")

        assert result is True
        mock_post.assert_called_once()

    @patch("src.services.futures_scheduler.load_scheduler_config")
    def test_send_scheduler_notification_skips_when_disabled(self, mock_config):
        """Should not send when webhook is disabled."""
        mock_config.return_value = {
            "scheduler_webhook": {
                "url": "https://discord.com/webhook",
                "enabled": False,
            }
        }

        result = futures_scheduler.send_scheduler_notification("Test message")

        assert result is False

    @patch("src.services.futures_scheduler.load_scheduler_config")
    def test_send_scheduler_notification_skips_when_no_url(self, mock_config):
        """Should not send when no webhook URL configured."""
        mock_config.return_value = {
            "scheduler_webhook": {
                "url": "",
                "enabled": True,
            }
        }

        result = futures_scheduler.send_scheduler_notification("Test message")

        assert result is False

    @patch("src.services.futures_scheduler.requests.post")
    @patch("src.services.futures_scheduler.load_scheduler_config")
    def test_send_scheduler_notification_handles_error_status(
        self, mock_config, mock_post
    ):
        """Should return False on non-200 status."""
        mock_config.return_value = {
            "scheduler_webhook": {
                "url": "https://discord.com/webhook",
                "enabled": True,
            }
        }
        mock_post.return_value = Mock(status_code=400)

        result = futures_scheduler.send_scheduler_notification("Test message")

        assert result is False

    @patch("src.services.futures_scheduler.requests.post")
    @patch("src.services.futures_scheduler.load_scheduler_config")
    def test_send_scheduler_notification_handles_exception(self, mock_config, mock_post):
        """Should handle request exceptions."""
        mock_config.return_value = {
            "scheduler_webhook": {
                "url": "https://discord.com/webhook",
                "enabled": True,
            }
        }
        mock_post.side_effect = Exception("Network error")

        result = futures_scheduler.send_scheduler_notification("Test message")

        assert result is False


class TestMessageFormatting:
    """Tests for message formatting helpers."""

    def test_format_stats_for_message_with_dicts(self):
        """Should format stats dictionaries into message."""
        price_stats = {"updated": 60, "failed": 2, "skipped": 0}
        alert_stats = {"total": 10, "success": 2, "errors": 0}

        result = futures_scheduler._format_stats_for_message(price_stats, alert_stats)

        assert "upd 60" in result
        assert "fail 2" in result
        assert "total 10" in result
        assert "trig 2" in result

    def test_format_stats_for_message_with_non_dicts(self):
        """Should handle non-dict stats gracefully."""
        result = futures_scheduler._format_stats_for_message("price", "alert")

        assert "price" in result
        assert "alert" in result

    def test_format_duration_handles_seconds(self):
        """Should format seconds-only duration."""
        result = futures_scheduler._format_duration(45)

        assert "45s" in result

    def test_format_duration_handles_minutes_and_seconds(self):
        """Should format minutes and seconds."""
        result = futures_scheduler._format_duration(125)  # 2m 5s

        assert "2m" in result
        assert "5s" in result

    def test_format_duration_handles_hours_minutes_seconds(self):
        """Should format hours, minutes, and seconds."""
        result = futures_scheduler._format_duration(3665)  # 1h 1m 5s

        assert "1h" in result
        assert "1m" in result
        assert "5s" in result


class TestJobExecution:
    """Tests for job execution logic."""

    def setup_method(self):
        """Reset job locks before each test."""
        futures_scheduler._job_locks.clear()

    @patch("src.services.futures_scheduler.send_scheduler_notification")
    @patch("src.services.futures_scheduler.update_scheduler_status")
    @patch("src.services.futures_scheduler.is_ib_available")
    @patch("src.services.futures_scheduler.load_scheduler_config")
    def test_execute_futures_job_skips_when_already_running(
        self, mock_config, mock_ib, mock_status, mock_notify
    ):
        """Should skip job when already running."""
        mock_config.return_value = {"notification_settings": {}}
        futures_scheduler.acquire_job_lock("futures_combined")

        result = futures_scheduler.execute_futures_job()

        assert result is None

    @patch("src.services.futures_scheduler.send_scheduler_notification")
    @patch("src.services.futures_scheduler.update_scheduler_status")
    @patch("src.services.futures_scheduler.is_ib_available")
    @patch("src.services.futures_scheduler.load_scheduler_config")
    def test_execute_futures_job_skips_outside_ib_hours(
        self, mock_config, mock_ib, mock_status, mock_notify
    ):
        """Should skip job when outside IB hours."""
        mock_config.return_value = {"notification_settings": {}}
        mock_ib.return_value = False

        futures_scheduler.execute_futures_job()

        # Should update status with skip reason
        assert any(
            call_args[1].get("last_run", {}).get("skipped", False)
            for call_args in mock_status.call_args_list
        )

    @patch("src.services.futures_scheduler._run_job_subprocess")
    @patch("src.services.futures_scheduler.send_scheduler_notification")
    @patch("src.services.futures_scheduler.update_scheduler_status")
    @patch("src.services.futures_scheduler.is_ib_available")
    @patch("src.services.futures_scheduler.load_scheduler_config")
    def test_execute_futures_job_runs_successfully(
        self, mock_config, mock_ib, mock_status, mock_notify, mock_subprocess
    ):
        """Should execute job successfully."""
        mock_config.return_value = {
            "notification_settings": {
                "send_start_notification": True,
                "send_completion_notification": True,
                "job_timeout_seconds": 900,
            }
        }
        mock_ib.return_value = True
        mock_subprocess.return_value = (
            {"updated": 60, "failed": 2},
            {"total": 10, "triggered": 2},
            None,
        )

        result = futures_scheduler.execute_futures_job()

        assert result is not None
        assert result["price_stats"]["updated"] == 60
        assert result["alert_stats"]["total"] == 10


class TestSchedulerLifecycle:
    """Tests for scheduler start/stop lifecycle."""

    @patch("src.services.futures_scheduler.is_scheduler_running")
    def test_start_futures_scheduler_returns_true_when_already_running(self, mock_running):
        """Should return True when scheduler is already running."""
        mock_running.return_value = True

        result = futures_scheduler.start_futures_scheduler()

        assert result is True

    @patch("src.services.futures_scheduler.os.getppid")
    @patch("src.services.futures_scheduler.os.getpid")
    @patch("src.services.futures_scheduler.BackgroundScheduler")
    @patch("src.services.futures_scheduler.send_scheduler_notification")
    @patch("src.services.futures_scheduler.update_scheduler_status")
    @patch("src.services.futures_scheduler._acquire_process_lock")
    @patch("src.services.futures_scheduler.is_scheduler_running")
    @patch("src.services.futures_scheduler.load_scheduler_config")
    def test_start_futures_scheduler_initializes_scheduler(
        self, mock_config, mock_running, mock_lock, mock_status, mock_notify, mock_scheduler_class, mock_getpid, mock_getppid
    ):
        """Should initialize and start scheduler."""
        # Make pids match so it doesn't spawn subprocess
        mock_getpid.return_value = 1000
        mock_getppid.return_value = 1000

        mock_running.return_value = False
        mock_lock.return_value = True
        mock_scheduler = Mock()
        mock_scheduler_class.return_value = mock_scheduler
        mock_config.return_value = {
            "update_times": ["06:00"],
            "update_on_start": False,
            "notification_settings": {},
        }

        result = futures_scheduler.start_futures_scheduler()

        assert result is True
        mock_scheduler.start.assert_called_once()

    @patch("src.services.futures_scheduler.os.getppid")
    @patch("src.services.futures_scheduler.os.getpid")
    @patch("src.services.futures_scheduler._acquire_process_lock")
    @patch("src.services.futures_scheduler.is_scheduler_running")
    def test_start_futures_scheduler_fails_when_cant_acquire_lock(
        self, mock_running, mock_lock, mock_getpid, mock_getppid
    ):
        """Should return False when can't acquire process lock."""
        # Make pids match so it doesn't spawn subprocess
        mock_getpid.return_value = 1000
        mock_getppid.return_value = 1000

        mock_running.return_value = False
        mock_lock.return_value = False

        result = futures_scheduler.start_futures_scheduler()

        assert result is False

    @patch("src.services.futures_scheduler.update_scheduler_status")
    @patch("src.services.futures_scheduler.send_scheduler_notification")
    @patch("src.services.futures_scheduler._release_process_lock")
    def test_stop_futures_scheduler_shuts_down_cleanly(
        self, mock_release, mock_notify, mock_status
    ):
        """Should shut down scheduler cleanly."""
        mock_scheduler = Mock()
        mock_scheduler.running = True
        futures_scheduler.scheduler = mock_scheduler

        result = futures_scheduler.stop_futures_scheduler()

        assert result is True
        mock_scheduler.shutdown.assert_called_once_with(wait=True)
        mock_release.assert_called_once()

    @patch("src.services.futures_scheduler.update_scheduler_status")
    @patch("src.services.futures_scheduler.send_scheduler_notification")
    @patch("src.services.futures_scheduler._release_process_lock")
    def test_stop_futures_scheduler_handles_no_scheduler(
        self, mock_release, mock_notify, mock_status
    ):
        """Should handle stopping when no scheduler exists."""
        futures_scheduler.scheduler = None

        result = futures_scheduler.stop_futures_scheduler()

        assert result is True
        mock_release.assert_called_once()

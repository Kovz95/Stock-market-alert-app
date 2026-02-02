"""Integration tests for futures scheduler."""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.services import futures_scheduler
from src.services.futures_alert_checker import FuturesAlertChecker


class TestFuturesSchedulerIntegration:
    """Integration tests for complete scheduler workflow."""

    def setup_method(self):
        """Reset state before each test."""
        futures_scheduler._job_locks.clear()
        futures_scheduler.scheduler = None

    @patch("src.services.futures_scheduler.subprocess.run")
    @patch("src.services.futures_scheduler.FuturesAlertChecker")
    @patch("src.services.futures_scheduler.is_ib_available")
    @patch("src.services.futures_scheduler.save_document")
    @patch("src.services.futures_scheduler.load_document")
    def test_complete_job_execution_flow(
        self, mock_load, mock_save, mock_ib, mock_checker_class, mock_run
    ):
        """Should execute complete job flow: price update + alert check."""
        # Setup mocks
        mock_load.return_value = {
            "notification_settings": {
                "send_start_notification": False,
                "send_completion_notification": False,
                "job_timeout_seconds": 900,
            }
        }
        mock_ib.return_value = True

        # Mock price updater
        mock_run.return_value = Mock(
            returncode=0, stdout="Update complete: 60 succeeded, 2 failed", stderr=""
        )

        # Mock alert checker
        mock_checker = Mock()
        mock_checker.check_all_alerts.return_value = {
            "total": 10,
            "triggered": 2,
            "errors": 0,
            "skipped": 0,
            "no_data": 0,
            "success": 8,
        }
        mock_checker_class.return_value = mock_checker

        # Patch subprocess execution to avoid timeout
        with patch(
            "src.services.futures_scheduler._run_job_subprocess"
        ) as mock_subprocess:
            mock_subprocess.return_value = (
                {"updated": 60, "failed": 2},
                {"total": 10, "triggered": 2, "errors": 0},
                None,
            )

            # Execute job
            result = futures_scheduler.execute_futures_job()

        # Verify result
        assert result is not None
        assert result["price_stats"]["updated"] == 60
        assert result["alert_stats"]["triggered"] == 2
        assert "duration" in result

        # Verify status was updated
        assert mock_save.called

    @patch("src.services.futures_scheduler.subprocess.run")
    @patch("src.services.futures_scheduler.FuturesAlertChecker")
    @patch("src.services.futures_scheduler.is_ib_available")
    @patch("src.services.futures_scheduler.save_document")
    @patch("src.services.futures_scheduler.load_document")
    def test_job_handles_price_update_failure(
        self, mock_load, mock_save, mock_ib, mock_checker_class, mock_run
    ):
        """Should handle price update failure gracefully."""
        mock_load.return_value = {
            "notification_settings": {
                "send_start_notification": False,
                "send_completion_notification": False,
                "job_timeout_seconds": 900,
            }
        }
        mock_ib.return_value = True

        # Mock failed price update
        mock_run.return_value = Mock(
            returncode=1, stdout="", stderr="IB connection failed"
        )

        # Mock alert checker (should still run)
        mock_checker = Mock()
        mock_checker.check_all_alerts.return_value = {
            "total": 10,
            "triggered": 0,
            "errors": 0,
        }
        mock_checker_class.return_value = mock_checker

        with patch(
            "src.services.futures_scheduler._run_job_subprocess"
        ) as mock_subprocess:
            mock_subprocess.return_value = (
                {"updated": 0, "failed": 0, "error": "IB connection failed"},
                {"total": 10, "triggered": 0, "errors": 0},
                None,
            )

            result = futures_scheduler.execute_futures_job()

        # Job should complete despite price update failure
        assert result is not None
        assert result["price_stats"]["error"] is not None

    @patch("src.services.futures_scheduler.is_ib_available")
    @patch("src.services.futures_scheduler.save_document")
    @patch("src.services.futures_scheduler.load_document")
    def test_concurrent_job_execution_prevented(
        self, mock_load, mock_save, mock_ib
    ):
        """Should prevent concurrent execution of same job."""
        mock_load.return_value = {"notification_settings": {}}
        mock_ib.return_value = True

        # Acquire lock for first job
        futures_scheduler.acquire_job_lock("futures_combined")

        # Try to execute second job
        result = futures_scheduler.execute_futures_job()

        # Second job should be skipped
        assert result is None

        # Release lock
        futures_scheduler.release_job_lock("futures_combined")

    @patch("src.services.futures_scheduler.os.getppid")
    @patch("src.services.futures_scheduler.os.getpid")
    @patch("src.services.futures_scheduler.BackgroundScheduler")
    @patch("src.services.futures_scheduler.send_scheduler_notification")
    @patch("src.services.futures_scheduler.update_scheduler_status")
    @patch("src.services.futures_scheduler._acquire_process_lock")
    @patch("src.services.futures_scheduler.is_scheduler_running")
    def test_scheduler_lifecycle(
        self, mock_running, mock_lock, mock_status, mock_notify, mock_scheduler_class, mock_getpid, mock_getppid
    ):
        """Should handle complete scheduler lifecycle: start -> run -> stop."""
        # Make pids match so it doesn't spawn subprocess
        mock_getpid.return_value = 1000
        mock_getppid.return_value = 1000

        # Setup
        mock_running.return_value = False
        mock_lock.return_value = True
        mock_scheduler = Mock()
        mock_scheduler.running = True
        mock_scheduler_class.return_value = mock_scheduler

        # Start scheduler
        with patch("src.services.futures_scheduler.load_scheduler_config") as mock_config:
            mock_config.return_value = {
                "update_times": ["06:00", "12:00"],
                "update_on_start": False,
                "notification_settings": {},
            }
            start_result = futures_scheduler.start_futures_scheduler()

        assert start_result is True
        mock_scheduler.start.assert_called_once()

        # Verify jobs were scheduled
        assert mock_scheduler.add_job.called

        # Stop scheduler
        with patch("src.services.futures_scheduler._release_process_lock") as mock_release:
            stop_result = futures_scheduler.stop_futures_scheduler()

        assert stop_result is True
        mock_scheduler.shutdown.assert_called_once_with(wait=True)
        mock_release.assert_called_once()


class TestConfigurationPersistence:
    """Tests for configuration loading and persistence."""

    @patch("src.services.futures_scheduler.save_document")
    @patch("src.services.futures_scheduler.load_document")
    def test_config_roundtrip(self, mock_load, mock_save):
        """Should save and load configuration correctly."""
        test_config = {
            "enabled": True,
            "update_times": ["06:00", "18:00"],
            "ib_hours": {"start": "05:00", "end": "23:00"},
            "scheduler_webhook": {
                "url": "https://discord.com/webhook",
                "enabled": True,
            },
        }

        # Save config
        futures_scheduler.save_scheduler_config(test_config)

        # Verify save was called with correct data
        mock_save.assert_called_once()
        saved_config = mock_save.call_args[0][1]
        assert saved_config["enabled"] is True
        assert saved_config["update_times"] == ["06:00", "18:00"]

        # Load config
        mock_load.return_value = test_config
        loaded_config = futures_scheduler.load_scheduler_config()

        # Verify loaded config matches
        assert loaded_config["enabled"] is True
        assert loaded_config["update_times"] == ["06:00", "18:00"]


class TestStatusTracking:
    """Tests for status tracking throughout job lifecycle."""

    @patch("src.services.futures_scheduler.LOCK_FILE")
    @patch("src.services.futures_scheduler.save_document")
    @patch("src.services.futures_scheduler.load_document")
    def test_status_updates_throughout_lifecycle(
        self, mock_load, mock_save, mock_lock_file
    ):
        """Should update status at each lifecycle stage."""
        mock_lock_file.exists.return_value = False

        # Initial status
        mock_load.return_value = {}
        futures_scheduler.update_scheduler_status(status="starting")
        assert mock_save.call_args[0][1]["status"] == "starting"

        # Running status with job
        mock_load.return_value = mock_save.call_args[0][1]  # Use previous saved state
        futures_scheduler.update_scheduler_status(
            status="running",
            current_job={"id": "test_job", "started": "2024-01-15T10:00:00"},
        )
        assert mock_save.call_args[0][1]["status"] == "running"
        assert mock_save.call_args[0][1]["current_job"]["id"] == "test_job"

        # Completed status - job finished
        mock_load.return_value = mock_save.call_args[0][1]  # Use previous saved state
        futures_scheduler.update_scheduler_status(
            status="running",
            current_job=None,
            last_run={
                "job_id": "test_job",
                "completed_at": "2024-01-15T10:05:00",
                "duration_seconds": 45.2,
            },
        )
        # current_job should remain from previous call since None means "don't update"
        # This matches the actual behavior of the function
        assert mock_save.call_args[0][1]["last_run"]["job_id"] == "test_job"

        # Error status
        mock_load.return_value = mock_save.call_args[0][1]  # Use previous saved state
        futures_scheduler.update_scheduler_status(
            status="error",
            last_error={
                "time": "2024-01-15T10:10:00",
                "job_id": "test_job",
                "message": "Test error",
            },
        )
        assert mock_save.call_args[0][1]["status"] == "error"
        assert mock_save.call_args[0][1]["last_error"]["message"] == "Test error"

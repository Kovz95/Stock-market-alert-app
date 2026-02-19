"""
Shared infrastructure for all scheduler job types.

Provides a SchedulerServices class that centralises job-type-agnostic
infrastructure: configuration loading, Discord webhook posting, job locking,
status management, process-level locking, and heartbeat updates.

This module is consumed by scheduler_job_handler.py (Template Method handlers)
and auto_scheduler_v2.py (the thin orchestrator/facade).

Usage:
    from src.services.scheduler_services import SchedulerServices

    services = SchedulerServices()
    config = services.load_config()
    services.send_notification("Scheduler started")
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import psutil
import requests

from src.data_access.db_config import db_config
from src.data_access.document_store import load_document, save_document
from src.data_access.redis_support import build_key, delete_key

logger = logging.getLogger("auto_scheduler_v2")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
LOCK_FILE = BASE_DIR / "scheduler_v2.lock"
STATUS_DOCUMENT_KEY = "scheduler_status"  # Base key, will be mode-specific if needed
CONFIG_DOCUMENT_KEY = "scheduler_config"
JOB_TIMEOUT_SECONDS = int(os.getenv("SCHEDULER_JOB_TIMEOUT", "900"))
HEARTBEAT_INTERVAL = int(os.getenv("SCHEDULER_HEARTBEAT_INTERVAL", "60"))


# ---------------------------------------------------------------------------
# Module-level formatting helpers
# ---------------------------------------------------------------------------

def format_stats_for_message(price_stats: Any, alert_stats: Any) -> str:
    """Build a compact one-line summary for Discord."""
    price_parts: list[str] = []
    if isinstance(price_stats, dict):
        price_parts.append(f"upd {price_stats.get('updated', 0):,}")
        price_parts.append(f"fail {price_stats.get('failed', 0):,}")
        price_parts.append(f"skip {price_stats.get('skipped', 0):,}")
    else:
        price_parts.append(str(price_stats))

    alert_parts: list[str] = []
    if isinstance(alert_stats, dict):
        alert_parts.append(f"total {alert_stats.get('total', 0):,}")
        triggered = alert_stats.get("success", alert_stats.get("triggered", 0))
        errors = alert_stats.get("errors", 0)
        alert_parts.append(f"trig {triggered:,}")
        alert_parts.append(f"err {errors:,}")
    else:
        alert_parts.append(str(alert_stats))

    return f"Price ({', '.join(price_parts)}) | Alerts ({', '.join(alert_parts)})"


def format_duration(seconds: float) -> str:
    """Return a human-friendly duration string."""
    seconds = int(max(0, round(seconds)))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# SchedulerServices
# ---------------------------------------------------------------------------

class SchedulerServices:
    """Shared infrastructure for all scheduler job types.

    Centralises configuration loading, Discord notifications, job locking
    (thread-safe), status/heartbeat management, and process-level lock files.
    """

    def __init__(
        self,
        lock_file: Optional[Union[Path, str]] = None,
        mode: Optional[str] = None
    ) -> None:
        """
        Initialize scheduler services.

        Args:
            lock_file: Optional custom lock file path.
            mode: Optional scheduler mode ('daily', 'weekly', 'hourly').
                  If provided, uses mode-specific status key.
        """
        self._job_locks: Dict[str, bool] = {}
        self._lock_lock = threading.Lock()
        self._lock_file: Path = Path(lock_file) if lock_file else LOCK_FILE
        self._mode = mode

        # Use mode-specific status key if mode is provided
        if mode:
            self._status_key = f"{STATUS_DOCUMENT_KEY}_{mode}"
        else:
            self._status_key = STATUS_DOCUMENT_KEY

    # -- Configuration -------------------------------------------------------

    def load_config(self) -> Dict[str, Any]:
        """Load scheduler configuration from document store."""
        default_config = {
            "scheduler_webhook": {
                "url": "",
                "enabled": False,
                "name": "Scheduler Status",
            },
            "notification_settings": {
                "send_start_notification": True,
                "send_completion_notification": True,
                "send_progress_updates": False,
                "progress_update_interval": 300,
                "include_summary_stats": True,
                "job_timeout_seconds": JOB_TIMEOUT_SECONDS,
            },
        }

        try:
            config = load_document(
                CONFIG_DOCUMENT_KEY,
                default=default_config,
                fallback_path=str(BASE_DIR / "scheduler_config.json"),
            )
            if not isinstance(config, dict):
                return default_config
            return config
        except Exception as exc:
            logger.warning("Failed to load scheduler config: %s", exc)
            return default_config

    # -- Discord Notifications -----------------------------------------------

    def send_notification(self, message: str, event: str = "info") -> bool:
        """Send a scheduler status message to Discord (if configured)."""
        config = self.load_config()
        webhook_cfg = (config or {}).get("scheduler_webhook", {})

        if not webhook_cfg.get("enabled") or not webhook_cfg.get("url"):
            return False
        from src.utils.discord_env import get_discord_environment_tag, is_discord_send_enabled
        if not is_discord_send_enabled():
            return False

        payload = {
            "content": get_discord_environment_tag() + message,
            "username": webhook_cfg.get("name") or "Scheduler",
        }

        try:
            response = requests.post(webhook_cfg["url"], json=payload, timeout=10)
            if response.status_code not in (200, 204):
                logger.warning(
                    "Failed to send scheduler notification (%s): HTTP %s",
                    event,
                    response.status_code,
                )
                return False
            return True
        except Exception as exc:
            logger.warning("Error sending scheduler notification (%s): %s", event, exc)
            return False

    # -- Job Locking ---------------------------------------------------------

    @staticmethod
    def sanitize_job_id(job_type: str, exchange_name: str) -> str:
        """Create a consistent job ID from type and exchange."""
        return f"{job_type}_{exchange_name}".lower().replace(" ", "_")

    def acquire_job_lock(self, job_id: str) -> bool:
        """Acquire a lock for a job. Returns True if acquired, False if already locked."""
        with self._lock_lock:
            if self._job_locks.get(job_id):
                return False
            self._job_locks[job_id] = True
            return True

    def release_job_lock(self, job_id: str) -> None:
        """Release a job lock."""
        with self._lock_lock:
            self._job_locks.pop(job_id, None)

    # -- Status Management ---------------------------------------------------

    def update_status(
        self,
        status: str = "running",
        current_job: Optional[Dict[str, Any]] = None,
        last_run: Optional[Dict[str, Any]] = None,
        last_result: Optional[Dict[str, Any]] = None,
        last_error: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update scheduler status in document store."""
        try:
            # Use mode-specific fallback path if mode is set
            fallback_filename = f"scheduler_status_{self._mode}.json" if self._mode else "scheduler_status.json"
            fallback_path = str(BASE_DIR / fallback_filename)

            existing_status = load_document(
                self._status_key,
                default={},
                fallback_path=fallback_path,
            )
            if not isinstance(existing_status, dict):
                existing_status = {}

            existing_status["status"] = status
            existing_status["heartbeat"] = datetime.now(tz=timezone.utc).isoformat()
            existing_status["mode"] = self._mode  # Add mode to status for identification

            if current_job is not None:
                existing_status["current_job"] = current_job
            elif "current_job" not in existing_status:
                existing_status["current_job"] = None

            if last_run is not None:
                existing_status["last_run"] = last_run

            if last_result is not None:
                existing_status["last_result"] = last_result

            if last_error is not None:
                existing_status["last_error"] = last_error

            save_document(
                self._status_key,
                existing_status,
                fallback_path=fallback_path,
            )

            if self._lock_file.exists():
                try:
                    lock_data = json.loads(self._lock_file.read_text())
                    lock_data["heartbeat"] = datetime.now(tz=timezone.utc).isoformat()
                    self._lock_file.write_text(json.dumps(lock_data, indent=2))
                except Exception:
                    pass

        except Exception as exc:
            logger.warning("Failed to update scheduler status: %s", exc)

    def get_info(self, scheduler_instance: Any = None) -> Optional[Dict[str, Any]]:
        """Get current scheduler status and information.

        Args:
            scheduler_instance: Optional APScheduler BackgroundScheduler instance
                to include job counts and next run time.
        """
        try:
            fallback_filename = f"scheduler_status_{self._mode}.json" if self._mode else "scheduler_status.json"
            status = load_document(
                self._status_key,
                default={},
                fallback_path=str(BASE_DIR / fallback_filename),
            )

            if not isinstance(status, dict):
                return None

            if scheduler_instance:
                jobs = scheduler_instance.get_jobs()
                daily_jobs = [j for j in jobs if "daily" in j.id]
                weekly_jobs = [j for j in jobs if "weekly" in j.id]

                status["total_daily_jobs"] = len(daily_jobs)
                status["total_weekly_jobs"] = len(weekly_jobs)

                if jobs:
                    next_job = min(jobs, key=lambda j: j.next_run_time or datetime.max)
                    if next_job.next_run_time:
                        status["next_run"] = next_job.next_run_time.isoformat()

            return status

        except Exception as exc:
            logger.warning("Failed to get scheduler info: %s", exc)
            return None

    # -- Process Management --------------------------------------------------

    @staticmethod
    def process_matches(pid: int) -> bool:
        """Check if PID matches our scheduler process (running as main script)."""
        try:
            process = psutil.Process(pid)
            cmdline = process.cmdline() or []
            return any(segment.endswith("auto_scheduler_v2.py") for segment in cmdline)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return False

    def acquire_process_lock(self) -> bool:
        """Acquire process-level lock file."""
        current_pid = os.getpid()

        if self._lock_file.exists():
            try:
                info = json.loads(self._lock_file.read_text())
                existing_pid = info.get("pid")
            except Exception:
                existing_pid = None

            if existing_pid == current_pid:
                logger.debug(
                    "Lock file exists for current process (PID %s), updating timestamp",
                    current_pid,
                )
            elif existing_pid and self.process_matches(existing_pid):
                logger.warning("Another scheduler instance (PID %s) is running.", existing_pid)
                return False
            else:
                self._lock_file.unlink(missing_ok=True)

        payload = {
            "pid": current_pid,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "heartbeat": datetime.now(tz=timezone.utc).isoformat(),
            "started_at": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "type": "auto_scheduler_v2",
        }
        self._lock_file.write_text(json.dumps(payload, indent=2))
        return True

    def release_process_lock(self) -> None:
        """Release process-level lock file."""
        try:
            self._lock_file.unlink(missing_ok=True)
        except OSError as exc:
            logger.error("Failed to remove lock file: %s", exc)

    @staticmethod
    def is_scheduler_running() -> bool:
        """Check if scheduler is currently running as a main script."""
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            try:
                cmdline = proc.info.get("cmdline") or []
                if any(part.endswith("auto_scheduler_v2.py") for part in cmdline):
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return False

    # -- Heartbeat -----------------------------------------------------------

    def heartbeat(self) -> None:
        """Update heartbeat timestamp in lock file and status document.

        Writes the heartbeat timestamp directly without reading the current
        status first, keeping the operation lightweight even under DB load.
        """
        now = datetime.now(tz=timezone.utc).isoformat()

        # Update lock file (pure file I/O, always fast)
        try:
            if self._lock_file.exists():
                lock_data = json.loads(self._lock_file.read_text())
                lock_data["heartbeat"] = now
                self._lock_file.write_text(json.dumps(lock_data, indent=2))
        except Exception as exc:
            logger.debug("Heartbeat lock file update failed: %s", exc)

        # Write only the heartbeat timestamp directly to DB — no read required
        try:
            conn = db_config.get_connection()
            try:
                cur = conn.cursor()
                try:
                    cur.execute(
                        """
                        INSERT INTO app_documents (document_key, payload, updated_at)
                        VALUES (%s, jsonb_build_object('status', 'running', 'heartbeat', %s::text), NOW())
                        ON CONFLICT (document_key) DO UPDATE SET
                            payload = app_documents.payload
                                || jsonb_build_object('status', 'running', 'heartbeat', EXCLUDED.payload->>'heartbeat'),
                            updated_at = EXCLUDED.updated_at
                        """,
                        (self._status_key, now),
                    )
                finally:
                    cur.close()
                conn.commit()
            finally:
                db_config.close_connection(conn)

            # Invalidate Redis cache so next read gets fresh data
            delete_key(build_key(f"document:{self._status_key}"))
        except Exception as exc:
            logger.debug("Heartbeat DB update failed: %s", exc)

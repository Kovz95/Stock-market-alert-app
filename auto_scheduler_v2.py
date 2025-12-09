"""
Compatibility shim for the missing auto_scheduler_v2 source module.

The original Python file is unavailable, but the compiled bytecode still
exists under __pycache__/auto_scheduler_v2_legacy.cpython-311.pyc.  This
loader executes that bytecode so imports continue to work until the real
source is restored.
"""

from __future__ import annotations

import os
import marshal
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from types import CodeType
from typing import Any, Dict
from datetime import datetime

import requests
from data_access.metadata_repository import fetch_stock_metadata_df

_LEGACY_BYTECODE = Path(__file__).parent / "__pycache__/auto_scheduler_v2_legacy.cpython-311.pyc"
_PYC_HEADER_SIZE = 16
_BOOTSTRAPPED = False
JOB_TIMEOUT_SECONDS = int(os.getenv("SCHEDULER_JOB_TIMEOUT", "900"))  # 15 minutes default


def _load_legacy_code() -> CodeType:
    if not _LEGACY_BYTECODE.exists():
        raise FileNotFoundError(
            f"Missing legacy bytecode: {_LEGACY_BYTECODE}. "
            "auto_scheduler_v2 cannot be bootstrapped."
        )
    with _LEGACY_BYTECODE.open("rb") as fh:
        fh.read(_PYC_HEADER_SIZE)
        return marshal.load(fh)


def _bootstrap(namespace: Dict[str, Any]) -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return
    code = _load_legacy_code()
    exec(code, namespace)
    _BOOTSTRAPPED = True


_bootstrap(globals())

# --- Patched scheduler Discord notifications ---------------------------------
# The legacy bytecode doesn't expose readable source, so we override the
# notification helpers here to fix the webhook status-code check and emit
# start/complete/error messages for each exchange job.

# Keep references to the legacy implementations in case they're needed later.
_legacy_send_scheduler_notification = globals().get("send_scheduler_notification")
_legacy_execute_exchange_job = globals().get("execute_exchange_job")
_legacy_run_daily_job = globals().get("run_daily_job")
_legacy_run_weekly_job = globals().get("run_weekly_job")


def _list_exchanges():
    """Return sorted unique exchanges from stock metadata."""
    try:
        df = fetch_stock_metadata_df()
        if df is None or df.empty or "exchange" not in df.columns:
            return []
        exchanges = sorted({str(x) for x in df["exchange"].dropna().unique() if str(x).strip()})
        return exchanges
    except Exception:
        return []


def send_scheduler_notification(message: str, event: str = "info") -> bool:
    """Send a scheduler status message to Discord (if configured)."""
    config = load_scheduler_config()
    webhook_cfg = (config or {}).get("scheduler_webhook", {})

    if not webhook_cfg.get("enabled") or not webhook_cfg.get("url"):
        logger.info("Scheduler webhook not configured or disabled.")
        return False

    payload = {
        "content": message,
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
    except Exception as exc:  # pragma: no cover - network edge cases
        logger.warning("Error sending scheduler notification (%s): %s", event, exc)
        return False


def _format_stats_for_message(price_stats, alert_stats) -> str:
    """Build a compact one-line summary for Discord."""
    price_parts = []
    if isinstance(price_stats, dict):
        price_parts.append(f"upd {price_stats.get('updated', 0):,}")
        price_parts.append(f"fail {price_stats.get('failed', 0):,}")
        price_parts.append(f"skip {price_stats.get('skipped', 0):,}")
        price_parts.append(f"stale {price_stats.get('stale', 0):,}")
        price_parts.append(f"new {price_stats.get('new', 0):,}")
        skipped_list = price_stats.get("skipped_tickers") or []
        if skipped_list:
            sample = ", ".join(skipped_list[:8])
            if len(skipped_list) > 8:
                sample += ", …"
            price_parts.append(f"skipped_tickers [{sample}]")
    else:
        price_parts.append(str(price_stats))

    alert_parts = []
    if isinstance(alert_stats, dict):
        alert_parts.append(f"total {alert_stats.get('total', 0):,}")
        triggered = alert_stats.get("success", alert_stats.get("triggered", 0))
        errors = alert_stats.get("errors", 0)
        no_data = alert_stats.get("no_data", 0)
        stale = alert_stats.get("stale_data", 0)
        not_triggered = max(alert_stats.get("total", 0) - (triggered + errors + no_data + stale), 0)
        alert_parts.append(f"trig {triggered:,}")
        alert_parts.append(f"not {not_triggered:,}")
        alert_parts.append(f"err {errors:,}")
        alert_parts.append(f"no-data {no_data:,}")
        alert_parts.append(f"stale {stale:,}")
    else:
        alert_parts.append(str(alert_stats))

    return f"Price ({', '.join(price_parts)}) | Alerts ({', '.join(alert_parts)})"


def _format_duration(seconds: float) -> str:
    """Return a human-friendly duration string."""
    seconds = int(max(0, round(seconds)))
    minutes, secs = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    parts = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def _run_with_timeout(func, timeout_seconds: int, *args, **kwargs):
    """Execute a callable with a hard timeout to prevent scheduler hangs."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        return future.result(timeout=timeout_seconds)


def _job_worker(queue, exchange_name: str, resample_weekly: bool):
    """Worker executed in a subprocess to isolate hangs/crashes."""
    try:
        timeframe_key = "weekly" if resample_weekly else "daily"
        price_stats = update_prices_for_exchanges([exchange_name], resample_weekly=resample_weekly)
        alert_stats = run_alert_checks([exchange_name], timeframe_key)
        queue.put({"ok": True, "price_stats": price_stats, "alert_stats": alert_stats})
    except Exception as exc:  # pragma: no cover - executed in child
        queue.put({"ok": False, "error": str(exc)})


def _run_job_subprocess(exchange_name: str, resample_weekly: bool, timeout_seconds: int):
    """
    Run a single exchange job in a subprocess with a hard timeout.
    Returns (price_stats, alert_stats, error_message)
    """
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_job_worker, args=(queue, exchange_name, resample_weekly))
    proc.start()
    start = time.time()
    last_progress = start

    notify_settings = (load_scheduler_config() or {}).get("notification_settings", {})
    send_progress = notify_settings.get("send_progress_updates", False)
    progress_interval = max(int(notify_settings.get("progress_update_interval", 0) or 0), 0)

    def _maybe_send_progress(elapsed: float, remaining: float):
        if not send_progress or progress_interval <= 0:
            return
        try:
            send_scheduler_notification(
                "\n".join(
                    [
                        "⏳ **Job in progress**",
                        f"• Exchange: {exchange_name}",
                        f"• Job ID: `{sanitize_job_id('weekly' if resample_weekly else 'daily', exchange_name)}`",
                        f"• Elapsed: {_format_duration(elapsed)}",
                        f"• Time remaining: {_format_duration(max(remaining, 0))}",
                    ]
                ),
                event="progress",
            )
        except Exception:
            logger.debug("Progress notification failed", exc_info=True)

    while True:
        elapsed = time.time() - start
        remaining = timeout_seconds - elapsed
        if remaining <= 0:
            proc.terminate()
            proc.join(5)
            return None, None, f"timeout after {timeout_seconds}s"

        wait_time = min(progress_interval or remaining, remaining, 5)
        proc.join(wait_time)
        if not proc.is_alive():
            break

        if progress_interval and time.time() - last_progress >= progress_interval:
            _maybe_send_progress(elapsed, remaining)
            last_progress = time.time()

    if queue.empty():
        return None, None, "no result from worker"

    result = queue.get()
    if not result.get("ok"):
        return None, None, result.get("error", "unknown error")

    return result.get("price_stats"), result.get("alert_stats"), None


def execute_exchange_job(exchange_name: str, job_type: str):
    """
    Override of the legacy job runner to add Discord notifications while
    preserving the original scheduling behaviour.
    """
    job_id = sanitize_job_id(job_type, exchange_name)
    if not acquire_job_lock(job_id):
        logger.warning("Job %s is already running; skipping duplicate execution.", job_id)
        return None

    notify_settings = (load_scheduler_config() or {}).get("notification_settings", {})
    send_start = notify_settings.get("send_start_notification", True)
    send_complete = notify_settings.get("send_completion_notification", True)
    job_timeout = max(int(notify_settings.get("job_timeout_seconds", JOB_TIMEOUT_SECONDS)), 60)
    deadline = time.time() + job_timeout

    start_time = time.time()
    if send_start:
        send_scheduler_notification(
            "\n".join(
                [
                    f"🚀 **{job_type.title()} job started**",
                    f"• Exchange: {exchange_name}",
                    f"• Job ID: `{job_id}`",
                    f"• Start (UTC): {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}Z",
                    f"• Timeout: {job_timeout}s",
                ]
            ),
            event="start",
        )

    try:
        update_scheduler_status(
            status="running",
            current_job={
                "id": job_id,
                "exchange": exchange_name,
                "job_type": job_type,
                "started": datetime.utcnow().isoformat(),
            },
        )

        exchanges = [exchange_name]
        resample_weekly = job_type == "weekly"
        timeframe_key = "weekly" if resample_weekly else "daily"

        remaining = max(int(deadline - time.time()), 1)
        price_stats, alert_stats, worker_err = _run_job_subprocess(
            exchange_name,
            resample_weekly=resample_weekly,
            timeout_seconds=remaining,
        )

        if worker_err:
            raise TimeoutError(worker_err) if "timeout" in worker_err else RuntimeError(worker_err)

        duration = round(time.time() - start_time, 2)

        update_scheduler_status(
            status="running",
            current_job=None,
            last_run={
                "job_id": job_id,
                "exchange": exchange_name,
                "job_type": job_type,
                "completed_at": datetime.utcnow().isoformat(),
                "duration_seconds": duration,
            },
            last_result={
                "price_stats": price_stats,
                "alert_stats": alert_stats,
            },
        )

        logger.info(
            "%s job finished for %s (price updates: %s, alerts processed: %s)",
            job_type.upper(),
            exchange_name,
            price_stats.get("updated", 0) if isinstance(price_stats, dict) else price_stats,
            alert_stats.get("total", 0) if isinstance(alert_stats, dict) else alert_stats,
        )

        if send_complete:
            send_scheduler_notification(
                "\n".join(
                    [
                        f"🏁 **{job_type.title()} job complete**",
                        f"• Exchange: {exchange_name}",
                        f"• Duration: {duration}s",
                        f"• Summary: {_format_stats_for_message(price_stats, alert_stats)}",
                    ]
                ),
                event="complete",
            )

        return {
            "price_stats": price_stats,
            "alert_stats": alert_stats,
            "duration": duration,
        }
    except Exception as exc:  # pragma: no cover - handled operationally
        err_msg = (
            f"Timeout after {job_timeout}s"
            if isinstance(exc, TimeoutError)
            else str(exc)
        )
        logger.exception("Error running %s job for %s: %s", job_type, exchange_name, err_msg)
        update_scheduler_status(
            status="error",
            current_job=None,
            last_error={
                "time": datetime.utcnow().isoformat(),
                "job_id": job_id,
                "exchange": exchange_name,
                "job_type": job_type,
                "message": err_msg,
            },
        )
        send_scheduler_notification(
            "\n".join(
                [
                    f"❌ **{job_type.title()} job failed**",
                    f"• Exchange: {exchange_name}",
                    f"• Error: {err_msg}",
                    f"• Duration: {round(time.time() - start_time, 2)}s",
                ]
            ),
            event="error",
        )
        return None
    finally:
        release_job_lock(job_id)


def run_daily_job(exchange_name: str):
    """
    Extend legacy daily job to support 'ALL' which iterates all exchanges.
    """
    if exchange_name and str(exchange_name).upper() == "ALL":
        exchanges = _list_exchanges()
        if not exchanges:
            logger.warning("No exchanges found for ALL daily job.")
            return None
        for ex in exchanges:
            try:
                if _legacy_run_daily_job:
                    _legacy_run_daily_job(ex)
            except Exception:
                logger.exception("Daily job failed for %s during ALL run", ex)
        return None

    if _legacy_run_daily_job:
        return _legacy_run_daily_job(exchange_name)
    raise RuntimeError("Legacy run_daily_job implementation missing.")


def run_weekly_job(exchange_name: str):
    """
    Extend legacy weekly job to support 'ALL' which iterates all exchanges.
    """
    if exchange_name and str(exchange_name).upper() == "ALL":
        exchanges = _list_exchanges()
        if not exchanges:
            logger.warning("No exchanges found for ALL weekly job.")
            return None
        for ex in exchanges:
            try:
                if _legacy_run_weekly_job:
                    _legacy_run_weekly_job(ex)
            except Exception:
                logger.exception("Weekly job failed for %s during ALL run", ex)
        return None

    if _legacy_run_weekly_job:
        return _legacy_run_weekly_job(exchange_name)
    raise RuntimeError("Legacy run_weekly_job implementation missing.")

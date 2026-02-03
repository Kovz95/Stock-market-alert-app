#!/usr/bin/env python3
"""
Daily/Weekly Scheduler Status Page
----------------------------------

Display the health of the daily/weekly background schedulers and the upcoming
exchange checks using the exchange-calendars integration.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import psutil
import pytz
import streamlit as st

# Ensure we can import the project modules when running via Streamlit
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from src.data_access.db_config import db_config  # noqa: E402
from src.services.auto_scheduler_v2 import (  # noqa: E402
    get_scheduler_info,
    run_daily_job,
    start_auto_scheduler,
    stop_auto_scheduler,
)
from src.config.exchange_schedule_config import (  # noqa: E402
    EXCHANGE_SCHEDULES,
    get_exchanges_by_closing_time,
    is_dst_active,
)
from src.services.calendar_adapter import get_calendar_timezone  # noqa: E402
from src.services.pivot_support_resistance import (
    PIVOT_SR,
    PIVOT_SR_CROSSOVER,
    PIVOT_SR_PROXIMITY,
)
from src.data_access.document_store import load_document, save_document  # noqa: E402
from src.utils.docker_utils import (  # noqa: E402
    is_container_running,
    get_container_status,
    start_container,
    stop_container,
)

SCHEDULER_PREF_FILE = BASE_DIR / "scheduler_preference.json"
SCHEDULER_PREF_DOCUMENT = "scheduler_preference"


def _trigger_rerun() -> None:
    """Use the supported rerun API for the installed Streamlit version."""
    rerun_fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rerun_fn:
        rerun_fn()


def is_scheduler_running_local() -> bool:
    """Check if scheduler is running as a local process."""
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if any("auto_scheduler_v2" in part for part in cmdline):
                return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return False


def is_scheduler_process_running() -> bool:
    """Check if daily scheduler is running (Docker container or local process)."""
    # Check Docker container first
    if is_container_running("daily"):
        return True
    # Fall back to local process check
    return is_scheduler_running_local()


def is_running_in_docker_mode() -> bool:
    """Check if the daily scheduler is running as a Docker container."""
    return is_container_running("daily")


def start_scheduler_process() -> bool:
    """Start the daily scheduler (Docker container or local process)."""
    # Try Docker first
    try:
        success, message = start_container("daily")
        if success:
            time.sleep(2)
            return True
    except Exception:
        pass  # Fall through to local start

    # Fall back to local start
    try:
        if start_auto_scheduler():
            time.sleep(2)
            return True
        return is_scheduler_process_running()
    except Exception as exc:
        st.error(f"Failed to start scheduler: {exc}")
        return False


def stop_scheduler_process() -> bool:
    """Stop the daily scheduler (Docker container or local process)."""
    # Try Docker first
    try:
        if is_container_running("daily"):
            success, message = stop_container("daily")
            if success:
                time.sleep(2)
                return True
    except Exception:
        pass  # Fall through to local stop

    # Fall back to local stop
    try:
        if stop_auto_scheduler():
            time.sleep(2)
            return True
        return not is_scheduler_process_running()
    except Exception as exc:
        st.error(f"Failed to stop scheduler: {exc}")
        return False


def _format_timedelta(delta: timedelta) -> str:
    seconds = int(delta.total_seconds())
    if seconds <= 0:
        return "Now"
    hours, remainder = divmod(seconds, 3600)
    minutes, _ = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _region_from_tz(tz_name: Optional[str]) -> str:
    if not tz_name:
        return "Unknown"
    if tz_name.startswith(("Asia/", "Australia/", "Pacific/")):
        return "Asia-Pacific"
    if tz_name.startswith(("America/", "Atlantic/")):
        return "Americas"
    return "Europe"


def build_schedule_dataframe(
    time_groups: Dict[str, List[dict]],
    current_time_et: datetime,
    eastern_tz: pytz.BaseTzInfo,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for exchanges in time_groups.values():
        for info in exchanges:
            exchange = info["exchange"]
            config = EXCHANGE_SCHEDULES.get(exchange, {})

            run_utc = pd.Timestamp(info.get("run_utc") or info.get("close_utc"))
            run_et = run_utc.tz_convert(eastern_tz)
            close_utc = pd.Timestamp(info.get("close_utc"))
            tz_name = get_calendar_timezone(exchange)
            close_local = close_utc.tz_convert(tz_name) if tz_name else close_utc

            rows.append(
                {
                    "Exchange": config.get("name", exchange),
                    "Symbol": exchange,
                    "Region": _region_from_tz(tz_name),
                    "Run (ET)": run_et,
                    "Run (UTC)": run_utc,
                    "Local Close": close_local,
                    "Local TZ": tz_name,
                    "Time Remaining": run_et - current_time_et,
                }
            )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.sort_values("Run (ET)", inplace=True)
    return df.reset_index(drop=True)


def _normalize_datetime_column(df: pd.DataFrame, column: str) -> None:
    """Ensure column is datetime and drop timezone for display."""
    if column not in df.columns:
        return
    if not pd.api.types.is_datetime64_any_dtype(df[column]):
        df[column] = pd.to_datetime(df[column], errors="coerce")
    if pd.api.types.is_datetime64_any_dtype(df[column]):
        df[column] = df[column].dt.tz_localize(None)


def _format_datetime_string(value: Optional[pd.Timestamp], tz_label: str | None = None) -> str:
    if pd.isna(value):
        return ""
    formatted = pd.Timestamp(value).strftime("%Y-%m-%d %I:%M %p")
    if tz_label:
        return f"{formatted} {tz_label}"
    return formatted


def _parse_iso_datetime(value: str, tz) -> Optional[pd.Timestamp]:
    """Parse ISO string and convert to provided timezone."""
    if not value:
        return None
    ts = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(ts):
        return None
    try:
        return ts.tz_convert(tz)
    except Exception:
        return None


def _fetch_last_runs(exchange_symbols: List[str]) -> Dict[str, Dict[str, pd.Timestamp]]:
    """
    Return the most recent start/end timestamps per exchange from alert_audits.
    Uses the last day we have data for each exchange.
    """
    if not exchange_symbols:
        return {}

    query = """
        WITH per_day AS (
            SELECT
                exchange,
                DATE(timestamp) AS day,
                MIN(timestamp) AS start_ts,
                MAX(timestamp) AS end_ts
            FROM alert_audits
            WHERE exchange = ANY(%s)
              AND evaluation_type IN ('scheduled', 'parallel')
            GROUP BY exchange, DATE(timestamp)
        ),
        latest AS (
            SELECT DISTINCT ON (exchange)
                exchange, day, start_ts, end_ts
            FROM per_day
            ORDER BY exchange, day DESC
        )
        SELECT exchange, day, start_ts, end_ts
        FROM latest;
    """

    results: Dict[str, Dict[str, pd.Timestamp]] = {}
    try:
        with db_config.connection(role="alerts") as conn:
            with conn.cursor() as cur:
                cur.execute(query, (exchange_symbols,))
                for exchange, day, start_ts, end_ts in cur.fetchall():
                    start_dt = pd.to_datetime(start_ts)
                    end_dt = pd.to_datetime(end_ts)
                    if start_dt.tzinfo is None:
                        start_dt = start_dt.tz_localize("UTC")
                    if end_dt.tzinfo is None:
                        end_dt = end_dt.tz_localize("UTC")
                    results[str(exchange)] = {
                        "day": pd.to_datetime(day),
                        "start": start_dt,
                        "end": end_dt,
                    }
    except Exception:
        # If the database is unreachable or the table is missing, just return no data.
        return {}

    return results


def _set_run_now_exchange(exchange: str) -> None:
    """Callback: queue this exchange for 'run now' on next rerun."""
    st.session_state["run_now_exchange"] = exchange


def _format_last_run(
    exchange: str,
    last_runs: Dict[str, Dict[str, pd.Timestamp]],
    tz_label: str,
    tz: pytz.BaseTzInfo,
) -> str:
    info = last_runs.get(exchange)
    if not info:
        return "—"

    start_et = info["start"].tz_convert(tz)
    end_et = info["end"].tz_convert(tz)
    date_str = info["day"].strftime("%Y-%m-%d")

    start_str = _format_datetime_string(start_et, tz_label)
    end_str = _format_datetime_string(end_et, tz_label)
    return f"{date_str} • Start {start_str} → Done {end_str}"


def main() -> None:
    st.set_page_config(page_title="Daily/Weekly Scheduler Status", page_icon="⏰", layout="wide")
    st.title("⏰ Daily/Weekly Scheduler Status")

    eastern_tz = pytz.timezone("America/New_York")
    current_time_et = datetime.now(eastern_tz)
    is_edt = is_dst_active()
    tz_label = "EDT" if is_edt else "EST"

    status_col, time_col, day_col, dst_col = st.columns([2, 1, 1, 1])

    with status_col:
        st.subheader("📊 Scheduler Status")
        running = is_scheduler_process_running()
        docker_running = is_running_in_docker_mode()
        if running:
            run_mode = "Docker" if docker_running else "Local"
            st.success(f"✅ Scheduler is RUNNING ({run_mode})")
        else:
            st.error("❌ Scheduler is NOT running")

        scheduler_info = get_scheduler_info()
        if scheduler_info:
            daily_jobs = scheduler_info.get("total_daily_jobs", scheduler_info.get("daily_jobs", 0))
            weekly_jobs = scheduler_info.get("total_weekly_jobs", scheduler_info.get("weekly_jobs", 0))
            st.caption(f"Daily jobs: {daily_jobs} • Weekly jobs: {weekly_jobs}")
            next_run = scheduler_info.get("next_run")
            if next_run:
                st.caption(f"Next run: {next_run}")
            last_run_info = scheduler_info.get("last_run") or {}
            last_completed = _parse_iso_datetime(last_run_info.get("completed_at"), eastern_tz)
            if last_completed is not None:
                ago = current_time_et - last_completed
                exchange = last_run_info.get("exchange", "Unknown")
                job_type = last_run_info.get("job_type", "daily/weekly")
                duration = last_run_info.get("duration_seconds", "?")
                st.caption(
                    f"Last job: {exchange} ({job_type}) at "
                    f"{_format_datetime_string(last_completed, tz_label)} • "
                    f"{_format_timedelta(ago)} ago • Duration: {duration}s"
                )

        start_btn, stop_btn = st.columns(2)
        with start_btn:
            if st.button("▶️ Start Scheduler", disabled=running):
                with st.spinner("Starting scheduler..."):
                    if start_scheduler_process():
                        run_mode = "Docker container" if is_running_in_docker_mode() else "background process"
                        st.success(f"Scheduler started as {run_mode}")
                        pref = load_document(
                            SCHEDULER_PREF_DOCUMENT,
                            default={},
                            fallback_path=str(SCHEDULER_PREF_FILE),
                        )
                        if not isinstance(pref, dict):
                            pref = {}
                        pref["enabled"] = True
                        save_document(
                            SCHEDULER_PREF_DOCUMENT,
                            pref,
                            fallback_path=str(SCHEDULER_PREF_FILE),
                        )
                        _trigger_rerun()
                    else:
                        st.error(
                            "Failed to start scheduler. Check logs for errors."
                        )
                        st.code("docker compose logs daily-scheduler", language="bash")
        with stop_btn:
            if st.button("⏹️ Stop Scheduler", disabled=not running):
                with st.spinner("Stopping scheduler..."):
                    if stop_scheduler_process():
                        st.success("Scheduler stopped")
                        pref = load_document(
                            SCHEDULER_PREF_DOCUMENT,
                            default={},
                            fallback_path=str(SCHEDULER_PREF_FILE),
                        )
                        if not isinstance(pref, dict):
                            pref = {}
                        pref["enabled"] = False
                        save_document(
                            SCHEDULER_PREF_DOCUMENT,
                            pref,
                            fallback_path=str(SCHEDULER_PREF_FILE),
                        )
                        _trigger_rerun()
                    else:
                        st.error("Failed to stop scheduler")
                        if docker_running:
                            st.code("docker compose stop daily-scheduler", language="bash")

    with time_col:
        st.subheader("🕒 Current Time")
        st.info(f"**{current_time_et.strftime('%I:%M %p')} {tz_label}**")
        st.caption(current_time_et.strftime("%Y-%m-%d"))

    with day_col:
        st.subheader("📅 Today")
        day_name = current_time_et.strftime("%A")
        st.info(f"**{day_name}**")
        st.caption("Trading day" if day_name not in {"Saturday", "Sunday"} else "Weekend")

    with dst_col:
        st.subheader("🌡️ DST Status")
        if is_edt:
            st.info("**EDT Active**")
            st.caption("Summer time")
        else:
            st.info("**EST Active**")
            st.caption("Winter time")

    st.markdown("---")

    time_groups = get_exchanges_by_closing_time()
    schedule_df = build_schedule_dataframe(time_groups, current_time_et, eastern_tz)
    last_runs = _fetch_last_runs(schedule_df["Symbol"].tolist())

    if schedule_df.empty:
        st.warning("No exchange schedule data available.")
    else:
        upcoming = schedule_df[schedule_df["Run (ET)"] > current_time_et].head(5)
        st.subheader("🔜 Next 5 Exchange Checks")
        if upcoming.empty:
            st.info("No additional checks scheduled today.")
        else:
            for _, row in upcoming.iterrows():
                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.markdown(f"**{row['Exchange']}**")
                    st.caption(f"{row['Region']} • {row['Symbol']}")
                with col2:
                    st.metric("Run (ET)", _format_datetime_string(row["Run (ET)"], tz_label))
                with col3:
                    st.metric("Time Remaining", _format_timedelta(row["Time Remaining"]))

        st.markdown("---")
        st.subheader("📅 Complete Exchange Schedule")

        df_display = schedule_df.copy()
        df_display["Last Run (ET)"] = df_display["Symbol"].apply(
            lambda sym: _format_last_run(sym, last_runs, tz_label, eastern_tz)
        )
        for column in ["Run (ET)", "Run (UTC)", "Local Close"]:
            _normalize_datetime_column(df_display, column)
        df_display["Run (ET)"] = df_display["Run (ET)"].apply(lambda dt: _format_datetime_string(dt, tz_label))
        df_display["Run (UTC)"] = df_display["Run (UTC)"].apply(lambda dt: _format_datetime_string(dt, "UTC"))
        df_display["Local Close"] = df_display.apply(
            lambda row: _format_datetime_string(row["Local Close"], row.get("Local TZ")), axis=1
        )
        df_display["Time Remaining"] = df_display["Time Remaining"].apply(_format_timedelta)

        col1, col2 = st.columns(2)
        with col1:
            region_filter = st.selectbox("Filter by Region", ["All", "Asia-Pacific", "Europe", "Americas"])
        with col2:
            search_term = st.text_input("Search Exchange")

        filtered_df = df_display.copy()
        if region_filter != "All":
            filtered_df = filtered_df[filtered_df["Region"] == region_filter]
        if search_term:
            mask = filtered_df["Exchange"].str.contains(search_term, case=False, na=False) | filtered_df[
                "Symbol"
            ].str.contains(search_term, case=False, na=False)
            filtered_df = filtered_df[mask]

        # Show one-time result from a previous "Run now"
        run_result = st.session_state.pop("run_now_result", None)
        if run_result:
            if run_result.get("success"):
                st.success(
                    f"Daily job for **{run_result.get('exchange', '')}** completed in "
                    f"{run_result.get('duration', 0):.1f}s."
                )
            else:
                st.error(
                    f"Daily job for **{run_result.get('exchange', '')}** failed: {run_result.get('error', 'Unknown error')}"
                )

        # Execute "Run now" if a button was clicked
        run_exchange = st.session_state.pop("run_now_exchange", None)
        if run_exchange:
            with st.spinner(f"Running daily job for **{run_exchange}**… This may take several minutes."):
                try:
                    result = run_daily_job(run_exchange)
                    if result:
                        st.session_state["run_now_result"] = {
                            "success": True,
                            "exchange": run_exchange,
                            "duration": result.get("duration", 0),
                        }
                    else:
                        st.session_state["run_now_result"] = {
                            "success": False,
                            "exchange": run_exchange,
                            "error": "Job skipped (already running or lock not acquired).",
                        }
                except Exception as exc:
                    st.session_state["run_now_result"] = {
                        "success": False,
                        "exchange": run_exchange,
                        "error": str(exc),
                    }
            _trigger_rerun()

        columns = [
            "Exchange",
            "Symbol",
            "Region",
            "Run (ET)",
            "Run (UTC)",
            "Local Close",
            "Local TZ",
            "Time Remaining",
            "Last Run (ET)",
        ]
        # Header row
        header_cols = st.columns([2, 1, 1, 2, 2, 2, 1, 1, 2, 1])
        for i, col_name in enumerate(columns + ["Run now"]):
            with header_cols[i]:
                st.markdown(f"**{col_name}**")
        # Data rows with Run now button per exchange
        for _, row in filtered_df.iterrows():
            row_cols = st.columns([2, 1, 1, 2, 2, 2, 1, 1, 2, 1])
            for i, col_name in enumerate(columns):
                with row_cols[i]:
                    st.write(str(row.get(col_name, "")))
            with row_cols[-1]:
                symbol = row["Symbol"]
                st.button(
                    "Run now",
                    key=f"run_now_{symbol}",
                    on_click=_set_run_now_exchange,
                    args=(symbol,),
                    type="secondary",
                )

    st.markdown("---")
    st.subheader("ℹ️ How It Works")
    st.info(
        f"""
        **Automatic Alert Checking with Calendar Awareness**
        - Times displayed in **{tz_label}** automatically adjust for DST.
        - Daily jobs run after each exchange closes using calendar-derived schedules.
        - Weekly jobs execute on the exchange's final session of the week.
        - Hourly checks respect session open/close boundaries.
        """
    )

    if st.checkbox("🔄 Auto-refresh (every 60 seconds)", value=False):
        time.sleep(60)
        _trigger_rerun()


if __name__ == "__main__":
    main()

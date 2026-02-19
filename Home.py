#!/usr/bin/env python3
"""Streamlit entrypoint for Stock Alert App."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Load .env from project root first so FMP_API_KEY etc. are available regardless of cwd
from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)

import pandas as pd
import streamlit as st

from src.data_access.alert_repository import list_alerts, refresh_alert_cache
from src.data_access.document_store import load_document
from src.data_access.metadata_repository import fetch_stock_metadata_map
from src.data_access.redis_support import build_key, get_json

# Scheduler status keys match document_store cache keys (mode-specific)
DAILY_SCHEDULER_STATUS_KEY = build_key("document:scheduler_status_daily")
WEEKLY_SCHEDULER_STATUS_KEY = build_key("document:scheduler_status_weekly")
HOURLY_SCHEDULER_STATUS_KEY = build_key("document:scheduler_status_hourly")


def _ensure_event_loop() -> asyncio.AbstractEventLoop:
    """Return a running event loop (required by legacy async helpers)."""
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def _load_scheduler_status_by_mode(mode: str, redis_key: str, doc_key: str) -> Optional[dict]:
    """
    Load scheduler status for a specific mode.

    Args:
        mode: Scheduler mode ('daily', 'weekly', 'hourly')
        redis_key: Redis cache key
        doc_key: Document store key

    Returns:
        Status dictionary if found, None otherwise
    """
    # Try Redis cache first (document_store writes here)
    snapshot = get_json(redis_key)
    if isinstance(snapshot, dict):
        # document_store wraps payload in {"payload": ..., "cached_at": ...}
        payload = snapshot.get("payload") if "payload" in snapshot else snapshot
        if payload:
            return payload

    # Fallback to document_store (which queries PostgreSQL)
    try:
        document = load_document(doc_key, default=None, fallback_path=f"scheduler_status_{mode}.json")
        if isinstance(document, dict):
            return document
    except Exception:
        pass

    # Final fallback to legacy JSON file
    status_path = Path(f"scheduler_status_{mode}.json")
    if status_path.exists():
        try:
            data = json.loads(status_path.read_text())
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return None


def _load_auto_scheduler_status() -> Optional[dict]:
    """
    Load combined daily/weekly scheduler status.

    Tries to load both daily and weekly scheduler status and returns
    the most recently updated one, or merges them if both exist.
    """
    daily_status = _load_scheduler_status_by_mode("daily", DAILY_SCHEDULER_STATUS_KEY, "scheduler_status_daily")
    weekly_status = _load_scheduler_status_by_mode("weekly", WEEKLY_SCHEDULER_STATUS_KEY, "scheduler_status_weekly")

    # If both exist, return the most recently updated
    if daily_status and weekly_status:
        daily_time = daily_status.get("heartbeat", "")
        weekly_time = weekly_status.get("heartbeat", "")
        # Return the one with the most recent heartbeat
        if weekly_time > daily_time:
            return weekly_status
        return daily_status

    # Return whichever one exists
    return daily_status or weekly_status


def _load_hourly_scheduler_status() -> Optional[dict]:
    """Load hourly scheduler status."""
    return _load_scheduler_status_by_mode("hourly", HOURLY_SCHEDULER_STATUS_KEY, "scheduler_status_hourly")


@st.cache_data(ttl=60)
def load_alert_dataframe() -> pd.DataFrame:
    """Return all alerts as a DataFrame for easier aggregation."""
    refresh_alert_cache()
    alerts = list_alerts()
    if not alerts:
        return pd.DataFrame()
    df = pd.DataFrame(alerts)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    if "conditions" in df.columns:
        df["conditions_text"] = df["conditions"].apply(_format_conditions)
    else:
        df["conditions_text"] = ""
    return df


def _format_conditions(value: object) -> str:
    if not value:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("conditions") or item.get("expression")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "; ".join(parts)
    return str(value)


@st.cache_data(ttl=300)
def load_exchange_symbol_counts() -> Dict[str, int]:
    metadata = fetch_stock_metadata_map()
    counts: Dict[str, int] = defaultdict(int)
    for info in metadata.values():
        exchange = info.get("exchange")
        if exchange:
            counts[exchange] += 1
    return counts


def format_timestamp(iso_timestamp: Optional[str]) -> str:
    """Format ISO timestamp to readable format."""
    if not iso_timestamp:
        return "Never"
    try:
        dt = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00"))
        # Convert to local time
        local_dt = dt.astimezone()
        # Format as "Jan 15, 2:30 PM" or "2 hours ago" for recent times
        now = datetime.now(local_dt.tzinfo)
        diff = now - local_dt

        if diff.total_seconds() < 60:
            return "Just now"
        elif diff.total_seconds() < 3600:
            minutes = int(diff.total_seconds() / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif diff.total_seconds() < 86400:
            hours = int(diff.total_seconds() / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        else:
            return local_dt.strftime("%b %d, %I:%M %p")
    except Exception:
        return iso_timestamp


def format_scheduler_status_detailed(status: Optional[dict]) -> dict:
    """
    Format scheduler status into a detailed display dict.

    Returns:
        Dict with 'state', 'emoji', 'details' keys for display
    """
    if not isinstance(status, dict):
        return {
            "state": "Unknown",
            "emoji": "❓",
            "details": "No status information available",
            "is_healthy": False
        }

    state = status.get("status", "unknown").lower()
    mode = status.get("mode", "")
    heartbeat = status.get("heartbeat")
    last_run = status.get("last_run")
    last_result = status.get("last_result")
    current_job = status.get("current_job")

    # Determine emoji and health based on state
    emoji_map = {
        "running": "🟢",
        "updating": "🔄",
        "stopped": "🔴",
        "error": "❌",
        "paused": "⏸️"
    }
    emoji = emoji_map.get(state, "⚪")
    is_healthy = state in ("running", "updating")

    # Build details list
    details = []

    # Heartbeat info
    if heartbeat:
        heartbeat_str = format_timestamp(heartbeat)
        details.append(f"**Last heartbeat:** {heartbeat_str}")

    # Current job info
    if current_job and isinstance(current_job, dict):
        job_name = current_job.get("exchange", current_job.get("name", "Unknown"))
        details.append(f"**Current job:** {job_name}")

    # Last run info
    if last_run and isinstance(last_run, dict):
        last_run_time = last_run.get("timestamp") or last_run.get("time")
        if last_run_time:
            details.append(f"**Last run:** {format_timestamp(last_run_time)}")

        exchange = last_run.get("exchange")
        if exchange:
            details.append(f"**Last exchange:** {exchange}")

    # Last result stats
    if last_result and isinstance(last_result, dict):
        price_stats = last_result.get("price_stats", {})
        alert_stats = last_result.get("alert_stats", {})

        if price_stats:
            updated = price_stats.get("updated", 0)
            failed = price_stats.get("failed", 0)
            skipped = price_stats.get("skipped", 0)
            details.append(f"**Last update:** {updated:,} updated, {failed:,} failed, {skipped:,} skipped")

        if alert_stats:
            triggered = alert_stats.get("triggered", 0)
            total = alert_stats.get("total", 0)
            if total > 0:
                details.append(f"**Alerts:** {triggered:,} triggered of {total:,} checked")

    # Mode info
    if mode:
        details.append(f"**Mode:** {mode.capitalize()}")

    details_text = "\n\n".join(details) if details else "No detailed information available"

    return {
        "state": state.capitalize(),
        "emoji": emoji,
        "details": details_text,
        "is_healthy": is_healthy
    }


def summarize_alerts(df: pd.DataFrame) -> Dict[str, int]:
    total = len(df)
    return {
        "total": total,
        "active": total,  # alerts table only stores active alerts
        "inactive": 0,
    }


def build_alert_download(df: pd.DataFrame) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    if df.empty:
        writer.writerow(["No alerts"])
    else:
        writer.writerow(df.columns)
        for row in df.itertuples(index=False):
            writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------


def render_alert_summary(df: pd.DataFrame, filtered_count: int) -> None:
    stats = summarize_alerts(df)
    st.subheader("📊 Alert Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Alerts", f"{stats['total']:,}")
    col2.metric("Active Alerts", f"{stats['active']:,}")
    col3.metric("Inactive Alerts", f"{stats['inactive']:,}")
    col4.metric("Matches Current Filters", f"{filtered_count:,}")
    st.caption("All alerts stored in the database are active; inactive alerts are kept in audit logs.")


def render_scheduler_cards() -> None:
    """Render scheduler status cards with detailed information."""
    auto_status = _load_auto_scheduler_status()
    hourly_status = _load_hourly_scheduler_status()
    auto_formatted = format_scheduler_status_detailed(auto_status)
    hourly_formatted = format_scheduler_status_detailed(hourly_status)

    st.subheader("⏱️ Scheduler Status")

    col1, col2 = st.columns(2)

    # Daily/Weekly Scheduler Card
    with col1:
        st.markdown(f"### {auto_formatted['emoji']} Daily/Weekly Scheduler")

        # Status badge
        if auto_formatted['is_healthy']:
            st.success(f"**Status:** {auto_formatted['state']}")
        elif auto_formatted['state'].lower() == 'stopped':
            st.warning(f"**Status:** {auto_formatted['state']}")
        else:
            st.error(f"**Status:** {auto_formatted['state']}")

        # Details in expander
        with st.expander("📊 Details", expanded=False):
            st.markdown(auto_formatted['details'])

    # Hourly Scheduler Card
    with col2:
        st.markdown(f"### {hourly_formatted['emoji']} Hourly Scheduler")

        # Status badge
        if hourly_formatted['is_healthy']:
            st.success(f"**Status:** {hourly_formatted['state']}")
        elif hourly_formatted['state'].lower() == 'stopped':
            st.warning(f"**Status:** {hourly_formatted['state']}")
        else:
            st.error(f"**Status:** {hourly_formatted['state']}")

        # Details in expander
        with st.expander("📊 Details", expanded=False):
            st.markdown(hourly_formatted['details'])


def render_recent_activity(df: pd.DataFrame) -> None:
    st.subheader("🕒 Recent Activity")
    if df.empty or "created_at" not in df.columns:
        st.info("No alert activity available.")
        return
    recent = df.sort_values("created_at", ascending=False).head(20)
    st.dataframe(
        recent[["alert_id", "ticker", "exchange", "active", "created_at"]].rename(
            columns={"alert_id": "id", "active": "is_active"}
        ),
        use_container_width=True,
    )


def render_alert_list(df: pd.DataFrame, total_count: int) -> None:
    st.subheader("📋 All Alerts")
    if df.empty:
        st.info("No alerts found.")
        return

    st.caption(f"Showing {len(df):,} of {total_count:,} alerts.")
    display_columns = [
        "alert_id",
        "name",
        "ticker",
        "exchange",
        "timeframe",
        "action",
        "ratio",
        "conditions_text",
    ]
    missing_cols = [col for col in display_columns if col not in df.columns]
    extended_df = df.copy()
    for col in missing_cols:
        extended_df[col] = ""
    view = extended_df[display_columns].rename(
        columns={
            "alert_id": "id",
            "conditions_text": "conditions",
        }
    )
    st.dataframe(view, use_container_width=True)
    csv_data = build_alert_download(view)
    st.download_button(
        label="📥 Download alerts CSV",
        data=csv_data,
        file_name=f"alerts_{datetime.utcnow():%Y%m%d_%H%M%S}.csv",
        mime="text/csv",
    )


def apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("🔎 Alert Filters")
    search = st.sidebar.text_input("Search ticker or company")

    filtered = df.copy()
    for column, default in [
        ("exchange", "Unknown"),
        ("timeframe", "Unknown"),
        ("action", "Unknown"),
        ("ratio", "No"),
    ]:
        if column not in filtered.columns:
            filtered[column] = default
        else:
            filtered[column] = filtered[column].fillna(default)

    exchange_options = sorted(filtered["exchange"].dropna().unique())
    selected_exchanges = st.sidebar.multiselect("Exchanges", exchange_options)

    timeframe_options = sorted(filtered["timeframe"].dropna().unique())
    selected_timeframes = st.sidebar.multiselect("Timeframes", timeframe_options)

    action_options = sorted(filtered["action"].dropna().unique())
    selected_actions = st.sidebar.multiselect("Actions", action_options)

    ratio_choice = st.sidebar.selectbox("Ratio Alerts", ["All", "Yes", "No"])

    result = filtered
    if search:
        token = search.strip().lower()
        contains = (
            result["ticker"].astype(str).str.lower().str.contains(token)
            | result["name"].astype(str).str.lower().str.contains(token)
            | result["exchange"].astype(str).str.lower().str.contains(token)
            | result["conditions_text"].astype(str).str.lower().str.contains(token)
        )
        result = result[contains]
    if selected_exchanges:
        result = result[result["exchange"].isin(selected_exchanges)]
    if selected_timeframes:
        result = result[result["timeframe"].isin(selected_timeframes)]
    if selected_actions:
        result = result[result["action"].isin(selected_actions)]
    if ratio_choice != "All":
        result = result[result["ratio"] == ratio_choice]
    st.sidebar.caption(f"{len(result):,} alerts match filters.")
    return result


def main() -> None:
    st.set_page_config(page_title="Stock Alert Dashboard", layout="wide", page_icon="📈")
    st.title("📈 Stock Alert Dashboard")
    st.caption("Live overview of alert coverage, scheduler health, and recent activity.")

    alert_df = load_alert_dataframe()
    filtered_alerts = apply_sidebar_filters(alert_df)

    render_scheduler_cards()
    render_alert_summary(alert_df, len(filtered_alerts))
    render_recent_activity(filtered_alerts)
    render_alert_list(filtered_alerts, len(alert_df))


if __name__ == "__main__":
    _ensure_event_loop()
    main()

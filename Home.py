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

import pandas as pd
import streamlit as st

from src.data_access.alert_repository import list_alerts, refresh_alert_cache
from src.data_access.document_store import load_document
from src.data_access.metadata_repository import fetch_stock_metadata_map
from src.data_access.redis_support import build_key, get_json

AUTO_SCHEDULER_STATUS_KEY = build_key("auto_scheduler_status")
HOURLY_STATUS_KEY = build_key("hourly_scheduler_status")


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


def _load_auto_scheduler_status() -> Optional[dict]:
    snapshot = get_json(AUTO_SCHEDULER_STATUS_KEY)
    if isinstance(snapshot, dict):
        return snapshot
    try:
        document = load_document("scheduler_status", default=None, fallback_path="scheduler_status.json")
        if isinstance(document, dict):
            return document
    except Exception:
        pass
    status_path = Path("scheduler_status.json")
    if status_path.exists():
        try:
            data = json.loads(status_path.read_text())
            if isinstance(data, dict):
                return data
        except Exception:
            return None
    return None


def _load_hourly_scheduler_status() -> Optional[dict]:
    snapshot = get_json(HOURLY_STATUS_KEY)
    if isinstance(snapshot, dict):
        return snapshot
    try:
        document = load_document("hourly_scheduler_status", default=None, fallback_path="hourly_scheduler_status.json")
        if isinstance(document, dict):
            return document
    except Exception:
        pass
    status_path = Path("hourly_scheduler_status.json")
    if status_path.exists():
        try:
            data = json.loads(status_path.read_text())
            if isinstance(data, dict):
                return data
        except Exception:
            return None
    return None


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


def format_scheduler_status(status: Optional[dict]) -> Tuple[str, str]:
    if not isinstance(status, dict):
        return "unknown", "No status available"
    state = status.get("status", "unknown")
    last_run = status.get("last_run")
    next_run = status.get("next_run")
    lines = [f"Status: {state}"]
    if last_run:
        lines.append(f"Last run: {last_run}")
    if next_run:
        lines.append(f"Next run: {next_run}")
    return state, " | ".join(lines)


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
    auto_status = _load_auto_scheduler_status()
    hourly_status = _load_hourly_scheduler_status()
    auto_state, auto_details = format_scheduler_status(auto_status)
    hourly_state, hourly_details = format_scheduler_status(hourly_status)

    st.subheader("⏱️ Scheduler Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Daily/Weekly Scheduler",
            "Running" if auto_state in {"running", "updating"} else auto_state.title(),
        )
        st.caption(auto_details)
    with col2:
        st.metric(
            "Hourly Scheduler",
            "Running" if hourly_state in {"running", "updating"} else hourly_state.title(),
        )
        st.caption(hourly_details)


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

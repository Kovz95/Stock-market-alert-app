"""
Streamlit page showing calendar-aware market hours in Eastern Time.

Uses the calendar_adapter helpers so DST and holiday calendars are respected.
All times are shown in ET (EST/EDT automatically), formatted in 12-hour time.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import pandas as pd
import pytz
import streamlit as st

from src.services.calendar_adapter import (
    EXCHANGE_SCHEDULES,
    get_calendar_timezone,
    get_session_bounds,
    is_exchange_open,
)


st.set_page_config(
    page_title="Market Hours (ET)",
    page_icon="⏰",
    layout="wide",
)

st.title("⏰ Market Hours (Eastern Time)")
st.caption("Calendar-aware open/close times in ET (EST/EDT automatically).")


def _format_et(ts: pd.Timestamp) -> str:
    """Return a friendly ET string with day + 12-hour clock."""
    ts_et = ts.tz_convert(pytz.timezone("America/New_York"))
    day = ts_et.strftime("%a %b %d")
    time = ts_et.strftime("%I:%M %p").lstrip("0")
    return f"{day} {time} {ts_et.strftime('%Z')}"


@st.cache_data(ttl=300, show_spinner=False)
def load_market_hours(reference_utc: pd.Timestamp) -> pd.DataFrame:
    """Build a table of the next session open/close for each exchange."""
    rows: List[Dict[str, object]] = []
    et = pytz.timezone("America/New_York")
    today_et = reference_utc.tz_convert(et).date()

    for code, config in EXCHANGE_SCHEDULES.items():
        try:
            open_ts, close_ts = get_session_bounds(code, reference_utc, next_if_closed=True)
        except Exception as exc:
            rows.append(
                {
                    "Exchange": code,
                    "Name": config.get("name", code),
                    "Status": "Unknown",
                    "Opens": pd.NaT,
                    "Closes": pd.NaT,
                    "Day Offset": None,
                    "Timezone": get_calendar_timezone(code) or "",
                    "Notes": config.get("notes", ""),
                    "Error": str(exc),
                }
            )
            continue

        try:
            status = "Open" if is_exchange_open(code, reference_utc) else "Closed"
        except Exception:
            status = "Unknown"

        open_et = pd.Timestamp(open_ts).tz_convert(et)
        close_et = pd.Timestamp(close_ts).tz_convert(et)
        day_offset = (open_et.date() - today_et).days

        rows.append(
            {
                "Exchange": code,
                "Name": config.get("name", code),
                "Status": status,
                "Opens": open_et,
                "Closes": close_et,
                "Day Offset": day_offset,
                "Timezone": get_calendar_timezone(code) or "",
                "Notes": config.get("notes", ""),
                "Error": "",
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Opens (ET)"] = df["Opens"].apply(lambda x: _format_et(pd.Timestamp(x)) if pd.notna(x) else "N/A")
    df["Closes (ET)"] = df["Closes"].apply(lambda x: _format_et(pd.Timestamp(x)) if pd.notna(x) else "N/A")
    return df


# Reference time (UTC & ET)
now_utc = pd.Timestamp.now(tz="UTC")
now_et = now_utc.tz_convert(pytz.timezone("America/New_York"))
st.metric("Current Time (ET)", now_et.strftime("%a %b %d %I:%M %p %Z"))

df = load_market_hours(now_utc)

if df.empty:
    st.error("Could not load market hours. Please try again.")
    st.stop()

# Filters
col_search, col_status, col_sort = st.columns([2, 1, 1])
with col_search:
    query = st.text_input("Search exchange/name", placeholder="e.g., NASDAQ, LONDON, XETRA").strip().lower()
with col_status:
    status_filter = st.multiselect("Status", options=["Open", "Closed", "Unknown"], default=["Open", "Closed"])
with col_sort:
    sort_choice = st.selectbox("Sort by", ["Opens (ET)", "Closes (ET)", "Name"])

filtered = df.copy()
if query:
    filtered = filtered[
        filtered["Exchange"].str.lower().str.contains(query)
        | filtered["Name"].str.lower().str.contains(query)
    ]

if status_filter:
    filtered = filtered[filtered["Status"].isin(status_filter)]

sort_col = "Opens" if sort_choice == "Opens (ET)" else ("Closes" if sort_choice == "Closes (ET)" else "Name")
filtered = filtered.sort_values(sort_col)

# Display
display_cols = [
    "Exchange",
    "Name",
    "Status",
    "Opens (ET)",
    "Closes (ET)",
    "Timezone",
    "Notes",
]

st.dataframe(
    filtered[display_cols],
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Times are converted to Eastern Time with DST awareness. "
    "Values come from the shared exchange calendar, so holidays and weekend closures are reflected automatically."
)

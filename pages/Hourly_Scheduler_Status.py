"""
Streamlit page for Hourly Data Scheduler Status
Shows the current status, next run time, and statistics with live updates
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import psutil
import pytz
import streamlit as st

from src.services.hourly_data_scheduler import is_exchange_open
from src.data_access.redis_support import build_key, get_json
from src.data_access.metadata_repository import fetch_stock_metadata_map

TIMEZONE_OPTIONS = {
    "US/Eastern": "Eastern Time (EST/EDT)",
    "US/Pacific": "Western Time (PST/PDT)",
}


def parse_utc_timestamp(raw_value: Optional[str]) -> Optional[datetime]:
    """Parse scheduler timestamps (stored in UTC) into aware datetimes."""
    if not raw_value or not isinstance(raw_value, str):
        return None
    value = raw_value.strip()
    if not value or "Skipped" in value:
        return None
    # Normalize common suffixes
    if value.endswith("Z"):
        value = value[:-1]
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(value, fmt)
            return pytz.UTC.localize(dt)
        except ValueError:
            continue
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            return pytz.UTC.localize(dt)
        return dt.astimezone(pytz.UTC)
    except ValueError:
        return None


def format_local_timestamp(raw_value: Optional[str], tz: pytz.BaseTzInfo) -> Optional[str]:
    """Convert stored UTC timestamp strings to the selected timezone for display."""
    dt = parse_utc_timestamp(raw_value)
    if not dt:
        return None
    local_dt = dt.astimezone(tz)
    return local_dt.strftime("%b %d %I:%M %p %Z")


def convert_utc_clock_string(time_str: str, tz: pytz.BaseTzInfo, reference_date: datetime) -> str:
    """
    Convert strings like '13:30' that represent UTC clock times into the selected timezone.
    """
    try:
        hour, minute = map(int, time_str.split(":"))
    except ValueError:
        return time_str
    utc_dt = datetime(reference_date.year, reference_date.month, reference_date.day, hour, minute, tzinfo=pytz.UTC)
    return utc_dt.astimezone(tz).strftime("%I:%M %p %Z")


def display_timestamp_value(raw_value: Optional[str], tz: pytz.BaseTzInfo, default: str = "N/A") -> str:
    """Utility to convert timestamps while keeping fallback messaging."""
    formatted = format_local_timestamp(raw_value, tz)
    if formatted:
        return formatted
    if raw_value:
        return raw_value
    return default

st.set_page_config(
    page_title="Hourly Scheduler Status",
    page_icon="⏰",
    layout="wide"
)

# Initialize session state for smooth updates
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

if 'last_run_display' not in st.session_state:
    st.session_state.last_run_display = None

st.title("⏰ Hourly Data Scheduler Status")
st.markdown("🟢 **Live monitoring** - Updates automatically without refresh")

selected_tz_key = st.selectbox(
    "Display times in",
    options=list(TIMEZONE_OPTIONS.keys()),
    format_func=lambda tz: TIMEZONE_OPTIONS[tz],
    key="hourly_status_timezone",
)
SELECTED_TZ = pytz.timezone(selected_tz_key)
SELECTED_TZ_ABBREV = datetime.now(SELECTED_TZ).tzname() or SELECTED_TZ.zone
st.caption(
    f"All times on this page are shown in {TIMEZONE_OPTIONS[selected_tz_key]} "
    f"({SELECTED_TZ_ABBREV}); backend scheduling continues to run in UTC."
)

# Files to monitor - use LOG_DIR environment variable if set
LOG_DIR = Path(os.getenv("LOG_DIR", "."))
LOCK_FILE = Path("hourly_scheduler.lock")
REDIS_STATUS_KEY = build_key("hourly_scheduler_status")
LOG_FILE = LOG_DIR / "hourly_data_scheduler.log"
STATUS_FILE = Path("hourly_scheduler_status.json")

@st.cache_data(show_spinner=False)
def load_stock_database_data():
    """Return the master ticker database used for hourly scheduling."""
    try:
        return fetch_stock_metadata_map()
    except Exception:
        return {}

@st.cache_data(show_spinner=False)
def get_exchange_symbol_counts():
    """Count symbols per exchange for schedule summaries."""
    data = load_stock_database_data()
    counts = {}
    for info in data.values():
        exchange = info.get('exchange')
        if exchange:
            counts[exchange] = counts.get(exchange, 0) + 1
    return counts

@st.cache_data(ttl=3, show_spinner=False)
def load_status():
    """Load scheduler status from Redis."""
    snapshot = get_json(REDIS_STATUS_KEY)
    if isinstance(snapshot, dict):
        return snapshot
    try:
        if STATUS_FILE.exists():
            payload = json.loads(STATUS_FILE.read_text())
            if isinstance(payload, dict):
                return payload
    except Exception:
        return None
    return None


def load_lock():
    """Load scheduler lock file (best-effort)."""
    try:
        if LOCK_FILE.exists():
            return json.loads(LOCK_FILE.read_text())
        return None
    except Exception:
        return None


def status_indicates_running(status: Optional[dict]) -> bool:
    if not isinstance(status, dict):
        return False
    return status.get("status") in {"running", "updating", "starting"}


def status_pid_alive(status: Optional[dict]) -> bool:
    if not isinstance(status, dict):
        return False
    pid = status.get("pid")
    return bool(pid and psutil.pid_exists(pid))


def lock_pid_alive(lock: Optional[dict]) -> bool:
    """Return True when the lock file contains a live process."""
    if not isinstance(lock, dict):
        return False
    pid = lock.get("pid")
    return bool(pid and psutil.pid_exists(int(pid)))

def start_scheduler_process():
    """Start the hourly scheduler as a separate background process (console-less)"""
    try:
        # If the lock already points to a running process, treat as success
        existing_lock = load_lock()
        if lock_pid_alive(existing_lock):
            return True

        # Get the Python executable path
        python_path = sys.executable
        if not python_path:
            # Fallback to common Python paths on Windows
            python_path = r"C:\Users\NickK\AppData\Local\Programs\Python\Python313\python.exe"

        # Start the scheduler in a new process with proper working directory
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Use CREATE_NO_WINDOW to run without console window
        process = subprocess.Popen(
            [python_path, os.path.join(parent_dir, "hourly_data_scheduler.py")],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            cwd=parent_dir,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        )

        # Answer 'n' to the initial prompt automatically
        try:
            process.stdin.write(b'n\n')
            process.stdin.flush()
        except:
            pass

        # Give it time to initialize
        time.sleep(3)
        load_status.clear()
        status_snapshot = load_status()
        if (
            status_pid_alive(status_snapshot)
            or status_indicates_running(status_snapshot)
            or lock_pid_alive(load_lock())
        ):
            return True
        return False
    except Exception as e:
        st.error(f"Failed to start hourly scheduler: {e}")
        return False

def stop_scheduler_process():
    """Stop the hourly scheduler process"""
    try:
        # Kill all Python processes running hourly_data_scheduler.py
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info.get('cmdline')
                if cmdline and any('hourly_data_scheduler.py' in arg for arg in cmdline):
                    proc.terminate()  # Try graceful termination first
                    time.sleep(0.5)
                    if proc.is_running():
                        proc.kill()  # Force kill if still running
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Remove lock file if it exists
        LOCK_FILE.unlink(missing_ok=True)

        time.sleep(1)  # Give it a moment to fully stop
        load_status.clear()
        return True
    except Exception as e:
        st.error(f"Failed to stop hourly scheduler: {e}")
        return False

def get_log_tail(num_lines=50):
    """Get last N lines from log file"""
    try:
        if not LOG_FILE.exists():
            return "Log file not found"

        with LOG_FILE.open('rb') as fh:
            fh.seek(0, os.SEEK_END)
            file_size = fh.tell()
            buffer = bytearray()
            block_size = 4096
            while len(buffer.splitlines()) <= num_lines and file_size > 0:
                read_size = min(block_size, file_size)
                file_size -= read_size
                fh.seek(file_size)
                buffer = fh.read(read_size) + buffer

        lines = buffer.splitlines()[-num_lines:]
        return b"\n".join(lines).decode("utf-8", errors="ignore")
    except Exception as exc:
        return f"Error reading log: {exc}"

st.markdown("---")

# Live monitoring section with auto-refresh fragment
@st.fragment(run_every=3)
def live_status_monitor():
    """Auto-refreshing fragment for live status updates"""
    # Load current status
    status = load_status()
    lock = load_lock()
    state = status.get("status") if isinstance(status, dict) else None
    pid_alive = status_pid_alive(status) or lock_pid_alive(lock)
    actively_running = status_indicates_running(status) or (pid_alive and state not in {"stopped", "error"})
    is_idle = state == "idle" and pid_alive

    if status:
        last_run_value = status.get('last_run')
        if last_run_value:
            st.session_state.last_run_display = last_run_value
        elif status.get('status') in {'stopped', 'error'}:
            st.session_state.last_run_display = None

    # Status Overview
    st.subheader("📊 Status Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if actively_running:
            st.metric("Scheduler Status", "🟢 Running", delta=None)
        elif is_idle:
            st.metric("Scheduler Status", "🟡 Idle", delta=None)
        else:
            st.metric("Scheduler Status", "🔴 Stopped", delta=None)

    with col2:
        if status and status.get('next_run'):
            st.metric("Next Run", display_timestamp_value(status.get('next_run'), SELECTED_TZ))
        else:
            st.metric("Next Run", "N/A")

    with col3:
        display_last_run = st.session_state.get('last_run_display')
        if display_last_run:
            if display_last_run.startswith('Skipped - '):
                st.metric("Last Run", "Skipped")
                st.caption(f"Reason: {display_last_run.replace('Skipped - ', '')}")
            else:
                formatted_last_run = format_local_timestamp(display_last_run, SELECTED_TZ)
                st.metric("Last Run", formatted_last_run or display_last_run)
        else:
            st.metric("Last Run", "Waiting...")

    with col4:
        pid_display = status.get('pid') if isinstance(status, dict) else None
        if not pid_display and isinstance(lock, dict) and lock_pid_alive(lock):
            pid_display = lock.get('pid')
        st.metric("Process ID", pid_display or "N/A")

    st.markdown("### Update Run Summary")
    exchange_counts = get_exchange_symbol_counts()
    if not exchange_counts:
        st.info("Symbol counts could not be loaded from metadata table. Showing schedule with zero counts.")

    schedule_blocks = [
        {
            "order": 1,
            "time": ":05",
            "label": "On-the-hour candles (:00 close)",
            "exchanges": [
                "TOKYO", "TAIWAN", "ASX", "SINGAPORE", "MALAYSIA", "THAILAND", "INDONESIA",
                "LONDON", "XETRA", "EURONEXT AMSTERDAM", "EURONEXT BRUSSELS", "EURONEXT LISBON",
                "EURONEXT DUBLIN", "MILAN", "SPAIN", "SIX SWISS", "SIX",
                "OMX NORDIC STOCKHOLM", "OMX NORDIC COPENHAGEN", "OMX NORDIC HELSINKI",
                "OSLO", "WARSAW", "VIENNA", "PRAGUE", "BUDAPEST", "ISTANBUL",
                "SAO PAULO", "JSE"
            ]
        },
        {
            "order": 2,
            "time": ":20",
            "label": "Quarter-hour candles (:15 close)",
            "exchanges": ["BSE INDIA", "NSE INDIA"]
        },
        {
            "order": 3,
            "time": ":35",
            "label": "Half-hour candles (:30 close)",
            "exchanges": [
                "NASDAQ", "NYSE", "NYSE ARCA", "NYSE AMERICAN", "CBOE BZX",
                "TORONTO", "MEXICO", "SANTIAGO", "BUENOS AIRES", "COLOMBIA",
                "HONG KONG", "EURONEXT PARIS", "ATHENS", "OMX NORDIC ICELAND"
            ]
        }
    ]

    now_utc = datetime.now(pytz.UTC)

    def format_time_delta(delta: timedelta) -> str:
        total_seconds = int(delta.total_seconds())
        if total_seconds <= 0:
            return "Now"
        hours, remainder = divmod(total_seconds, 3600)
        minutes, _ = divmod(remainder, 60)
        parts = []
        if hours:
            parts.append(f"{hours}h")
        parts.append(f"{minutes}m")
        return " ".join(parts)

    summary_rows = []
    next_run_column_label = f"Next Run ({SELECTED_TZ_ABBREV})"

    for block in schedule_blocks:
        exchanges = block["exchanges"]
        run_minute = block["time"]
        minute_value = int(run_minute.replace(":", ""))

        minutes_ahead = (minute_value - now_utc.minute) % 60
        if minutes_ahead == 0 and now_utc.second > 0:
            minutes_ahead = 60
        next_run_dt = (now_utc + timedelta(minutes=minutes_ahead)).replace(second=0, microsecond=0)
        time_until = format_time_delta(next_run_dt - now_utc)

        open_exchanges = [ex for ex in exchanges if is_exchange_open(ex, now_utc)]
        open_count = len(open_exchanges)
        open_symbols = sum(exchange_counts.get(ex, 0) for ex in open_exchanges)
        open_display = ", ".join(open_exchanges) if open_exchanges else "None (markets closed)"

        summary_rows.append({
            "Order": block["order"],
            "Run Minute": run_minute,
            next_run_column_label: next_run_dt.astimezone(SELECTED_TZ).strftime("%Y-%m-%d %I:%M %p %Z"),
            "Starts In": time_until,
            "Open Exchanges": open_display,
            "Open Exchange Count": open_count,
            "Symbols": open_symbols
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("Order")
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.caption(
        f"Live view shows only exchanges currently open for each run window. Times reflect {TIMEZONE_OPTIONS[selected_tz_key]}."
        " Symbol totals come from the stock_metadata table."
    )


    st.markdown("---")

    # Exchange Update Schedule
    st.subheader("🌍 Exchange Update Schedule")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**:05 Update (Candles at :00)**")
        st.caption("Exchanges with hourly candles at :00")
        exchanges_at_05 = [
            "🌏 Asia: Tokyo, Taiwan, ASX, Singapore, Malaysia, Thailand, Indonesia",
            "🇬🇧🇪🇺 Europe West: London, XETRA, Amsterdam, Brussels, Lisbon, Dublin",
            "🇮🇹🇪🇸🇨🇭 Europe South: Milan, Spain, Swiss",
            "🇸🇪🇩🇰🇫🇮🇳🇴 Europe Nordic: Stockholm, Copenhagen, Helsinki, Oslo",
            "🇵🇱🇦🇹🇨🇿🇭🇺🇹🇷 Europe East: Warsaw, Vienna, Prague, Budapest, Istanbul",
            "🌎 Americas: Sao Paulo",
            "🌍 Africa: JSE (South Africa)"
        ]
        for ex in exchanges_at_05:
            st.text(ex)

    with col2:
        st.markdown("**:20 Update (Candles at :15)**")
        st.caption("India exchanges with hourly candles at :15")
        st.text("🇮🇳 India: BSE, NSE")
        st.caption("")

        st.markdown("**:35 Update (Candles at :30)**")
        st.caption("US, Canada, Latin America, Hong Kong, Paris, Athens, Iceland")
        exchanges_at_35 = [
            "🇺🇸 US Markets: NASDAQ, NYSE, NYSE ARCA, CBOE BZX",
            "🇨🇦 Canada: TORONTO",
            "🌎 Latin America: Mexico, Santiago, Buenos Aires, Colombia",
            "🇭🇰 Asia: Hong Kong",
            "🇫🇷🇬🇷🇮🇸 Europe: Paris, Athens, Iceland"
        ]
        for ex in exchanges_at_35:
            st.text(ex)
        st.caption("")
        st.info("💡 Only exchanges currently OPEN are updated (closed exchanges skipped)")

    st.markdown("---")

    # Exchange Trading Hours Table
    st.subheader(f"🕐 Exchange Trading Hours ({SELECTED_TZ_ABBREV})")

    # Check current DST status for accurate display
    eastern = pytz.timezone('US/Eastern')
    now_eastern = datetime.now(eastern)
    is_edt = bool(now_eastern.dst())
    tz_name = now_eastern.tzname()

    # Check Europe DST status
    europe_tz = pytz.timezone('Europe/London')
    now_europe = datetime.now(europe_tz)
    is_cest = bool(now_europe.dst())
    europe_tz_name = 'CEST/BST' if is_cest else 'CET/GMT'

    # Calculate DST-adjusted times
    us_open = '13:30' if is_edt else '14:30'
    us_close = '20:00' if is_edt else '21:00'

    # Europe DST offset
    # Base times are summer (DST) times
    # During summer (CEST): dst_offset = 0 (use base times)
    # During winter (CET): dst_offset = 1 (add 1 hour - markets open later in UTC)
    dst_offset = 0 if is_cest else 1

    display_tz_now = datetime.now(SELECTED_TZ)
    st.caption(
        f"⏰ Display timezone: {TIMEZONE_OPTIONS[selected_tz_key]} ({display_tz_now.tzname()}) | "
        f"US Eastern now: {tz_name} (DST {'Active' if is_edt else 'Inactive'}) | "
        f"Europe core: {europe_tz_name} (DST {'Active' if is_cest else 'Inactive'})"
    )

    # Create detailed exchange hours data
    exchange_hours = {
        '🇺🇸 United States': [
            ('NASDAQ', us_open, us_close, ':30'),
            ('NYSE', us_open, us_close, ':30'),
            ('NYSE ARCA', us_open, us_close, ':30'),
            ('NYSE AMERICAN', us_open, us_close, ':30'),
            ('CBOE BZX', us_open, us_close, ':30'),
        ],
        '🇨🇦 Canada': [
            ('TORONTO', us_open, us_close, ':30'),  # Toronto follows US ET time (9:30 AM - 4:00 PM ET)
        ],
        '🇧🇷 Brazil': [
            ('SAO PAULO', '13:00', '20:00', ':00'),
        ],
        '🌎 Latin America': [
            ('MEXICO', '14:30', '21:00', ':30'),
            ('SANTIAGO', '12:30', '19:00', ':30'),
            ('BUENOS AIRES', '13:30', '20:00', ':30'),
            ('COLOMBIA', '12:30', '19:00', ':30'),
        ],
        '🇯🇵🇭🇰🇹🇼 Asia-Pacific': [
            ('TOKYO', '00:00', '06:30', ':00'),
            ('HONG KONG', '01:30', '08:00', ':30'),
            ('TAIWAN', '01:00', '05:30', ':00'),
            ('ASX (Australia)', '00:00', '06:00', ':00'),
            ('SINGAPORE', '01:00', '09:00', ':00'),
            ('MALAYSIA', '01:00', '09:00', ':00'),
            ('THAILAND', '03:00', '09:30', ':00'),
            ('INDONESIA', '02:00', '09:00', ':00'),
            ('BSE INDIA', '03:45', '10:00', ':15'),
            ('NSE INDIA', '03:45', '10:00', ':15'),
        ],
        '🇬🇧🇩🇪🇫🇷 Europe': [
            ('LONDON', f'{7+dst_offset:02.0f}:00', f'{15+dst_offset:02.0f}:30', ':00'),
            ('XETRA (Germany)', f'{6+dst_offset:02.0f}:00', f'{20+dst_offset:02.0f}:00', ':00'),
            ('EURONEXT PARIS', f'{7+dst_offset:02.0f}:30', f'{15+dst_offset:02.0f}:30', ':30'),
            ('EURONEXT AMSTERDAM', f'{7+dst_offset:02.0f}:00', f'{15+dst_offset:02.0f}:30', ':00'),
            ('EURONEXT BRUSSELS', f'{7+dst_offset:02.0f}:00', f'{15+dst_offset:02.0f}:30', ':00'),
            ('EURONEXT LISBON', f'{7+dst_offset:02.0f}:00', f'{15+dst_offset:02.0f}:30', ':00'),
            ('EURONEXT DUBLIN', f'{7+dst_offset:02.0f}:00', f'{15+dst_offset:02.0f}:30', ':00'),
            ('MILAN', f'{7+dst_offset:02.0f}:00', f'{15+dst_offset:02.0f}:30', ':00'),
            ('SPAIN', f'{7+dst_offset:02.0f}:00', f'{15+dst_offset:02.0f}:30', ':00'),
            ('SIX SWISS', f'{7+dst_offset:02.0f}:00', f'{15+dst_offset:02.0f}:30', ':00'),
            ('OMX STOCKHOLM', f'{7+dst_offset:02.0f}:00', f'{15+dst_offset:02.0f}:30', ':00'),
            ('OMX COPENHAGEN', f'{7+dst_offset:02.0f}:00', f'{15+dst_offset:02.0f}:00', ':00'),
            ('OMX HELSINKI', f'{7+dst_offset:02.0f}:00', f'{15+dst_offset:02.0f}:30', ':00'),
            ('OMX ICELAND', f'{9+dst_offset:02.0f}:30', f'{15+dst_offset:02.0f}:30', ':30'),
            ('OSLO', f'{7+dst_offset:02.0f}:00', f'{14+dst_offset:02.0f}:30', ':00'),
            ('WARSAW', f'{7+dst_offset:02.0f}:00', f'{15+dst_offset:02.0f}:00', ':00'),
            ('VIENNA', f'{7+dst_offset:02.0f}:00', f'{15+dst_offset:02.0f}:30', ':00'),
            ('ATHENS', f'{7+dst_offset:02.0f}:30', f'{14+dst_offset:02.0f}:00', ':30'),
            ('PRAGUE', f'{7+dst_offset:02.0f}:00', f'{14+dst_offset:02.0f}:10', ':00'),
            ('BUDAPEST', f'{7+dst_offset:02.0f}:00', f'{15+dst_offset:02.0f}:00', ':00'),
            ('ISTANBUL', '07:00', '15:00', ':00'),  # Turkey does not observe DST
        ],
        '🇿🇦 Africa': [
            ('JSE (South Africa)', '07:00', '15:00', ':00'),
        ],
    }

    # Create tabs for each region
    region_tabs = st.tabs(list(exchange_hours.keys()))
    open_col = f"Open ({SELECTED_TZ_ABBREV})"
    close_col = f"Close ({SELECTED_TZ_ABBREV})"

    for i, (region, exchanges) in enumerate(exchange_hours.items()):
        with region_tabs[i]:
            df = pd.DataFrame(exchanges, columns=['Exchange', 'Open (UTC)', 'Close (UTC)', 'Candle Type'])
            df[open_col] = df['Open (UTC)'].apply(lambda t: convert_utc_clock_string(t, SELECTED_TZ, now_utc))
            df[close_col] = df['Close (UTC)'].apply(lambda t: convert_utc_clock_string(t, SELECTED_TZ, now_utc))
            df = df[['Exchange', open_col, close_col, 'Candle Type']]
            st.dataframe(df, use_container_width=True, hide_index=True)

    st.caption(
        "💡 Candle Type shows when hourly bars close (:00, :15, or :30). Open/Close columns now reflect your selected timezone."
    )

    st.markdown("---")

    # Detailed Status
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Scheduler Details")

        if status:
            details_data = {
                "Status": status.get('status', 'Unknown'),
                "Type": status.get('type', 'Unknown'),
                "Last Update": display_timestamp_value(status.get('last_update'), SELECTED_TZ),
                "Next Run": display_timestamp_value(status.get('next_run'), SELECTED_TZ),
                "Last Run": display_timestamp_value(status.get('last_run'), SELECTED_TZ)
            }

            for key, value in details_data.items():
                st.text(f"{key}: {value}")
        else:
            st.info("No status information available. Start the scheduler to see details.")

    with col2:
        st.subheader("🔧 Process Information")

        if lock:
            lock_data = {
                "PID": lock.get('pid', 'N/A'),
                "Started At": display_timestamp_value(lock.get('started_at'), SELECTED_TZ),
                "Lock Type": lock.get('type', 'N/A'),
                "Lock Timestamp": display_timestamp_value(lock.get('timestamp'), SELECTED_TZ)
            }

            for key, value in lock_data.items():
                st.text(f"{key}: {value}")
        else:
            st.warning("Scheduler is not running. No lock file found.")

    st.markdown("---")

    # Live Progress Tracker
    if status and status.get('current_progress'):
        st.subheader("🔄 Live Update Progress")

        progress = status['current_progress']
        current = progress.get('current', 0)
        total = progress.get('total', 1)
        percentage = progress.get('percentage', 0)
        eta_seconds = progress.get('eta_seconds', 0)

        # Progress bar
        st.progress(percentage / 100, text=f"Processing: {current:,} / {total:,} tickers ({percentage}%)")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Current Ticker", progress.get('current_ticker', 'N/A'))

        with col2:
            st.metric("Processed", f"{current:,} / {total:,}")

        with col3:
            success = progress.get('success', 0)
            st.metric("Success", f"{success:,}", delta=f"+{success}")

        with col4:
            eta_mins = eta_seconds // 60
            eta_secs = eta_seconds % 60
            st.metric("ETA", f"{eta_mins}m {eta_secs}s")

        st.markdown("---")

    # Last Run Statistics
    st.subheader("📈 Last Run Statistics")

    if status and 'last_run_stats' in status:
        stats = status['last_run_stats']

        # Check if we have skipped_closed stats
        has_skipped = 'skipped_closed' in stats and stats.get('skipped_closed', 0) > 0

        if has_skipped:
            col1, col2, col3, col4, col5 = st.columns(5)
        else:
            col1, col2, col3, col4 = st.columns(4)
            col5 = None

        with col1:
            st.metric("Updated", f"{stats.get('total', 0):,}")

        with col2:
            success = stats.get('success', 0)
            st.metric("Successful", f"{success:,}", delta=None)

        with col3:
            failed = stats.get('failed', 0)
            if failed > 0:
                st.metric("Failed", f"{failed:,}", delta=f"-{failed}", delta_color="inverse")
            else:
                st.metric("Failed", "0")

        with col4:
            total = stats.get('total', 1)
            success_rate = (success / total * 100) if total > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")

        if col5:
            with col5:
                skipped = stats.get('skipped_closed', 0)
                st.metric("Skipped (Closed)", f"{skipped:,}", help="Tickers skipped because their exchange was closed")

        # Stats breakdown
        if stats.get('total', 0) > 0:
            st.markdown("##### Update Breakdown")
            categories = ['Successful', 'Failed', 'Updated']
            counts = [stats.get('success', 0), stats.get('failed', 0), stats.get('total', 0)]

            if has_skipped:
                categories.append('Skipped (Closed)')
                counts.append(stats.get('skipped_closed', 0))

            percentages = [
                f"{(stats.get('success', 0) / stats.get('total', 1) * 100):.1f}%",
                f"{(stats.get('failed', 0) / stats.get('total', 1) * 100):.1f}%",
                "100.0%"
            ]

            if has_skipped:
                total_all = stats.get('total', 0) + stats.get('skipped_closed', 0)
                percentages.append(f"{(stats.get('skipped_closed', 0) / max(total_all, 1) * 100):.1f}%")

            stats_df = pd.DataFrame({
                'Category': categories,
                'Count': counts,
                'Percentage': percentages
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)
    else:
        st.info("No statistics available yet. Statistics will appear after the first run.")

    st.markdown("---")

    # Show last update time
    st.caption(
        f"🟢 Live updates every 3 seconds | Last updated: {datetime.now(SELECTED_TZ).strftime('%I:%M:%S %p %Z')}"
    )

# Call the live monitoring fragment
live_status_monitor()

st.markdown("---")

# Load status for control section
status = load_status()
lock_info = load_lock()
running = (
    status_indicates_running(status)
    or status_pid_alive(status)
    or lock_pid_alive(lock_info)
)

# Control Section
st.subheader("🎮 Scheduler Control")

col1, col2, col3 = st.columns(3)

with col1:
    if not running:
        if st.button("▶️ Start Scheduler", use_container_width=True, type="primary"):
            with st.spinner("Starting hourly data scheduler..."):
                if start_scheduler_process():
                    st.success("✅ Hourly scheduler started successfully!")
                    st.info("The scheduler is now running in the background (no console window).")
                    time.sleep(2)
                    load_status.clear()
                    st.rerun()
                else:
                    st.error("❌ Failed to start scheduler. Check the logs for details.")
    else:
        st.success("✅ Scheduler is currently running")

with col2:
    if running:
        if st.button("⏹️ Stop Scheduler", use_container_width=True, type="secondary"):
            with st.spinner("Stopping hourly data scheduler..."):
                if stop_scheduler_process():
                    st.success("✅ Hourly scheduler stopped successfully!")
                    time.sleep(2)
                    load_status.clear()
                    st.rerun()
                else:
                    st.error("❌ Failed to stop scheduler. You may need to close it manually.")
                    pid_hint = None
                    if isinstance(status, dict):
                        pid_hint = status.get('pid')
                    if not pid_hint and lock_info and 'pid' in lock_info:
                        pid_hint = lock_info['pid']
                    if pid_hint:
                        st.code(f"taskkill /F /PID {pid_hint}", language="bash")

with col3:
    if st.button("📊 View Full Logs", use_container_width=True):
        st.session_state.show_logs = not st.session_state.get('show_logs', False)

st.markdown("---")

# Log Viewer
if st.session_state.get('show_logs', False):
    st.subheader("📄 Recent Log Entries")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("**Showing last 50 lines:**")

    with col2:
        log_lines = st.selectbox("Lines to show", [20, 50, 100, 200], index=1, key="log_lines_count")

    log_content = get_log_tail(log_lines)
    st.code(log_content, language="log")

    if st.button("🔽 Download Full Log"):
        if LOG_FILE.exists():
            log_data = LOG_FILE.read_text(encoding='utf-8', errors='ignore')

            st.download_button(
                label="📥 Download hourly_data_scheduler.log",
                data=log_data,
                file_name="hourly_data_scheduler.log",
                mime="text/plain"
            )

st.markdown("---")

# Information Section
st.subheader("ℹ️ About Hourly Data Scheduler")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **What it does:**
    - Automatically updates hourly price data for all tickers
    - Runs every hour at :05 past the hour
    - Only fetches new data since last update
    - Skips updates when markets are closed
    - Processes all stocks and ETFs in the database

    **Schedule:**
    - ⏰ Runs at: :05 past every hour (e.g., 9:05, 10:05, 11:05)
    - 📅 Active: Monday-Friday during market hours
    - 🌍 Covers: US, Europe, and Asia markets
    - ⏸️ Skips: Weekends and off-market hours
    """)

with col2:
    status_location = build_key("hourly_scheduler_status")
    st.markdown(f"""
    **Files & Keys:**
    - `hourly_scheduler.lock` - Lock file (separate from main scheduler)
    - Redis `{status_location}` (preferred) or `hourly_scheduler_status.json` for status snapshots
    - `hourly_data_scheduler.log` - Detailed logs
    - `start_hourly_data_scheduler.bat` - Startup script

    **Features:**
    - ✅ Separate process from alert scheduler
    - ✅ Rate limiting (3,000 calls/minute)
    - ✅ Skips already up-to-date tickers
    - ✅ Market hours detection
    - ✅ Comprehensive logging
    """)

st.markdown("---")

"""
Streamlit page for viewing and filtering price database
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
from typing import Any, Dict, List
from contextlib import contextmanager

sys.path.append(str(Path(__file__).parent.parent))
from src.utils.stale_data import is_data_stale, get_last_trading_day
from src.services.backend_fmp_optimized import OptimizedDailyPriceCollector
import time
try:
    import psycopg2
    OperationalError = psycopg2.OperationalError
except Exception:  # psycopg2 not installed or unavailable in this env
    OperationalError = Exception

from src.data_access.metadata_repository import fetch_stock_metadata_map
from src.data_access.db_config import db_config
from src.services.calendar_adapter import get_session_bounds, get_hourly_alignment

# Page config
st.set_page_config(
    page_title="Price Database Viewer",
    page_icon="📊",
    layout="wide"
)

# Title and description
st.title("📊 Price Database Viewer")
st.markdown("View and analyze historical price data with advanced filtering options")

# Initialize session state
if 'selected_ticker' not in st.session_state:
    st.session_state.selected_ticker = None

DEFAULT_PRICE_FILTERS = {
    "timeframe": "daily",
    "selected_exchanges": [],
    "search_type": "Ticker",
    "search_input": "",
    "selected_tickers": [],
    "max_rows": 5000,
    "day_filter": "All days",
}

if "pd_filters" not in st.session_state:
    st.session_state.pd_filters = DEFAULT_PRICE_FILTERS.copy()

@st.cache_resource(ttl=1800)  # Cache for 30 minutes
def load_main_database():
    """Load main database for ticker metadata"""
    try:
        return fetch_stock_metadata_map()
    except Exception:
        return {}

PRICE_INDEXES_CREATED = False


def _ensure_price_indexes(conn) -> None:
    """Create indexes that dramatically speed up ticker/date lookups."""
    global PRICE_INDEXES_CREATED
    if PRICE_INDEXES_CREATED:
        return

    statements = [
        ("daily_prices", "CREATE INDEX IF NOT EXISTS idx_daily_prices_ticker_date ON daily_prices (ticker, date DESC)"),
        ("hourly_prices", "CREATE INDEX IF NOT EXISTS idx_hourly_prices_ticker_datetime ON hourly_prices (ticker, datetime DESC)"),
        ("weekly_prices", "CREATE INDEX IF NOT EXISTS idx_weekly_prices_ticker_week ON weekly_prices (ticker, week_ending DESC)"),
    ]

    cursor = conn.cursor()
    try:
        for table, statement in statements:
            if not _table_exists(conn, table):
                continue
            cursor.execute(statement)
        conn.commit()
        PRICE_INDEXES_CREATED = True
    except Exception as exc:
        st.warning(f"Failed to ensure price indexes: {exc}")
        try:
            conn.rollback()
        except Exception:
            pass
    finally:
        cursor.close()


def _compute_hourly_data_quality(conn) -> Dict[str, Any]:
    """Return freshness and gap metrics for hourly_prices."""
    results: Dict[str, Any] = {
        "total_tickers": 0,
        "stale_tickers": 0,
        "oldest_stale": None,
        "gap_tickers": 0,
        "worst_gap_hours": 0.0,
        "worst_calendar_gap_hours": 0.0,
    }

    stale_query = """
        WITH last_points AS (
            SELECT ticker, MAX(datetime) AS last_dt
            FROM hourly_prices
            GROUP BY ticker
        )
        SELECT
            COUNT(*) AS total_tickers,
            COUNT(*) FILTER (WHERE last_dt < NOW() - INTERVAL '48 hours') AS stale_tickers,
            MIN(last_dt) FILTER (WHERE last_dt < NOW() - INTERVAL '48 hours') AS oldest_stale
        FROM last_points;
    """

    gap_query = """
        WITH recent AS (
            SELECT ticker, datetime
            FROM hourly_prices
            WHERE datetime >= NOW() - INTERVAL '60 days'
        ),
        ordered AS (
            SELECT
                ticker,
                datetime,
                LEAD(datetime) OVER (PARTITION BY ticker ORDER BY datetime) AS next_dt
            FROM recent
        ),
        gaps AS (
            SELECT
                ticker,
                -- Raw calendar gap (for reference/diagnostics)
                EXTRACT(EPOCH FROM (next_dt - datetime))/3600.0 AS gap_hours,
                -- Trading-hour aware gap: count only weekday hours (Mon-Fri)
                (
                    SELECT COUNT(*)::float
                    FROM generate_series(
                        datetime,
                        next_dt - INTERVAL '1 hour',
                        INTERVAL '1 hour'
                    ) AS gs(dt)
                    WHERE EXTRACT(ISODOW FROM gs.dt) BETWEEN 1 AND 5
                ) AS trading_gap_hours
            FROM ordered
            WHERE next_dt IS NOT NULL
        )
        SELECT
            COUNT(DISTINCT ticker) FILTER (WHERE trading_gap_hours > 72) AS gap_tickers,
            COALESCE(MAX(trading_gap_hours), 0) AS worst_gap_hours,
            COALESCE(MAX(gap_hours), 0) AS worst_calendar_gap_hours
        FROM gaps;
    """

    try:
        stale_df = read_sql_with_engine(stale_query)
        if not stale_df.empty:
            row = stale_df.iloc[0]
            results["total_tickers"] = int(row.get("total_tickers") or 0)
            results["stale_tickers"] = int(row.get("stale_tickers") or 0)
            results["oldest_stale"] = row.get("oldest_stale")
    except Exception as exc:
        st.error(f"Error calculating stale hourly tickers: {exc}")

    try:
        gap_df = read_sql_with_engine(gap_query)
        if not gap_df.empty:
            row = gap_df.iloc[0]
            results["gap_tickers"] = int(row.get("gap_tickers") or 0)
            results["worst_gap_hours"] = float(row.get("worst_gap_hours") or 0.0)
            results["worst_calendar_gap_hours"] = float(row.get("worst_calendar_gap_hours") or 0.0)
    except Exception as exc:
        st.error(f"Error calculating hourly gap metrics: {exc}")

    return results


def _table_exists(conn, table_name: str) -> bool:
    """Check whether a table exists in PostgreSQL."""
    cursor = conn.cursor()
    try:
        cursor.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = current_schema()
                  AND table_name = %s
            )
            """,
            (table_name,),
        )
        row = cursor.fetchone()
        return bool(row and row[0])
    finally:
        cursor.close()


@contextmanager
def get_database_connection():
    """Yield a database connection, ensuring cleanup after use."""
    conn = db_config.get_connection(role="prices")
    try:
        _ensure_price_indexes(conn)
        yield conn
    finally:
        db_config.close_connection(conn)

def read_sql_with_engine(query, params=None):
    """
    Execute a SQL query using SQLAlchemy engine (preferred by pandas).
    This eliminates the pandas UserWarning about using raw DBAPI connections.
    Handles both ? (SQLite-style) and %s (PostgreSQL-style) placeholders.
    """
    from sqlalchemy import text

    engine = db_config.get_sqlalchemy_engine()
    if params:
        # Convert placeholders to named parameters for SQLAlchemy
        # Handle both ? (SQLite) and %s (PostgreSQL) styles
        query_modified = query
        param_count = query.count('?')

        if param_count > 0:
            # SQLite-style placeholders (?)
            param_names = [f'param{i}' for i in range(param_count)]
            for name in param_names:
                query_modified = query_modified.replace('?', f':{name}', 1)
            params_dict = dict(zip(param_names, params))
        elif '%s' in query:
            # PostgreSQL-style placeholders (%s) - convert to named parameters
            param_count = query.count('%s')
            param_names = [f'param{i}' for i in range(param_count)]
            for name in param_names:
                query_modified = query_modified.replace('%s', f':{name}', 1)
            params_dict = dict(zip(param_names, params))
        else:
            # No placeholders, use params as-is (might be a dict already)
            params_dict = params if isinstance(params, dict) else {}

        # Use connection from engine for proper parameter binding
        with engine.connect() as conn:
            return pd.read_sql_query(text(query_modified), conn, params=params_dict)

    return pd.read_sql_query(query, engine)

def get_unique_exchanges(main_db):
    """Get unique exchanges from main database"""
    exchanges = set()
    for ticker, data in main_db.items():
        exchange = data.get('exchange', '')
        if exchange and exchange != 'N/A':
            exchanges.add(exchange)
    return sorted(list(exchanges))

def get_unique_tickers_by_exchange(main_db, selected_exchanges):
    """Get tickers filtered by exchange"""
    tickers = []
    for ticker, data in main_db.items():
        if not selected_exchanges or data.get('exchange', '') in selected_exchanges:
            tickers.append(ticker)
    return sorted(tickers)

def load_price_data(
    conn,
    tickers=None,
    start_date=None,
    end_date=None,
    timeframe='daily',
    limit=None,
    day_filter: str = "All days",
):
    """Load price data with filters while only selecting necessary columns."""
    columns_map = {
        'hourly': ('datetime', "ticker, datetime, open, high, low, close, volume"),
        'daily': ('date', "ticker, date, open, high, low, close, volume"),
        'weekly': ('week_ending', "ticker, week_ending, open, high, low, close, volume"),
    }
    date_col, select_cols = columns_map.get(timeframe, columns_map['daily'])

    query = f"SELECT {select_cols} FROM {timeframe}_prices WHERE 1=1"
    params: List[str] = []

    if not _table_exists(conn, f"{timeframe}_prices"):
        st.info(f"The table '{timeframe}_prices' does not exist in the current database.")
        return pd.DataFrame()

    if tickers:
        placeholders = ','.join(['?' for _ in tickers])
        query += f" AND ticker IN ({placeholders})"
        params.extend(tickers)

    if start_date:
        start_str = start_date.strftime('%Y-%m-%d')
        if timeframe == 'hourly':
            query += f" AND {date_col} >= ?"
            params.append(f"{start_str} 00:00:00")
        else:
            query += f" AND {date_col} >= ?"
            params.append(start_str)

    if end_date:
        end_str = end_date.strftime('%Y-%m-%d')
        if timeframe == 'hourly':
            query += f" AND {date_col} <= ?"
            params.append(f"{end_str} 23:59:59")
        else:
            query += f" AND {date_col} <= ?"
            params.append(end_str)

    # Day-of-week filter
    if day_filter == "Weekdays only (Mon-Fri)":
        query += f" AND EXTRACT(ISODOW FROM {date_col}) BETWEEN 1 AND 5"
    elif day_filter == "Weekends only (Sat-Sun)":
        query += f" AND EXTRACT(ISODOW FROM {date_col}) IN (6,7)"

    query += f" ORDER BY ticker, {date_col} DESC"
    if limit:
        query += f" LIMIT {int(limit)}"

    try:
        df = read_sql_with_engine(query, params=params if params else None)
        if not df.empty and date_col in df.columns:
            df['date'] = pd.to_datetime(df[date_col])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


def _apply_day_filter(df: pd.DataFrame, timeframe: str, day_filter: str) -> pd.DataFrame:
    """Filter dataframe by weekdays/weekends if requested."""
    if df.empty or day_filter == "All days":
        return df

    date_col_map = {"hourly": "datetime", "daily": "date", "weekly": "week_ending"}
    col = date_col_map.get(timeframe)
    if col not in df.columns:
        return df

    df = df.copy()
    df[col] = pd.to_datetime(df[col], errors="coerce")
    dow = df[col].dt.isocalendar().day  # 1=Mon, 7=Sun

    if day_filter.startswith("Weekends"):
        mask = dow.isin([6, 7])
    elif day_filter.startswith("Weekdays"):
        mask = dow.isin([1, 2, 3, 4, 5])
    else:
        return df

    return df[mask]

@st.cache_data(ttl=300, show_spinner=False)  # Cache for 5 minutes
def get_database_stats() -> Dict[str, Any]:
    """Get database statistics from PostgreSQL/SQLite."""
    stats: Dict[str, Any] = {}

    # Retry once on OperationalError to avoid crashing the page if a connection died
    last_error = None
    for attempt in range(2):
        try:
            with db_config.connection(role="prices") as conn:
                cursor = conn.cursor()
                try:
                    hourly_stats = daily_stats = weekly_stats = None

                    if _table_exists(conn, "hourly_prices"):
                        cursor.execute(
                            """
                            SELECT COUNT(*) as count,
                                   COUNT(DISTINCT ticker) as tickers,
                                   MIN(datetime) as min_date,
                                   MAX(datetime) as max_date
                            FROM hourly_prices
                            """
                        )
                        hourly_stats = cursor.fetchone()

                    if _table_exists(conn, "daily_prices"):
                        cursor.execute(
                            """
                            SELECT COUNT(*) as count,
                                   COUNT(DISTINCT ticker) as tickers,
                                   MIN(date) as min_date,
                                   MAX(date) as max_date
                            FROM daily_prices
                            """
                        )
                        daily_stats = cursor.fetchone()

                    if _table_exists(conn, "weekly_prices"):
                        cursor.execute(
                            """
                            SELECT COUNT(*) as count,
                                   COUNT(DISTINCT ticker) as tickers,
                                   MIN(week_ending) as min_date,
                                   MAX(week_ending) as max_date
                            FROM weekly_prices
                            """
                        )
                        weekly_stats = cursor.fetchone()

                    hourly_count, hourly_tickers, hourly_min, hourly_max = hourly_stats or (0, 0, None, None)
                    daily_count, daily_tickers, daily_min, daily_max = daily_stats or (0, 0, None, None)
                    weekly_count, weekly_tickers, weekly_min, weekly_max = weekly_stats or (0, 0, None, None)

                    stats = {
                        'hourly_records': hourly_count,
                        'daily_records': daily_count,
                        'weekly_records': weekly_count,
                        'hourly_tickers': hourly_tickers,
                        'daily_tickers': daily_tickers,
                        'weekly_tickers': weekly_tickers,
                        'hourly_date_range': (hourly_min, hourly_max) if hourly_count > 0 else (None, None),
                        'daily_date_range': (daily_min, daily_max) if daily_count > 0 else (None, None),
                        'weekly_date_range': (weekly_min, weekly_max) if weekly_count > 0 else (None, None),
                    }
                    return stats
                finally:
                    cursor.close()
        except OperationalError as exc:
            last_error = exc
            # Retry once after an OperationalError (connection dropped)
            continue
        except Exception as exc:
            last_error = exc
            break

    if last_error:
        st.warning(f"Database stats unavailable right now: {last_error}")

    return stats

def fetch_stale_hourly_data(limit: int | None = None, main_db: Dict[str, Dict[str, Any]] | None = None) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Return tickers missing the most recent hourly bar that exists in the database.
    Uses an exchange-aware expected hour based on trading sessions; falls back to the
    latest plausible timestamp in the table (<= now+1h) for unknown exchanges.
    """
    meta: Dict[str, Any] = {
        "latest_hour": None,
        "latest_hour_raw": None,
        "latest_within_now": None,
        "clamped_to_now": False,
        "total_tickers": 0,
        "up_to_date": 0,
        "table_exists": False,
    }

    with db_config.connection(role="prices") as conn:
        if not _table_exists(conn, "hourly_prices"):
            return pd.DataFrame(), meta

        meta["table_exists"] = True
    per_ticker = read_sql_with_engine(
        """
        SELECT ticker, MAX(datetime) AS last_hour
        FROM hourly_prices
        GROUP BY ticker
        """
    )

    if per_ticker.empty:
        return pd.DataFrame(), meta

    per_ticker["last_hour"] = pd.to_datetime(per_ticker["last_hour"], utc=True)
    per_ticker["hour_bucket"] = per_ticker["last_hour"].dt.floor("h")

    # Map exchanges from metadata for exchange-aware expectations
    exchange_map: Dict[str, str] = {}
    if main_db:
        for t in per_ticker["ticker"]:
            if t in main_db:
                exchange_map[t] = main_db[t].get("exchange")
    per_ticker["exchange"] = per_ticker["ticker"].map(exchange_map).fillna("")

    # Determine a sane global fallback reference hour (latest non-future hour)
    now_utc = pd.Timestamp.now(tz="UTC")
    allowed_max = now_utc + pd.Timedelta(hours=1)
    latest_raw = per_ticker["last_hour"].max()
    latest_within_now = per_ticker.loc[per_ticker["last_hour"] <= allowed_max, "last_hour"].max()

    anchor_hour = None
    recent_mask = per_ticker["last_hour"] <= allowed_max
    if recent_mask.any():
        modes = per_ticker.loc[recent_mask, "hour_bucket"].mode()
        if not modes.empty:
            anchor_hour = modes.iloc[0]
    if anchor_hour is None and not per_ticker["hour_bucket"].isna().all():
        modes = per_ticker["hour_bucket"].mode()
        if not modes.empty:
            anchor_hour = modes.iloc[0]

    reference_hour = anchor_hour if anchor_hour is not None else latest_within_now
    if reference_hour is None or pd.isna(reference_hour):
        reference_hour = allowed_max.floor("h")
    if reference_hour > allowed_max:
        reference_hour = allowed_max.floor("h")
        meta["clamped_to_now"] = True
    cap_upper = reference_hour + pd.Timedelta(hours=2)
    meta["anchor_hour"] = anchor_hour

    # Compute expected hour per exchange using trading sessions
    def _expected_hour_for_exchange(exchange: str) -> pd.Timestamp | None:
        if not exchange:
            return None
        try:
            reference_ts = reference_hour
            if reference_ts.tzinfo is None:
                reference_ts = reference_ts.tz_localize("UTC")
            open_ts, close_ts = get_session_bounds(exchange, reference_ts, next_if_closed=False)
            alignment = get_hourly_alignment(exchange)
        except Exception:
            return None
        if open_ts is None or close_ts is None:
            return None
        open_utc = pd.Timestamp(open_ts).tz_convert("UTC")
        close_utc = pd.Timestamp(close_ts).tz_convert("UTC")
        reference_ts = close_utc if reference_ts >= close_utc else reference_ts
        freq = {"quarter": "15min", "half": "30min", "hour": "60min"}.get(alignment, "60min")
        expected = reference_ts.floor(freq)
        # If before the session open, fall back to previous session close (which get_session_bounds should give)
        if expected < open_utc.floor(freq):
            expected = close_utc.floor(freq)
        return expected.tz_convert("UTC") if expected.tzinfo else expected

    exchanges = per_ticker["exchange"].unique()
    exchange_expected: Dict[str, pd.Timestamp] = {}
    for ex in exchanges:
        if not ex:
            continue
        expected = _expected_hour_for_exchange(ex)
        if expected is not None:
            exchange_expected[ex] = expected

    meta["latest_hour_raw"] = latest_raw
    meta["latest_within_now"] = latest_within_now
    meta["latest_hour"] = reference_hour
    meta["total_tickers"] = len(per_ticker)

    per_ticker["expected_hour"] = per_ticker["exchange"].map(exchange_expected)
    per_ticker["expected_hour"] = per_ticker["expected_hour"].fillna(latest_within_now if pd.notna(latest_within_now) else reference_hour)
    per_ticker["expected_hour"] = pd.to_datetime(per_ticker["expected_hour"], utc=True)

    # Cap obviously future data when computing freshness to avoid skew from bad rows
    per_ticker["last_hour_capped"] = per_ticker["last_hour"].where(per_ticker["last_hour"] <= cap_upper, cap_upper)

    per_ticker["Hours Behind"] = (per_ticker["expected_hour"] - per_ticker["last_hour_capped"]).dt.total_seconds() / 3600.0
    per_ticker["Hours Behind"] = per_ticker["Hours Behind"].fillna(0).clip(lower=0).round(1)
    meta["up_to_date"] = int((per_ticker["last_hour_capped"] >= per_ticker["expected_hour"]).sum())

    stale_df = per_ticker[per_ticker["last_hour_capped"] < per_ticker["expected_hour"]].copy()
    if stale_df.empty:
        return stale_df, meta

    stale_df = stale_df.rename(
        columns={
            "ticker": "Ticker",
            "last_hour": "Last Hour",
        }
    )
    stale_df = stale_df.sort_values(["Hours Behind", "Last Hour"], ascending=[False, True])
    if limit:
        stale_df = stale_df.head(limit)

    return stale_df, meta

@st.cache_data(ttl=300, show_spinner=False)
def fetch_stale_ticker_dataframe(
    timeframe: str,
    *,
    stale_after_days: int | None,
    limit: int | None,
) -> pd.DataFrame:
    """
    Return tickers missing the latest bar:
    - Daily: last_date < last_expected_trading_day
    - Weekly: last_date < last_week_ending
    """
    table_name = f"{timeframe}_prices"
    date_col = "date" if timeframe == "daily" else "week_ending"

    with db_config.connection(role="prices") as conn:
        if not _table_exists(conn, table_name):
            return pd.DataFrame()

        if timeframe == "daily":
            # Compute expected date per exchange using local time (approx: needs today's bar if past 6pm local)
            from calendar_adapter import get_calendar_timezone
            from src.data_access.metadata_repository import fetch_stock_metadata_map

            metadata = fetch_stock_metadata_map()

            def expected_daily_date(exchange: str):
                try:
                    tz_name = get_calendar_timezone(exchange)
                    import pytz
                    tz = pytz.timezone(tz_name) if tz_name else None
                except Exception:
                    tz = None
                now_local = datetime.now(tz) if tz else datetime.utcnow()
                weekday = now_local.weekday()
                # Determine last trading day (weekends -> Friday)
                if weekday >= 5:  # Sat/Sun
                    days_back = weekday - 4
                    last_trading = now_local.date() - timedelta(days=days_back)
                    return last_trading
                # If past ~6pm local, expect today's bar; otherwise yesterday's trading day.
                if now_local.hour >= 18:
                    return now_local.date()
                # previous weekday
                if weekday == 0:  # Monday -> Friday
                    return (now_local - timedelta(days=3)).date()
                return (now_local - timedelta(days=1)).date()

            last_df = read_sql_with_engine(
                f"SELECT ticker, MAX({date_col}) AS last_date FROM {table_name} GROUP BY ticker"
            )
            if last_df.empty:
                return pd.DataFrame()

            last_df["last_date"] = pd.to_datetime(last_df["last_date"])
            last_df["exchange"] = last_df["ticker"].apply(lambda t: metadata.get(t, {}).get("exchange"))
            last_df["expected_date"] = last_df["exchange"].apply(expected_daily_date)
            last_df["Days Old"] = (pd.to_datetime(last_df["expected_date"]) - last_df["last_date"]).dt.days
            stale_df = last_df[last_df["last_date"].dt.date < last_df["expected_date"]]
            stale_df = stale_df.sort_values("last_date")
            if limit:
                stale_df = stale_df.head(limit)

            stale_df["Timeframe"] = "Daily"
            return stale_df.rename(
                columns={
                    "ticker": "Ticker",
                    "last_date": "Last Update",
                }
            )
        else:
            # Weekly: expect the most recent completed Friday.
            # If it's Friday but before close+40m (US), use the prior Friday.
            now_utc = datetime.utcnow()
            close_buffer_hour_utc = 21 + 40 / 60  # 16:40 ET ~= 21.67 UTC
            weekday = now_utc.weekday()
            current_hour = now_utc.hour + now_utc.minute / 60.0

            if weekday > 4:  # Sat/Sun -> this past Friday
                expected_week_end = now_utc.date() - timedelta(days=weekday - 4)
            elif weekday == 4:  # Friday
                if current_hour >= close_buffer_hour_utc:
                    expected_week_end = now_utc.date()  # today’s Friday is complete
                else:
                    expected_week_end = now_utc.date() - timedelta(days=7)  # wait until after close
            else:  # Mon-Thu -> last Friday
                expected_week_end = now_utc.date() - timedelta(days=weekday + 3)

            query = f"""
                SELECT
                    ticker,
                    MAX({date_col}) AS last_date,
                    CAST(%s::date - MAX({date_col}) AS INTEGER) AS days_old
                FROM {table_name}
                GROUP BY ticker
                HAVING MAX({date_col}) < %s::date
                ORDER BY MAX({date_col}) ASC
            """
            params = (expected_week_end, expected_week_end)

        if limit:
            query += " LIMIT %s"
            params = (*params, limit)

        df = read_sql_with_engine(query, params=list(params))

    if df.empty:
        return df

    df["last_date"] = pd.to_datetime(df["last_date"])
    df["Timeframe"] = timeframe.capitalize()
    return df.rename(
        columns={
            "ticker": "Ticker",
            "last_date": "Last Update",
            "days_old": "Days Old",
        }
    )

def refresh_stale_tickers(conn, stale_tickers, timeframe='daily'):
    """Refresh multiple stale tickers"""
    from datetime import datetime, timezone

    table_name = f"{timeframe}_prices"
    if not _table_exists(conn, table_name):
        st.error(f"Cannot refresh tickers: '{table_name}' table does not exist in the database.")
        return {"success": [], "failed": [t['Ticker'] for t in stale_tickers], "skipped_market_open": []}

    stock_db = load_main_database()

    # Market close times in UTC (with 30-minute buffer for data availability)
    # Format: exchange -> close time in UTC (24-hour format)
    market_close_times_utc = {
        'NASDAQ': 20.5,    # 4:00 PM ET = 20:00 UTC, + 30 min buffer = 20:30 UTC
        'NYSE': 20.5,      # 4:00 PM ET = 20:00 UTC, + 30 min buffer = 20:30 UTC
        'AMEX': 20.5,      # 4:00 PM ET = 20:00 UTC, + 30 min buffer = 20:30 UTC
        'SAO PAULO': 21.5, # 6:00 PM BRT = 21:00 UTC, + 30 min buffer = 21:30 UTC
    }

    collector = OptimizedDailyPriceCollector()
    results = {
        'success': [],
        'failed': [],
        'skipped_market_open': []
    }

    # Only resample weekly on Fridays (weekday 4)
    is_friday = datetime.now().weekday() == 4

    # Get current UTC time
    now_utc = datetime.now(timezone.utc)
    current_hour_utc = now_utc.hour + now_utc.minute / 60.0

    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    total = len(stale_tickers)

    for i, ticker_info in enumerate(stale_tickers):
        ticker = ticker_info['Ticker']

        # Update progress less frequently to reduce UI overhead
        if i == 0 or (i + 1) == total or (i + 1) % 25 == 0:
            progress = (i + 1) / total
            progress_bar.progress(progress)
            status_text.text(f"Updating {ticker} ({i+1}/{total})...")

        try:
            # Check if market is closed (only on weekdays)
            should_skip = False
            if now_utc.weekday() < 5:  # Monday-Friday (0-4)
                exchange = stock_db.get(ticker, {}).get('exchange', 'UNKNOWN')
                close_time_utc = market_close_times_utc.get(exchange)

                if close_time_utc and current_hour_utc < close_time_utc:
                    # Market hasn't closed yet, skip this ticker
                    should_skip = True
                    results['skipped_market_open'].append(ticker)

            if should_skip:
                # Skip to next ticker without updating
                if i < total - 1:
                    time.sleep(0.05)
                continue

            # Update the ticker - only resample weekly on Fridays
            success = collector.update_ticker(ticker, resample_weekly=is_friday)

            if success:
                results['success'].append(ticker)
            else:
                results['failed'].append(ticker)

        except Exception as e:
            results['failed'].append(ticker)
            st.warning(f"Failed to update {ticker}: {str(e)}")

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    return results


def _filter_stale_for_closed_markets(df: pd.DataFrame, main_db: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Drop tickers that only appear stale because their exchange is closed today (holidays).
    We derive the most recent completed session date per exchange from the trading calendar.
    """
    if df.empty or "Ticker" not in df.columns or "Last Update" not in df.columns:
        return df

    reference = pd.Timestamp.utcnow()
    exchange_expected: Dict[str, pd.Timestamp] = {}

    def expected_close_date(exchange: str) -> pd.Timestamp | None:
        if exchange in exchange_expected:
            return exchange_expected[exchange]
        try:
            _, close_ts = get_session_bounds(exchange, reference, next_if_closed=False)
            expected = pd.Timestamp(close_ts).normalize()
        except Exception:
            expected = None
        exchange_expected[exchange] = expected
        return expected

    filtered_rows = []
    for row in df.to_dict(orient="records"):
        ticker = row.get("Ticker")
        last_update = row.get("Last Update")
        exchange = main_db.get(ticker, {}).get("exchange") if main_db else None
        if exchange:
            exp_date = expected_close_date(exchange)
            if exp_date is not None:
                try:
                    last_dt = pd.to_datetime(last_update)
                    if pd.notna(last_dt):
                        last_date = last_dt.date()
                        exp_date_only = pd.Timestamp(exp_date).date()
                        if last_date >= exp_date_only:
                            # Exchange closed today/holiday; data is up-to-date for its last session
                            continue
                except Exception:
                    # Fallback: string comparison on ISO dates
                    if str(last_update)[:10] >= str(exp_date)[:10]:
                        # Exchange closed today/holiday; data is up-to-date for its last session
                        continue
        filtered_rows.append(row)

    return pd.DataFrame(filtered_rows)

def resample_weekly_only(stale_tickers):
    """Resample weekly data from existing daily data without fetching new data"""
    collector = OptimizedDailyPriceCollector()
    results = {
        'success': [],
        'failed': []
    }

    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    total = len(stale_tickers)

    for i, ticker_info in enumerate(stale_tickers):
        ticker = ticker_info['Ticker']

        # Update progress
        progress = (i + 1) / total
        progress_bar.progress(progress)
        status_text.text(f"Resampling {ticker} ({i+1}/{total})...")

        try:
            # Get existing daily data (last 250 days for proper weekly aggregation)
            recent_df = collector.db.get_daily_prices(ticker, limit=250)

            if recent_df is None or recent_df.empty:
                results['failed'].append(ticker)
                continue

            # Resample to weekly
            weekly_df = collector._resample_to_weekly(recent_df, ticker)

            if weekly_df is not None and not weekly_df.empty:
                weekly_records = collector.db.store_weekly_prices(ticker, weekly_df)
                if weekly_records >= 0:
                    results['success'].append(ticker)
                else:
                    results['failed'].append(ticker)
            else:
                results['failed'].append(ticker)

        except Exception as e:
            results['failed'].append(ticker)
            st.warning(f"Failed to resample {ticker}: {str(e)}")

    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()

    return results

def _update_hourly_batch(tickers: List[str], days_back: int, skip_existing: bool = True) -> Dict[str, List[str]]:
    """Update hourly data for a specific list of tickers with progress feedback."""
    if not tickers:
        return {"success": [], "failed": []}

    import sys

    sys.path.append(str(Path(__file__).parent.parent))
    from src.services.hourly_price_collector import HourlyPriceCollector

    collector = HourlyPriceCollector()
    progress_bar = st.progress(0)
    status_text = st.empty()

    results: Dict[str, List[str]] = {"success": [], "failed": []}
    total = len(tickers)

    for i, ticker in enumerate(tickers, 1):
        try:
            updated = collector.update_ticker_hourly(ticker, days_back, skip_existing=skip_existing)
            if updated:
                results["success"].append(ticker)
            else:
                results["failed"].append(ticker)
        except Exception as exc:
            results["failed"].append(ticker)
            st.warning(f"Failed updating {ticker}: {exc}")

        if i == 1 or i == total or (i % 25 == 0):
            progress_bar.progress(i / total)
            status_text.text(f"Updating {ticker} ({i}/{total})... "
                             f"Success: {len(results['success'])} | Failed: {len(results['failed'])}")

    progress_bar.empty()
    status_text.empty()
    return results

def create_price_chart(df, ticker, show_volume=True):
    """Create interactive price chart"""
    ticker_data = df[df['ticker'] == ticker].sort_values('date')

    if ticker_data.empty:
        return None

    # Create figure with subplots
    if show_volume and 'volume' in ticker_data.columns:
        fig = go.Figure()

        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=ticker_data['date'],
            open=ticker_data['open'],
            high=ticker_data['high'],
            low=ticker_data['low'],
            close=ticker_data['close'],
            name='Price',
            yaxis='y'
        ))

        # Volume bars
        fig.add_trace(go.Bar(
            x=ticker_data['date'],
            y=ticker_data['volume'],
            name='Volume',
            yaxis='y2',
            opacity=0.3
        ))

        # Update layout
        fig.update_layout(
            title=f'{ticker} Price and Volume',
            yaxis=dict(title='Price', side='left'),
            yaxis2=dict(title='Volume', side='right', overlaying='y'),
            xaxis=dict(title='Date'),
            hovermode='x unified',
            height=600
        )
    else:
        # Simple line chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ticker_data['date'],
            y=ticker_data['close'],
            mode='lines',
            name='Close Price',
            line=dict(color='blue', width=2)
        ))

        fig.update_layout(
            title=f'{ticker} Close Price',
            xaxis_title='Date',
            yaxis_title='Price',
            height=500
        )

    return fig

# Main app
def main():
    main_db = load_main_database()

    # Sidebar filters
    st.sidebar.header("🔍 Filters")

    all_tickers = sorted(list(main_db.keys()))
    default_filters = st.session_state.pd_filters

    with st.sidebar.form("price_filters"):
        timeframe_input = st.radio(
            "Timeframe",
            options=['hourly', 'daily', 'weekly'],
            format_func=lambda x: x.capitalize(),
            index=['hourly', 'daily', 'weekly'].index(default_filters['timeframe'])
        )

        exchanges = get_unique_exchanges(main_db)
        selected_exchanges_input = st.multiselect(
            "Select Exchanges",
            options=exchanges,
            default=default_filters['selected_exchanges'],
            help="Leave empty to show all exchanges"
        )

        search_type_input = st.radio(
            "Search by",
            options=["Ticker", "Company Name"],
            horizontal=True,
            index=0 if default_filters['search_type'] == "Ticker" else 1,
        )

        search_input_value = st.text_input(
            "Search" if search_type_input == "Ticker" else "Search Company",
            value=default_filters['search_input'],
            placeholder="Enter ticker symbol..." if search_type_input == "Ticker" else "Enter company name...",
        )

        available_filtered = get_unique_tickers_by_exchange(main_db, selected_exchanges_input)
        selected_tickers_default = default_filters.get('selected_tickers', [])
        selected_tickers_input = selected_tickers_default

        if search_input_value:
            if search_type_input == "Ticker":
                if ',' in search_input_value:
                    search_tickers = [t.strip().upper() for t in search_input_value.split(',') if t.strip()]
                    selected_tickers_input = [t for t in search_tickers if t in all_tickers]
                    if selected_tickers_input:
                        st.caption(f"Found {len(selected_tickers_input)} ticker(s)")
                    else:
                        st.warning("No matching tickers found")
                else:
                    search_upper = search_input_value.upper().strip()
                    filtered_matches = [t for t in all_tickers if t.upper().startswith(search_upper)]
                    display_tickers = filtered_matches[:100]
                    default_selection = [t for t in selected_tickers_default if t in display_tickers]
                    if not default_selection and search_upper in display_tickers:
                        default_selection = [search_upper]
                    selected_tickers_input = st.multiselect(
                        "Select Tickers",
                        options=display_tickers,
                        default=default_selection,
                        help=f"Showing first 100 of {len(filtered_matches)} matching tickers."
                    )
            else:
                search_lower = search_input_value.lower().strip()
                matching = []
                ticker_display_map = {}
                for ticker in all_tickers:
                    info = main_db.get(ticker)
                    if not info:
                        continue
                    name = info.get('name', '')
                    lower = name.lower()
                    if (lower.startswith(search_lower) or
                            f" {search_lower}" in lower or
                            f"-{search_lower}" in lower or
                            f"({search_lower}" in lower):
                        matching.append(ticker)
                        ticker_display_map[ticker] = f"{ticker} - {name[:50]}"

                matching.sort()
                display_tickers = matching[:100]
                display_options = [ticker_display_map[t] for t in display_tickers]
                default_display = [ticker_display_map[t] for t in selected_tickers_default if t in display_tickers]
                selected_display = st.multiselect(
                    "Select Companies",
                    options=display_options,
                    default=default_display,
                    help=f"Showing first 100 of {len(matching)} matching companies."
                )
                selected_tickers_input = [s.split(' - ')[0] for s in selected_display]
        else:
            default_selection = [t for t in selected_tickers_default if t in available_filtered]
            selected_tickers_input = st.multiselect(
                "Select Tickers",
                options=available_filtered[:100],
                default=default_selection[:100],
                help=f"Showing first 100 of {len(available_filtered)} tickers. Use search to find specific ones."
            )

        max_rows_input = st.number_input(
            "Maximum rows",
            min_value=100,
            max_value=50000,
            value=int(default_filters['max_rows']),
            step=500,
            help="Limits the number of price rows returned in a single query."
        )

        day_filter_input = st.selectbox(
            "Day filter",
            ["All days", "Weekdays only (Mon-Fri)", "Weekends only (Sat-Sun)"],
            index=["All days", "Weekdays only (Mon-Fri)", "Weekends only (Sat-Sun)"].index(
                default_filters.get("day_filter", "All days")
            ),
            help="Filter loaded price rows by weekdays/weekends.",
        )

        apply_filters = st.form_submit_button("Apply Filters")

        if apply_filters:
            st.session_state.pd_filters.update({
                "timeframe": timeframe_input,
                "selected_exchanges": selected_exchanges_input,
                "search_type": search_type_input,
                "search_input": search_input_value,
                "selected_tickers": selected_tickers_input,
                "max_rows": int(max_rows_input),
                "day_filter": day_filter_input,
            })

    filters = st.session_state.pd_filters
    timeframe = filters["timeframe"]
    selected_exchanges = filters["selected_exchanges"]
    selected_tickers = filters.get("selected_tickers", [])
    search_input = filters["search_input"]
    search_type = filters["search_type"]
    max_rows = int(filters["max_rows"])
    day_filter = filters.get("day_filter", "All days")
    available_tickers = get_unique_tickers_by_exchange(main_db, selected_exchanges)

    # Get database date range
    stats = get_database_stats()
    if timeframe == 'hourly' and stats.get('hourly_date_range'):
        min_date, max_date = stats['hourly_date_range']
    elif timeframe == 'daily' and stats.get('daily_date_range'):
        min_date, max_date = stats['daily_date_range']
    elif timeframe == 'weekly' and stats.get('weekly_date_range'):
        min_date, max_date = stats['weekly_date_range']
    else:
        min_date, max_date = None, None

    if min_date and max_date:
        min_date = pd.to_datetime(min_date).date()
        max_date = pd.to_datetime(max_date).date()

        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=min_date,  # Default to first available date
                min_value=min_date,
                max_value=max_date
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date
            )
    else:
        start_date = None
        end_date = None

    # Load data button
    if st.sidebar.button("🔄 Load Data", type="primary", use_container_width=True):
        with st.spinner("Loading price data..."):
            if selected_tickers:
                tickers_to_load = selected_tickers
            elif selected_exchanges:
                tickers_to_load = available_tickers
            else:
                tickers_to_load = None

            with get_database_connection() as conn:
                df = load_price_data(
                    conn,
                    tickers=tickers_to_load,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=timeframe,
                    limit=max_rows,
                    day_filter=day_filter,
                )

            st.session_state.price_data = df

            if tickers_to_load:
                st.success(f"Loaded {len(df):,} records for {len(tickers_to_load)} ticker(s)")
            else:
                st.success(f"Loaded {len(df):,} records")

            if max_rows and len(df) == max_rows:
                st.info(f"Showing the first {max_rows:,} rows. Increase 'Maximum rows' to retrieve more data.")

    # Main content area
    # Database statistics
    st.subheader("📈 Database Statistics")

    if stats:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Hourly Records", f"{stats.get('hourly_records', 0):,}")
            st.metric("Hourly Tickers", f"{stats.get('hourly_tickers', 0):,}")

        with col2:
            st.metric("Daily Records", f"{stats.get('daily_records', 0):,}")
            st.metric("Daily Tickers", f"{stats.get('daily_tickers', 0):,}")

        with col3:
            st.metric("Weekly Records", f"{stats.get('weekly_records', 0):,}")
            st.metric("Weekly Tickers", f"{stats.get('weekly_tickers', 0):,}")

    # Display data
    if 'price_data' in st.session_state and not st.session_state.price_data.empty:
        df = st.session_state.price_data

        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Table", "📈 Charts", "📉 Analysis", "💾 Export"])

        with tab1:
            st.subheader("Price Data")

            # Add company names if available
            if not df.empty and main_db:
                df['company'] = df['ticker'].apply(lambda x: main_db.get(x, {}).get('name', ''))
                df['exchange'] = df['ticker'].apply(lambda x: main_db.get(x, {}).get('exchange', ''))

            # Display options
            col1, col2, col3 = st.columns(3)
            with col1:
                show_columns = st.multiselect(
                    "Show Columns",
                    options=df.columns.tolist(),
                    default=['ticker', 'company', 'date', 'close', 'volume', 'exchange'][:6]
                )
            with col2:
                rows_per_page = st.selectbox(
                    "Rows per page",
                    options=[10, 25, 50, 100, 500],
                    index=1
                )
            with col3:
                sort_column = st.selectbox(
                    "Sort by",
                    options=show_columns,
                    index=min(2, len(show_columns)-1) if 'date' in show_columns else 0
                )

            # Sort data
            df_display = df[show_columns].sort_values(sort_column, ascending=False)

            # Pagination
            total_rows = len(df_display)
            total_pages = (total_rows - 1) // rows_per_page + 1

            page = st.number_input(
                f"Page (1-{total_pages})",
                min_value=1,
                max_value=max(1, total_pages),
                value=1
            )

            start_idx = (page - 1) * rows_per_page
            end_idx = min(start_idx + rows_per_page, total_rows)

            st.dataframe(
                df_display.iloc[start_idx:end_idx],
                use_container_width=True,
                hide_index=True
            )

            st.caption(f"Showing rows {start_idx + 1} to {end_idx} of {total_rows}")

        with tab2:
            st.subheader("Interactive Charts")

            # Select ticker for charting
            chart_tickers = df['ticker'].unique()

            col1, col2 = st.columns([3, 1])
            with col1:
                selected_chart_ticker = st.selectbox(
                    "Select Ticker for Chart",
                    options=sorted(chart_tickers),
                    index=0 if chart_tickers.size > 0 else None
                )
            with col2:
                show_volume = st.checkbox("Show Volume", value=True)

            if selected_chart_ticker:
                # Get company name
                company_name = main_db.get(selected_chart_ticker, {}).get('name', '')
                if company_name:
                    st.write(f"**{selected_chart_ticker}** - {company_name}")

                # Create and display chart
                fig = create_price_chart(df, selected_chart_ticker, show_volume)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)

                # Show latest data for selected ticker
                latest_data = df[df['ticker'] == selected_chart_ticker].iloc[0] if not df[df['ticker'] == selected_chart_ticker].empty else None
                if latest_data is not None:
                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric("Latest Close", f"${latest_data['close']:.2f}")
                    with col2:
                        st.metric("High", f"${latest_data['high']:.2f}")
                    with col3:
                        st.metric("Low", f"${latest_data['low']:.2f}")
                    with col4:
                        if 'volume' in latest_data:
                            st.metric("Volume", f"{latest_data['volume']:,.0f}")
                    with col5:
                        st.metric("Date", latest_data['date'].strftime('%Y-%m-%d'))

        with tab3:
            st.subheader("Data Analysis")

            # Summary statistics
            if len(df['ticker'].unique()) <= 20:  # Only show for reasonable number of tickers
                st.write("### Summary Statistics by Ticker")

                summary_stats = df.groupby('ticker').agg({
                    'close': ['mean', 'std', 'min', 'max'],
                    'volume': 'mean' if 'volume' in df.columns else 'count',
                    'date': ['min', 'max', 'count']
                }).round(2)

                st.dataframe(summary_stats, use_container_width=True)

            # Price changes
            st.write("### Recent Price Changes")

            recent_changes = []
            for ticker in df['ticker'].unique()[:50]:  # Limit to 50 tickers
                ticker_data = df[df['ticker'] == ticker].sort_values('date')
                if len(ticker_data) >= 2:
                    latest = ticker_data.iloc[0]
                    previous = ticker_data.iloc[1]
                    change = latest['close'] - previous['close']
                    change_pct = (change / previous['close']) * 100

                    recent_changes.append({
                        'Ticker': ticker,
                        'Company': main_db.get(ticker, {}).get('name', '')[:50],
                        'Latest Close': latest['close'],
                        'Previous Close': previous['close'],
                        'Change': change,
                        'Change %': change_pct,
                        'Date': latest['date'].strftime('%Y-%m-%d')
                    })

            if recent_changes:
                changes_df = pd.DataFrame(recent_changes)
                changes_df = changes_df.sort_values('Change %', ascending=False)

                # Style the dataframe
                def color_negative_red(val):
                    color = 'red' if val < 0 else 'green'
                    return f'color: {color}'

                styled_df = changes_df.style.applymap(
                    color_negative_red,
                    subset=['Change', 'Change %']
                ).format({
                    'Latest Close': '{:.2f}',
                    'Previous Close': '{:.2f}',
                    'Change': '{:.2f}',
                    'Change %': '{:.2f}%'
                })

                st.dataframe(styled_df, use_container_width=True, hide_index=True)

        with tab4:
            st.subheader("Export Data")

            col1, col2 = st.columns(2)

            with col1:
                # CSV export
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name=f"price_data_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col2:
                # Excel export - completely rewritten to avoid the empty sheet issue
                excel_disabled = True
                excel_data = None

                # Only try to create Excel if we have actual data rows
                if df is not None and isinstance(df, pd.DataFrame) and len(df) > 0:
                    try:
                        import io
                        buffer = io.BytesIO()

                        # Create a copy to ensure we don't modify the original
                        export_df = df.copy()

                        # Write to Excel with error handling
                        try:
                            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                                export_df.to_excel(writer, sheet_name='Price Data', index=False)

                            buffer.seek(0)
                            excel_data = buffer.getvalue()
                            excel_disabled = False
                        except:
                            # If openpyxl fails, try xlsxwriter as fallback
                            buffer = io.BytesIO()
                            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                export_df.to_excel(writer, sheet_name='Price Data', index=False)
                            buffer.seek(0)
                            excel_data = buffer.getvalue()
                            excel_disabled = False

                    except Exception as e:
                        # If all else fails, just show the error
                        st.caption(f"Excel export unavailable: {str(e)[:50]}")
                        excel_disabled = True

                # Show the download button (enabled or disabled based on success)
                if excel_disabled:
                    st.button("📥 Download as Excel", disabled=True, use_container_width=True)
                    if df is None or len(df) == 0:
                        st.caption("No data to export")
                else:
                    st.download_button(
                        label="📥 Download as Excel",
                        data=excel_data,
                        file_name=f"price_data_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

            # Show export summary only if data exists
            if not df.empty:
                st.info(f"""
                **Export Summary:**
                - Records: {len(df):,}
                - Tickers: {df['ticker'].nunique()}
                - Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}
                - Timeframe: {timeframe.capitalize()}
                """)
            else:
                st.warning("No data available for export")

    else:
        # No data loaded yet
        st.info("👈 Use the filters in the sidebar to load price data")

    if 'hourly_stale_df' not in st.session_state:
        st.session_state.hourly_stale_df = None
    if 'hourly_stale_meta' not in st.session_state:
        st.session_state.hourly_stale_meta = {}
    if 'daily_stale_df' not in st.session_state:
        st.session_state.daily_stale_df = None
    if 'weekly_stale_df' not in st.session_state:
        st.session_state.weekly_stale_df = None

    # Stale Data Section - on-demand analysis
    st.divider()
    st.header("⚠️ Stale Data Monitor")
    st.caption("Note: Weekly data is automatically updated when daily data is refreshed")

    # Add tabs for daily and weekly stale data
    stale_tab1, stale_tab2 = st.tabs(["📅 Daily Stale Data", "📆 Weekly Stale Data"])

    with stale_tab1:
        col_title, col_hint = st.columns([3, 1])
        with col_title:
            st.subheader("Daily Tickers with Stale Data")
        with col_hint:
            st.caption("Scan checks for missing latest daily bar (market-aware).")

        if st.button("🔍 Scan Daily Data", key="scan_daily_stale", use_container_width=True):
            with st.spinner("Scanning daily prices for stale tickers..."):
                daily_df = fetch_stale_ticker_dataframe(
                    "daily",
                    stale_after_days=None,
                    limit=None,
                )
                if not daily_df.empty and main_db:
                    daily_df["Company"] = daily_df["Ticker"].apply(lambda t: main_db.get(t, {}).get("name", ""))
                    daily_df["ISIN"] = daily_df["Ticker"].apply(lambda t: main_db.get(t, {}).get("isin", ""))
                    daily_df["Exchange"] = daily_df["Ticker"].apply(lambda t: main_db.get(t, {}).get("exchange", ""))
                    daily_df = _filter_stale_for_closed_markets(daily_df, main_db)
                st.session_state.daily_stale_df = daily_df

        daily_stale_df = st.session_state.daily_stale_df

        if daily_stale_df is None:
            st.info("Click **Scan Daily Data** to analyse your universe for stale daily bars.")
        elif daily_stale_df.empty:
            st.success("✅ No stale daily data found within the selected threshold.")
        else:
            if st.button("🔄 Refresh All", key="refresh_daily_all", help="Updates both daily and weekly data", use_container_width=True):
                payload = daily_stale_df.to_dict(orient="records")
                if payload:
                    st.info(f"Starting update of {len(payload)} stale tickers (daily + weekly data)...")
                    with get_database_connection() as conn:
                        results = refresh_stale_tickers(conn, payload, timeframe='daily')
                    if results['success']:
                        st.success(f"✅ Successfully updated {len(results['success'])} tickers (daily & weekly)")
                    if results['skipped_market_open']:
                        st.info(f"⏸️ Skipped {len(results['skipped_market_open'])} tickers – markets still open")
                    if results['failed']:
                        st.error(f"❌ Failed to update {len(results['failed'])} tickers: {', '.join(results['failed'][:5])}")
                    time.sleep(2)
                    st.session_state.daily_stale_df = None
                    st.rerun()
                else:
                    st.info("No stale daily tickers to refresh.")

            display_df = daily_stale_df.sort_values('Days Old', ascending=False)

            def color_days_old(val: int) -> str:
                if val >= 7:
                    return 'background-color: #ffcccc'
                if val >= 3:
                    return 'background-color: #ffe6cc'
                return 'background-color: #ffffcc'

            styled_df = display_df.style.applymap(color_days_old, subset=['Days Old'])

            st.dataframe(
                styled_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                    "Company": st.column_config.TextColumn("Company", width="medium"),
                    "ISIN": st.column_config.TextColumn("ISIN", width="small"),
                    "Exchange": st.column_config.TextColumn("Exchange", width="small"),
                    "Last Update": st.column_config.TextColumn("Last Update", width="small"),
                    "Days Old": st.column_config.NumberColumn("Days Old", width="small"),
                },
            )

            st.info(
                f"""
                **Daily Stale Data Summary:**
                - Total stale tickers: {len(display_df)}
                - Average days old: {display_df['Days Old'].mean():.1f}
                - Most stale: {display_df['Days Old'].max()} days
                - Last expected trading day: {get_last_trading_day().strftime('%Y-%m-%d')}
                """
            )

    with stale_tab2:
        col_title_w, col_hint_w = st.columns([3, 1])
        with col_title_w:
            st.subheader("Weekly Tickers with Stale Data")
        with col_hint_w:
            st.caption("Resample weekly bars directly from recent daily history; flags if last week’s bar is missing.")

        if st.button("🔍 Scan Weekly Data", key="scan_weekly_stale", use_container_width=True):
            with st.spinner("Scanning weekly prices for stale tickers..."):
                weekly_df = fetch_stale_ticker_dataframe(
                    "weekly",
                    stale_after_days=None,
                    limit=None,
                )
                if not weekly_df.empty and main_db:
                    weekly_df["Company"] = weekly_df["Ticker"].apply(lambda t: main_db.get(t, {}).get("name", ""))
                    weekly_df["Exchange"] = weekly_df["Ticker"].apply(lambda t: main_db.get(t, {}).get("exchange", ""))
                st.session_state.weekly_stale_df = weekly_df

        weekly_stale_df = st.session_state.weekly_stale_df

        if weekly_stale_df is None:
            st.info("Click **Scan Weekly Data** to analyse weekly bars.")
        elif weekly_stale_df.empty:
            st.success("✅ No stale weekly data detected within the selected threshold.")
        else:
            if st.button("📊 Resample Weekly", key="resample_weekly_all", help="Resample weekly data from existing daily data (no API calls)", use_container_width=True):
                payload = weekly_stale_df.to_dict(orient='records')
                if payload:
                    st.info(f"Starting resample of {len(payload)} tickers (weekly only)...")
                    results = resample_weekly_only(payload)
                    if results['success']:
                        st.success(f"✅ Successfully resampled {len(results['success'])} tickers to weekly data")
                    if results['failed']:
                        st.error(f"❌ Failed to resample {len(results['failed'])} tickers: {', '.join(results['failed'][:5])}")
                    time.sleep(2)
                    st.session_state.weekly_stale_df = None
                    st.rerun()
                else:
                    st.info("No stale weekly tickers to resample.")

            st.info("💡 Weekly data can be resampled from existing daily data (no API calls), or automatically updated when you refresh daily data")
            display_weekly_df = weekly_stale_df.sort_values('Days Old', ascending=False)

            def color_days_old_weekly(val: int) -> str:
                if val >= 14:
                    return 'background-color: #ffcccc'
                if val >= 7:
                    return 'background-color: #ffe6cc'
                return 'background-color: #ffffcc'

            styled_weekly_df = display_weekly_df.style.applymap(
                color_days_old_weekly,
                subset=['Days Old']
            )

            st.dataframe(
                styled_weekly_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                    "Company": st.column_config.TextColumn("Company", width="medium"),
                    "Exchange": st.column_config.TextColumn("Exchange", width="small"),
                    "Last Update": st.column_config.TextColumn("Last Update", width="small"),
                    "Days Old": st.column_config.NumberColumn("Days Old", width="small"),
                }
            )

            st.info(
                f"""
                **Weekly Stale Data Summary:**
                - Total stale tickers: {len(display_weekly_df)}
                - Average days old: {display_weekly_df['Days Old'].mean():.1f}
                - Most stale: {display_weekly_df['Days Old'].max()} days
                """
            )

    # Hourly Data Update Section
    st.divider()
    st.header("⏰ Hourly Data Management")
    st.markdown("Update hourly OHLC data for all stocks and ETFs in the database")

    col1, col2, col3 = st.columns(3)

    with col1:
        days_back = st.number_input(
            "Days of Hourly Data to Fetch:",
            min_value=1,
            max_value=1095,  # 3 years
            value=756,  # Default to 756 days (about 2 years)
            help="How many days back to fetch hourly data"
        )

        skip_existing = st.checkbox(
            "Skip existing data (only fetch new)",
            value=True,
            help="If checked, only fetch data that's not already in the database"
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⏰ Update All Hourly Data", key="update_all_hourly", help="Fetch hourly data for all tickers"):
            st.info(f"Starting hourly data update for all tickers ({days_back} days back)...")
            st.warning("⚠️ This will take approximately 1 minute due to API rate limiting (3,000 calls/minute)")

            tickers_to_update = sorted(main_db.keys())
            total_tickers = len(tickers_to_update)
            results = _update_hourly_batch(tickers_to_update, int(days_back), skip_existing=bool(skip_existing))
            success_count = len(results["success"])
            failed_count = len(results["failed"])
            success_rate = (success_count / total_tickers * 100) if total_tickers else 0

            st.success(f"""
            **Hourly Data Update Complete:**
            - Total processed: {total_tickers}
            - Successful: {success_count}
            - Failed: {failed_count}
            - Success rate: {success_rate:.1f}%
            """)
            if failed_count:
                st.error(f"❌ Failed tickers (first 10): {', '.join(results['failed'][:10])}")

    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        st.info(f"""
        **Info:**
        - Rate limit: 3,000 calls/min
        - Est. time: ~1 min for all tickers
        - Data: Hourly OHLC bars
        """)

    st.markdown("### 🔍 Stale Hourly Coverage")
    st.caption("Checks for tickers that are missing the latest hourly bar present in the database.")

    if st.button("🔍 Check for stale hourly data", key="scan_hourly_stale", use_container_width=True):
        with st.spinner("Scanning hourly prices for tickers missing the latest hour..."):
            hourly_df, hourly_meta = fetch_stale_hourly_data(main_db=main_db)
            if not hourly_df.empty and main_db:
                hourly_df["Company"] = hourly_df["Ticker"].apply(lambda t: main_db.get(t, {}).get("name", ""))
                hourly_df["Exchange"] = hourly_df["Ticker"].apply(lambda t: main_db.get(t, {}).get("exchange", ""))
            st.session_state.hourly_stale_df = hourly_df
            st.session_state.hourly_stale_meta = hourly_meta

    hourly_stale_df = st.session_state.hourly_stale_df
    hourly_meta = st.session_state.get("hourly_stale_meta", {})
    latest_hour = hourly_meta.get("latest_hour")

    def _fmt_ts(ts_val):
        if ts_val is None:
            return None
        try:
            return pd.to_datetime(ts_val).tz_localize(None).strftime("%Y-%m-%d %I:%M %p")
        except Exception:
            return str(ts_val)

    latest_display = _fmt_ts(latest_hour)
    latest_raw_display = _fmt_ts(hourly_meta.get("latest_hour_raw"))
    latest_within_display = _fmt_ts(hourly_meta.get("latest_within_now"))
    anchor_display = _fmt_ts(hourly_meta.get("anchor_hour"))

    total_hourly_tickers = int(hourly_meta.get("total_tickers") or 0)
    up_to_date = int(hourly_meta.get("up_to_date") or 0)
    coverage_pct = (up_to_date / total_hourly_tickers * 100) if total_hourly_tickers else 0

    if hourly_stale_df is None:
        st.info("Click **Check for stale hourly data** to find tickers behind the latest available hour.")
    elif not hourly_meta.get("table_exists") and hourly_stale_df.empty:
        st.warning("The 'hourly_prices' table is missing, so hourly freshness cannot be checked.")
    elif latest_hour is None:
        st.warning("No hourly data found to scan yet. Fetch some hourly bars first.")
    elif hourly_stale_df.empty:
        st.success(
            f"✅ All tracked tickers have data at the latest hour ({latest_display}). "
            f"Coverage: {coverage_pct:.1f}% ({up_to_date:,}/{total_hourly_tickers:,})."
        )
    else:
        st.info(
            f"Found {len(hourly_stale_df)} tickers missing the latest hour "
            f"({latest_display}). Coverage: {coverage_pct:.1f}% "
            f"({up_to_date:,}/{total_hourly_tickers:,})."
        )

    if hourly_meta.get("clamped_to_now"):
        st.caption(
            f"Reference hour clamped to current time to ignore far-future timestamps. "
            f"Max timestamp in table: {latest_raw_display}."
        )
    elif latest_raw_display and latest_raw_display != latest_display:
        st.caption(
            f"Using nearest recent hour ({latest_display}) to avoid future-dated rows (raw max: {latest_raw_display})."
        )
    if anchor_display:
        st.caption(f"Reference hour anchored to most common last bar: {anchor_display}.")

    if hourly_stale_df is not None and not hourly_stale_df.empty:
        if st.button(
            "⚡ Update stale hourly data",
            key="update_stale_hourly",
            help="Fetch hourly data only for tickers missing the latest hour",
            use_container_width=True,
        ):
            tickers_to_update = hourly_stale_df["Ticker"].tolist()
            st.info(f"Updating {len(tickers_to_update)} stale hourly tickers with {int(days_back)} days of history...")
            results = _update_hourly_batch(tickers_to_update, int(days_back), skip_existing=bool(skip_existing))
            if results["success"]:
                st.success(f"✅ Updated {len(results['success'])} tickers to the latest hour.")
            if results["failed"]:
                st.error(f"❌ Failed to update {len(results['failed'])} tickers: {', '.join(results['failed'][:10])}")
            st.session_state.hourly_stale_df = None
            st.session_state.hourly_stale_meta = {}

        display_hourly_df = hourly_stale_df.sort_values(["Hours Behind", "Last Hour"], ascending=[False, True])
        display_columns = ["Ticker", "Company", "Exchange", "Last Hour", "Hours Behind"]
        display_columns = [c for c in display_columns if c in display_hourly_df.columns]
        column_cfg = {
            "Ticker": st.column_config.TextColumn("Ticker", width="small"),
            "Last Hour": st.column_config.TextColumn("Last Hour", width="medium"),
            "Hours Behind": st.column_config.NumberColumn("Hours Behind", format="%.1f", width="small"),
        }
        if "Company" in display_hourly_df.columns:
            column_cfg["Company"] = st.column_config.TextColumn("Company", width="medium")
        if "Exchange" in display_hourly_df.columns:
            column_cfg["Exchange"] = st.column_config.TextColumn("Exchange", width="small")

        st.dataframe(
            display_hourly_df[display_columns],
            use_container_width=True,
            hide_index=True,
            column_config=column_cfg,
        )

    # Show hourly data stats
    st.subheader("📊 Hourly Data Statistics")

    # Query hourly data stats
    hourly_stats = pd.DataFrame()
    hourly_table_exists = False
    quality = {}
    with get_database_connection() as conn:
        if _table_exists(conn, "hourly_prices"):
            hourly_table_exists = True
            hourly_stats_query = """
                SELECT
                    COUNT(DISTINCT ticker) as tickers_with_data,
                    COUNT(*) as total_hourly_records,
                    MIN(datetime) as earliest_hour,
                    MAX(datetime) as latest_hour
                FROM hourly_prices
            """
            try:
                hourly_stats = read_sql_with_engine(hourly_stats_query)
            except Exception as exc:
                st.error(f"Error loading hourly stats: {exc}")
                hourly_stats = pd.DataFrame()
            quality = _compute_hourly_data_quality(conn)

    if hourly_table_exists and not hourly_stats.empty and hourly_stats['total_hourly_records'].iloc[0] > 0:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Tickers with Hourly Data", f"{hourly_stats['tickers_with_data'].iloc[0]:,}")

        with col2:
            st.metric("Total Hourly Records", f"{hourly_stats['total_hourly_records'].iloc[0]:,}")

        with col3:
            earliest = hourly_stats['earliest_hour'].iloc[0]
            if pd.notna(earliest):
                earliest_display = pd.to_datetime(earliest).tz_localize(None)
                earliest_display = earliest_display.strftime("%Y-%m-%d %I:%M %p")
            else:
                earliest_display = "N/A"
            st.metric("Earliest Hour", earliest_display)

        with col4:
            latest = hourly_stats['latest_hour'].iloc[0]
            latest_display = "N/A"
            cap_ref = hourly_meta.get("anchor_hour") or pd.Timestamp.utcnow()
            try:
                cap_ref_ts = pd.to_datetime(cap_ref, utc=True)
            except Exception:
                cap_ref_ts = pd.Timestamp.utcnow()
            cap_upper = cap_ref_ts + pd.Timedelta(hours=2)
            if pd.notna(latest):
                latest_ts = pd.to_datetime(latest, utc=True)
                if latest_ts > cap_upper:
                    latest_ts = cap_upper.floor("h")
                    st.caption("Latest hour capped to anchor time to ignore future-dated rows.")
                latest_display = latest_ts.tz_localize(None).strftime("%Y-%m-%d %I:%M %p")
            st.metric("Latest Hour", latest_display)
    elif hourly_table_exists:
        st.info("No hourly data available yet. Click 'Update All Hourly Data' to fetch hourly OHLC data.")
    else:
        st.warning("The 'hourly_prices' table is missing in the configured database. Please complete the PostgreSQL migration.")

    if hourly_table_exists and quality:
        st.markdown("### ✅ Hourly Data Quality")
        q_col1, q_col2, q_col3 = st.columns(3)

        total_tickers = quality.get("total_tickers", 0) or 1  # avoid divide-by-zero
        stale = quality.get("stale_tickers", 0) or 0
        gap_tickers = quality.get("gap_tickers", 0) or 0
        oldest_stale = quality.get("oldest_stale")
        worst_gap = quality.get("worst_gap_hours", 0.0) or 0.0
        worst_calendar_gap = quality.get("worst_calendar_gap_hours", 0.0) or 0.0

        with q_col1:
            st.metric(
                "Stale Tickers (>48h)",
                f"{stale:,}",
                delta=f"-{stale/total_tickers*100:.1f}% of coverage",
            )
            if oldest_stale:
                oldest_display = pd.to_datetime(oldest_stale).tz_localize(None).strftime("%Y-%m-%d %I:%M %p")
                st.caption(f"Oldest last bar: {oldest_display}")
        with q_col2:
            st.metric(
                "Tickers with Gaps >72 trading hours (last 60d)",
                f"{gap_tickers:,}",
            )
            st.caption(f"Worst trading gap: {worst_gap:.0f}h (calendar gap max: {worst_calendar_gap:.0f}h)")
        with q_col3:
            st.info(
                "Stale = no hourly bar in last 48h. Gaps exclude weekend hours (Mon-Fri only) over the past 60 days; thresholds still compare to 72h."
            )

if __name__ == "__main__":
    main()

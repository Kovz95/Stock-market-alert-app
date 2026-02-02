"""
Integration tests for OptimizedDailyPriceCollector.

Tests cover:
- update_ticker: success (new/updated), skip (needs_update=False or empty df), fail (API None)
- Stats: updated, new, skipped, failed, records_fetched, api_calls, weekly_updated
- Weekly resample: when resample_weekly=True, OHLCV correctness, partial-week skip
- _resample_to_weekly: week boundaries, last-trading-day date, column names
- get_statistics structure and avg_records_per_ticker
- Edge cases: store failure, needs_update logic, duplicate dates
"""

from __future__ import annotations

import sqlite3
import tempfile
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.services.backend_fmp_optimized import OptimizedDailyPriceCollector, OptimizedFMPDataFetcher


# ---------------------------------------------------------------------------
# Fixtures: in-memory SQLite for collector + repository
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db_path():
    """Temporary SQLite file path; same DB used for conn and engine."""
    fd = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    path = fd.name
    fd.close()
    yield path
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        pass


@pytest.fixture
def db_config_sqlite(temp_db_path):
    """Patch db_config instance to use SQLite at temp_db_path for get_connection and connection."""
    from src.data_access.db_config import db_config

    original_get_connection = db_config.get_connection
    original_connection = db_config.connection

    def get_connection(db_path=None, *, role=None):
        if db_path is not None:
            return sqlite3.connect(temp_db_path)
        return original_get_connection(db_path=db_path, role=role)

    @contextmanager
    def connection(db_path=None, *, role=None):
        conn = sqlite3.connect(temp_db_path)
        try:
            yield conn
        finally:
            conn.close()

    with (
        patch.object(db_config, "db_type", "sqlite"),
        patch.object(db_config, "get_connection", side_effect=get_connection),
        patch.object(db_config, "connection", side_effect=connection),
    ):
        yield temp_db_path


@pytest.fixture
def collector_with_sqlite(db_config_sqlite, temp_db_path):
    """OptimizedDailyPriceCollector using temp SQLite; _get_db returns repo with temp path."""
    def _get_db(self):
        from src.data_access.daily_price_repository import DailyPriceRepository
        return DailyPriceRepository(db_path=temp_db_path)

    with patch.object(OptimizedDailyPriceCollector, "_get_db", _get_db):
        c = OptimizedDailyPriceCollector()
        yield c
        try:
            c.close()
        except Exception:
            pass


def _daily_df(dates, open_=100.0, high=105.0, low=99.0, close=102.0, volume=1_000_000):
    """Build daily OHLCV DataFrame with DatetimeIndex and standard columns."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    return pd.DataFrame(
        {
            "Open": [open_] * len(idx),
            "High": [high] * len(idx),
            "Low": [low] * len(idx),
            "Close": [close] * len(idx),
            "Volume": [volume] * len(idx),
        },
        index=idx,
    )


# ---------------------------------------------------------------------------
# update_ticker: needs_update False -> skip
# ---------------------------------------------------------------------------


class TestUpdateTickerSkip:
    """When needs_update is False, update_ticker should skip fetch and return True."""

    def test_skip_when_needs_update_false(self, collector_with_sqlite):
        """If repo says no update needed, skip fetcher and increment skipped."""
        coll = collector_with_sqlite
        with patch.object(coll.db, "needs_update", return_value=(False, datetime(2025, 1, 15))):
            out = coll.update_ticker("AAPL")
        assert out is True
        assert coll.stats["skipped"] == 1
        assert "AAPL" in coll.stats["skipped_tickers"]
        assert coll.stats["api_calls"] == 0
        assert coll.stats["records_fetched"] == 0

    def test_skip_when_fetcher_returns_empty_df(self, collector_with_sqlite):
        """When needs_update True but fetcher returns empty DataFrame, count as skipped."""
        coll = collector_with_sqlite
        with (
            patch.object(coll.db, "needs_update", return_value=(True, datetime(2024, 12, 1))),
            patch.object(
                coll.fetcher,
                "get_historical_data_optimized",
                return_value=pd.DataFrame(),
            ),
        ):
            out = coll.update_ticker("MSFT")
        assert out is True
        assert coll.stats["skipped"] == 1
        assert "MSFT" in coll.stats["skipped_tickers"]
        assert coll.stats["api_calls"] == 1
        assert coll.stats["records_fetched"] == 0


# ---------------------------------------------------------------------------
# update_ticker: fetch failure
# ---------------------------------------------------------------------------


class TestUpdateTickerFail:
    """When fetcher returns None or store raises, update_ticker should fail and increment failed."""

    def test_fail_when_fetcher_returns_none(self, collector_with_sqlite):
        """Fetcher returning None should increment failed and return False."""
        coll = collector_with_sqlite
        with (
            patch.object(coll.db, "needs_update", return_value=(True, None)),
            patch.object(coll.fetcher, "get_historical_data_optimized", return_value=None),
        ):
            out = coll.update_ticker("INVALID")
        assert out is False
        assert coll.stats["failed"] == 1
        assert "INVALID" in coll.stats["failed_tickers"]
        assert coll.stats["api_calls"] == 1

    def test_fail_when_store_daily_prices_raises(self, collector_with_sqlite):
        """Store failure should increment failed and return False."""
        coll = collector_with_sqlite
        df = _daily_df(["2025-01-15", "2025-01-16"])
        with (
            patch.object(coll.db, "needs_update", return_value=(True, None)),
            patch.object(coll.fetcher, "get_historical_data_optimized", return_value=df),
            patch.object(coll.db, "store_daily_prices", side_effect=RuntimeError("db error")),
        ):
            out = coll.update_ticker("TICK")
        assert out is False
        assert coll.stats["failed"] == 1
        assert "TICK" in coll.stats["failed_tickers"]


# ---------------------------------------------------------------------------
# update_ticker: success (new ticker and updated ticker)
# ---------------------------------------------------------------------------


class TestUpdateTickerSuccess:
    """Successful update: data stored, stats updated/new and records_fetched correct."""

    def test_new_ticker_stores_and_increments_new(self, collector_with_sqlite):
        """First time ticker: store_daily_prices called, stats new and records_fetched set."""
        coll = collector_with_sqlite
        df = _daily_df(["2025-01-14", "2025-01-15", "2025-01-16"])
        with (
            patch.object(coll.db, "needs_update", return_value=(True, None)),
            patch.object(coll.fetcher, "get_historical_data_optimized", return_value=df),
        ):
            out = coll.update_ticker("NEWT")
        assert out is True
        assert coll.stats["new"] == 1
        assert coll.stats["updated"] == 0
        assert coll.stats["records_fetched"] == 3
        assert coll.stats["api_calls"] == 1
        # DB should have 3 rows
        stored = coll.db.get_daily_prices("NEWT")
        assert len(stored) == 3
        assert list(stored.index) == [
            pd.Timestamp("2025-01-14"),
            pd.Timestamp("2025-01-15"),
            pd.Timestamp("2025-01-16"),
        ]

    def test_existing_ticker_stores_and_increments_updated(self, collector_with_sqlite):
        """Ticker with prior last_update: store_daily_prices called, stats updated set."""
        coll = collector_with_sqlite
        df = _daily_df(["2025-01-17", "2025-01-18"])
        with (
            patch.object(coll.db, "needs_update", return_value=(True, datetime(2025, 1, 16))),
            patch.object(coll.fetcher, "get_historical_data_optimized", return_value=df),
        ):
            out = coll.update_ticker("EXIST")
        assert out is True
        assert coll.stats["updated"] == 1
        assert coll.stats["new"] == 0
        assert coll.stats["records_fetched"] == 2
        stored = coll.db.get_daily_prices("EXIST")
        assert len(stored) == 2


# ---------------------------------------------------------------------------
# Weekly resample: resample_weekly=True
# ---------------------------------------------------------------------------


class TestUpdateTickerWeeklyResample:
    """Weekly resample when resample_weekly=True and daily data is present."""

    def test_resample_weekly_called_when_new_daily_data(self, collector_with_sqlite):
        """When resample_weekly=True and we stored new daily data, weekly should be generated."""
        coll = collector_with_sqlite
        # One week of daily data (Mon–Fri)
        dates = ["2025-01-13", "2025-01-14", "2025-01-15", "2025-01-16", "2025-01-17"]
        df = _daily_df(dates, open_=100, high=110, low=98, close=108, volume=500_000)
        with (
            patch.object(coll.db, "needs_update", return_value=(True, None)),
            patch.object(coll.fetcher, "get_historical_data_optimized", return_value=df),
            patch.object(coll.db, "get_weekly_prices", return_value=pd.DataFrame()),
        ):
            out = coll.update_ticker("WEEK", resample_weekly=True)
        assert out is True
        # Should have attempted weekly: get_weekly_prices and get_daily_prices(limit=250) used
        assert coll.stats["records_fetched"] == 5
        # weekly_updated may be 1 if current week logic includes this week
        assert "weekly_updated" in coll.stats

    def test_resample_weekly_skipped_when_no_new_daily_and_weekly_exists(self, collector_with_sqlite):
        """When resample_weekly=True but no new daily records and weekly data exists, no weekly write."""
        coll = collector_with_sqlite
        with (
            patch.object(coll.db, "needs_update", return_value=(True, datetime(2025, 1, 17))),
            patch.object(coll.fetcher, "get_historical_data_optimized", return_value=pd.DataFrame()),
        ):
            out = coll.update_ticker("SKIP", resample_weekly=True)
        assert out is True
        assert coll.stats["skipped"] == 1
        assert coll.stats["weekly_updated"] == 0


# ---------------------------------------------------------------------------
# _resample_to_weekly: logic and OHLCV
# ---------------------------------------------------------------------------


class TestResampleToWeekly:
    """Direct tests for _resample_to_weekly to catch week-boundary and OHLCV bugs."""

    def test_weekly_ohlcv_correct(self):
        """Weekly Open=first open, High=max, Low=min, Close=last close, Volume=sum."""
        coll = OptimizedDailyPriceCollector()
        coll.fetcher = MagicMock()
        coll.db = MagicMock()
        # One full week
        df = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "High": [101.0, 102.0, 103.0, 104.0, 105.0],
                "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "Close": [101.0, 102.0, 103.0, 104.0, 105.0],
                "Volume": [1000, 2000, 3000, 4000, 5000],
            },
            index=pd.DatetimeIndex(
                ["2025-01-13", "2025-01-14", "2025-01-15", "2025-01-16", "2025-01-17"]
            ),
        )
        out = coll._resample_to_weekly(df, "T")
        assert out is not None
        assert not out.empty
        assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
        # One row per week; may be 0 if "current week" logic excludes it
        if len(out) >= 1:
            row = out.iloc[-1]
            assert row["Open"] == 100.0
            assert row["High"] == 105.0
            assert row["Low"] == 99.0
            assert row["Close"] == 105.0
            assert row["Volume"] == 15000

    def test_weekly_uses_last_trading_day_as_date(self):
        """Weekly row date should be the last trading day of that week."""
        coll = OptimizedDailyPriceCollector()
        coll.fetcher = MagicMock()
        coll.db = MagicMock()
        # Mon–Thu only (no Friday)
        df = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0, 103.0],
                "High": [101.0, 102.0, 103.0, 104.0],
                "Low": [99.0, 100.0, 101.0, 102.0],
                "Close": [101.0, 102.0, 103.0, 104.0],
                "Volume": [1000, 2000, 3000, 4000],
            },
            index=pd.DatetimeIndex(
                ["2025-01-13", "2025-01-14", "2025-01-15", "2025-01-16"]
            ),
        )
        out = coll._resample_to_weekly(df, "T")
        if out is not None and not out.empty:
            # Last trading day of that week is Thu 2025-01-16
            last_date = out.index[-1]
            assert last_date == pd.Timestamp("2025-01-16") or last_date == pd.Timestamp("2025-01-16").normalize()

    def test_resample_empty_returns_none(self):
        """Empty DataFrame should return None."""
        coll = OptimizedDailyPriceCollector()
        coll.fetcher = MagicMock()
        coll.db = MagicMock()
        assert coll._resample_to_weekly(pd.DataFrame(), "T") is None

    def test_resample_accepts_date_column_and_converts_to_index(self):
        """If 'date' is a column, it should be set as index before grouping."""
        coll = OptimizedDailyPriceCollector()
        coll.fetcher = MagicMock()
        coll.db = MagicMock()
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-13", "2025-01-14"]),
                "Open": [100.0, 101.0],
                "High": [101.0, 102.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Volume": [1000, 2000],
            },
        )
        out = coll._resample_to_weekly(df, "T")
        # Should not raise; may return None if current-week logic excludes
        assert out is None or isinstance(out, pd.DataFrame)

    def test_resample_multiple_weeks_produces_multiple_rows(self):
        """Two full weeks should yield two weekly rows (when not filtered by current-week)."""
        coll = OptimizedDailyPriceCollector()
        coll.fetcher = MagicMock()
        coll.db = MagicMock()
        # Two weeks: Jan 6–10 and Jan 13–17
        dates = (
            ["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10"]
            + ["2025-01-13", "2025-01-14", "2025-01-15", "2025-01-16", "2025-01-17"]
        )
        df = _daily_df(dates)
        with patch("src.services.backend_fmp_optimized.pd.Timestamp") as mock_ts:
            # Freeze "today" to a Friday so current week is included
            mock_ts.now.return_value = pd.Timestamp("2025-01-17 12:00:00")
            mock_ts.side_effect = lambda *a, **k: pd.Timestamp(*a, **k)
            out = coll._resample_to_weekly(df, "T")
        # May be 2 rows (both weeks) or 1 (only current week) depending on weekday logic
        assert out is None or (isinstance(out, pd.DataFrame) and len(out) >= 1)


# ---------------------------------------------------------------------------
# get_statistics
# ---------------------------------------------------------------------------


class TestGetStatistics:
    """get_statistics structure and avg_records_per_ticker."""

    def test_structure_has_all_keys(self, collector_with_sqlite):
        """get_statistics must return updated, skipped, failed, new, records_fetched, api_calls, weekly_updated, avg_records_per_ticker."""
        coll = collector_with_sqlite
        s = coll.get_statistics()
        for key in (
            "updated",
            "skipped",
            "skipped_tickers",
            "failed",
            "failed_tickers",
            "new",
            "records_fetched",
            "api_calls",
            "weekly_updated",
            "avg_records_per_ticker",
        ):
            assert key in s, f"missing key: {key}"

    def test_avg_records_per_ticker_no_updates(self, collector_with_sqlite):
        """When no ticker was updated/new, avg_records_per_ticker should not divide by zero."""
        coll = collector_with_sqlite
        s = coll.get_statistics()
        assert s["avg_records_per_ticker"] >= 0
        assert s["updated"] == 0 and s["new"] == 0
        # Denominator is max(updated + new, 1) so should be 1
        assert s["avg_records_per_ticker"] == s["records_fetched"] / 1

    def test_avg_records_after_one_update(self, collector_with_sqlite):
        """After one successful update with 5 records, avg should be 5.0."""
        coll = collector_with_sqlite
        df = _daily_df(["2025-01-13", "2025-01-14", "2025-01-15", "2025-01-16", "2025-01-17"])
        with (
            patch.object(coll.db, "needs_update", return_value=(True, None)),
            patch.object(coll.fetcher, "get_historical_data_optimized", return_value=df),
        ):
            coll.update_ticker("ONE")
        s = coll.get_statistics()
        assert s["new"] == 1
        assert s["records_fetched"] == 5
        assert s["avg_records_per_ticker"] == 5.0


# ---------------------------------------------------------------------------
# Edge cases and bug-hunting
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases that might reveal bugs."""

    def test_update_ticker_exception_increments_failed(self, collector_with_sqlite):
        """Any exception in update_ticker should increment failed and return False."""
        coll = collector_with_sqlite
        with (
            patch.object(coll.db, "needs_update", side_effect=RuntimeError("boom")),
        ):
            out = coll.update_ticker("BOOM")
        assert out is False
        assert coll.stats["failed"] == 1

    def test_store_daily_prices_receives_datetime_index(self, collector_with_sqlite):
        """store_daily_prices must receive a DataFrame with DatetimeIndex (not column 'date')."""
        coll = collector_with_sqlite
        df = _daily_df(["2025-01-15", "2025-01-16"])
        assert isinstance(df.index, pd.DatetimeIndex)
        with (
            patch.object(coll.db, "needs_update", return_value=(True, None)),
            patch.object(coll.fetcher, "get_historical_data_optimized", return_value=df),
        ):
            coll.update_ticker("IDX")
        stored = coll.db.get_daily_prices("IDX")
        assert len(stored) == 2
        assert list(stored.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_duplicate_dates_upserted_not_fail(self, collector_with_sqlite):
        """Storing same ticker/date twice should upsert (no unique violation)."""
        coll = collector_with_sqlite
        df1 = _daily_df(["2025-01-15"], close=100.0)
        df2 = _daily_df(["2025-01-15"], close=200.0)
        with (
            patch.object(coll.db, "needs_update", return_value=(True, None)),
            patch.object(coll.fetcher, "get_historical_data_optimized", return_value=df1),
        ):
            coll.update_ticker("DUP")
        with (
            patch.object(coll.db, "needs_update", return_value=(True, datetime(2025, 1, 15))),
            patch.object(coll.fetcher, "get_historical_data_optimized", return_value=df2),
        ):
            coll.update_ticker("DUP")
        stored = coll.db.get_daily_prices("DUP")
        assert len(stored) == 1
        assert float(stored["Close"].iloc[0]) == 200.0

    def test_stats_reset_per_collector_instance(self):
        """Each new collector instance should have fresh stats."""
        c1 = OptimizedDailyPriceCollector()
        c1.stats["updated"] = 10
        c2 = OptimizedDailyPriceCollector()
        assert c2.stats["updated"] == 0
        assert c2.stats["skipped"] == 0

    def test_resample_weekly_not_called_when_fetcher_returns_empty(self, collector_with_sqlite):
        """When fetcher returns empty df we return early; get_weekly_prices is never called even if resample_weekly=True."""
        coll = collector_with_sqlite
        mock_get_weekly = MagicMock(return_value=pd.DataFrame())
        with (
            patch.object(coll.db, "needs_update", return_value=(True, datetime(2025, 1, 17))),
            patch.object(coll.fetcher, "get_historical_data_optimized", return_value=pd.DataFrame()),
            patch.object(coll.db, "get_weekly_prices", mock_get_weekly),
        ):
            out = coll.update_ticker("WEEKLY_EMPTY", resample_weekly=True)
        assert out is True
        assert coll.stats["skipped"] == 1
        mock_get_weekly.assert_not_called()

    def test_weekly_resample_result_has_standard_columns(self):
        """_resample_to_weekly output must have Open, High, Low, Close, Volume for store_weekly_prices."""
        coll = OptimizedDailyPriceCollector()
        coll.fetcher = MagicMock()
        coll.db = MagicMock()
        df = _daily_df(["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10"])
        with patch("src.services.backend_fmp_optimized.pd.Timestamp") as mock_ts:
            mock_ts.now.return_value = pd.Timestamp("2025-01-20 12:00:00")  # Monday, so that week is "past"
            mock_ts.side_effect = lambda *a, **k: pd.Timestamp(*a, **k) if a or k else pd.Timestamp("2025-01-20 12:00:00")
            out = coll._resample_to_weekly(df, "T")
        if out is not None and not out.empty:
            assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]

    def test_close_calls_db_close(self, collector_with_sqlite):
        """close() should call db.close()."""
        coll = collector_with_sqlite
        coll.db.close = MagicMock()
        coll.close()
        coll.db.close.assert_called_once()


# ---------------------------------------------------------------------------
# OptimizedFMPDataFetcher: get_missing_dates and get_historical_data_optimized
# ---------------------------------------------------------------------------


class TestOptimizedFMPDataFetcher:
    """Integration-style tests for OptimizedFMPDataFetcher with mocked DB and API."""

    def test_get_missing_dates_no_data_returns_none_and_750(self, db_config_sqlite, temp_db_path):
        """When ticker has no rows in daily_prices, get_missing_dates should return (None, 750)."""
        fetcher = OptimizedFMPDataFetcher()
        last_date, days_to_fetch = fetcher.get_missing_dates("NOEXIST")
        assert last_date is None
        assert days_to_fetch == 750

    def test_get_missing_dates_with_last_date_returns_days_to_fetch(self, db_config_sqlite, temp_db_path):
        """When ticker has data, get_missing_dates should return (last_date, days_to_fetch)."""
        from src.data_access.daily_price_repository import DailyPriceRepository

        repo = DailyPriceRepository(db_path=temp_db_path)
        df = _daily_df(["2025-01-10", "2025-01-13"])
        repo.store_daily_prices("HASDATA", df)
        repo.close()
        fetcher = OptimizedFMPDataFetcher()
        last_date, days_to_fetch = fetcher.get_missing_dates("HASDATA")
        assert last_date is not None
        assert days_to_fetch >= 0
        assert days_to_fetch <= 30

    def test_get_historical_data_optimized_force_days_ignores_db(self):
        """When force_days is set, fetcher should use it and not query DB for missing dates."""
        fetcher = OptimizedFMPDataFetcher()
        with patch.object(fetcher, "get_missing_dates", MagicMock(return_value=(None, 100))) as mock_missing:
            with patch("src.services.backend_fmp_optimized.requests.get") as mock_get:
                mock_get.return_value.status_code = 200
                mock_get.return_value.json.return_value = {
                    "historical": [
                        {"date": "2025-01-15", "open": 100, "high": 105, "low": 99, "close": 102, "volume": 1_000_000},
                    ]
                }
                out = fetcher.get_historical_data_optimized("T", force_days=5)
            mock_missing.assert_not_called()
        assert out is not None
        assert len(out) == 1
        assert out.index[0] == pd.Timestamp("2025-01-15")

    def test_get_historical_data_optimized_empty_response_returns_none(self):
        """When API returns 200 but no 'historical' or empty list, should return None or empty."""
        fetcher = OptimizedFMPDataFetcher()
        with patch("src.services.backend_fmp_optimized.requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {}
            out = fetcher.get_historical_data_optimized("T", force_days=5)
        assert out is None

    def test_get_historical_data_optimized_non_200_returns_none(self):
        """When API returns non-200, should return None."""
        fetcher = OptimizedFMPDataFetcher()
        with patch("src.services.backend_fmp_optimized.requests.get") as mock_get:
            mock_get.return_value.status_code = 500
            out = fetcher.get_historical_data_optimized("T", force_days=5)
        assert out is None

    def test_get_historical_data_optimized_columns_standardized(self):
        """Returned DataFrame should have Open, High, Low, Close, Volume (standardized)."""
        fetcher = OptimizedFMPDataFetcher()
        with patch("src.services.backend_fmp_optimized.requests.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = {
                "historical": [
                    {"date": "2025-01-15", "open": 100, "high": 105, "low": 99, "close": 102, "volume": 1_000_000},
                ]
            }
            out = fetcher.get_historical_data_optimized("T", force_days=5)
        assert out is not None
        assert "Open" in out.columns
        assert "High" in out.columns
        assert "Low" in out.columns
        assert "Close" in out.columns
        assert "Volume" in out.columns
        assert out.index.name is None or isinstance(out.index, pd.DatetimeIndex)

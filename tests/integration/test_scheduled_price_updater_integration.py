"""Integration tests for ScheduledPriceUpdater.

Tests cover:
- Ticker resolution by exchange and country (including edge cases)
- Price update flow: batching, stats aggregation, skip vs updated vs failed
- run_scheduled_update with exchanges/countries/empty/duplicates
- update_prices_for_exchanges convenience and close()
- run_alert_checks timeframe filtering and error handling
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.services.scheduled_price_updater import (
    ScheduledPriceUpdater,
    update_prices_for_exchanges,
    run_alert_checks,
)


# ---------------------------------------------------------------------------
# Fixtures: metadata and exchange mapping
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_metadata_map():
    """Metadata keyed by symbol; used by get_tickers_for_exchange/country."""
    return {
        "AAPL": {"symbol": "AAPL", "exchange": "NASDAQ", "name": "Apple"},
        "MSFT": {"symbol": "MSFT", "exchange": "NASDAQ", "name": "Microsoft"},
        "GOOGL": {"symbol": "GOOGL", "exchange": "NASDAQ", "name": "Alphabet"},
        "JPM": {"symbol": "JPM", "exchange": "NYSE", "name": "JPMorgan"},
        "LSE1": {"symbol": "LSE1", "exchange": "LONDON", "name": "UK Co"},
        "LSE2": {"symbol": "LSE2", "exchange": "LONDON", "name": "UK Co 2"},
        "TOK1": {"symbol": "TOK1", "exchange": "TOKYO", "name": "Japan Co"},
    }


@pytest.fixture
def sample_exchange_country_map():
    """Exchange -> country for get_exchanges_for_country / get_tickers_for_country."""
    return {
        "NASDAQ": "United States",
        "NYSE": "United States",
        "LONDON": "United Kingdom",
        "TOKYO": "Japan",
    }


@pytest.fixture
def updater_with_mocked_deps(sample_metadata_map, sample_exchange_country_map):
    """ScheduledPriceUpdater with metadata and exchange mapping patched; collector as mock."""
    with (
        patch(
            "src.services.scheduled_price_updater.fetch_stock_metadata_map",
            return_value=sample_metadata_map,
        ),
        patch(
            "src.services.scheduled_price_updater.EXCHANGE_COUNTRY_MAP",
            sample_exchange_country_map,
        ),
        patch(
            "src.services.scheduled_price_updater.OptimizedDailyPriceCollector",
            MagicMock(),
        ),
    ):
        updater = ScheduledPriceUpdater()
        updater.metadata = sample_metadata_map
        updater.exchange_mapping = sample_exchange_country_map
        yield updater


# ---------------------------------------------------------------------------
# get_tickers_for_exchange
# ---------------------------------------------------------------------------


class TestGetTickersForExchange:
    """Tests for get_tickers_for_exchange."""

    def test_returns_sorted_tickers_for_exchange(self, updater_with_mocked_deps):
        updater = updater_with_mocked_deps
        result = updater.get_tickers_for_exchange("NASDAQ")
        assert result == ["AAPL", "GOOGL", "MSFT"]

    def test_returns_single_ticker_for_exchange(self, updater_with_mocked_deps):
        updater = updater_with_mocked_deps
        result = updater.get_tickers_for_exchange("NYSE")
        assert result == ["JPM"]

    def test_returns_empty_for_unknown_exchange(self, updater_with_mocked_deps):
        updater = updater_with_mocked_deps
        result = updater.get_tickers_for_exchange("UNKNOWN")
        assert result == []

    def test_ignores_non_dict_metadata_entries(self, sample_exchange_country_map):
        """Entries that are not dicts must be skipped to avoid AttributeError."""
        bad_metadata = {
            "AAPL": {"symbol": "AAPL", "exchange": "NASDAQ"},
            "BAD": "not a dict",
            "MSFT": {"symbol": "MSFT", "exchange": "NASDAQ"},
        }
        with (
            patch(
                "src.services.scheduled_price_updater.fetch_stock_metadata_map",
                return_value=bad_metadata,
            ),
            patch(
                "src.services.scheduled_price_updater.EXCHANGE_COUNTRY_MAP",
                sample_exchange_country_map,
            ),
            patch(
                "src.services.scheduled_price_updater.OptimizedDailyPriceCollector",
                MagicMock(),
            ),
        ):
            updater = ScheduledPriceUpdater()
            updater.metadata = bad_metadata
            updater.exchange_mapping = sample_exchange_country_map
            result = updater.get_tickers_for_exchange("NASDAQ")
        assert result == ["AAPL", "MSFT"]

    def test_ignores_entries_without_exchange_key(self, sample_exchange_country_map):
        metadata = {
            "AAPL": {"symbol": "AAPL", "exchange": "NASDAQ"},
            "NOEX": {"symbol": "NOEX"},
        }
        with (
            patch(
                "src.services.scheduled_price_updater.fetch_stock_metadata_map",
                return_value=metadata,
            ),
            patch(
                "src.services.scheduled_price_updater.EXCHANGE_COUNTRY_MAP",
                sample_exchange_country_map,
            ),
            patch(
                "src.services.scheduled_price_updater.OptimizedDailyPriceCollector",
                MagicMock(),
            ),
        ):
            updater = ScheduledPriceUpdater()
            updater.metadata = metadata
            updater.exchange_mapping = sample_exchange_country_map
            result = updater.get_tickers_for_exchange("NASDAQ")
        assert result == ["AAPL"]


# ---------------------------------------------------------------------------
# get_tickers_for_country
# ---------------------------------------------------------------------------


class TestGetTickersForCountry:
    """Tests for get_tickers_for_country."""

    def test_returns_tickers_from_all_exchanges_in_country(self, updater_with_mocked_deps):
        updater = updater_with_mocked_deps
        result = updater.get_tickers_for_country("United States")
        assert sorted(result) == ["AAPL", "GOOGL", "JPM", "MSFT"]

    def test_returns_sorted_list(self, updater_with_mocked_deps):
        updater = updater_with_mocked_deps
        result = updater.get_tickers_for_country("United Kingdom")
        assert result == ["LSE1", "LSE2"]

    def test_returns_empty_for_unknown_country(self, updater_with_mocked_deps):
        updater = updater_with_mocked_deps
        result = updater.get_tickers_for_country("Mars")
        assert result == []

    def test_exchange_matching_is_case_insensitive(self, sample_metadata_map, sample_exchange_country_map):
        """Metadata may have 'nasdaq' or 'NASDAQ'; country lookup uses exchange_mapping keys (e.g. NASDAQ)."""
        metadata = {
            "AAPL": {"symbol": "AAPL", "exchange": "nasdaq"},
            "JPM": {"symbol": "JPM", "exchange": "NYSE"},
        }
        with (
            patch(
                "src.services.scheduled_price_updater.fetch_stock_metadata_map",
                return_value=metadata,
            ),
            patch(
                "src.services.scheduled_price_updater.EXCHANGE_COUNTRY_MAP",
                sample_exchange_country_map,
            ),
            patch(
                "src.services.scheduled_price_updater.OptimizedDailyPriceCollector",
                MagicMock(),
            ),
        ):
            updater = ScheduledPriceUpdater()
            updater.metadata = metadata
            updater.exchange_mapping = sample_exchange_country_map
            result = updater.get_tickers_for_country("United States")
        assert sorted(result) == ["AAPL", "JPM"]


# ---------------------------------------------------------------------------
# get_exchanges_for_country
# ---------------------------------------------------------------------------


class TestGetExchangesForCountry:
    """Tests for get_exchanges_for_country."""

    def test_returns_exchanges_for_country(self, updater_with_mocked_deps):
        updater = updater_with_mocked_deps
        result = updater.get_exchanges_for_country("United States")
        assert sorted(result) == ["NASDAQ", "NYSE"]

    def test_returns_single_exchange(self, updater_with_mocked_deps):
        updater = updater_with_mocked_deps
        result = updater.get_exchanges_for_country("Japan")
        assert result == ["TOKYO"]

    def test_returns_empty_for_unknown_country(self, updater_with_mocked_deps):
        updater = updater_with_mocked_deps
        result = updater.get_exchanges_for_country("Unknown")
        assert result == []


# ---------------------------------------------------------------------------
# update_exchange_prices
# ---------------------------------------------------------------------------


class TestUpdateExchangePrices:
    """Tests for update_exchange_prices: stats, batching, failures, monitor."""

    def test_empty_tickers_returns_initial_stats_no_calls(
        self, updater_with_mocked_deps
    ):
        updater = updater_with_mocked_deps
        result = updater.update_exchange_prices([])
        assert result["total"] == 0
        assert result["updated"] == 0
        assert result["skipped"] == 0
        assert result["failed"] == 0
        # Implementation returns early for empty tickers and does not set duration_seconds
        updater.collector.update_ticker.assert_not_called()

    def test_stats_structure_present(self, updater_with_mocked_deps):
        updater = updater_with_mocked_deps
        updater.collector.update_ticker.return_value = True
        updater.collector.get_statistics.return_value = {
            "skipped_tickers": [],
            "updated": 1,
            "new": 0,
        }
        result = updater.update_exchange_prices(["AAPL"], rate_limit_delay=0)
        assert "total" in result
        assert "updated" in result
        assert "skipped" in result
        assert "skipped_tickers" in result
        assert "failed" in result
        assert "failed_tickers" in result
        assert "new" in result
        assert "stale" in result
        assert "records_fetched" in result
        assert "duration_seconds" in result

    def test_single_ticker_updated_counted_correctly(self, updater_with_mocked_deps):
        updater = updater_with_mocked_deps
        updater.collector.update_ticker.return_value = True
        updater.collector.get_statistics.return_value = {"skipped_tickers": []}
        result = updater.update_exchange_prices(["AAPL"], rate_limit_delay=0)
        assert result["total"] == 1
        assert result["updated"] == 1
        assert result["skipped"] == 0
        assert result["failed"] == 0
        assert result["skipped_tickers"] == []
        assert result["failed_tickers"] == []

    def test_single_ticker_skipped_when_in_collector_skipped_tickers(
        self, updater_with_mocked_deps
    ):
        updater = updater_with_mocked_deps
        updater.collector.update_ticker.return_value = True
        updater.collector.get_statistics.return_value = {"skipped_tickers": ["AAPL"]}
        result = updater.update_exchange_prices(["AAPL"], rate_limit_delay=0)
        assert result["total"] == 1
        assert result["updated"] == 0
        assert result["skipped"] == 1
        assert result["skipped_tickers"] == ["AAPL"]
        assert result["failed"] == 0

    def test_mix_updated_skipped_failed(self, updater_with_mocked_deps):
        """Multiple tickers: some updated, some skipped, one failed; stats and monitor called correctly."""
        updater = updater_with_mocked_deps
        skipped_so_far = []

        def update_ticker(ticker, **kwargs):
            if ticker == "FAIL":
                return False
            return True

        def get_statistics():
            return {"skipped_tickers": list(skipped_so_far)}

        def update_ticker_side_effect(ticker, **kwargs):
            if ticker == "SKIP":
                skipped_so_far.append("SKIP")
            result = update_ticker(ticker, **kwargs)
            return result

        updater.collector.update_ticker.side_effect = update_ticker_side_effect
        updater.collector.get_statistics.side_effect = get_statistics

        result = updater.update_exchange_prices(
            ["AAPL", "SKIP", "MSFT", "FAIL"], rate_limit_delay=0
        )
        assert result["total"] == 4
        assert result["updated"] == 2
        assert result["skipped"] == 1
        assert result["failed"] == 1
        assert "AAPL" not in result["skipped_tickers"]
        assert "SKIP" in result["skipped_tickers"]
        assert "MSFT" not in result["skipped_tickers"]
        assert len(result["failed_tickers"]) == 1
        assert result["failed_tickers"][0]["ticker"] == "FAIL"
        assert "error" in result["failed_tickers"][0]

    def test_exception_during_update_increments_failed_and_appends_error(
        self, updater_with_mocked_deps
    ):
        updater = updater_with_mocked_deps
        updater.collector.update_ticker.side_effect = RuntimeError("API timeout")
        result = updater.update_exchange_prices(["AAPL"], rate_limit_delay=0)
        assert result["failed"] == 1
        assert len(result["failed_tickers"]) == 1
        assert result["failed_tickers"][0]["ticker"] == "AAPL"
        assert "API timeout" in result["failed_tickers"][0]["error"]

    def test_false_return_from_collector_records_error_message(self, updater_with_mocked_deps):
        updater = updater_with_mocked_deps
        updater.collector.update_ticker.return_value = False
        result = updater.update_exchange_prices(["X"], rate_limit_delay=0)
        assert result["failed_tickers"][0]["error"] == "Update returned False"

    def test_batch_size_controls_inner_loop(self, updater_with_mocked_deps):
        updater = updater_with_mocked_deps
        updater.collector.update_ticker.return_value = True
        updater.collector.get_statistics.return_value = {"skipped_tickers": []}
        tickers = ["A", "B", "C", "D", "E"]
        updater.update_exchange_prices(tickers, batch_size=2, rate_limit_delay=0)
        assert updater.collector.update_ticker.call_count == 5
        assert [c[0][0] for c in updater.collector.update_ticker.call_args_list] == [
            "A", "B", "C", "D", "E"
        ]

    def test_resample_weekly_passed_to_collector(self, updater_with_mocked_deps):
        updater = updater_with_mocked_deps
        updater.collector.update_ticker.return_value = True
        updater.collector.get_statistics.return_value = {"skipped_tickers": []}
        updater.update_exchange_prices(["AAPL"], resample_weekly=True, rate_limit_delay=0)
        updater.collector.update_ticker.assert_called_once_with(
            "AAPL", resample_weekly=True
        )

    def test_duration_seconds_recorded(self, updater_with_mocked_deps):
        updater = updater_with_mocked_deps
        updater.collector.update_ticker.return_value = True
        updater.collector.get_statistics.return_value = {"skipped_tickers": []}
        result = updater.update_exchange_prices(["AAPL"], rate_limit_delay=0)
        assert isinstance(result["duration_seconds"], (int, float))
        assert result["duration_seconds"] >= 0


# ---------------------------------------------------------------------------
# run_scheduled_update
# ---------------------------------------------------------------------------


class TestRunScheduledUpdate:
    """Tests for run_scheduled_update: exchanges/countries, dedup, summary."""

    def test_exchanges_only_collects_tickers_and_calls_update(
        self, updater_with_mocked_deps
    ):
        updater = updater_with_mocked_deps
        updater.collector.update_ticker.return_value = True
        updater.collector.get_statistics.return_value = {"skipped_tickers": []}
        result = updater.run_scheduled_update(
            exchanges=["NASDAQ", "NYSE"],
            resample_weekly=False,
        )
        # NASDAQ: AAPL, GOOGL, MSFT; NYSE: JPM -> 4 tickers
        assert result["total"] == 4
        assert "exchanges" in result
        assert "NASDAQ" in result["exchanges"] and "NYSE" in result["exchanges"]

    def test_countries_only_collects_tickers_and_reports(self, updater_with_mocked_deps):
        updater = updater_with_mocked_deps
        updater.collector.update_ticker.return_value = True
        updater.collector.get_statistics.return_value = {"skipped_tickers": []}
        result = updater.run_scheduled_update(
            countries=["United States"],
            resample_weekly=False,
        )
        assert result["total"] == 4
        assert "exchanges" in result

    def test_exchanges_and_countries_merge_tickers_without_duplicates(
        self, updater_with_mocked_deps
    ):
        updater = updater_with_mocked_deps
        updater.collector.update_ticker.return_value = True
        updater.collector.get_statistics.return_value = {"skipped_tickers": []}
        result = updater.run_scheduled_update(
            exchanges=["NASDAQ"],
            countries=["United States"],
            resample_weekly=False,
        )
        # NASDAQ + US (NASDAQ+NYSE) -> 4 unique tickers
        assert result["total"] == 4
        assert len(result.get("exchanges", [])) >= 1

    def test_empty_exchanges_and_countries_still_calls_update(
        self, updater_with_mocked_deps
    ):
        updater = updater_with_mocked_deps
        result = updater.run_scheduled_update(
            exchanges=[],
            countries=[],
        )
        assert result["total"] == 0
        assert result["updated"] == 0

    def test_duplicate_exchanges_deduplicate_tickers(self, updater_with_mocked_deps):
        updater = updater_with_mocked_deps
        updater.collector.update_ticker.return_value = True
        updater.collector.get_statistics.return_value = {"skipped_tickers": []}
        result = updater.run_scheduled_update(
            exchanges=["NASDAQ", "NASDAQ"],
            resample_weekly=False,
        )
        assert result["total"] == 3


# ---------------------------------------------------------------------------
# update_prices_for_exchanges (convenience) and close
# ---------------------------------------------------------------------------


class TestUpdatePricesForExchanges:
    """Tests for module-level update_prices_for_exchanges and close."""

    @patch(
        "src.services.scheduled_price_updater.fetch_stock_metadata_map",
        return_value={"AAPL": {"symbol": "AAPL", "exchange": "NASDAQ"}},
    )
    @patch(
        "src.services.scheduled_price_updater.EXCHANGE_COUNTRY_MAP",
        {"NASDAQ": "United States"},
    )
    @patch("src.services.scheduled_price_updater.OptimizedDailyPriceCollector")
    def test_creates_updater_runs_and_closes(
        self, mock_collector_cls, *_ignore
    ):
        mock_collector = MagicMock()
        mock_collector.update_ticker.return_value = True
        mock_collector.get_statistics.return_value = {"skipped_tickers": []}
        mock_collector_cls.return_value = mock_collector
        mock_collector.close = MagicMock()

        result = update_prices_for_exchanges(["NASDAQ"], resample_weekly=False)

        assert result["total"] == 1
        mock_collector.close.assert_called_once()

    @patch(
        "src.services.scheduled_price_updater.fetch_stock_metadata_map",
        return_value={},
    )
    @patch("src.services.scheduled_price_updater.EXCHANGE_COUNTRY_MAP", {})
    @patch("src.services.scheduled_price_updater.OptimizedDailyPriceCollector")
    def test_close_called_even_if_run_raises(
        self, mock_collector_cls, *_ignore
    ):
        mock_collector = MagicMock()
        mock_collector_cls.return_value = mock_collector
        mock_collector.close = MagicMock()
        from src.services.scheduled_price_updater import ScheduledPriceUpdater

        with patch.object(
            ScheduledPriceUpdater,
            "run_scheduled_update",
            side_effect=RuntimeError("fail"),
        ):
            with pytest.raises(RuntimeError):
                update_prices_for_exchanges(["NASDAQ"])
        mock_collector.close.assert_called_once()


class TestClose:
    """Tests for ScheduledPriceUpdater.close."""

    def test_close_calls_collector_close_when_present(self, updater_with_mocked_deps):
        updater = updater_with_mocked_deps
        updater.collector.close = MagicMock()
        updater.close()
        updater.collector.close.assert_called_once()

    def test_close_safe_when_collector_has_no_close(self, sample_metadata_map, sample_exchange_country_map):
        with (
            patch(
                "src.services.scheduled_price_updater.fetch_stock_metadata_map",
                return_value=sample_metadata_map,
            ),
            patch(
                "src.services.scheduled_price_updater.EXCHANGE_COUNTRY_MAP",
                sample_exchange_country_map,
            ),
            patch(
                "src.services.scheduled_price_updater.OptimizedDailyPriceCollector",
                MagicMock(spec=[]),
            ),
        ):
            updater = ScheduledPriceUpdater()
            updater.close()


# ---------------------------------------------------------------------------
# run_alert_checks
# ---------------------------------------------------------------------------


class TestRunAlertChecks:
    """Tests for run_alert_checks: timeframe filtering, totals, exception handling."""

    def test_returns_stats_structure_when_no_alerts(self):
        with (
            patch(
                "src.services.scheduled_price_updater.fetch_stock_metadata_map",
                return_value={"AAPL": {"symbol": "AAPL", "exchange": "NASDAQ"}},
            ),
            patch(
                "src.data_access.alert_repository.list_alerts",
                return_value=[],
            ),
        ):
            result = run_alert_checks(["NASDAQ"], timeframe_key="daily")
        assert result["total"] == 0
        assert result["success"] == 0
        assert result["triggered"] == 0
        assert result["errors"] == 0

    def test_daily_timeframe_includes_only_daily_alerts(self):
        alerts = [
            {
                "alert_id": "1",
                "ticker": "AAPL",
                "exchange": "NASDAQ",
                "timeframe": "daily",
                "action": "on",
            },
            {
                "alert_id": "2",
                "ticker": "MSFT",
                "exchange": "NASDAQ",
                "timeframe": "weekly",
                "action": "on",
            },
        ]
        with (
            patch(
                "src.services.scheduled_price_updater.fetch_stock_metadata_map",
                return_value={"AAPL": {"exchange": "NASDAQ"}, "MSFT": {"exchange": "NASDAQ"}},
            ),
            patch(
                "src.data_access.alert_repository.list_alerts",
                return_value=alerts,
            ),
        ):
            result = run_alert_checks(["NASDAQ"], timeframe_key="daily")
        assert result["total"] == 1

    def test_weekly_timeframe_includes_only_weekly_alerts(self):
        alerts = [
            {"alert_id": "1", "ticker": "AAPL", "exchange": "NASDAQ", "timeframe": "daily"},
            {"alert_id": "2", "ticker": "MSFT", "exchange": "NASDAQ", "timeframe": "1wk"},
        ]
        with (
            patch(
                "src.services.scheduled_price_updater.fetch_stock_metadata_map",
                return_value={"AAPL": {"exchange": "NASDAQ"}, "MSFT": {"exchange": "NASDAQ"}},
            ),
            patch(
                "src.data_access.alert_repository.list_alerts",
                return_value=alerts,
            ),
        ):
            result = run_alert_checks(["NASDAQ"], timeframe_key="weekly")
        assert result["total"] == 1

    def test_hourly_timeframe_includes_hourly_1h_1hr(self):
        alerts = [
            {"alert_id": "1", "ticker": "AAPL", "exchange": "NASDAQ", "timeframe": "hourly"},
            {"alert_id": "2", "ticker": "MSFT", "exchange": "NASDAQ", "timeframe": "1h"},
            {"alert_id": "3", "ticker": "GOOGL", "exchange": "NASDAQ", "timeframe": "1hr"},
            {"alert_id": "4", "ticker": "JPM", "exchange": "NASDAQ", "timeframe": "daily"},
        ]
        with (
            patch(
                "src.services.scheduled_price_updater.fetch_stock_metadata_map",
                return_value={
                    "AAPL": {"exchange": "NASDAQ"},
                    "MSFT": {"exchange": "NASDAQ"},
                    "GOOGL": {"exchange": "NASDAQ"},
                    "JPM": {"exchange": "NASDAQ"},
                },
            ),
            patch(
                "src.data_access.alert_repository.list_alerts",
                return_value=alerts,
            ),
        ):
            result = run_alert_checks(["NASDAQ"], timeframe_key="hourly")
        assert result["total"] == 3

    def test_exception_sets_errors_and_returns_stats(self):
        with (
            patch(
                "src.services.scheduled_price_updater.fetch_stock_metadata_map",
                side_effect=RuntimeError("DB down"),
            ),
        ):
            result = run_alert_checks(["NASDAQ"], timeframe_key="daily")
        assert result["errors"] == 1
        assert result["total"] == 0

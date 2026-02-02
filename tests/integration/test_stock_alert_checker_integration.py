"""Integration tests for StockAlertChecker.

Tests cover:
- Condition extraction (all formats: list of strings, dicts, nested lists)
- Condition evaluation (AND/OR, empty conditions, backend integration)
- should_skip_alert (today vs yesterday, timezone, invalid dates)
- get_price_data (timeframe mapping, cache, FMP integration)
- check_alert (disabled, skip today, no ticker, ratio, no data, triggered flow)
- check_alerts (timeframe filter, stats aggregation)
- check_all_alerts / check_alerts_for_exchanges
- Edge cases that may reveal logic bugs
"""

from __future__ import annotations

from datetime import datetime, date, timedelta, timezone
from unittest.mock import MagicMock, patch, call

import pandas as pd
import pytest

from src.services.stock_alert_checker import (
    StockAlertChecker,
    check_stock_alerts,
)


# ---------------------------------------------------------------------------
# Fixtures: sample data
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_ohlcv_df():
    """OHLCV DataFrame suitable for condition evaluation (Close[-1] etc.)."""
    return pd.DataFrame({
        "Open": [98.0, 99.0, 100.0, 101.0, 102.0],
        "High": [99.0, 100.0, 101.0, 102.0, 103.0],
        "Low": [97.0, 98.0, 99.0, 100.0, 101.0],
        "Close": [98.5, 99.5, 100.5, 101.5, 102.5],
        "Volume": [1_000_000] * 5,
    })


@pytest.fixture
def sample_alert():
    """Minimal alert dict with conditions that can be evaluated."""
    return {
        "alert_id": "test-1",
        "ticker": "AAPL",
        "name": "Test Alert",
        "conditions": [{"conditions": "Close[-1] > 0"}],
        "combination_logic": "AND",
        "timeframe": "1d",
        "action": "on",
    }


# ---------------------------------------------------------------------------
# extract_conditions
# ---------------------------------------------------------------------------


class TestExtractConditions:
    """Tests for extract_conditions logic (all supported formats)."""

    def test_list_of_dicts_with_conditions_key(self):
        """Dict format: {"conditions": "Close[-1] > 100"} is extracted."""
        checker = StockAlertChecker()
        alert = {"conditions": [{"conditions": "Close[-1] > 100"}, {"conditions": "Open[-1] < 200"}]}
        out = checker.extract_conditions(alert)
        assert out == ["Close[-1] > 100", "Open[-1] < 200"]

    def test_list_of_strings(self):
        """Plain list of condition strings is extracted."""
        checker = StockAlertChecker()
        alert = {"conditions": ["Close[-1] > 100", "Volume[-1] > 0"]}
        out = checker.extract_conditions(alert)
        assert out == ["Close[-1] > 100", "Volume[-1] > 0"]

    def test_nested_list_format(self):
        """Nested list format [["Close[-1] > 100"]] is extracted (first element as string)."""
        checker = StockAlertChecker()
        alert = {"conditions": [["Close[-1] > 100"], ["Open[-1] < 200"]]}
        out = checker.extract_conditions(alert)
        assert out == ["Close[-1] > 100", "Open[-1] < 200"]

    def test_empty_conditions_returns_empty_list(self):
        """Missing or empty conditions returns []."""
        checker = StockAlertChecker()
        assert checker.extract_conditions({}) == []
        assert checker.extract_conditions({"conditions": []}) == []
        assert checker.extract_conditions({"conditions": None}) == []

    def test_dict_with_empty_conditions_key_skipped(self):
        """Dict entries with empty 'conditions' key are not added."""
        checker = StockAlertChecker()
        alert = {"conditions": [{"conditions": "A > 1"}, {"conditions": ""}, {"conditions": "B < 2"}]}
        out = checker.extract_conditions(alert)
        assert out == ["A > 1", "B < 2"]

    def test_non_list_conditions_returns_empty(self):
        """If conditions is not a list (e.g. a string), implementation returns [] (only iterates when list)."""
        checker = StockAlertChecker()
        alert = {"conditions": "Close[-1] > 100"}  # string, not list
        out = checker.extract_conditions(alert)
        assert isinstance(out, list)
        assert out == []

    def test_mixed_list_dict_and_string(self):
        """Mix of dict and string entries both contribute."""
        checker = StockAlertChecker()
        alert = {"conditions": [{"conditions": "A > 1"}, "B < 2"]}
        out = checker.extract_conditions(alert)
        assert out == ["A > 1", "B < 2"]


# ---------------------------------------------------------------------------
# evaluate_alert
# ---------------------------------------------------------------------------


class TestEvaluateAlert:
    """Tests for evaluate_alert (real backend expression evaluation)."""

    def test_single_condition_true(self, sample_ohlcv_df, sample_alert):
        """Condition Close[-1] > 0 is True for sample data (last Close 102.5)."""
        checker = StockAlertChecker()
        sample_alert["conditions"] = [{"conditions": "Close[-1] > 0"}]
        assert checker.evaluate_alert(sample_alert, sample_ohlcv_df) is True

    def test_single_condition_false(self, sample_ohlcv_df, sample_alert):
        """Condition Close[-1] > 1000 is False."""
        checker = StockAlertChecker()
        sample_alert["conditions"] = [{"conditions": "Close[-1] > 1000"}]
        assert checker.evaluate_alert(sample_alert, sample_ohlcv_df) is False

    def test_no_conditions_returns_false(self, sample_ohlcv_df, sample_alert):
        """Alert with no conditions returns False."""
        checker = StockAlertChecker()
        sample_alert["conditions"] = []
        assert checker.evaluate_alert(sample_alert, sample_ohlcv_df) is False

    def test_and_combination_both_true(self, sample_ohlcv_df, sample_alert):
        """AND with both conditions True returns True."""
        checker = StockAlertChecker()
        sample_alert["conditions"] = [
            {"conditions": "Close[-1] > 0"},
            {"conditions": "Close[-1] < 10000"},
        ]
        sample_alert["combination_logic"] = "AND"
        assert checker.evaluate_alert(sample_alert, sample_ohlcv_df) is True

    def test_and_combination_one_false(self, sample_ohlcv_df, sample_alert):
        """AND with one condition False returns False."""
        checker = StockAlertChecker()
        sample_alert["conditions"] = [
            {"conditions": "Close[-1] > 0"},
            {"conditions": "Close[-1] > 1000"},
        ]
        sample_alert["combination_logic"] = "AND"
        assert checker.evaluate_alert(sample_alert, sample_ohlcv_df) is False

    def test_or_combination_one_true(self, sample_ohlcv_df, sample_alert):
        """OR with one condition True returns True."""
        checker = StockAlertChecker()
        sample_alert["conditions"] = [
            {"conditions": "Close[-1] > 1000"},
            {"conditions": "Close[-1] > 0"},
        ]
        sample_alert["combination_logic"] = "OR"
        assert checker.evaluate_alert(sample_alert, sample_ohlcv_df) is True

    def test_or_combination_both_false(self, sample_ohlcv_df, sample_alert):
        """OR with both False returns False."""
        checker = StockAlertChecker()
        sample_alert["conditions"] = [
            {"conditions": "Close[-1] > 1000"},
            {"conditions": "Close[-1] < 0"},
        ]
        sample_alert["combination_logic"] = "OR"
        assert checker.evaluate_alert(sample_alert, sample_ohlcv_df) is False

    def test_combination_logic_default_and(self, sample_ohlcv_df, sample_alert):
        """Missing combination_logic is treated as AND (backend default)."""
        checker = StockAlertChecker()
        sample_alert["conditions"] = [
            {"conditions": "Close[-1] > 0"},
            {"conditions": "Close[-1] < 10000"},
        ]
        sample_alert.pop("combination_logic", None)
        assert checker.evaluate_alert(sample_alert, sample_ohlcv_df) is True

    def test_empty_dataframe_raises_or_returns_false(self, sample_alert):
        """Empty DataFrame may cause backend to return False or raise; we expect False from evaluate_alert."""
        checker = StockAlertChecker()
        sample_alert["conditions"] = [{"conditions": "Close[-1] > 0"}]
        empty_df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        # Backend might raise or return False; evaluate_alert catches and returns False
        result = checker.evaluate_alert(sample_alert, empty_df)
        assert result is False


# ---------------------------------------------------------------------------
# should_skip_alert
# ---------------------------------------------------------------------------


class TestShouldSkipAlert:
    """Tests for should_skip_alert (last_triggered today vs other)."""

    def test_no_last_triggered_does_not_skip(self):
        """Missing last_triggered -> False (do not skip)."""
        checker = StockAlertChecker()
        assert checker.should_skip_alert({}) is False
        assert checker.should_skip_alert({"last_triggered": None}) is False
        assert checker.should_skip_alert({"last_triggered": ""}) is False

    def test_triggered_today_skips(self):
        """last_triggered set to today (ISO) -> True (skip)."""
        checker = StockAlertChecker()
        today_iso = date.today().isoformat() + "T12:00:00"
        assert checker.should_skip_alert({"last_triggered": today_iso}) is True

    def test_triggered_yesterday_does_not_skip(self):
        """last_triggered yesterday -> False (do not skip)."""
        checker = StockAlertChecker()
        yesterday = date.today() - timedelta(days=1)
        yesterday_iso = yesterday.isoformat() + "T12:00:00"
        assert checker.should_skip_alert({"last_triggered": yesterday_iso}) is False

    def test_triggered_with_z_timezone_parsed(self):
        """ISO string with Z suffix is parsed (UTC); skip if that date is today in local date."""
        checker = StockAlertChecker()
        # Use today in UTC at midnight so local date might still be today or yesterday
        today_utc = datetime.now(timezone.utc).date().isoformat() + "T00:00:00Z"
        # After replace Z with +00:00, fromisoformat gives UTC; .date() is UTC date. Then compare with datetime.now().date() (local).
        # So if we're in US evening, UTC might be next day -> skip. If we're in UK morning, UTC might be same day -> skip.
        # Just ensure we don't crash and that we get a bool.
        result = checker.should_skip_alert({"last_triggered": today_utc})
        assert isinstance(result, bool)

    def test_invalid_last_triggered_does_not_skip(self):
        """Invalid last_triggered string -> False (do not skip), no exception."""
        checker = StockAlertChecker()
        assert checker.should_skip_alert({"last_triggered": "not-a-date"}) is False
        assert checker.should_skip_alert({"last_triggered": "2024-13-45"}) is False


# ---------------------------------------------------------------------------
# format_alert_message
# ---------------------------------------------------------------------------


class TestFormatAlertMessage:
    """Tests for format_alert_message output structure."""

    def test_message_contains_ticker_price_timeframe(self, sample_ohlcv_df, sample_alert):
        """Formatted message includes ticker, price, timeframe, conditions."""
        checker = StockAlertChecker()
        msg = checker.format_alert_message(sample_alert, sample_ohlcv_df)
        assert "AAPL" in msg or "Test Alert" in msg
        assert "102.50" in msg  # last Close
        assert "1d" in msg or "daily" in msg
        assert "Close[-1] > 0" in msg or "Conditions" in msg

    def test_empty_dataframe_uses_zero_price(self, sample_alert):
        """Empty DataFrame -> price 0 in message."""
        checker = StockAlertChecker()
        empty_df = pd.DataFrame(columns=["Close"])
        msg = checker.format_alert_message(sample_alert, empty_df)
        assert "$0.00" in msg or "0.00" in msg


# ---------------------------------------------------------------------------
# get_price_data (timeframe mapping and cache)
# ---------------------------------------------------------------------------


class TestGetPriceData:
    """Tests for get_price_data timeframe mapping and caching."""

    @patch("src.services.stock_alert_checker.FMPDataFetcher")
    def test_daily_timeframe_calls_fmp_daily(self, mock_fmp_class):
        """Default/1d timeframe calls get_historical_data with period 1day (no timeframe for daily)."""
        mock_fmp = MagicMock()
        mock_fmp.get_historical_data.return_value = pd.DataFrame({"Close": [100.0]})
        mock_fmp_class.return_value = mock_fmp

        checker = StockAlertChecker()
        checker.get_price_data("AAPL", "1d")

        mock_fmp.get_historical_data.assert_called_once()
        args = mock_fmp.get_historical_data.call_args[0]
        kwargs = mock_fmp.get_historical_data.call_args[1]
        assert args[0] == "AAPL"
        assert kwargs.get("period") == "1day"

    @patch("src.services.stock_alert_checker.FMPDataFetcher")
    def test_weekly_timeframe_calls_fmp_weekly(self, mock_fmp_class):
        """timeframe 1wk or weekly calls get_historical_data with period=1day, timeframe=1wk."""
        mock_fmp = MagicMock()
        mock_fmp.get_historical_data.return_value = pd.DataFrame({"Close": [100.0]})
        mock_fmp_class.return_value = mock_fmp

        checker = StockAlertChecker()
        checker.get_price_data("AAPL", "1wk")
        mock_fmp.get_historical_data.assert_called_once()
        args, kwargs = mock_fmp.get_historical_data.call_args
        assert args[0] == "AAPL"
        assert kwargs.get("period") == "1day" and kwargs.get("timeframe") == "1wk"

        mock_fmp.reset_mock()
        checker.get_price_data("MSFT", "weekly")
        assert mock_fmp.get_historical_data.called

    @patch("src.services.stock_alert_checker.FMPDataFetcher")
    def test_hourly_timeframe_calls_fmp_hourly(self, mock_fmp_class):
        """timeframe 1h or hourly calls get_historical_data with period=1hour."""
        mock_fmp = MagicMock()
        mock_fmp.get_historical_data.return_value = pd.DataFrame({"Close": [100.0]})
        mock_fmp_class.return_value = mock_fmp

        checker = StockAlertChecker()
        checker.get_price_data("AAPL", "1h")
        mock_fmp.get_historical_data.assert_called_once()
        args = mock_fmp.get_historical_data.call_args[0]
        kwargs = mock_fmp.get_historical_data.call_args[1]
        assert args[0] == "AAPL"
        assert kwargs.get("period") == "1hour"

    @patch("src.services.stock_alert_checker.FMPDataFetcher")
    def test_cache_returns_same_frame(self, mock_fmp_class):
        """Second call for same ticker/timeframe returns cached DataFrame without new FMP call."""
        mock_fmp = MagicMock()
        df = pd.DataFrame({"Close": [100.0]})
        mock_fmp.get_historical_data.return_value = df.copy()
        mock_fmp_class.return_value = mock_fmp

        checker = StockAlertChecker()
        first = checker.get_price_data("AAPL", "1d")
        second = checker.get_price_data("AAPL", "1d")
        assert first is second
        mock_fmp.get_historical_data.assert_called_once()

    @patch("src.services.stock_alert_checker.FMPDataFetcher")
    def test_none_from_fmp_returns_none(self, mock_fmp_class):
        """FMP returning None yields None from get_price_data."""
        mock_fmp = MagicMock()
        mock_fmp.get_historical_data.return_value = None
        mock_fmp_class.return_value = mock_fmp

        checker = StockAlertChecker()
        assert checker.get_price_data("INVALID", "1d") is None

    @patch("src.services.stock_alert_checker.FMPDataFetcher")
    def test_empty_dataframe_returns_none(self, mock_fmp_class):
        """FMP returning empty DataFrame yields None."""
        mock_fmp = MagicMock()
        mock_fmp.get_historical_data.return_value = pd.DataFrame()
        mock_fmp_class.return_value = mock_fmp

        checker = StockAlertChecker()
        assert checker.get_price_data("TICK", "1d") is None


# ---------------------------------------------------------------------------
# check_alert (full flow)
# ---------------------------------------------------------------------------


class TestCheckAlert:
    """Integration tests for check_alert flow."""

    @patch("src.services.stock_alert_checker.log_completion")
    @patch("src.services.stock_alert_checker.log_conditions_evaluated")
    @patch("src.services.stock_alert_checker.log_price_data_pulled")
    @patch("src.services.stock_alert_checker.log_alert_check_start", return_value="audit-1")
    def test_disabled_alert_skipped(
        self, mock_start, _mock_pulled, _mock_eval, _mock_completion
    ):
        """action=off -> skipped, skip_reason=disabled, no FMP/Discord/update."""
        checker = StockAlertChecker()
        alert = {
            "alert_id": "a1",
            "ticker": "AAPL",
            "action": "off",
            "conditions": [{"conditions": "Close[-1] > 0"}],
            "timeframe": "1d",
        }
        with patch.object(checker, "get_price_data") as mock_get:
            result = checker.check_alert(alert)
        assert result["skipped"] is True
        assert result.get("skip_reason") == "disabled"
        assert result["triggered"] is False
        mock_get.assert_not_called()

    @patch("src.services.stock_alert_checker.log_completion")
    @patch("src.services.stock_alert_checker.log_conditions_evaluated")
    @patch("src.services.stock_alert_checker.log_price_data_pulled")
    @patch("src.services.stock_alert_checker.log_alert_check_start", return_value="audit-1")
    def test_skip_already_triggered_today(
        self, mock_start, _mock_pulled, _mock_eval, _mock_completion
    ):
        """last_triggered today -> skipped, no FMP call."""
        checker = StockAlertChecker()
        today_iso = date.today().isoformat() + "T10:00:00"
        alert = {
            "alert_id": "a1",
            "ticker": "AAPL",
            "action": "on",
            "last_triggered": today_iso,
            "conditions": [{"conditions": "Close[-1] > 0"}],
            "timeframe": "1d",
        }
        with patch.object(checker, "get_price_data") as mock_get:
            result = checker.check_alert(alert)
        assert result["skipped"] is True
        assert result.get("skip_reason") == "already_triggered_today"
        mock_get.assert_not_called()

    @patch("src.services.stock_alert_checker.log_completion")
    @patch("src.services.stock_alert_checker.log_error")
    @patch("src.services.stock_alert_checker.log_alert_check_start", return_value="audit-1")
    def test_no_ticker_returns_error(self, mock_start, _mock_err, _mock_completion):
        """Missing ticker (and not ratio) -> error, no FMP."""
        checker = StockAlertChecker()
        alert = {
            "alert_id": "a1",
            "ticker": "",
            "action": "on",
            "conditions": [{"conditions": "Close[-1] > 0"}],
            "timeframe": "1d",
        }
        with patch.object(checker, "get_price_data") as mock_get:
            result = checker.check_alert(alert)
        assert result["error"] == "No ticker specified"
        assert result["triggered"] is False
        mock_get.assert_not_called()

    @patch("src.services.stock_alert_checker.log_completion")
    @patch("src.services.stock_alert_checker.log_error")
    @patch("src.services.stock_alert_checker.log_alert_check_start", return_value="audit-1")
    def test_ratio_alert_missing_ticker2_returns_error(self, mock_start, _mock_err, _mock_completion):
        """Ratio alert with missing ticker1 or ticker2 -> error."""
        checker = StockAlertChecker()
        alert = {
            "alert_id": "a1",
            "ticker": "",
            "ticker1": "AAPL",
            "ticker2": "",
            "ratio": "Yes",
            "action": "on",
            "conditions": [{"conditions": "Close[-1] > 0"}],
            "timeframe": "1d",
        }
        with patch.object(checker, "get_price_data") as mock_get:
            result = checker.check_alert(alert)
        assert "ticker" in result["error"].lower() or "Missing" in result["error"]
        mock_get.assert_not_called()

    @patch("src.services.stock_alert_checker.log_completion")
    @patch("src.services.stock_alert_checker.log_no_data_available")
    @patch("src.services.stock_alert_checker.log_alert_check_start", return_value="audit-1")
    def test_no_price_data_returns_error(self, mock_start, _mock_no_data, _mock_completion):
        """get_price_data returns None -> error 'No price data for ...'."""
        checker = StockAlertChecker()
        alert = {
            "alert_id": "a1",
            "ticker": "AAPL",
            "action": "on",
            "conditions": [{"conditions": "Close[-1] > 0"}],
            "timeframe": "1d",
        }
        with patch.object(checker, "get_price_data", return_value=None):
            result = checker.check_alert(alert)
        assert result["error"] == "No price data for AAPL"
        assert result["triggered"] is False

    @patch("src.services.stock_alert_checker.update_alert")
    @patch("src.services.stock_alert_checker.send_economy_discord_alert", return_value=True)
    @patch("src.services.stock_alert_checker.log_completion")
    @patch("src.services.stock_alert_checker.log_conditions_evaluated")
    @patch("src.services.stock_alert_checker.log_price_data_pulled")
    @patch("src.services.stock_alert_checker.log_alert_check_start", return_value="audit-1")
    def test_triggered_calls_discord_and_update_alert(
        self, mock_start, _mock_pulled, _mock_eval, _mock_completion, mock_discord, mock_update
    ):
        """When conditions met: Discord sent and update_alert(last_triggered) called."""
        checker = StockAlertChecker()
        df = pd.DataFrame({"Close": [100.0], "Open": [99.0], "High": [101.0], "Low": [98.0], "Volume": [1e6]})
        alert = {
            "alert_id": "a1",
            "ticker": "AAPL",
            "name": "Test",
            "action": "on",
            "conditions": [{"conditions": "Close[-1] > 0"}],
            "combination_logic": "AND",
            "timeframe": "1d",
        }
        with patch.object(checker, "get_price_data", return_value=df):
            result = checker.check_alert(alert)
        assert result["triggered"] is True
        mock_discord.assert_called_once()
        args = mock_discord.call_args[0]
        assert args[0] == alert
        assert "Test" in args[1] or "AAPL" in args[1]
        mock_update.assert_called_once_with("a1", {"last_triggered": mock_update.call_args[0][1]["last_triggered"]})
        assert "last_triggered" in mock_update.call_args[0][1]

    @patch("src.services.stock_alert_checker.update_alert")
    @patch("src.services.stock_alert_checker.send_economy_discord_alert")
    @patch("src.services.stock_alert_checker.log_completion")
    @patch("src.services.stock_alert_checker.log_conditions_evaluated")
    @patch("src.services.stock_alert_checker.log_price_data_pulled")
    @patch("src.services.stock_alert_checker.log_alert_check_start", return_value="audit-1")
    def test_not_triggered_does_not_call_update_alert(
        self, mock_start, _mock_pulled, _mock_eval, _mock_completion, mock_discord, mock_update
    ):
        """When conditions not met: update_alert and Discord not called."""
        checker = StockAlertChecker()
        df = pd.DataFrame({"Close": [100.0], "Open": [99.0], "High": [101.0], "Low": [98.0], "Volume": [1e6]})
        alert = {
            "alert_id": "a1",
            "ticker": "AAPL",
            "action": "on",
            "conditions": [{"conditions": "Close[-1] > 10000"}],
            "timeframe": "1d",
        }
        with patch.object(checker, "get_price_data", return_value=df):
            result = checker.check_alert(alert)
        assert result["triggered"] is False
        mock_discord.assert_not_called()
        mock_update.assert_not_called()

    @patch("src.services.stock_alert_checker.update_alert")
    @patch("src.services.stock_alert_checker.send_economy_discord_alert", return_value=True)
    @patch("src.services.stock_alert_checker.log_completion")
    @patch("src.services.stock_alert_checker.log_conditions_evaluated")
    @patch("src.services.stock_alert_checker.log_price_data_pulled")
    @patch("src.services.stock_alert_checker.log_alert_check_start", return_value="audit-1")
    def test_ratio_alert_uses_ticker1_for_data_fetch(
        self, mock_start, _mock_pulled, _mock_eval, _mock_completion, _mock_discord, _mock_update
    ):
        """Ratio alert with ticker1/ticker2 uses ticker1 for get_price_data (not ticker)."""
        checker = StockAlertChecker()
        df = pd.DataFrame({"Close": [100.0], "Open": [99.0], "High": [101.0], "Low": [98.0], "Volume": [1e6]})
        alert = {
            "alert_id": "a1",
            "ticker": "",
            "ticker1": "AAPL",
            "ticker2": "MSFT",
            "ratio": "Yes",
            "action": "on",
            "conditions": [{"conditions": "Close[-1] > 0"}],
            "timeframe": "1d",
        }
        with patch.object(checker, "get_price_data", return_value=df) as mock_get:
            checker.check_alert(alert)
        mock_get.assert_called_once()
        assert mock_get.call_args[0][0] == "AAPL"

    @patch("src.services.stock_alert_checker.log_completion")
    @patch("src.services.stock_alert_checker.log_error")
    @patch("src.services.stock_alert_checker.log_alert_check_start", return_value="audit-1")
    def test_is_ratio_missing_ticker2_returns_error(self, mock_start, _mock_err, _mock_completion):
        """is_ratio=True with missing ticker2 returns error (same as ratio='Yes')."""
        checker = StockAlertChecker()
        alert = {
            "alert_id": "a1",
            "ticker": "X",
            "ticker1": "AAPL",
            "ticker2": "",
            "is_ratio": True,
            "action": "on",
            "conditions": [{"conditions": "Close[-1] > 0"}],
            "timeframe": "1d",
        }
        with patch.object(checker, "get_price_data") as mock_get:
            result = checker.check_alert(alert)
        assert "ticker" in result["error"].lower() or "Missing" in result["error"]
        mock_get.assert_not_called()

    @patch("src.services.stock_alert_checker.log_completion")
    @patch("src.services.stock_alert_checker.log_conditions_evaluated")
    @patch("src.services.stock_alert_checker.log_price_data_pulled")
    @patch("src.services.stock_alert_checker.log_alert_check_start", return_value="audit-1")
    def test_alert_without_action_key_not_skipped(self, mock_start, _mock_pulled, _mock_eval, _mock_completion):
        """Alert with no 'action' key is treated as on (not skipped)."""
        checker = StockAlertChecker()
        alert = {
            "alert_id": "a1",
            "ticker": "AAPL",
            "conditions": [{"conditions": "Close[-1] > 10000"}],
            "timeframe": "1d",
        }
        alert.pop("action", None)
        df = pd.DataFrame({"Close": [100.0], "Open": [99.0], "High": [101.0], "Low": [98.0], "Volume": [1e6]})
        with patch.object(checker, "get_price_data", return_value=df):
            result = checker.check_alert(alert)
        assert result.get("skipped") is not True
        assert "skip_reason" not in result or result.get("skip_reason") != "disabled"


# ---------------------------------------------------------------------------
# check_alerts (timeframe filter and stats)
# ---------------------------------------------------------------------------


class TestCheckAlerts:
    """Tests for check_alerts timeframe filtering and stats."""

    @patch("src.services.stock_alert_checker.StockAlertChecker.check_alert")
    def test_timeframe_filter_daily_includes_only_daily_and_1d(self, mock_check_alert):
        """timeframe_filter='daily' only includes alerts with timeframe daily or 1d."""
        mock_check_alert.return_value = {"triggered": False, "skipped": False, "error": None}
        checker = StockAlertChecker()
        alerts = [
            {"alert_id": "1", "ticker": "A", "timeframe": "daily", "action": "on"},
            {"alert_id": "2", "ticker": "B", "timeframe": "1d", "action": "on"},
            {"alert_id": "3", "ticker": "C", "timeframe": "weekly", "action": "on"},
        ]
        stats = checker.check_alerts(alerts, timeframe_filter="daily")
        assert stats["total"] == 2
        assert mock_check_alert.call_count == 2
        # Check that weekly was not passed
        called_ids = [call[0][0]["alert_id"] for call in mock_check_alert.call_args_list]
        assert "3" not in called_ids

    @patch("src.services.stock_alert_checker.StockAlertChecker.check_alert")
    def test_timeframe_filter_weekly_includes_only_weekly_and_1wk(self, mock_check_alert):
        """timeframe_filter='weekly' only includes weekly and 1wk."""
        mock_check_alert.return_value = {"triggered": False, "skipped": False, "error": None}
        checker = StockAlertChecker()
        alerts = [
            {"alert_id": "1", "ticker": "A", "timeframe": "weekly", "action": "on"},
            {"alert_id": "2", "ticker": "B", "timeframe": "1wk", "action": "on"},
            {"alert_id": "3", "ticker": "C", "timeframe": "daily", "action": "on"},
        ]
        stats = checker.check_alerts(alerts, timeframe_filter="weekly")
        assert stats["total"] == 2
        called_ids = [call[0][0]["alert_id"] for call in mock_check_alert.call_args_list]
        assert "3" not in called_ids

    @patch("src.services.stock_alert_checker.StockAlertChecker.check_alert")
    def test_timeframe_filter_hourly_includes_1h_1hr_hourly(self, mock_check_alert):
        """timeframe_filter='hourly' includes hourly, 1h, 1hr."""
        mock_check_alert.return_value = {"triggered": False, "skipped": False, "error": None}
        checker = StockAlertChecker()
        alerts = [
            {"alert_id": "1", "ticker": "A", "timeframe": "hourly", "action": "on"},
            {"alert_id": "2", "ticker": "B", "timeframe": "1h", "action": "on"},
            {"alert_id": "3", "ticker": "C", "timeframe": "1hr", "action": "on"},
            {"alert_id": "4", "ticker": "D", "timeframe": "daily", "action": "on"},
        ]
        stats = checker.check_alerts(alerts, timeframe_filter="hourly")
        assert stats["total"] == 3
        called_ids = [call[0][0]["alert_id"] for call in mock_check_alert.call_args_list]
        assert "4" not in called_ids

    @patch("src.services.stock_alert_checker.StockAlertChecker.check_alert")
    def test_stats_aggregation(self, mock_check_alert):
        """Stats correctly count triggered, skipped, errors, no_data, success."""
        checker = StockAlertChecker()
        alerts = [
            {"alert_id": "1", "ticker": "A", "timeframe": "daily", "action": "on"},
            {"alert_id": "2", "ticker": "B", "timeframe": "daily", "action": "on"},
            {"alert_id": "3", "ticker": "C", "timeframe": "daily", "action": "on"},
        ]
        mock_check_alert.side_effect = [
            {"triggered": True, "skipped": False, "error": None},
            {"triggered": False, "skipped": True, "error": None},
            {"triggered": False, "skipped": False, "error": "No price data for C"},
        ]
        stats = checker.check_alerts(alerts)
        assert stats["total"] == 3
        assert stats["triggered"] == 1
        assert stats["skipped"] == 1
        assert stats["no_data"] == 1
        assert stats["errors"] == 0
        # success = count of non-skipped, non-error checks that completed (triggered or not)
        assert stats["success"] == 1  # only the triggered one; skipped and no_data do not add to success

    @patch("src.services.stock_alert_checker.StockAlertChecker.check_alert")
    def test_non_dict_alerts_skipped_from_total(self, mock_check_alert):
        """Alerts that are not dicts are not counted in total and check_alert not called for them."""
        checker = StockAlertChecker()
        alerts = [
            {"alert_id": "1", "ticker": "A", "timeframe": "daily", "action": "on"},
            None,
            "not-a-dict",
        ]
        mock_check_alert.return_value = {"triggered": False, "skipped": False, "error": None}
        stats = checker.check_alerts(alerts)
        assert stats["total"] == 1
        mock_check_alert.assert_called_once()

    @patch("src.services.stock_alert_checker.StockAlertChecker.check_alert")
    def test_default_timeframe_lowercase_daily(self, mock_check_alert):
        """Alert with no timeframe defaults to 'daily' for filter (get('timeframe','daily').lower())."""
        mock_check_alert.return_value = {"triggered": False, "skipped": False, "error": None}
        checker = StockAlertChecker()
        alerts = [
            {"alert_id": "1", "ticker": "A", "action": "on"},  # no timeframe
        ]
        stats = checker.check_alerts(alerts, timeframe_filter="daily")
        assert stats["total"] == 1
        mock_check_alert.assert_called_once()
        assert mock_check_alert.call_args[0][0].get("timeframe", "daily").lower() == "daily"


# ---------------------------------------------------------------------------
# check_all_alerts / check_alerts_for_exchanges
# ---------------------------------------------------------------------------


class TestCheckAllAlerts:
    """Tests for check_all_alerts loading and filtering."""

    @patch("src.services.stock_alert_checker.StockAlertChecker.check_alerts")
    @patch("src.services.stock_alert_checker.list_alerts")
    def test_loads_alerts_and_filters_action_on(self, mock_list, mock_check_alerts):
        """check_all_alerts loads from list_alerts and only checks alerts with action=on."""
        mock_list.return_value = [
            {"alert_id": "1", "ticker": "A", "action": "on", "timeframe": "daily"},
            {"alert_id": "2", "ticker": "B", "action": "off", "timeframe": "daily"},
        ]
        mock_check_alerts.return_value = {"total": 1, "triggered": 0, "errors": 0, "skipped": 0, "no_data": 0, "success": 1}
        checker = StockAlertChecker()
        checker.check_all_alerts()
        mock_list.assert_called_once()
        call_args = mock_check_alerts.call_args[0]
        assert len(call_args[0]) == 1
        assert call_args[0][0]["alert_id"] == "1"

    @patch("src.services.stock_alert_checker.list_alerts")
    def test_exception_returns_error_stats(self, mock_list):
        """If list_alerts raises, check_all_alerts returns error stats."""
        mock_list.side_effect = RuntimeError("DB error")
        checker = StockAlertChecker()
        stats = checker.check_all_alerts()
        assert stats["errors"] == 1
        assert stats["total"] == 0


class TestCheckAlertsForExchanges:
    """Tests for check_alerts_for_exchanges."""

    @patch("src.services.stock_alert_checker.StockAlertChecker.check_alerts")
    @patch("src.services.stock_alert_checker.list_alerts")
    def test_filters_by_exchange(self, mock_list, mock_check_alerts):
        """Only alerts whose exchange is in the list are checked."""
        mock_list.return_value = [
            {"alert_id": "1", "ticker": "A", "exchange": "NASDAQ", "timeframe": "daily"},
            {"alert_id": "2", "ticker": "B", "exchange": "NYSE", "timeframe": "daily"},
        ]
        mock_check_alerts.return_value = {"total": 1, "triggered": 0, "errors": 0, "skipped": 0, "no_data": 0, "success": 1}
        checker = StockAlertChecker()
        checker.check_alerts_for_exchanges(["NASDAQ"])
        call_args = mock_check_alerts.call_args[0]
        assert len(call_args[0]) == 1
        assert call_args[0][0]["exchange"] == "NASDAQ"


# ---------------------------------------------------------------------------
# check_stock_alerts convenience
# ---------------------------------------------------------------------------


class TestCheckStockAlertsFunction:
    """Tests for module-level check_stock_alerts."""

    @patch("src.services.stock_alert_checker.StockAlertChecker")
    def test_with_exchanges_calls_check_alerts_for_exchanges(self, mock_checker_class):
        """check_stock_alerts(exchanges=[...]) uses check_alerts_for_exchanges."""
        mock_checker = MagicMock()
        mock_checker.check_alerts_for_exchanges.return_value = {"total": 0}
        mock_checker_class.return_value = mock_checker
        result = check_stock_alerts(exchanges=["NASDAQ"], timeframe_filter="daily")
        mock_checker.check_alerts_for_exchanges.assert_called_once_with(["NASDAQ"], "daily")
        assert result["total"] == 0

    @patch("src.services.stock_alert_checker.StockAlertChecker")
    def test_without_exchanges_calls_check_all_alerts(self, mock_checker_class):
        """check_stock_alerts() uses check_all_alerts."""
        mock_checker = MagicMock()
        mock_checker.check_all_alerts.return_value = {"total": 0}
        mock_checker_class.return_value = mock_checker
        result = check_stock_alerts(timeframe_filter="weekly")
        mock_checker.check_all_alerts.assert_called_once_with("weekly")
        assert result["total"] == 0

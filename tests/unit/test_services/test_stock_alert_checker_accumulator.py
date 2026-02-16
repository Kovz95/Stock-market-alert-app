"""
Unit tests for the DiscordMessageAccumulator integration in
src/services/stock_alert_checker.py

Covers:
- check_alert(): uses accumulator.add() when accumulator is provided
- check_alert(): adds to both primary and custom webhook URLs
- check_alert(): falls back to async queue when accumulator is None
- check_alert(): skips accumulator when alert does not trigger
- check_alerts(): creates accumulator and passes it to check_alert
- check_alerts(): calls flush_all() after all alerts are processed
- check_alerts(): logs accumulator stats
- check_alerts(): accumulator works in both sequential and parallel modes

Note: The test_services conftest mocks src.services.stock_alert_checker in
sys.modules to prevent database imports at collection time.  We use
importlib to load the real module inside fixtures after all session-scoped
mocks are in place.
"""

from __future__ import annotations

import importlib
import sys
import pytest
from unittest.mock import MagicMock, Mock, patch, call

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Fixtures – import the real StockAlertChecker module
# ---------------------------------------------------------------------------

@pytest.fixture()
def _real_checker_module():
    """
    Import the real stock_alert_checker module, bypassing the session-scoped
    MagicMock that the test_services conftest installs.

    The module's transitive imports (backend, backend_fmp, alert_repository,
    metadata_repository, alert_audit_logger) are all patched via sys.modules
    so no database or API connections are created.
    """
    # Save the session mock so we can restore it afterward
    saved = sys.modules.pop("src.services.stock_alert_checker", None)

    # Ensure heavy transitive dependencies are mocked
    deps_to_mock = [
        "src.services.backend",
        "src.services.backend_fmp",
        "src.data_access.alert_repository",
        "src.data_access.metadata_repository",
        "src.services.alert_audit_logger",
        "pytz",
    ]
    saved_deps = {}
    for dep in deps_to_mock:
        if dep not in sys.modules:
            saved_deps[dep] = None
            sys.modules[dep] = MagicMock()
        else:
            saved_deps[dep] = sys.modules[dep]

    # pytz.timezone needs to return a callable (used in format_alert_message)
    sys.modules["pytz"].timezone = MagicMock(return_value=MagicMock())

    try:
        mod = importlib.import_module("src.services.stock_alert_checker")
        importlib.reload(mod)  # force a fresh load with current sys.modules
        yield mod
    finally:
        # Restore the session mock
        if saved is not None:
            sys.modules["src.services.stock_alert_checker"] = saved
        elif "src.services.stock_alert_checker" in sys.modules:
            del sys.modules["src.services.stock_alert_checker"]

        # Restore deps
        for dep, original in saved_deps.items():
            if original is None and dep in sys.modules:
                del sys.modules[dep]
            elif original is not None:
                sys.modules[dep] = original


@pytest.fixture()
def StockAlertChecker(_real_checker_module):
    """Return the real StockAlertChecker class."""
    return _real_checker_module.StockAlertChecker


@pytest.fixture()
def sample_df():
    """Minimal OHLCV DataFrame for testing."""
    np.random.seed(42)
    n = 20
    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "Open": np.random.uniform(100, 200, n),
            "High": np.random.uniform(100, 200, n),
            "Low": np.random.uniform(100, 200, n),
            "Close": np.random.uniform(100, 200, n),
            "Volume": np.random.randint(1_000_000, 10_000_000, n),
        },
        index=dates,
    )


@pytest.fixture()
def triggered_alert():
    """An alert dict that will be treated as triggered."""
    return {
        "alert_id": "test-123",
        "ticker": "AAPL",
        "name": "Apple Buy Signal",
        "action": "on",
        "timeframe": "1d",
        "conditions": [{"conditions": "Close[-1] > 100"}],
        "combination_logic": "AND",
        "last_triggered": "",
    }


@pytest.fixture()
def non_triggered_alert():
    """An alert dict that will NOT trigger."""
    return {
        "alert_id": "test-456",
        "ticker": "MSFT",
        "name": "Microsoft Sell Signal",
        "action": "on",
        "timeframe": "1d",
        "conditions": [{"conditions": "Close[-1] < 0"}],
        "combination_logic": "AND",
        "last_triggered": "",
    }


WEBHOOK_PRIMARY = "https://discord.com/api/webhooks/111/primary"
WEBHOOK_CUSTOM = "https://discord.com/api/webhooks/222/custom"


# ---------------------------------------------------------------------------
# check_alert() with accumulator
# ---------------------------------------------------------------------------

class TestCheckAlertWithAccumulator:
    """Tests that check_alert uses the accumulator when provided."""

    def test_triggered_alert_adds_to_accumulator(
        self, StockAlertChecker, triggered_alert, sample_df
    ):
        """When an alert triggers and accumulator is provided, it should add embeds."""
        accumulator = MagicMock()

        with (
            patch.object(StockAlertChecker, "get_price_data", return_value=sample_df) as _,
            patch.object(StockAlertChecker, "evaluate_alert", return_value=True),
            patch.object(StockAlertChecker, "format_alert_message", return_value="msg"),
            patch(
                "src.services.stock_alert_checker.resolve_alert_webhook_url",
                return_value=WEBHOOK_PRIMARY,
            ),
            patch(
                "src.services.stock_alert_checker.resolve_alert_custom_webhook_urls",
                return_value=[],
            ),
            patch(
                "src.services.stock_alert_checker.format_alert_as_embed",
                return_value={"title": "embed"},
            ),
            patch("src.services.stock_alert_checker.log_alert_check_start", return_value="aud1"),
            patch("src.services.stock_alert_checker.log_price_data_pulled"),
            patch("src.services.stock_alert_checker.log_conditions_evaluated"),
            patch("src.services.stock_alert_checker.log_completion"),
            patch("src.services.stock_alert_checker.update_alert"),
        ):
            checker = StockAlertChecker(async_discord=False)
            result = checker.check_alert(triggered_alert, accumulator=accumulator)

        assert result["triggered"] is True
        accumulator.add.assert_called_once_with(WEBHOOK_PRIMARY, {"title": "embed"})

    def test_triggered_alert_adds_custom_urls(
        self, StockAlertChecker, triggered_alert, sample_df
    ):
        """Triggered alert should add embeds for custom webhook URLs too."""
        accumulator = MagicMock()

        with (
            patch.object(StockAlertChecker, "get_price_data", return_value=sample_df),
            patch.object(StockAlertChecker, "evaluate_alert", return_value=True),
            patch.object(StockAlertChecker, "format_alert_message", return_value="msg"),
            patch(
                "src.services.stock_alert_checker.resolve_alert_webhook_url",
                return_value=WEBHOOK_PRIMARY,
            ),
            patch(
                "src.services.stock_alert_checker.resolve_alert_custom_webhook_urls",
                return_value=[WEBHOOK_CUSTOM],
            ),
            patch(
                "src.services.stock_alert_checker.format_alert_as_embed",
                return_value={"title": "embed"},
            ),
            patch("src.services.stock_alert_checker.log_alert_check_start", return_value="aud1"),
            patch("src.services.stock_alert_checker.log_price_data_pulled"),
            patch("src.services.stock_alert_checker.log_conditions_evaluated"),
            patch("src.services.stock_alert_checker.log_completion"),
            patch("src.services.stock_alert_checker.update_alert"),
        ):
            checker = StockAlertChecker(async_discord=False)
            result = checker.check_alert(triggered_alert, accumulator=accumulator)

        assert result["triggered"] is True
        assert accumulator.add.call_count == 2
        accumulator.add.assert_any_call(WEBHOOK_PRIMARY, {"title": "embed"})
        accumulator.add.assert_any_call(WEBHOOK_CUSTOM, {"title": "embed"})

    def test_non_triggered_alert_does_not_add(
        self, StockAlertChecker, non_triggered_alert, sample_df
    ):
        """When an alert does NOT trigger, nothing should be added to the accumulator."""
        accumulator = MagicMock()

        with (
            patch.object(StockAlertChecker, "get_price_data", return_value=sample_df),
            patch.object(StockAlertChecker, "evaluate_alert", return_value=False),
            patch("src.services.stock_alert_checker.log_alert_check_start", return_value="aud1"),
            patch("src.services.stock_alert_checker.log_price_data_pulled"),
            patch("src.services.stock_alert_checker.log_conditions_evaluated"),
            patch("src.services.stock_alert_checker.log_completion"),
        ):
            checker = StockAlertChecker(async_discord=False)
            result = checker.check_alert(non_triggered_alert, accumulator=accumulator)

        assert result["triggered"] is False
        accumulator.add.assert_not_called()

    def test_skips_accumulator_when_webhook_is_none(
        self, StockAlertChecker, triggered_alert, sample_df
    ):
        """If webhook URL resolves to None, accumulator.add should not be called."""
        accumulator = MagicMock()

        with (
            patch.object(StockAlertChecker, "get_price_data", return_value=sample_df),
            patch.object(StockAlertChecker, "evaluate_alert", return_value=True),
            patch.object(StockAlertChecker, "format_alert_message", return_value="msg"),
            patch(
                "src.services.stock_alert_checker.resolve_alert_webhook_url",
                return_value=None,
            ),
            patch(
                "src.services.stock_alert_checker.resolve_alert_custom_webhook_urls",
                return_value=[],
            ),
            patch(
                "src.services.stock_alert_checker.format_alert_as_embed",
                return_value={"title": "embed"},
            ),
            patch("src.services.stock_alert_checker.log_alert_check_start", return_value="aud1"),
            patch("src.services.stock_alert_checker.log_price_data_pulled"),
            patch("src.services.stock_alert_checker.log_conditions_evaluated"),
            patch("src.services.stock_alert_checker.log_completion"),
            patch("src.services.stock_alert_checker.update_alert"),
        ):
            checker = StockAlertChecker(async_discord=False)
            result = checker.check_alert(triggered_alert, accumulator=accumulator)

        # Should not add when webhook_url is None (guard: if webhook_url and embed)
        accumulator.add.assert_not_called()


# ---------------------------------------------------------------------------
# check_alert() fallback to async queue
# ---------------------------------------------------------------------------

class TestCheckAlertFallback:
    """Tests that check_alert falls back to async queue when no accumulator."""

    def test_async_queue_used_when_no_accumulator(
        self, StockAlertChecker, triggered_alert, sample_df
    ):
        """Without accumulator, triggered alerts should use queue_discord_notification."""
        with (
            patch.object(StockAlertChecker, "get_price_data", return_value=sample_df),
            patch.object(StockAlertChecker, "evaluate_alert", return_value=True),
            patch.object(StockAlertChecker, "format_alert_message", return_value="msg"),
            patch(
                "src.services.stock_alert_checker.resolve_alert_webhook_url",
                return_value=WEBHOOK_PRIMARY,
            ),
            patch(
                "src.services.stock_alert_checker.resolve_alert_custom_webhook_urls",
                return_value=[],
            ),
            patch(
                "src.services.stock_alert_checker.format_alert_as_embed",
                return_value={"title": "embed"},
            ),
            patch(
                "src.services.stock_alert_checker.queue_discord_notification",
                return_value=True,
            ) as mock_queue,
            patch("src.services.stock_alert_checker.log_alert_check_start", return_value="aud1"),
            patch("src.services.stock_alert_checker.log_price_data_pulled"),
            patch("src.services.stock_alert_checker.log_conditions_evaluated"),
            patch("src.services.stock_alert_checker.log_completion"),
            patch("src.services.stock_alert_checker.update_alert"),
        ):
            checker = StockAlertChecker(async_discord=True)
            result = checker.check_alert(triggered_alert, accumulator=None)

        assert result["triggered"] is True
        mock_queue.assert_called()

    def test_sync_send_when_async_disabled_and_no_accumulator(
        self, StockAlertChecker, triggered_alert, sample_df
    ):
        """With async_discord=False and no accumulator, should use sync send."""
        with (
            patch.object(StockAlertChecker, "get_price_data", return_value=sample_df),
            patch.object(StockAlertChecker, "evaluate_alert", return_value=True),
            patch.object(StockAlertChecker, "format_alert_message", return_value="msg"),
            patch(
                "src.services.stock_alert_checker.send_economy_discord_alert",
                return_value=True,
            ) as mock_sync_send,
            patch("src.services.stock_alert_checker.log_alert_check_start", return_value="aud1"),
            patch("src.services.stock_alert_checker.log_price_data_pulled"),
            patch("src.services.stock_alert_checker.log_conditions_evaluated"),
            patch("src.services.stock_alert_checker.log_completion"),
            patch("src.services.stock_alert_checker.update_alert"),
        ):
            checker = StockAlertChecker(async_discord=False)
            result = checker.check_alert(triggered_alert, accumulator=None)

        assert result["triggered"] is True
        mock_sync_send.assert_called_once()


# ---------------------------------------------------------------------------
# check_alerts() – accumulator lifecycle
# ---------------------------------------------------------------------------

class TestCheckAlertsAccumulatorLifecycle:
    """Tests that check_alerts creates, passes, and flushes the accumulator."""

    def test_creates_accumulator_and_flushes(
        self, _real_checker_module, StockAlertChecker, triggered_alert, sample_df
    ):
        """check_alerts should create an accumulator and call flush_all after processing."""
        mock_accumulator_cls = MagicMock()
        mock_accumulator_instance = MagicMock()
        mock_accumulator_instance.get_stats.return_value = {
            "added": 1, "sent": 1, "failed": 0, "flushes": 1,
        }
        mock_accumulator_cls.return_value = mock_accumulator_instance

        with (
            patch.object(
                _real_checker_module,
                "DiscordMessageAccumulator",
                mock_accumulator_cls,
            ),
            patch.object(
                _real_checker_module,
                "get_rate_limiter",
                return_value=MagicMock(),
            ),
            patch.object(StockAlertChecker, "check_alert") as mock_check_alert,
        ):
            mock_check_alert.return_value = {
                "alert_id": "test-123",
                "ticker": "AAPL",
                "triggered": True,
                "error": None,
                "skipped": False,
            }

            checker = StockAlertChecker(async_discord=False)
            stats = checker.check_alerts([triggered_alert], max_workers=1)

        # Accumulator was created
        mock_accumulator_cls.assert_called_once()

        # check_alert was called with the accumulator
        mock_check_alert.assert_called_once()
        call_kwargs = mock_check_alert.call_args
        # The accumulator is passed as keyword arg or positional
        assert mock_accumulator_instance in call_kwargs[1].values() or \
               mock_accumulator_instance in call_kwargs[0]

        # flush_all was called
        mock_accumulator_instance.flush_all.assert_called_once()

        # get_stats was called (for logging)
        mock_accumulator_instance.get_stats.assert_called_once()

    def test_accumulator_flushed_even_on_empty_alerts(
        self, _real_checker_module, StockAlertChecker
    ):
        """flush_all should be called even when no alerts are processed."""
        mock_accumulator_cls = MagicMock()
        mock_accumulator_instance = MagicMock()
        mock_accumulator_instance.get_stats.return_value = {
            "added": 0, "sent": 0, "failed": 0, "flushes": 0,
        }
        mock_accumulator_cls.return_value = mock_accumulator_instance

        with (
            patch.object(
                _real_checker_module,
                "DiscordMessageAccumulator",
                mock_accumulator_cls,
            ),
            patch.object(
                _real_checker_module,
                "get_rate_limiter",
                return_value=None,
            ),
        ):
            checker = StockAlertChecker(async_discord=False)
            stats = checker.check_alerts([], max_workers=1)

        mock_accumulator_instance.flush_all.assert_called_once()
        assert stats["total"] == 0

    def test_accumulator_passed_in_parallel_mode(
        self, _real_checker_module, StockAlertChecker, triggered_alert
    ):
        """check_alerts with workers > 1 should still pass accumulator to check_alert."""
        mock_accumulator_cls = MagicMock()
        mock_accumulator_instance = MagicMock()
        mock_accumulator_instance.get_stats.return_value = {
            "added": 1, "sent": 1, "failed": 0, "flushes": 1,
        }
        mock_accumulator_cls.return_value = mock_accumulator_instance

        with (
            patch.object(
                _real_checker_module,
                "DiscordMessageAccumulator",
                mock_accumulator_cls,
            ),
            patch.object(
                _real_checker_module,
                "get_rate_limiter",
                return_value=MagicMock(),
            ),
            patch.object(StockAlertChecker, "check_alert") as mock_check_alert,
        ):
            mock_check_alert.return_value = {
                "alert_id": "test-123",
                "ticker": "AAPL",
                "triggered": False,
                "error": None,
                "skipped": False,
            }

            checker = StockAlertChecker(async_discord=False)
            stats = checker.check_alerts([triggered_alert], max_workers=3)

        # check_alert was called with the accumulator instance
        mock_check_alert.assert_called_once()
        args, kwargs = mock_check_alert.call_args
        # In the parallel path, accumulator is passed as a positional arg
        assert mock_accumulator_instance in args

        mock_accumulator_instance.flush_all.assert_called_once()

    def test_accumulator_flushed_after_error_in_check_alert(
        self, _real_checker_module, StockAlertChecker, triggered_alert
    ):
        """flush_all should be called even if check_alert raises an exception."""
        mock_accumulator_cls = MagicMock()
        mock_accumulator_instance = MagicMock()
        mock_accumulator_instance.get_stats.return_value = {
            "added": 0, "sent": 0, "failed": 0, "flushes": 0,
        }
        mock_accumulator_cls.return_value = mock_accumulator_instance

        with (
            patch.object(
                _real_checker_module,
                "DiscordMessageAccumulator",
                mock_accumulator_cls,
            ),
            patch.object(
                _real_checker_module,
                "get_rate_limiter",
                return_value=None,
            ),
            patch.object(
                StockAlertChecker,
                "check_alert",
                side_effect=RuntimeError("boom"),
            ),
        ):
            checker = StockAlertChecker(async_discord=False)
            stats = checker.check_alerts([triggered_alert], max_workers=1)

        # flush_all must still be called
        mock_accumulator_instance.flush_all.assert_called_once()
        assert stats["errors"] == 1

    def test_rate_limiter_passed_to_accumulator(
        self, _real_checker_module, StockAlertChecker
    ):
        """check_alerts should pass the rate limiter from get_rate_limiter()."""
        mock_rl = MagicMock(name="rate_limiter")
        mock_accumulator_cls = MagicMock()
        mock_accumulator_instance = MagicMock()
        mock_accumulator_instance.get_stats.return_value = {
            "added": 0, "sent": 0, "failed": 0, "flushes": 0,
        }
        mock_accumulator_cls.return_value = mock_accumulator_instance

        with (
            patch.object(
                _real_checker_module,
                "DiscordMessageAccumulator",
                mock_accumulator_cls,
            ),
            patch.object(
                _real_checker_module,
                "get_rate_limiter",
                return_value=mock_rl,
            ),
        ):
            checker = StockAlertChecker(async_discord=False)
            checker.check_alerts([], max_workers=1)

        mock_accumulator_cls.assert_called_once_with(rate_limiter=mock_rl)

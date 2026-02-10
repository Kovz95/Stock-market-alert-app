"""
Unit tests for src/services/scheduler_discord.py

Covers:
- _utc_str and _format_duration helpers
- DailySchedulerDiscord and WeeklySchedulerDiscord property values
- _load_config: happy path, missing key, exception, None document
- _load_scheduler_webhook: enabled/disabled/missing url/exception
- _post: discord disabled, missing webhook, placeholder url, fallback webhook,
         HTTP 200, HTTP 204, HTTP error, requests exception, env tag prepended
- notify_start / notify_skipped / notify_complete / notify_error /
  notify_scheduler_start / notify_scheduler_stop message content
- create_scheduler_discord factory: daily, weekly, unknown type
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from src.services.scheduler_discord import (
    BaseSchedulerDiscord,
    DailySchedulerDiscord,
    WeeklySchedulerDiscord,
    _format_duration,
    _utc_str,
    create_scheduler_discord,
)

# ---------------------------------------------------------------------------
# Fixed timestamp used across message-content tests
# ---------------------------------------------------------------------------
RUN_TIME = datetime(2025, 3, 15, 14, 30, 0, tzinfo=timezone.utc)
RUN_TIME_STR = "2025-03-15 14:30:00Z"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestUtcStr:
    """Tests for the _utc_str formatting helper."""

    def test_naive_datetime_treated_as_utc(self):
        """Naive datetimes should be assumed UTC."""
        result = _utc_str(datetime(2025, 1, 15, 12, 0, 0))
        assert result == "2025-01-15 12:00:00Z"

    def test_aware_utc_datetime(self):
        """UTC-aware datetimes should format with Z suffix."""
        dt = datetime(2025, 6, 1, 9, 30, 0, tzinfo=timezone.utc)
        assert _utc_str(dt) == "2025-06-01 09:30:00Z"

    def test_aware_non_utc_datetime_converted(self):
        """Non-UTC timezones should be converted before formatting."""
        import pytz
        et = pytz.timezone("America/New_York")
        # 7 AM ET in winter = 12 PM UTC
        dt = et.localize(datetime(2025, 1, 15, 7, 0, 0))
        assert _utc_str(dt) == "2025-01-15 12:00:00Z"

    def test_result_ends_with_z(self):
        """Result must always end with Z."""
        result = _utc_str(datetime(2025, 1, 1, 0, 0, 0))
        assert result.endswith("Z")


class TestFormatDuration:
    """Tests for the _format_duration formatting helper."""

    def test_seconds_only(self):
        assert _format_duration(45) == "45s"

    def test_exactly_one_minute(self):
        assert _format_duration(60) == "1m 0s"

    def test_minutes_and_seconds(self):
        assert _format_duration(125) == "2m 5s"

    def test_hours_minutes_seconds(self):
        assert _format_duration(3725) == "1h 2m 5s"

    def test_zero_seconds(self):
        assert _format_duration(0) == "0s"

    def test_negative_rounds_to_zero(self):
        assert _format_duration(-10) == "0s"

    def test_fractional_seconds_rounded(self):
        # Python 3 uses banker's rounding: round(120.5) == 120 (rounds to even)
        assert _format_duration(120.5) == "2m 0s"
        # 121.5 rounds to 122 = 2m 2s
        assert _format_duration(121.5) == "2m 2s"


# ---------------------------------------------------------------------------
# Fixtures: notifiers with config pre-loaded
# ---------------------------------------------------------------------------

def _make_daily(webhook_url: str | None = "https://discord.com/daily/TOKEN"):
    """Return a DailySchedulerDiscord with the given webhook in config."""
    config = {}
    if webhook_url:
        config["logging_channels"] = {
            "Daily_Scheduler_Status": {"webhook_url": webhook_url}
        }
    with patch("src.services.scheduler_discord.load_document", return_value=config):
        notifier = DailySchedulerDiscord()
    notifier.fallback_webhook = None
    return notifier


def _make_weekly(webhook_url: str | None = "https://discord.com/weekly/TOKEN"):
    """Return a WeeklySchedulerDiscord with the given webhook in config."""
    config = {}
    if webhook_url:
        config["logging_channels"] = {
            "Weekly_Scheduler_Status": {"webhook_url": webhook_url}
        }
    with patch("src.services.scheduler_discord.load_document", return_value=config):
        notifier = WeeklySchedulerDiscord()
    notifier.fallback_webhook = None
    return notifier


@pytest.fixture
def daily():
    return _make_daily()


@pytest.fixture
def weekly():
    return _make_weekly()


@pytest.fixture
def daily_mocked_post(daily):
    """Daily notifier with _post replaced by a Mock."""
    daily._post = Mock(return_value=True)
    return daily


@pytest.fixture
def weekly_mocked_post(weekly):
    """Weekly notifier with _post replaced by a Mock."""
    weekly._post = Mock(return_value=True)
    return weekly


# ---------------------------------------------------------------------------
# Subclass properties
# ---------------------------------------------------------------------------

class TestDailyProperties:
    """DailySchedulerDiscord must expose the correct property values."""

    def test_job_label(self, daily):
        assert daily.job_label == "Daily"

    def test_timeframe_key(self, daily):
        assert daily.timeframe_key == "1d"

    def test_config_key(self, daily):
        assert daily.config_key == "Daily_Scheduler_Status"

    def test_is_base_scheduler_discord(self, daily):
        assert isinstance(daily, BaseSchedulerDiscord)


class TestWeeklyProperties:
    """WeeklySchedulerDiscord must expose the correct property values."""

    def test_job_label(self, weekly):
        assert weekly.job_label == "Weekly"

    def test_timeframe_key(self, weekly):
        assert weekly.timeframe_key == "1wk"

    def test_config_key(self, weekly):
        assert weekly.config_key == "Weekly_Scheduler_Status"

    def test_is_base_scheduler_discord(self, weekly):
        assert isinstance(weekly, BaseSchedulerDiscord)


# ---------------------------------------------------------------------------
# _load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    """Tests for BaseSchedulerDiscord._load_config."""

    def test_loads_daily_webhook_from_document_store(self):
        doc = {
            "logging_channels": {
                "Daily_Scheduler_Status": {"webhook_url": "https://discord.com/daily"}
            }
        }
        with patch("src.services.scheduler_discord.load_document", return_value=doc):
            n = DailySchedulerDiscord()
        assert n.config == {"webhook_url": "https://discord.com/daily"}

    def test_loads_weekly_webhook_from_document_store(self):
        doc = {
            "logging_channels": {
                "Weekly_Scheduler_Status": {"webhook_url": "https://discord.com/weekly"}
            }
        }
        with patch("src.services.scheduler_discord.load_document", return_value=doc):
            n = WeeklySchedulerDiscord()
        assert n.config == {"webhook_url": "https://discord.com/weekly"}

    def test_returns_empty_dict_when_config_key_absent(self):
        doc = {"logging_channels": {}}
        with patch("src.services.scheduler_discord.load_document", return_value=doc):
            n = DailySchedulerDiscord()
        assert n.config == {}

    def test_returns_empty_dict_when_logging_channels_absent(self):
        with patch("src.services.scheduler_discord.load_document", return_value={}):
            n = DailySchedulerDiscord()
        assert n.config == {}

    def test_returns_empty_dict_when_document_is_none(self):
        with patch("src.services.scheduler_discord.load_document", return_value=None):
            n = DailySchedulerDiscord()
        assert n.config == {}

    def test_returns_empty_dict_on_exception(self):
        with patch(
            "src.services.scheduler_discord.load_document",
            side_effect=Exception("DB error"),
        ):
            n = DailySchedulerDiscord()
        assert n.config == {}


# ---------------------------------------------------------------------------
# _load_scheduler_webhook (fallback)
# ---------------------------------------------------------------------------

class TestLoadSchedulerWebhook:
    """Tests for BaseSchedulerDiscord._load_scheduler_webhook."""

    def _make(self, discord_doc, scheduler_doc):
        """Helper: build notifier with controlled load_document responses."""
        with patch(
            "src.services.scheduler_discord.load_document",
            side_effect=[discord_doc, scheduler_doc],
        ):
            return DailySchedulerDiscord()

    def test_returns_url_when_enabled(self):
        n = self._make(
            {},
            {"scheduler_webhook": {"enabled": True, "url": "https://discord.com/fallback"}},
        )
        assert n.fallback_webhook == "https://discord.com/fallback"

    def test_returns_none_when_webhook_disabled(self):
        n = self._make(
            {},
            {"scheduler_webhook": {"enabled": False, "url": "https://discord.com/fallback"}},
        )
        assert n.fallback_webhook is None

    def test_returns_none_when_url_is_empty(self):
        n = self._make(
            {},
            {"scheduler_webhook": {"enabled": True, "url": ""}},
        )
        assert n.fallback_webhook is None

    def test_returns_none_when_scheduler_webhook_key_missing(self):
        n = self._make({}, {})
        assert n.fallback_webhook is None

    def test_returns_none_on_exception(self):
        with patch(
            "src.services.scheduler_discord.load_document",
            side_effect=[{}, Exception("error")],
        ):
            n = DailySchedulerDiscord()
        assert n.fallback_webhook is None


# ---------------------------------------------------------------------------
# _post
# ---------------------------------------------------------------------------

class TestPost:
    """Tests for BaseSchedulerDiscord._post."""

    def test_returns_false_when_discord_disabled(self, daily):
        with patch("src.utils.discord_env.is_discord_send_enabled", return_value=False):
            assert daily._post("msg") is False

    def test_returns_false_when_no_webhook_configured(self):
        n = _make_daily(webhook_url=None)
        n.fallback_webhook = None
        with patch("src.utils.discord_env.is_discord_send_enabled", return_value=True):
            assert n._post("msg") is False

    def test_returns_false_for_placeholder_webhook(self):
        n = _make_daily(webhook_url="YOUR_WEBHOOK_URL_HERE")
        with patch("src.utils.discord_env.is_discord_send_enabled", return_value=True):
            assert n._post("msg") is False

    def test_uses_fallback_when_config_webhook_empty(self):
        n = _make_daily(webhook_url=None)
        n.fallback_webhook = "https://discord.com/fallback"
        mock_resp = Mock(status_code=204)
        with patch("src.utils.discord_env.is_discord_send_enabled", return_value=True), \
             patch("src.utils.discord_env.get_discord_environment_tag", return_value=""), \
             patch("src.services.scheduler_discord.requests.post", return_value=mock_resp) as p:
            result = n._post("hello")
        assert result is True
        assert p.call_args[0][0] == "https://discord.com/fallback"

    def test_returns_true_on_http_200(self, daily):
        mock_resp = Mock(status_code=200)
        with patch("src.utils.discord_env.is_discord_send_enabled", return_value=True), \
             patch("src.utils.discord_env.get_discord_environment_tag", return_value=""), \
             patch("src.services.scheduler_discord.requests.post", return_value=mock_resp):
            assert daily._post("hello") is True

    def test_returns_true_on_http_204(self, daily):
        mock_resp = Mock(status_code=204)
        with patch("src.utils.discord_env.is_discord_send_enabled", return_value=True), \
             patch("src.utils.discord_env.get_discord_environment_tag", return_value=""), \
             patch("src.services.scheduler_discord.requests.post", return_value=mock_resp):
            assert daily._post("hello") is True

    def test_returns_false_on_http_error_status(self, daily):
        mock_resp = Mock(status_code=400, text="Bad Request")
        with patch("src.utils.discord_env.is_discord_send_enabled", return_value=True), \
             patch("src.utils.discord_env.get_discord_environment_tag", return_value=""), \
             patch("src.services.scheduler_discord.requests.post", return_value=mock_resp):
            assert daily._post("hello") is False

    def test_returns_false_on_requests_exception(self, daily):
        with patch("src.utils.discord_env.is_discord_send_enabled", return_value=True), \
             patch("src.utils.discord_env.get_discord_environment_tag", return_value=""), \
             patch(
                 "src.services.scheduler_discord.requests.post",
                 side_effect=Exception("connection error"),
             ):
            assert daily._post("hello") is False

    def test_environment_tag_prepended_to_message(self, daily):
        mock_resp = Mock(status_code=204)
        with patch("src.utils.discord_env.is_discord_send_enabled", return_value=True), \
             patch("src.utils.discord_env.get_discord_environment_tag", return_value="[PROD] "), \
             patch("src.services.scheduler_discord.requests.post", return_value=mock_resp) as p:
            daily._post("test content")
        assert p.call_args[1]["json"]["content"] == "[PROD] test content"

    def test_posts_to_configured_webhook_url(self, daily):
        mock_resp = Mock(status_code=204)
        with patch("src.utils.discord_env.is_discord_send_enabled", return_value=True), \
             patch("src.utils.discord_env.get_discord_environment_tag", return_value=""), \
             patch("src.services.scheduler_discord.requests.post", return_value=mock_resp) as p:
            daily._post("test")
        assert p.call_args[0][0] == "https://discord.com/daily/TOKEN"

    def test_request_uses_10s_timeout(self, daily):
        mock_resp = Mock(status_code=204)
        with patch("src.utils.discord_env.is_discord_send_enabled", return_value=True), \
             patch("src.utils.discord_env.get_discord_environment_tag", return_value=""), \
             patch("src.services.scheduler_discord.requests.post", return_value=mock_resp) as p:
            daily._post("test")
        assert p.call_args[1]["timeout"] == 10


# ---------------------------------------------------------------------------
# notify_start
# ---------------------------------------------------------------------------

class TestNotifyStart:
    """Tests for BaseSchedulerDiscord.notify_start message format."""

    def test_daily_message_header(self, daily_mocked_post):
        daily_mocked_post.notify_start(RUN_TIME, "NYSE")
        msg = daily_mocked_post._post.call_args[0][0]
        assert "✅ **Daily Alert Check Started**" in msg

    def test_contains_run_time(self, daily_mocked_post):
        daily_mocked_post.notify_start(RUN_TIME, "NYSE")
        msg = daily_mocked_post._post.call_args[0][0]
        assert RUN_TIME_STR in msg

    def test_contains_daily_timeframe(self, daily_mocked_post):
        daily_mocked_post.notify_start(RUN_TIME, "NYSE")
        assert "1d" in daily_mocked_post._post.call_args[0][0]

    def test_contains_exchange(self, daily_mocked_post):
        daily_mocked_post.notify_start(RUN_TIME, "NYSE")
        assert "NYSE" in daily_mocked_post._post.call_args[0][0]

    def test_weekly_message_header(self, weekly_mocked_post):
        weekly_mocked_post.notify_start(RUN_TIME, "LSE")
        msg = weekly_mocked_post._post.call_args[0][0]
        assert "✅ **Weekly Alert Check Started**" in msg

    def test_weekly_contains_1wk_timeframe(self, weekly_mocked_post):
        weekly_mocked_post.notify_start(RUN_TIME, "LSE")
        assert "1wk" in weekly_mocked_post._post.call_args[0][0]


# ---------------------------------------------------------------------------
# notify_skipped
# ---------------------------------------------------------------------------

class TestNotifySkipped:
    """Tests for BaseSchedulerDiscord.notify_skipped message format."""

    def test_daily_skipped_header(self, daily_mocked_post):
        daily_mocked_post.notify_skipped(RUN_TIME, "No market data")
        msg = daily_mocked_post._post.call_args[0][0]
        assert "⚪ **Daily Alert Check Skipped**" in msg

    def test_contains_run_time(self, daily_mocked_post):
        daily_mocked_post.notify_skipped(RUN_TIME, "Holiday")
        assert RUN_TIME_STR in daily_mocked_post._post.call_args[0][0]

    def test_contains_reason(self, daily_mocked_post):
        daily_mocked_post.notify_skipped(RUN_TIME, "Market closed today")
        assert "Market closed today" in daily_mocked_post._post.call_args[0][0]

    def test_weekly_skipped_header(self, weekly_mocked_post):
        weekly_mocked_post.notify_skipped(RUN_TIME, "reason")
        assert "⚪ **Weekly Alert Check Skipped**" in weekly_mocked_post._post.call_args[0][0]


# ---------------------------------------------------------------------------
# notify_complete
# ---------------------------------------------------------------------------

class TestNotifyComplete:
    """Tests for BaseSchedulerDiscord.notify_complete message format."""

    PRICE = {"updated": 50, "failed": 2, "skipped": 3}
    ALERTS = {
        "total": 100,
        "triggered": 5,
        "not_triggered": 90,
        "no_data": 3,
        "stale_data": 1,
        "errors": 1,
    }

    def test_daily_complete_header(self, daily_mocked_post):
        daily_mocked_post.notify_complete(RUN_TIME, 60, self.PRICE, self.ALERTS, "NYSE")
        assert "✅ **Daily Alert Check Complete**" in daily_mocked_post._post.call_args[0][0]

    def test_weekly_complete_header(self, weekly_mocked_post):
        weekly_mocked_post.notify_complete(RUN_TIME, 60, self.PRICE, self.ALERTS, "TSE")
        assert "✅ **Weekly Alert Check Complete**" in weekly_mocked_post._post.call_args[0][0]

    def test_contains_run_time(self, daily_mocked_post):
        daily_mocked_post.notify_complete(RUN_TIME, 60, self.PRICE, self.ALERTS, "NYSE")
        assert RUN_TIME_STR in daily_mocked_post._post.call_args[0][0]

    def test_contains_duration(self, daily_mocked_post):
        # 125 seconds = 2m 5s
        daily_mocked_post.notify_complete(RUN_TIME, 125, self.PRICE, self.ALERTS, "NYSE")
        assert "2m 5s" in daily_mocked_post._post.call_args[0][0]

    def test_daily_timeframe_in_message(self, daily_mocked_post):
        daily_mocked_post.notify_complete(RUN_TIME, 60, self.PRICE, self.ALERTS, "NYSE")
        assert "1d" in daily_mocked_post._post.call_args[0][0]

    def test_weekly_timeframe_in_message(self, weekly_mocked_post):
        weekly_mocked_post.notify_complete(RUN_TIME, 60, self.PRICE, self.ALERTS, "TSE")
        assert "1wk" in weekly_mocked_post._post.call_args[0][0]

    def test_contains_exchange(self, daily_mocked_post):
        daily_mocked_post.notify_complete(RUN_TIME, 60, self.PRICE, self.ALERTS, "NYSE")
        assert "NYSE" in daily_mocked_post._post.call_args[0][0]

    def test_price_stats_in_message(self, daily_mocked_post):
        daily_mocked_post.notify_complete(RUN_TIME, 60, self.PRICE, self.ALERTS, "NYSE")
        msg = daily_mocked_post._post.call_args[0][0]
        assert "50" in msg   # updated
        assert "failed 2" in msg
        assert "skipped 3" in msg

    def test_alert_stats_in_message(self, daily_mocked_post):
        daily_mocked_post.notify_complete(RUN_TIME, 60, self.PRICE, self.ALERTS, "NYSE")
        msg = daily_mocked_post._post.call_args[0][0]
        assert "total 100" in msg
        assert "triggered 5" in msg
        assert "not triggered 90" in msg
        assert "no data 3" in msg
        assert "stale 1" in msg
        assert "errors 1" in msg

    def test_first_failure_shown_when_failures_exist(self, daily_mocked_post):
        price = {"updated": 48, "failed": 2, "skipped": 0}
        daily_mocked_post.notify_complete(
            RUN_TIME, 90, price, {}, "NYSE",
            first_failure_reason="Connection refused",
        )
        msg = daily_mocked_post._post.call_args[0][0]
        assert "First failure" in msg
        assert "Connection refused" in msg

    def test_first_failure_omitted_when_no_failures(self, daily_mocked_post):
        price = {"updated": 50, "failed": 0, "skipped": 0}
        daily_mocked_post.notify_complete(
            RUN_TIME, 90, price, {}, "NYSE",
            first_failure_reason="should not appear",
        )
        msg = daily_mocked_post._post.call_args[0][0]
        assert "First failure" not in msg
        assert "should not appear" not in msg

    def test_first_failure_reason_truncated_at_200_chars(self, daily_mocked_post):
        price = {"failed": 1}
        long_reason = "E" * 250
        daily_mocked_post.notify_complete(
            RUN_TIME, 30, price, {}, "NYSE",
            first_failure_reason=long_reason,
        )
        msg = daily_mocked_post._post.call_args[0][0]
        assert "..." in msg
        # The full 250-char string must not appear verbatim
        assert long_reason not in msg

    def test_handles_none_price_stats_gracefully(self, daily_mocked_post):
        """Passing None for price_stats must not raise."""
        daily_mocked_post.notify_complete(RUN_TIME, 30, None, None, "NYSE")
        assert daily_mocked_post._post.called

    def test_handles_empty_stats_dicts(self, daily_mocked_post):
        daily_mocked_post.notify_complete(RUN_TIME, 30, {}, {}, "NYSE")
        msg = daily_mocked_post._post.call_args[0][0]
        # All counters should default to 0
        assert "0" in msg

    def test_post_called_once(self, daily_mocked_post):
        daily_mocked_post.notify_complete(RUN_TIME, 60, self.PRICE, self.ALERTS, "NYSE")
        daily_mocked_post._post.assert_called_once()


# ---------------------------------------------------------------------------
# notify_error
# ---------------------------------------------------------------------------

class TestNotifyError:
    """Tests for BaseSchedulerDiscord.notify_error message format."""

    def test_daily_error_header(self, daily_mocked_post):
        daily_mocked_post.notify_error(RUN_TIME, "Something exploded")
        assert "❌ **Daily Scheduler Error**" in daily_mocked_post._post.call_args[0][0]

    def test_weekly_error_header(self, weekly_mocked_post):
        weekly_mocked_post.notify_error(RUN_TIME, "Timeout")
        assert "❌ **Weekly Scheduler Error**" in weekly_mocked_post._post.call_args[0][0]

    def test_contains_run_time(self, daily_mocked_post):
        daily_mocked_post.notify_error(RUN_TIME, "err")
        assert RUN_TIME_STR in daily_mocked_post._post.call_args[0][0]

    def test_contains_error_message(self, daily_mocked_post):
        daily_mocked_post.notify_error(RUN_TIME, "DB connection failed")
        assert "DB connection failed" in daily_mocked_post._post.call_args[0][0]


# ---------------------------------------------------------------------------
# notify_scheduler_start / notify_scheduler_stop
# ---------------------------------------------------------------------------

class TestNotifySchedulerStart:
    """Tests for BaseSchedulerDiscord.notify_scheduler_start."""

    def test_daily_online_header(self, daily_mocked_post):
        daily_mocked_post.notify_scheduler_start("NYSE at 16:00 ET")
        msg = daily_mocked_post._post.call_args[0][0]
        assert "✅ **Daily Scheduler Online**" in msg

    def test_contains_schedule_info(self, daily_mocked_post):
        daily_mocked_post.notify_scheduler_start("NYSE at 16:00 ET")
        assert "NYSE at 16:00 ET" in daily_mocked_post._post.call_args[0][0]

    def test_weekly_online_header(self, weekly_mocked_post):
        weekly_mocked_post.notify_scheduler_start("TSE at 15:30 ET")
        assert "✅ **Weekly Scheduler Online**" in weekly_mocked_post._post.call_args[0][0]


class TestNotifySchedulerStop:
    """Tests for BaseSchedulerDiscord.notify_scheduler_stop."""

    def test_daily_stopped_header(self, daily_mocked_post):
        daily_mocked_post.notify_scheduler_stop()
        assert "⏹️ **Daily Scheduler Stopped**" in daily_mocked_post._post.call_args[0][0]

    def test_weekly_stopped_header(self, weekly_mocked_post):
        weekly_mocked_post.notify_scheduler_stop()
        assert "⏹️ **Weekly Scheduler Stopped**" in weekly_mocked_post._post.call_args[0][0]

    def test_contains_timestamp(self, daily_mocked_post):
        daily_mocked_post.notify_scheduler_stop()
        msg = daily_mocked_post._post.call_args[0][0]
        # Should contain a UTC timestamp string (ends with Z)
        assert "Z" in msg


# ---------------------------------------------------------------------------
# create_scheduler_discord factory
# ---------------------------------------------------------------------------

class TestCreateSchedulerDiscord:
    """Tests for the create_scheduler_discord factory function."""

    def test_returns_daily_instance_for_daily(self):
        with patch("src.services.scheduler_discord.load_document", return_value={}):
            n = create_scheduler_discord("daily")
        assert isinstance(n, DailySchedulerDiscord)

    def test_returns_weekly_instance_for_weekly(self):
        with patch("src.services.scheduler_discord.load_document", return_value={}):
            n = create_scheduler_discord("weekly")
        assert isinstance(n, WeeklySchedulerDiscord)

    def test_returned_instance_is_base_scheduler_discord(self):
        with patch("src.services.scheduler_discord.load_document", return_value={}):
            n = create_scheduler_discord("daily")
        assert isinstance(n, BaseSchedulerDiscord)

    def test_raises_value_error_for_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown job_type"):
            create_scheduler_discord("hourly")

    def test_raises_value_error_for_empty_string(self):
        with pytest.raises(ValueError):
            create_scheduler_discord("")

    def test_raises_value_error_for_uppercase_daily(self):
        """Factory is case-sensitive."""
        with pytest.raises(ValueError):
            create_scheduler_discord("Daily")

    def test_raises_value_error_for_uppercase_weekly(self):
        with pytest.raises(ValueError):
            create_scheduler_discord("Weekly")

    def test_error_message_includes_job_type(self):
        with pytest.raises(ValueError) as exc_info:
            create_scheduler_discord("unknown_type")
        assert "unknown_type" in str(exc_info.value)

    def test_daily_has_correct_properties(self):
        with patch("src.services.scheduler_discord.load_document", return_value={}):
            n = create_scheduler_discord("daily")
        assert n.job_label == "Daily"
        assert n.timeframe_key == "1d"

    def test_weekly_has_correct_properties(self):
        with patch("src.services.scheduler_discord.load_document", return_value={}):
            n = create_scheduler_discord("weekly")
        assert n.job_label == "Weekly"
        assert n.timeframe_key == "1wk"

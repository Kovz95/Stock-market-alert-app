"""
Unit tests for settings configuration module.

Tests the Pydantic settings class for proper environment variable loading
and validation.
"""

import pytest
from pydantic import ValidationError

from src.stock_alert.config.settings import Settings, get_fmp_api_key, get_settings, get_webhook_url


class TestSettings:
    """Test suite for Settings configuration class"""

    def test_settings_loads_from_environment(self, monkeypatch):
        """Test that settings properly load from environment variables"""
        # Set required environment variables
        monkeypatch.setenv("FMP_API_KEY", "test_api_key_123")
        monkeypatch.setenv("WEBHOOK_URL_LOGGING", "https://discord.com/api/webhooks/123/test1")
        monkeypatch.setenv("WEBHOOK_URL_LOGGING_2", "https://discord.com/api/webhooks/123/test2")

        # Clear the cache to force reload
        get_settings.cache_clear()

        settings = Settings()

        assert settings.fmp_api_key == "test_api_key_123"
        assert settings.webhook_url_logging == "https://discord.com/api/webhooks/123/test1"
        assert settings.webhook_url_logging_2 == "https://discord.com/api/webhooks/123/test2"

    def test_settings_requires_fmp_api_key(self, monkeypatch):
        """Test that FMP_API_KEY is required"""
        # Clear FMP_API_KEY if set
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        monkeypatch.setenv("WEBHOOK_URL_LOGGING", "https://discord.com/api/webhooks/123/test1")
        monkeypatch.setenv("WEBHOOK_URL_LOGGING_2", "https://discord.com/api/webhooks/123/test2")

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        assert "fmp_api_key" in str(exc_info.value).lower()

    def test_settings_requires_webhook_logging_urls(self, monkeypatch):
        """Test that webhook logging URLs are required"""
        monkeypatch.setenv("FMP_API_KEY", "test_api_key_123")
        monkeypatch.delenv("WEBHOOK_URL_LOGGING", raising=False)
        monkeypatch.delenv("WEBHOOK_URL_LOGGING_2", raising=False)

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        error_msg = str(exc_info.value).lower()
        assert "webhook_url_logging" in error_msg

    def test_webhook_url_validation(self, monkeypatch):
        """Test that webhook URLs must be valid Discord webhook URLs"""
        monkeypatch.setenv("FMP_API_KEY", "test_api_key_123")
        monkeypatch.setenv("WEBHOOK_URL_LOGGING", "https://example.com/webhook")  # Invalid
        monkeypatch.setenv("WEBHOOK_URL_LOGGING_2", "https://discord.com/api/webhooks/123/test2")

        with pytest.raises(ValidationError) as exc_info:
            Settings()

        assert "Invalid Discord webhook URL" in str(exc_info.value)

    def test_environment_detection(self, monkeypatch):
        """Test environment detection properties"""
        monkeypatch.setenv("FMP_API_KEY", "test_api_key_123")
        monkeypatch.setenv("WEBHOOK_URL_LOGGING", "https://discord.com/api/webhooks/123/test1")
        monkeypatch.setenv("WEBHOOK_URL_LOGGING_2", "https://discord.com/api/webhooks/123/test2")
        monkeypatch.setenv("ENVIRONMENT", "production")

        settings = Settings()

        assert settings.is_production is True
        assert settings.is_development is False

        monkeypatch.setenv("ENVIRONMENT", "development")
        settings = Settings()

        assert settings.is_production is False
        assert settings.is_development is True

    def test_optional_fields_have_defaults(self, monkeypatch):
        """Test that optional fields have sensible defaults"""
        monkeypatch.setenv("FMP_API_KEY", "test_api_key_123")
        monkeypatch.setenv("WEBHOOK_URL_LOGGING", "https://discord.com/api/webhooks/123/test1")
        monkeypatch.setenv("WEBHOOK_URL_LOGGING_2", "https://discord.com/api/webhooks/123/test2")
        # Explicitly unset ENVIRONMENT to test default
        monkeypatch.delenv("ENVIRONMENT", raising=False)

        settings = Settings()

        assert settings.environment == "development"
        assert settings.debug is False
        assert settings.ib_host == "127.0.0.1"
        assert settings.ib_port == 7497
        assert settings.ib_client_id == 1


class TestSettingsHelpers:
    """Test suite for settings helper functions"""

    def test_get_fmp_api_key(self, monkeypatch):
        """Test get_fmp_api_key helper function"""
        monkeypatch.setenv("FMP_API_KEY", "test_key_abc")
        monkeypatch.setenv("WEBHOOK_URL_LOGGING", "https://discord.com/api/webhooks/123/test1")
        monkeypatch.setenv("WEBHOOK_URL_LOGGING_2", "https://discord.com/api/webhooks/123/test2")

        get_settings.cache_clear()
        api_key = get_fmp_api_key()

        assert api_key == "test_key_abc"

    def test_get_webhook_url_logging(self, monkeypatch):
        """Test get_webhook_url for logging webhooks"""
        monkeypatch.setenv("FMP_API_KEY", "test_key_abc")
        monkeypatch.setenv("WEBHOOK_URL_LOGGING", "https://discord.com/api/webhooks/123/primary")
        monkeypatch.setenv(
            "WEBHOOK_URL_LOGGING_2", "https://discord.com/api/webhooks/123/secondary"
        )

        get_settings.cache_clear()

        # Primary logging webhook
        url = get_webhook_url(logging=True, secondary=False)
        assert url == "https://discord.com/api/webhooks/123/primary"

        # Secondary logging webhook
        url = get_webhook_url(logging=True, secondary=True)
        assert url == "https://discord.com/api/webhooks/123/secondary"

    def test_get_webhook_url_alerts(self, monkeypatch):
        """Test get_webhook_url for alert webhooks"""
        monkeypatch.setenv("FMP_API_KEY", "test_key_abc")
        monkeypatch.setenv("WEBHOOK_URL", "https://discord.com/api/webhooks/123/alert1")
        monkeypatch.setenv("WEBHOOK_URL_2", "https://discord.com/api/webhooks/123/alert2")
        monkeypatch.setenv("WEBHOOK_URL_LOGGING", "https://discord.com/api/webhooks/123/log1")
        monkeypatch.setenv("WEBHOOK_URL_LOGGING_2", "https://discord.com/api/webhooks/123/log2")

        get_settings.cache_clear()

        # Primary alert webhook
        url = get_webhook_url(logging=False, secondary=False)
        assert url == "https://discord.com/api/webhooks/123/alert1"

        # Secondary alert webhook
        url = get_webhook_url(logging=False, secondary=True)
        assert url == "https://discord.com/api/webhooks/123/alert2"

    def test_get_settings_caching(self, monkeypatch):
        """Test that get_settings returns cached instance"""
        monkeypatch.setenv("FMP_API_KEY", "test_key_abc")
        monkeypatch.setenv("WEBHOOK_URL_LOGGING", "https://discord.com/api/webhooks/123/test1")
        monkeypatch.setenv("WEBHOOK_URL_LOGGING_2", "https://discord.com/api/webhooks/123/test2")

        get_settings.cache_clear()

        settings1 = get_settings()
        settings2 = get_settings()

        # Should return the same cached instance
        assert settings1 is settings2

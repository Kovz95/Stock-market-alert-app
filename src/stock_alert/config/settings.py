"""
Application settings management using Pydantic Settings.

Loads configuration from environment variables with validation.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings can be overridden via environment variables.
    See .env.example for required variables.
    """

    # API Keys
    FMP_API_KEY: str | None = Field(None, description="Financial Modeling Prep API key")

    # Discord Webhooks - Alerts
    WEBHOOK_URL: str | None = Field(None, description="Primary Discord webhook for alerts")
    WEBHOOK_URL_2: str | None = Field(None, description="Secondary Discord webhook for alerts")

    # Discord Webhooks - Logging
    WEBHOOK_URL_LOGGING: str | None = Field(None, description="Primary Discord webhook for logging")
    WEBHOOK_URL_LOGGING_2: str | None = Field(
        None, description="Secondary Discord webhook for logging"
    )

    # Database
    DATABASE_URL: str | None = Field(None, description="PostgreSQL database connection URL")
    REDIS_URL: str | None = Field("redis://localhost:6379", description="Redis connection URL")

    # Environment
    ENVIRONMENT: str = Field(
        "development", description="Environment (development/production/testing)"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="allow",  # Allow extra environment variables
    )


# Create a singleton instance
_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """
    Get application settings singleton.

    Returns:
        Settings instance
    """
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance

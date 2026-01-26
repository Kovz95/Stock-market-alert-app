"""
Application settings using Pydantic for type-safe configuration management.
All secrets are loaded from environment variables with no hardcoded defaults.
"""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Environment variables are loaded from:
    1. Environment variables (highest priority)
    2. .env file in project root
    3. No defaults for secrets (will raise ValidationError if not set)
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # API Keys
    fmp_api_key: str = Field(
        ..., description="Financial Modeling Prep API Key", alias="FMP_API_KEY"
    )

    # Discord Webhooks
    webhook_url: str | None = Field(
        None, description="Primary Discord webhook for alerts", alias="WEBHOOK_URL"
    )

    webhook_url_2: str | None = Field(
        None, description="Secondary Discord webhook for alerts", alias="WEBHOOK_URL_2"
    )

    webhook_url_logging: str = Field(
        ..., description="Discord webhook for application logging", alias="WEBHOOK_URL_LOGGING"
    )

    webhook_url_logging_2: str = Field(
        ...,
        description="Secondary Discord webhook for application logging",
        alias="WEBHOOK_URL_LOGGING_2",
    )

    # Application Settings
    environment: str = Field(
        default="development",
        description="Application environment (development, staging, production)",
    )

    debug: bool = Field(default=False, description="Enable debug mode")

    # Database (optional overrides)
    database_url: str | None = Field(
        None, description="PostgreSQL database URL override", alias="DATABASE_URL"
    )

    redis_url: str | None = Field(None, description="Redis cache URL", alias="REDIS_URL")

    # Interactive Brokers (optional)
    ib_host: str = Field(
        default="127.0.0.1", description="Interactive Brokers Gateway/TWS host", alias="IB_HOST"
    )

    ib_port: int = Field(
        default=7497, description="Interactive Brokers Gateway/TWS port", alias="IB_PORT"
    )

    ib_client_id: int = Field(
        default=1, description="Interactive Brokers client ID", alias="IB_CLIENT_ID"
    )

    @field_validator("fmp_api_key", "webhook_url_logging", "webhook_url_logging_2")
    @classmethod
    def validate_required_fields(cls, v: str, info) -> str:
        """Ensure required fields are not empty"""
        if not v or v.strip() == "":
            raise ValueError(f"{info.field_name} cannot be empty")
        return v.strip()

    @field_validator("webhook_url", "webhook_url_2", "webhook_url_logging", "webhook_url_logging_2")
    @classmethod
    def validate_webhook_urls(cls, v: str | None) -> str | None:
        """Validate webhook URLs if provided"""
        if v and not v.startswith("https://discord.com/api/webhooks/"):
            raise ValueError("Invalid Discord webhook URL format")
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment.lower() == "development"


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    This function is cached to avoid re-reading environment variables
    on every call. Use this instead of instantiating Settings directly.

    Returns:
        Settings: Application settings instance

    Raises:
        ValidationError: If required environment variables are missing
    """
    return Settings()


# Convenience function for backward compatibility
def get_fmp_api_key() -> str:
    """Get FMP API key from settings"""
    return get_settings().fmp_api_key


def get_webhook_url(logging: bool = False, secondary: bool = False) -> str | None:
    """
    Get Discord webhook URL from settings

    Args:
        logging: If True, return logging webhook, else return alert webhook
        secondary: If True, return secondary webhook

    Returns:
        Webhook URL or None if not configured
    """
    settings = get_settings()

    if logging:
        return settings.webhook_url_logging_2 if secondary else settings.webhook_url_logging
    else:
        return settings.webhook_url_2 if secondary else settings.webhook_url

"""
Discord environment tag for messages.

Adds a short prefix to Discord messages so you can tell development/local
from production (e.g. [DEV] vs [PROD]). Uses the ENVIRONMENT env var,
consistent with db_config and deployment.
"""

import os


def get_discord_environment_tag() -> str:
    """
    Return a short tag to prepend to Discord messages for environment identification.

    Reads ENVIRONMENT (default "development"). Normalizes to "production" vs
    everything else (dev/local).

    Returns:
        A string like "[PROD] " or "[DEV] " (with trailing space).
    """
    raw = (os.getenv("ENVIRONMENT") or "development").strip().lower()
    if raw in ("production", "prod"):
        return "[PROD] "
    return "[DEV] "


def with_discord_environment(text: str) -> str:
    """
    Prepend the environment tag to non-empty text for Discord.

    Args:
        text: Message body (can be empty).

    Returns:
        Tag + text if text is non-empty, otherwise empty string.
    """
    if not (text and text.strip()):
        return ""
    return get_discord_environment_tag() + text.strip()

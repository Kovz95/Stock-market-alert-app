"""
Discord service module for Stock Market Alert App.

Provides Discord webhook integration for alerts and logging.
"""

from .client import DiscordClient

__all__ = ["DiscordClient"]

# DiscordEconomyRouter is available in .routing module
# but not auto-imported due to heavy dependencies

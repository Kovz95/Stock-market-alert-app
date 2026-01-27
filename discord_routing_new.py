"""
Backward compatibility shim for discord_routing.

This module has been moved to src/stock_alert/services/discord/routing.py
Importing from this location is deprecated but supported for backward compatibility.
"""

import warnings

# Issue deprecation warning
warnings.warn(
    "Importing from discord_routing.py is deprecated. "
    "Please use: from src.stock_alert.services.discord.routing import DiscordEconomyRouter",
    DeprecationWarning,
    stacklevel=2
)

# Import from new location
try:
    from src.stock_alert.services.discord.routing import *
except ImportError:
    # Fallback to old location if new one not available
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from discord_routing_old import *

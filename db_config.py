"""
Backward compatibility shim for db_config module.

This module maintains backward compatibility for existing code that imports from db_config.
All functionality has been moved to src.stock_alert.data_access.database.

New code should import from src.stock_alert.data_access.database instead.
"""

from __future__ import annotations

# Import and re-export from new location
from src.stock_alert.data_access.database import (
    DatabaseConfig,
    PostgresConnectionProxy,
    PostgresCursorProxy,
    db_config,
)

__all__ = ["DatabaseConfig", "PostgresConnectionProxy", "PostgresCursorProxy", "db_config"]

"""
Simple futures symbol checker that doesn't require IB connection
Thread-safe and can be used in batch processing
"""

import logging
from typing import Dict, Any

from data_access.document_store import load_document

logger = logging.getLogger(__name__)

# Cache for futures symbols
_futures_symbols_cache = None
_futures_metadata_cache: Dict[str, Any] | None = None

def load_futures_symbols():
    """Load futures symbols from JSON database"""
    global _futures_symbols_cache
    global _futures_metadata_cache

    if _futures_symbols_cache is not None:
        return _futures_symbols_cache

    try:
        futures_db = load_document(
            "futures_database",
            default={},
            fallback_path='futures_database.json',
        )
        if isinstance(futures_db, dict):
            _futures_metadata_cache = futures_db
            _futures_symbols_cache = set(futures_db.keys())
            logger.debug(f"Loaded {len(_futures_symbols_cache)} futures symbols")
            return _futures_symbols_cache
    except Exception as e:
        logger.warning(f"Failed to load futures database: {e}")

    # Return empty set if loading fails
    _futures_symbols_cache = set()
    return _futures_symbols_cache

def is_futures_symbol_simple(ticker):
    """
    Check if a ticker is a futures symbol using simple JSON lookup
    This is thread-safe and doesn't require IB connection
    """
    # Remove any exchange suffix
    base_symbol = ticker.split('.')[0] if '.' in ticker else ticker

    # Load futures symbols
    futures_symbols = load_futures_symbols()

    # Check if symbol is in futures database
    return base_symbol in futures_symbols

def get_futures_metadata(ticker):
    """Get metadata for a futures symbol if it exists"""
    base_symbol = ticker.split('.')[0] if '.' in ticker else ticker
    global _futures_metadata_cache

    if _futures_metadata_cache is None:
        load_futures_symbols()

    try:
        if isinstance(_futures_metadata_cache, dict):
            return _futures_metadata_cache.get(base_symbol)
    except Exception as e:
        logger.warning(f"Failed to get futures metadata: {e}")

    return None

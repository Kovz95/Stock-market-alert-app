"""
Data model for price data fetch results.

This module defines the result type returned by the SmartPriceFetcher service,
including price data, source information, and freshness status.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

import pandas as pd


@dataclass
class PriceDataResult:
    """
    Result of a price data fetch operation.

    Contains the price DataFrame along with metadata about where the data
    came from (cache/database/API) and how fresh it is.

    Attributes:
        df: Price data DataFrame with OHLCV columns, or None if fetch failed.
        source: Where the data was retrieved from.
        freshness: How fresh the data is relative to market expectations.
        last_update: Timestamp of the most recent data point.
        error: Error message if fetch failed, None otherwise.

    Example:
        >>> result = PriceDataResult(
        ...     df=price_df,
        ...     source='cache',
        ...     freshness='fresh',
        ...     last_update=datetime.now(),
        ...     error=None
        ... )
        >>> if result.df is not None:
        ...     print(f"Got {len(result.df)} bars from {result.source}")
    """

    df: Optional[pd.DataFrame]
    source: Literal['cache', 'database', 'api']
    freshness: Literal['fresh', 'recent', 'stale', 'error']
    last_update: Optional[datetime]
    error: Optional[str] = None

    @property
    def is_successful(self) -> bool:
        """Check if data fetch was successful."""
        return self.df is not None and not self.df.empty

    @property
    def is_fresh(self) -> bool:
        """Check if data is considered fresh."""
        return self.freshness == 'fresh'

    def __repr__(self) -> str:
        """String representation of result."""
        if self.error:
            return f"PriceDataResult(error='{self.error}')"

        rows = len(self.df) if self.df is not None else 0
        return (
            f"PriceDataResult(rows={rows}, source='{self.source}', "
            f"freshness='{self.freshness}', last_update={self.last_update})"
        )

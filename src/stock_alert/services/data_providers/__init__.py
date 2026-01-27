"""
Data providers module for Stock Market Alert App.

Provides access to various market data sources:
- FMP (Financial Modeling Prep) for stocks
- Interactive Brokers for futures
"""

from .base import AbstractDataProvider
from .fmp import FMPDataProvider, OptimizedFMPDataProvider

try:
    from .ib_futures import IBFuturesProvider

    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    IBFuturesProvider = None

__all__ = [
    "AbstractDataProvider",
    "FMPDataProvider",
    "OptimizedFMPDataProvider",
]

if IB_AVAILABLE:
    __all__.append("IBFuturesProvider")

"""
Factory for creating data provider instances.

Provides a unified way to instantiate the appropriate provider
based on asset type or provider name.
"""

import logging
from typing import Optional, Union

from .base import AbstractDataProvider
from .fmp import FMPDataProvider, OptimizedFMPDataProvider

try:
    from .ib_futures import IBFuturesProvider

    IB_AVAILABLE = True
except ImportError:
    IB_AVAILABLE = False
    IBFuturesProvider = None

logger = logging.getLogger(__name__)


class DataProviderFactory:
    """
    Factory for creating data provider instances.

    Handles selection of the appropriate provider based on:
    - Asset type (stock, futures, etc.)
    - Provider name (FMP, IB, etc.)
    - Configuration settings
    """

    @staticmethod
    def create_provider(
        provider_name: str | None = None, asset_type: str | None = None, **kwargs
    ) -> Union[AbstractDataProvider, "IBFuturesProvider"] | None:
        """
        Create a data provider instance.

        Args:
            provider_name: Name of provider ('fmp', 'ib', 'fmp_optimized')
            asset_type: Type of asset ('stock', 'futures')
            **kwargs: Additional arguments passed to provider constructor

        Returns:
            Provider instance or None if unavailable

        Examples:
            >>> # Create FMP provider
            >>> provider = DataProviderFactory.create_provider('fmp', api_key='...')
            >>> # Create IB provider for futures
            >>> provider = DataProviderFactory.create_provider('ib', port=7497)
            >>> # Create provider by asset type
            >>> provider = DataProviderFactory.create_provider(asset_type='futures')
        """
        # Normalize inputs
        if provider_name:
            provider_name = provider_name.lower()

        if asset_type:
            asset_type = asset_type.lower()

        # If provider explicitly specified
        if provider_name == "fmp":
            return FMPDataProvider(**kwargs)

        elif provider_name == "fmp_optimized":
            return OptimizedFMPDataProvider(**kwargs)

        elif provider_name == "ib":
            if not IB_AVAILABLE:
                logger.error("Interactive Brokers provider not available (ib_insync not installed)")
                return None
            return IBFuturesProvider(**kwargs)

        # If asset type specified, select appropriate provider
        if asset_type == "stock" or asset_type == "stocks":
            # Default to optimized FMP for stocks
            return OptimizedFMPDataProvider(**kwargs)

        elif asset_type == "futures" or asset_type == "future":
            if not IB_AVAILABLE:
                logger.error("Interactive Brokers provider not available for futures")
                return None
            return IBFuturesProvider(**kwargs)

        # Default to optimized FMP
        logger.warning("No provider/asset_type specified, defaulting to FMP")
        return OptimizedFMPDataProvider(**kwargs)

    @staticmethod
    def create_stock_provider(optimized: bool = True, **kwargs) -> AbstractDataProvider:
        """
        Convenience method to create a stock data provider.

        Args:
            optimized: Use optimized FMP provider with database integration
            **kwargs: Additional arguments passed to provider

        Returns:
            FMP data provider instance
        """
        if optimized:
            return OptimizedFMPDataProvider(**kwargs)
        return FMPDataProvider(**kwargs)

    @staticmethod
    def create_futures_provider(**kwargs) -> Optional["IBFuturesProvider"]:
        """
        Convenience method to create a futures data provider.

        Args:
            **kwargs: Additional arguments passed to provider

        Returns:
            IB futures provider instance or None if unavailable
        """
        if not IB_AVAILABLE:
            logger.error("Interactive Brokers provider not available (ib_insync not installed)")
            return None
        return IBFuturesProvider(**kwargs)

    @staticmethod
    def get_provider_for_symbol(
        symbol: str, **kwargs
    ) -> Union[AbstractDataProvider, "IBFuturesProvider"] | None:
        """
        Automatically select provider based on symbol format.

        Args:
            symbol: Ticker or futures symbol
            **kwargs: Additional arguments passed to provider

        Returns:
            Appropriate provider instance

        Logic:
            - Checks if symbol is a known futures contract
            - Falls back to stock provider for other symbols
        """
        # Check if it's a known futures symbol
        if IB_AVAILABLE:
            temp_ib = IBFuturesProvider()
            if temp_ib.is_futures_symbol(symbol):
                return IBFuturesProvider(**kwargs)

        # Default to stock provider
        return OptimizedFMPDataProvider(**kwargs)


# Convenience functions
def get_stock_provider(optimized: bool = True, **kwargs) -> AbstractDataProvider:
    """Get a stock data provider"""
    return DataProviderFactory.create_stock_provider(optimized=optimized, **kwargs)


def get_futures_provider(**kwargs) -> Optional["IBFuturesProvider"]:
    """Get a futures data provider"""
    return DataProviderFactory.create_futures_provider(**kwargs)


def get_provider_for_symbol(symbol: str, **kwargs):
    """Get appropriate provider for a symbol"""
    return DataProviderFactory.get_provider_for_symbol(symbol, **kwargs)

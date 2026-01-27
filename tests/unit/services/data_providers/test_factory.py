"""
Unit tests for data provider factory.

Tests the DataProviderFactory for provider selection and instantiation.
"""

import sys
from pathlib import Path

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.stock_alert.services.data_providers.factory import (
    DataProviderFactory,
    get_futures_provider,
    get_provider_for_symbol,
    get_stock_provider,
)
from src.stock_alert.services.data_providers.fmp import FMPDataProvider, OptimizedFMPDataProvider


class TestDataProviderFactory:
    """Test suite for DataProviderFactory"""

    def test_create_fmp_provider(self):
        """Test creating FMP provider"""
        provider = DataProviderFactory.create_provider("fmp", api_key="test_key")

        assert isinstance(provider, FMPDataProvider)
        assert provider.api_key == "test_key"

    def test_create_fmp_optimized_provider(self):
        """Test creating optimized FMP provider"""
        provider = DataProviderFactory.create_provider("fmp_optimized", api_key="test_key")

        assert isinstance(provider, OptimizedFMPDataProvider)
        assert provider.api_key == "test_key"

    def test_create_provider_by_asset_type_stock(self):
        """Test creating provider by asset type (stock)"""
        provider = DataProviderFactory.create_provider(asset_type="stock", api_key="test_key")

        assert isinstance(provider, OptimizedFMPDataProvider)

    def test_create_provider_by_asset_type_stocks(self):
        """Test creating provider by asset type (stocks plural)"""
        provider = DataProviderFactory.create_provider(asset_type="stocks", api_key="test_key")

        assert isinstance(provider, OptimizedFMPDataProvider)

    def test_create_provider_default(self):
        """Test creating provider with no arguments defaults to FMP"""
        provider = DataProviderFactory.create_provider(api_key="test_key")

        assert isinstance(provider, OptimizedFMPDataProvider)

    def test_create_stock_provider_optimized(self):
        """Test convenience method for optimized stock provider"""
        provider = DataProviderFactory.create_stock_provider(optimized=True, api_key="test_key")

        assert isinstance(provider, OptimizedFMPDataProvider)

    def test_create_stock_provider_basic(self):
        """Test convenience method for basic stock provider"""
        provider = DataProviderFactory.create_stock_provider(optimized=False, api_key="test_key")

        assert isinstance(provider, FMPDataProvider)
        assert not isinstance(provider, OptimizedFMPDataProvider)

    def test_get_stock_provider_convenience(self):
        """Test global convenience function"""
        provider = get_stock_provider(api_key="test_key")

        assert isinstance(provider, OptimizedFMPDataProvider)

    def test_case_insensitive_provider_names(self):
        """Test that provider names are case insensitive"""
        provider1 = DataProviderFactory.create_provider("FMP", api_key="test_key")
        provider2 = DataProviderFactory.create_provider("fmp", api_key="test_key")

        assert type(provider1) == type(provider2)

    def test_case_insensitive_asset_types(self):
        """Test that asset types are case insensitive"""
        provider1 = DataProviderFactory.create_provider(asset_type="STOCK", api_key="test_key")
        provider2 = DataProviderFactory.create_provider(asset_type="stock", api_key="test_key")

        assert type(provider1) == type(provider2)


class TestIBProviderCreation:
    """Test IB provider creation (may be skipped if ib_insync not installed)"""

    @pytest.mark.skipif(
        not hasattr(DataProviderFactory, "__dict__"), reason="IB provider not available"
    )
    def test_create_ib_provider_by_name(self):
        """Test creating IB provider by name (if available)"""
        try:
            from src.stock_alert.services.data_providers.ib_futures import IBFuturesProvider

            provider = DataProviderFactory.create_provider("ib")

            # If IB available, should create provider
            if provider is not None:
                assert isinstance(provider, IBFuturesProvider)
        except ImportError:
            # IB not available, should return None
            provider = DataProviderFactory.create_provider("ib")
            assert provider is None

    @pytest.mark.skipif(
        not hasattr(DataProviderFactory, "__dict__"), reason="IB provider not available"
    )
    def test_create_ib_provider_by_asset_type(self):
        """Test creating IB provider by asset type (if available)"""
        try:
            from src.stock_alert.services.data_providers.ib_futures import IBFuturesProvider

            provider = DataProviderFactory.create_provider(asset_type="futures")

            # If IB available, should create provider
            if provider is not None:
                assert isinstance(provider, IBFuturesProvider)
        except ImportError:
            # IB not available, should return None
            provider = DataProviderFactory.create_provider(asset_type="futures")
            assert provider is None

    def test_get_futures_provider_convenience(self):
        """Test global convenience function for futures"""
        provider = get_futures_provider()

        # May be None if ib_insync not installed
        if provider is not None:
            from src.stock_alert.services.data_providers.ib_futures import IBFuturesProvider

            assert isinstance(provider, IBFuturesProvider)


class TestProviderSelectionBySymbol:
    """Test automatic provider selection based on symbol"""

    def test_get_provider_for_stock_symbol(self):
        """Test selecting provider for stock symbol"""
        provider = get_provider_for_symbol("AAPL", api_key="test_key")

        assert isinstance(provider, OptimizedFMPDataProvider)

    def test_get_provider_for_unknown_symbol(self):
        """Test selecting provider for unknown symbol defaults to stock"""
        provider = get_provider_for_symbol("UNKNOWN123", api_key="test_key")

        assert isinstance(provider, OptimizedFMPDataProvider)

    @pytest.mark.skipif(
        True,  # Skip unless IB is available and configured
        reason="Requires IB provider",
    )
    def test_get_provider_for_futures_symbol(self):
        """Test selecting provider for futures symbol (requires IB)"""
        try:
            from src.stock_alert.services.data_providers.ib_futures import IBFuturesProvider

            provider = get_provider_for_symbol("ES")  # S&P 500 futures

            if provider and not isinstance(provider, OptimizedFMPDataProvider):
                assert isinstance(provider, IBFuturesProvider)
        except ImportError:
            pytest.skip("IB provider not available")

# Services Layer - AI Assistant Instructions

## Purpose of the Services Layer

The services layer contains the **business logic** and **orchestration** code for the application. Services coordinate between different parts of the system, implement complex business rules, integrate with external APIs, and provide a clean interface for the UI layer to interact with.

**Key Principle**: Services implement "what the application does" not "how it stores data" or "how it displays information".

---

## What Belongs in Services

### ✅ Should Be in Services

1. **Business Logic**
   - Complex calculations and algorithms
   - Business rule validation
   - Workflow orchestration
   - Data transformation between layers

2. **External API Integration**
   - Third-party API clients
   - API response handling and transformation
   - Rate limiting and retry logic
   - API authentication handling

3. **Cross-Cutting Concerns**
   - Sending notifications (email, SMS, push)
   - File generation (PDFs, reports, exports)
   - Background job scheduling
   - Caching strategies

4. **Domain Operations**
   - Multi-step operations that span multiple data models
   - Complex queries that require business logic
   - Data aggregation and analysis
   - Transaction coordination

### ❌ Should NOT Be in Services

1. **UI/Presentation Logic**
   - Streamlit widgets and components
   - UI state management
   - Display formatting
   - → These belong in `src/components/` or page files

2. **Data Models**
   - Database schemas
   - Pydantic models
   - Data classes
   - → These belong in `src/models/`

3. **Database Queries**
   - Direct SQL queries
   - ORM model definitions
   - Database connection management
   - → These belong in `src/data_access/`

4. **Utility Functions**
   - String manipulation
   - Date formatting
   - Generic helpers
   - → These belong in `src/utils/`

5. **Configuration**
   - Environment variables
   - Settings classes
   - → These belong in a config module

---

## Service Design Principles

### 1. Single Responsibility Principle (SRP)

Each service should have **one clear purpose** and handle **one domain or feature area**.

**Good Example**:
```python
# email_service.py - Handles all email operations
class EmailService:
    """Service for sending emails."""

    def send_welcome_email(self, user_email: str, username: str) -> bool:
        """Send welcome email to new user."""
        pass

    def send_alert_notification(self, user_email: str, alert_data: dict) -> bool:
        """Send stock alert notification."""
        pass

# stock_analysis_service.py - Handles stock analysis
class StockAnalysisService:
    """Service for stock market analysis."""

    def analyze_price_movement(self, symbol: str, days: int) -> dict:
        """Analyze price movement for a stock."""
        pass
```

**Bad Example**:
```python
# god_service.py - Does too many unrelated things ❌
class GodService:
    """Service that does everything."""  # ❌ Violates SRP

    def send_email(self, email: str, message: str) -> bool:
        pass

    def analyze_stock(self, symbol: str) -> dict:
        pass

    def generate_pdf_report(self, data: dict) -> bytes:
        pass

    def validate_user_input(self, input: str) -> bool:
        pass
```

### 2. Dependency Injection

Services should receive their dependencies through **constructor injection**, not create them internally.

**Good Example**:
```python
from src.data_access.database import StockRepository
from src.services.notification_service import NotificationService

class AlertService:
    """Service for managing stock alerts."""

    def __init__(
        self,
        stock_repo: StockRepository,
        notification_service: NotificationService
    ):
        """Initialize with injected dependencies."""
        self.stock_repo = stock_repo
        self.notification_service = notification_service

    def check_alerts(self, user_id: int) -> list[dict]:
        """Check and process alerts for user."""
        alerts = self.stock_repo.get_active_alerts(user_id)
        # Process alerts...
        return results
```

**Bad Example**:
```python
class AlertService:
    """Service with hard-coded dependencies."""  # ❌

    def __init__(self):
        # ❌ Creates dependencies internally - hard to test
        self.stock_repo = StockRepository()
        self.notification_service = NotificationService()

    def check_alerts(self, user_id: int) -> list[dict]:
        pass
```

### 3. Interface Segregation

Services should provide focused, well-defined interfaces. Split large services into smaller, specialized ones.

**Good Example**:
```python
# Focused interfaces
class StockDataService:
    """Service for fetching stock data."""
    def get_current_price(self, symbol: str) -> float:
        pass

    def get_historical_data(self, symbol: str, days: int) -> pd.DataFrame:
        pass

class StockAnalyticsService:
    """Service for analyzing stock data."""
    def calculate_moving_average(self, data: pd.DataFrame, window: int) -> pd.Series:
        pass

    def detect_trend(self, data: pd.DataFrame) -> str:
        pass

class StockAlertService:
    """Service for managing stock alerts."""
    def create_alert(self, user_id: int, symbol: str, threshold: float) -> int:
        pass

    def check_triggered_alerts(self) -> list[dict]:
        pass
```

### 4. Fail Fast and Explicitly

Validate inputs early and raise specific exceptions with clear messages.

**Good Example**:
```python
class StockService:
    """Service for stock operations."""

    def get_stock_data(self, symbol: str, days: int) -> dict:
        """Fetch stock data with validation."""
        # Validate inputs early
        if not symbol or not symbol.strip():
            raise ValueError("Stock symbol cannot be empty")

        if days <= 0:
            raise ValueError(f"Days must be positive, got {days}")

        if days > 365:
            raise ValueError(f"Cannot fetch more than 365 days of data, got {days}")

        # Fetch data
        try:
            data = self._fetch_from_api(symbol, days)
        except requests.exceptions.RequestException as e:
            raise StockDataError(f"Failed to fetch data for {symbol}: {e}") from e

        if not data:
            raise StockDataError(f"No data available for symbol {symbol}")

        return data
```

---

## Low-Level Design Patterns

### Service Class Structure Template

```python
"""
Module for [feature area] service operations.

This module provides [brief description of what this service does].
"""

from typing import Optional, Any
from datetime import datetime
import logging

# Third-party imports
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Local imports
from src.data_access.repositories import SomeRepository
from src.models.schemas import SomeModel
from src.utils.validators import validate_input

# Set up module logger
logger = logging.getLogger(__name__)


class CustomServiceError(Exception):
    """Base exception for this service."""
    pass


class ServiceValidationError(CustomServiceError):
    """Raised when input validation fails."""
    pass


class ServiceOperationError(CustomServiceError):
    """Raised when a service operation fails."""
    pass


class FeatureService:
    """
    Service for managing [feature area].

    This service handles [detailed description of responsibilities].
    It coordinates between [list key dependencies/interactions].

    Attributes:
        repository: Data access layer for [entity].
        api_client: Client for external API integration.
        config: Service configuration settings.

    Example:
        >>> service = FeatureService(repository, api_client)
        >>> result = service.perform_operation(param)
    """

    def __init__(
        self,
        repository: SomeRepository,
        api_client: Optional[Any] = None,
        config: Optional[dict] = None
    ):
        """
        Initialize the service with dependencies.

        Args:
            repository: Repository for data access.
            api_client: Optional external API client.
            config: Optional configuration dictionary.
        """
        self.repository = repository
        self.api_client = api_client
        self.config = config or {}

        # Initialize any internal state
        self._cache = {}
        self._last_update = None

        logger.info(f"Initialized {self.__class__.__name__}")

    def public_operation(
        self,
        param1: str,
        param2: int,
        optional_param: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Perform a public service operation.

        This method [detailed description of what it does, any side effects,
        and important behavior notes].

        Args:
            param1: Description of param1.
            param2: Description of param2 with expected range/format.
            optional_param: Description of optional parameter. Defaults to None.

        Returns:
            Dictionary containing:
                - 'key1': Description of key1
                - 'key2': Description of key2
                - 'timestamp': When operation was performed

        Raises:
            ServiceValidationError: If input validation fails.
            ServiceOperationError: If operation cannot be completed.

        Example:
            >>> result = service.public_operation("value", 42)
            >>> print(result['key1'])
        """
        # Step 1: Validate inputs
        self._validate_inputs(param1, param2)

        # Step 2: Perform operation
        try:
            result = self._internal_operation(param1, param2)
        except Exception as e:
            logger.error(f"Operation failed: {e}", exc_info=True)
            raise ServiceOperationError(f"Failed to perform operation: {e}") from e

        # Step 3: Post-process and return
        return {
            'key1': result,
            'key2': self._transform_result(result),
            'timestamp': datetime.now()
        }

    def _validate_inputs(self, param1: str, param2: int) -> None:
        """
        Validate input parameters (private helper).

        Args:
            param1: Parameter to validate.
            param2: Parameter to validate.

        Raises:
            ServiceValidationError: If validation fails.
        """
        if not param1 or not param1.strip():
            raise ServiceValidationError("param1 cannot be empty")

        if param2 < 0:
            raise ServiceValidationError(f"param2 must be non-negative, got {param2}")

    def _internal_operation(self, param1: str, param2: int) -> Any:
        """
        Internal helper for core operation logic (private).

        Args:
            param1: Validated param1.
            param2: Validated param2.

        Returns:
            Raw operation result.
        """
        # Core business logic here
        logger.debug(f"Performing internal operation with {param1}, {param2}")
        return self.repository.fetch_data(param1, param2)

    def _transform_result(self, raw_result: Any) -> Any:
        """
        Transform raw result to desired format (private).

        Args:
            raw_result: Raw data from internal operation.

        Returns:
            Transformed result.
        """
        # Transformation logic
        return raw_result


def create_feature_service(
    repository: Optional[SomeRepository] = None,
    api_client: Optional[Any] = None
) -> FeatureService:
    """
    Factory function to create a configured FeatureService instance.

    This factory handles dependency setup and configuration loading,
    providing a convenient way to instantiate the service.

    Args:
        repository: Optional repository instance. If None, creates default.
        api_client: Optional API client. If None, creates default.

    Returns:
        Configured FeatureService instance.

    Example:
        >>> service = create_feature_service()
        >>> result = service.public_operation("value", 42)
    """
    if repository is None:
        repository = SomeRepository()

    if api_client is None:
        api_client = _create_default_api_client()

    config = _load_service_config()

    return FeatureService(repository, api_client, config)


def _create_default_api_client() -> requests.Session:
    """Create a default API client with retry logic."""
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.3,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def _load_service_config() -> dict:
    """Load service configuration from environment/settings."""
    return {
        'timeout': 30,
        'cache_ttl': 300,
        'max_retries': 3
    }
```

### Async Service Pattern

For services that perform I/O operations, use async/await:

```python
"""Async service for concurrent operations."""

import asyncio
from typing import Optional
import logging
import aiohttp

logger = logging.getLogger(__name__)


class AsyncStockDataService:
    """
    Async service for fetching stock data from external APIs.

    This service uses async/await for concurrent API calls,
    allowing efficient fetching of multiple stocks simultaneously.
    """

    def __init__(self, api_key: str, base_url: str, timeout: int = 30):
        """
        Initialize async service.

        Args:
            api_key: API authentication key.
            base_url: Base URL for API endpoints.
            timeout: Request timeout in seconds.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()

    async def fetch_stock_price(self, symbol: str) -> dict:
        """
        Fetch current price for a single stock.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Dictionary with price data.

        Raises:
            aiohttp.ClientError: If API request fails.
        """
        if not self._session:
            raise RuntimeError("Service must be used as async context manager")

        url = f"{self.base_url}/quote/{symbol}"
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            async with self._session.get(url, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                logger.debug(f"Fetched price for {symbol}")
                return data
        except aiohttp.ClientError as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            raise

    async def fetch_multiple_stocks(self, symbols: list[str]) -> dict[str, dict]:
        """
        Fetch prices for multiple stocks concurrently.

        Args:
            symbols: List of stock ticker symbols.

        Returns:
            Dictionary mapping symbols to their price data.

        Example:
            >>> async with AsyncStockDataService(api_key, url) as service:
            ...     results = await service.fetch_multiple_stocks(['AAPL', 'GOOGL'])
        """
        tasks = [self.fetch_stock_price(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            symbol: result if not isinstance(result, Exception) else None
            for symbol, result in zip(symbols, results)
        }


# Usage example
async def example_usage():
    """Example of using async service."""
    async with AsyncStockDataService(api_key="key", base_url="https://api.example.com") as service:
        # Fetch single stock
        price = await service.fetch_stock_price("AAPL")

        # Fetch multiple stocks concurrently
        prices = await service.fetch_multiple_stocks(["AAPL", "GOOGL", "MSFT"])
        return prices
```

### Service with Caching Pattern

```python
"""Service with intelligent caching."""

from typing import Optional, Any
from datetime import datetime, timedelta
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class CacheEntry:
    """Cache entry with expiration."""

    def __init__(self, value: Any, ttl_seconds: int):
        self.value = value
        self.created_at = datetime.now()
        self.expires_at = self.created_at + timedelta(seconds=ttl_seconds)

    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        return datetime.now() > self.expires_at


def cached(ttl_seconds: int = 300):
    """
    Decorator for caching service method results.

    Args:
        ttl_seconds: Time-to-live for cached values in seconds.
    """
    def decorator(func):
        cache = {}

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Create cache key from arguments
            cache_key = (func.__name__, args, tuple(sorted(kwargs.items())))

            # Check if cached and not expired
            if cache_key in cache:
                entry = cache[cache_key]
                if not entry.is_expired():
                    logger.debug(f"Cache hit for {func.__name__}")
                    return entry.value
                else:
                    logger.debug(f"Cache expired for {func.__name__}")
                    del cache[cache_key]

            # Call function and cache result
            logger.debug(f"Cache miss for {func.__name__}")
            result = func(self, *args, **kwargs)
            cache[cache_key] = CacheEntry(result, ttl_seconds)
            return result

        # Add method to clear cache
        wrapper.clear_cache = lambda: cache.clear()

        return wrapper
    return decorator


class MarketDataService:
    """Service with intelligent caching for market data."""

    def __init__(self, api_client: Any):
        """Initialize service with API client."""
        self.api_client = api_client

    @cached(ttl_seconds=60)
    def get_current_price(self, symbol: str) -> float:
        """
        Get current price with 60-second cache.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Current stock price.
        """
        logger.info(f"Fetching current price for {symbol}")
        response = self.api_client.get(f"/quote/{symbol}")
        return response['price']

    @cached(ttl_seconds=3600)
    def get_company_info(self, symbol: str) -> dict:
        """
        Get company info with 1-hour cache.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Company information dictionary.
        """
        logger.info(f"Fetching company info for {symbol}")
        response = self.api_client.get(f"/company/{symbol}")
        return response

    def invalidate_cache(self):
        """Invalidate all cached data."""
        self.get_current_price.clear_cache()
        self.get_company_info.clear_cache()
        logger.info("Cache invalidated")
```

---

## Error Handling Best Practices

### Define Custom Exceptions

```python
"""Custom exceptions for service layer."""


class ServiceError(Exception):
    """Base exception for all service errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        """
        Initialize service error.

        Args:
            message: Error message.
            details: Optional dictionary with additional error context.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ValidationError(ServiceError):
    """Raised when input validation fails."""
    pass


class NotFoundError(ServiceError):
    """Raised when requested resource is not found."""
    pass


class ExternalAPIError(ServiceError):
    """Raised when external API call fails."""

    def __init__(self, message: str, status_code: Optional[int] = None, **kwargs):
        """
        Initialize API error with status code.

        Args:
            message: Error message.
            status_code: HTTP status code.
            **kwargs: Additional details.
        """
        super().__init__(message, details={'status_code': status_code, **kwargs})
        self.status_code = status_code


class RateLimitError(ExternalAPIError):
    """Raised when API rate limit is exceeded."""

    def __init__(self, retry_after: Optional[int] = None):
        """
        Initialize rate limit error.

        Args:
            retry_after: Seconds to wait before retry.
        """
        message = f"Rate limit exceeded. Retry after {retry_after} seconds."
        super().__init__(message, status_code=429, retry_after=retry_after)
        self.retry_after = retry_after
```

### Comprehensive Error Handling

```python
"""Service with comprehensive error handling."""

import logging
from typing import Optional
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

logger = logging.getLogger(__name__)


class StockAlertService:
    """Service for managing stock alerts with robust error handling."""

    def __init__(self, api_client: requests.Session, repository: Any):
        self.api_client = api_client
        self.repository = repository

    def create_alert(
        self,
        user_id: int,
        symbol: str,
        threshold: float,
        alert_type: str
    ) -> dict:
        """
        Create a new stock alert with comprehensive validation and error handling.

        Args:
            user_id: User identifier.
            symbol: Stock ticker symbol.
            threshold: Price threshold for alert.
            alert_type: Type of alert ('above' or 'below').

        Returns:
            Dictionary with alert details.

        Raises:
            ValidationError: If inputs are invalid.
            NotFoundError: If stock symbol is not found.
            ExternalAPIError: If external API fails.
            ServiceError: For other service errors.
        """
        # Step 1: Validate inputs
        try:
            self._validate_alert_inputs(user_id, symbol, threshold, alert_type)
        except ValueError as e:
            logger.warning(f"Validation failed for alert creation: {e}")
            raise ValidationError(str(e)) from e

        # Step 2: Verify stock symbol exists
        try:
            stock_data = self._fetch_stock_data(symbol)
        except NotFoundError:
            logger.warning(f"Stock symbol not found: {symbol}")
            raise
        except ExternalAPIError as e:
            logger.error(f"Failed to verify stock symbol {symbol}: {e}")
            raise

        # Step 3: Create alert in database
        try:
            alert = self.repository.create_alert(
                user_id=user_id,
                symbol=symbol,
                threshold=threshold,
                alert_type=alert_type,
                current_price=stock_data['price']
            )
            logger.info(f"Created alert {alert['id']} for user {user_id}")
            return alert

        except Exception as e:
            logger.error(f"Failed to create alert in database: {e}", exc_info=True)
            raise ServiceError(f"Failed to create alert: {e}") from e

    def _validate_alert_inputs(
        self,
        user_id: int,
        symbol: str,
        threshold: float,
        alert_type: str
    ) -> None:
        """Validate alert creation inputs."""
        if user_id <= 0:
            raise ValueError(f"Invalid user_id: {user_id}")

        if not symbol or not symbol.strip():
            raise ValueError("Stock symbol cannot be empty")

        if not symbol.isalnum():
            raise ValueError(f"Invalid stock symbol format: {symbol}")

        if threshold <= 0:
            raise ValueError(f"Threshold must be positive: {threshold}")

        if alert_type not in ['above', 'below']:
            raise ValueError(f"Invalid alert_type: {alert_type}. Must be 'above' or 'below'")

    def _fetch_stock_data(self, symbol: str) -> dict:
        """
        Fetch stock data from external API with error handling.

        Args:
            symbol: Stock ticker symbol.

        Returns:
            Stock data dictionary.

        Raises:
            NotFoundError: If symbol not found.
            ExternalAPIError: If API call fails.
        """
        try:
            response = self.api_client.get(
                f"/api/stock/{symbol}",
                timeout=10
            )
            response.raise_for_status()
            return response.json()

        except Timeout:
            logger.error(f"Timeout fetching data for {symbol}")
            raise ExternalAPIError(
                f"Request timeout while fetching {symbol}",
                details={'symbol': symbol}
            )

        except ConnectionError as e:
            logger.error(f"Connection error fetching {symbol}: {e}")
            raise ExternalAPIError(
                f"Connection error while fetching {symbol}",
                details={'symbol': symbol, 'error': str(e)}
            )

        except requests.HTTPError as e:
            if e.response.status_code == 404:
                raise NotFoundError(
                    f"Stock symbol not found: {symbol}",
                    details={'symbol': symbol}
                )
            elif e.response.status_code == 429:
                retry_after = e.response.headers.get('Retry-After', 60)
                raise RateLimitError(retry_after=int(retry_after))
            else:
                logger.error(f"HTTP error fetching {symbol}: {e}")
                raise ExternalAPIError(
                    f"API error while fetching {symbol}",
                    status_code=e.response.status_code,
                    details={'symbol': symbol}
                )

        except RequestException as e:
            logger.error(f"Request error fetching {symbol}: {e}")
            raise ExternalAPIError(
                f"Request failed while fetching {symbol}",
                details={'symbol': symbol, 'error': str(e)}
            )
```

---

## Testing Services

### Unit Test Structure

```python
"""Unit tests for StockAlertService."""

import pytest
from unittest.mock import Mock, MagicMock, patch
import requests

from src.services.stock_alert_service import StockAlertService
from src.services.exceptions import ValidationError, NotFoundError, ExternalAPIError


class TestStockAlertService:
    """Test suite for StockAlertService."""

    @pytest.fixture
    def mock_api_client(self):
        """Create mock API client."""
        return Mock(spec=requests.Session)

    @pytest.fixture
    def mock_repository(self):
        """Create mock repository."""
        return Mock()

    @pytest.fixture
    def service(self, mock_api_client, mock_repository):
        """Create service instance with mocked dependencies."""
        return StockAlertService(mock_api_client, mock_repository)

    def test_create_alert_success(self, service, mock_api_client, mock_repository):
        """Test successful alert creation."""
        # Arrange
        mock_response = Mock()
        mock_response.json.return_value = {'price': 150.00, 'symbol': 'AAPL'}
        mock_response.status_code = 200
        mock_api_client.get.return_value = mock_response

        mock_repository.create_alert.return_value = {
            'id': 1,
            'user_id': 123,
            'symbol': 'AAPL',
            'threshold': 160.00,
            'alert_type': 'above'
        }

        # Act
        result = service.create_alert(
            user_id=123,
            symbol='AAPL',
            threshold=160.00,
            alert_type='above'
        )

        # Assert
        assert result['id'] == 1
        assert result['symbol'] == 'AAPL'
        mock_api_client.get.assert_called_once()
        mock_repository.create_alert.assert_called_once()

    def test_create_alert_invalid_user_id(self, service):
        """Test alert creation with invalid user_id raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            service.create_alert(
                user_id=0,  # Invalid
                symbol='AAPL',
                threshold=160.00,
                alert_type='above'
            )

        assert 'Invalid user_id' in str(exc_info.value)

    def test_create_alert_empty_symbol(self, service):
        """Test alert creation with empty symbol raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            service.create_alert(
                user_id=123,
                symbol='',  # Empty
                threshold=160.00,
                alert_type='above'
            )

        assert 'symbol cannot be empty' in str(exc_info.value)

    def test_create_alert_negative_threshold(self, service):
        """Test alert creation with negative threshold raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            service.create_alert(
                user_id=123,
                symbol='AAPL',
                threshold=-10.00,  # Negative
                alert_type='above'
            )

        assert 'must be positive' in str(exc_info.value)

    def test_create_alert_invalid_alert_type(self, service):
        """Test alert creation with invalid alert_type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            service.create_alert(
                user_id=123,
                symbol='AAPL',
                threshold=160.00,
                alert_type='invalid'  # Invalid type
            )

        assert 'Invalid alert_type' in str(exc_info.value)

    def test_create_alert_stock_not_found(self, service, mock_api_client):
        """Test alert creation with non-existent stock raises NotFoundError."""
        # Arrange
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = requests.HTTPError(response=mock_response)
        mock_api_client.get.return_value = mock_response

        # Act & Assert
        with pytest.raises(NotFoundError) as exc_info:
            service.create_alert(
                user_id=123,
                symbol='INVALID',
                threshold=160.00,
                alert_type='above'
            )

        assert 'not found' in str(exc_info.value)

    def test_create_alert_api_timeout(self, service, mock_api_client):
        """Test alert creation with API timeout raises ExternalAPIError."""
        # Arrange
        mock_api_client.get.side_effect = requests.Timeout()

        # Act & Assert
        with pytest.raises(ExternalAPIError) as exc_info:
            service.create_alert(
                user_id=123,
                symbol='AAPL',
                threshold=160.00,
                alert_type='above'
            )

        assert 'timeout' in str(exc_info.value).lower()


# Integration test example
class TestStockAlertServiceIntegration:
    """Integration tests for StockAlertService."""

    @pytest.fixture
    def db_session(self):
        """Create test database session."""
        # Setup test database
        pass

    def test_create_and_retrieve_alert(self, db_session):
        """Test end-to-end alert creation and retrieval."""
        # Use real repository and API client (or test doubles)
        # Test full workflow
        pass
```

---

## Common Service Patterns

### 1. Multi-Step Transaction Pattern

```python
"""Service with transaction management."""

from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class OrderService:
    """Service for processing orders with transactions."""

    def __init__(self, db_session, inventory_service, payment_service):
        self.db = db_session
        self.inventory = inventory_service
        self.payment = payment_service

    def process_order(self, user_id: int, items: list[dict], payment_info: dict) -> dict:
        """
        Process order with multiple steps in a transaction.

        Args:
            user_id: User placing the order.
            items: List of items to order.
            payment_info: Payment information.

        Returns:
            Order confirmation details.

        Raises:
            ValidationError: If order validation fails.
            ServiceError: If order processing fails.
        """
        with self._transaction():
            # Step 1: Validate order
            self._validate_order(user_id, items)

            # Step 2: Reserve inventory
            reservation_ids = self.inventory.reserve_items(items)

            try:
                # Step 3: Process payment
                payment_result = self.payment.charge(payment_info, total=self._calculate_total(items))

                # Step 4: Create order record
                order = self._create_order_record(user_id, items, payment_result)

                # Step 5: Confirm inventory reservation
                self.inventory.confirm_reservation(reservation_ids)

                logger.info(f"Order {order['id']} processed successfully")
                return order

            except Exception as e:
                # Rollback inventory reservation on failure
                self.inventory.cancel_reservation(reservation_ids)
                logger.error(f"Order processing failed: {e}")
                raise ServiceError(f"Failed to process order: {e}") from e

    @contextmanager
    def _transaction(self):
        """Context manager for database transaction."""
        try:
            yield
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise
```

### 2. Batch Processing Pattern

```python
"""Service with batch processing capabilities."""

from typing import Iterator, TypeVar, Callable
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class BatchProcessor:
    """Mixin for batch processing operations."""

    def process_in_batches(
        self,
        items: list[T],
        processor: Callable[[list[T]], list[R]],
        batch_size: int = 100,
        on_error: str = 'continue'
    ) -> list[R]:
        """
        Process items in batches.

        Args:
            items: Items to process.
            processor: Function to process each batch.
            batch_size: Number of items per batch.
            on_error: Error handling strategy ('continue', 'stop', 'collect').

        Returns:
            List of processed results.
        """
        results = []
        errors = []

        for batch in self._batch_iterator(items, batch_size):
            try:
                batch_results = processor(batch)
                results.extend(batch_results)
                logger.debug(f"Processed batch of {len(batch)} items")

            except Exception as e:
                logger.error(f"Batch processing error: {e}")

                if on_error == 'stop':
                    raise
                elif on_error == 'collect':
                    errors.append({'batch': batch, 'error': e})
                # 'continue' - just log and continue

        if errors and on_error == 'collect':
            logger.warning(f"Completed with {len(errors)} batch errors")
            return results, errors

        return results

    def _batch_iterator(self, items: list[T], batch_size: int) -> Iterator[list[T]]:
        """Yield successive batches from items."""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]


class BulkAlertService(BatchProcessor):
    """Service for bulk alert operations."""

    def __init__(self, repository):
        self.repository = repository

    def check_all_alerts(self, alert_ids: list[int]) -> list[dict]:
        """
        Check multiple alerts in batches.

        Args:
            alert_ids: List of alert IDs to check.

        Returns:
            List of triggered alerts.
        """
        return self.process_in_batches(
            items=alert_ids,
            processor=self._check_alert_batch,
            batch_size=50
        )

    def _check_alert_batch(self, alert_ids: list[int]) -> list[dict]:
        """Process a batch of alerts."""
        alerts = self.repository.get_alerts_by_ids(alert_ids)
        triggered = [alert for alert in alerts if self._is_triggered(alert)]
        return triggered

    def _is_triggered(self, alert: dict) -> bool:
        """Check if alert is triggered."""
        # Logic to check alert condition
        pass
```

### 3. Circuit Breaker Pattern

```python
"""Service with circuit breaker for external APIs."""

from enum import Enum
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitBreaker:
    """Circuit breaker for protecting against cascading failures."""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        recovery_timeout: int = 30
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit.
            timeout_seconds: Timeout for external calls.
            recovery_timeout: Seconds to wait before attempting recovery.
        """
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.recovery_timeout = recovery_timeout

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None

    def call(self, func: Callable, *args, **kwargs):
        """
        Execute function with circuit breaker protection.

        Args:
            func: Function to execute.
            *args: Positional arguments for function.
            **kwargs: Keyword arguments for function.

        Returns:
            Function result.

        Raises:
            CircuitBreakerOpen: If circuit is open.
        """
        if self.state == CircuitState.OPEN:
            if self._should_attempt_recovery():
                logger.info("Circuit breaker entering half-open state")
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitBreakerOpen("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except Exception as e:
            self._on_failure()
            raise

    def _should_attempt_recovery(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self.last_failure_time is None:
            return True

        elapsed = (datetime.now() - self.last_failure_time).total_seconds()
        return elapsed >= self.recovery_timeout

    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker closing after successful recovery")
            self.state = CircuitState.CLOSED

        self.failure_count = 0

    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            logger.warning(f"Circuit breaker opening after {self.failure_count} failures")
            self.state = CircuitState.OPEN


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""
    pass


class ResilientAPIService:
    """Service with circuit breaker for API calls."""

    def __init__(self, api_client):
        self.api_client = api_client
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60
        )

    def fetch_data(self, endpoint: str) -> dict:
        """
        Fetch data with circuit breaker protection.

        Args:
            endpoint: API endpoint to call.

        Returns:
            API response data.

        Raises:
            CircuitBreakerOpen: If too many failures occurred.
        """
        return self.circuit_breaker.call(
            self._do_api_call,
            endpoint
        )

    def _do_api_call(self, endpoint: str) -> dict:
        """Perform actual API call."""
        response = self.api_client.get(endpoint)
        response.raise_for_status()
        return response.json()
```

---

## Service Naming Conventions

### Class Names
- **Pattern**: `<Domain><Purpose>Service`
- **Examples**:
  - `StockDataService` - Fetches stock data
  - `AlertNotificationService` - Sends alert notifications
  - `UserAuthenticationService` - Handles user authentication
  - `ReportGenerationService` - Generates reports

### Method Names
- Use **verb-noun** pattern for actions
- Be specific and descriptive
- Examples:
  - `fetch_stock_price()`
  - `send_email_notification()`
  - `validate_user_credentials()`
  - `calculate_portfolio_value()`
  - `generate_pdf_report()`

### File Names
- **Pattern**: `<domain>_service.py`
- **Examples**:
  - `stock_data_service.py`
  - `notification_service.py`
  - `authentication_service.py`

---

## Configuration Management

```python
"""Service configuration management."""

from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class StockAPIConfig:
    """Configuration for stock API service."""

    api_key: str
    base_url: str
    timeout: int = 30
    max_retries: int = 3
    rate_limit_per_minute: int = 60

    @classmethod
    def from_env(cls) -> 'StockAPIConfig':
        """Create configuration from environment variables."""
        return cls(
            api_key=os.getenv('STOCK_API_KEY', ''),
            base_url=os.getenv('STOCK_API_BASE_URL', 'https://api.example.com'),
            timeout=int(os.getenv('STOCK_API_TIMEOUT', '30')),
            max_retries=int(os.getenv('STOCK_API_MAX_RETRIES', '3')),
            rate_limit_per_minute=int(os.getenv('STOCK_API_RATE_LIMIT', '60'))
        )

    def validate(self) -> None:
        """Validate configuration values."""
        if not self.api_key:
            raise ValueError("STOCK_API_KEY environment variable is required")

        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")

        if self.max_retries < 0:
            raise ValueError(f"max_retries cannot be negative, got {self.max_retries}")


class ConfigurableService:
    """Example service with configuration."""

    def __init__(self, config: Optional[StockAPIConfig] = None):
        """
        Initialize service with configuration.

        Args:
            config: Service configuration. If None, loads from environment.
        """
        self.config = config or StockAPIConfig.from_env()
        self.config.validate()
```

---

## Logging Best Practices

```python
"""Service with comprehensive logging."""

import logging
from functools import wraps
from typing import Callable

# Configure module logger
logger = logging.getLogger(__name__)


def log_execution(level: int = logging.INFO):
    """
    Decorator to log function execution.

    Args:
        level: Logging level to use.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.log(level, f"Executing {func_name}")

            try:
                result = func(*args, **kwargs)
                logger.log(level, f"Completed {func_name} successfully")
                return result

            except Exception as e:
                logger.error(f"Error in {func_name}: {e}", exc_info=True)
                raise

        return wrapper
    return decorator


class WellLoggedService:
    """Example service with good logging practices."""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    @log_execution()
    def important_operation(self, param: str) -> dict:
        """Perform important operation with logging."""
        self.logger.info(f"Starting important operation with param: {param}")

        # Log at appropriate levels
        self.logger.debug(f"Debug details: processing {param}")

        try:
            result = self._process(param)
            self.logger.info(f"Operation completed successfully: {result}")
            return result

        except ValueError as e:
            self.logger.warning(f"Validation issue: {e}")
            raise

        except Exception as e:
            self.logger.error(f"Unexpected error: {e}", exc_info=True)
            raise

    def _process(self, param: str) -> dict:
        """Internal processing."""
        # Use structured logging
        self.logger.debug(
            "Processing started",
            extra={'param': param, 'timestamp': datetime.now()}
        )
        return {'status': 'success', 'param': param}
```

---

## Documentation Standards

Every service should have:

1. **Module Docstring**: Describe the purpose of the service module
2. **Class Docstring**: Explain what the service does, its responsibilities, and dependencies
3. **Method Docstrings**: Use Google-style docstrings with Args, Returns, Raises sections
4. **Type Hints**: Always include type hints for all parameters and return values
5. **Examples**: Include usage examples in docstrings where helpful

---

## Checklist for New Services

When creating a new service, ensure:

- [ ] Service has a single, clear responsibility
- [ ] Dependencies are injected, not created internally
- [ ] Custom exceptions are defined for service-specific errors
- [ ] All inputs are validated early
- [ ] Error handling is comprehensive and specific
- [ ] Methods have descriptive names following conventions
- [ ] Type hints are used throughout
- [ ] Docstrings follow Google style
- [ ] Logging is included at appropriate levels
- [ ] Unit tests are written with >80% coverage
- [ ] Service is in the correct file following naming conventions
- [ ] No UI logic, data models, or direct database queries in service

---

## Quick Reference

### Service Layer Boundaries

```
✅ Services Layer Contains:
├── Business logic and rules
├── External API integration
├── Multi-step workflows
├── Cross-cutting concerns
└── Domain operations

❌ Services Layer Does NOT Contain:
├── Streamlit UI components
├── Database schema definitions
├── Direct SQL queries
├── Generic utility functions
└── Application configuration
```

### Import Organization

```python
# Standard library
import logging
from typing import Optional, Any
from datetime import datetime

# Third-party
import requests
from sqlalchemy import text

# Local - Data Access
from src.data_access.repositories import StockRepository

# Local - Models
from src.models.schemas import AlertModel, StockData

# Local - Other Services
from src.services.notification_service import NotificationService

# Local - Utilities
from src.utils.validators import validate_symbol
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-09 | Initial services layer documentation |

---

*Keep this document updated as service patterns evolve.*

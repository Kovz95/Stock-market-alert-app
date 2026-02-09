# Data Access Layer - AI Assistant Instructions

## Purpose of the Data Access Layer

The data access layer is the **single point of interaction** with the database. It encapsulates all database operations, queries, and data persistence logic, providing a clean interface for the service layer to work with data without knowing the underlying database structure or SQL.

**Key Principle**: The data access layer is responsible for "how data is stored and retrieved" not "what business operations are performed with that data".

---

## What Belongs in Data Access

### ✅ Should Be in Data Access

1. **Database Queries**
   - SELECT, INSERT, UPDATE, DELETE operations
   - Complex joins and aggregations
   - Raw SQL queries when needed
   - ORM query builders

2. **Repository Pattern Implementation**
   - CRUD operations for entities
   - Custom query methods
   - Data filtering and pagination
   - Bulk operations

3. **Database Connection Management**
   - Connection pooling
   - Session management
   - Transaction handling
   - Connection lifecycle

4. **Data Mapping**
   - Converting database rows to domain models
   - Mapping domain models to database records
   - Handling NULL values and defaults
   - Type conversions

5. **Query Optimization**
   - Index usage
   - Query performance monitoring
   - Eager/lazy loading strategies
   - Query caching

### ❌ Should NOT Be in Data Access

1. **Business Logic**
   - Complex calculations
   - Business rule validation
   - Workflow orchestration
   - → These belong in `src/services/`

2. **External API Calls**
   - HTTP requests to third parties
   - API integration logic
   - → These belong in `src/services/`

3. **UI Components**
   - Streamlit widgets
   - Display formatting
   - → These belong in `src/components/` or page files

4. **Data Models/Schemas**
   - Pydantic models for validation
   - API request/response schemas
   - → These belong in `src/models/`
   - Note: ORM models (SQLAlchemy models) DO belong here

5. **Utility Functions**
   - Generic helpers
   - String manipulation
   - → These belong in `src/utils/`

---

## Repository Pattern

The repository pattern provides a collection-like interface for accessing domain objects. Each repository handles data access for a specific entity or aggregate.

### Repository Design Principles

1. **One Repository Per Aggregate Root**: Create repositories for main entities, not every table
2. **Domain-Focused Interface**: Methods should reflect domain operations, not database operations
3. **Hide Implementation Details**: Don't expose ORM models or SQL directly
4. **Return Domain Models**: Convert database records to domain models before returning

---

## Low-Level Design Patterns

### Repository Base Class Template

```python
"""
Base repository providing common CRUD operations.

This module defines the base repository pattern that all repositories
should inherit from, providing consistent data access patterns.
"""

from typing import TypeVar, Generic, Optional, Type, Any
from abc import ABC, abstractmethod
from datetime import datetime
import logging

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError
from sqlalchemy import select, update, delete, func

logger = logging.getLogger(__name__)


# Type variable for the domain model
TModel = TypeVar('TModel')


class RepositoryError(Exception):
    """Base exception for repository errors."""
    pass


class NotFoundError(RepositoryError):
    """Raised when a requested entity is not found."""

    def __init__(self, entity_type: str, entity_id: Any):
        """
        Initialize not found error.

        Args:
            entity_type: Type of entity that wasn't found.
            entity_id: ID of the entity.
        """
        self.entity_type = entity_type
        self.entity_id = entity_id
        super().__init__(f"{entity_type} with id {entity_id} not found")


class DuplicateError(RepositoryError):
    """Raised when trying to create a duplicate entity."""

    def __init__(self, entity_type: str, field: str, value: Any):
        """
        Initialize duplicate error.

        Args:
            entity_type: Type of entity.
            field: Field that has duplicate value.
            value: The duplicate value.
        """
        self.entity_type = entity_type
        self.field = field
        self.value = value
        super().__init__(
            f"{entity_type} with {field}='{value}' already exists"
        )


class BaseRepository(ABC, Generic[TModel]):
    """
    Base repository providing common CRUD operations.

    This abstract base class provides standard create, read, update, and
    delete operations for database entities. All concrete repositories
    should inherit from this class.

    Type Parameters:
        TModel: The SQLAlchemy model class for this repository.

    Attributes:
        session: SQLAlchemy database session.
        model_class: The SQLAlchemy model class.

    Example:
        >>> class UserRepository(BaseRepository[User]):
        ...     def __init__(self, session: Session):
        ...         super().__init__(session, User)
    """

    def __init__(self, session: Session, model_class: Type[TModel]):
        """
        Initialize repository with database session.

        Args:
            session: SQLAlchemy database session.
            model_class: The SQLAlchemy model class for this repository.
        """
        self.session = session
        self.model_class = model_class
        self._entity_name = model_class.__name__

        logger.debug(f"Initialized {self.__class__.__name__}")

    def get_by_id(self, entity_id: Any) -> Optional[TModel]:
        """
        Retrieve an entity by its ID.

        Args:
            entity_id: Primary key value of the entity.

        Returns:
            The entity if found, None otherwise.

        Example:
            >>> user = user_repo.get_by_id(123)
            >>> if user:
            ...     print(user.name)
        """
        try:
            result = self.session.get(self.model_class, entity_id)
            if result:
                logger.debug(f"Retrieved {self._entity_name} with id {entity_id}")
            return result

        except SQLAlchemyError as e:
            logger.error(f"Error retrieving {self._entity_name} {entity_id}: {e}")
            raise RepositoryError(f"Failed to retrieve {self._entity_name}") from e

    def get_by_id_or_raise(self, entity_id: Any) -> TModel:
        """
        Retrieve an entity by ID or raise exception if not found.

        Args:
            entity_id: Primary key value of the entity.

        Returns:
            The entity.

        Raises:
            NotFoundError: If entity is not found.

        Example:
            >>> user = user_repo.get_by_id_or_raise(123)
        """
        result = self.get_by_id(entity_id)
        if result is None:
            raise NotFoundError(self._entity_name, entity_id)
        return result

    def get_all(
        self,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> list[TModel]:
        """
        Retrieve all entities with optional pagination.

        Args:
            limit: Maximum number of entities to return.
            offset: Number of entities to skip.

        Returns:
            List of entities.

        Example:
            >>> # Get first 10 users
            >>> users = user_repo.get_all(limit=10, offset=0)
            >>> # Get next 10 users
            >>> users = user_repo.get_all(limit=10, offset=10)
        """
        try:
            query = select(self.model_class)

            if offset:
                query = query.offset(offset)
            if limit:
                query = query.limit(limit)

            result = self.session.execute(query).scalars().all()
            logger.debug(f"Retrieved {len(result)} {self._entity_name} entities")
            return list(result)

        except SQLAlchemyError as e:
            logger.error(f"Error retrieving all {self._entity_name}: {e}")
            raise RepositoryError(f"Failed to retrieve {self._entity_name} list") from e

    def count(self) -> int:
        """
        Count total number of entities.

        Returns:
            Total count of entities.

        Example:
            >>> total_users = user_repo.count()
        """
        try:
            query = select(func.count()).select_from(self.model_class)
            result = self.session.execute(query).scalar_one()
            return result

        except SQLAlchemyError as e:
            logger.error(f"Error counting {self._entity_name}: {e}")
            raise RepositoryError(f"Failed to count {self._entity_name}") from e

    def create(self, entity: TModel) -> TModel:
        """
        Create a new entity in the database.

        Args:
            entity: The entity instance to create.

        Returns:
            The created entity with populated ID.

        Raises:
            DuplicateError: If entity violates uniqueness constraint.
            RepositoryError: If creation fails.

        Example:
            >>> new_user = User(name="John", email="john@example.com")
            >>> created_user = user_repo.create(new_user)
            >>> print(created_user.id)  # ID is now populated
        """
        try:
            self.session.add(entity)
            self.session.flush()  # Flush to get the ID
            self.session.refresh(entity)  # Refresh to get any defaults

            logger.info(f"Created {self._entity_name} with id {entity.id}")
            return entity

        except IntegrityError as e:
            self.session.rollback()
            logger.warning(f"Integrity error creating {self._entity_name}: {e}")
            raise DuplicateError(
                self._entity_name,
                "unknown_field",
                "unknown_value"
            ) from e

        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Error creating {self._entity_name}: {e}")
            raise RepositoryError(f"Failed to create {self._entity_name}") from e

    def create_many(self, entities: list[TModel]) -> list[TModel]:
        """
        Create multiple entities in a single operation.

        Args:
            entities: List of entity instances to create.

        Returns:
            List of created entities with populated IDs.

        Raises:
            RepositoryError: If bulk creation fails.

        Example:
            >>> users = [User(name=f"User{i}") for i in range(100)]
            >>> created_users = user_repo.create_many(users)
        """
        try:
            self.session.add_all(entities)
            self.session.flush()

            for entity in entities:
                self.session.refresh(entity)

            logger.info(f"Created {len(entities)} {self._entity_name} entities")
            return entities

        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Error creating multiple {self._entity_name}: {e}")
            raise RepositoryError(f"Failed to create {self._entity_name} batch") from e

    def update(self, entity: TModel) -> TModel:
        """
        Update an existing entity.

        Args:
            entity: The entity instance with updated values.

        Returns:
            The updated entity.

        Raises:
            RepositoryError: If update fails.

        Example:
            >>> user = user_repo.get_by_id(123)
            >>> user.name = "Updated Name"
            >>> updated_user = user_repo.update(user)
        """
        try:
            # Mark entity as modified
            self.session.merge(entity)
            self.session.flush()
            self.session.refresh(entity)

            logger.info(f"Updated {self._entity_name} with id {entity.id}")
            return entity

        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Error updating {self._entity_name}: {e}")
            raise RepositoryError(f"Failed to update {self._entity_name}") from e

    def delete(self, entity_id: Any) -> bool:
        """
        Delete an entity by its ID.

        Args:
            entity_id: Primary key value of the entity to delete.

        Returns:
            True if entity was deleted, False if not found.

        Raises:
            RepositoryError: If deletion fails.

        Example:
            >>> success = user_repo.delete(123)
            >>> if success:
            ...     print("User deleted")
        """
        try:
            stmt = delete(self.model_class).where(
                self.model_class.id == entity_id
            )
            result = self.session.execute(stmt)
            self.session.flush()

            deleted = result.rowcount > 0
            if deleted:
                logger.info(f"Deleted {self._entity_name} with id {entity_id}")
            else:
                logger.warning(f"{self._entity_name} {entity_id} not found for deletion")

            return deleted

        except SQLAlchemyError as e:
            self.session.rollback()
            logger.error(f"Error deleting {self._entity_name} {entity_id}: {e}")
            raise RepositoryError(f"Failed to delete {self._entity_name}") from e

    def exists(self, entity_id: Any) -> bool:
        """
        Check if an entity exists by ID.

        Args:
            entity_id: Primary key value to check.

        Returns:
            True if entity exists, False otherwise.

        Example:
            >>> if user_repo.exists(123):
            ...     print("User exists")
        """
        try:
            query = select(func.count()).select_from(self.model_class).where(
                self.model_class.id == entity_id
            )
            count = self.session.execute(query).scalar_one()
            return count > 0

        except SQLAlchemyError as e:
            logger.error(f"Error checking existence of {self._entity_name} {entity_id}: {e}")
            raise RepositoryError(f"Failed to check {self._entity_name} existence") from e

    def refresh(self, entity: TModel) -> TModel:
        """
        Refresh entity from database to get latest values.

        Args:
            entity: Entity instance to refresh.

        Returns:
            Refreshed entity.

        Example:
            >>> user = user_repo.get_by_id(123)
            >>> # ... some time passes, data might have changed ...
            >>> user = user_repo.refresh(user)
        """
        try:
            self.session.refresh(entity)
            return entity
        except SQLAlchemyError as e:
            logger.error(f"Error refreshing {self._entity_name}: {e}")
            raise RepositoryError(f"Failed to refresh {self._entity_name}") from e
```

### Concrete Repository Example

```python
"""
Repository for Stock Alert entities.

This module provides data access methods for stock alerts,
including creating, retrieving, and managing alert records.
"""

from typing import Optional
from datetime import datetime, timedelta
import logging

from sqlalchemy import select, and_, or_
from sqlalchemy.orm import Session

from src.data_access.base_repository import BaseRepository, NotFoundError
from src.data_access.models import StockAlert  # ORM model
from src.models.schemas import AlertStatus, AlertType  # Domain enums

logger = logging.getLogger(__name__)


class StockAlertRepository(BaseRepository[StockAlert]):
    """
    Repository for managing stock alert data access.

    Provides methods for creating, retrieving, and managing stock alerts
    in the database. Handles all database operations related to alerts.

    Example:
        >>> from src.data_access.database import get_session
        >>> with get_session() as session:
        ...     repo = StockAlertRepository(session)
        ...     alert = repo.get_by_id(123)
    """

    def __init__(self, session: Session):
        """
        Initialize StockAlertRepository.

        Args:
            session: SQLAlchemy database session.
        """
        super().__init__(session, StockAlert)

    def get_by_user_id(
        self,
        user_id: int,
        active_only: bool = False
    ) -> list[StockAlert]:
        """
        Retrieve all alerts for a specific user.

        Args:
            user_id: User ID to filter by.
            active_only: If True, return only active alerts. Defaults to False.

        Returns:
            List of alert entities for the user.

        Example:
            >>> alerts = repo.get_by_user_id(123, active_only=True)
            >>> for alert in alerts:
            ...     print(alert.symbol, alert.threshold)
        """
        try:
            query = select(StockAlert).where(StockAlert.user_id == user_id)

            if active_only:
                query = query.where(StockAlert.status == AlertStatus.ACTIVE)

            query = query.order_by(StockAlert.created_at.desc())

            result = self.session.execute(query).scalars().all()
            logger.debug(f"Retrieved {len(result)} alerts for user {user_id}")
            return list(result)

        except Exception as e:
            logger.error(f"Error retrieving alerts for user {user_id}: {e}")
            raise

    def get_by_symbol(
        self,
        symbol: str,
        user_id: Optional[int] = None
    ) -> list[StockAlert]:
        """
        Retrieve alerts for a specific stock symbol.

        Args:
            symbol: Stock ticker symbol.
            user_id: Optional user ID to filter by.

        Returns:
            List of alerts for the symbol.

        Example:
            >>> # Get all AAPL alerts
            >>> alerts = repo.get_by_symbol('AAPL')
            >>> # Get AAPL alerts for specific user
            >>> alerts = repo.get_by_symbol('AAPL', user_id=123)
        """
        try:
            query = select(StockAlert).where(StockAlert.symbol == symbol.upper())

            if user_id is not None:
                query = query.where(StockAlert.user_id == user_id)

            query = query.order_by(StockAlert.created_at.desc())

            result = self.session.execute(query).scalars().all()
            logger.debug(f"Retrieved {len(result)} alerts for symbol {symbol}")
            return list(result)

        except Exception as e:
            logger.error(f"Error retrieving alerts for symbol {symbol}: {e}")
            raise

    def get_triggered_alerts(
        self,
        current_prices: dict[str, float]
    ) -> list[StockAlert]:
        """
        Find all alerts that should be triggered based on current prices.

        This method performs a complex query to find alerts where the
        threshold condition is met based on the alert type.

        Args:
            current_prices: Dictionary mapping symbols to current prices.

        Returns:
            List of alerts that have been triggered.

        Example:
            >>> prices = {'AAPL': 175.50, 'GOOGL': 140.25}
            >>> triggered = repo.get_triggered_alerts(prices)
        """
        if not current_prices:
            return []

        try:
            symbols = list(current_prices.keys())

            # Get all active alerts for these symbols
            query = select(StockAlert).where(
                and_(
                    StockAlert.symbol.in_(symbols),
                    StockAlert.status == AlertStatus.ACTIVE
                )
            )

            alerts = self.session.execute(query).scalars().all()

            # Filter alerts based on price conditions
            triggered = []
            for alert in alerts:
                current_price = current_prices.get(alert.symbol)
                if current_price is None:
                    continue

                is_triggered = self._check_alert_condition(
                    alert.alert_type,
                    current_price,
                    alert.threshold
                )

                if is_triggered:
                    triggered.append(alert)

            logger.info(f"Found {len(triggered)} triggered alerts")
            return triggered

        except Exception as e:
            logger.error(f"Error finding triggered alerts: {e}")
            raise

    def get_alerts_by_status(
        self,
        status: AlertStatus,
        limit: Optional[int] = None
    ) -> list[StockAlert]:
        """
        Retrieve alerts by status.

        Args:
            status: Alert status to filter by.
            limit: Maximum number of alerts to return.

        Returns:
            List of alerts with the specified status.

        Example:
            >>> active_alerts = repo.get_alerts_by_status(AlertStatus.ACTIVE)
        """
        try:
            query = select(StockAlert).where(StockAlert.status == status)
            query = query.order_by(StockAlert.created_at.desc())

            if limit:
                query = query.limit(limit)

            result = self.session.execute(query).scalars().all()
            return list(result)

        except Exception as e:
            logger.error(f"Error retrieving alerts by status {status}: {e}")
            raise

    def mark_as_triggered(
        self,
        alert_id: int,
        triggered_price: float
    ) -> StockAlert:
        """
        Mark an alert as triggered and record the price.

        Args:
            alert_id: ID of the alert to mark.
            triggered_price: Price at which alert was triggered.

        Returns:
            Updated alert entity.

        Raises:
            NotFoundError: If alert is not found.

        Example:
            >>> alert = repo.mark_as_triggered(123, 175.50)
        """
        alert = self.get_by_id_or_raise(alert_id)

        alert.status = AlertStatus.TRIGGERED
        alert.triggered_at = datetime.now()
        alert.triggered_price = triggered_price

        return self.update(alert)

    def deactivate_alert(self, alert_id: int) -> StockAlert:
        """
        Deactivate an alert.

        Args:
            alert_id: ID of the alert to deactivate.

        Returns:
            Updated alert entity.

        Raises:
            NotFoundError: If alert is not found.

        Example:
            >>> alert = repo.deactivate_alert(123)
        """
        alert = self.get_by_id_or_raise(alert_id)
        alert.status = AlertStatus.INACTIVE
        return self.update(alert)

    def delete_old_triggered_alerts(self, days: int = 30) -> int:
        """
        Delete triggered alerts older than specified days.

        This is a maintenance operation to clean up old triggered alerts.

        Args:
            days: Delete alerts triggered more than this many days ago.

        Returns:
            Number of alerts deleted.

        Example:
            >>> # Delete alerts triggered more than 30 days ago
            >>> deleted_count = repo.delete_old_triggered_alerts(30)
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)

            query = select(StockAlert).where(
                and_(
                    StockAlert.status == AlertStatus.TRIGGERED,
                    StockAlert.triggered_at < cutoff_date
                )
            )

            alerts_to_delete = self.session.execute(query).scalars().all()
            count = len(alerts_to_delete)

            for alert in alerts_to_delete:
                self.session.delete(alert)

            self.session.flush()

            logger.info(f"Deleted {count} old triggered alerts")
            return count

        except Exception as e:
            self.session.rollback()
            logger.error(f"Error deleting old alerts: {e}")
            raise

    def _check_alert_condition(
        self,
        alert_type: AlertType,
        current_price: float,
        threshold: float
    ) -> bool:
        """
        Check if alert condition is met (private helper).

        Args:
            alert_type: Type of alert (above/below).
            current_price: Current stock price.
            threshold: Alert threshold price.

        Returns:
            True if alert condition is met.
        """
        if alert_type == AlertType.ABOVE:
            return current_price >= threshold
        elif alert_type == AlertType.BELOW:
            return current_price <= threshold
        else:
            logger.warning(f"Unknown alert type: {alert_type}")
            return False
```

### ORM Model Definition

```python
"""
SQLAlchemy ORM models for the database.

This module defines the database schema using SQLAlchemy ORM.
Models defined here represent database tables and relationships.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Boolean,
    Enum,
    ForeignKey,
    Index,
    UniqueConstraint,
    CheckConstraint
)
from sqlalchemy.orm import relationship, DeclarativeBase
import enum


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class AlertStatus(enum.Enum):
    """Alert status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRIGGERED = "triggered"


class AlertType(enum.Enum):
    """Alert type enumeration."""
    ABOVE = "above"
    BELOW = "below"


class StockAlert(Base):
    """
    ORM model for stock alerts table.

    Represents a price alert set by a user for a specific stock.

    Attributes:
        id: Primary key.
        user_id: ID of the user who created the alert.
        symbol: Stock ticker symbol.
        threshold: Price threshold for the alert.
        alert_type: Type of alert (above/below).
        status: Current status of the alert.
        created_at: When alert was created.
        triggered_at: When alert was triggered (if applicable).
        triggered_price: Price at which alert was triggered.
    """

    __tablename__ = "stock_alerts"

    # Primary key
    id = Column(Integer, primary_key=True, autoincrement=True)

    # Foreign keys
    user_id = Column(Integer, nullable=False, index=True)

    # Alert data
    symbol = Column(String(10), nullable=False, index=True)
    threshold = Column(Float, nullable=False)
    alert_type = Column(Enum(AlertType), nullable=False)
    status = Column(
        Enum(AlertStatus),
        nullable=False,
        default=AlertStatus.ACTIVE,
        index=True
    )

    # Metadata
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    triggered_at = Column(DateTime, nullable=True)
    triggered_price = Column(Float, nullable=True)

    # Indexes for common queries
    __table_args__ = (
        # Composite index for user's active alerts
        Index('idx_user_status', 'user_id', 'status'),
        # Composite index for symbol queries
        Index('idx_symbol_status', 'symbol', 'status'),
        # Check constraint for threshold
        CheckConstraint('threshold > 0', name='check_positive_threshold'),
        # Unique constraint to prevent duplicate alerts
        UniqueConstraint(
            'user_id',
            'symbol',
            'threshold',
            'alert_type',
            'status',
            name='uq_active_alert'
        ),
    )

    def __repr__(self) -> str:
        """String representation of StockAlert."""
        return (
            f"<StockAlert(id={self.id}, user_id={self.user_id}, "
            f"symbol='{self.symbol}', threshold={self.threshold}, "
            f"type={self.alert_type.value}, status={self.status.value})>"
        )


class User(Base):
    """
    ORM model for users table.

    Represents a user in the system.

    Attributes:
        id: Primary key.
        email: User's email address (unique).
        name: User's display name.
        created_at: When user was created.
        is_active: Whether user account is active.
        alerts: Relationship to user's alerts.
    """

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    is_active = Column(Boolean, nullable=False, default=True)

    # Relationships (if needed)
    # alerts = relationship("StockAlert", back_populates="user")

    def __repr__(self) -> str:
        """String representation of User."""
        return f"<User(id={self.id}, email='{self.email}', name='{self.name}')>"
```

---

## Database Connection Management

### Session Management Pattern

```python
"""
Database connection and session management.

This module provides database connection pooling, session management,
and transaction handling utilities.
"""

from typing import Generator, Optional
from contextlib import contextmanager
import logging

from sqlalchemy import create_engine, event, Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import Pool

from src.data_access.models import Base

logger = logging.getLogger(__name__)


class DatabaseConfig:
    """Database configuration."""

    def __init__(
        self,
        database_url: str,
        echo: bool = False,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_pre_ping: bool = True,
        pool_recycle: int = 3600
    ):
        """
        Initialize database configuration.

        Args:
            database_url: Database connection URL.
            echo: Whether to log SQL statements.
            pool_size: Number of connections to keep in pool.
            max_overflow: Maximum connections beyond pool_size.
            pool_pre_ping: Test connections before using.
            pool_recycle: Recycle connections after this many seconds.
        """
        self.database_url = database_url
        self.echo = echo
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_pre_ping = pool_pre_ping
        self.pool_recycle = pool_recycle


class Database:
    """
    Database manager for connection and session handling.

    Provides a centralized way to manage database connections,
    sessions, and transactions throughout the application.

    Example:
        >>> db = Database(config)
        >>> db.init_db()
        >>> with db.get_session() as session:
        ...     repo = UserRepository(session)
        ...     user = repo.get_by_id(123)
    """

    def __init__(self, config: DatabaseConfig):
        """
        Initialize database manager.

        Args:
            config: Database configuration.
        """
        self.config = config
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None

        logger.info("Database manager initialized")

    @property
    def engine(self) -> Engine:
        """Get database engine, creating it if necessary."""
        if self._engine is None:
            self._create_engine()
        return self._engine

    @property
    def session_factory(self) -> sessionmaker:
        """Get session factory, creating it if necessary."""
        if self._session_factory is None:
            self._create_session_factory()
        return self._session_factory

    def _create_engine(self) -> None:
        """Create database engine with connection pooling."""
        logger.info(f"Creating database engine")

        self._engine = create_engine(
            self.config.database_url,
            echo=self.config.echo,
            pool_size=self.config.pool_size,
            max_overflow=self.config.max_overflow,
            pool_pre_ping=self.config.pool_pre_ping,
            pool_recycle=self.config.pool_recycle,
        )

        # Register event listeners
        event.listen(self._engine, "connect", self._on_connect)
        event.listen(self._engine, "checkout", self._on_checkout)

        logger.info("Database engine created successfully")

    def _create_session_factory(self) -> None:
        """Create session factory."""
        self._session_factory = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False  # Keep objects usable after commit
        )
        logger.debug("Session factory created")

    def _on_connect(self, dbapi_conn, connection_record):
        """Event handler when connection is created."""
        logger.debug("New database connection established")

    def _on_checkout(self, dbapi_conn, connection_record, connection_proxy):
        """Event handler when connection is checked out from pool."""
        logger.debug("Connection checked out from pool")

    def init_db(self) -> None:
        """
        Initialize database schema.

        Creates all tables defined in ORM models. This should be
        called during application setup.

        Example:
            >>> db = Database(config)
            >>> db.init_db()
        """
        logger.info("Initializing database schema")
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database schema initialized")

    def drop_all(self) -> None:
        """
        Drop all database tables.

        WARNING: This will delete all data. Use only in development/testing.

        Example:
            >>> db.drop_all()  # Be very careful with this!
        """
        logger.warning("Dropping all database tables")
        Base.metadata.drop_all(bind=self.engine)
        logger.warning("All tables dropped")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get a database session with automatic cleanup.

        This context manager provides a session and handles commit/rollback
        automatically. The session is committed on success and rolled back
        on exception.

        Yields:
            Database session.

        Example:
            >>> with db.get_session() as session:
            ...     repo = UserRepository(session)
            ...     user = repo.create(new_user)
            ...     # Session is committed automatically
        """
        session = self.session_factory()
        try:
            logger.debug("Database session started")
            yield session
            session.commit()
            logger.debug("Database session committed")

        except Exception as e:
            logger.error(f"Error in database session, rolling back: {e}")
            session.rollback()
            raise

        finally:
            session.close()
            logger.debug("Database session closed")

    def get_raw_session(self) -> Session:
        """
        Get a raw database session without automatic management.

        Use this when you need manual control over session lifecycle.
        You MUST call session.close() when done.

        Returns:
            Database session.

        Example:
            >>> session = db.get_raw_session()
            >>> try:
            ...     # Use session
            ...     session.commit()
            ... finally:
            ...     session.close()
        """
        return self.session_factory()

    def dispose(self) -> None:
        """
        Dispose of database engine and connection pool.

        Call this on application shutdown to cleanly close all connections.

        Example:
            >>> db.dispose()
        """
        if self._engine:
            logger.info("Disposing database engine")
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Database engine disposed")


# Global database instance (initialize in your app startup)
_db_instance: Optional[Database] = None


def init_database(config: DatabaseConfig) -> Database:
    """
    Initialize global database instance.

    Call this during application startup.

    Args:
        config: Database configuration.

    Returns:
        Database instance.

    Example:
        >>> config = DatabaseConfig(database_url="postgresql://...")
        >>> db = init_database(config)
        >>> db.init_db()
    """
    global _db_instance
    _db_instance = Database(config)
    return _db_instance


def get_database() -> Database:
    """
    Get global database instance.

    Returns:
        Database instance.

    Raises:
        RuntimeError: If database not initialized.

    Example:
        >>> db = get_database()
        >>> with db.get_session() as session:
        ...     # Use session
    """
    if _db_instance is None:
        raise RuntimeError(
            "Database not initialized. Call init_database() first."
        )
    return _db_instance


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Convenience function to get a database session.

    This is a shortcut for get_database().get_session().

    Yields:
        Database session.

    Example:
        >>> from src.data_access.database import get_session
        >>> with get_session() as session:
        ...     repo = UserRepository(session)
        ...     users = repo.get_all()
    """
    db = get_database()
    with db.get_session() as session:
        yield session
```

---

## Transaction Management

### Manual Transaction Control

```python
"""Transaction management patterns."""

from typing import Callable, TypeVar
from contextlib import contextmanager
import logging

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

T = TypeVar('T')


@contextmanager
def transaction(session: Session):
    """
    Context manager for explicit transaction control.

    Use when you need to control commit/rollback manually or
    when working with nested operations.

    Args:
        session: Database session.

    Yields:
        The same session.

    Example:
        >>> with get_session() as session:
        ...     with transaction(session):
        ...         repo.create(entity1)
        ...         repo.create(entity2)
        ...         # Both committed together
    """
    try:
        yield session
        session.commit()
        logger.debug("Transaction committed")

    except Exception as e:
        logger.error(f"Transaction failed, rolling back: {e}")
        session.rollback()
        raise


def transactional(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator for methods that should run in a transaction.

    Use this on repository methods that perform multiple operations
    that must be atomic.

    Args:
        func: Function to wrap in transaction.

    Returns:
        Wrapped function.

    Example:
        >>> class MyRepository:
        ...     @transactional
        ...     def complex_operation(self, data):
        ...         self.create(entity1)
        ...         self.update(entity2)
        ...         self.delete(entity3)
    """
    def wrapper(self, *args, **kwargs):
        try:
            result = func(self, *args, **kwargs)
            self.session.commit()
            return result
        except Exception as e:
            logger.error(f"Transaction failed in {func.__name__}: {e}")
            self.session.rollback()
            raise

    return wrapper


class UnitOfWork:
    """
    Unit of Work pattern for managing related operations.

    Groups multiple repository operations into a single transaction.

    Example:
        >>> with get_session() as session:
        ...     uow = UnitOfWork(session)
        ...     user_repo = UserRepository(session)
        ...     alert_repo = AlertRepository(session)
        ...
        ...     user = user_repo.create(new_user)
        ...     alert = alert_repo.create(new_alert)
        ...     uow.commit()
    """

    def __init__(self, session: Session):
        """
        Initialize unit of work.

        Args:
            session: Database session.
        """
        self.session = session
        self._committed = False

    def commit(self) -> None:
        """Commit all pending changes."""
        if not self._committed:
            self.session.commit()
            self._committed = True
            logger.debug("Unit of work committed")

    def rollback(self) -> None:
        """Rollback all pending changes."""
        if not self._committed:
            self.session.rollback()
            logger.debug("Unit of work rolled back")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic commit/rollback."""
        if exc_type is None:
            self.commit()
        else:
            self.rollback()
```

---

## Query Optimization Patterns

### Eager Loading for Relationships

```python
"""Query optimization with eager loading."""

from sqlalchemy import select
from sqlalchemy.orm import selectinload, joinedload


class OptimizedRepository(BaseRepository[User]):
    """Repository with optimized queries."""

    def get_users_with_alerts(self, user_ids: list[int]) -> list[User]:
        """
        Get users with their alerts in a single query (N+1 prevention).

        Uses eager loading to fetch related alerts without additional queries.

        Args:
            user_ids: List of user IDs.

        Returns:
            List of users with alerts loaded.

        Example:
            >>> users = repo.get_users_with_alerts([1, 2, 3])
            >>> for user in users:
            ...     print(f"{user.name} has {len(user.alerts)} alerts")
            ...     # No additional query - alerts already loaded!
        """
        query = (
            select(User)
            .where(User.id.in_(user_ids))
            .options(selectinload(User.alerts))  # Eager load alerts
        )

        result = self.session.execute(query).scalars().all()
        return list(result)

    def get_user_with_active_alerts(self, user_id: int) -> Optional[User]:
        """
        Get user with only active alerts loaded.

        Uses filtered eager loading to load only specific related records.

        Args:
            user_id: User ID.

        Returns:
            User with active alerts loaded.
        """
        query = (
            select(User)
            .where(User.id == user_id)
            .options(
                selectinload(User.alerts).where(
                    StockAlert.status == AlertStatus.ACTIVE
                )
            )
        )

        result = self.session.execute(query).scalar_one_or_none()
        return result
```

### Pagination Pattern

```python
"""Pagination helpers for large result sets."""

from typing import Generic, TypeVar
from dataclasses import dataclass

T = TypeVar('T')


@dataclass
class Page(Generic[T]):
    """
    Paginated result container.

    Attributes:
        items: Items on current page.
        total: Total number of items across all pages.
        page: Current page number (1-indexed).
        page_size: Number of items per page.
        total_pages: Total number of pages.
    """

    items: list[T]
    total: int
    page: int
    page_size: int

    @property
    def total_pages(self) -> int:
        """Calculate total number of pages."""
        return (self.total + self.page_size - 1) // self.page_size

    @property
    def has_next(self) -> bool:
        """Check if there is a next page."""
        return self.page < self.total_pages

    @property
    def has_previous(self) -> bool:
        """Check if there is a previous page."""
        return self.page > 1


class PaginatedRepository(BaseRepository[StockAlert]):
    """Repository with pagination support."""

    def get_paginated(
        self,
        page: int = 1,
        page_size: int = 20,
        user_id: Optional[int] = None
    ) -> Page[StockAlert]:
        """
        Get paginated results.

        Args:
            page: Page number (1-indexed).
            page_size: Number of items per page.
            user_id: Optional user ID filter.

        Returns:
            Page object with results and metadata.

        Example:
            >>> page_result = repo.get_paginated(page=1, page_size=10)
            >>> print(f"Showing page {page_result.page} of {page_result.total_pages}")
            >>> for alert in page_result.items:
            ...     print(alert.symbol)
            >>> if page_result.has_next:
            ...     next_page = repo.get_paginated(page=2, page_size=10)
        """
        # Build base query
        query = select(StockAlert)

        if user_id is not None:
            query = query.where(StockAlert.user_id == user_id)

        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total = self.session.execute(count_query).scalar_one()

        # Get page of results
        query = query.order_by(StockAlert.created_at.desc())
        offset = (page - 1) * page_size
        query = query.offset(offset).limit(page_size)

        items = self.session.execute(query).scalars().all()

        return Page(
            items=list(items),
            total=total,
            page=page,
            page_size=page_size
        )
```

### Bulk Operations Pattern

```python
"""Efficient bulk operations."""

from sqlalchemy import insert, update as sql_update


class BulkOperationRepository(BaseRepository[StockAlert]):
    """Repository with optimized bulk operations."""

    def bulk_insert(self, alert_dicts: list[dict]) -> int:
        """
        Efficiently insert multiple alerts.

        Uses SQLAlchemy's bulk insert for better performance.

        Args:
            alert_dicts: List of dictionaries with alert data.

        Returns:
            Number of alerts inserted.

        Example:
            >>> alerts_data = [
            ...     {'user_id': 1, 'symbol': 'AAPL', 'threshold': 150.0, ...},
            ...     {'user_id': 1, 'symbol': 'GOOGL', 'threshold': 140.0, ...},
            ... ]
            >>> count = repo.bulk_insert(alerts_data)
        """
        if not alert_dicts:
            return 0

        stmt = insert(StockAlert).values(alert_dicts)
        result = self.session.execute(stmt)
        self.session.flush()

        logger.info(f"Bulk inserted {len(alert_dicts)} alerts")
        return len(alert_dicts)

    def bulk_update_status(
        self,
        alert_ids: list[int],
        new_status: AlertStatus
    ) -> int:
        """
        Efficiently update status for multiple alerts.

        Args:
            alert_ids: List of alert IDs to update.
            new_status: New status value.

        Returns:
            Number of alerts updated.

        Example:
            >>> updated = repo.bulk_update_status([1, 2, 3], AlertStatus.INACTIVE)
        """
        if not alert_ids:
            return 0

        stmt = (
            sql_update(StockAlert)
            .where(StockAlert.id.in_(alert_ids))
            .values(status=new_status)
        )

        result = self.session.execute(stmt)
        self.session.flush()

        count = result.rowcount
        logger.info(f"Bulk updated {count} alerts to status {new_status.value}")
        return count
```

---

## Testing Data Access Layer

### Repository Unit Tests

```python
"""Unit tests for repositories."""

import pytest
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.data_access.models import Base, StockAlert, AlertStatus, AlertType
from src.data_access.repositories import StockAlertRepository
from src.data_access.base_repository import NotFoundError


@pytest.fixture(scope="function")
def test_engine():
    """Create in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    yield engine
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture(scope="function")
def test_session(test_engine):
    """Create test database session."""
    SessionLocal = sessionmaker(bind=test_engine)
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def alert_repository(test_session):
    """Create alert repository for testing."""
    return StockAlertRepository(test_session)


@pytest.fixture
def sample_alert(test_session):
    """Create a sample alert for testing."""
    alert = StockAlert(
        user_id=1,
        symbol="AAPL",
        threshold=150.00,
        alert_type=AlertType.ABOVE,
        status=AlertStatus.ACTIVE
    )
    test_session.add(alert)
    test_session.commit()
    test_session.refresh(alert)
    return alert


class TestStockAlertRepository:
    """Test suite for StockAlertRepository."""

    def test_create_alert(self, alert_repository, test_session):
        """Test creating a new alert."""
        alert = StockAlert(
            user_id=1,
            symbol="GOOGL",
            threshold=140.00,
            alert_type=AlertType.BELOW,
            status=AlertStatus.ACTIVE
        )

        created = alert_repository.create(alert)

        assert created.id is not None
        assert created.symbol == "GOOGL"
        assert created.threshold == 140.00
        assert created.status == AlertStatus.ACTIVE

    def test_get_by_id_existing(self, alert_repository, sample_alert):
        """Test retrieving an existing alert by ID."""
        result = alert_repository.get_by_id(sample_alert.id)

        assert result is not None
        assert result.id == sample_alert.id
        assert result.symbol == sample_alert.symbol

    def test_get_by_id_nonexistent(self, alert_repository):
        """Test retrieving a non-existent alert returns None."""
        result = alert_repository.get_by_id(99999)
        assert result is None

    def test_get_by_id_or_raise_success(self, alert_repository, sample_alert):
        """Test get_by_id_or_raise with existing alert."""
        result = alert_repository.get_by_id_or_raise(sample_alert.id)
        assert result.id == sample_alert.id

    def test_get_by_id_or_raise_not_found(self, alert_repository):
        """Test get_by_id_or_raise raises NotFoundError for missing alert."""
        with pytest.raises(NotFoundError) as exc_info:
            alert_repository.get_by_id_or_raise(99999)

        assert "not found" in str(exc_info.value)

    def test_get_by_user_id(self, alert_repository, test_session):
        """Test retrieving alerts by user ID."""
        # Create multiple alerts for user
        for i in range(3):
            alert = StockAlert(
                user_id=1,
                symbol=f"TEST{i}",
                threshold=100.00 + i,
                alert_type=AlertType.ABOVE,
                status=AlertStatus.ACTIVE
            )
            test_session.add(alert)
        test_session.commit()

        results = alert_repository.get_by_user_id(1)

        assert len(results) == 3
        assert all(a.user_id == 1 for a in results)

    def test_get_by_user_id_active_only(self, alert_repository, test_session):
        """Test retrieving only active alerts for user."""
        # Create active and inactive alerts
        active_alert = StockAlert(
            user_id=1,
            symbol="ACTIVE",
            threshold=100.00,
            alert_type=AlertType.ABOVE,
            status=AlertStatus.ACTIVE
        )
        inactive_alert = StockAlert(
            user_id=1,
            symbol="INACTIVE",
            threshold=100.00,
            alert_type=AlertType.ABOVE,
            status=AlertStatus.INACTIVE
        )
        test_session.add_all([active_alert, inactive_alert])
        test_session.commit()

        results = alert_repository.get_by_user_id(1, active_only=True)

        assert len(results) == 1
        assert results[0].status == AlertStatus.ACTIVE

    def test_update_alert(self, alert_repository, sample_alert):
        """Test updating an alert."""
        sample_alert.threshold = 175.00

        updated = alert_repository.update(sample_alert)

        assert updated.threshold == 175.00

    def test_delete_alert(self, alert_repository, sample_alert):
        """Test deleting an alert."""
        alert_id = sample_alert.id

        deleted = alert_repository.delete(alert_id)

        assert deleted is True
        assert alert_repository.get_by_id(alert_id) is None

    def test_mark_as_triggered(self, alert_repository, sample_alert):
        """Test marking alert as triggered."""
        triggered = alert_repository.mark_as_triggered(
            sample_alert.id,
            triggered_price=155.50
        )

        assert triggered.status == AlertStatus.TRIGGERED
        assert triggered.triggered_price == 155.50
        assert triggered.triggered_at is not None

    def test_count(self, alert_repository, test_session):
        """Test counting total alerts."""
        # Create multiple alerts
        for i in range(5):
            alert = StockAlert(
                user_id=1,
                symbol=f"TEST{i}",
                threshold=100.00,
                alert_type=AlertType.ABOVE,
                status=AlertStatus.ACTIVE
            )
            test_session.add(alert)
        test_session.commit()

        count = alert_repository.count()
        assert count == 5


class TestStockAlertRepositoryIntegration:
    """Integration tests for StockAlertRepository with real database operations."""

    def test_transaction_rollback_on_error(self, alert_repository, test_session):
        """Test that transaction rolls back on error."""
        # Create alert
        alert = StockAlert(
            user_id=1,
            symbol="TEST",
            threshold=100.00,
            alert_type=AlertType.ABOVE,
            status=AlertStatus.ACTIVE
        )
        alert_repository.create(alert)

        initial_count = alert_repository.count()

        # Try to create invalid alert
        try:
            invalid_alert = StockAlert(
                user_id=1,
                symbol="TEST",
                threshold=-100.00,  # Will violate check constraint
                alert_type=AlertType.ABOVE,
                status=AlertStatus.ACTIVE
            )
            alert_repository.create(invalid_alert)
        except Exception:
            test_session.rollback()

        # Count should remain the same
        assert alert_repository.count() == initial_count
```

### Test Fixtures and Factories

```python
"""Test fixtures and data factories."""

import factory
from factory.alchemy import SQLAlchemyModelFactory

from src.data_access.models import StockAlert, AlertType, AlertStatus


class StockAlertFactory(SQLAlchemyModelFactory):
    """Factory for creating StockAlert test data."""

    class Meta:
        model = StockAlert
        sqlalchemy_session_persistence = "commit"

    user_id = factory.Sequence(lambda n: n)
    symbol = factory.Sequence(lambda n: f"TEST{n}")
    threshold = factory.Faker("pyfloat", min_value=1, max_value=1000)
    alert_type = factory.Iterator([AlertType.ABOVE, AlertType.BELOW])
    status = AlertStatus.ACTIVE


# Usage in tests
def test_with_factory(test_session):
    """Example test using factory."""
    # Create 10 test alerts easily
    alerts = [StockAlertFactory.create() for _ in range(10)]

    assert len(alerts) == 10
```

---

## Best Practices Checklist

When working with the data access layer:

### Repository Design
- [ ] One repository per aggregate root/main entity
- [ ] Inherit from BaseRepository for consistency
- [ ] Use type hints for all method signatures
- [ ] Return domain models, not ORM models (when applicable)
- [ ] Method names reflect domain operations, not SQL operations
- [ ] Custom exceptions for domain-specific errors

### Query Optimization
- [ ] Use eager loading to prevent N+1 queries
- [ ] Implement pagination for large result sets
- [ ] Use bulk operations for multiple records
- [ ] Add database indexes for frequently queried columns
- [ ] Monitor and log slow queries

### Transaction Management
- [ ] Use context managers for automatic session management
- [ ] Group related operations in transactions
- [ ] Handle rollback on errors
- [ ] Don't hold sessions longer than necessary

### ORM Models
- [ ] Define appropriate indexes and constraints
- [ ] Use enums for fixed value fields
- [ ] Add check constraints for data validation
- [ ] Include timestamps (created_at, updated_at)
- [ ] Implement __repr__ for debugging

### Testing
- [ ] Use in-memory SQLite for fast tests
- [ ] Create test fixtures for common data
- [ ] Test both success and error cases
- [ ] Test transaction rollback behavior
- [ ] Mock external dependencies, not database

### Security
- [ ] Never expose raw SQL to upper layers
- [ ] Use parameterized queries (ORM handles this)
- [ ] Validate data before persisting
- [ ] Don't log sensitive data (passwords, tokens)

---

## Common Patterns Quick Reference

### Basic CRUD Operations
```python
# Create
new_alert = StockAlert(user_id=1, symbol="AAPL", ...)
created = repo.create(new_alert)

# Read
alert = repo.get_by_id(123)
alert = repo.get_by_id_or_raise(123)  # Raises if not found
alerts = repo.get_all(limit=10, offset=0)

# Update
alert.threshold = 160.00
updated = repo.update(alert)

# Delete
deleted = repo.delete(123)  # Returns True/False

# Count
total = repo.count()

# Check existence
exists = repo.exists(123)
```

### Session Management
```python
# Recommended: Context manager
with get_session() as session:
    repo = UserRepository(session)
    user = repo.create(new_user)
    # Automatically commits on success, rolls back on error

# Manual session management (use sparingly)
session = db.get_raw_session()
try:
    repo = UserRepository(session)
    # Operations...
    session.commit()
finally:
    session.close()
```

### Custom Queries
```python
# Filter by multiple conditions
query = select(StockAlert).where(
    and_(
        StockAlert.user_id == user_id,
        StockAlert.status == AlertStatus.ACTIVE,
        StockAlert.threshold >= 100.00
    )
)
results = session.execute(query).scalars().all()

# Order and limit
query = (
    select(StockAlert)
    .where(StockAlert.user_id == user_id)
    .order_by(StockAlert.created_at.desc())
    .limit(10)
)
```

---

## Naming Conventions

### Repository Classes
- **Pattern**: `<Entity>Repository`
- **Examples**: `StockAlertRepository`, `UserRepository`, `PortfolioRepository`

### Repository Methods
- `get_by_<field>()` - Retrieve by specific field
- `get_<plural>_by_<criteria>()` - Retrieve multiple records
- `create()`, `update()`, `delete()` - Basic CRUD
- `count()`, `exists()` - Query helpers
- Domain-specific methods: `mark_as_triggered()`, `deactivate_alert()`

### ORM Model Classes
- **Pattern**: Singular entity name (e.g., `StockAlert`, not `StockAlerts`)
- Table name: Plural lowercase with underscores (e.g., `stock_alerts`)

### File Organization
```
src/data_access/
├── __init__.py
├── CLAUDE.md              # This file
├── database.py            # Connection and session management
├── base_repository.py     # Base repository class
├── models.py              # ORM model definitions
└── repositories.py        # Concrete repository implementations
    # OR separate files:
    ├── stock_alert_repository.py
    ├── user_repository.py
    └── portfolio_repository.py
```

---

## Migration from Direct SQL

If migrating from direct SQL queries to repositories:

### Before (Direct SQL)
```python
# ❌ Don't do this in new code
def get_user_alerts(db_connection, user_id):
    cursor = db_connection.cursor()
    cursor.execute(
        "SELECT * FROM stock_alerts WHERE user_id = ? AND status = 'active'",
        (user_id,)
    )
    return cursor.fetchall()
```

### After (Repository Pattern)
```python
# ✅ Use repository instead
def get_user_alerts(session, user_id):
    repo = StockAlertRepository(session)
    return repo.get_by_user_id(user_id, active_only=True)
```

---

## Performance Guidelines

### Query Performance
1. **Add indexes** for columns used in WHERE, JOIN, ORDER BY clauses
2. **Use select_** methods for loading only needed columns
3. **Batch operations** when working with many records
4. **Eager load** relationships to prevent N+1 queries
5. **Paginate** large result sets

### Connection Pooling
1. Configure appropriate `pool_size` for your workload
2. Use `pool_pre_ping=True` to handle stale connections
3. Set `pool_recycle` to prevent connection timeouts
4. Monitor pool statistics in production

### Session Lifecycle
1. Keep sessions short-lived
2. Don't store sessions in global variables
3. Use context managers for automatic cleanup
4. Commit or rollback explicitly in long-running operations

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-09 | Initial data access layer documentation |

---

*Keep this document updated as data access patterns evolve.*

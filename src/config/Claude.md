# Claude.md - Configuration Directory Guidelines

## Overview

The `src/config/` directory contains application-wide configuration modules that define system behavior, schedules, performance settings, and other operational parameters. These configurations are designed to be:

- **Centralized**: Single source of truth for configuration values
- **Type-safe**: Use type hints and validation where appropriate
- **Documented**: Clear docstrings explaining purpose and usage
- **Testable**: Easy to test and validate
- **Maintainable**: Well-organized and easy to update

---

## Directory Structure

```
src/config/
├── __init__.py                    # Package initialization and exports
├── exchange_schedule_config.py     # Stock exchange schedules and timezone handling
├── performance_config.py           # Performance tuning and optimization settings
└── Claude.md                       # This file
```

---

## Current Configuration Modules

### 1. Exchange Schedule Configuration (`exchange_schedule_config.py`)

**Purpose**: Manages stock exchange closing times, timezone conversions, and DST (Daylight Saving Time) handling for global markets.

**Key Features**:
- Exchange closing times in both EST (winter) and EDT (summer)
- DST transition period handling
- Day offset logic for Asia/Pacific markets
- Timezone-aware scheduling

**Key Exports**:
- `EXCHANGE_SCHEDULES`: Dictionary mapping exchange names to their schedule configurations
- `EXCHANGE_TIMEZONES`: Mapping of exchange names to their timezone identifiers
- `get_exchange_close_time()`: Function to determine correct close time accounting for DST
- `is_dst_active()`: Check if US Eastern Daylight Time is currently active
- `is_in_dst_transition_period()`: Detect DST misalignment periods
- `get_market_days_for_exchange()`: Get US days when exchange alerts should be checked
- `get_exchanges_by_closing_time()`: Group exchanges by their closing times
- `get_weekly_schedule()`: Get weekly schedule for a specific exchange

**Configuration Structure**:
```python
EXCHANGE_SCHEDULES = {
    "EXCHANGE_NAME": {
        "est_close_hour": int,           # Hour in EST (winter)
        "est_close_minute": int,          # Minute in EST (winter)
        "edt_close_hour": int,            # Hour in EDT (summer)
        "edt_close_minute": int,          # Minute in EDT (summer)
        "name": str,                      # Human-readable exchange name
        "notes": str,                     # Additional notes about timing
        "day_offset": int,                # Optional: -1 for Asia/Pacific markets
        "timezone": str,                  # Optional: IANA timezone identifier
        "dst_transition_override": {      # Optional: DST transition handling
            "condition": str,             # "both_dst", "misaligned_dst", or date periods
            "edt_close_hour": int,
            "edt_close_minute": int,
            "periods": [                  # Optional: date-based periods
                {
                    "start_month": int,
                    "start_day": int,
                    "end_month": int,
                    "end_day": int
                }
            ]
        }
    }
}
```

**Usage Example**:
```python
from src.config.exchange_schedule_config import (
    EXCHANGE_SCHEDULES,
    get_exchange_close_time,
    is_dst_active
)

# Get exchange configuration
nasdaq_config = EXCHANGE_SCHEDULES["NASDAQ"]

# Get current close time (auto-detects DST)
hour, minute = get_exchange_close_time(nasdaq_config)

# Check DST status
is_edt = is_dst_active()
```

**Important Notes**:
- Times include a 40-minute delay after market close for data availability
- Asian/Pacific markets that close after midnight ET are processed on the PREVIOUS US day
- DST transition periods require special handling when US and international DST don't align
- Exchange names must match exactly with database exchange names

---

### 2. Performance Configuration (`performance_config.py`)

**Purpose**: Centralizes performance tuning parameters, batch processing settings, caching configurations, and optimization thresholds.

**Key Features**:
- Batch processing settings
- Caching configuration
- API rate limiting
- Memory management
- Performance profiles for different environments
- Dynamic configuration updates

**Key Exports**:
- `BATCH_SIZE`: Number of alerts to process in each batch
- `MAX_WORKERS`: Number of parallel workers for processing
- `CACHE_TTL`: Cache time-to-live in seconds
- `CACHE_MAX_SIZE`: Maximum number of cached items
- `API_RATE_LIMIT`: Requests per second for API calls
- `PERFORMANCE_PROFILES`: Predefined profiles for different environments
- `get_performance_profile()`: Get settings for a specific profile
- `update_performance_settings()`: Dynamically update settings
- `get_optimal_settings()`: Get optimal settings based on alert count

**Configuration Categories**:

1. **Batch Processing**:
   - `BATCH_SIZE`: Default 1000 alerts per batch
   - `MAX_WORKERS`: Default 10 parallel workers

2. **Caching**:
   - `CACHE_TTL`: Default 300 seconds (5 minutes)
   - `CACHE_MAX_SIZE`: Default 1000 items

3. **API Settings**:
   - `API_RATE_LIMIT`: Default 3 requests per second
   - `API_TIMEOUT`: Default 30 seconds

4. **Memory Management**:
   - `MAX_MEMORY_USAGE`: Default 80% before cleanup
   - `MEMORY_CLEANUP_INTERVAL`: Cleanup every 1000 alerts

5. **Performance Thresholds**:
   - `LARGE_ALERT_THRESHOLD`: Switch to batch processing above 1000
   - `OPTIMIZATION_THRESHOLD`: Enable advanced optimizations above 5000

**Performance Profiles**:
```python
PERFORMANCE_PROFILES = {
    "development": {
        "batch_size": 100,
        "max_workers": 5,
        "cache_ttl": 60,
        "monitoring_enabled": True,
    },
    "production": {
        "batch_size": 1000,
        "max_workers": 10,
        "cache_ttl": 300,
        "monitoring_enabled": True,
    },
    "high_volume": {
        "batch_size": 2000,
        "max_workers": 20,
        "cache_ttl": 600,
        "monitoring_enabled": True,
    },
}
```

**Usage Example**:
```python
from src.config.performance_config import (
    BATCH_SIZE,
    get_performance_profile,
    get_optimal_settings,
    update_performance_settings
)

# Use default settings
batch_size = BATCH_SIZE

# Get profile-based settings
prod_settings = get_performance_profile("production")

# Get optimal settings based on alert count
optimal = get_optimal_settings(alert_count=5000)

# Dynamically update settings
update_performance_settings({"batch_size": 2000, "max_workers": 15})
```

---

## Configuration Patterns and Conventions

### 1. Module Structure

Each configuration module should follow this structure:

```python
#!/usr/bin/env python3
"""
Module-level docstring explaining the purpose and scope of the configuration.
"""

import logging
from typing import Any, Dict, Optional  # Add appropriate imports

logger = logging.getLogger(__name__)

# Constants section (uppercase with underscores)
CONFIG_CONSTANT = "value"

# Configuration dictionaries (uppercase with underscores)
CONFIG_DICT: Dict[str, Any] = {
    "key": "value",
}

# Helper functions (lowercase with underscores)
def get_config_value(key: str) -> Optional[Any]:
    """Get a configuration value.
    
    Args:
        key: Configuration key to retrieve.
        
    Returns:
        Configuration value or None if not found.
    """
    return CONFIG_DICT.get(key)

# Validation functions
def validate_config() -> bool:
    """Validate configuration values."""
    # Validation logic
    return True
```

### 2. Naming Conventions

- **Constants**: `UPPER_SNAKE_CASE` (e.g., `BATCH_SIZE`, `CACHE_TTL`)
- **Dictionaries**: `UPPER_SNAKE_CASE` (e.g., `EXCHANGE_SCHEDULES`, `PERFORMANCE_PROFILES`)
- **Functions**: `lower_snake_case` (e.g., `get_exchange_close_time()`, `is_dst_active()`)
- **Classes**: `PascalCase` (if needed, though config modules typically don't use classes)

### 3. Type Hints

Always use type hints for function signatures:

```python
def get_config(
    exchange_name: str,
    use_edt: Optional[bool] = None,
    current_date: Optional[date] = None
) -> tuple[int, int]:
    """Get configuration with type hints."""
    pass
```

### 4. Documentation Standards

- **Module-level docstrings**: Explain the purpose, scope, and key concepts
- **Function docstrings**: Use Google-style docstrings with Args, Returns, Raises sections
- **Inline comments**: Explain complex logic, especially timezone/DST handling
- **Configuration comments**: Document non-obvious values and their rationale

### 5. Error Handling

```python
def get_config_safely(key: str, default: Any = None) -> Any:
    """Get configuration with safe error handling."""
    try:
        return CONFIG_DICT[key]
    except KeyError:
        logger.warning(f"Configuration key '{key}' not found, using default")
        return default
    except Exception as e:
        logger.error(f"Error accessing configuration: {e}")
        return default
```

### 6. Validation

Include validation functions for critical configurations:

```python
def validate_exchange_config(config: Dict[str, Any]) -> bool:
    """Validate exchange configuration structure."""
    required_keys = ["est_close_hour", "est_close_minute", "edt_close_hour", "edt_close_minute"]
    return all(key in config for key in required_keys)
```

---

## Timezone Handling Best Practices

### 1. Always Use IANA Timezone Identifiers

```python
# ✅ Good
import pytz
eastern = pytz.timezone('America/New_York')
london = pytz.timezone('Europe/London')

# ❌ Bad
# Don't use ambiguous abbreviations like 'EST' or 'EDT'
```

### 2. Handle DST Transitions Explicitly

```python
def get_time_with_dst(exchange_config: Dict[str, Any]) -> tuple[int, int]:
    """Get time accounting for DST transitions."""
    use_edt = is_dst_active()
    in_transition = is_in_dst_transition_period()
    
    # Check for DST override
    if in_transition and exchange_config.get('dst_transition_override'):
        override = exchange_config['dst_transition_override']
        return override['edt_close_hour'], override['edt_close_minute']
    
    # Use standard EDT/EST times
    if use_edt:
        return exchange_config['edt_close_hour'], exchange_config['edt_close_minute']
    else:
        return exchange_config['est_close_hour'], exchange_config['est_close_minute']
```

### 3. Document Timezone Assumptions

Always document:
- What timezone the configuration values are in
- How DST transitions are handled
- Any special cases (e.g., markets that don't observe DST)

### 4. Use `pytz` for Timezone Operations

```python
import pytz
from datetime import datetime

# Convert to specific timezone
eastern = pytz.timezone('America/New_York')
now_et = datetime.now(eastern)

# Check DST status
is_dst = bool(now_et.dst())
```

---

## Performance Configuration Best Practices

### 1. Provide Multiple Profiles

Create profiles for different environments:
- `development`: Lower resource usage, more verbose logging
- `production`: Balanced settings for normal operation
- `high_volume`: Optimized for large-scale processing

### 2. Make Settings Adjustable

Allow dynamic updates for runtime tuning:

```python
def update_performance_settings(settings_dict: Dict[str, Any]) -> None:
    """Update performance settings dynamically."""
    global BATCH_SIZE, MAX_WORKERS
    
    if "batch_size" in settings_dict:
        BATCH_SIZE = settings_dict["batch_size"]
        logger.info(f"Updated BATCH_SIZE to {BATCH_SIZE}")
```

### 3. Provide Adaptive Settings

Offer functions that adjust settings based on workload:

```python
def get_optimal_settings(alert_count: int) -> Dict[str, int]:
    """Get optimal settings based on alert count."""
    if alert_count < 100:
        return {"batch_size": 50, "max_workers": 5}
    elif alert_count < 1000:
        return {"batch_size": 200, "max_workers": 8}
    # ... etc
```

### 4. Document Performance Trade-offs

Explain the impact of different settings:
- Higher batch sizes: More memory usage, fewer database calls
- More workers: Better parallelism, but more resource consumption
- Longer cache TTL: Better performance, but potentially stale data

---

## Creating New Configuration Modules

### Step-by-Step Guide

1. **Create the module file**:
   ```python
   # src/config/new_config.py
   ```

2. **Add module docstring**:
   ```python
   """
   Brief description of what this configuration manages.
   
   Detailed explanation of purpose, scope, and key concepts.
   """
   ```

3. **Import required dependencies**:
   ```python
   import logging
   from typing import Any, Dict, Optional
   
   logger = logging.getLogger(__name__)
   ```

4. **Define configuration constants**:
   ```python
   # Configuration constants
   DEFAULT_VALUE = "default"
   MAX_VALUE = 1000
   ```

5. **Define configuration dictionaries**:
   ```python
   CONFIG_DICT: Dict[str, Any] = {
       "key1": "value1",
       "key2": "value2",
   }
   ```

6. **Add helper functions**:
   ```python
   def get_config(key: str, default: Any = None) -> Any:
       """Get configuration value."""
       return CONFIG_DICT.get(key, default)
   ```

7. **Add validation** (if needed):
   ```python
   def validate_config() -> bool:
       """Validate configuration values."""
       # Validation logic
       return True
   ```

8. **Export in `__init__.py`**:
   ```python
   # src/config/__init__.py
   from src.config import new_config
   
   __all__ = ["performance_config", "exchange_schedule_config", "new_config"]
   ```

### Configuration Module Template

```python
#!/usr/bin/env python3
"""
[Module Name] Configuration

[Detailed description of what this configuration manages]
"""

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration Constants
# ============================================================================

DEFAULT_SETTING = "default_value"
MAX_SETTING = 100

# ============================================================================
# Configuration Dictionaries
# ============================================================================

CONFIGURATION: Dict[str, Any] = {
    "setting1": "value1",
    "setting2": 100,
    "setting3": {
        "nested": "value"
    }
}

# ============================================================================
# Helper Functions
# ============================================================================

def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value by key.
    
    Args:
        key: Configuration key to retrieve.
        default: Default value if key not found.
        
    Returns:
        Configuration value or default.
    """
    return CONFIGURATION.get(key, default)


def update_config(updates: Dict[str, Any]) -> None:
    """Update configuration values.
    
    Args:
        updates: Dictionary of configuration updates.
    """
    CONFIGURATION.update(updates)
    logger.info(f"Updated configuration: {updates}")


def validate_config() -> bool:
    """Validate configuration values.
    
    Returns:
        True if configuration is valid, False otherwise.
    """
    # Add validation logic
    return True


# ============================================================================
# Main Execution (for testing)
# ============================================================================

if __name__ == "__main__":
    print(f"Configuration loaded: {len(CONFIGURATION)} settings")
    print(f"Validation: {validate_config()}")
```

---

## Integration with Application

### Import Patterns

**Preferred**: Import specific functions/constants
```python
from src.config.exchange_schedule_config import EXCHANGE_SCHEDULES, get_exchange_close_time
from src.config.performance_config import BATCH_SIZE, get_performance_profile
```

**Acceptable**: Import entire module if needed
```python
from src.config import exchange_schedule_config
config = exchange_schedule_config.EXCHANGE_SCHEDULES["NASDAQ"]
```

**Avoid**: Wildcard imports
```python
# ❌ Don't do this
from src.config.exchange_schedule_config import *
```

### Usage in Services

Configuration modules are typically used in:
- **Schedulers**: `auto_scheduler_v2.py`, `hourly_data_scheduler.py`
- **Services**: `calendar_adapter.py`, `performance_monitor.py`
- **Pages**: `Daily_Weekly_Scheduler_Status.py`

Example:
```python
from src.config.exchange_schedule_config import (
    EXCHANGE_SCHEDULES,
    get_exchange_close_time,
    is_dst_active
)

def schedule_exchange_alerts(exchange_name: str):
    """Schedule alerts for an exchange."""
    config = EXCHANGE_SCHEDULES.get(exchange_name)
    if not config:
        logger.error(f"Exchange {exchange_name} not found")
        return
    
    hour, minute = get_exchange_close_time(config, exchange_name=exchange_name)
    # Schedule logic...
```

---

## Testing Configuration Modules

### Unit Test Structure

```python
# tests/test_config/test_exchange_schedule_config.py
import pytest
from datetime import date
from src.config.exchange_schedule_config import (
    EXCHANGE_SCHEDULES,
    get_exchange_close_time,
    is_dst_active
)

class TestExchangeScheduleConfig:
    """Tests for exchange schedule configuration."""
    
    def test_exchange_schedules_not_empty(self):
        """Test that exchange schedules are loaded."""
        assert len(EXCHANGE_SCHEDULES) > 0
    
    def test_nasdaq_config_exists(self):
        """Test that NASDAQ configuration exists."""
        assert "NASDAQ" in EXCHANGE_SCHEDULES
        config = EXCHANGE_SCHEDULES["NASDAQ"]
        assert "est_close_hour" in config
        assert "edt_close_hour" in config
    
    def test_get_exchange_close_time(self):
        """Test getting exchange close time."""
        config = EXCHANGE_SCHEDULES["NASDAQ"]
        hour, minute = get_exchange_close_time(config)
        assert isinstance(hour, int)
        assert isinstance(minute, int)
        assert 0 <= hour < 24
        assert 0 <= minute < 60
    
    def test_is_dst_active_returns_bool(self):
        """Test that DST check returns boolean."""
        result = is_dst_active()
        assert isinstance(result, bool)
```

### Running Tests

```bash
# Run all config tests
pytest tests/test_config/ -v

# Run specific config module tests
pytest tests/test_config/test_exchange_schedule_config.py -v

# Run with coverage
pytest tests/test_config/ --cov=src.config --cov-report=html
```

---

## Common Patterns and Solutions

### Pattern 1: Environment-Based Configuration

```python
import os

ENVIRONMENT = os.getenv("ENVIRONMENT", "production")

if ENVIRONMENT == "development":
    BATCH_SIZE = 100
    MAX_WORKERS = 5
elif ENVIRONMENT == "production":
    BATCH_SIZE = 1000
    MAX_WORKERS = 10
else:
    BATCH_SIZE = 2000
    MAX_WORKERS = 20
```

### Pattern 2: Configuration Validation on Import

```python
def _validate_config():
    """Validate configuration on module import."""
    required_exchanges = ["NASDAQ", "NYSE", "LONDON"]
    for exchange in required_exchanges:
        if exchange not in EXCHANGE_SCHEDULES:
            raise ValueError(f"Required exchange '{exchange}' not configured")

# Validate on import
_validate_config()
```

### Pattern 3: Configuration Reloading

```python
def reload_config():
    """Reload configuration from source."""
    global CONFIGURATION
    # Reload logic (e.g., from file, database, etc.)
    CONFIGURATION = load_from_source()
    logger.info("Configuration reloaded")
```

### Pattern 4: Configuration Versioning

```python
CONFIG_VERSION = "2.0.0"

CONFIGURATION = {
    "version": CONFIG_VERSION,
    "settings": {
        # Actual settings
    }
}

def check_config_version() -> bool:
    """Check if configuration version is compatible."""
    return CONFIGURATION.get("version") == CONFIG_VERSION
```

---

## AI Assistant Guidelines

When working with configuration modules:

1. **Follow existing patterns**: Match the structure and style of `exchange_schedule_config.py` and `performance_config.py`

2. **Document timezone handling**: If dealing with timezones, clearly document:
   - What timezone values are in
   - How DST transitions are handled
   - Any special cases or edge cases

3. **Use type hints**: Always include type hints for function signatures and important variables

4. **Add validation**: Include validation functions for critical configurations

5. **Provide helper functions**: Create utility functions to access configuration values safely

6. **Consider performance**: For frequently accessed configurations, consider caching or optimization

7. **Test configurations**: When adding new config modules, include unit tests

8. **Update `__init__.py`**: Always export new config modules in `src/config/__init__.py`

9. **Document defaults**: Clearly document default values and their rationale

10. **Handle errors gracefully**: Use try/except blocks and provide meaningful error messages

---

## Quick Reference

### Exchange Schedule Config

```python
# Get exchange configuration
config = EXCHANGE_SCHEDULES["NASDAQ"]

# Get close time (auto-detects DST)
hour, minute = get_exchange_close_time(config, exchange_name="NASDAQ")

# Check DST status
is_edt = is_dst_active()

# Check DST transition period
in_transition = is_in_dst_transition_period()

# Get market days for exchange
days = get_market_days_for_exchange(config)
```

### Performance Config

```python
# Access constants
batch_size = BATCH_SIZE
max_workers = MAX_WORKERS

# Get profile settings
settings = get_performance_profile("production")

# Get optimal settings
optimal = get_optimal_settings(alert_count=5000)

# Update settings dynamically
update_performance_settings({"batch_size": 2000})
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-02-09 | Initial Claude.md creation for config directory |

---

*This document should be updated as new configuration modules are added or existing ones are modified. Keep it in sync with actual implementation patterns.*

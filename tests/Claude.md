# Claude.md - AI Assistant Instructions for Tests Directory

## Overview

This directory contains all tests for the Streamlit application. When working on tests in this project, follow the conventions, patterns, and best practices outlined in this document.

---

## Testing Stack

| Tool | Purpose | Version |
|------|---------|---------|
| `pytest` | Test framework | 7.x+ |
| `pytest-cov` | Coverage reporting | Latest |
| `pytest-mock` | Mocking utilities | Latest |
| `pytest-asyncio` | Async test support | Latest |
| `streamlit.testing` | Streamlit app testing | Built-in (Streamlit 1.28+) |
| `hypothesis` | Property-based testing | Optional |
| `freezegun` | Time mocking | Optional |
| `responses` | HTTP mocking | Optional |
| `factory_boy` | Test data factories | Optional |

---

## Directory Structure

```
tests/
├── __init__.py
├── conftest.py                 # Shared fixtures and configuration
├── Claude.md                   # This file
├── unit/                       # Unit tests (isolated, fast)
│   ├── __init__.py
│   ├── test_utils/
│   │   ├── __init__.py
│   │   ├── test_data_processing.py
│   │   ├── test_validators.py
│   │   └── test_formatters.py
│   ├── test_models/
│   │   ├── __init__.py
│   │   └── test_schemas.py
│   └── test_services/
│       ├── __init__.py
│       ├── test_database.py
│       └── test_api_client.py
├── integration/                # Integration tests (multiple components)
│   ├── __init__.py
│   ├── test_data_pipeline.py
│   └── test_auth_flow.py
├── e2e/                        # End-to-end tests (full app)
│   ├── __init__.py
│   ├── test_app_navigation.py
│   └── test_user_workflows.py
├── fixtures/                   # Test data and fixtures
│   ├── __init__.py
│   ├── sample_data.py
│   ├── mock_responses.py
│   └── data/
│       ├── sample_input.csv
│       ├── expected_output.json
│       └── test_config.yaml
└── helpers/                    # Test utilities
    ├── __init__.py
    ├── assertions.py
    ├── factories.py
    └── streamlit_helpers.py
```

---

## Test Naming Conventions

### File Naming
```
test_<module_name>.py           # Test file for a specific module
test_<feature>_<aspect>.py      # Test file for specific feature aspect
```

### Function Naming
```python
def test_<function_name>_<scenario>_<expected_result>():
    """Test description."""
    pass

# Examples:
def test_calculate_total_with_valid_input_returns_sum():
def test_calculate_total_with_empty_list_returns_zero():
def test_calculate_total_with_negative_values_raises_error():
def test_user_login_with_invalid_credentials_returns_false():
```

### Class Naming
```python
class TestClassName:
    """Tests for ClassName."""
    pass

class TestFeatureName:
    """Tests for feature behavior."""
    pass

# Examples:
class TestDataProcessor:
class TestUserAuthentication:
class TestDashboardCharts:
```

---

## Fixture Patterns

### conftest.py Structure
```python
# tests/conftest.py
"""Shared pytest fixtures and configuration."""

import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

# ============================================================================
# DATA FIXTURES
# ============================================================================

@pytest.fixture
def sample_dataframe():
    """Create a standard sample dataframe for testing."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "value": [100, 200, 150, 300, 250],
        "category": ["A", "B", "A", "C", "B"],
        "date": pd.date_range("2024-01-01", periods=5),
    })


@pytest.fixture
def empty_dataframe():
    """Create an empty dataframe with expected columns."""
    return pd.DataFrame(columns=["id", "name", "value", "category", "date"])


@pytest.fixture
def dataframe_with_nulls():
    """Create a dataframe containing null values."""
    return pd.DataFrame({
        "id": [1, 2, 3, None, 5],
        "name": ["Alice", None, "Charlie", "Diana", None],
        "value": [100, 200, None, 300, 250],
    })


# ============================================================================
# MOCK FIXTURES
# ============================================================================

@pytest.fixture
def mock_database_connection():
    """Mock database connection."""
    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = []
    mock_conn.execute.return_value.fetchone.return_value = None
    return mock_conn


@pytest.fixture
def mock_api_response():
    """Mock successful API response."""
    return {
        "status": "success",
        "data": {"id": 1, "result": "test"},
        "timestamp": datetime.now().isoformat(),
    }


@pytest.fixture
def mock_streamlit_session_state():
    """Mock Streamlit session state."""
    with patch("streamlit.session_state", {}) as mock_state:
        yield mock_state


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def test_config():
    """Test configuration settings."""
    return {
        "debug": True,
        "database_url": "sqlite:///:memory:",
        "api_timeout": 5,
        "max_retries": 3,
    }


@pytest.fixture
def temp_directory(tmp_path):
    """Create a temporary directory for test files."""
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()
    return test_dir


# ============================================================================
# STREAMLIT-SPECIFIC FIXTURES
# ============================================================================

@pytest.fixture
def mock_st():
    """Comprehensive Streamlit mock."""
    with patch.multiple(
        "streamlit",
        write=MagicMock(),
        error=MagicMock(),
        warning=MagicMock(),
        success=MagicMock(),
        info=MagicMock(),
        dataframe=MagicMock(),
        cache_data=lambda ttl=None, show_spinner=None: lambda f: f,
        cache_resource=lambda: lambda f: f,
    ) as mocks:
        yield mocks


# ============================================================================
# AUTHENTICATION FIXTURES
# ============================================================================

@pytest.fixture
def authenticated_user():
    """Return an authenticated user object."""
    return {
        "id": 1,
        "username": "testuser",
        "email": "test@example.com",
        "roles": ["user", "editor"],
        "authenticated": True,
    }


@pytest.fixture
def admin_user():
    """Return an admin user object."""
    return {
        "id": 0,
        "username": "admin",
        "email": "admin@example.com",
        "roles": ["user", "editor", "admin"],
        "authenticated": True,
    }
```

### Fixture Scopes
```python
# Function scope (default) - created for each test
@pytest.fixture
def per_test_fixture():
    return {"fresh": "data"}

# Class scope - shared across tests in a class
@pytest.fixture(scope="class")
def per_class_fixture():
    return expensive_setup()

# Module scope - shared across tests in a module
@pytest.fixture(scope="module")
def per_module_fixture():
    connection = create_connection()
    yield connection
    connection.close()

# Session scope - shared across entire test session
@pytest.fixture(scope="session")
def per_session_fixture():
    return load_large_test_data()
```

---

## Test Patterns

### Unit Test Pattern
```python
# tests/unit/test_utils/test_data_processing.py
"""Unit tests for data processing utilities."""

import pytest
import pandas as pd
from src.utils.data_processing import (
    clean_data,
    calculate_metrics,
    transform_columns,
    validate_schema,
)


class TestCleanData:
    """Tests for clean_data function."""

    def test_removes_null_rows(self, dataframe_with_nulls):
        """Should remove rows containing null values."""
        result = clean_data(dataframe_with_nulls, drop_nulls=True)
        
        assert len(result) == 2  # Only rows 1 and 4 have no nulls
        assert result["id"].isna().sum() == 0
        assert result["name"].isna().sum() == 0

    def test_fills_null_with_default(self, dataframe_with_nulls):
        """Should fill null values with specified default."""
        result = clean_data(
            dataframe_with_nulls, 
            drop_nulls=False, 
            fill_value=0
        )
        
        assert len(result) == 5
        assert result["value"].isna().sum() == 0

    def test_preserves_data_types(self, sample_dataframe):
        """Should preserve original column data types."""
        result = clean_data(sample_dataframe)
        
        assert result["id"].dtype == sample_dataframe["id"].dtype
        assert result["value"].dtype == sample_dataframe["value"].dtype

    def test_empty_dataframe_returns_empty(self, empty_dataframe):
        """Should handle empty dataframe gracefully."""
        result = clean_data(empty_dataframe)
        
        assert len(result) == 0
        assert list(result.columns) == list(empty_dataframe.columns)

    def test_invalid_input_raises_type_error(self):
        """Should raise TypeError for non-DataFrame input."""
        with pytest.raises(TypeError, match="Expected DataFrame"):
            clean_data([1, 2, 3])


class TestCalculateMetrics:
    """Tests for calculate_metrics function."""

    @pytest.mark.parametrize("data,expected_mean", [
        ([1, 2, 3, 4, 5], 3.0),
        ([10, 20, 30], 20.0),
        ([0, 0, 0], 0.0),
        ([-5, 5], 0.0),
    ])
    def test_calculates_mean_correctly(self, data, expected_mean):
        """Should calculate mean correctly for various inputs."""
        result = calculate_metrics(data)
        
        assert result["mean"] == pytest.approx(expected_mean)

    def test_returns_all_expected_keys(self):
        """Should return dictionary with all metric keys."""
        result = calculate_metrics([1, 2, 3])
        
        expected_keys = {"mean", "median", "std", "min", "max", "count"}
        assert set(result.keys()) == expected_keys

    def test_empty_list_raises_value_error(self):
        """Should raise ValueError for empty input."""
        with pytest.raises(ValueError, match="Cannot calculate metrics"):
            calculate_metrics([])

    def test_single_value_handles_std(self):
        """Should handle standard deviation for single value."""
        result = calculate_metrics([42])
        
        assert result["mean"] == 42
        assert result["std"] == 0.0
```

### Integration Test Pattern
```python
# tests/integration/test_data_pipeline.py
"""Integration tests for data pipeline."""

import pytest
import pandas as pd
from src.services.database import DatabaseService
from src.utils.data_processing import clean_data, transform_columns
from src.services.api_client import APIClient


class TestDataPipeline:
    """Tests for complete data pipeline flow."""

    @pytest.fixture
    def pipeline_components(self, mock_database_connection, test_config):
        """Set up pipeline components for testing."""
        db_service = DatabaseService(mock_database_connection)
        api_client = APIClient(test_config)
        return {
            "db": db_service,
            "api": api_client,
        }

    def test_full_etl_pipeline(self, pipeline_components, sample_dataframe):
        """Should execute complete ETL pipeline successfully."""
        # Extract
        raw_data = pipeline_components["db"].fetch_data("SELECT * FROM source")
        
        # Transform
        cleaned = clean_data(raw_data)
        transformed = transform_columns(cleaned, mappings={"old": "new"})
        
        # Load
        result = pipeline_components["db"].insert_data(transformed, "target")
        
        assert result["rows_affected"] > 0
        assert result["status"] == "success"

    def test_pipeline_handles_api_failure_gracefully(
        self, pipeline_components, mock_api_response
    ):
        """Should handle API failures without crashing."""
        with pytest.raises(APIError) as exc_info:
            pipeline_components["api"].fetch_with_retry(
                "/endpoint",
                max_retries=0  # Force immediate failure
            )
        
        assert "Max retries exceeded" in str(exc_info.value)

    def test_data_consistency_across_transformations(self, sample_dataframe):
        """Should maintain data consistency through pipeline."""
        original_count = len(sample_dataframe)
        original_columns = set(sample_dataframe.columns)
        
        # Run through multiple transformations
        step1 = clean_data(sample_dataframe)
        step2 = transform_columns(step1, mappings={})
        final = step2.copy()
        
        # Verify no data loss
        assert len(final) == original_count
        assert set(final.columns) == original_columns
```

### Streamlit App Test Pattern
```python
# tests/e2e/test_app_navigation.py
"""End-to-end tests for Streamlit app using AppTest."""

import pytest
from streamlit.testing.v1 import AppTest


class TestAppNavigation:
    """Tests for app navigation and basic functionality."""

    @pytest.fixture
    def app(self):
        """Create AppTest instance."""
        return AppTest.from_file("app.py", default_timeout=10)

    def test_app_loads_successfully(self, app):
        """App should load without errors."""
        app.run()
        
        assert not app.exception
        assert len(app.error) == 0

    def test_title_displayed(self, app):
        """App should display correct title."""
        app.run()
        
        assert "My Streamlit App" in app.title[0].value

    def test_sidebar_renders(self, app):
        """Sidebar should render with navigation options."""
        app.run()
        
        assert len(app.sidebar) > 0

    def test_selectbox_interaction(self, app):
        """Selectbox should update app state."""
        app.run()
        
        # Find and interact with selectbox
        selectbox = app.selectbox[0]
        selectbox.select("Option B")
        app.run()
        
        assert app.session_state["selected_option"] == "Option B"

    def test_button_click_triggers_action(self, app):
        """Button click should trigger expected action."""
        app.run()
        
        # Click submit button
        app.button[0].click()
        app.run()
        
        assert app.session_state.get("form_submitted") is True

    def test_form_submission(self, app):
        """Form should process input correctly."""
        app.run()
        
        # Fill form fields
        app.text_input[0].input("Test User")
        app.text_input[1].input("test@example.com")
        
        # Submit form
        app.button("Submit").click()
        app.run()
        
        assert "success" in app.success[0].value.lower()

    def test_file_upload_handling(self, app, tmp_path):
        """File upload should be processed correctly."""
        # Create test file
        test_file = tmp_path / "test.csv"
        test_file.write_text("col1,col2\n1,2\n3,4")
        
        app.run()
        
        # Simulate file upload
        with open(test_file, "rb") as f:
            app.file_uploader[0].upload(f, "test.csv")
        app.run()
        
        assert app.session_state.get("file_uploaded") is True


class TestDashboardPage:
    """Tests for dashboard page functionality."""

    @pytest.fixture
    def dashboard_app(self):
        """Create AppTest for dashboard page."""
        return AppTest.from_file("pages/1_Dashboard.py", default_timeout=10)

    def test_charts_render(self, dashboard_app):
        """Dashboard charts should render without errors."""
        dashboard_app.run()
        
        assert not dashboard_app.exception
        # Check for plotly or other chart elements

    def test_filters_affect_display(self, dashboard_app):
        """Applying filters should update displayed data."""
        dashboard_app.run()
        
        # Apply date filter
        dashboard_app.date_input[0].set_value("2024-01-01")
        dashboard_app.run()
        
        # Verify filtered state
        assert dashboard_app.session_state["filter_applied"] is True
```

---

## Mocking Patterns

### Mocking External APIs
```python
import responses
import requests

@responses.activate
def test_api_call_success():
    """Test successful API call."""
    responses.add(
        responses.GET,
        "https://api.example.com/data",
        json={"status": "ok", "data": [1, 2, 3]},
        status=200,
    )
    
    result = fetch_data_from_api()
    
    assert result["status"] == "ok"
    assert len(responses.calls) == 1


@responses.activate
def test_api_call_handles_error():
    """Test API error handling."""
    responses.add(
        responses.GET,
        "https://api.example.com/data",
        json={"error": "Not found"},
        status=404,
    )
    
    with pytest.raises(APIError):
        fetch_data_from_api()
```

### Mocking Database
```python
from unittest.mock import MagicMock, patch

def test_database_query(mock_database_connection):
    """Test database query execution."""
    # Setup mock return
    mock_database_connection.execute.return_value.fetchall.return_value = [
        (1, "Alice", 100),
        (2, "Bob", 200),
    ]
    
    with patch("src.services.database.get_connection", return_value=mock_database_connection):
        result = query_users()
    
    assert len(result) == 2
    mock_database_connection.execute.assert_called_once()
```

### Mocking Streamlit Components
```python
from unittest.mock import patch, MagicMock

def test_component_with_mocked_streamlit():
    """Test component with mocked Streamlit."""
    mock_session_state = {"user": "test", "authenticated": True}
    
    with patch("streamlit.session_state", mock_session_state):
        with patch("streamlit.write") as mock_write:
            render_user_greeting()
            
            mock_write.assert_called_with("Welcome, test!")
```

### Mocking Time
```python
from freezegun import freeze_time
from datetime import datetime

@freeze_time("2024-06-15 12:00:00")
def test_time_dependent_function():
    """Test function that depends on current time."""
    result = get_greeting()
    
    assert result == "Good afternoon"  # Based on frozen time

@freeze_time("2024-01-01")
def test_new_year_banner():
    """Test New Year banner display."""
    result = should_show_new_year_banner()
    
    assert result is True
```

---

## Assertion Helpers

### Custom Assertions (tests/helpers/assertions.py)
```python
"""Custom assertion helpers for tests."""

import pandas as pd
from typing import Any


def assert_dataframe_equal(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    check_dtype: bool = True,
    check_order: bool = True,
) -> None:
    """Assert two DataFrames are equal with detailed error messages."""
    # Check shape
    assert actual.shape == expected.shape, (
        f"Shape mismatch: {actual.shape} != {expected.shape}"
    )
    
    # Check columns
    if check_order:
        assert list(actual.columns) == list(expected.columns), (
            f"Column mismatch: {list(actual.columns)} != {list(expected.columns)}"
        )
    else:
        assert set(actual.columns) == set(expected.columns), (
            f"Column mismatch: {set(actual.columns)} != {set(expected.columns)}"
        )
    
    # Check dtypes
    if check_dtype:
        for col in actual.columns:
            assert actual[col].dtype == expected[col].dtype, (
                f"Dtype mismatch for '{col}': {actual[col].dtype} != {expected[col].dtype}"
            )
    
    # Check values
    pd.testing.assert_frame_equal(actual, expected)


def assert_dict_contains(actual: dict, expected_subset: dict) -> None:
    """Assert dictionary contains expected key-value pairs."""
    for key, value in expected_subset.items():
        assert key in actual, f"Missing key: {key}"
        assert actual[key] == value, (
            f"Value mismatch for '{key}': {actual[key]} != {value}"
        )


def assert_valid_response(response: dict, required_fields: list[str]) -> None:
    """Assert API response has required structure."""
    for field in required_fields:
        assert field in response, f"Missing required field: {field}"


def assert_called_with_dataframe(mock_call, expected_df: pd.DataFrame) -> None:
    """Assert mock was called with specific DataFrame."""
    actual_df = mock_call.call_args[0][0]
    assert_dataframe_equal(actual_df, expected_df)
```

---

## Test Data Factories

### Factory Pattern (tests/helpers/factories.py)
```python
"""Test data factories."""

import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import random
import string


class DataFrameFactory:
    """Factory for creating test DataFrames."""

    @staticmethod
    def create_users(
        count: int = 5,
        include_nulls: bool = False,
    ) -> pd.DataFrame:
        """Create a users DataFrame."""
        data = {
            "id": range(1, count + 1),
            "name": [f"User_{i}" for i in range(1, count + 1)],
            "email": [f"user{i}@example.com" for i in range(1, count + 1)],
            "created_at": [
                datetime.now() - timedelta(days=i) for i in range(count)
            ],
        }
        
        df = pd.DataFrame(data)
        
        if include_nulls and count > 2:
            df.loc[1, "name"] = None
            df.loc[2, "email"] = None
        
        return df

    @staticmethod
    def create_transactions(
        count: int = 10,
        user_ids: Optional[list[int]] = None,
    ) -> pd.DataFrame:
        """Create a transactions DataFrame."""
        if user_ids is None:
            user_ids = list(range(1, 6))
        
        return pd.DataFrame({
            "id": range(1, count + 1),
            "user_id": [random.choice(user_ids) for _ in range(count)],
            "amount": [round(random.uniform(10, 1000), 2) for _ in range(count)],
            "category": [
                random.choice(["food", "transport", "entertainment", "utilities"])
                for _ in range(count)
            ],
            "date": [
                datetime.now() - timedelta(days=random.randint(0, 30))
                for _ in range(count)
            ],
        })

    @staticmethod
    def create_time_series(
        start_date: str = "2024-01-01",
        periods: int = 30,
        columns: list[str] = None,
    ) -> pd.DataFrame:
        """Create a time series DataFrame."""
        if columns is None:
            columns = ["value"]
        
        dates = pd.date_range(start=start_date, periods=periods, freq="D")
        data = {"date": dates}
        
        for col in columns:
            data[col] = [random.uniform(0, 100) for _ in range(periods)]
        
        return pd.DataFrame(data)


class UserFactory:
    """Factory for creating test user objects."""

    _counter = 0

    @classmethod
    def create(
        cls,
        username: Optional[str] = None,
        email: Optional[str] = None,
        role: str = "user",
        authenticated: bool = True,
    ) -> dict:
        """Create a user dictionary."""
        cls._counter += 1
        return {
            "id": cls._counter,
            "username": username or f"user_{cls._counter}",
            "email": email or f"user{cls._counter}@example.com",
            "role": role,
            "authenticated": authenticated,
            "created_at": datetime.now().isoformat(),
        }

    @classmethod
    def create_admin(cls) -> dict:
        """Create an admin user."""
        return cls.create(role="admin", username="admin")

    @classmethod
    def reset(cls) -> None:
        """Reset the counter."""
        cls._counter = 0
```

---

## Running Tests

### Command Reference
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_utils/test_data_processing.py

# Run specific test class
pytest tests/unit/test_utils/test_data_processing.py::TestCleanData

# Run specific test function
pytest tests/unit/test_utils/test_data_processing.py::TestCleanData::test_removes_null_rows

# Run tests matching pattern
pytest -k "test_calculate"

# Run tests with specific marker
pytest -m "slow"
pytest -m "not slow"

# Run with coverage
pytest --cov=src --cov-report=html --cov-report=term-missing

# Run in parallel (requires pytest-xdist)
pytest -n auto

# Run with debug output
pytest -s  # Show print statements
pytest --tb=long  # Longer traceback

# Stop on first failure
pytest -x

# Run last failed tests
pytest --lf

# Generate JUnit XML report (for CI)
pytest --junitxml=reports/junit.xml
```

### pytest.ini Configuration
```ini
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --strict-markers
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    e2e: marks tests as end-to-end tests
    requires_db: marks tests that require database
    requires_api: marks tests that require external API
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
```

### pyproject.toml Configuration
```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
markers = [
    "slow: marks tests as slow",
    "integration: integration tests",
    "e2e: end-to-end tests",
]

[tool.coverage.run]
source = ["src"]
branch = true
omit = [
    "*/tests/*",
    "*/__init__.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
]
fail_under = 80
show_missing = true
```

---

## Test Markers

### Using Markers
```python
import pytest

@pytest.mark.slow
def test_large_data_processing():
    """This test takes a long time."""
    pass

@pytest.mark.integration
def test_database_connection():
    """This test requires database."""
    pass

@pytest.mark.e2e
def test_full_user_workflow():
    """End-to-end test."""
    pass

@pytest.mark.skip(reason="Feature not implemented yet")
def test_future_feature():
    pass

@pytest.mark.skipif(
    condition=os.environ.get("CI") == "true",
    reason="Skip in CI environment"
)
def test_local_only():
    pass

@pytest.mark.xfail(reason="Known bug #123")
def test_known_failing():
    pass

@pytest.mark.parametrize("input,expected", [
    (1, 2),
    (2, 4),
    (3, 6),
])
def test_doubling(input, expected):
    assert input * 2 == expected
```

---

## Coverage Requirements

### Minimum Coverage Targets
| Category | Target |
|----------|--------|
| Overall | 80% |
| Utils | 90% |
| Models | 85% |
| Services | 80% |
| Components | 75% |

### Excluding from Coverage
```python
# Exclude specific lines
if __name__ == "__main__":  # pragma: no cover
    main()

# Exclude entire function
def debug_only_function():  # pragma: no cover
    """This is only for debugging."""
    pass
```

---

## CI/CD Integration

### GitHub Actions Example
```yaml
# .github/workflows/tests.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      
      - name: Run tests
        run: pytest --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          files: coverage.xml
```

---

## AI Assistant Guidelines for Tests

When assisting with tests in this project:

1. **Follow existing patterns**: Match the test style, naming conventions, and structure already established in the codebase.

2. **Write focused tests**: Each test should verify one specific behavior. Use descriptive names that explain what's being tested.

3. **Use fixtures appropriately**: Leverage existing fixtures from conftest.py; create new fixtures for reusable test setup.

4. **Mock external dependencies**: Always mock databases, APIs, file systems, and Streamlit components in unit tests.

5. **Consider edge cases**: Include tests for empty inputs, null values, boundary conditions, and error scenarios.

6. **Keep tests fast**: Unit tests should execute in milliseconds. Mark slow tests with `@pytest.mark.slow`.

7. **Maintain test isolation**: Tests should not depend on each other or share mutable state.

8. **Use parametrize for variations**: When testing the same logic with different inputs, use `@pytest.mark.parametrize`.

9. **Assert with clarity**: Use specific assertions with helpful error messages. Prefer multiple specific assertions over one generic assertion.

10. **Document complex tests**: Add docstrings explaining what's being tested and why, especially for integration tests.

---

## Quick Reference

### Common pytest Commands
```bash
pytest                          # Run all tests
pytest -v                       # Verbose output
pytest -x                       # Stop on first failure
pytest --lf                     # Run last failed
pytest -k "pattern"             # Match test names
pytest -m "marker"              # Run marked tests
pytest --cov=src               # With coverage
pytest -n auto                  # Parallel execution
```

### Common Assertions
```python
assert x == y                   # Equality
assert x != y                   # Inequality
assert x is None                # None check
assert x is not None            # Not None
assert x in collection          # Membership
assert isinstance(x, Type)      # Type check
pytest.approx(x, rel=1e-3)     # Float comparison
pytest.raises(Exception)        # Exception expected
```

### Fixture Scopes
```python
@pytest.fixture                 # Function (default)
@pytest.fixture(scope="class")  # Class
@pytest.fixture(scope="module") # Module
@pytest.fixture(scope="session")# Session
```

---

*Keep this document updated as testing patterns evolve in the project.*

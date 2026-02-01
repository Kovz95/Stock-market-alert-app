# Claude.md - AI Assistant Instructions for Streamlit Application

## Project Overview

This is a Streamlit web application. When working on this project, follow the conventions, patterns, and best practices outlined in this document to ensure consistent, high-quality code and helpful assistance.

---

## Technology Stack

### Core Framework
- **Streamlit**: Primary web framework for building the application UI
- **Python**: 3.13+ (ensure compatibility with type hints and modern Python features)
- **UV** Package Manager: For managing dependencies and ensuring compatibility with modern Python features

### Common Dependencies
- `streamlit` - Web application framework
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `plotly` / `altair` / `matplotlib` - Data visualization
- `sqlalchemy` - Database ORM (if applicable)
- `requests` - HTTP client for API calls
- `python-dotenv` - Environment variable management

---

## Project Structure

```
project_root/
├── .streamlit/
│   ├── config.toml          # Streamlit configuration
│   └── secrets.toml         # Secrets (API keys, DB credentials) - DO NOT COMMIT
├── src/
│   ├── __init__.py
│   ├── components/          # Reusable UI components
│   │   ├── __init__.py
│   │   ├── sidebar.py
│   │   ├── charts.py
│   │   └── forms.py
│   ├── utils/               # Utility functions
│   │   ├── __init__.py
│   │   ├── data_processing.py
│   │   ├── api_client.py
│   │   └── validators.py
│   ├── models/              # Data models and schemas
│   │   ├── __init__.py
│   │   └── schemas.py
│   ├── services/            # Business logic and external services
│   │   ├── __init__.py
│   │   ├── auth.py
│   └── data_access/         # Data access layer (database queries, ORM)
│       ├── __init__.py
│       └── database.py
├── pages/                   # Multi-page app pages (Streamlit convention)
│   ├── 1_Dashboard.py
│   ├── 2_Analytics.py
│   └── 3_Settings.py
├── scripts/                 # Utility and operational scripts
│   ├── analysis/            # Analysis and monitoring scripts
│   │   ├── analyze_alerts.py
│   │   └── check_scheduler_status.py
│   ├── maintenance/         # Database and system maintenance scripts
│   │   ├── fix_database_constraints.py
│   │   └── cleanup_old_data.py
│   └── migration/           # Data migration and schema update scripts
│       └── migrate_legacy_data.py
├── data/                    # Static data files
├── tests/                   # Test files
│   ├── __init__.py
│   ├── test_utils.py
│   └── test_components.py
├── pyproject.toml          # Project metadata and tool configuration
├── .env.example            # Example environment variables
├── .gitignore
├── .venv                   # Virtual environment (DO NOT COMMIT)
├── uv.lock                 # Virtual environment lock file
├── README.md
└── Claude.md               # This file
```

---

## Scripts Directory Organization

The `scripts/` directory contains operational and utility scripts that are not part of the core application but support development, operations, and maintenance tasks.

### Directory Structure

```
scripts/
├── analysis/          # Analysis, monitoring, and diagnostic scripts
├── maintenance/       # Database and system maintenance scripts
└── migration/         # Data migration and schema update scripts
```

### Script Categories

#### 1. Analysis Scripts (`scripts/analysis/`)
Scripts for analyzing system behavior, monitoring, and diagnostics.

**Examples:**
- Alert system performance analysis
- Scheduler status checking
- Log analysis and reporting
- Data quality audits
- System health checks

**Characteristics:**
- Read-only operations (no data modification)
- Generate reports and insights
- CLI-based with argument parsing
- Can be run on production data safely

#### 2. Maintenance Scripts (`scripts/maintenance/`)
Scripts for routine maintenance, cleanup, and system upkeep.

**Examples:**
- Database constraint fixes
- Data cleanup and archival
- Index rebuilding
- Cache clearing
- Backup verification

**Characteristics:**
- May modify data or system state
- Should have confirmation prompts for destructive operations
- Often scheduled or run periodically
- Should include logging and error handling

#### 3. Migration Scripts (`scripts/migration/`)
Scripts for one-time or periodic data migrations and schema updates.

**Examples:**
- Legacy data format conversions
- Schema migrations not handled by ORM
- Bulk data transformations
- Historical data imports
- Database version upgrades

**Characteristics:**
- Typically run once or infrequently
- Should be idempotent when possible
- Include rollback capabilities
- Extensive logging and validation

### Script Development Guidelines

When creating new scripts:

1. **Choose the Right Location**:
   - Analysis scripts → `scripts/analysis/`
   - Maintenance scripts → `scripts/maintenance/`
   - Migration scripts → `scripts/migration/`

2. **Script Template**:
   ```python
   #!/usr/bin/env python3
   """
   Brief description of what this script does.

   Usage:
       python scripts/category/script_name.py [options]

   Examples:
       python scripts/analysis/analyze_alerts.py --days 7
   """

   import sys
   import os
   from pathlib import Path
   from datetime import datetime
   import argparse

   # Set UTF-8 encoding for Windows console
   if sys.platform == 'win32':
       import codecs
       sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
       sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

   # Add project root to path
   BASE_DIR = Path(__file__).resolve().parent.parent
   sys.path.append(str(BASE_DIR))


   def main():
       """Main script logic."""
       parser = argparse.ArgumentParser(description="Script description")
       parser.add_argument('--option', type=str, help='Option description')
       args = parser.parse_args()

       # Script logic here
       print(f"Script executed at {datetime.now()}")


   if __name__ == "__main__":
       main()
   ```

3. **Best Practices**:
   - Use shebang (`#!/usr/bin/env python3`) for direct execution
   - Include comprehensive docstrings
   - Use argparse for CLI arguments
   - Handle errors gracefully with try/except blocks
   - Include progress indicators for long-running operations
   - Log important actions and errors
   - Use type hints for function signatures
   - Make scripts executable: `chmod +x scripts/category/script_name.py`

4. **Documentation Requirements**:
   - Clear description at the top of the file
   - Usage examples in docstring
   - Inline comments for complex logic
   - Document any prerequisites or dependencies
   - Note any required environment variables or configuration

5. **Safety Considerations**:
   - For destructive operations, require explicit confirmation
   - Include dry-run mode for testing (`--dry-run`)
   - Validate inputs before processing
   - Backup data before modifications
   - Include rollback procedures for migrations

### Common Script Patterns

#### Progress Indicators
```python
from tqdm import tqdm

def process_items(items: list):
    """Process items with progress bar."""
    for item in tqdm(items, desc="Processing"):
        process_single_item(item)
```

#### Confirmation Prompts
```python
def confirm_action(message: str) -> bool:
    """Prompt user for confirmation."""
    response = input(f"{message} (yes/no): ").lower().strip()
    return response in ['yes', 'y']

# Usage
if not confirm_action("This will delete old records. Continue?"):
    print("Operation cancelled.")
    sys.exit(0)
```

#### Dry Run Mode
```python
def delete_records(dry_run: bool = False):
    """Delete records with dry-run support."""
    if dry_run:
        print("[DRY RUN] Would delete 100 records")
    else:
        print("Deleting 100 records...")
        # Actual deletion logic
```

---

## Coding Standards and Conventions

### Python Style Guide

1. **Follow PEP 8** with these specific preferences:
   - Line length: 88 characters (Black formatter default)
   - Use 4 spaces for indentation
   - Use double quotes for strings (consistent with Black)

2. **Type Hints**: Always use type hints for function signatures
   ```python
   def process_data(df: pd.DataFrame, column: str) -> pd.DataFrame:
       """Process the dataframe and return filtered results."""
       return df[df[column].notna()]
   ```

3. **Docstrings**: Use Google-style docstrings for all functions and classes
   ```python
   def calculate_metrics(data: list[float], threshold: float = 0.5) -> dict[str, float]:
       """Calculate statistical metrics from the provided data.

       Args:
           data: A list of numerical values to analyze.
           threshold: Minimum value threshold for filtering. Defaults to 0.5.

       Returns:
           A dictionary containing 'mean', 'median', and 'std' keys.

       Raises:
           ValueError: If data is empty or contains non-numeric values.
       """
       pass
   ```

4. **Import Organization**:
   ```python
   # Standard library imports
   import os
   from datetime import datetime
   from typing import Optional, Union

   # Third-party imports
   import pandas as pd
   import streamlit as st

   # Local imports
   from src.utils.data_processing import clean_data
   from src.components.charts import render_chart
   ```

### Streamlit-Specific Conventions

1. **Page Configuration**: Always set at the top of the main file
   ```python
   st.set_page_config(
       page_title="App Title",
       page_icon="🚀",
       layout="wide",
       initial_sidebar_state="expanded",
       menu_items={
           "Get Help": "https://docs.example.com",
           "Report a bug": "https://github.com/user/repo/issues",
           "About": "# App Description\nVersion 1.0.0"
       }
   )
   ```

2. **Session State Management**: Use structured patterns
   ```python
   # Initialize session state with defaults
   def init_session_state():
       """Initialize all session state variables with defaults."""
       defaults = {
           "user_data": None,
           "current_page": "home",
           "filters": {},
           "is_authenticated": False,
       }
       for key, value in defaults.items():
           if key not in st.session_state:
               st.session_state[key] = value

   # Call at app start
   init_session_state()
   ```

3. **Caching Strategy**:
   ```python
   # For data that doesn't change often
   @st.cache_data(ttl=3600)  # Cache for 1 hour
   def load_data(file_path: str) -> pd.DataFrame:
       """Load and cache data from file."""
       return pd.read_csv(file_path)

   # For resources like database connections
   @st.cache_resource
   def get_database_connection():
       """Create and cache database connection."""
       return create_engine(st.secrets["database"]["url"])
   ```

4. **Component Organization**: Create reusable components
   ```python
   # src/components/sidebar.py
   def render_sidebar() -> dict:
       """Render sidebar and return selected filter values."""
       with st.sidebar:
           st.header("Filters")
           date_range = st.date_input("Date Range", value=[])
           category = st.selectbox("Category", options=["All", "A", "B", "C"])
           
           return {
               "date_range": date_range,
               "category": category
           }
   ```

---

## State Management Patterns

### Callback Pattern for Forms
```python
def on_form_submit():
    """Handle form submission."""
    st.session_state.form_submitted = True
    st.session_state.form_data = {
        "name": st.session_state.input_name,
        "email": st.session_state.input_email,
    }

with st.form("user_form"):
    st.text_input("Name", key="input_name")
    st.text_input("Email", key="input_email")
    st.form_submit_button("Submit", on_click=on_form_submit)
```

### Navigation State Pattern
```python
def navigate_to(page: str):
    """Navigate to a different page."""
    st.session_state.current_page = page
    st.rerun()

# Usage in sidebar
if st.sidebar.button("Go to Dashboard"):
    navigate_to("dashboard")
```

---

## Error Handling

### Standard Error Handling Pattern
```python
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@contextmanager
def handle_errors(operation_name: str):
    """Context manager for consistent error handling."""
    try:
        yield
    except FileNotFoundError as e:
        logger.error(f"{operation_name} failed: File not found - {e}")
        st.error(f"❌ File not found. Please check the file path.")
    except pd.errors.EmptyDataError:
        logger.error(f"{operation_name} failed: Empty data")
        st.warning("⚠️ The data source is empty.")
    except Exception as e:
        logger.exception(f"{operation_name} failed with unexpected error")
        st.error(f"❌ An unexpected error occurred: {str(e)}")

# Usage
with handle_errors("Data loading"):
    data = load_data("data.csv")
    st.dataframe(data)
```

### User-Friendly Error Messages
```python
def display_error(error_type: str, details: str = ""):
    """Display a user-friendly error message."""
    error_messages = {
        "auth": "🔐 Authentication failed. Please check your credentials.",
        "network": "🌐 Network error. Please check your connection.",
        "data": "📊 Data processing error. Please verify your input.",
        "permission": "🚫 You don't have permission to perform this action.",
    }
    message = error_messages.get(error_type, "❌ An error occurred.")
    if details:
        message += f"\n\nDetails: {details}"
    st.error(message)
```

---

## Performance Optimization

### Data Loading Best Practices
```python
@st.cache_data(ttl=1800, show_spinner="Loading data...")
def load_large_dataset(query: str) -> pd.DataFrame:
    """Load large dataset with progress indication."""
    # Use chunked reading for very large files
    chunks = []
    for chunk in pd.read_csv("large_file.csv", chunksize=10000):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)
```

### Lazy Loading Pattern
```python
def render_dashboard():
    """Render dashboard with lazy-loaded components."""
    tab1, tab2, tab3 = st.tabs(["Overview", "Details", "Export"])
    
    with tab1:
        # Load only what's needed for this tab
        summary_data = load_summary_data()
        render_summary_charts(summary_data)
    
    with tab2:
        # Data loads only when tab is selected
        if st.session_state.get("details_loaded") is None:
            with st.spinner("Loading details..."):
                st.session_state.details_loaded = load_detailed_data()
        render_details(st.session_state.details_loaded)
```

---

## Security Guidelines

### Secrets Management
```python
# Access secrets securely
api_key = st.secrets["api"]["key"]
db_url = st.secrets["database"]["url"]

# Never do this:
# api_key = "hardcoded_key_123"  # ❌ NEVER hardcode secrets
```

### Input Validation
```python
import re
from typing import Optional

def validate_email(email: str) -> Optional[str]:
    """Validate email format and return cleaned email or None."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    email = email.strip().lower()
    if re.match(pattern, email):
        return email
    return None

def sanitize_input(text: str, max_length: int = 1000) -> str:
    """Sanitize user input text."""
    # Remove potentially dangerous characters
    text = text.strip()
    text = re.sub(r'[<>]', '', text)
    return text[:max_length]
```

### SQL Injection Prevention
```python
from sqlalchemy import text

# Always use parameterized queries
def get_user_data(user_id: int) -> pd.DataFrame:
    """Fetch user data safely."""
    query = text("SELECT * FROM users WHERE id = :user_id")
    return pd.read_sql(query, engine, params={"user_id": user_id})
```

---

## Testing Guidelines

### Unit Test Structure
```python
# tests/test_utils.py
import pytest
import pandas as pd
from src.utils.data_processing import clean_data, calculate_metrics

class TestDataProcessing:
    """Tests for data processing utilities."""

    @pytest.fixture
    def sample_dataframe(self):
        """Create sample dataframe for testing."""
        return pd.DataFrame({
            "name": ["Alice", "Bob", None, "Charlie"],
            "value": [10, 20, 30, 40]
        })

    def test_clean_data_removes_nulls(self, sample_dataframe):
        """Test that clean_data removes null values."""
        result = clean_data(sample_dataframe, "name")
        assert len(result) == 3
        assert result["name"].isna().sum() == 0

    def test_calculate_metrics_returns_expected_keys(self):
        """Test that calculate_metrics returns all expected keys."""
        data = [1, 2, 3, 4, 5]
        result = calculate_metrics(data)
        assert "mean" in result
        assert "median" in result
        assert "std" in result
```

### Running Tests
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_utils.py -v
```

---

## Deployment Checklist

### Pre-Deployment
- [ ] All tests passing
- [ ] No hardcoded secrets or credentials
- [ ] `requirements.txt` is up to date
- [ ] `.gitignore` includes sensitive files
- [ ] Error handling covers edge cases
- [ ] Performance tested with expected data volumes
- [ ] Mobile responsiveness verified (if applicable)

### Streamlit Cloud Deployment
```toml
# .streamlit/config.toml
[server]
headless = true
port = 8501
enableCORS = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Environment Variables
```bash
# .env.example (commit this)
DATABASE_URL=postgresql://user:password@localhost:5432/dbname
API_KEY=your_api_key_here
DEBUG=false

# .env (DO NOT commit)
DATABASE_URL=postgresql://actual_user:actual_password@production:5432/proddb
API_KEY=actual_api_key_12345
DEBUG=false
```

---

## Common Patterns and Solutions

### Multi-Page Navigation
```python
# app.py - Main entry point
import streamlit as st

# Define pages
pages = {
    "Home": "pages/home.py",
    "Dashboard": "pages/dashboard.py",
    "Settings": "pages/settings.py",
}

# Sidebar navigation
selected_page = st.sidebar.radio("Navigation", list(pages.keys()))

# This is handled automatically by Streamlit's pages/ folder convention
```

### Data Upload and Processing
```python
def handle_file_upload() -> Optional[pd.DataFrame]:
    """Handle file upload with validation."""
    uploaded_file = st.file_uploader(
        "Upload your data",
        type=["csv", "xlsx", "json"],
        help="Supported formats: CSV, Excel, JSON"
    )
    
    if uploaded_file is None:
        return None
    
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".json"):
            df = pd.read_json(uploaded_file)
        
        st.success(f"✅ Loaded {len(df)} rows from {uploaded_file.name}")
        return df
        
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")
        return None
```

### Authentication Pattern
```python
def check_authentication() -> bool:
    """Check if user is authenticated."""
    if st.session_state.get("authenticated"):
        return True
    
    with st.form("login_form"):
        st.subheader("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if authenticate_user(username, password):
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid credentials")
    
    return False

# Usage in main app
if not check_authentication():
    st.stop()

# Rest of the app (only runs if authenticated)
st.write(f"Welcome, {st.session_state.username}!")
```

### Progress Indicators
```python
def process_with_progress(items: list) -> list:
    """Process items with progress bar."""
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, item in enumerate(items):
        status_text.text(f"Processing item {i + 1} of {len(items)}...")
        result = process_item(item)
        results.append(result)
        progress_bar.progress((i + 1) / len(items))
    
    status_text.text("✅ Processing complete!")
    return results
```

---

## Debugging Tips

### Enable Debug Mode
```python
import streamlit as st
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Debug session state
if st.secrets.get("debug", False):
    with st.expander("🔧 Debug: Session State"):
        st.json(dict(st.session_state))
```

### Common Issues and Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Widget state reset | Missing unique keys | Add `key` parameter to all widgets |
| Slow performance | Uncached data operations | Use `@st.cache_data` or `@st.cache_resource` |
| Memory issues | Large dataframes in session state | Store file paths instead, load on demand |
| Duplicate content | Missing `st.rerun()` | Add rerun after state changes |

---

## AI Assistant Guidelines

When assisting with this Streamlit project:

1. **Always consider Streamlit's execution model**: Code runs top-to-bottom on every interaction; use session state for persistence.

2. **Prefer built-in Streamlit components**: Use native widgets before reaching for custom solutions.

3. **Follow the established patterns**: Match existing code style, component structure, and naming conventions in the project.

4. **Scripts directory organization**: When creating utility or operational scripts, always place them in the appropriate `scripts/` subdirectory:
   - Analysis/monitoring scripts → `scripts/analysis/`
   - Maintenance/cleanup scripts → `scripts/maintenance/`
   - Migration/data conversion scripts → `scripts/migration/`
   - Never place operational scripts in the project root

5. **Consider performance**: Suggest caching strategies for data operations and avoid unnecessary reruns.

6. **Security first**: Never suggest hardcoding secrets; always use `st.secrets` or environment variables.

7. **Test suggestions**: Provide testable code with clear input/output expectations.

8. **Explain trade-offs**: When suggesting alternatives, explain the pros and cons of each approach.

9. **Incremental improvements**: Suggest small, focused changes rather than large rewrites unless explicitly requested.

---

## Quick Reference

### Essential Streamlit Commands
```python
# Display
st.write()           # Universal display
st.markdown()        # Markdown text
st.dataframe()       # Interactive table
st.table()           # Static table
st.metric()          # KPI display
st.json()            # JSON viewer

# Input
st.text_input()      # Text field
st.number_input()    # Number field
st.selectbox()       # Dropdown
st.multiselect()     # Multi-select
st.slider()          # Slider
st.file_uploader()   # File upload
st.button()          # Button
st.form()            # Form container

# Layout
st.columns()         # Side-by-side columns
st.tabs()            # Tabbed interface
st.expander()        # Collapsible section
st.sidebar           # Sidebar container
st.container()       # Generic container

# State & Control
st.session_state     # Persistent state
st.rerun()           # Trigger rerun
st.stop()            # Stop execution
st.cache_data()      # Cache data
st.cache_resource()  # Cache resources
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.1.0 | 2026-01-31 | Added Scripts Directory Organization section with subdirectory structure and guidelines |
| 1.0.0 | Initial | Initial Claude.md creation |

---

*This document should be updated as the project evolves. Keep it in sync with actual project practices.*

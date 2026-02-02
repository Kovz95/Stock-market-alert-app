"""
Pytest configuration and shared fixtures.

This conftest.py file adds the project root to the Python path
so that tests can import from the src package.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Add src/data_access to path for legacy db_config imports
src_data_access = project_root / "src" / "data_access"
if str(src_data_access) not in sys.path:
    sys.path.insert(0, str(src_data_access))

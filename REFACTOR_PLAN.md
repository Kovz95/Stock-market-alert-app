 Ready to code?

 Here is Claude's plan:
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 Stock Market Alert App - Restructuring Plan

 Executive Summary

 Current State:
 - 60+ Python files in root directory with no clear organization
 - Zero testing infrastructure (no pytest, no tests, no CI/CD)
 - No code quality tooling (no linting, formatting, or type checking)
 - CRITICAL: Hardcoded API keys and Discord webhooks in source code
 - Large monolithic files: Add_Alert.py (2,991 lines), Scanner.py (2,722 lines), utils.py (1,791 lines)
 - Syntax error in view_scheduler_logs.py preventing it from running

 Team Context:
 - Small team (2-5 developers) working collaboratively
 - 1-2 month timeline for comprehensive modernization
 - All pain points: hard to find code, unsafe changes, overwhelming files, no quality confidence

 Target State:
 - Modern Python package structure with clear separation of concerns
 - Comprehensive testing infrastructure with 60%+ coverage
 - Automated code quality checks (linting, formatting, type checking)
 - Secure secrets management with zero hardcoded credentials
 - Modular, maintainable codebase with files under 500 lines
 - CI/CD pipeline ensuring quality on every commit

 ---
 New Directory Structure

 stock-market-alert-app/
 ├── pyproject.toml              # Enhanced with dev tools & dependencies
 ├── .env.example                # Template for required secrets
 ├── .gitignore                  # Already excludes .env ✓
 ├── README.md                   # Updated setup guide
 ├── .pre-commit-config.yaml     # Pre-commit hooks
 ├── pytest.ini                  # Pytest configuration
 │
 ├── .github/
 │   └── workflows/
 │       └── ci.yml              # GitHub Actions CI/CD
 │
 ├── src/
 │   └── stock_alert/            # Main application package
 │       ├── __init__.py
 │       │
 │       ├── web/                # Streamlit UI layer
 │       │   ├── Home.py         # Entry point (keep for Streamlit)
 │       │   └── pages/          # Organized by feature
 │       │       ├── alerts/     # Alert management pages
 │       │       ├── scanning/   # Scanner page
 │       │       ├── monitoring/ # Status/audit pages
 │       │       └── data/       # Database viewer pages
 │       │
 │       ├── core/               # Business logic (extracted from monoliths)
 │       │   ├── alerts/         # Alert domain logic
 │       │   │   ├── models.py
 │       │   │   ├── validator.py
 │       │   │   ├── processor.py
 │       │   │   └── suggestions.py
 │       │   ├── scanning/       # Scanner business logic
 │       │   │   ├── scanner.py
 │       │   │   ├── filters.py
 │       │   │   └── evaluator.py
 │       │   └── indicators/     # Technical indicators
 │       │       ├── library.py
 │       │       └── catalog.py
 │       │
 │       ├── data_access/        # Existing repository layer (move here)
 │       │   ├── alert_repository.py
 │       │   ├── portfolio_repository.py
 │       │   ├── metadata_repository.py
 │       │   └── database.py     # Consolidated db_config
 │       │
 │       ├── services/           # External integrations
 │       │   ├── data_providers/
 │       │   │   ├── base.py     # Abstract provider
 │       │   │   ├── fmp.py      # Consolidate 3 FMP backends
 │       │   │   └── ib.py       # Interactive Brokers
 │       │   ├── discord/
 │       │   │   ├── client.py
 │       │   │   └── routing.py
 │       │   └── cache/
 │       │       └── redis_client.py
 │       │
 │       ├── schedulers/         # Job scheduling
 │       │   ├── daily_scheduler.py
 │       │   ├── hourly_scheduler.py
 │       │   └── futures_scheduler.py
 │       │
 │       ├── config/             # Configuration management
 │       │   └── settings.py     # Pydantic settings for secrets
 │       │
 │       └── utils/              # Utilities (broken down from utils.py)
 │           ├── logging.py
 │           └── validation.py
 │
 ├── tests/                      # Test suite
 │   ├── conftest.py             # Shared fixtures
 │   ├── unit/                   # Unit tests
 │   │   ├── core/
 │   │   ├── services/
 │   │   └── data_access/
 │   ├── integration/            # Integration tests
 │   └── fixtures/               # Test data
 │
 ├── scripts/                    # Utility scripts (moved from root)
 │   ├── migration/              # DB migration scripts
 │   ├── maintenance/            # Maintenance tasks
 │   └── analysis/               # Analysis scripts
 │
 ├── db/
 │   └── schemas/                # Database schemas
 │
 └── docs/                       # Documentation
     ├── architecture.md
     ├── migration_guide.md
     └── testing_guide.md

 ---
 Implementation Plan (8 Weeks)

 Week 1: Foundation & Critical Fixes

 Goals: Fix critical security issues, set up testing infrastructure, configure quality tools

 Day 1-2: Security Fixes (CRITICAL)
 - Create .env.example with all required environment variables
 - Create src/stock_alert/config/settings.py using Pydantic Settings
 - Identify all hardcoded secrets:
   - utils.py lines 57, 62-63 (FMP_API_KEY, WEBHOOK_URL_LOGGING)
   - backend_fmp_optimized.py (FMP_API_KEY)
   - pages/Scanner.py (FMP_API_KEY)
   - Search for other instances: grep -r "webhook" --include="*.py"
 - Update all files to use Settings() instead of hardcoded values
 - CRITICAL: Fix syntax error in view_scheduler_logs.py line 14
   - Missing indented block after with statement
   - Indent lines 16-87 properly
 - Test with real credentials to ensure nothing breaks
 - Document all required env vars in README

 Day 3-4: Testing Infrastructure
 - Update pyproject.toml with dev dependencies:
 [project.optional-dependencies]
 dev = [
     "pytest>=7.4.0",
     "pytest-cov>=4.1.0",
     "pytest-asyncio>=0.21.0",
     "pytest-mock>=3.11.0",
     "pytest-xdist>=3.3.0",
     "freezegun>=1.2.0",
     "faker>=19.0.0",
 ]
 - Create pytest.ini with coverage thresholds (60% initial target)
 - Create tests/conftest.py with basic fixtures:
   - mock_database - In-memory test database
   - mock_fmp_client - Mocked FMP API
   - sample_alert_data - Test alert configurations
 - Write first 10 unit tests (data_access layer)
 - Ensure tests run: uv run pytest

 Day 5: Code Quality Tools
 - Configure ruff in pyproject.toml:
 [tool.ruff]
 line-length = 100
 target-version = "py313"
 select = ["E", "W", "F", "I", "N", "UP", "B", "C4", "SIM"]
 - Configure mypy (permissive initially)
 - Create .pre-commit-config.yaml with ruff, trailing whitespace checks
 - Install pre-commit: pre-commit install
 - Run first formatting pass: ruff format .
 - Create .github/workflows/ci.yml for GitHub Actions
 - Verify CI passes with current codebase

 Deliverables:
 - ✅ Zero hardcoded secrets in repository
 - ✅ Syntax error fixed
 - ✅ Working test suite (even if minimal coverage)
 - ✅ CI/CD pipeline passing
 - ✅ Pre-commit hooks running
 - ✅ All team members can run tests locally

 ---
 Week 2: Data Access Layer Migration

 Goals: Migrate well-organized data_access/ to new structure, add comprehensive tests

 Tasks:
 - Create src/stock_alert/ package structure
 - Move data_access/ → src/stock_alert/data_access/
 - Consolidate db_config.py → src/stock_alert/data_access/database.py
 - Add type hints to all repository methods (use mypy to verify)
 - Write comprehensive repository tests:
   - tests/unit/data_access/test_alert_repository.py
   - tests/unit/data_access/test_portfolio_repository.py
   - tests/unit/data_access/test_metadata_repository.py
 - Update imports in dependent files (Home.py, pages/, schedulers)
 - Test Streamlit app to ensure no breakage
 - Verify scheduler jobs still work

 Verification:
 - Run uv run pytest tests/unit/data_access/ -v
 - Start Streamlit: streamlit run Home.py
 - Test alert creation, deletion, viewing
 - Check scheduler status pages work

 Deliverables:
 - ✅ data_access/ properly packaged in src/
 - ✅ 80%+ test coverage for repositories
 - ✅ Type hints on all public methods
 - ✅ Streamlit app still functional

 ---
 Week 3: Service Layer Refactoring

 Goals: Consolidate backend implementations, extract Discord and cache services

 Tasks:
 - Create abstract AbstractDataProvider interface
 - Consolidate 3 FMP backends into services/data_providers/fmp.py:
   - Merge backend_fmp.py, backend_fmp_optimized.py
   - Keep best implementation (optimized version)
 - Create services/data_providers/ib.py from backend_futures_ib.py
 - Create services/data_providers/factory.py for provider selection
 - Extract Discord client from utils.py → services/discord/client.py
 - Extract routing logic → services/discord/routing.py
 - Move Redis logic → services/cache/redis_client.py
 - Write service layer tests:
   - tests/unit/services/test_fmp_provider.py (mock API responses)
   - tests/unit/services/test_discord.py (mock webhooks)
   - tests/unit/services/test_cache.py
 - Update dependent code to use new services

 Verification:
 - Run full test suite: uv run pytest
 - Test data fetching for stocks and futures
 - Test Discord notifications
 - Verify caching behavior

 Deliverables:
 - ✅ Consolidated data providers (6 backends → 2)
 - ✅ Extracted service layer
 - ✅ 70%+ test coverage for services
 - ✅ All integrations working

 ---
 Week 4-5: Core Business Logic Extraction

 Goals: Extract business logic from monolithic files, create domain models

 Week 4 Tasks:
 - Extract from utils.py (1,791 lines):
   - Market data loading → core/market_data/utils.py
   - Indicator calculations → core/indicators/library.py
   - Alert utilities → core/alerts/processor.py
   - Formatting → utils/formatting.py
   - Validation → utils/validation.py
 - Create alert domain models in core/alerts/models.py
 - Extract alert validation → core/alerts/validator.py
 - Write unit tests for core/alerts/ module
 - Write unit tests for core/indicators/ module

 Week 5 Tasks:
 - Extract from Scanner.py (2,722 lines):
   - Filtering logic → core/scanning/filters.py
   - Condition evaluation → core/scanning/evaluator.py
   - Scanner business logic → core/scanning/scanner.py
 - Keep UI code in web/pages/scanning/scanner.py (400-500 lines)
 - Write comprehensive scanning tests
 - Update Scanner page to use new core modules

 Verification:
 - utils.py reduced to < 500 lines
 - Scanner.py reduced to < 500 lines
 - All tests passing
 - Scanner page still functional
 - Core logic coverage > 75%

 Deliverables:
 - ✅ Business logic separated from UI
 - ✅ Reusable, testable core modules
 - ✅ Major files under 500 lines
 - ✅ High test coverage for core logic

 ---
 Week 6: Web Layer Refactoring

 Goals: Refactor Streamlit pages to use new core/services, reduce page complexity

 Priority Order:
 1. Add_Alert.py (2,991 lines) → web/pages/alerts/add_alert.py (300-400 lines)
   - Extract all business logic to core/alerts/
   - Keep only Streamlit UI code
   - Use new validator, processor, suggestions modules
 2. Scanner.py (already extracted in Week 5)
 3. Other pages (simpler, refactor as needed):
   - Edit_Alert.py
   - Delete_Alert.py
   - Alert_History.py
   - Price_Database.py
   - Stock_Database.py

 Tasks:
 - Create src/stock_alert/web/pages/ directory structure
 - Refactor Add_Alert.py (most critical):
   - UI rendering only in page file
   - All validation → core/alerts/validator.py
   - All processing → core/alerts/processor.py
   - All suggestions → core/alerts/suggestions.py
 - Test Add_Alert thoroughly with real data
 - Refactor remaining pages one by one
 - Keep old pages as *_old.py backup until verified
 - Update Home.py imports

 Verification:
 - All Streamlit pages load without errors
 - Alert creation/editing/deletion works
 - Scanner produces correct results
 - Database viewers display data correctly
 - Run app for 1 week in parallel with old code

 Deliverables:
 - ✅ All page files under 500 lines
 - ✅ Clear separation of UI and business logic
 - ✅ Streamlit app fully functional
 - ✅ Team can safely make changes

 ---
 Week 7: Schedulers & Scripts Organization

 Goals: Organize scheduler code, migrate maintenance scripts

 Tasks:
 - Create schedulers/ module structure
 - Move scheduler files:
   - auto_scheduler_v2.py → schedulers/daily_scheduler.py
   - hourly_data_scheduler.py → schedulers/hourly_scheduler.py
   - futures_auto_scheduler.py → schedulers/futures_scheduler.py
   - scheduler_watchdog_v2.py → schedulers/watchdog.py
 - Extract common scheduler logic to base class
 - Add scheduler tests
 - Organize scripts by purpose:
   - Migration scripts → scripts/migration/
   - Maintenance tasks → scripts/maintenance/
   - Analysis scripts → scripts/analysis/
   - Startup scripts → scripts/startup/
 - Update script imports
 - Update startup .bat files with new paths

 Verification:
 - Daily scheduler runs successfully
 - Hourly scheduler runs successfully
 - Futures scheduler runs successfully
 - All maintenance scripts work
 - Watchdog monitoring functional

 Deliverables:
 - ✅ Organized scheduler modules
 - ✅ Scripts categorized by purpose
 - ✅ Clean root directory (< 10 files)
 - ✅ All automation working

 ---
 Week 8: Documentation, Cleanup & Final Polish

 Goals: Comprehensive documentation, remove old code, achieve quality targets

 Tasks:
 - Write documentation:
   - docs/architecture.md - System architecture overview
   - docs/migration_guide.md - Guide for team on new structure
   - docs/testing_guide.md - How to write/run tests
   - docs/deployment.md - Deployment procedures
 - Update README with:
   - New setup instructions
   - Required environment variables
   - How to run tests
   - Project structure overview
 - Remove old backup files (*_old.py)
 - Final code quality pass:
   - Run ruff check . --fix
   - Run mypy src/
   - Fix any remaining issues
 - Achieve coverage targets:
   - Overall: 60%+
   - Core modules: 75%+
   - Services: 70%+
 - Team review and sign-off
 - Production deployment preparation

 Final Verification Checklist:
 - All tests passing: uv run pytest
 - Coverage meets targets: uv run pytest --cov
 - Linting passes: uv run ruff check .
 - Type checking passes: uv run mypy src/
 - CI/CD pipeline green
 - Streamlit app fully functional
 - All schedulers running
 - Zero hardcoded secrets
 - Documentation complete

 Deliverables:
 - ✅ Comprehensive documentation
 - ✅ Clean, organized codebase
 - ✅ Quality targets achieved
 - ✅ Team trained on new structure
 - ✅ Ready for production

 ---
 Critical Files to Modify

 Week 1 (Foundation):

 1. pyproject.toml - Add dev dependencies, configure tools
 2. utils.py:57-63 - Remove hardcoded FMP_API_KEY and WEBHOOK_URLs
 3. view_scheduler_logs.py:14 - Fix syntax error (indent lines 16-87)
 4. backend_fmp_optimized.py:25 - Remove hardcoded FMP_API_KEY
 5. pages/Scanner.py:4 - Remove hardcoded FMP_API_KEY
 6. .env.example (create) - Template for all required secrets
 7. src/stock_alert/config/settings.py (create) - Pydantic settings
 8. pytest.ini (create) - Test configuration
 9. .pre-commit-config.yaml (create) - Pre-commit hooks
 10. .github/workflows/ci.yml (create) - CI/CD pipeline

 Week 2-3 (Data & Services):

 11. data_access/*.py - Move to src/, add type hints
 12. db_config.py - Consolidate to data_access/database.py
 13. backend_fmp*.py - Consolidate to services/data_providers/fmp.py
 14. backend_futures_ib.py - Move to services/data_providers/ib.py
 15. discord_routing.py - Extract to services/discord/

 Week 4-5 (Core Logic):

 16. utils.py - Break into multiple modules (biggest refactor)
 17. indicators_lib.py - Move to core/indicators/library.py
 18. pages/Scanner.py - Extract business logic to core/scanning/

 Week 6 (UI):

 19. pages/Add_Alert.py - Major refactor to reduce from 2,991 lines
 20. Home.py - Update imports for new structure

 ---
 Testing Strategy

 Unit Tests (Target: 80% coverage)

 tests/unit/
 ├── core/
 │   ├── test_alerts.py          # Alert validation, processing
 │   ├── test_scanning.py        # Scanner filters, evaluation
 │   └── test_indicators.py      # Indicator calculations
 ├── services/
 │   ├── test_fmp_provider.py    # Mock FMP API responses
 │   ├── test_discord.py         # Mock webhooks
 │   └── test_cache.py           # Redis mocking
 └── data_access/
     └── test_repositories.py    # Database operations

 Key Testing Patterns:
 - Mock all external dependencies (database, APIs, Discord)
 - Use freezegun for time-dependent tests
 - Use faker for generating test data
 - Parametrize tests for multiple scenarios
 - Fast execution (< 1 second per test)

 Integration Tests (Target: 60% coverage)

 tests/integration/
 ├── test_database.py            # Real database operations (test schema)
 ├── test_alert_workflow.py      # Create → Process → Notify flow
 └── test_data_fetching.py       # Provider integration

 Fixtures (tests/conftest.py)

 @pytest.fixture
 def mock_database():
     """In-memory SQLite for testing"""

 @pytest.fixture
 def mock_fmp_client():
     """Mocked FMP API with sample responses"""

 @pytest.fixture
 def sample_alert_data():
     """Sample alert configurations for testing"""

 @pytest.fixture
 def sample_price_data():
     """Sample OHLCV data"""

 ---
 Code Quality Configuration

 pyproject.toml Additions

 [tool.ruff]
 line-length = 100
 target-version = "py313"
 select = ["E", "W", "F", "I", "N", "UP", "B", "C4", "SIM", "TCH"]
 ignore = ["E501"]  # Line too long (formatter handles it)

 [tool.ruff.per-file-ignores]
 "__init__.py" = ["F401"]  # Unused imports OK
 "tests/*" = ["S101"]      # Assert in tests OK

 [tool.mypy]
 python_version = "3.13"
 warn_return_any = true
 warn_unused_configs = true
 disallow_untyped_defs = false  # Start permissive
 ignore_missing_imports = true
 exclude = [".venv", "build"]

 [tool.pytest.ini_options]
 testpaths = ["tests"]
 addopts = "--cov=src/stock_alert --cov-report=html --cov-report=term-missing --cov-fail-under=60 -v"

 Pre-commit Hooks

 repos:
   - repo: https://github.com/astral-sh/ruff-pre-commit
     rev: v0.1.0
     hooks:
       - id: ruff
         args: [--fix]
       - id: ruff-format

   - repo: https://github.com/pre-commit/pre-commit-hooks
     rev: v4.5.0
     hooks:
       - id: trailing-whitespace
       - id: end-of-file-fixer
       - id: check-yaml
       - id: check-json
       - id: detect-private-key

 ---
 Team Coordination

 For 2-5 Developer Team:

 1. Week-by-week sprints - Each week has clear deliverables
 2. Branch strategy:
   - main - production code
   - develop - integration branch
   - feature/week-X-* - weekly feature branches
 3. Daily standups (15 min) - Progress, blockers, coordination
 4. Weekly demos - Show completed work to team
 5. Pair programming for complex refactors (Add_Alert.py, Scanner.py)
 6. Code review required for all PRs
 7. Documentation as you go - Update docs with each change

 Risk Mitigation:

 1. Backward compatibility - Keep old files until new ones verified
 2. Feature flags - Can switch between old/new implementations
 3. Comprehensive testing - Catch regressions early
 4. Parallel running - Run both old/new code side-by-side initially
 5. Easy rollback - Git branches allow quick reversion
 6. Incremental deployment - One module at a time

 ---
 Success Metrics

 Code Quality (End of Week 8):

 - ✅ Test coverage ≥ 60% overall, ≥ 75% for core modules
 - ✅ All linting checks pass (ruff)
 - ✅ Type hints on ≥ 50% of functions
 - ✅ All files < 500 lines
 - ✅ Zero hardcoded secrets

 Structural Metrics:

 - ✅ < 10 Python files in root directory
 - ✅ Clear package structure (web, core, services, data_access)
 - ✅ Documented architecture
 - ✅ Consistent import patterns

 Developer Experience:

 - ✅ CI/CD pipeline runs on all PRs
 - ✅ Pre-commit hooks catch issues locally
 - ✅ New developer setup < 30 minutes
 - ✅ Team confident making changes without breaking things

 Functional:

 - ✅ All Streamlit pages working
 - ✅ All schedulers running
 - ✅ All integrations functional (Discord, FMP, IB)
 - ✅ Zero production incidents during migration

 ---
 Quick Start After Approval

 Day 1 Commands:

 # 1. Install dev dependencies
 uv sync --all-extras

 # 2. Create environment template
 cat > .env.example << 'EOF'
 # API Keys
 FMP_API_KEY=your_fmp_api_key_here

 # Discord Webhooks
 WEBHOOK_URL_LOGGING=your_discord_webhook_url
 WEBHOOK_URL_LOGGING_2=your_discord_webhook_url_2

 # Database
 DATABASE_URL=postgresql://user:pass@localhost/dbname
 REDIS_URL=redis://localhost:6379
 EOF

 # 3. Create your actual .env (add real values)
 cp .env.example .env
 # Edit .env with your actual secrets

 # 4. Fix syntax error
 # Edit view_scheduler_logs.py line 14-87, indent properly

 # 5. Set up pre-commit
 pip install pre-commit
 pre-commit install

 # 6. Run first format pass
 ruff format .

 # 7. Create pytest.ini
 # (see configuration above)

 # 8. Run tests (will fail initially, that's OK)
 uv run pytest

 This plan provides a clear 8-week roadmap to modernize the Stock Market Alert App while maintaining functionality
 throughout the migration.

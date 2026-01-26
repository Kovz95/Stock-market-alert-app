# Stock Market Alert App

A comprehensive stock market alert and scanning application with automated notifications, technical indicators, and scheduled data collection.

## Version 2.0 - Modernization Project

This application is currently undergoing a comprehensive modernization to improve maintainability, security, and code quality. See the [Refactoring Plan](REFACTOR_PLAN.md) for details.

## Features

- **Real-time Stock Alerts**: Configure custom alerts based on technical indicators
- **Stock Scanner**: Scan markets based on custom conditions and filters
- **Automated Scheduling**: Hourly and daily data collection and alert processing
- **Multi-Exchange Support**: Stocks, futures, and international markets
- **Discord Integration**: Automated notifications via Discord webhooks
- **Technical Indicators**: RSI, SMA, EMA, MACD, Bollinger Bands, and more
- **Portfolio Management**: Track multiple portfolios and watchlists

## Prerequisites

- **Python 3.13+** (required)
- **PostgreSQL** (for data storage)
- **Redis** (optional, for caching)
- **uv** package manager (recommended)

## Installation

### 1. Install uv Package Manager

```bash
# Windows (PowerShell)
pip install uv

# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Add `uv` to your PATH environment variable if not automatically added.

### 2. Clone the Repository

```bash
git clone <repository-url>
cd stock-market-alert-app
```

### 3. Set Up Environment Variables

Copy the example environment file and configure with your credentials:

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:

```env
# Financial Modeling Prep API Key (required)
FMP_API_KEY=your_fmp_api_key_here

# Discord Webhooks (required for notifications)
WEBHOOK_URL=your_primary_webhook_url
WEBHOOK_URL_2=your_secondary_webhook_url
WEBHOOK_URL_LOGGING=your_logging_webhook_url
WEBHOOK_URL_LOGGING_2=your_secondary_logging_webhook_url

# Application Settings
ENVIRONMENT=development
DEBUG=false
```

**IMPORTANT**: Never commit your `.env` file with real credentials to version control.

### 4. Install Dependencies

```bash
# Install all dependencies including dev tools
uv sync --all-extras

# Or install just production dependencies
uv sync
```

### 5. Set Up Pre-commit Hooks

```bash
uv run pre-commit install
```

This will automatically run code quality checks before each commit.

## Running the Application

### Streamlit Web Interface

```bash
streamlit run Home.py
```

Access the web interface at `http://localhost:8501`

### Schedulers

The application includes several automated schedulers:

```bash
# Daily alert scheduler
python auto_scheduler_v2.py

# Hourly data collector
python hourly_data_scheduler.py

# Futures scheduler
python futures_auto_scheduler.py
```

## Development

### Running Tests

```bash
# Run all tests with coverage
uv run pytest

# Run specific test file
uv run pytest tests/unit/config/test_settings.py

# Run with verbose output
uv run pytest -v

# Run only unit tests
uv run pytest tests/unit/

# Run only integration tests
uv run pytest tests/integration/
```

### Code Quality Checks

```bash
# Format code with ruff
uv run ruff format .

# Lint code with ruff
uv run ruff check .

# Auto-fix linting issues
uv run ruff check . --fix

# Type checking with mypy
uv run mypy src/

# Run all pre-commit hooks manually
uv run pre-commit run --all-files
```

### Code Coverage

After running tests, view the coverage report:

```bash
# Terminal report
uv run pytest --cov-report=term-missing

# HTML report (opens in browser)
# Coverage report is automatically generated in htmlcov/
# Open htmlcov/index.html in your browser
```

## Project Structure

```
stock-market-alert-app/
├── src/
│   └── stock_alert/          # Main application package
│       ├── config/            # Configuration management
│       ├── core/              # Business logic
│       ├── services/          # External integrations
│       ├── data_access/       # Database layer
│       └── web/               # Streamlit UI
│
├── tests/                     # Test suite
│   ├── unit/                  # Unit tests
│   ├── integration/           # Integration tests
│   └── conftest.py            # Shared fixtures
│
├── pages/                     # Streamlit pages (legacy)
├── data_access/               # Database repositories (legacy)
├── scripts/                   # Utility scripts
├── docs/                      # Documentation
│
├── .env                       # Environment variables (DO NOT COMMIT)
├── .env.example               # Environment template
├── pyproject.toml             # Project configuration
├── .pre-commit-config.yaml    # Pre-commit hooks
└── README.md                  # This file
```

## Configuration

### Required Environment Variables

- `FMP_API_KEY`: Financial Modeling Prep API key
- `WEBHOOK_URL_LOGGING`: Discord webhook for application logs
- `WEBHOOK_URL_LOGGING_2`: Secondary Discord webhook for logs

### Optional Environment Variables

- `WEBHOOK_URL`: Discord webhook for alert notifications
- `WEBHOOK_URL_2`: Secondary Discord webhook for alerts
- `ENVIRONMENT`: Application environment (development/production)
- `DEBUG`: Enable debug mode (true/false)
- `DATABASE_URL`: PostgreSQL connection string override
- `REDIS_URL`: Redis connection string

See `.env.example` for complete list with descriptions.

## API Keys and Webhooks

### Financial Modeling Prep (FMP)

Get your API key from: https://financialmodelingprep.com/developer/docs/

### Discord Webhooks

1. Open Discord Server Settings
2. Navigate to Integrations > Webhooks
3. Create New Webhook
4. Copy the webhook URL
5. Add to `.env` file

## Security Best Practices

✅ **DO:**
- Store credentials in `.env` file
- Use the `Settings` class from `src/stock_alert/config/settings.py`
- Keep `.env` in `.gitignore`
- Rotate API keys and webhooks regularly

❌ **DON'T:**
- Hardcode credentials in source code
- Commit `.env` file to version control
- Share credentials in chat/email
- Use production credentials in development

## Testing Infrastructure

The project uses:
- **pytest**: Test framework
- **pytest-cov**: Coverage reporting
- **pytest-mock**: Mocking utilities
- **faker**: Test data generation
- **freezegun**: Time mocking

See `tests/conftest.py` for available fixtures.

## Code Quality Tools

- **ruff**: Fast Python linter and formatter
- **mypy**: Static type checking
- **pre-commit**: Git hooks for automated checks
- **bandit**: Security vulnerability scanning

## CI/CD Pipeline

GitHub Actions automatically runs on every push and pull request:
- Code formatting checks (ruff format)
- Linting (ruff check)
- Type checking (mypy)
- Test suite with coverage
- Security checks (bandit, secret detection)

See `.github/workflows/ci.yml` for configuration.

## Week 1 Deliverables ✅

- [x] Zero hardcoded secrets in repository
- [x] Syntax error in `view_scheduler_logs.py` fixed
- [x] Working test suite with 98% coverage for config module
- [x] CI/CD pipeline configured
- [x] Pre-commit hooks installed
- [x] Code formatting and linting tools configured

## Contributing

1. Create a feature branch from `develop`
2. Make your changes
3. Run tests and quality checks
4. Submit a pull request

All PRs must pass CI checks before merging.

## Troubleshooting

### Import Errors

If you encounter import errors, ensure:
- You're using the virtual environment: `uv venv` and activate it
- All dependencies are installed: `uv sync --all-extras`
- PYTHONPATH includes the project root

### Missing Environment Variables

If you see "environment variable is required" errors:
1. Ensure `.env` file exists in project root
2. Check all required variables are set (see `.env.example`)
3. Restart your application/terminal after modifying `.env`

### Test Failures

If tests fail:
- Ensure test environment variables are set (see `tests/conftest.py`)
- Clear pytest cache: `uv run pytest --cache-clear`
- Check for stale Python bytecode: `find . -name "*.pyc" -delete`

## Documentation

- [Refactoring Plan](REFACTOR_PLAN.md) - Detailed modernization roadmap
- [Architecture Documentation](docs/architecture.md) - Coming soon
- [Migration Guide](docs/migration_guide.md) - Coming soon
- [Testing Guide](docs/testing_guide.md) - Coming soon

## License

[Add license information]

## Support

For issues and questions:
- Create an issue in the GitHub repository
- Check existing documentation in `docs/`
- Review the refactoring plan for ongoing changes

---

**Note**: This application is actively being modernized. See [REFACTOR_PLAN.md](REFACTOR_PLAN.md) for the 8-week modernization roadmap. We're currently in Week 1 - Foundation & Critical Fixes.

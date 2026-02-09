# Stock Market Alert Application

A comprehensive, production-ready stock market alert system built with Streamlit, featuring real-time price monitoring, technical indicator-based alerts, and Discord notifications.

## Features

### Core Capabilities
- **Multi-Timeframe Alerts**: Daily, weekly, and hourly alert monitoring
- **Technical Indicators**: RSI, Moving Averages, Bollinger Bands, MACD, and more (powered by TA-Lib)
- **Dual Market Support**: Track both regular stocks and futures contracts
- **Real-Time Data**: Integration with Financial Modeling Prep API and Interactive Brokers
- **Smart Notifications**: Discord webhook integration for instant alert delivery
- **Portfolio Tracking**: Monitor your positions with price targets and stop losses
- **Historical Analysis**: View alert history and performance analytics

### Application Pages
- **Home**: Dashboard with scheduler status and alert statistics
- **Add/Edit/Delete Alerts**: Manage your alert configurations
- **Alert History**: Review triggered alerts and performance
- **Scanner**: Search and filter securities
- **Database Views**: Browse stock, futures, and price data
- **Portfolio Management**: Track your positions and targets
- **Discord Management**: Configure notification channels
- **Scheduler Status**: Monitor daily/weekly and hourly schedulers
- **Market Hours**: Check trading hours and market status
- **Alert Audit Logs**: Track all alert modifications

## Technology Stack

### Core Framework
- **Python**: 3.13+
- **Streamlit**: Web application framework
- **UV**: Modern Python package manager

### Data & Storage
- **PostgreSQL**: Primary database
- **Redis**: Caching and session management
- **SQLAlchemy**: Database ORM
- **Pandas**: Data manipulation

### Market Data
- **fmpsdk**: Financial Modeling Prep API client
- **ib-insync**: Interactive Brokers integration
- **ta-lib**: Technical analysis indicators

### Scheduling & Monitoring
- **APScheduler**: Job scheduling for automated checks
- **exchange-calendars**: Market calendar support
- **psutil**: System monitoring

### Testing
- **pytest**: Test framework
- **pytest-asyncio**: Async test support
- **pytest-cov**: Coverage reporting
- **hypothesis**: Property-based testing
- **factory-boy**: Test fixture generation

## Installation

### Prerequisites
- Python 3.13 or higher
- PostgreSQL 12+
- Redis 6+
- TA-Lib C library ([installation guide](https://github.com/ta-lib/ta-lib-python#dependencies))

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Stock-market-alert-app
   ```

2. **Install UV package manager**
   ```bash
   pip install uv
   ```

3. **Create and activate virtual environment**
   ```bash
   uv venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux
   source .venv/bin/activate
   ```

4. **Install dependencies**
   ```bash
   uv pip install -e .
   ```

5. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

6. **Set up the database**
   ```bash
   # Using Docker Compose (recommended)
   docker-compose up -d postgres redis

   # Or set up PostgreSQL and Redis manually
   ```

7. **Initialize the database schema**
   ```bash
   python -m src.data_access.database
   ```

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`. Key configurations:

#### Database
```env
DATABASE_URL=postgresql://user:password@localhost:5432/stockalertapp
POSTGRES_POOL_MIN=5
POSTGRES_POOL_MAX=50
```

#### Redis
```env
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_KEY_PREFIX=stockalert:
```

#### API Keys
```env
FMP_API_KEY=your_fmp_api_key_here
```

#### Discord Notifications
```env
DISCORD_SEND_ENABLED=true
WEBHOOK_URL=https://discord.com/api/webhooks/your_webhook_id/token
WEBHOOK_URL_LOGGING=https://discord.com/api/webhooks/your_webhook_id/token
```

#### Scheduler Settings
```env
SCHEDULER_JOB_TIMEOUT=900
SCHEDULER_HEARTBEAT_INTERVAL=60
SCHEDULER_ALERT_CHECK_WORKERS=5
```

### Alert Configuration

Alert behavior can be customized via `alert_processing_config.json`:
```json
{
  "max_alerts_per_run": 100,
  "rate_limit_delay_ms": 500,
  "retry_attempts": 3
}
```

## Usage

### Running the Application

#### Local Development
```bash
streamlit run Home.py
```

The application will be available at `http://localhost:8501`

#### Docker Deployment
```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f app

# Stop services
docker-compose down
```

### Running the Schedulers

The application includes automated schedulers for checking alerts:

**Daily/Weekly Scheduler**
```bash
python -m src.services.scheduler_service
```

**Hourly Scheduler**
```bash
python -m src.services.hourly_scheduler
```

**Futures Scheduler**
```bash
python -m src.services.futures_scheduler
```

### Running the Watchdog

Monitor scheduler health and send alerts if schedulers stop:
```bash
python -m src.services.watchdog
```

## Project Structure

```
Stock-market-alert-app/
├── Home.py                          # Main Streamlit entry point
├── pages/                           # Streamlit pages
│   ├── Add_Alert.py                 # Create new alerts
│   ├── Edit_Alert.py                # Modify existing alerts
│   ├── Alert_History.py             # View triggered alerts
│   ├── Scanner.py                   # Search securities
│   ├── My_Portfolio.py              # Portfolio tracking
│   ├── Futures_Alerts_Home.py       # Futures alert management
│   ├── Daily_Weekly_Scheduler_Status.py
│   ├── Hourly_Scheduler_Status.py
│   └── ...
├── src/
│   ├── data_access/                 # Database layer
│   │   ├── database.py              # Core database setup
│   │   ├── alert_repository.py      # Alert CRUD operations
│   │   ├── metadata_repository.py   # Stock metadata
│   │   ├── redis_support.py         # Redis operations
│   │   └── document_store.py        # Document storage
│   ├── services/                    # Business logic
│   │   ├── scheduler_service.py     # Daily/weekly scheduler
│   │   ├── hourly_scheduler.py      # Hourly alert checker
│   │   ├── futures_scheduler.py     # Futures alerts
│   │   ├── watchdog.py              # Scheduler monitoring
│   │   ├── discord_support.py       # Discord webhooks
│   │   └── backend_*.py             # Market data backends
│   ├── utils/                       # Utilities
│   └── models/                      # Data models
├── scripts/                         # Utility scripts
│   ├── analysis/                    # Analysis scripts
│   ├── maintenance/                 # Maintenance tasks
│   └── migration/                   # Data migrations
├── tests/                           # Test suite
├── docker-compose.yml               # Docker services
├── Dockerfile                       # Application container
├── pyproject.toml                   # Project metadata
├── .env.example                     # Example configuration
└── CLAUDE.md                        # AI assistant instructions
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test markers
pytest -m backend_e2e -v   # Backend integration tests
pytest -m e2e -v           # End-to-end tests
```

### Code Quality

The project follows PEP 8 style guidelines with type hints:

```python
def process_alert(alert_id: int, threshold: float = 0.5) -> Optional[dict]:
    """Process an alert and return results.

    Args:
        alert_id: The alert identifier
        threshold: Minimum threshold value

    Returns:
        Dictionary with alert results or None
    """
    pass
```

### Database Migrations

Migration scripts are located in the `migration/` directory:
- `import_sqlite_to_postgres.py`: Migrate from SQLite
- `load_json_to_postgres.py`: Import JSON data

### Utility Scripts

Located in `scripts/` subdirectories:

**Analysis** (`scripts/analysis/`)
```bash
python scripts/analysis/check_latest_hourly_coverage.py
```

**Maintenance** (add as needed)
**Migration** (add as needed)

## Deployment

### Docker Deployment

1. **Configure environment**
   ```bash
   cp .env.example .env.production
   # Edit .env.production with production values
   ```

2. **Build and deploy**
   ```bash
   docker-compose -f docker-compose.yml up -d
   ```

3. **Verify services**
   ```bash
   docker-compose ps
   docker-compose logs -f app
   ```

### Production Checklist
- [ ] Set `ENVIRONMENT=production` in `.env`
- [ ] Configure secure database credentials
- [ ] Set up Redis password protection
- [ ] Configure Discord webhooks
- [ ] Enable scheduler watchdog
- [ ] Set up automated backups
- [ ] Configure logging and monitoring
- [ ] Review rate limits and timeouts

## Monitoring & Maintenance

### Health Checks
- Scheduler status pages in the Streamlit UI
- Watchdog alerts via Discord
- Redis cache monitoring
- Database connection pool status

### Logs
- Application logs: Check Docker logs or Streamlit output
- Scheduler logs: `scheduler_watchdog.log`
- Discord audit trail: Alert Audit Logs page

### Backup & Recovery
- Regular PostgreSQL backups
- Redis persistence configuration
- Environment variable backups

## Troubleshooting

### Common Issues

**Database Connection Errors**
```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Verify connection string
psql postgresql://user:password@localhost:5432/dbname
```

**Redis Connection Errors**
```bash
# Check Redis is running
redis-cli ping

# Verify Redis configuration
docker-compose logs redis
```

**API Rate Limits**
- Adjust `SCHEDULER_ALERT_CHECK_WORKERS` to reduce concurrent requests
- Increase rate limit delays in configuration

**Missing TA-Lib**
- Ensure TA-Lib C library is installed
- Windows: Download and install from [ta-lib.org](https://ta-lib.org/)
- macOS: `brew install ta-lib`
- Linux: `sudo apt-get install ta-lib`

## Contributing

1. Create a feature branch
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

[Specify your license here]

## Support

For issues, questions, or feature requests, please open an issue on the repository.

---

**Version**: 0.1.0
**Last Updated**: February 2026

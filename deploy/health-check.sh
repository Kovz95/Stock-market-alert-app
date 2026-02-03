#!/bin/bash
# Health check script - verifies deployment was successful
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Running health checks ==="

FAILED=0

# Check if Streamlit app is responding
echo "Checking Streamlit app..."
if curl -s -o /dev/null -w "%{http_code}" http://localhost:8501 | grep -q "200\|302"; then
    echo "  Streamlit app: OK"
else
    echo "  Streamlit app: FAILED"
    FAILED=1
fi

# Check systemd services
SERVICES=(
    "stockalert-app"
    "stockalert-scheduler"
    "stockalert-hourly"
    "stockalert-futures"
    "stockalert-watchdog"
)

echo ""
echo "Checking services..."
for service in "${SERVICES[@]}"; do
    if systemctl is-enabled --quiet "$service" 2>/dev/null; then
        if systemctl is-active --quiet "$service"; then
            echo "  $service: OK"
        else
            echo "  $service: FAILED"
            FAILED=1
        fi
    fi
done

# Check database connectivity
echo ""
echo "Checking database connection..."
cd "$APP_DIR"
if uv run python -c "from src.data_access.db_config import get_engine; get_engine().connect().close(); print('  Database: OK')" 2>/dev/null; then
    :
else
    echo "  Database: FAILED"
    FAILED=1
fi

# Check Redis connectivity
echo "Checking Redis connection..."
if uv run python -c "from src.data_access.redis_support import get_redis_client; r = get_redis_client(); r.ping(); print('  Redis: OK')" 2>/dev/null; then
    :
else
    echo "  Redis: FAILED (may be optional)"
fi

echo ""
if [ $FAILED -eq 0 ]; then
    echo "=== All health checks passed ==="
    exit 0
else
    echo "=== Some health checks failed ==="
    exit 1
fi

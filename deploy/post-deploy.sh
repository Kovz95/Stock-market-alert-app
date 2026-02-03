#!/bin/bash
# Post-deployment script - runs on the server after rsync
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Post-deployment script started ==="
echo "App directory: $APP_DIR"
cd "$APP_DIR"

# Install/update dependencies with uv
echo "Installing dependencies..."
if command -v uv &> /dev/null; then
    uv sync --frozen
else
    echo "ERROR: uv not found. Please install uv first."
    exit 1
fi

# Reload systemd daemon to pick up any service file changes
echo "Reloading systemd daemon..."
sudo systemctl daemon-reload

# Restart all services
echo "Restarting services..."

# Array of services to restart
SERVICES=(
    "stockalert-app"
    "stockalert-scheduler"
    "stockalert-hourly"
    "stockalert-futures"
    "stockalert-watchdog"
)

for service in "${SERVICES[@]}"; do
    if systemctl is-enabled --quiet "$service" 2>/dev/null; then
        echo "  Restarting $service..."
        sudo systemctl restart "$service"
    else
        echo "  Skipping $service (not enabled)"
    fi
done

# Wait a moment for services to start
sleep 3

# Show status
echo ""
echo "=== Service Status ==="
for service in "${SERVICES[@]}"; do
    if systemctl is-enabled --quiet "$service" 2>/dev/null; then
        status=$(systemctl is-active "$service" 2>/dev/null || echo "unknown")
        echo "  $service: $status"
    fi
done

echo ""
echo "=== Deployment completed ==="

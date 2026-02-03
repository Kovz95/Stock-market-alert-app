#!/bin/bash
# Server setup script - run once on the server to configure deployment
# Usage: sudo bash deploy/setup-server.sh
set -e

APP_USER="${APP_USER:-stockalert}"
APP_DIR="${APP_DIR:-/opt/stockalert}"

echo "=== Stock Alert Server Setup ==="
echo "User: $APP_USER"
echo "Directory: $APP_DIR"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run as root (sudo)"
    exit 1
fi

# Create application user if it doesn't exist
if ! id "$APP_USER" &>/dev/null; then
    echo "Creating user $APP_USER..."
    useradd -r -m -s /bin/bash "$APP_USER"
else
    echo "User $APP_USER already exists"
fi

# Create application directory
echo "Setting up application directory..."
mkdir -p "$APP_DIR"
chown "$APP_USER:$APP_USER" "$APP_DIR"

# Install uv for the app user
echo "Installing uv for $APP_USER..."
sudo -u "$APP_USER" bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'

# Copy systemd service files
echo "Installing systemd services..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Update service files with correct paths if needed
for service_file in "$SCRIPT_DIR"/systemd/*.service; do
    if [ -f "$service_file" ]; then
        service_name=$(basename "$service_file")
        echo "  Installing $service_name..."

        # Copy and update paths
        sed -e "s|/opt/stockalert|$APP_DIR|g" \
            -e "s|User=stockalert|User=$APP_USER|g" \
            -e "s|Group=stockalert|Group=$APP_USER|g" \
            -e "s|/home/stockalert|/home/$APP_USER|g" \
            "$service_file" > "/etc/systemd/system/$service_name"
    fi
done

# Reload systemd
echo "Reloading systemd..."
systemctl daemon-reload

# Enable services
echo "Enabling services..."
systemctl enable stockalert-app
systemctl enable stockalert-scheduler
systemctl enable stockalert-hourly
systemctl enable stockalert-futures
systemctl enable stockalert-watchdog

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Copy your application files to $APP_DIR"
echo "2. Create $APP_DIR/.env with your environment variables (see .env.example)"
echo "3. Run: cd $APP_DIR && sudo -u $APP_USER uv sync"
echo "4. Start services: sudo systemctl start stockalert-app stockalert-scheduler stockalert-hourly stockalert-futures stockalert-watchdog"
echo ""
echo "Useful commands:"
echo "  sudo systemctl status stockalert-app"
echo "  sudo journalctl -u stockalert-app -f"
echo "  sudo systemctl restart stockalert-app"

#!/bin/bash
# Deploy to Docker on remote server
# Usage: ./deploy/deploy-docker.sh [server] [path]
#
# Examples:
#   ./deploy/deploy-docker.sh                          # Uses defaults
#   ./deploy/deploy-docker.sh root@stockviz.example.com
#   ./deploy/deploy-docker.sh root@stockviz.example.com /opt/stockalert

set -e

# Configuration - customize these or pass as arguments
SERVER="${1:-root@45.63.20.126}"
REMOTE_PATH="${2:-/opt/stockalert}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Stock Alert Docker Deployment ===${NC}"
echo "Server: $SERVER"
echo "Remote path: $REMOTE_PATH"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(dirname "$SCRIPT_DIR")"

cd "$APP_DIR"

# Step 1: Package and copy files to server using scp
echo -e "${YELLOW}[1/4] Packaging and copying files to server...${NC}"

# Create temporary archive
TEMP_ARCHIVE="stockalert-deploy.tar.gz"
echo "Creating archive..."

# Create archive excluding unwanted files
tar --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='.env' \
    --exclude='node_modules' \
    --exclude='*.log' \
    --exclude='logs/' \
    --exclude='.pytest_cache' \
    --exclude='.mypy_cache' \
    --exclude='*.egg-info' \
    --exclude="$TEMP_ARCHIVE" \
    -czf "$TEMP_ARCHIVE" -C . .

# Copy archive to server using scp
echo "Copying to server..."
scp "$TEMP_ARCHIVE" "$SERVER:$REMOTE_PATH/"

# Extract on server
echo "Extracting on server..."
ssh "$SERVER" "cd $REMOTE_PATH && tar -xzf $TEMP_ARCHIVE && rm $TEMP_ARCHIVE"

# Clean up local archive
rm "$TEMP_ARCHIVE"
echo "Cleanup complete."

# Step 2: Copy .env.production to server as .env
echo -e "${YELLOW}[2/4] Deploying production environment file...${NC}"

# Check if .env.production exists locally
if [ ! -f ".env.production" ]; then
    echo -e "${RED}ERROR: .env.production file not found in project root${NC}"
    echo ""
    echo "Please ensure .env.production exists with required variables:"
    echo "  - POSTGRES_PASSWORD"
    echo "  - FMP_API_KEY"
    echo "  - WEBHOOK_URL (Discord)"
    exit 1
fi

echo "Copying .env.production to server as .env..."
scp ".env.production" "$SERVER:$REMOTE_PATH/.env"
echo -e "${GREEN}Environment file deployed successfully.${NC}"

# Step 3: Stop existing containers and rebuild
echo -e "${YELLOW}[3/4] Building and starting Docker containers...${NC}"
ssh "$SERVER" "cd $REMOTE_PATH && docker compose down && docker compose up -d --build"

# Step 4: Show status
echo -e "${YELLOW}[4/4] Checking container status...${NC}"
sleep 5
ssh "$SERVER" "cd $REMOTE_PATH && docker compose ps"

echo ""
echo -e "${GREEN}=== Deployment Complete ===${NC}"
echo ""
echo "Useful commands:"
echo "  View logs:     ssh $SERVER 'cd $REMOTE_PATH && docker compose logs -f'"
echo "  View app logs: ssh $SERVER 'cd $REMOTE_PATH && docker compose logs -f streamlit'"
echo "  Restart:       ssh $SERVER 'cd $REMOTE_PATH && docker compose restart'"
echo "  Stop:          ssh $SERVER 'cd $REMOTE_PATH && docker compose down'"
echo ""

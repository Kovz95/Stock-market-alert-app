# Deployment Guide

This guide explains how to deploy the Stock Alert application to a Vultr server using GitHub Actions.

## Prerequisites

### On the Server
- Ubuntu 22.04+ or Debian 12+
- PostgreSQL installed and running
- Redis installed and running
- Python 3.13+ available

### On GitHub
- Access to repository settings for secrets

## Initial Server Setup

### 1. Run the Setup Script

SSH into your server and run the initial setup:

```bash
# Clone or copy the deploy directory to the server first
cd /tmp
git clone <your-repo> stockalert-setup
cd stockalert-setup

# Run setup (creates user, installs services)
sudo bash deploy/setup-server.sh
```

This will:
- Create a `stockalert` user
- Create `/opt/stockalert` directory
- Install `uv` package manager
- Install and enable systemd services

### 2. Configure Environment Variables

Create the environment file on the server:

```bash
sudo -u stockalert nano /opt/stockalert/.env
```

Add your configuration:

```env
# Environment
ENVIRONMENT=production

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/stockalert

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# API Keys
FMP_API_KEY=your_fmp_api_key

# Discord Webhooks
WEBHOOK_URL=https://discord.com/api/webhooks/...
WEBHOOK_URL_LOGGING=https://discord.com/api/webhooks/...

# Scheduler settings (optional)
SCHEDULER_JOB_TIMEOUT=900
SCHEDULER_HEARTBEAT_INTERVAL=60

# Watchdog settings (optional)
WATCHDOG_ENABLE_DISCORD=true
WATCHDOG_DISCORD_WEBHOOK=https://discord.com/api/webhooks/...
```

### 3. Set Up SSH Key for Deployment

Generate a deploy key (on your local machine):

```bash
ssh-keygen -t ed25519 -C "github-deploy" -f ~/.ssh/stockalert_deploy -N ""
```

Add the public key to the server:

```bash
# On the server
sudo -u stockalert mkdir -p /home/stockalert/.ssh
sudo -u stockalert nano /home/stockalert/.ssh/authorized_keys
# Paste the content of ~/.ssh/stockalert_deploy.pub
sudo chmod 700 /home/stockalert/.ssh
sudo chmod 600 /home/stockalert/.ssh/authorized_keys
```

## GitHub Actions Setup

### Required Secrets

Add these secrets in your GitHub repository settings (Settings > Secrets and variables > Actions):

| Secret | Description | Example |
|--------|-------------|---------|
| `SSH_PRIVATE_KEY` | Contents of your deploy private key | `-----BEGIN OPENSSH PRIVATE KEY-----...` |
| `SERVER_HOST` | Your Vultr server IP or hostname | `123.45.67.89` |
| `SSH_USER` | SSH username for deployment | `stockalert` |
| `DEPLOY_PATH` | Path to deploy to | `/opt/stockalert` |
| `DISCORD_WEBHOOK_URL` | (Optional) Webhook for deploy notifications | `https://discord.com/api/webhooks/...` |

### Setting Up Secrets

1. Go to your GitHub repository
2. Click **Settings** > **Secrets and variables** > **Actions**
3. Click **New repository secret**
4. Add each secret listed above

For `SSH_PRIVATE_KEY`, copy the entire contents of your private key file:
```bash
cat ~/.ssh/stockalert_deploy
```

## Deployment

### Automatic Deployment

Push to the `main` branch to trigger automatic deployment:

```bash
git push origin main
```

### Manual Deployment

Trigger a manual deployment from GitHub:

1. Go to **Actions** > **Deploy to Vultr**
2. Click **Run workflow**
3. Optionally check "Skip tests" for faster deployment
4. Click **Run workflow**

## Services

### Service Management

```bash
# View status of all services
sudo systemctl status stockalert-*

# Restart a specific service
sudo systemctl restart stockalert-app

# View logs
sudo journalctl -u stockalert-app -f

# Stop all services
sudo systemctl stop stockalert-app stockalert-scheduler stockalert-hourly stockalert-futures stockalert-watchdog

# Start all services
sudo systemctl start stockalert-app stockalert-scheduler stockalert-hourly stockalert-futures stockalert-watchdog
```

### Service Descriptions

| Service | Description | Port |
|---------|-------------|------|
| `stockalert-app` | Streamlit web application | 8501 |
| `stockalert-scheduler` | Daily/weekly alert scheduler | - |
| `stockalert-hourly` | Hourly price data collector | - |
| `stockalert-futures` | Futures market scheduler | - |
| `stockalert-watchdog` | Monitors scheduler health | - |

## Troubleshooting

### Deployment Fails

1. Check GitHub Actions logs for specific errors
2. Verify SSH connectivity: `ssh -i ~/.ssh/stockalert_deploy stockalert@YOUR_SERVER`
3. Check server logs: `sudo journalctl -u stockalert-app -n 50`

### Service Won't Start

```bash
# Check service status and logs
sudo systemctl status stockalert-app
sudo journalctl -u stockalert-app -n 100

# Check if dependencies are installed
cd /opt/stockalert
sudo -u stockalert uv sync

# Verify environment file exists
ls -la /opt/stockalert/.env
```

### Database Connection Issues

```bash
# Test database connection
cd /opt/stockalert
sudo -u stockalert uv run python -c "from src.data_access.db_config import get_engine; print(get_engine().connect())"
```

### Permission Issues

```bash
# Fix ownership
sudo chown -R stockalert:stockalert /opt/stockalert

# Fix permissions
sudo chmod -R 755 /opt/stockalert
```

## Reverse Proxy Setup (Optional)

To serve the app on port 80/443 with Nginx:

```nginx
# /etc/nginx/sites-available/stockalert
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
    }
}
```

Enable the site:
```bash
sudo ln -s /etc/nginx/sites-available/stockalert /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## File Structure

```
deploy/
├── README.md                  # This file
├── README-docker-systemd.md   # Running schedulers with systemd inside Docker
├── post-deploy.sh             # Runs after rsync (restarts services)
├── health-check.sh            # Verifies deployment success
├── setup-server.sh            # Initial server setup
├── systemd/                   # Units for bare-metal / VPS
│   ├── stockalert-app.service
│   ├── stockalert-scheduler.service
│   ├── stockalert-hourly.service
│   ├── stockalert-futures.service
│   └── stockalert-watchdog.service
└── systemd-docker/            # Units for Docker (paths /app, user appuser)
    ├── *.service
    └── entrypoint.sh
```

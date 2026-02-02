# Deploy: Local vs Production

## Local (Docker)

- Postgres and Redis run in Docker (`docker-compose up`).
- Schedulers and app run on the host (or in separate containers if you add them to `docker-compose.yml`).
- Use the `.bat` launchers on Windows or `./deploy/start_*.sh` on macOS/Linux to run schedulers manually.

## Production (Debian Linux, root)

- App and schedulers run on a Debian server as **root**.
- Install systemd units and replace `/path/to/stock-market-alert-app` with the real app root (e.g. `/opt/stock-market-alert-app`).

### Install all schedulers on production

```bash
# 1. Copy units and set app root (e.g. /opt/stock-market-alert-app)
APP_ROOT=/opt/stock-market-alert-app
for f in deploy/*.service; do
  sudo cp "$f" /etc/systemd/system/
  sudo sed -i "s|/path/to/stock-market-alert-app|$APP_ROOT|g" /etc/systemd/system/$(basename "$f")
done
sudo systemctl daemon-reload

# 2. Start services (watchdog monitors the other two; start it after them or enable all on boot)
sudo systemctl start auto-scheduler
sudo systemctl start hourly-data-scheduler
sudo systemctl start futures-scheduler
sudo systemctl start scheduler-watchdog

# 3. Optional: enable on boot
sudo systemctl enable auto-scheduler hourly-data-scheduler futures-scheduler scheduler-watchdog
```

### Logs

```bash
journalctl -u auto-scheduler -f
journalctl -u hourly-data-scheduler -f
journalctl -u futures-scheduler -f
journalctl -u scheduler-watchdog -f
```

All units use `User=root` and only the app path placeholder needs to be replaced.

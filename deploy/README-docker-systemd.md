# Running Schedulers with systemd Inside Docker

This setup runs the Streamlit app and all schedulers (daily/weekly, hourly, futures, watchdog) as **systemd services inside a single Docker container**, instead of starting only Streamlit.

## What Runs Under systemd

| Service | Unit | Description |
|--------|------|-------------|
| Streamlit | `stockalert-app` | Web app on port 8501 |
| Daily/Weekly | `stockalert-scheduler` | `auto_scheduler_v2` |
| Hourly | `stockalert-hourly` | `hourly_data_scheduler` |
| Futures | `stockalert-futures` | `futures_scheduler` |
| Watchdog | `stockalert-watchdog` | Monitors and restarts schedulers |

## Requirements

- **Docker with cgroup v2** (default on recent Docker Desktop and many Linux hosts). systemd in the container needs cgroups; the compose file uses a read-only mount of `/sys/fs/cgroup`.
- **`.env`** on the host (same as for the non-systemd setup). It is mounted into the container so all systemd units can use it via `EnvironmentFile=-/app/.env`.

## Build and Run

### Option A: Docker Compose (recommended)

From the project root:

```bash
# Build the systemd image
docker compose -f docker-compose.yml -f docker-compose.systemd.yml build

# Start postgres + redis + systemd container
docker compose -f docker-compose.yml -f docker-compose.systemd.yml up -d

# View logs (systemd journal from the container)
docker compose -f docker-compose.yml -f docker-compose.systemd.yml exec streamlit journalctl -u stockalert-app -f
docker compose -f docker-compose.yml -f docker-compose.systemd.yml exec streamlit journalctl -u stockalert-scheduler -f
```

### Option B: Plain Docker

```bash
docker build -f Dockerfile.systemd -t stockalert:systemd .

# With cgroup mount (Linux with cgroup v2)
docker run -d --name stockalert-systemd \
  --tmpfs /run --tmpfs /run/lock \
  -v /sys/fs/cgroup:/sys/fs/cgroup:ro \
  -v "$(pwd)/.env:/app/.env:ro" \
  -p 8501:8501 \
  -e DATABASE_URL=postgresql://user:pass@host:5432/db \
  -e REDIS_URL=redis://host:6379 \
  stockalert:systemd
```

If the container exits immediately (e.g. systemd can’t use cgroups), try:

```bash
docker run -d --privileged --name stockalert-systemd \
  -v "$(pwd)/.env:/app/.env:ro" \
  -p 8501:8501 \
  -e DATABASE_URL=... -e REDIS_URL=... \
  stockalert:systemd
```

## Inspecting and Controlling Services

Enter the container and use `systemctl`:

```bash
docker exec -it stockalert-systemd bash   # or: ... exec streamlit bash with compose

# Status of all app services
systemctl status stockalert-app stockalert-scheduler stockalert-hourly stockalert-futures stockalert-watchdog

# Restart one service
sudo systemctl restart stockalert-scheduler

# Follow logs
journalctl -u stockalert-app -f
journalctl -u stockalert-scheduler -f
```

## Files Added/Used

- **`deploy/systemd-docker/`** – systemd unit files and entrypoint for Docker:
  - `stockalert-app.service`, `stockalert-scheduler.service`, `stockalert-hourly.service`, `stockalert-futures.service`, `stockalert-watchdog.service`
  - `entrypoint.sh` – enables units and starts systemd as PID 1
- **`Dockerfile.systemd`** – image that installs systemd and copies these units
- **`docker-compose.systemd.yml`** – override that builds and runs the systemd image with the required tmpfs and cgroup mount

## Differences from Bare-Metal systemd

- Unit files use **`User=appuser`**, **`WorkingDirectory=/app`**, and **`/app/.venv/bin`** so they match the container layout.
- **`EnvironmentFile=-/app/.env`** so the same env used by Compose is available to every service; mount `.env` at `/app/.env` (as in the compose file).
- Security options like `ProtectSystem=strict` are relaxed so the app can run in the container without extra bind mounts.

## Troubleshooting

1. **Container exits with code 1**  
   systemd often needs cgroups. Ensure `/sys/fs/cgroup` is mounted as in `docker-compose.systemd.yml`, or run with `--privileged`.

2. **Streamlit not reachable**  
   Wait for `start_period` (e.g. 90s); then check:
   - `docker exec ... systemctl status stockalert-app`
   - `docker exec ... journalctl -u stockalert-app -n 50`

3. **Schedulers not running**  
   - `docker exec ... systemctl status stockalert-scheduler stockalert-hourly stockalert-futures stockalert-watchdog`
   - Confirm `DATABASE_URL` and `REDIS_URL` in `.env` point to postgres/redis (e.g. `postgres`/`redis` hostnames when using compose).

4. **Windows / Docker Desktop**  
   cgroup mounts may behave differently. If the container fails to start, try `privileged: true` in `docker-compose.systemd.yml` for the streamlit service (see comment there).

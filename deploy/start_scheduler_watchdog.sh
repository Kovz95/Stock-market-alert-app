#!/usr/bin/env bash
# Start the scheduler watchdog (monitors daily/weekly and hourly schedulers).
# Run from project root or from deploy/; or set PROJECT_ROOT.
# Usage: ./deploy/start_scheduler_watchdog.sh
#
# Optional env: SCHEDULER_WATCHDOG_INTERVAL, SCHEDULER_HEARTBEAT_MAX_AGE,
#               WATCHDOG_ENABLE_DISCORD, WATCHDOG_DISCORD_WEBHOOK, WATCHDOG_LOG_FILE

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT

if command -v uv >/dev/null 2>&1; then
  exec uv run python scripts/maintenance/run_scheduler_watchdog.py
else
  exec python scripts/maintenance/run_scheduler_watchdog.py
fi

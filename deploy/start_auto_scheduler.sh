#!/usr/bin/env bash
# Start the daily/weekly auto scheduler (stock alerts, calendar-aware).
# Run from project root or from deploy/; or set PROJECT_ROOT.
# Usage: ./deploy/start_auto_scheduler.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT

if command -v uv >/dev/null 2>&1; then
  exec uv run python src/services/auto_scheduler_v2.py
else
  exec python src/services/auto_scheduler_v2.py
fi

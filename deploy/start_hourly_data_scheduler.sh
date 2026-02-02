#!/usr/bin/env bash
# Start the hourly data scheduler (Linux/macOS).
# Run from project root or from deploy/; or set PROJECT_ROOT.
# Usage: ./deploy/start_hourly_data_scheduler.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT

if command -v uv >/dev/null 2>&1; then
  exec uv run python src/services/hourly_data_scheduler.py
else
  exec python src/services/hourly_data_scheduler.py
fi

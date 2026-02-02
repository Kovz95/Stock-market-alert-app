#!/usr/bin/env bash
# Start the futures scheduler (Linux/macOS).
# Run from project root or from deploy/; or set PROJECT_ROOT.
# Usage: ./deploy/start_futures_scheduler.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"
export PROJECT_ROOT

if command -v uv >/dev/null 2>&1; then
  exec uv run python src/services/futures_scheduler.py
else
  exec python src/services/futures_scheduler.py
fi

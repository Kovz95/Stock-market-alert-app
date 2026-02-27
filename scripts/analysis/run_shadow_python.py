#!/usr/bin/env python3
"""
Run Python alert checks for shadow mode and write results to JSON.

Used to compare Go vs Python alert trigger results (Week 10 shadow mode).
Writes the same JSON shape as the Go scheduler so compare_shadow_results.py can diff.

Usage:
    python scripts/analysis/run_shadow_python.py --exchange NYSE --timeframe daily
    python scripts/analysis/run_shadow_python.py --exchange NASDAQ --timeframe hourly --out-dir shadow_results

Examples:
    python scripts/analysis/run_shadow_python.py --exchange NYSE --timeframe daily
    python scripts/analysis/run_shadow_python.py --exchange NASDAQ --timeframe weekly --out-dir ./shadow_results
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_access.alert_repository import list_alerts  # noqa: E402
from src.data_access.metadata_repository import fetch_stock_metadata_map  # noqa: E402
from src.services.stock_alert_checker import StockAlertChecker  # noqa: E402


def get_relevant_alerts(exchanges: list[str], timeframe_key: str) -> list[dict]:
    """Filter alerts by exchange and timeframe (same logic as run_alert_checks)."""
    metadata = fetch_stock_metadata_map()
    exchange_tickers = set()
    for symbol, info in metadata.items():
        if isinstance(info, dict) and info.get("exchange") in exchanges:
            exchange_tickers.add(symbol)

    all_alerts = list_alerts()
    relevant = []
    for alert in all_alerts:
        alert_ticker = alert.get("ticker")
        alert_exchange = alert.get("exchange")
        alert_timeframe = alert.get("timeframe", "daily")

        if alert.get("action", "on") == "off":
            continue
        if alert_exchange not in exchanges and alert_ticker not in exchange_tickers:
            continue
        if timeframe_key == "weekly" and alert_timeframe.lower() not in ("weekly", "1wk"):
            continue
        if timeframe_key == "daily" and alert_timeframe.lower() not in ("daily", "1d"):
            continue
        if timeframe_key == "hourly" and alert_timeframe.lower() not in ("hourly", "1h", "1hr"):
            continue
        relevant.append(alert)
    return relevant


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Python alert checks and write shadow JSON")
    parser.add_argument("--exchange", required=True, help="Exchange name (e.g. NYSE, NASDAQ)")
    parser.add_argument("--timeframe", required=True, choices=["daily", "weekly", "hourly"])
    parser.add_argument("--out-dir", default="shadow_results", help="Output directory for JSON file")
    args = parser.parse_args()

    exchanges = [args.exchange]
    timeframe_key = args.timeframe

    relevant = get_relevant_alerts(exchanges, timeframe_key)
    triggered_list: list[dict] = []

    def on_triggered(alert: dict, result: dict) -> None:
        triggered_list.append({
            "alert_id": str(alert.get("alert_id", result.get("alert_id", ""))),
            "ticker": result.get("ticker", alert.get("ticker", "")),
        })

    checker = StockAlertChecker()
    stats = checker.check_alerts(
        relevant,
        timeframe_filter=timeframe_key,
        max_workers=1,
        on_triggered=on_triggered,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_time = datetime.now(timezone.utc)
    ts = run_time.strftime("%Y%m%d_%H%M%S")
    filename = f"python_{args.exchange}_{timeframe_key}_{ts}.json"
    path = out_dir / filename

    payload = {
        "exchange": args.exchange,
        "timeframe": timeframe_key,
        "utc_timestamp": run_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "triggered": triggered_list,
        "total": stats.get("total", 0),
        "triggered_count": stats.get("triggered", 0),
        "errors": stats.get("errors", 0),
        "skipped": stats.get("skipped", 0),
        "no_data": stats.get("no_data", 0),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {path} ({len(triggered_list)} triggered)")
    sys.exit(0)


if __name__ == "__main__":
    main()

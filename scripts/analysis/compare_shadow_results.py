#!/usr/bin/env python3
"""
Compare Go vs Python shadow alert trigger results.

Reads two JSON files (or the latest file from each prefix in a directory)
and reports differences in triggered alert sets and stats.

Usage:
    python scripts/analysis/compare_shadow_results.py go_NYSE_daily_20260101_120000.json python_NYSE_daily_20260101_120100.json
    python scripts/analysis/compare_shadow_results.py --dir shadow_results --exchange NYSE --timeframe daily

Examples:
    python scripts/analysis/compare_shadow_results.py --dir shadow_results --exchange NYSE --timeframe daily
"""

import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")


def load_shadow(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def triggered_set(data: dict) -> set[tuple[str, str]]:
    """Return set of (alert_id, ticker) for comparison."""
    out = set()
    for r in data.get("triggered", []):
        aid = r.get("alert_id", "")
        ticker = r.get("ticker", "")
        out.add((str(aid).lower(), str(ticker).upper()))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Go and Python shadow result JSON files")
    parser.add_argument("go_file", nargs="?", help="Go shadow JSON file")
    parser.add_argument("python_file", nargs="?", help="Python shadow JSON file")
    parser.add_argument("--dir", default="shadow_results", help="Directory containing shadow JSON files")
    parser.add_argument("--exchange", help="Filter: exchange name (with --dir)")
    parser.add_argument("--timeframe", help="Filter: timeframe (with --dir)")
    args = parser.parse_args()

    if args.go_file and args.python_file:
        go_path = Path(args.go_file)
        python_path = Path(args.python_file)
        if not go_path.is_file():
            print(f"Error: not a file: {go_path}", file=sys.stderr)
            sys.exit(1)
        if not python_path.is_file():
            print(f"Error: not a file: {python_path}", file=sys.stderr)
            sys.exit(1)
    elif args.dir:
        d = Path(args.dir)
        if not d.is_dir():
            print(f"Error: not a directory: {d}", file=sys.stderr)
            sys.exit(1)
        prefix_go = "go_"
        prefix_py = "python_"
        if args.exchange:
            prefix_go += f"{args.exchange}_"
            prefix_py += f"{args.exchange}_"
        if args.timeframe:
            prefix_go += f"{args.timeframe}_"
            prefix_py += f"{args.timeframe}_"
        go_files = sorted(d.glob(f"{prefix_go}*.json"), reverse=True)
        py_files = sorted(d.glob(f"{prefix_py}*.json"), reverse=True)
        if not go_files:
            print(f"No Go shadow files found in {d} matching {prefix_go}*.json", file=sys.stderr)
            sys.exit(1)
        if not py_files:
            print(f"No Python shadow files found in {d} matching {prefix_py}*.json", file=sys.stderr)
            sys.exit(1)
        go_path = go_files[0]
        python_path = py_files[0]
        print(f"Using Go:    {go_path}")
        print(f"Using Python: {python_path}")
    else:
        parser.print_help()
        sys.exit(1)

    go_data = load_shadow(go_path)
    py_data = load_shadow(python_path)

    go_set = triggered_set(go_data)
    py_set = triggered_set(py_data)

    only_go = go_set - py_set
    only_py = py_set - go_set
    common = go_set & py_set

    print()
    print("=== Stats ===")
    print(f"  Go:     total={go_data.get('total')} triggered={go_data.get('triggered_count')} errors={go_data.get('errors')} skipped={go_data.get('skipped')} no_data={go_data.get('no_data')}")
    print(f"  Python: total={py_data.get('total')} triggered={py_data.get('triggered_count')} errors={py_data.get('errors')} skipped={py_data.get('skipped')} no_data={py_data.get('no_data')}")

    print()
    print("=== Triggered set comparison ===")
    print(f"  In both:     {len(common)}")
    print(f"  Only in Go:   {len(only_go)}")
    print(f"  Only in Python: {len(only_py)}")

    if only_go:
        print()
        print("  Triggered only in Go:")
        for aid, ticker in sorted(only_go):
            print(f"    {aid} {ticker}")
    if only_py:
        print()
        print("  Triggered only in Python:")
        for aid, ticker in sorted(only_py):
            print(f"    {aid} {ticker}")

    if only_go or only_py:
        sys.exit(1)
    print()
    print("No differences in triggered sets.")
    sys.exit(0)


if __name__ == "__main__":
    main()

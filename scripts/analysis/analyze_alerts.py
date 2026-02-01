#!/usr/bin/env python3
"""
Alert Analysis Script
Comprehensive analysis of alert evaluations, triggers, and errors
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

import pandas as pd
from alert_audit_logger import AlertAuditLogger


def get_evaluation_summary(hours=24):
    """Get comprehensive evaluation summary"""
    print("\n" + "=" * 70)
    print(f"EVALUATION SUMMARY (Last {hours} hours)")
    print("=" * 70)
    
    try:
        logger = AlertAuditLogger()
        cutoff = datetime.now() - timedelta(hours=hours)
        
        query = """
        SELECT 
            COUNT(*) as total_checks,
            SUM(CASE WHEN alert_triggered = true THEN 1 ELSE 0 END) as triggered,
            SUM(CASE WHEN error_message IS NOT NULL THEN 1 ELSE 0 END) as errors,
            SUM(CASE WHEN price_data_pulled = false THEN 1 ELSE 0 END) as no_data,
            SUM(CASE WHEN conditions_evaluated = true THEN 1 ELSE 0 END) as evaluated,
            COUNT(DISTINCT ticker) as unique_tickers,
            COUNT(DISTINCT alert_id) as unique_alerts
        FROM alert_audits
        WHERE timestamp >= %s
        """
        
        with logger._connection() as conn:
            result = pd.read_sql_query(query, conn, params=[cutoff.isoformat()])
        
        if result.empty or result.iloc[0]['total_checks'] == 0:
            print(f"\nNo evaluations found in the last {hours} hours")
            return None
        
        row = result.iloc[0]
        print(f"\nOverall Statistics:")
        print(f"  Total Checks: {row['total_checks']}")
        print(f"  Unique Alerts: {row['unique_alerts']}")
        print(f"  Unique Tickers: {row['unique_tickers']}")
        print(f"  Conditions Evaluated: {row['evaluated']}")
        print(f"  Triggered: {row['triggered']}")
        print(f"  Errors: {row['errors']}")
        print(f"  No Data: {row['no_data']}")
        
        if row['total_checks'] > 0:
            success_rate = ((row['total_checks'] - row['errors']) / row['total_checks']) * 100
            trigger_rate = (row['triggered'] / row['total_checks']) * 100
            data_rate = ((row['total_checks'] - row['no_data']) / row['total_checks']) * 100
            
            print(f"\nRates:")
            print(f"  Success Rate: {success_rate:.1f}%")
            print(f"  Data Availability: {data_rate:.1f}%")
            print(f"  Trigger Rate: {trigger_rate:.1f}%")
        
        return result
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_recently_triggered_alerts(hours=24):
    """Get alerts that were triggered recently"""
    print("\n" + "=" * 70)
    print(f"RECENTLY TRIGGERED ALERTS (Last {hours} hours)")
    print("=" * 70)
    
    try:
        logger = AlertAuditLogger()
        cutoff = datetime.now() - timedelta(hours=hours)
        
        query = """
        SELECT 
            timestamp,
            alert_id,
            ticker,
            stock_name,
            trigger_reason,
            timeframe
        FROM alert_audits
        WHERE timestamp >= %s
        AND alert_triggered = true
        ORDER BY timestamp DESC
        """
        
        with logger._connection() as conn:
            triggered = pd.read_sql_query(query, conn, params=[cutoff.isoformat()])
        
        if triggered.empty:
            print(f"\nNo triggered alerts in the last {hours} hours")
            return None
        
        print(f"\nFound {len(triggered)} triggered alert(s):")
        print()
        
        for idx, row in triggered.iterrows():
            print(f"  [{row['timestamp']}]")
            print(f"    Alert ID: {row['alert_id']}")
            print(f"    Ticker: {row['ticker']}")
            print(f"    Name: {row['stock_name']}")
            print(f"    Timeframe: {row['timeframe']}")
            if row['trigger_reason']:
                print(f"    Reason: {row['trigger_reason']}")
            print()
        
        return triggered
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_failed_checks(hours=24):
    """Get checks that failed with errors"""
    print("\n" + "=" * 70)
    print(f"FAILED CHECKS (Last {hours} hours)")
    print("=" * 70)
    
    try:
        logger = AlertAuditLogger()
        cutoff = datetime.now() - timedelta(hours=hours)
        
        query = """
        SELECT 
            timestamp,
            alert_id,
            ticker,
            stock_name,
            error_message,
            evaluation_type
        FROM alert_audits
        WHERE timestamp >= %s
        AND error_message IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT 20
        """
        
        with logger._connection() as conn:
            errors = pd.read_sql_query(query, conn, params=[cutoff.isoformat()])
        
        if errors.empty:
            print(f"\n✅ No failed checks in the last {hours} hours")
            return None
        
        print(f"\n❌ Found {len(errors)} failed check(s):")
        print()
        
        for idx, row in errors.iterrows():
            print(f"  [{row['timestamp']}]")
            print(f"    Ticker: {row['ticker']}")
            print(f"    Alert ID: {row['alert_id']}")
            print(f"    Error: {row['error_message']}")
            print()
        
        return errors
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_ticker_performance(ticker, days=7):
    """Analyze alert performance for a specific ticker"""
    print("\n" + "=" * 70)
    print(f"TICKER ANALYSIS: {ticker} (Last {days} days)")
    print("=" * 70)
    
    try:
        logger = AlertAuditLogger()
        cutoff = datetime.now() - timedelta(days=days)
        
        query = """
        SELECT 
            timestamp,
            alert_id,
            stock_name,
            alert_triggered,
            error_message,
            price_data_pulled,
            conditions_evaluated,
            execution_time_ms
        FROM alert_audits
        WHERE timestamp >= %s
        AND ticker = %s
        ORDER BY timestamp DESC
        """
        
        with logger._connection() as conn:
            checks = pd.read_sql_query(query, conn, params=[cutoff.isoformat(), ticker])
        
        if checks.empty:
            print(f"\nNo checks found for {ticker} in the last {days} days")
            return None
        
        total = len(checks)
        triggered = checks['alert_triggered'].sum()
        errors = checks['error_message'].notna().sum()
        no_data = (~checks['price_data_pulled']).sum()
        
        print(f"\nStatistics for {ticker}:")
        print(f"  Total Checks: {total}")
        print(f"  Triggered: {triggered}")
        print(f"  Errors: {errors}")
        print(f"  No Data: {no_data}")
        
        if total > 0:
            print(f"  Trigger Rate: {(triggered / total * 100):.1f}%")
        
        # Show execution times
        valid_times = checks['execution_time_ms'].dropna()
        if not valid_times.empty:
            print(f"\nExecution Times:")
            print(f"  Average: {valid_times.mean():.1f}ms")
            print(f"  Min: {valid_times.min():.1f}ms")
            print(f"  Max: {valid_times.max():.1f}ms")
        
        # Show recent triggers
        triggered_checks = checks[checks['alert_triggered']]
        if not triggered_checks.empty:
            print(f"\nRecent Triggers:")
            for idx, row in triggered_checks.head(5).iterrows():
                print(f"  - {row['timestamp']}: {row['stock_name']}")
        
        return checks
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_alert_performance(alert_id, days=7):
    """Analyze performance of a specific alert"""
    print("\n" + "=" * 70)
    print(f"ALERT ANALYSIS: {alert_id} (Last {days} days)")
    print("=" * 70)
    
    try:
        logger = AlertAuditLogger()
        cutoff = datetime.now() - timedelta(days=days)
        
        query = """
        SELECT 
            timestamp,
            ticker,
            stock_name,
            alert_triggered,
            error_message,
            price_data_pulled,
            conditions_evaluated,
            trigger_reason,
            execution_time_ms
        FROM alert_audits
        WHERE timestamp >= %s
        AND alert_id = %s
        ORDER BY timestamp DESC
        """
        
        with logger._connection() as conn:
            checks = pd.read_sql_query(query, conn, params=[cutoff.isoformat(), alert_id])
        
        if checks.empty:
            print(f"\nNo checks found for alert {alert_id}")
            print("This could mean:")
            print("  - Alert hasn't been checked yet")
            print("  - Alert ID is incorrect")
            print("  - Alert was created after the time period")
            return None
        
        # Get alert details from first row
        ticker = checks.iloc[0]['ticker']
        name = checks.iloc[0]['stock_name']
        
        print(f"\nAlert Details:")
        print(f"  Name: {name}")
        print(f"  Ticker: {ticker}")
        print(f"  Alert ID: {alert_id}")
        
        # Statistics
        total = len(checks)
        triggered = checks['alert_triggered'].sum()
        errors = checks['error_message'].notna().sum()
        no_data = (~checks['price_data_pulled']).sum()
        evaluated = checks['conditions_evaluated'].sum()
        
        print(f"\nPerformance Statistics:")
        print(f"  Total Checks: {total}")
        print(f"  Conditions Evaluated: {evaluated}")
        print(f"  Triggered: {triggered}")
        print(f"  Errors: {errors}")
        print(f"  No Data: {no_data}")
        
        if total > 0:
            print(f"\nRates:")
            print(f"  Evaluation Rate: {(evaluated / total * 100):.1f}%")
            print(f"  Trigger Rate: {(triggered / total * 100):.1f}%")
            print(f"  Error Rate: {(errors / total * 100):.1f}%")
        
        # Show trigger history
        triggered_checks = checks[checks['alert_triggered']]
        if not triggered_checks.empty:
            print(f"\nTrigger History ({len(triggered_checks)} times):")
            for idx, row in triggered_checks.head(10).iterrows():
                print(f"  [{row['timestamp']}]")
                if row['trigger_reason']:
                    print(f"    Reason: {row['trigger_reason']}")
        else:
            print(f"\nTrigger History: Never triggered in this period")
        
        # Show errors
        error_checks = checks[checks['error_message'].notna()]
        if not error_checks.empty:
            print(f"\nErrors ({len(error_checks)} times):")
            for idx, row in error_checks.head(5).iterrows():
                print(f"  [{row['timestamp']}] {row['error_message']}")
        
        return checks
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_timeframe_breakdown(hours=24):
    """Get breakdown by timeframe"""
    print("\n" + "=" * 70)
    print(f"TIMEFRAME BREAKDOWN (Last {hours} hours)")
    print("=" * 70)
    
    try:
        logger = AlertAuditLogger()
        cutoff = datetime.now() - timedelta(hours=hours)
        
        query = """
        SELECT 
            timeframe,
            COUNT(*) as checks,
            SUM(CASE WHEN alert_triggered = true THEN 1 ELSE 0 END) as triggered,
            SUM(CASE WHEN error_message IS NOT NULL THEN 1 ELSE 0 END) as errors
        FROM alert_audits
        WHERE timestamp >= %s
        GROUP BY timeframe
        ORDER BY checks DESC
        """
        
        with logger._connection() as conn:
            breakdown = pd.read_sql_query(query, conn, params=[cutoff.isoformat()])
        
        if breakdown.empty:
            print(f"\nNo data for the last {hours} hours")
            return None
        
        print(f"\nBreakdown by timeframe:")
        print()
        print(f"  {'Timeframe':<12} {'Checks':>8} {'Triggered':>10} {'Errors':>8}")
        print(f"  {'-'*12} {'-'*8} {'-'*10} {'-'*8}")
        
        for idx, row in breakdown.iterrows():
            timeframe = row['timeframe'] or 'Unknown'
            print(f"  {timeframe:<12} {row['checks']:>8} {row['triggered']:>10} {row['errors']:>8}")
        
        return breakdown
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main analysis function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze alert system performance")
    parser.add_argument('--summary', type=int, default=24,
                       help='Show evaluation summary (hours, default: 24)')
    parser.add_argument('--triggered', type=int,
                       help='Show recently triggered alerts (hours)')
    parser.add_argument('--errors', type=int,
                       help='Show failed checks (hours)')
    parser.add_argument('--ticker', type=str,
                       help='Analyze specific ticker')
    parser.add_argument('--alert', type=str,
                       help='Analyze specific alert ID')
    parser.add_argument('--days', type=int, default=7,
                       help='Days of history (default: 7)')
    parser.add_argument('--timeframe', type=int,
                       help='Show timeframe breakdown (hours)')
    parser.add_argument('--all', action='store_true',
                       help='Show all analysis')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("ALERT SYSTEM ANALYSIS")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.all:
        # Show everything
        get_evaluation_summary(args.summary)
        get_timeframe_breakdown(args.summary)
        get_recently_triggered_alerts(args.summary)
        get_failed_checks(args.summary)
    elif args.ticker:
        get_ticker_performance(args.ticker, args.days)
    elif args.alert:
        get_alert_performance(args.alert, args.days)
    else:
        # Default analysis
        if args.summary:
            get_evaluation_summary(args.summary)
        if args.timeframe:
            get_timeframe_breakdown(args.timeframe)
        if args.triggered:
            get_recently_triggered_alerts(args.triggered)
        if args.errors:
            get_failed_checks(args.errors)
        
        # If nothing specified, show summary
        if not any([args.summary, args.triggered, args.errors, args.timeframe]):
            get_evaluation_summary(24)
            get_timeframe_breakdown(24)
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Alert Monitoring Script
Monitor audit logs, recent checks, and alert history
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import json

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

import pandas as pd
from src.services.alert_audit_logger import AlertAuditLogger
from src.data_access.document_store import load_document


def monitor_recent_checks(hours=1):
    """Monitor recent alert checks from audit log"""
    print("\n" + "=" * 70)
    print(f"RECENT ALERT CHECKS (Last {hours} hour(s))")
    print("=" * 70)
    
    try:
        logger = AlertAuditLogger()
        
        # Get recent checks
        cutoff = datetime.now() - timedelta(hours=hours)
        
        # Query using raw SQL with correct PostgreSQL column names
        query = """
        SELECT 
            timestamp,
            alert_id,
            ticker,
            stock_name as alert_name,
            alert_triggered,
            error_message,
            conditions_evaluated,
            price_data_pulled
        FROM alert_audits
        WHERE timestamp >= %s
        ORDER BY timestamp DESC
        LIMIT 50
        """
        
        with logger._connection() as conn:
            recent = pd.read_sql_query(query, conn, params=[cutoff.isoformat()])
        
        if recent.empty:
            print(f"\nNo alert checks found in the last {hours} hour(s)")
            print("This is normal if:")
            print("  - Scheduler hasn't run yet today")
            print("  - No alerts are scheduled for this time")
            print("  - Alerts only trigger during market hours")
        else:
            print(f"\nFound {len(recent)} alert check(s):")
            print()
            
            # Show summary
            triggered_count = recent['alert_triggered'].sum()
            error_count = recent['error_message'].notna().sum()
            
            print(f"Summary:")
            print(f"  Total Checks: {len(recent)}")
            print(f"  Triggered: {triggered_count}")
            print(f"  Errors: {error_count}")
            print(f"  Success Rate: {((len(recent) - error_count) / len(recent) * 100):.1f}%")
            
            # Show details
            print(f"\nRecent checks:")
            for idx, row in recent.head(10).iterrows():
                timestamp = row['timestamp']
                ticker = row['ticker']
                triggered = 'YES' if row['alert_triggered'] else 'NO'
                status = 'ERROR' if pd.notna(row['error_message']) else 'OK'
                
                print(f"  [{timestamp}] {ticker:8} | Triggered: {triggered:3} | Status: {status}")
            
            if len(recent) > 10:
                print(f"  ... and {len(recent) - 10} more")
        
        return recent
        
    except Exception as e:
        print(f"\nError querying audit log: {e}")
        print("\nThis might mean:")
        print("  - PostgreSQL is not running")
        print("  - Database connection issue")
        print("  - Audit table doesn't exist yet")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def monitor_alert_history(alert_id, days=7):
    """Monitor history for a specific alert"""
    print("\n" + "=" * 70)
    print(f"ALERT HISTORY: {alert_id} (Last {days} days)")
    print("=" * 70)
    
    try:
        logger = AlertAuditLogger()
        
        cutoff = datetime.now() - timedelta(days=days)
        
        query = """
        SELECT 
            timestamp,
            alert_triggered,
            conditions_evaluated,
            price_data_pulled,
            error_message
        FROM alert_audits
        WHERE alert_id = %s
        AND timestamp >= %s
        ORDER BY timestamp DESC
        LIMIT 50
        """
        
        with logger._connection() as conn:
            history = pd.read_sql_query(query, conn, params=[alert_id, cutoff.isoformat()])
        
        if history.empty:
            print(f"\nNo history found for alert {alert_id}")
            print("This could mean:")
            print("  - Alert was never checked")
            print("  - Alert ID is incorrect")
            print("  - Alert is new (created recently)")
        else:
            print(f"\nFound {len(history)} check(s) for this alert:")
            
            triggered_count = history['alert_triggered'].sum()
            error_count = history['error_message'].notna().sum()
            
            print(f"\nSummary:")
            print(f"  Total Checks: {len(history)}")
            print(f"  Times Triggered: {triggered_count}")
            print(f"  Errors: {error_count}")
            
            if triggered_count > 0:
                print(f"\nTriggered on:")
                for idx, row in history[history['alert_triggered']].iterrows():
                    print(f"  - {row['timestamp']}")
            
            if error_count > 0:
                print(f"\nErrors encountered:")
                for idx, row in history[history['error_message'].notna()].head(5).iterrows():
                    print(f"  [{row['timestamp']}] {row['error_message']}")
        
        return history
        
    except Exception as e:
        print(f"\nError querying alert history: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def check_scheduler_status():
    """Check current scheduler status from document store"""
    print("\n" + "=" * 70)
    print("SCHEDULER STATUS (From Document Store)")
    print("=" * 70)
    
    try:
        status = load_document('scheduler_status')
        
        if not status:
            print("\nNo scheduler status found")
            print("This means the scheduler hasn't been started yet")
            return None
        
        print(f"\nStatus: {status.get('status', 'Unknown')}")
        print(f"Heartbeat: {status.get('heartbeat', 'N/A')}")
        
        current_job = status.get('current_job')
        if current_job:
            print(f"\nCurrent Job:")
            print(f"  ID: {current_job.get('id', 'N/A')}")
            print(f"  Exchange: {current_job.get('exchange', 'N/A')}")
            print(f"  Type: {current_job.get('job_type', 'N/A')}")
            print(f"  Started: {current_job.get('started', 'N/A')}")
        else:
            print(f"\nCurrent Job: None (idle)")
        
        last_run = status.get('last_run')
        if last_run:
            print(f"\nLast Run:")
            print(f"  Job ID: {last_run.get('job_id', 'N/A')}")
            print(f"  Exchange: {last_run.get('exchange', 'N/A')}")
            print(f"  Completed: {last_run.get('completed_at', 'N/A')}")
            print(f"  Duration: {last_run.get('duration_seconds', 'N/A')}s")
        
        last_result = status.get('last_result')
        if last_result:
            print(f"\nLast Result:")
            price_stats = last_result.get('price_stats', {})
            alert_stats = last_result.get('alert_stats', {})
            
            if isinstance(price_stats, dict):
                print(f"  Prices: {price_stats.get('updated', 0)} updated, "
                      f"{price_stats.get('failed', 0)} failed")
            
            if isinstance(alert_stats, dict):
                print(f"  Alerts: {alert_stats.get('total', 0)} checked, "
                      f"{alert_stats.get('triggered', 0)} triggered, "
                      f"{alert_stats.get('errors', 0)} errors")
        
        last_error = status.get('last_error')
        if last_error:
            print(f"\nLast Error:")
            print(f"  Time: {last_error.get('time', 'N/A')}")
            print(f"  Message: {last_error.get('message', 'N/A')}")
        
        return status
        
    except Exception as e:
        print(f"\nError loading scheduler status: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_evaluation_summary(hours=24):
    """Get summary of alert evaluations"""
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
            COUNT(DISTINCT ticker) as unique_tickers,
            COUNT(DISTINCT alert_id) as unique_alerts
        FROM alert_audits
        WHERE timestamp >= %s
        """
        
        with logger._connection() as conn:
            result = pd.read_sql_query(query, conn, params=[cutoff.isoformat()])
        
        if result.empty or result.iloc[0]['total_checks'] == 0:
            print(f"\nNo evaluations found in the last {hours} hours")
        else:
            row = result.iloc[0]
            print(f"\nTotal Checks: {row['total_checks']}")
            print(f"Unique Alerts: {row['unique_alerts']}")
            print(f"Unique Tickers: {row['unique_tickers']}")
            print(f"Triggered: {row['triggered']}")
            print(f"Errors: {row['errors']}")
            print(f"No Data: {row['no_data']}")
            
            if row['total_checks'] > 0:
                success_rate = ((row['total_checks'] - row['errors']) / row['total_checks']) * 100
                trigger_rate = (row['triggered'] / row['total_checks']) * 100
                print(f"\nSuccess Rate: {success_rate:.1f}%")
                print(f"Trigger Rate: {trigger_rate:.1f}%")
        
        return result
        
    except Exception as e:
        print(f"\nError getting evaluation summary: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def main():
    """Main monitoring function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor alert system activity")
    parser.add_argument('--recent', type=int, default=1, 
                       help='Show recent checks (hours)')
    parser.add_argument('--alert', type=str,
                       help='Show history for specific alert ID')
    parser.add_argument('--alert-days', type=int, default=7,
                       help='Days of history for alert (default: 7)')
    parser.add_argument('--summary', type=int, default=24,
                       help='Show evaluation summary (hours)')
    parser.add_argument('--all', action='store_true',
                       help='Show all monitoring information')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("ALERT SYSTEM MONITORING")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.all or not any([args.alert, args.summary]):
        # Default: show everything
        check_scheduler_status()
        get_evaluation_summary(args.summary)
        monitor_recent_checks(args.recent)
    else:
        # Specific queries
        if args.alert:
            monitor_alert_history(args.alert, args.alert_days)
        
        if args.summary:
            get_evaluation_summary(args.summary)
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

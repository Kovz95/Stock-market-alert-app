#!/usr/bin/env python3
"""
Alert System Monitoring Dashboard
Real-time monitoring of scheduler health and alert activity
"""

import sys
import os
import time
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
from auto_scheduler_v2 import get_scheduler_info, is_scheduler_running
from alert_audit_logger import AlertAuditLogger
from data_access.document_store import load_document


def clear_screen():
    """Clear the console screen"""
    os.system('cls' if os.name == 'nt' else 'clear')


def show_header():
    """Show dashboard header"""
    print("\n" + "=" * 80)
    print("STOCK ALERT SYSTEM - MONITORING DASHBOARD")
    print("=" * 80)
    print(f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


def show_scheduler_status():
    """Show scheduler status section"""
    print("\n" + "─" * 80)
    print("SCHEDULER STATUS")
    print("─" * 80)
    
    try:
        # Check if running
        running = is_scheduler_running()
        
        if running:
            print("Status: ✅ RUNNING")
        else:
            print("Status: ❌ NOT RUNNING")
            print("  To start: python auto_scheduler_v2.py")
            return
        
        # Get detailed info
        info = get_scheduler_info()
        
        if not info:
            print("  No status information available")
            return
        
        # Heartbeat
        heartbeat = info.get('heartbeat')
        if heartbeat:
            try:
                hb_time = datetime.fromisoformat(heartbeat.replace('Z', '+00:00'))
                age = (datetime.now(hb_time.tzinfo) - hb_time).total_seconds()
                if age < 120:
                    print(f"Heartbeat: ✅ Fresh ({int(age)}s ago)")
                else:
                    print(f"Heartbeat: ⚠️  Stale ({int(age)}s ago)")
            except:
                print(f"Heartbeat: {heartbeat}")
        
        # Current job
        current_job = info.get('current_job')
        if current_job:
            print(f"Current Job: 🔄 {current_job.get('exchange', 'N/A')} ({current_job.get('job_type', 'N/A')})")
            print(f"  Started: {current_job.get('started', 'N/A')}")
        else:
            print("Current Job: ⏸️  Idle")
        
        # Last run
        last_run = info.get('last_run')
        if last_run:
            print(f"Last Run: {last_run.get('exchange', 'N/A')} at {last_run.get('completed_at', 'N/A')}")
        
        # Job counts
        daily_jobs = info.get('total_daily_jobs', 0)
        weekly_jobs = info.get('total_weekly_jobs', 0)
        print(f"Scheduled Jobs: {daily_jobs} daily, {weekly_jobs} weekly")
        
        # Next run
        next_run = info.get('next_run')
        if next_run:
            try:
                next_time = datetime.fromisoformat(next_run.replace('Z', '+00:00'))
                until_next = (next_time - datetime.now(next_time.tzinfo)).total_seconds()
                if until_next > 0:
                    hours = int(until_next // 3600)
                    minutes = int((until_next % 3600) // 60)
                    print(f"Next Run: {next_run} (in {hours}h {minutes}m)")
                else:
                    print(f"Next Run: {next_run} (overdue)")
            except:
                print(f"Next Run: {next_run}")
        
    except Exception as e:
        print(f"Error checking scheduler: {e}")


def show_recent_activity(hours=1):
    """Show recent alert activity"""
    print("\n" + "─" * 80)
    print(f"RECENT ACTIVITY (Last {hours} hour(s))")
    print("─" * 80)
    
    try:
        logger = AlertAuditLogger()
        cutoff = datetime.now() - timedelta(hours=hours)
        
        query = """
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN alert_triggered = true THEN 1 ELSE 0 END) as triggered,
            SUM(CASE WHEN error_message IS NOT NULL THEN 1 ELSE 0 END) as errors,
            SUM(CASE WHEN price_data_pulled = false THEN 1 ELSE 0 END) as no_data
        FROM alert_audits
        WHERE timestamp >= %s
        """
        
        with logger._connection() as conn:
            result = pd.read_sql_query(query, conn, params=[cutoff.isoformat()])
        
        if result.empty or result.iloc[0]['total'] == 0:
            print("No activity in the last hour")
            print("  This is normal if:")
            print("    - No alerts scheduled for this time")
            print("    - Markets are closed")
            print("    - Scheduler hasn't run yet")
        else:
            row = result.iloc[0]
            print(f"Total Checks: {row['total']}")
            print(f"Triggered: {row['triggered']}")
            print(f"Errors: {row['errors']}")
            print(f"No Data: {row['no_data']}")
            
            if row['total'] > 0:
                trigger_rate = (row['triggered'] / row['total']) * 100
                error_rate = (row['errors'] / row['total']) * 100
                print(f"Trigger Rate: {trigger_rate:.1f}%")
                print(f"Error Rate: {error_rate:.1f}%")
        
    except Exception as e:
        print(f"Error checking activity: {e}")


def show_last_triggered(limit=5):
    """Show last triggered alerts"""
    print("\n" + "─" * 80)
    print(f"RECENTLY TRIGGERED ALERTS (Last {limit})")
    print("─" * 80)
    
    try:
        logger = AlertAuditLogger()
        
        query = """
        SELECT 
            timestamp,
            ticker,
            stock_name,
            timeframe
        FROM alert_audits
        WHERE alert_triggered = true
        ORDER BY timestamp DESC
        LIMIT %s
        """
        
        with logger._connection() as conn:
            triggered = pd.read_sql_query(query, conn, params=[limit])
        
        if triggered.empty:
            print("No recently triggered alerts")
        else:
            for idx, row in triggered.iterrows():
                timestamp = row['timestamp']
                if isinstance(timestamp, str):
                    try:
                        ts_obj = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        age = (datetime.now(ts_obj.tzinfo) - ts_obj).total_seconds()
                        age_str = f"{int(age/3600)}h ago" if age > 3600 else f"{int(age/60)}m ago"
                        timestamp_display = f"{ts_obj.strftime('%H:%M:%S')} ({age_str})"
                    except:
                        timestamp_display = str(timestamp)
                else:
                    timestamp_display = str(timestamp)
                
                print(f"  [{timestamp_display}] {row['ticker']:8} - {row['stock_name']}")
        
    except Exception as e:
        print(f"Error checking triggered alerts: {e}")


def show_24h_summary():
    """Show 24-hour summary"""
    print("\n" + "─" * 80)
    print("24-HOUR SUMMARY")
    print("─" * 80)
    
    try:
        logger = AlertAuditLogger()
        cutoff = datetime.now() - timedelta(hours=24)
        
        query = """
        SELECT 
            COUNT(*) as total_checks,
            SUM(CASE WHEN alert_triggered = true THEN 1 ELSE 0 END) as triggered,
            SUM(CASE WHEN error_message IS NOT NULL THEN 1 ELSE 0 END) as errors,
            COUNT(DISTINCT ticker) as unique_tickers,
            COUNT(DISTINCT alert_id) as unique_alerts
        FROM alert_audits
        WHERE timestamp >= %s
        """
        
        with logger._connection() as conn:
            result = pd.read_sql_query(query, conn, params=[cutoff.isoformat()])
        
        if result.empty or result.iloc[0]['total_checks'] == 0:
            print("No checks in the last 24 hours")
        else:
            row = result.iloc[0]
            print(f"Total Checks: {row['total_checks']:,}")
            print(f"Unique Alerts: {row['unique_alerts']:,}")
            print(f"Unique Tickers: {row['unique_tickers']:,}")
            print(f"Triggered: {row['triggered']:,}")
            print(f"Errors: {row['errors']:,}")
            
            if row['total_checks'] > 0:
                success_rate = ((row['total_checks'] - row['errors']) / row['total_checks']) * 100
                trigger_rate = (row['triggered'] / row['total_checks']) * 100
                print(f"Success Rate: {success_rate:.1f}%")
                print(f"Trigger Rate: {trigger_rate:.1f}%")
        
    except Exception as e:
        print(f"Error getting summary: {e}")


def show_errors(hours=24):
    """Show recent errors"""
    print("\n" + "─" * 80)
    print(f"RECENT ERRORS (Last {hours} hours)")
    print("─" * 80)
    
    try:
        logger = AlertAuditLogger()
        cutoff = datetime.now() - timedelta(hours=hours)
        
        query = """
        SELECT 
            timestamp,
            ticker,
            error_message
        FROM alert_audits
        WHERE timestamp >= %s
        AND error_message IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT 5
        """
        
        with logger._connection() as conn:
            errors = pd.read_sql_query(query, conn, params=[cutoff.isoformat()])
        
        if errors.empty:
            print("✅ No errors in the last 24 hours")
        else:
            print(f"⚠️  Found {len(errors)} error(s):")
            for idx, row in errors.iterrows():
                print(f"  [{row['timestamp']}] {row['ticker']}: {row['error_message'][:50]}")
        
    except Exception as e:
        print(f"Error checking errors: {e}")


def show_footer():
    """Show dashboard footer"""
    print("\n" + "=" * 80)
    print("Press Ctrl+C to exit | Refreshes every 5 minutes")
    print("=" * 80)


def show_dashboard():
    """Show complete dashboard"""
    clear_screen()
    show_header()
    show_scheduler_status()
    show_recent_activity(hours=1)
    show_last_triggered(limit=5)
    show_24h_summary()
    show_errors(hours=24)
    show_footer()


def main():
    """Main dashboard function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Alert system monitoring dashboard")
    parser.add_argument('--once', action='store_true',
                       help='Run once and exit (no continuous monitoring)')
    parser.add_argument('--interval', type=int, default=300,
                       help='Refresh interval in seconds (default: 300)')
    
    args = parser.parse_args()
    
    if args.once:
        # Single run
        show_dashboard()
    else:
        # Continuous monitoring
        try:
            while True:
                show_dashboard()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user")


if __name__ == "__main__":
    main()

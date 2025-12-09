#!/usr/bin/env python3
"""
View scheduler and alert logs
"""

import pandas as pd
from datetime import datetime, timedelta
import sys
from db_config import db_config

def view_recent_alerts(hours=1):
    """View recent alert checks from the audit database"""
    try:
        with db_config.connection(role="alerts") as conn:

        # Calculate time threshold
        time_threshold = datetime.now() - timedelta(hours=hours)
        time_str = time_threshold.strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"\nALERT CHECKS IN LAST {hours} HOUR(S)")
        print("=" * 80)
        
        # Get summary stats
        query = """
        SELECT 
            COUNT(*) as total_checks,
            SUM(CASE WHEN alert_triggered = 1 THEN 1 ELSE 0 END) as triggered,
            SUM(CASE WHEN error_message IS NOT NULL THEN 1 ELSE 0 END) as errors,
            AVG(execution_time_ms) as avg_time_ms,
            COUNT(DISTINCT exchange) as exchanges_checked
        FROM alert_audits 
        WHERE timestamp > %s
        """
        
        df = pd.read_sql_query(query, conn, params=(time_str,))
        
        if not df.empty and df.iloc[0]['total_checks'] > 0:
            stats = df.iloc[0]
            print(f"Total checks: {stats['total_checks']}")
            print(f"Triggered: {stats['triggered']}")
            print(f"Errors: {stats['errors']}")
            print(f"Exchanges: {stats['exchanges_checked']}")
            print(f"Avg time: {stats['avg_time_ms']:.0f}ms")
            
            # Get recent triggered alerts
            print("\nRECENT TRIGGERED ALERTS:")
            print("-" * 80)
            
            triggered_query = """
            SELECT timestamp, ticker, stock_name, exchange, trigger_reason
            FROM alert_audits 
            WHERE alert_triggered = 1 AND timestamp > %s
            ORDER BY timestamp DESC
            LIMIT 10
            """
            
            triggered_df = pd.read_sql_query(triggered_query, conn, params=(time_str,))
            
            if not triggered_df.empty:
                for _, row in triggered_df.iterrows():
                    print(f"{row['timestamp'][:19]} | {row['ticker']:8} | {row['exchange']:15} | {row['stock_name'][:30]}")
            else:
                print("No alerts triggered")
                
            # Get recent errors
            print("\nRECENT ERRORS:")
            print("-" * 80)
            
            error_query = """
            SELECT timestamp, ticker, error_message
            FROM alert_audits 
            WHERE error_message IS NOT NULL AND timestamp > %s
            ORDER BY timestamp DESC
            LIMIT 5
            """
            
            error_df = pd.read_sql_query(error_query, conn, params=(time_str,))
            
            if not error_df.empty:
                for _, row in error_df.iterrows():
                    print(f"{row['timestamp'][:19]} | {row['ticker']:8} | {row['error_message'][:50]}")
            else:
                print("No errors")
                
        else:
            print("No alert checks found in this time period")
            
    except Exception as e:
        print(f"Error reading audit database: {e}")

def view_scheduler_status():
    """Check if scheduler is running"""
    print("\nSCHEDULER STATUS")
    print("=" * 80)
    
    try:
        from auto_scheduler_v2 import is_scheduler_running, get_scheduler_info
        
        if is_scheduler_running():
            print("Status: RUNNING ✓")
            info = get_scheduler_info()
            print(f"Jobs scheduled: {len(info.get('jobs', []))}")
            if info.get('next_run'):
                print(f"Next run: {info['next_run']}")
        else:
            print("Status: STOPPED ✗")
            
    except Exception as e:
        print(f"Could not check scheduler status: {e}")

def main():
    """Main function"""
    hours = 24  # Default to last 24 hours
    
    if len(sys.argv) > 1:
        try:
            hours = int(sys.argv[1])
        except:
            pass
    
    print(f"\n{'=' * 80}")
    print(f"SCHEDULER AND ALERT LOGS")
    print(f"{'=' * 80}")
    
    view_scheduler_status()
    view_recent_alerts(hours)
    
    print(f"\n{'=' * 80}")
    print("Log storage: alert audits now reside in PostgreSQL (alert_audits table)")
    print("\nUsage: python view_scheduler_logs.py [hours]")
    print("Example: python view_scheduler_logs.py 24  (last 24 hours)")

if __name__ == "__main__":
    main()

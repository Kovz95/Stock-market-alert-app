#!/usr/bin/env python3
"""
Scheduler Status Checker
Comprehensive script to verify scheduler is running and healthy
"""

import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from auto_scheduler_v2 import is_scheduler_running, get_scheduler_info


def check_lock_file():
    """Check if scheduler lock file exists and is valid"""
    lock_file = BASE_DIR / "scheduler_v2.lock"
    
    print("\n" + "=" * 70)
    print("LOCK FILE CHECK")
    print("=" * 70)
    
    if lock_file.exists():
        try:
            with open(lock_file, 'r') as f:
                lock_data = json.load(f)
            
            print(f"✅ Lock file exists: {lock_file}")
            print(f"   PID: {lock_data.get('pid', 'N/A')}")
            print(f"   Started: {lock_data.get('started_at', 'N/A')}")
            print(f"   Last Heartbeat: {lock_data.get('heartbeat', 'N/A')}")
            print(f"   Type: {lock_data.get('type', 'N/A')}")
            return True
        except Exception as e:
            print(f"⚠️  Lock file exists but couldn't read it: {e}")
            return False
    else:
        print(f"❌ Lock file not found: {lock_file}")
        print("   Scheduler may not be running")
        return False


def check_process_status():
    """Check if scheduler process is running"""
    print("\n" + "=" * 70)
    print("PROCESS STATUS CHECK")
    print("=" * 70)
    
    try:
        running = is_scheduler_running()
        
        if running:
            print("✅ Scheduler process is RUNNING")
        else:
            print("❌ Scheduler process is NOT RUNNING")
        
        return running
    except Exception as e:
        print(f"❌ Error checking process status: {e}")
        return False


def check_scheduler_info():
    """Get detailed scheduler information"""
    print("\n" + "=" * 70)
    print("SCHEDULER DETAILED STATUS")
    print("=" * 70)
    
    try:
        info = get_scheduler_info()
        
        if info is None:
            print("⚠️  No scheduler status information available")
            return False
        
        # Main status
        status = info.get('status', 'Unknown')
        status_icon = "✅" if status == "running" else "⚠️"
        print(f"\n{status_icon} Status: {status}")
        
        # Heartbeat
        heartbeat = info.get('heartbeat')
        if heartbeat:
            print(f"💓 Last Heartbeat: {heartbeat}")
            # Parse and check if recent
            try:
                hb_time = datetime.fromisoformat(heartbeat.replace('Z', '+00:00'))
                age = (datetime.now(hb_time.tzinfo) - hb_time).total_seconds()
                if age < 120:  # Less than 2 minutes
                    print(f"   ✅ Heartbeat is fresh ({int(age)}s ago)")
                else:
                    print(f"   ⚠️  Heartbeat is stale ({int(age)}s ago)")
            except:
                pass
        else:
            print("💓 Last Heartbeat: N/A")
        
        # Current job
        current_job = info.get('current_job')
        if current_job:
            print(f"\n🔄 Current Job:")
            print(f"   Exchange: {current_job.get('exchange', 'N/A')}")
            print(f"   Type: {current_job.get('job_type', 'N/A')}")
            print(f"   Started: {current_job.get('started', 'N/A')}")
        else:
            print(f"\n🔄 Current Job: None (idle)")
        
        # Last run
        last_run = info.get('last_run')
        if last_run:
            print(f"\n📋 Last Run:")
            print(f"   Exchange: {last_run.get('exchange', 'N/A')}")
            print(f"   Type: {last_run.get('job_type', 'N/A')}")
            print(f"   Completed: {last_run.get('completed_at', 'N/A')}")
            print(f"   Duration: {last_run.get('duration_seconds', 'N/A')}s")
        
        # Last result
        last_result = info.get('last_result')
        if last_result:
            price_stats = last_result.get('price_stats', {})
            alert_stats = last_result.get('alert_stats', {})
            
            print(f"\n📊 Last Result:")
            if isinstance(price_stats, dict):
                print(f"   Price Updates: {price_stats.get('updated', 0)} updated, "
                      f"{price_stats.get('failed', 0)} failed, "
                      f"{price_stats.get('skipped', 0)} skipped")
            
            if isinstance(alert_stats, dict):
                print(f"   Alerts: {alert_stats.get('total', 0)} checked, "
                      f"{alert_stats.get('triggered', 0)} triggered, "
                      f"{alert_stats.get('errors', 0)} errors")
        
        # Last error
        last_error = info.get('last_error')
        if last_error:
            print(f"\n⚠️  Last Error:")
            print(f"   Time: {last_error.get('time', 'N/A')}")
            print(f"   Exchange: {last_error.get('exchange', 'N/A')}")
            print(f"   Message: {last_error.get('message', 'N/A')}")
        
        # Job counts
        daily_jobs = info.get('total_daily_jobs', 0)
        weekly_jobs = info.get('total_weekly_jobs', 0)
        
        print(f"\n📅 Scheduled Jobs:")
        print(f"   Daily Jobs: {daily_jobs}")
        print(f"   Weekly Jobs: {weekly_jobs}")
        print(f"   Total: {daily_jobs + weekly_jobs}")
        
        # Next run
        next_run = info.get('next_run')
        if next_run:
            print(f"\n⏰ Next Scheduled Run: {next_run}")
        
        return status == "running"
        
    except Exception as e:
        print(f"❌ Error getting scheduler info: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_log_file():
    """Check if log file exists and show recent entries"""
    print("\n" + "=" * 70)
    print("LOG FILE CHECK")
    print("=" * 70)
    
    log_file = BASE_DIR / "auto_scheduler_v2.log"
    
    if log_file.exists():
        print(f"✅ Log file exists: {log_file}")
        print(f"   Size: {log_file.stat().st_size:,} bytes")
        
        # Show last 10 lines
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                last_lines = lines[-10:] if len(lines) >= 10 else lines
            
            print(f"\n   Last {len(last_lines)} log entries:")
            for line in last_lines:
                print(f"   {line.rstrip()}")
            
            return True
        except Exception as e:
            print(f"   ⚠️  Couldn't read log file: {e}")
            return False
    else:
        print(f"❌ Log file not found: {log_file}")
        print("   Scheduler may never have been started")
        return False


def main():
    """Run all scheduler status checks"""
    print("\n" + "=" * 70)
    print("STOCK ALERT SCHEDULER - STATUS CHECK")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all checks
    lock_ok = check_lock_file()
    process_ok = check_process_status()
    info_ok = check_scheduler_info()
    log_ok = check_log_file()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    checks = [
        ("Lock File", lock_ok),
        ("Process Running", process_ok),
        ("Scheduler Info", info_ok),
        ("Log File", log_ok),
    ]
    
    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {check_name}")
    
    all_ok = all(result for _, result in checks)
    
    if all_ok:
        print("\n✅ Scheduler is HEALTHY and RUNNING")
        return 0
    else:
        print("\n⚠️  Scheduler has issues or is NOT RUNNING")
        print("\nTo start the scheduler:")
        print("  python auto_scheduler_v2.py")
        return 1


if __name__ == "__main__":
    sys.exit(main())

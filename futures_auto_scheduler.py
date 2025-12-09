"""
Futures Auto Scheduler - Automatically updates futures prices and checks alerts
Runs separately from the main stock scheduler
"""

import schedule
import time
import logging
import os
import sys
from datetime import datetime, timedelta
import subprocess
import threading
import traceback
import psutil
from futures_scheduler_discord import futures_discord
from data_access.document_store import load_document, save_document
from pprint import pprint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('futures_scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class FuturesAutoScheduler:
    def __init__(self):
        self.is_running = False
        self.config = self.load_config()
        self.lock_file = 'futures_scheduler.lock'
        self.python_path = sys.executable
        self.status_file = 'futures_scheduler_status.json'

    def load_config(self):
        """Load scheduler configuration"""
        default_config = {
            "update_times": ["06:00", "12:00", "16:00", "20:00"],
            "check_interval_minutes": 30,
            "enabled": True,
            "update_on_start": True,
            "ib_hours": {
                "start": "05:00",
                "end": "23:00",
            },
        }
        try:
            config = load_document(
                "futures_scheduler_config",
                default=default_config,
                fallback_path='futures_scheduler_config.json',
            )
            if not isinstance(config, dict):
                return default_config
            merged = {**default_config, **config}
            merged.setdefault("ib_hours", default_config["ib_hours"])
            return merged
        except Exception as e:
            logger.error(f"Failed to load scheduler config: {e}")
            return default_config

    def save_config(self):
        """Save configuration to file"""
        try:
            save_document(
                "futures_scheduler_config",
                self.config,
                fallback_path='futures_scheduler_config.json',
            )
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def is_ib_available(self):
        """Check if current time is within IB trading hours"""
        try:
            now = datetime.now()
            current_time = now.strftime("%H:%M")

            ib_start = self.config.get('ib_hours', {}).get('start', '05:00')
            ib_end = self.config.get('ib_hours', {}).get('end', '23:00')

            # Simple time comparison (assumes same day)
            return ib_start <= current_time <= ib_end

        except Exception as e:
            logger.error(f"Error checking IB availability: {e}")
            return True  # Assume available if can't determine

    def get_current_status(self):
        """Get current status data"""
        try:
            status = load_document(
                "futures_scheduler_status",
                default={},
                fallback_path=self.status_file,
            )
            return status if isinstance(status, dict) else {}
        except Exception:
            return {}

    def update_status(self, status, last_update=None, last_check=None, next_update=None, preserve_times=False):
        """Update scheduler status file"""
        try:
            # Get existing status if we need to preserve times
            current = self.get_current_status() if preserve_times else {}

            status_data = {
                "status": status,
                "last_update": last_update if last_update is not None else current.get('last_update', datetime.now().isoformat()),
                "last_check": last_check if last_check is not None else current.get('last_check', datetime.now().isoformat()),
                "next_update": next_update,
                "pid": os.getpid(),
                "started": current.get('started', datetime.now().isoformat())
            }

            save_document(
                "futures_scheduler_status",
                status_data,
                fallback_path=self.status_file,
            )

        except Exception as e:
            logger.error(f"Failed to update status: {e}")

    def run_price_update(self):
        """Run futures price update"""
        try:
            if not self.is_ib_available():
                logger.info("Skipping price update - outside IB hours")
                return

            logger.info("="*60)
            logger.info("Starting futures price update...")

            # Track start time for duration calculation
            start_time = datetime.now()

            # Update status
            self.update_status("Updating prices", last_update=datetime.now().isoformat())

            # Run the price updater
            result = subprocess.run(
                [self.python_path, "futures_price_updater.py"],
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode == 0:
                logger.info("Price update completed successfully")
                self.update_status("Running", last_update=datetime.now().isoformat())

                # Send Discord notification for successful update
                try:
                    # Parse output for counts if available
                    updated_count = 0
                    if "succeeded" in result.stdout:
                        try:
                            # Extract count from output like "Update complete: 60 succeeded"
                            for line in result.stdout.split('\n'):
                                if "succeeded" in line:
                                    updated_count = int(line.split()[2])
                                    break
                        except:
                            pass

                    duration = f"{(datetime.now() - start_time).seconds} seconds"
                    futures_discord.send_status_update('price_update_complete', {
                        'updated_count': updated_count,
                        'duration': duration
                    })
                except Exception as e:
                    logger.error(f"Failed to send Discord update notification: {e}")
            else:
                logger.error(f"Price update failed: {result.stderr}")
                self.update_status("Error in price update")

                # Send Discord notification for failed update
                try:
                    futures_discord.send_status_update('price_update_failed', {
                        'error': result.stderr[:500] if result.stderr else "Unknown error"
                    })
                except Exception as e:
                    logger.error(f"Failed to send Discord error notification: {e}")

        except subprocess.TimeoutExpired:
            logger.error("Price update timed out")
            self.update_status("Price update timeout")
        except Exception as e:
            logger.error(f"Error running price update: {e}")
            logger.error(traceback.format_exc())
            self.update_status("Error")

    def run_alert_check(self):
        """Run futures alert check"""
        try:
            logger.info("Starting futures alert check...")

            # Update status
            self.update_status("Checking alerts", last_check=datetime.now().isoformat())

            # Run the alert checker
            result = subprocess.run(
                [self.python_path, "futures_alert_checker.py"],
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )

            if result.returncode == 0:
                logger.info("Alert check completed successfully")
                # Only update last_check, preserve last_update time
                self.update_status("Running", last_check=datetime.now().isoformat(), preserve_times=True)

                # Parse output for triggered alerts
                triggered_count = 0
                if "TRIGGERED:" in result.stdout:
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if "TRIGGERED:" in line:
                            logger.info(f"Alert: {line}")
                            triggered_count += 1

                # Send Discord notification for alert check
                try:
                    # Count total alerts checked (could parse from output)
                    futures_discord.send_status_update('alert_check_complete', {
                        'alert_count': 74,  # We know there are 74 alerts
                        'triggered_count': triggered_count
                    })
                except Exception as e:
                    logger.error(f"Failed to send Discord alert check notification: {e}")
            else:
                logger.error(f"Alert check failed: {result.stderr}")
                self.update_status("Error in alert check")

                # Send Discord notification for failed alert check
                try:
                    futures_discord.send_status_update('error', {
                        'error': f"Alert check failed: {result.stderr[:500] if result.stderr else 'Unknown error'}"
                    })
                except Exception as e:
                    logger.error(f"Failed to send Discord error notification: {e}")

        except subprocess.TimeoutExpired:
            logger.error("Alert check timed out")
            self.update_status("Alert check timeout")
        except Exception as e:
            logger.error(f"Error running alert check: {e}")
            logger.error(traceback.format_exc())
            self.update_status("Error")

    def combined_update_and_check(self):
        """Run price update followed by alert check"""
        try:
            logger.info("="*60)
            logger.info("FUTURES SCHEDULER - Combined Update & Check")
            logger.info("="*60)

            # First update prices
            self.run_price_update()

            # Wait a bit for database to settle
            time.sleep(5)

            # Then check alerts
            self.run_alert_check()

            logger.info("Combined update and check completed")

        except Exception as e:
            logger.error(f"Error in combined update: {e}")

    def schedule_jobs(self):
        """Schedule all jobs"""
        try:
            # Clear existing jobs
            schedule.clear()

            # Schedule price updates with alert checks at specific times
            for update_time in self.config.get('update_times', ['06:00', '12:00', '16:00', '20:00']):
                schedule.every().day.at(update_time).do(self.combined_update_and_check)
                logger.info(f"Scheduled update at {update_time}")

            # NO LONGER scheduling periodic alert checks - only after price updates
            # Alert checks now only happen as part of combined_update_and_check

            # Run initial update if configured
            if self.config.get('update_on_start', True):
                logger.info("Running initial update...")
                threading.Thread(target=self.combined_update_and_check).start()

        except Exception as e:
            logger.error(f"Error scheduling jobs: {e}")

    def acquire_lock(self):
        """Acquire scheduler lock to prevent multiple instances"""
        try:
            if os.path.exists(self.lock_file):
                # Check if the process is still running
                try:
                    with open(self.lock_file, 'r') as f:
                        old_pid = int(f.read())

                    # Check if process exists
                    try:
                        os.kill(old_pid, 0)
                        logger.error(f"Another scheduler instance is running (PID: {old_pid})")
                        return False
                    except OSError:
                        # Process doesn't exist, remove stale lock
                        os.remove(self.lock_file)
                        logger.info("Removed stale lock file")
                except:
                    os.remove(self.lock_file)

            # Create new lock
            with open(self.lock_file, 'w') as f:
                f.write(str(os.getpid()))

            logger.info(f"Acquired lock (PID: {os.getpid()})")
            return True

        except Exception as e:
            logger.error(f"Failed to acquire lock: {e}")
            return False

    def release_lock(self):
        """Release scheduler lock"""
        try:
            if os.path.exists(self.lock_file):
                os.remove(self.lock_file)
                logger.info("Released lock")
        except Exception as e:
            logger.error(f"Failed to release lock: {e}")

    def start(self):
        """Start the scheduler"""
        try:
            if not self.acquire_lock():
                logger.error("Failed to acquire lock - another instance may be running")
                return False

            logger.info("="*60)
            logger.info("FUTURES AUTO SCHEDULER STARTED")
            logger.info(f"PID: {os.getpid()}")
            logger.info(f"Update times: {self.config.get('update_times', [])}")
            logger.info("Alert checks: After each price update only")
            logger.info("="*60)

            # Send Discord notification for scheduler start
            try:
                futures_discord.send_status_update('started', {
                    'schedule': ', '.join(self.config.get('update_times', [])),
                    'pid': os.getpid()
                })
            except Exception as e:
                logger.error(f"Failed to send Discord start notification: {e}")

            self.is_running = True
            self.schedule_jobs()
            self.update_status("Running")

            # Main scheduler loop
            while self.is_running:
                try:
                    schedule.run_pending()
                    time.sleep(30)  # Check every 30 seconds
                except KeyboardInterrupt:
                    logger.info("Scheduler interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"Error in scheduler loop: {e}")
                    time.sleep(60)  # Wait a bit before retrying

            self.stop()
            return True

        except Exception as e:
            logger.error(f"Failed to start scheduler: {e}")
            self.release_lock()
            return False

    def stop(self):
        """Stop the scheduler"""
        try:
            logger.info("Stopping futures scheduler...")
            self.is_running = False
            schedule.clear()
            self.update_status("Stopped")
            self.release_lock()
            logger.info("Futures scheduler stopped")
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")


def main():
    """Main entry point"""
    scheduler = FuturesAutoScheduler()

    try:
        # Check for command line arguments
        if len(sys.argv) > 1:
            command = sys.argv[1].lower()

            if command == 'stop':
                logger.info("Stopping futures scheduler...")
                if os.path.exists(scheduler.lock_file):
                    with open(scheduler.lock_file, 'r') as f:
                        pid = int(f.read())
                    try:
                        os.kill(pid, 15)  # SIGTERM
                        logger.info(f"Sent stop signal to PID {pid}")
                    except:
                        logger.error("Failed to stop scheduler")
                return

            elif command == 'status':
                status = scheduler.get_current_status()
                if status:
                    pprint(status)
                else:
                    print("Scheduler not running")
                return

            elif command == 'update':
                logger.info("Running manual update...")
                scheduler.combined_update_and_check()
                return

        # Start the scheduler
        scheduler.start()

    except KeyboardInterrupt:
        logger.info("Scheduler interrupted")
        scheduler.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        scheduler.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()

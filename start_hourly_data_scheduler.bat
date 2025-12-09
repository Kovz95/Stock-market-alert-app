@echo off
echo ================================================================
echo Starting Hourly Data Scheduler (SEPARATE from alert scheduler)
echo ================================================================
echo.
echo This scheduler will:
echo - Run SEPARATELY from auto_scheduler_v2.py
echo - Use its own lock file: hourly_scheduler.lock
echo - Update hourly price data every hour at :05 past the hour
echo - Log to: hourly_data_scheduler.log
echo.
echo Press Ctrl+C to stop the scheduler
echo ================================================================
echo.

"C:\Users\NickK\AppData\Local\Programs\Python\Python313\python.exe" hourly_data_scheduler.py

pause

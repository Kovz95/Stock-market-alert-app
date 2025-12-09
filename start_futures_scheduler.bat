@echo off
echo ========================================
echo Starting Futures Auto Scheduler
echo ========================================

REM Set Python path
set PYTHON_PATH=C:\Users\NickK\AppData\Local\Programs\Python\Python313\python.exe

REM Check if Python exists
if not exist "%PYTHON_PATH%" (
    echo Error: Python not found at %PYTHON_PATH%
    pause
    exit /b 1
)

REM Change to the app directory
cd /d "C:\Users\NickK\OneDrive\Documents\stock alert app"

echo.
echo Starting futures scheduler...
echo This will:
echo - Update futures prices from IB (6 times daily)
echo - Check futures alerts (every 15 minutes)
echo - Send Discord notifications for triggered alerts
echo.

REM Start the futures scheduler
"%PYTHON_PATH%" futures_auto_scheduler.py

echo.
echo Futures scheduler stopped.
pause
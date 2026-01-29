# Production Alert System Guide

## Overview

Your alert system runs automatically on a schedule without manual intervention. This guide explains how alerts are checked in production, how the scheduler works, and how it knows what to test.

## Architecture

The system consists of three separate schedulers:

### 1. **Stock Alert Scheduler** (`auto_scheduler_v2.py`)
- **Purpose:** Daily and weekly stock alerts
- **Runs:** Calendar-aware scheduling based on global exchange closing times
- **Lock File:** `scheduler_v2.lock`
- **Log File:** `auto_scheduler_v2.log`

### 2. **Futures Alert Scheduler** (`futures_auto_scheduler.py`)
- **Purpose:** Futures contract alerts
- **Runs:** Fixed schedule (4 times daily for prices, every 15-30 min for alerts)
- **Lock File:** `futures_scheduler.lock`
- **Log File:** `futures_scheduler.log`

### 3. **Hourly Data Scheduler** (`hourly_data_scheduler.py`)
- **Purpose:** Hourly price updates and hourly alerts
- **Runs:** Every hour at :05 past the hour
- **Lock File:** `hourly_scheduler.lock`
- **Log File:** `hourly_data_scheduler.log`

---

## 1. Stock Alert Scheduler (Main Scheduler)

### How It Works

```
┌─────────────────────────────────────────────────────────┐
│  AUTOMATIC DAILY WORKFLOW                               │
└─────────────────────────────────────────────────────────┘

1. Scheduler waits until exchange closing time + 40 min
   ↓
2. Update prices for that exchange
   ↓
3. Load ALL alerts from PostgreSQL database
   ↓
4. Filter alerts by:
   - Exchange match
   - Timeframe (daily/weekly)
   - Status (only "action: on")
   ↓
5. For each matching alert:
   - Fetch price data from FMP API
   - Evaluate conditions using backend.evaluate_expression()
   - If conditions met → Send Discord notification
   - Update last_triggered timestamp
   ↓
6. Send completion summary to Discord
```

### Schedule Configuration

The scheduler is **calendar-aware** using the `exchange-calendars` library:

- **When:** After each exchange closes (with 40-minute data delay)
- **Example:**
  - NYSE closes at 21:00 UTC → Job runs at 21:40 UTC
  - LSE closes at 16:30 UTC → Job runs at 17:10 UTC
  - TSX closes at 21:00 UTC → Job runs at 21:40 UTC

### What Gets Checked

```python
# The scheduler automatically:
# 1. Loads ALL alerts from database
all_alerts = list_alerts()

# 2. Filters to relevant alerts
relevant_alerts = [
    alert for alert in all_alerts
    if alert.get('action') == 'on'  # Only enabled alerts
    and alert.get('exchange') in current_exchanges  # Exchange match
    and alert.get('timeframe') in ('daily', '1d')  # Daily timeframe
]

# 3. Evaluates each alert
checker = StockAlertChecker()
for alert in relevant_alerts:
    result = checker.check_alert(alert)
    if result.get('triggered'):
        # Send Discord notification
        # Update last_triggered timestamp
```

### How It Knows Alert Conditions

The scheduler reads everything from the **PostgreSQL alerts table**:

```sql
SELECT 
    alert_id,
    ticker,
    conditions,           -- The actual condition expressions
    combination_logic,    -- AND/OR for multiple conditions
    timeframe,           -- daily/weekly/hourly
    action,              -- on/off (enabled/disabled)
    last_triggered       -- Last trigger timestamp
FROM alerts
WHERE action = 'on'
```

The `conditions` field contains the actual expressions like:
- `"Close[-1] > 200"`
- `"RSI < 30 AND SMA_20 > SMA_50"`
- `"HARSI_Flip(period = 14, smoothing = 3)[-1] > 0"`

### Starting the Stock Scheduler

**Manual Start:**
```bash
python auto_scheduler_v2.py
```

**The scheduler will:**
1. Check if another instance is already running (lock file check)
2. Load all exchange schedules from `exchange_schedule_config_v2.py`
3. Schedule jobs for 39+ global exchanges
4. Run continuously, executing jobs at scheduled times
5. Update heartbeat every 60 seconds
6. Send Discord notifications for job events

**Key Features:**
- ✅ Prevents duplicate execution (job locking)
- ✅ Timeout protection (15-minute default per job)
- ✅ Subprocess isolation (crashes don't affect scheduler)
- ✅ Automatic recovery (watchdog can restart)
- ✅ Discord notifications for all events

---

## 2. Futures Alert Scheduler

### How It Works

```
┌─────────────────────────────────────────────────────────┐
│  FUTURES WORKFLOW                                       │
└─────────────────────────────────────────────────────────┘

PRICE UPDATES (4x daily):
- Run at: 06:00, 12:00, 16:00, 20:00 UTC
- Fetch from: Interactive Brokers (IB) or FMP
- Store in: continuous_prices table

ALERT CHECKS (every 15-30 minutes):
- Load alerts from: futures_alerts.json (document store)
- Filter by: action == 'on'
- Evaluate: Custom condition logic
- Notify: Discord on trigger
```

### Starting the Futures Scheduler

**Manual Start:**
```bash
python futures_auto_scheduler.py

# Or use batch file:
start_futures_scheduler.bat
```

### What Gets Checked

```python
# Futures alerts are stored in JSON format
# Location: PostgreSQL document store → futures_alerts

# Structure:
{
    "alert_id": "abc-123",
    "ticker": "GC",  # Gold futures
    "name": "Gold Buy Signal",
    "entry_conditions": {
        "1": {"conditions": ["Close[-1] > SMA_50"]},
        "2": {"conditions": ["RSI < 30"]}
    },
    "entry_combination": "AND",
    "adjustment_method": "panama",
    "action": "on",
    "last_triggered": "2026-01-27T10:30:00"
}

# Scheduler loads these and evaluates conditions
```

---

## 3. Hourly Data Scheduler

### How It Works

```
┌─────────────────────────────────────────────────────────┐
│  HOURLY WORKFLOW                                        │
└─────────────────────────────────────────────────────────┘

Every hour at :05 past the hour (e.g., 10:05, 11:05):
1. Fetch hourly price data from FMP
2. Load all alerts from database
3. Filter to: timeframe == 'hourly' or '1h'
4. Evaluate hourly alert conditions
5. Send Discord notifications
```

### Starting the Hourly Scheduler

**Manual Start:**
```bash
python hourly_data_scheduler.py

# Or use batch file:
start_hourly_data_scheduler.bat
```

---

## How Schedulers Know What to Check

### The Data Flow

```
USER CREATES ALERT IN UI
        ↓
SAVED TO POSTGRESQL
(alerts table or document store)
        ↓
SCHEDULER LOADS ON NEXT RUN
        ↓
FILTERS BY EXCHANGE/TIMEFRAME
        ↓
EVALUATES CONDITIONS
        ↓
SENDS DISCORD IF TRIGGERED
```

### Alert Data Sources

| Scheduler | Data Source | Format |
|-----------|-------------|--------|
| Stock (Daily/Weekly) | PostgreSQL `alerts` table | Relational |
| Futures | Document store → `futures_alerts` | JSON |
| Hourly | PostgreSQL `alerts` table | Relational |

### Condition Evaluation

All schedulers use the same evaluation logic from `backend.py`:

```python
# Example: Alert has condition "Close[-1] > 200"

# 1. Fetch price data
df = fmp.get_historical_data(ticker, period="1day")

# 2. Create evaluation context
context = {
    'Close': df['close'].values,  # Array for indexing
    'Open': df['open'].values,
    'High': df['high'].values,
    'Low': df['low'].values,
    # ... plus indicators like SMA, RSI, etc.
}

# 3. Evaluate condition
result = eval(condition_string, context)
# "Close[-1] > 200" evaluates to True/False

# 4. If True → Trigger alert
if result:
    send_discord_notification()
    update_last_triggered()
```

---

## Checking Scheduler Status

### View Current Status

```bash
# Stock scheduler status
python -c "from auto_scheduler_v2 import get_scheduler_info; print(get_scheduler_info())"

# Futures scheduler status
python -c "from data_access.document_store import load_document; print(load_document('futures_scheduler_status'))"

# Check if running
python -c "from auto_scheduler_v2 import is_scheduler_running; print(is_scheduler_running())"
```

### Check Lock Files

```bash
# Stock scheduler
cat scheduler_v2.lock

# Futures scheduler  
cat futures_scheduler.lock

# Hourly scheduler
cat hourly_scheduler.lock
```

### View Logs

```bash
# Stock scheduler
tail -f auto_scheduler_v2.log

# Futures scheduler
tail -f futures_scheduler.log

# Hourly scheduler
tail -f hourly_data_scheduler.log
```

---

## Understanding the Schedule

### Stock Alerts (Exchange-Based)

The stock scheduler runs based on **real market calendars**:

```python
# Example schedule from exchange_schedule_config_v2.py

EXCHANGE_SCHEDULES = {
    'NASDAQ': {
        'close_time_utc': '21:00',  # 4:00 PM ET
        'trading_days': 'mon-fri',
        'data_delay_minutes': 40,
    },
    'LSE': {
        'close_time_utc': '16:30',  # 4:30 PM GMT
        'trading_days': 'mon-fri',
        'data_delay_minutes': 40,
    },
    # ... 39+ exchanges
}

# Jobs scheduled:
# - NASDAQ: Daily at 21:40 UTC, Weekly on Fridays at 21:40 UTC
# - LSE: Daily at 17:10 UTC, Weekly on Fridays at 17:10 UTC
```

### Futures Alerts (Fixed Schedule)

```python
# Default configuration
{
    "update_times": ["06:00", "12:00", "16:00", "20:00"],  # Price updates
    "check_interval_minutes": 30,  # Alert checks every 30 min
    "ib_hours": {
        "start": "05:00",
        "end": "23:00"
    }
}
```

### Hourly Alerts (Fixed Hourly)

```python
# Runs: Every hour at :05 past the hour
# Examples:
#   10:05, 11:05, 12:05, 13:05, etc.
```

---

## Configuration Files

### Stock Scheduler Config

**Location:** `scheduler_config.json` (document store)

```json
{
    "scheduler_webhook": {
        "url": "https://discord.com/api/webhooks/...",
        "enabled": true,
        "name": "Scheduler Status"
    },
    "notification_settings": {
        "send_start_notification": true,
        "send_completion_notification": true,
        "send_progress_updates": false,
        "job_timeout_seconds": 900
    }
}
```

### Futures Scheduler Config

**Location:** `futures_scheduler_config.json` (document store)

```json
{
    "update_times": ["06:00", "12:00", "16:00", "20:00"],
    "check_interval_minutes": 30,
    "enabled": true,
    "update_on_start": true
}
```

---

## Starting All Schedulers

### Recommended Startup Order

```bash
# 1. Start stock scheduler (main)
python auto_scheduler_v2.py &

# 2. Start futures scheduler
python futures_auto_scheduler.py &

# 3. Start hourly scheduler
python hourly_data_scheduler.py &

# Check all are running
ps aux | grep scheduler
```

### Using Batch Files (Windows)

```cmd
REM Start futures scheduler
start_futures_scheduler.bat

REM Start hourly scheduler
start_hourly_data_scheduler.bat

REM Stock scheduler (add to startup or run manually)
python auto_scheduler_v2.py
```

---

## How Alerts Get Into the System

### Stock Alerts

1. User creates alert in Streamlit UI (`pages/Add_Alert.py`)
2. UI calls `create_alert()` from `alert_repository.py`
3. Alert saved to PostgreSQL `alerts` table
4. Cache cleared to ensure fresh data
5. Next scheduler run loads the new alert
6. Alert evaluated on schedule

### Futures Alerts

1. User creates alert in Futures UI
2. Alert saved to `futures_alerts` document in document store
3. Next scheduler run loads the new alert
4. Alert evaluated every 15-30 minutes

---

## Troubleshooting

### Alerts Not Being Checked

**Check:**
1. Is scheduler running? `ps aux | grep scheduler`
2. Is alert enabled? `action == 'on'`
3. Is exchange/timeframe correct?
4. Check scheduler logs for errors

### Scheduler Won't Start

**Check:**
1. Lock file exists? Remove stale lock files
2. Another instance running? Kill it first
3. Database accessible?
4. Discord webhooks configured?

### Alerts Not Triggering

**Check:**
1. Conditions actually met? Use test script to verify
2. Already triggered today? Check `last_triggered`
3. Price data available? Check logs for "no data"
4. Condition syntax correct? Test with `evaluate_expression()`

---

## Manual Testing vs Production

| Aspect | Test Script | Production Scheduler |
|--------|-------------|---------------------|
| **Trigger** | Manual command | Automatic schedule |
| **Data Source** | Real-time from API | Real-time from API |
| **Condition Override** | ✅ Supported | ❌ Not possible |
| **Alerts Checked** | Single/selected | All enabled alerts |
| **Discord Send** | ✅ Real messages | ✅ Real messages |
| **Database Update** | ✅ Updates last_triggered | ✅ Updates last_triggered |
| **Purpose** | Testing & debugging | Production operation |

The test scripts (`test_alert_trigger.py`) are for **development and debugging**.

The schedulers are for **production operation**.

---

## Best Practices

1. **Test alerts before enabling** - Use `test_alert_trigger.py` first
2. **Monitor scheduler logs** - Check logs regularly for errors
3. **Keep one instance running** - Lock files prevent duplicates
4. **Configure Discord webhooks** - Essential for monitoring
5. **Check scheduler status** - Verify schedulers are running
6. **Review triggered alerts** - Check Discord channels daily
7. **Clean up old alerts** - Disable or delete unused alerts

---

## Summary

**How It Works:**
1. Schedulers run continuously in the background
2. They automatically load alerts from the database
3. On schedule, they fetch price data and evaluate conditions
4. When conditions are met, they send Discord notifications
5. They track execution status and errors

**Key Points:**
- ✅ Fully automatic - no manual intervention needed
- ✅ Calendar-aware for stock alerts
- ✅ Fixed schedule for futures and hourly alerts
- ✅ Reads conditions directly from database
- ✅ Evaluates conditions using `backend.evaluate_expression()`
- ✅ Sends Discord notifications on trigger
- ✅ Updates `last_triggered` to prevent duplicates
- ✅ Comprehensive logging and monitoring

Your alert system runs 24/7, checking alerts on schedule, and notifying you via Discord when conditions are met!

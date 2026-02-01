# Stock Alert Testing Toolkit Guide

## Overview

This guide documents all the testing and monitoring tools created to help you verify, test, and monitor your stock alert system.

## Quick Reference

| Tool | Purpose | Usage |
|------|---------|-------|
| `check_scheduler_status.py` | Check if scheduler is running | `python check_scheduler_status.py` |
| `test_monitor.py` | Monitor recent alert activity | `python test_monitor.py --all` |
| `analyze_alerts.py` | Analyze alert performance | `python analyze_alerts.py --all` |
| `test_alert_trigger.py` | Test specific alerts | `python test_alert_trigger.py --list` |
| `check_alert_routing.py` | Verify Discord routing | `python check_alert_routing.py --config` |
| `monitor_alerts_dashboard.py` | Live monitoring dashboard | `python monitor_alerts_dashboard.py` |

---

## 1. Verify Scheduler is Running

### Tool: `check_scheduler_status.py`

**What it does:**
- Checks if scheduler process is running
- Verifies lock file exists and is fresh
- Shows scheduler status and heartbeat
- Displays recent job execution
- Shows log file contents

**Usage:**
```bash
python check_scheduler_status.py
```

**Output:**
- ✅ PASS/❌ FAIL for each check
- Scheduler health summary
- Instructions if scheduler is not running

**When to use:**
- Before running tests to ensure scheduler is operational
- After system restart to verify scheduler started
- When alerts aren't triggering (first diagnostic step)

---

## 2. Monitor Recent Alert Activity

### Tool: `test_monitor.py`

**What it does:**
- Shows recent alert checks from audit log
- Displays scheduler status from document store
- Gets evaluation summary statistics
- Shows alert history for specific alerts

**Usage:**

```bash
# Show all monitoring info
python test_monitor.py --all

# Show recent checks (last 2 hours)
python test_monitor.py --recent 2

# Show specific alert history
python test_monitor.py --alert ALERT_ID --alert-days 7

# Show evaluation summary (last 48 hours)
python test_monitor.py --summary 48
```

**Key Information Displayed:**
- Total checks in time period
- Number triggered vs not triggered
- Error count and success rate
- Scheduler current job and last run

**When to use:**
- After scheduler runs to verify alerts were checked
- To see if specific alerts are being evaluated
- To check recent activity before testing

---

## 3. Analyze Alert Performance

### Tool: `analyze_alerts.py`

**What it does:**
- Comprehensive analysis of alert evaluations
- Shows triggered alerts history
- Identifies failed checks and errors
- Analyzes performance by ticker or alert
- Breaks down activity by timeframe

**Usage:**

```bash
# Full analysis report
python analyze_alerts.py --all

# Last 24 hours summary
python analyze_alerts.py --summary 24

# Recently triggered alerts
python analyze_alerts.py --triggered 24

# Failed checks
python analyze_alerts.py --errors 24

# Analyze specific ticker
python analyze_alerts.py --ticker AAPL --days 7

# Analyze specific alert
python analyze_alerts.py --alert ALERT_ID --days 7

# Timeframe breakdown
python analyze_alerts.py --timeframe 24
```

**Key Metrics:**
- Total checks, triggered count, error count
- Success rate, trigger rate, data availability
- Execution times (min/max/average)
- Per-ticker and per-alert statistics

**When to use:**
- To understand alert performance over time
- To identify problematic alerts with frequent errors
- To see which alerts trigger most often
- To analyze why an alert hasn't triggered

---

## 4. Test Specific Alerts

### Tool: `test_alert_trigger.py` (Already exists, documented here)

**What it does:**
- List all available alerts
- Test alerts with current market data
- Force trigger with condition override
- Create test alerts
- Test Discord routing only

**Usage:**

```bash
# List all alerts
python test_alert_trigger.py --list

# Test with current market data
python test_alert_trigger.py --test-alert ALERT_ID

# Force trigger (for testing Discord delivery)
python test_alert_trigger.py --test-override ALERT_ID

# Use custom override condition
python test_alert_trigger.py --test-override ALERT_ID --override-condition "Close[-1] > 100"

# Create test alert
python test_alert_trigger.py --create-test AAPL --test-condition "Close[-1] > 0"

# Test Discord routing only
python test_alert_trigger.py --test-discord AAPL
```

**Important Notes:**
- `--test-alert` uses real market conditions (may not trigger)
- `--test-override` forces trigger without modifying alert in database
- Both send real Discord notifications
- Both update `last_triggered` timestamp

**When to use:**
- To verify a new alert works correctly
- To test Discord delivery without waiting for conditions
- To debug why an alert isn't triggering
- To verify routing to correct Discord channels

---

## 5. Verify Discord Routing

### Tool: `check_alert_routing.py` (Enhanced with UTF-8 support)

**What it does:**
- Shows which Discord channel(s) an alert will be sent to
- Explains routing decision (economy/industry/ETF/default)
- Lists all custom channels that match
- Verifies webhook configuration

**Usage:**

```bash
# Check specific alert routing
python check_alert_routing.py --alert ALERT_ID

# Check where a ticker would route
python check_alert_routing.py --ticker AAPL

# Show all alerts grouped by channel
python check_alert_routing.py --all

# Show Discord channel configuration
python check_alert_routing.py --config
```

**Information Displayed:**
- Primary channel and routing reason
- Custom channels that match alert conditions
- Webhook URLs (masked for security)
- Configuration status (✅ configured / ⚠️ not configured)

**When to use:**
- After creating an alert to verify routing
- When Discord messages go to wrong channel
- To audit all webhook configurations
- To understand custom channel matching logic

---

## 6. Live Monitoring Dashboard

### Tool: `monitor_alerts_dashboard.py`

**What it does:**
- Real-time monitoring of scheduler and alerts
- Shows scheduler health and current job
- Displays recent activity and triggered alerts
- 24-hour summary statistics
- Recent errors

**Usage:**

```bash
# Single snapshot (run once and exit)
python monitor_alerts_dashboard.py --once

# Continuous monitoring (refreshes every 5 minutes)
python monitor_alerts_dashboard.py

# Custom refresh interval (every 2 minutes)
python monitor_alerts_dashboard.py --interval 120
```

**Dashboard Sections:**
1. **Scheduler Status** - Running/stopped, heartbeat, current job, next run
2. **Recent Activity** - Last hour statistics
3. **Recently Triggered** - Last 5 triggered alerts with timestamps
4. **24-Hour Summary** - Total checks, triggers, errors, rates
5. **Recent Errors** - Last errors in 24 hours

**When to use:**
- During active trading hours to monitor live activity
- To keep an eye on scheduler health
- When troubleshooting issues in real-time
- For ongoing system health monitoring

---

## Complete Testing Workflow

### 1. Before Testing

```bash
# Check scheduler is running
python check_scheduler_status.py

# Review recent activity
python test_monitor.py --all
```

### 2. Test a Specific Alert

```bash
# List alerts to find ID
python test_alert_trigger.py --list

# Check routing
python check_alert_routing.py --alert ALERT_ID

# Test with override (guaranteed trigger)
python test_alert_trigger.py --test-override ALERT_ID

# Verify Discord message received
# Check your Discord channels
```

### 3. Verify Correct Evaluation

```bash
# Test with real market data
python test_alert_trigger.py --test-alert ALERT_ID

# Check if it triggered
python analyze_alerts.py --alert ALERT_ID --days 1

# Review evaluation in audit log
python test_monitor.py --alert ALERT_ID --alert-days 1
```

### 4. Monitor Ongoing Performance

```bash
# Run dashboard
python monitor_alerts_dashboard.py

# Or get detailed analysis
python analyze_alerts.py --all
```

---

## Common Scenarios

### Scenario 1: Alert Not Triggering

```bash
# Step 1: Verify alert exists and is enabled
python test_alert_trigger.py --list | grep "TICKER"

# Step 2: Check if conditions are currently met
python test_alert_trigger.py --test-alert ALERT_ID

# Step 3: Force trigger to test Discord delivery
python test_alert_trigger.py --test-override ALERT_ID

# Step 4: Check alert history
python analyze_alerts.py --alert ALERT_ID --days 7

# Step 5: Check for errors
python analyze_alerts.py --errors 24
```

### Scenario 2: Wrong Discord Channel

```bash
# Step 1: Check current routing
python check_alert_routing.py --alert ALERT_ID

# Step 2: Check ticker routing
python check_alert_routing.py --ticker TICKER

# Step 3: Review all channel configs
python check_alert_routing.py --config

# Step 4: Test Discord delivery
python test_alert_trigger.py --test-discord TICKER
```

### Scenario 3: Scheduler Not Running

```bash
# Step 1: Check status
python check_scheduler_status.py

# Step 2: Start scheduler
python auto_scheduler_v2.py

# Step 3: Verify started
python check_scheduler_status.py

# Step 4: Monitor
python monitor_alerts_dashboard.py --once
```

### Scenario 4: Performance Issues

```bash
# Step 1: Get overall statistics
python analyze_alerts.py --summary 24

# Step 2: Check for errors
python analyze_alerts.py --errors 24

# Step 3: Analyze slow alerts (check execution times)
python analyze_alerts.py --all

# Step 4: Check data availability
python test_monitor.py --all
```

---

## Understanding the Output

### Status Indicators

- ✅ **Success/Pass** - Operation completed successfully
- ❌ **Fail/Error** - Operation failed or check didn't pass
- ⚠️ **Warning** - Potential issue but not critical
- 🔄 **Running** - Job is currently executing
- ⏸️ **Idle** - No active jobs

### Common Messages

**"No data available"**
- Price data couldn't be fetched from API
- Check if ticker symbol is correct
- Verify API is responding

**"Conditions not met"**
- Alert evaluated but trigger conditions weren't satisfied
- This is normal - use `--test-override` to force trigger for testing

**"Alert skipped"**
- Alert is disabled (action != 'on')
- Or already triggered today (check `last_triggered`)

**"Heartbeat stale"**
- Scheduler process may have hung or crashed
- Restart scheduler

---

## File Locations

All testing tools are in the project root:

```
Stock-market-alert-app/
├── check_scheduler_status.py       # Scheduler health check
├── test_monitor.py                 # Activity monitoring
├── analyze_alerts.py               # Performance analysis
├── test_alert_trigger.py           # Alert testing (existing)
├── check_alert_routing.py          # Routing verification (enhanced)
├── monitor_alerts_dashboard.py     # Live dashboard
└── TESTING_TOOLKIT_GUIDE.md       # This guide
```

---

## Troubleshooting

### Python Import Errors

```bash
# Activate virtual environment first
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

### Database Connection Errors

```bash
# Check PostgreSQL is running
# Windows: Services -> PostgreSQL
# Linux: systemctl status postgresql
```

### Unicode Encoding Errors (Windows)

All new tools automatically handle UTF-8 encoding on Windows. If you see encoding errors:
- Make sure you're using the latest versions of the scripts
- Try running in Windows Terminal instead of cmd.exe

### Discord Messages Not Received

```bash
# Verify webhooks configured
python check_alert_routing.py --config

# Test specific webhook
python test_alert_trigger.py --test-discord TICKER

# Check Discord rate limits (wait 30 seconds between tests)
```

---

## Best Practices

1. **Before Testing Production Alerts**
   - Always use `--test-override` first to verify Discord delivery
   - Check routing with `check_alert_routing.py`
   - Monitor with dashboard during market hours

2. **When Creating New Alerts**
   - Verify routing immediately after creation
   - Test with override before market opens
   - Monitor first few natural triggers

3. **Regular Monitoring**
   - Run dashboard during active hours
   - Check 24-hour summary daily
   - Review errors weekly

4. **Performance Optimization**
   - Identify slow alerts with `analyze_alerts.py`
   - Check data availability rates
   - Monitor execution times

---

## Quick Commands Cheat Sheet

```bash
# Check everything is working
python check_scheduler_status.py
python test_monitor.py --all

# Test an alert end-to-end
python test_alert_trigger.py --list
python test_alert_trigger.py --test-override ALERT_ID

# Check routing
python check_alert_routing.py --alert ALERT_ID

# Monitor live
python monitor_alerts_dashboard.py

# Analyze performance
python analyze_alerts.py --all

# Check specific ticker
python analyze_alerts.py --ticker AAPL --days 7


# Find errors
python analyze_alerts.py --errors 24
```

---

## Support

For more information, see:
- `ALERT_TESTING_GUIDE.md` - Original alert testing documentation
- `QUICK_START_TESTING.md` - Quick start guide
- `PRODUCTION_ALERT_SYSTEM_GUIDE.md` - Production system overview

---

**All tools are designed to be safe:**
- ✅ Read-only operations don't modify alerts
- ✅ Override tests don't change actual alert conditions
- ✅ All Discord notifications are clearly marked as tests
- ✅ Scheduler status checks don't affect running jobs

# Testing Futures Scheduler

This document explains how to test the new `src/services/futures_scheduler.py` to ensure it works correctly before deploying to production.

---

## Test Suite Overview

We've created three types of tests:

1. **Unit Tests** - Fast, isolated tests for individual components
2. **Integration Tests** - Tests for component interactions
3. **Manual Tests** - Real-world tests with actual services

---

## 1. Unit Tests (Automated)

### Run All Unit Tests

```bash
# Run all futures scheduler unit tests
pytest tests/unit/test_services/test_futures_scheduler.py -v

# Run with coverage
pytest tests/unit/test_services/test_futures_scheduler.py --cov=src.services.futures_scheduler --cov-report=term-missing

# Run specific test class
pytest tests/unit/test_services/test_futures_scheduler.py::TestConfiguration -v

# Run specific test
pytest tests/unit/test_services/test_futures_scheduler.py::TestConfiguration::test_load_scheduler_config_returns_defaults_when_no_config -v
```

### What Unit Tests Cover

✅ Configuration loading and saving
✅ IB hours checking
✅ Job locking mechanism
✅ Status tracking and updates
✅ Process management and locking
✅ Price update execution
✅ Alert checking execution
✅ Discord notification sending
✅ Message formatting
✅ Job execution logic
✅ Scheduler lifecycle (start/stop)

### Expected Results

All unit tests should pass:
```
============================= test session starts ==============================
collected 60 items

tests/unit/test_services/test_futures_scheduler.py::TestConfiguration::test_load_scheduler_config_returns_defaults_when_no_config PASSED
tests/unit/test_services/test_futures_scheduler.py::TestConfiguration::test_load_scheduler_config_merges_with_defaults PASSED
...
============================= 60 passed in 2.5s ================================
```

---

## 2. Integration Tests (Automated)

### Run Integration Tests

```bash
# Run all integration tests
pytest tests/integration/test_futures_scheduler_integration.py -v

# Run with verbose output
pytest tests/integration/test_futures_scheduler_integration.py -v -s
```

### What Integration Tests Cover

✅ Complete job execution flow (price + alerts)
✅ Error handling throughout pipeline
✅ Concurrent job execution prevention
✅ Scheduler lifecycle management
✅ Configuration persistence
✅ Status tracking throughout lifecycle

### Expected Results

All integration tests should pass:
```
============================= test session starts ==============================
collected 8 items

tests/integration/test_futures_scheduler_integration.py::TestFuturesSchedulerIntegration::test_complete_job_execution_flow PASSED
tests/integration/test_futures_scheduler_integration.py::TestFuturesSchedulerIntegration::test_job_handles_price_update_failure PASSED
...
============================= 8 passed in 1.2s =================================
```

---

## 3. Manual Tests (Real Environment)

Manual tests use your actual configuration, database, and services.

### Prerequisites

1. **Configuration file exists**: `futures_scheduler_config.json`
2. **Database access**: PostgreSQL with futures price data
3. **IB connection**: Interactive Brokers TWS or IB Gateway (for price updates)
4. **Discord webhook** (optional, for notification testing)

### Test Script Usage

```bash
# Check configuration
python scripts/analysis/test_futures_scheduler.py --check-config

# Check scheduler status
python scripts/analysis/test_futures_scheduler.py --check-status

# Test Discord notifications
python scripts/analysis/test_futures_scheduler.py --test-notifications

# Test price update only
python scripts/analysis/test_futures_scheduler.py --test-price-update

# Test alert checking only
python scripts/analysis/test_futures_scheduler.py --test-alerts

# Test complete job (price + alerts)
python scripts/analysis/test_futures_scheduler.py --test-job

# Run all non-destructive tests
python scripts/analysis/test_futures_scheduler.py --all
```

### Manual Test Checklist

#### ✅ Step 1: Check Configuration

```bash
python scripts/analysis/test_futures_scheduler.py --check-config
```

**Expected Output:**
```
============================================================
CHECKING FUTURES SCHEDULER CONFIGURATION
============================================================

✅ Configuration loaded successfully

Configuration:
{
  "enabled": true,
  "update_times": ["06:00", "12:00", "16:00", "20:00"],
  ...
}

✅ All required fields present

📅 IB Trading Hours: 05:00 - 23:00 UTC
   Current time within IB hours: Yes ✅

⏰ Scheduled Update Times (UTC): 06:00, 12:00, 16:00, 20:00

🔔 Discord Notifications: Enabled ✅
   Webhook URL: https://discord.com/api/webhooks/...
```

**What to Check:**
- Config file loads successfully
- Update times are correct for your needs
- IB hours match your requirements
- Discord webhook is configured (if you want notifications)

---

#### ✅ Step 2: Test Discord Notifications (Optional)

```bash
python scripts/analysis/test_futures_scheduler.py --test-notifications
```

**Expected Output:**
```
============================================================
TESTING DISCORD NOTIFICATIONS
============================================================

📡 Sending test notification to webhook...
✅ Test notification sent successfully!
   Check your Discord channel to verify delivery
```

**What to Check:**
- Test notification appears in Discord channel
- Message is properly formatted
- Webhook is working correctly

---

#### ✅ Step 3: Test Alert Checking

```bash
python scripts/analysis/test_futures_scheduler.py --test-alerts
```

**Expected Output:**
```
============================================================
TESTING FUTURES ALERT CHECKING
============================================================

🔍 Running futures alert check...

📊 Alert Check Results:
   Total Alerts: 74
   Triggered: 2
   Errors: 0
   Skipped: 10
   No Data: 0
   ✅ Alert check completed successfully
```

**What to Check:**
- Alert checker runs without errors
- Triggered alerts are reported correctly
- Discord notifications sent for triggered alerts (if configured)

---

#### ✅ Step 4: Test Price Update (Requires IB Connection)

```bash
python scripts/analysis/test_futures_scheduler.py --test-price-update
```

**Expected Output:**
```
============================================================
TESTING FUTURES PRICE UPDATE
============================================================

🔄 Running futures price update...
   (This may take several minutes)

📊 Price Update Results:
   Updated: 60
   Failed: 2
   ✅ Price update completed successfully
```

**What to Check:**
- IB connection is working
- Price data is being fetched
- Database is being updated
- Failed updates are expected (some contracts may be inactive)

**Note:** This requires:
- IB TWS or Gateway running
- Valid IB connection credentials
- Current time within IB hours

---

#### ✅ Step 5: Test Complete Job

```bash
python scripts/analysis/test_futures_scheduler.py --test-job
```

**Expected Output:**
```
============================================================
TESTING COMPLETE JOB EXECUTION
============================================================

🚀 Executing complete futures job...
   (This includes price update + alert check)
   (This may take several minutes)

✅ Job completed successfully!

📊 Results:

Price Update:
   Updated: 60
   Failed: 2

Alert Check:
   Total: 74
   Triggered: 2
   Errors: 0

Duration: 45.2 seconds
```

**What to Check:**
- Both price update and alert check run
- Results are logged correctly
- Duration is reasonable (should be < 15 minutes)
- Discord notifications sent (if configured)

---

#### ✅ Step 6: Start Scheduler and Monitor

```bash
# Start the scheduler
python src/services/futures_scheduler.py
```

**Expected Console Output:**
```
2024-01-15 10:00:00 [INFO] Futures scheduler starting (PID 12345)
2024-01-15 10:00:00 [INFO] Scheduled futures job at 06:00 UTC
2024-01-15 10:00:00 [INFO] Scheduled futures job at 12:00 UTC
2024-01-15 10:00:00 [INFO] Scheduled futures job at 16:00 UTC
2024-01-15 10:00:00 [INFO] Scheduled futures job at 20:00 UTC
2024-01-15 10:00:00 [INFO] Scheduled 4 futures jobs
2024-01-15 10:00:00 [INFO] Futures scheduler started successfully
2024-01-15 10:00:00 [INFO] Futures scheduler running. Press Ctrl+C to stop.
```

**Monitor in another terminal:**
```bash
# Watch log file
tail -f futures_scheduler.log

# Check status file
cat futures_scheduler_status.json

# Check lock file
cat futures_scheduler.lock
```

**Let it run for a few minutes, then check:**

1. **Status updates every minute** (heartbeat):
   ```bash
   cat futures_scheduler_status.json | jq '.heartbeat'
   ```

2. **Scheduler is responding**:
   ```bash
   python scripts/analysis/test_futures_scheduler.py --check-status
   ```

3. **Jobs execute at scheduled times**:
   ```bash
   # Wait for next scheduled time and check logs
   tail -f futures_scheduler.log
   ```

---

## Troubleshooting

### Common Issues

#### ❌ "Another futures scheduler instance is running"

**Cause:** Lock file exists from previous run

**Solution:**
```bash
# Check if process is actually running
python scripts/analysis/test_futures_scheduler.py --check-status

# If not running, remove stale lock
rm futures_scheduler.lock

# Then restart
python src/services/futures_scheduler.py
```

---

#### ❌ "Skipping job - outside IB hours"

**Cause:** Current time is outside configured IB hours

**Solution:**
- Adjust `ib_hours` in `futures_scheduler_config.json`
- Or wait until IB hours (default: 05:00-23:00 UTC)
- Or run manual test to override the check

---

#### ❌ "IB connection failed"

**Cause:** Interactive Brokers TWS/Gateway not running or not configured

**Solution:**
1. Start IB TWS or Gateway
2. Verify connection settings in `ib_futures_config.json`
3. Check IB API is enabled in TWS settings
4. Verify client ID is available

---

#### ❌ "No price data for symbol"

**Cause:** Symbol not available in IB or database

**Solution:**
- Check `futures_database.json` for symbol metadata
- Verify `ib_metadata_available` is true for the symbol
- Check IB permissions for futures data

---

#### ❌ Discord notifications not sending

**Cause:** Webhook not configured or invalid

**Solution:**
1. Check webhook URL in config is correct
2. Verify webhook is enabled: `scheduler_webhook.enabled = true`
3. Test webhook directly:
   ```bash
   python scripts/analysis/test_futures_scheduler.py --test-notifications
   ```

---

#### ❌ "Job timeout after 900s"

**Cause:** Job took longer than configured timeout

**Solution:**
- Increase `notification_settings.job_timeout_seconds` in config
- Check IB connection latency
- Reduce number of futures contracts

---

## Success Criteria

Before deploying to production, verify:

- [ ] All unit tests pass (60/60)
- [ ] All integration tests pass (8/8)
- [ ] Configuration loads correctly
- [ ] Discord notifications work (if configured)
- [ ] Price updates complete successfully
- [ ] Alert checks complete successfully
- [ ] Full job executes without errors
- [ ] Scheduler starts and runs continuously
- [ ] Status file updates every minute (heartbeat)
- [ ] Jobs execute at scheduled times
- [ ] Logs are being written correctly
- [ ] Lock file prevents duplicate instances

---

## Running Tests in CI/CD

Add to your CI pipeline:

```yaml
# .github/workflows/test.yml (example)
- name: Run futures scheduler tests
  run: |
    pytest tests/unit/test_services/test_futures_scheduler.py -v
    pytest tests/integration/test_futures_scheduler_integration.py -v
```

---

## Next Steps After Testing

Once all tests pass:

1. **Update watchdog** to monitor futures scheduler
2. **Clean up old files** (move `futures_auto_scheduler.py` to `scripts/migration/`)
3. **Update documentation** to reference new scheduler
4. **Deploy to production** using systemd/supervisor/etc.
5. **Monitor for first 24 hours** to ensure stability

---

## Getting Help

If you encounter issues:

1. **Check logs**: `futures_scheduler.log`
2. **Check status**: `futures_scheduler_status.json`
3. **Run diagnostics**: `python scripts/analysis/test_futures_scheduler.py --all`
4. **Review this guide**: Common issues section above
5. **Check migration guide**: `FUTURES_SCHEDULER_MIGRATION.md`

---

**Good luck testing! 🚀**

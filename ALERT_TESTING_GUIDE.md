# Alert Testing Guide

This guide explains how to test your alert system to ensure that alerts trigger properly and send Discord notifications when conditions are met.

## Overview

The alert system has two main components:
1. **Condition Evaluation** - Checks if alert conditions are met based on current market data
2. **Discord Notifications** - Sends messages to Discord channels when alerts trigger

## Testing Tools

### Main Test Script: `test_alert_trigger.py`

This script provides multiple testing strategies to validate your alert system.

## Testing Strategies

### 1. List Available Alerts

View all alerts in your system to identify which ones to test:

```bash
python test_alert_trigger.py --list
```

This will show:
- Alert ID
- Ticker symbol
- Alert name
- Status (on/off)
- Timeframe
- First few conditions

### 2. Test Specific Alert with Real Market Data

Test an alert using current market conditions to see if it would naturally trigger:

```bash
python test_alert_trigger.py --test-alert ALERT_ID
```

**What this does:**
- Fetches current price data for the alert's ticker
- Evaluates all alert conditions
- Sends Discord notification if conditions are met
- Shows detailed results in console

**Example:**
```bash
python test_alert_trigger.py --test-alert abc123-def456-ghi789
```

**Use this when:**
- You want to test if conditions are currently met
- You want to verify the alert logic works with real data
- You're debugging why an alert isn't triggering

### 3. Test with Condition Override (Guaranteed Trigger)

Override an alert's conditions temporarily to force it to trigger. This tests the Discord notification without waiting for real market conditions:

```bash
python test_alert_trigger.py --test-override ALERT_ID
```

**What this does:**
- Takes an existing alert
- Temporarily replaces conditions with `Close[-1] > 0` (always true)
- Evaluates the alert (will trigger)
- Sends real Discord notification
- Does NOT modify the actual alert in the database

**Custom override condition:**
```bash
python test_alert_trigger.py --test-override ALERT_ID --override-condition "Close[-1] > 100"
```

**Use this when:**
- You want to test Discord notification delivery
- You want to verify Discord channel routing
- You want to test the full alert flow end-to-end
- You don't want to wait for market conditions to align

### 4. Create a Temporary Test Alert

Create a new test alert that will definitely trigger:

```bash
python test_alert_trigger.py --create-test TICKER
```

**Example:**
```bash
python test_alert_trigger.py --create-test AAPL
```

**What this does:**
- Creates a new alert in the database
- Sets condition to `Close[-1] > 0` (always triggers)
- Returns the new alert ID
- Alert is marked as "TEST ALERT" in the name

**Custom test condition:**
```bash
python test_alert_trigger.py --create-test AAPL --test-condition "Close[-1] > 150"
```

**After creation, test it:**
```bash
python test_alert_trigger.py --test-alert [RETURNED_ALERT_ID]
```

**Use this when:**
- You want a dedicated test alert
- You want to test with different tickers
- You want to keep a test alert for repeated testing

**Note:** Remember to delete test alerts when done (via UI or database)

### 5. Test Discord Routing Only

Test Discord message delivery without evaluating any conditions:

```bash
python test_alert_trigger.py --test-discord TICKER
```

**Example:**
```bash
python test_alert_trigger.py --test-discord AAPL
```

**What this does:**
- Creates a test message
- Routes it to the appropriate Discord channel based on ticker
- Sends the message immediately
- Does NOT create any alerts
- Does NOT evaluate conditions

**Use this when:**
- You want to verify Discord webhook configuration
- You want to test channel routing logic
- You want to confirm Discord is receiving messages
- You're debugging Discord connectivity issues

## Understanding Test Results

### Success Indicators
- ✅ **ALERT TRIGGERED!** - Conditions were met and alert fired
- 🧪 **Test message sent successfully!** - Discord routing test passed

### Warning Indicators
- ⏭️ **Alert skipped** - Alert is disabled or already triggered today
- ⚠️ **No price data** - Could not fetch market data for ticker

### Error Indicators
- ❌ **Error** - Something went wrong (see error message)
- ❌ **Failed to send** - Discord notification failed

## Testing Workflow Examples

### Example 1: Test Existing Alert
```bash
# 1. List alerts to find the one you want to test
python test_alert_trigger.py --list

# 2. Test it with current market data
python test_alert_trigger.py --test-alert abc123-def456

# 3. If it doesn't trigger naturally, force a trigger to test Discord
python test_alert_trigger.py --test-override abc123-def456
```

### Example 2: Test Discord Configuration
```bash
# Test that Discord messages are being sent to the correct channel
python test_alert_trigger.py --test-discord AAPL
python test_alert_trigger.py --test-discord GC  # Futures
python test_alert_trigger.py --test-discord AAPL/SPY  # Ratio (if supported)
```

### Example 3: End-to-End Testing
```bash
# 1. Create a test alert
python test_alert_trigger.py --create-test MSFT

# 2. Note the returned alert ID, then test it
python test_alert_trigger.py --test-alert [ALERT_ID]

# 3. Verify Discord message received
# 4. Delete test alert via UI when done
```

## Troubleshooting

### Alert Not Triggering

**Possible causes:**
1. **Conditions not met** - Market data doesn't satisfy the condition
   - Solution: Use `--test-override` to force trigger
   
2. **Alert already triggered today** - Once-per-day trigger limit
   - Solution: Manually clear `last_triggered` timestamp or wait until tomorrow
   
3. **Alert disabled** - Action is set to "off"
   - Solution: Enable the alert via UI
   
4. **No price data** - Ticker data unavailable
   - Solution: Check ticker symbol, verify data source

### Discord Message Not Received

**Possible causes:**
1. **Webhook not configured** - Discord webhook URL missing or invalid
   - Solution: Check `discord_channels_config.json`
   - Solution: Verify webhook in Discord server settings
   
2. **Wrong channel routing** - Message sent to unexpected channel
   - Solution: Use `--test-discord` to verify routing
   - Solution: Check economy/industry classification
   
3. **Rate limiting** - Too many messages sent too quickly
   - Solution: Wait a few seconds between tests
   
4. **Discord server issues** - Discord API temporarily down
   - Solution: Check Discord status page

### Script Errors

**Import errors:**
```
ModuleNotFoundError: No module named 'xxx'
```
- Solution: Activate virtual environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
- Solution: Install dependencies: `pip install -r requirements.txt`

**Database connection errors:**
```
Failed to connect to database
```
- Solution: Verify PostgreSQL is running
- Solution: Check `db_config.py` settings
- Solution: Verify database credentials

## Advanced Usage

### Testing Multiple Alerts
```bash
# Create a shell script to test multiple alerts
for alert_id in alert1 alert2 alert3; do
    echo "Testing $alert_id"
    python test_alert_trigger.py --test-alert $alert_id
    sleep 2  # Avoid rate limiting
done
```

### Automated Testing in CI/CD
```bash
# Test Discord connectivity before deployment
python test_alert_trigger.py --test-discord AAPL || exit 1
```

### Custom Test Conditions

You can test complex conditions by creating test alerts with specific conditions:

```bash
# Test a complex technical indicator condition
python test_alert_trigger.py --create-test AAPL \
    --test-condition "SMA_20 > SMA_50 and RSI < 30"
```

## Integration with Schedulers

The alert system normally runs on a schedule:
- **Daily alerts** - Checked once per day after market close
- **Weekly alerts** - Checked on weekends
- **Hourly alerts** - Checked every hour during market hours

To manually trigger the scheduled check:
```bash
# For stock alerts
python stock_alert_checker.py

# For futures alerts
python futures_alert_checker.py
```

## Best Practices

1. **Test in development first** - Use test alerts before modifying production alerts
2. **Verify Discord routing** - Use `--test-discord` after changing Discord configuration
3. **Check logs** - Review console output for detailed error messages
4. **Clean up test alerts** - Delete test alerts after testing to avoid clutter
5. **Test incrementally** - Test condition evaluation separately from Discord delivery
6. **Use override for integration tests** - `--test-override` gives you full control

## Related Files

- `stock_alert_checker.py` - Main stock alert evaluation engine
- `futures_alert_checker.py` - Futures alert evaluation engine
- `discord_routing.py` - Discord channel routing logic
- `discord_channels_config.json` - Discord webhook configuration
- `data_access/alert_repository.py` - Alert database operations

## Need Help?

If you encounter issues not covered in this guide:
1. Check the logs in the console output
2. Verify your Discord webhook configuration
3. Ensure your database connection is working
4. Test with a simple condition first (e.g., `Close[-1] > 0`)
5. Use `--test-discord` to isolate Discord issues from condition evaluation issues

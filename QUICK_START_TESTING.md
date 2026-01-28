# Quick Start: Test Your Alert System

## 🚀 Get Started in 3 Steps

### Step 1: List Your Alerts
```bash
cd "C:\Users\ryan\Kovich\Stock-market-alert-app"
python test_alert_trigger.py --list
```

This shows all your alerts with their IDs.

### Step 2: Choose a Testing Method

#### Option A: Force a Test (Recommended for First Test)
```bash
python test_alert_trigger.py --test-override YOUR_ALERT_ID
```
✅ **Best for:** Quickly verifying Discord notifications work

#### Option B: Test with Real Market Data
```bash
python test_alert_trigger.py --test-alert YOUR_ALERT_ID
```
✅ **Best for:** Checking if conditions are actually met right now

#### Option C: Test Discord Routing Only
```bash
python test_alert_trigger.py --test-discord AAPL
```
✅ **Best for:** Verifying Discord webhook configuration

### Step 3: Check Discord
Go to your Discord channel and verify you received the alert message!

---

## 📋 Command Cheat Sheet

| Command | What It Does | When to Use |
|---------|-------------|-------------|
| `--list` | Show all alerts | Find alert IDs to test |
| `--test-alert ID` | Test with real data | Check if conditions are met |
| `--test-override ID` | Force trigger | Verify Discord delivery |
| `--test-discord TICKER` | Test routing only | Check Discord config |
| `--create-test TICKER` | Make test alert | Create dedicated test alert |

---

## 💡 Quick Examples

### Test if AAPL alert would trigger right now:
```bash
python test_alert_trigger.py --test-alert abc123-def456
```

### Force trigger to test Discord (doesn't modify alert):
```bash
python test_alert_trigger.py --test-override abc123-def456
```

### Test Discord routing for different tickers:
```bash
python test_alert_trigger.py --test-discord AAPL
python test_alert_trigger.py --test-discord TSLA
python test_alert_trigger.py --test-discord MSFT
```

### Create a test alert for AAPL:
```bash
python test_alert_trigger.py --create-test AAPL
# Then test it:
python test_alert_trigger.py --test-alert [RETURNED_ID]
```

---

## ✅ Success Looks Like This

```
============================================================
TESTING WITH CONDITION OVERRIDE: abc123-def456
============================================================
Original conditions: [{'conditions': 'Close[-1] > 200'}]
Test condition: Close[-1] > 0

Testing alert: My AAPL Alert (AAPL)
This will send a real Discord notification if successful!

✅ ALERT TRIGGERED!
Discord notification sent: True

⚠️ Note: Original alert conditions were NOT modified in the database
This was a temporary override for testing purposes only
```

Check your Discord channel - you should see the alert!

---

## 🔧 Troubleshooting

### "No module named 'xxx'"
Activate your virtual environment first:
```bash
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate
```

### "No alerts found"
Your alert database might be empty. Create an alert through your UI first.

### "Failed to send alert to Discord"
Check your Discord webhook configuration in `discord_channels_config.json`

---

## 📚 Need More Details?

See the full guide: [ALERT_TESTING_GUIDE.md](ALERT_TESTING_GUIDE.md)

---

## 🎯 Recommended First Test

The safest and most reliable first test:

```bash
# 1. Find an alert
python test_alert_trigger.py --list

# 2. Force it to trigger (tests everything without modifying the alert)
python test_alert_trigger.py --test-override [YOUR_ALERT_ID]

# 3. Check Discord for the message
```

This confirms:
- ✅ Your alert data loads correctly
- ✅ Price data is fetched
- ✅ Condition evaluation works
- ✅ Discord routing works
- ✅ Messages are delivered

All without modifying your actual alert or waiting for market conditions!

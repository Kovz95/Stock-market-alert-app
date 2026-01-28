# Discord Routing Guide

## How Discord Channel Routing Works

When an alert triggers, the system determines which Discord channel(s) to send the notification to based on several factors.

### Primary Channel Routing (Economy-Based)

The system routes alerts to different channels based on this priority:

1. **Ratio Alerts** → `#pairs-alerts`
   - Any alert with `ratio: "Yes"`
   - Example: SPY/QQQ ratio alerts

2. **ETF Alerts** → `#etfs-alerts`
   - Alerts where the ticker is classified as an ETF (based on `asset_type` field)
   - Example: SPY, QQQ, IWM

3. **Economy-Based Routing** → Industry-specific channels
   - Stocks are routed based on their RBICS economy classification
   - Example: AAPL → `#technology-alerts`, JPM → `#finance-alerts`
   - Requires stock metadata with economy classification

4. **Default Channel** → `#general-alerts`
   - Used when no economy classification is found
   - Fallback for any ticker not matching above rules

### Custom Channel Routing

In addition to the primary channel, alerts can **also** be sent to custom channels based on condition matching:

- Custom channels match specific alert conditions
- An alert can be sent to multiple custom channels
- Custom channels are in addition to the primary channel
- Example: All price level alerts also go to `#price-levels`

### Timeframe-Based Routing

Channels can have different configurations for different timeframes:

- **Daily alerts** (`timeframe: "1d"`) → Daily channel variants
- **Hourly alerts** (`timeframe: "1h"`) → Hourly channel variants
- **Weekly alerts** (`timeframe: "1wk"`) → Weekly channel variants

## Checking Alert Routing

### Check a Specific Alert

See exactly where an alert will be sent:

```bash
python check_alert_routing.py --alert ALERT_ID
```

**Output shows:**
- ✅ Primary channel and reason for routing
- ✅ Any custom channels that match
- ✅ Webhook configuration status
- ✅ Total number of channels alert will send to

**Example:**
```
============================================================
DISCORD ROUTING CHECK: 218355cc-a836-4b6f-8835-4af45e850fe1
============================================================

📋 Alert Information:
   ID: 218355cc-a836-4b6f-8835-4af45e850fe1
   Name: 1&1 AG Buy Alert
   Ticker: 1U1.DE
   Timeframe: 1d

🎯 Primary Channel Routing:
   Reason: No economy found - using default
   Channel: #general
   Webhook: https://discord.com/api/webhooks/...
   ✅ Webhook configured

📢 Additional Custom Channels (1):

   1. #price-levels-
      Matched Condition: price_level
      Webhook: https://discord.com/api/webhooks/...

📊 Summary:
   Total channels this alert will send to: 2
```

### Check All Alerts (Summary View)

See routing summary for all alerts grouped by channel:

```bash
python check_alert_routing.py --all
```

Shows which alerts go to which channels and identifies configuration issues.

### Check Routing for a Ticker

See where a specific ticker would be routed:

```bash
python check_alert_routing.py --ticker AAPL
```

Useful for understanding routing before creating an alert.

### Check Discord Configuration

View all configured Discord channels:

```bash
python check_alert_routing.py --config
```

Shows:
- All daily alert channels
- All hourly alert channels
- All custom channels
- Configuration status (✅ configured or ⚠️ not configured)

## Common Routing Scenarios

### Technology Stock (e.g., AAPL)
- **Economy:** Technology
- **Primary Channel:** `#technology-alerts`
- **Custom Channels:** May also go to `#price-levels` if condition matches

### Financial Stock (e.g., JPM)
- **Economy:** Financial Services
- **Primary Channel:** `#finance-alerts`
- **Custom Channels:** May also go to `#price-levels` if condition matches

### Foreign Stock with No Economy Data (e.g., 1U1.DE)
- **Economy:** None found
- **Primary Channel:** `#general-alerts` (default)
- **Custom Channels:** May also go to `#price-levels` if condition matches

### ETF (e.g., SPY)
- **Asset Type:** ETF
- **Primary Channel:** `#etfs-alerts`
- **Custom Channels:** May also go to custom channels based on conditions

### Ratio Alert (e.g., SPY/QQQ)
- **Type:** Ratio
- **Primary Channel:** `#pairs-alerts`
- **Custom Channels:** Not typically sent to custom channels

## Custom Channel Conditions

Custom channels match alerts based on their conditions. Common examples:

### Price Level Condition
- **Condition:** `price_level` (special keyword)
- **Matches:** Any alert with a price comparison condition
- **Examples:** `Close[-1] > 100`, `High[-1] < 50`, `Open[-1] >= 150.50`

### Specific Condition Match
- **Condition:** Exact condition string
- **Example:** `RSI < 30` matches only alerts with exactly that condition

### How Matching Works
1. System normalizes both channel condition and alert conditions (removes spaces, lowercase)
2. Checks each alert condition against channel condition
3. If any alert condition matches, alert is sent to that custom channel
4. Multiple custom channels can match the same alert

## Why Your Alert Might Not Send to Expected Channel

### 1. Economy Data Missing
**Symptom:** Alert goes to `#general` instead of industry-specific channel

**Cause:** Stock doesn't have economy classification in metadata

**Solution:**
```bash
# Check if stock has economy data
python check_alert_routing.py --ticker YOUR_TICKER

# If "Economy Classification: Not found", the stock needs metadata
```

### 2. Webhook Not Configured
**Symptom:** Logs show webhook errors or "not configured"

**Cause:** Discord webhook URL missing or invalid in config

**Solution:**
- Check `discord_channels_config.json`
- Ensure webhook URL is valid Discord webhook
- Verify webhook hasn't been deleted in Discord server

### 3. Custom Channel Not Triggering
**Symptom:** Alert doesn't go to expected custom channel

**Cause:** Condition doesn't match

**Solution:**
```bash
# Check what the routing system sees
python check_alert_routing.py --alert YOUR_ALERT_ID

# Verify custom channel condition matches alert condition exactly
```

### 4. Alert is Disabled
**Symptom:** No Discord message sent at all

**Cause:** Alert has `action` set to something other than `"on"`

**Solution:**
```bash
# Check alert status
python manage_alert_status.py --status ALERT_ID

# Enable if needed
python manage_alert_status.py --enable ALERT_ID
```

## Configuration Files

### discord_channels_config.json
Primary channel configuration:
- `channel_mappings` - Daily alert channels
- `channel_mappings_hourly` - Hourly alert channels
- `default_channel` - Fallback channel
- `enable_industry_routing` - Enable/disable economy-based routing

### custom_discord_channels.json
Custom channel configuration:
- Channel name
- Webhook URL
- Condition to match
- Enabled status

## Testing Discord Routing

### Test Without Triggering Alert
```bash
python test_alert_trigger.py --test-discord AAPL
```
Sends test message immediately without evaluating conditions.

### Test Full Alert Flow
```bash
python test_alert_trigger.py --test-override ALERT_ID
```
Forces alert to trigger and sends to appropriate channels.

### Check Routing Before Testing
```bash
python check_alert_routing.py --alert ALERT_ID
```
See where alert will be sent before actually triggering it.

## Best Practices

1. **Verify routing before testing** - Use `check_alert_routing.py` first
2. **Check alert status** - Ensure alert is enabled before testing
3. **Test with small set first** - Test one alert before bulk testing
4. **Monitor Discord rate limits** - Wait between tests to avoid rate limiting
5. **Keep webhooks secure** - Don't commit webhook URLs to version control
6. **Document custom channels** - Keep notes on what conditions match what channels

## Troubleshooting Commands

```bash
# Check where specific alert will go
python check_alert_routing.py --alert ALERT_ID

# Check if alert is enabled
python manage_alert_status.py --status ALERT_ID

# Enable alert for testing
python manage_alert_status.py --enable ALERT_ID

# Test Discord delivery
python test_alert_trigger.py --test-override ALERT_ID

# Check all channel configuration
python check_alert_routing.py --config

# See all disabled alerts
python manage_alert_status.py --list-disabled
```

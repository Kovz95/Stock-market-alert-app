# Discord Routing Explained

## How Alerts Are Routed to Discord Channels

Your alert system uses intelligent routing to send alerts to the appropriate Discord channels based on asset type, economy classification, and custom conditions.

## 🎯 Primary Routing Logic

When an alert triggers, the system determines which Discord channel to send to by following this priority order:

### 1. **Ratio Alerts** (Highest Priority)
- **Rule:** If `ratio == "Yes"` (pair trading alerts like AAPL/SPY)
- **Route to:** `Pairs` channel
- **Example:** AAPL/SPY ratio alert → `#pairs-alerts`

### 2. **ETF Alerts**
- **Rule:** If the ticker's `asset_type == "ETF"` in the metadata
- **Route to:** `ETFs` channel
- **Example:** SPY alert → `#etf-alerts`

### 3. **Economy-Based Routing** (Most Common)
- **Rule:** Based on RBICS Economy classification from stock metadata
- **Route to:** Economy-specific channel
- **Examples:**
  - AAPL (Technology economy) → `#technology-alerts`
  - JPM (Financial Services) → `#financial-services-alerts`
  - XOM (Energy & Utilities) → `#energy-utilities-alerts`

### 4. **Default Channel** (Fallback)
- **Rule:** If no economy found or no specific match
- **Route to:** Default channel (usually `General`)
- **Example:** Unknown ticker → `#general-alerts`

## 📊 Economy Classifications

The system uses RBICS (FactSet Revere Business Industry Classification System) economies:

| Economy | Example Tickers | Channel |
|---------|----------------|---------|
| Technology | AAPL, MSFT, GOOGL | `#technology-alerts` |
| Healthcare | JNJ, PFE, UNH | `#healthcare-alerts` |
| Financial Services | JPM, BAC, GS | `#financial-services-alerts` |
| Consumer Discretionary | AMZN, TSLA, NKE | `#consumer-discretionary-alerts` |
| Consumer Staples | PG, KO, WMT | `#consumer-staples-alerts` |
| Energy & Utilities | XOM, CVX, NEE | `#energy-utilities-alerts` |
| Industrials | BA, CAT, GE | `#industrials-alerts` |
| Materials | LIN, APD | `#materials-alerts` |
| Real Estate | AMT, PLD | `#real-estate-alerts` |
| Communication Services | META, DIS, NFLX | `#communication-services-alerts` |

## 🎨 Custom Channel Routing

In addition to primary routing, alerts can **also** be sent to custom channels based on specific conditions.

### How Custom Channels Work

Custom channels are **additive** - an alert is sent to:
1. Its primary channel (based on rules above)
2. **PLUS** any custom channels whose conditions match

### Custom Channel Matching

Custom channels match alerts based on their **conditions**. Examples:

```json
{
  "price-breakouts": {
    "enabled": true,
    "channel_name": "#price-breakouts",
    "webhook_url": "https://discord.com/api/webhooks/...",
    "condition": "price_level"
  }
}
```

- **Special keyword `price_level`**: Matches ANY alert with a price comparison condition
  - Matches: `Close[-1] > 100`, `Close[-1] < 50`, etc.
  
- **Exact condition matching**: Matches alerts with specific exact conditions
  - Condition: `RSI < 30` only matches alerts with exactly `RSI < 30`

## ⏰ Timeframe-Specific Channels

Channels can vary based on alert timeframe:

| Timeframe | Channel Suffix | Example |
|-----------|---------------|---------|
| Daily (1d) | `-alerts` | `#technology-alerts` |
| Hourly (1h) | `-hourly-alerts` | `#technology-hourly-alerts` |
| Weekly (1wk) | `-alerts` | `#technology-alerts` |

## 🔍 How to Check Routing

### Check a Specific Alert

```bash
python check_alert_routing.py --alert YOUR_ALERT_ID
```

**Output Example:**
```
============================================================
DISCORD ROUTING CHECK: abc123-def456
============================================================

📋 Alert Information:
   ID: abc123-def456
   Name: AAPL Oversold Alert
   Ticker: AAPL
   Timeframe: 1d

🎯 Primary Channel Routing:
   Reason: Economy: Technology
   Channel: #technology-alerts
   Webhook: https://discord.com/api/webhooks/...
   ✅ Webhook configured

📢 Additional Custom Channels (1):

   1. #price-breakouts
      Matched Condition: price_level
      Webhook: https://discord.com/api/webhooks/...

📊 Summary:
   Total channels this alert will send to: 2
```

### Check All Alerts Summary

```bash
python check_alert_routing.py --all
```

Shows all alerts grouped by channel:
```
📢 #technology-alerts (15 alerts)
   ✅ AAPL     | AAPL Oversold Alert
   ✅ MSFT     | MSFT Breakout
   ✅ GOOGL    | Google Support Level
   ... and 12 more

📢 #healthcare-alerts (8 alerts)
   ✅ JNJ      | JNJ RSI Alert
   ✅ PFE      | Pfizer Moving Average
   ... and 6 more
```

### Check Where a Ticker Routes

```bash
python check_alert_routing.py --ticker AAPL
```

Shows routing for any ticker without needing an alert:
```
📊 Ticker Analysis:
   Ticker: AAPL
   Economy Classification: Technology
   Is ETF: No
   Is Futures: No

🎯 Routing Decision:
   Channel: #technology-alerts
   Webhook: https://discord.com/api/webhooks/...
   ✅ Webhook is configured
```

### View All Channel Configuration

```bash
python check_alert_routing.py --config
```

Shows all configured Discord channels and their webhook status.

## 🔧 Configuration Files

### Main Channel Configuration
**File:** `discord_channels_config.json`

```json
{
  "enable_industry_routing": true,
  "default_channel": "General",
  "channel_mappings": {
    "Technology": {
      "webhook_url": "https://discord.com/api/webhooks/...",
      "channel_name": "#technology-alerts",
      "description": "Technology sector alerts"
    },
    "Healthcare": {
      "webhook_url": "https://discord.com/api/webhooks/...",
      "channel_name": "#healthcare-alerts",
      "description": "Healthcare sector alerts"
    }
  },
  "channel_mappings_hourly": {
    "Technology": {
      "webhook_url": "https://discord.com/api/webhooks/...",
      "channel_name": "#technology-hourly-alerts",
      "description": "Technology sector alerts (Hourly)"
    }
  }
}
```

### Custom Channels Configuration
**File:** `custom_discord_channels.json`

```json
{
  "price-breakouts": {
    "enabled": true,
    "channel_name": "#price-breakouts",
    "webhook_url": "https://discord.com/api/webhooks/...",
    "condition": "price_level",
    "description": "All price level breakout alerts"
  },
  "oversold-stocks": {
    "enabled": true,
    "channel_name": "#oversold-opportunities",
    "webhook_url": "https://discord.com/api/webhooks/...",
    "condition": "RSI < 30",
    "description": "Oversold stocks (RSI below 30)"
  }
}
```

## 📍 Routing in Test Script

When you test an alert, the routing information is automatically displayed:

```bash
python test_alert_trigger.py --test-override abc123-def456
```

**Output includes:**
```
📢 Discord Routing:
   Primary Channel: #technology-alerts
   Additional Custom Channels: 1
      • #price-breakouts
```

## 🐛 Troubleshooting Routing Issues

### Alert Not Going to Expected Channel

**Check the economy classification:**
```bash
python check_alert_routing.py --ticker AAPL
```

**Verify the alert's routing:**
```bash
python check_alert_routing.py --alert YOUR_ALERT_ID
```

### Webhook Not Configured Error

If you see `⚠️ PLACEHOLDER (not configured)`:

1. Open `discord_channels_config.json`
2. Find the channel name
3. Update the `webhook_url` with your actual Discord webhook
4. Save the file

### Custom Channel Not Matching

If a custom channel isn't matching your alert:

1. Check the `condition` field in `custom_discord_channels.json`
2. Verify the condition exactly matches your alert's condition
3. Use `price_level` keyword for any price comparison
4. Ensure `enabled: true`

### Alert Going to Wrong Channel

Possible reasons:
1. **Wrong ticker metadata** - Update stock metadata with correct economy
2. **Missing economy** - Ticker not in metadata, falls back to default
3. **ETF misclassification** - Check `asset_type` field in metadata
4. **Timeframe mismatch** - Hourly alerts use different channels

## 🔄 Routing Priority Summary

```
1. Is it a ratio alert (ticker1/ticker2)?
   YES → Pairs channel
   NO  → Continue

2. Is it an ETF?
   YES → ETFs channel
   NO  → Continue

3. Does ticker have economy classification?
   YES → Economy-specific channel
   NO  → Default channel

4. PLUS: Send to any matching custom channels
```

## 📚 Related Commands

```bash
# Check specific alert routing
python check_alert_routing.py --alert ALERT_ID

# Check all alerts by channel
python check_alert_routing.py --all

# Check ticker routing
python check_alert_routing.py --ticker SYMBOL

# View channel configuration
python check_alert_routing.py --config

# Test alert (shows routing info)
python test_alert_trigger.py --test-override ALERT_ID
```

## 💡 Pro Tips

1. **Multiple channels are intentional** - Alerts can go to both primary and custom channels
2. **Hourly alerts auto-route** - Hourly timeframe automatically uses hourly-specific channels
3. **Test routing before creating alerts** - Use `--ticker` to preview where new alerts will go
4. **Custom channels are powerful** - Create themed channels for specific trading strategies
5. **Economy data matters** - Keep your stock metadata updated for accurate routing

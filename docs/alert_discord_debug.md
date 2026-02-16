# Alert → Discord Flow: What This App Does & Debugging Checklist

## What This App Does (High Level)

1. **Schedulers** run on a schedule (by exchange/market):
   - **Stock (daily/weekly):** `auto_scheduler_v2.py` — after market close, updates prices for that exchange, then runs alert checks.
   - **Stock (hourly):** `hourly_data_scheduler.py` — updates hourly price data, then runs alert checks for "hourly" alerts.
   - **Futures:** `futures_scheduler.py` — updates futures prices from IB, then runs futures alert checks.

2. **Price data** is written to the DB by:
   - **Stocks:** `ScheduledPriceUpdater` / `OptimizedDailyPriceCollector` (daily) and `HourlyPriceCollector` (hourly) → `DailyPriceRepository`.
   - **Futures:** `futures_price_updater.py` → `continuous_prices` (Postgres).

3. **Alert checks** load alerts from the alert repository, get price data, evaluate conditions, and if triggered call **Discord routing** to send the message.

4. **Discord** routing (`discord_routing.py`) picks a channel by economy/ETF/Pairs and sends via webhook. Custom channels can also receive alerts based on condition matching.

---

## End-to-End Flow (Stocks)

```
auto_scheduler_v2 (or hourly_data_scheduler)
  → update_prices_for_exchanges(...)   // writes daily/hourly prices to DB
  → run_alert_checks(exchanges, "daily" | "weekly" | "hourly")
       → list_alerts() + filter by exchange + timeframe
       → StockAlertChecker().check_alerts(relevant_alerts, timeframe_key)
            → for each alert: get_price_data(ticker) → evaluate_alert() → if triggered:
                 → send_economy_discord_alert(alert, message)
                 → update_alert(last_triggered)
```

Important: **StockAlertChecker** gets price data from the **FMP API** (`FMPDataFetcher.get_historical_data`), not from the database. So alerts are evaluated on live FMP data. If FMP is down, rate-limited, or returns no data, you get `no_data` and no Discord send.

---

## Bugs That Were Fixed (As of This Doc)

1. **`scheduled_price_updater.run_alert_checks`**  
   It only counted alerts and never called `StockAlertChecker`. So any code path that used this function never actually ran alert logic or sent Discord. **Fix:** It now instantiates `StockAlertChecker` and calls `check_alerts(relevant_alerts, timeframe_key)` and merges the returned stats.

2. **Hourly alerts never checked**  
   In `auto_scheduler_v2.run_alert_checks`, when `timeframe_key == "hourly"` there was no branch adding alerts to `relevant_alerts`, so the hourly scheduler was effectively evaluating 0 alerts. **Fix:** Added an `elif timeframe_key == "hourly"` branch (and aligned `scheduled_price_updater` + `StockAlertChecker.check_alerts` to include `"hourly"` / `"1h"` / `"1hr"`).

---

## Where Schedulers Get Their Discord Channels

There are **two separate** Discord channel systems:

### 1. Scheduler status messages (start / complete / error)

Used by **daily**, **weekly**, and **hourly** schedulers to post “Alert Check Started” and “Alert Check Complete” (and errors) to a dedicated status channel.

| Scheduler | Config source | Config key | Fallback |
|-----------|----------------|------------|----------|
| **Daily** | `discord_channels_config` (document store or `discord_channels_config.json`) | `logging_channels.Daily_Scheduler_Status` | `scheduler_config.scheduler_webhook.url` |
| **Weekly** | same | `logging_channels.Weekly_Scheduler_Status` | `scheduler_config.scheduler_webhook.url` |
| **Hourly** | same | `logging_channels.Hourly_Scheduler_Status` | `scheduler_config.scheduler_webhook.url` |

- **Code:** `src/services/scheduler_discord.py` (BaseSchedulerDiscord, DailySchedulerDiscord, WeeklySchedulerDiscord) and `hourly_scheduler_discord.py` (HourlySchedulerDiscord). Each notifier’s `_load_config()` reads `config["logging_channels"][config_key]` for `webhook_url`.
- If a job-specific key has no `webhook_url`, the code falls back to the single **scheduler status webhook** in `scheduler_config` (document store or `scheduler_config.json`). So daily and weekly often end up using that “old” shared channel when you haven’t set up `Daily_Scheduler_Status` / `Weekly_Scheduler_Status`.

**To give daily and weekly their own channels:** Add entries under `logging_channels` with the right webhook URLs, for example:

```json
"logging_channels": {
  "Daily_Scheduler_Status":  { "webhook_url": "https://discord.com/api/webhooks/...", "channel_name": "#daily-scheduler" },
  "Weekly_Scheduler_Status": { "webhook_url": "https://discord.com/api/webhooks/...", "channel_name": "#weekly-scheduler" },
  "Hourly_Scheduler_Status": { "webhook_url": "https://discord.com/api/webhooks/...", "channel_name": "#hourly-scheduler" }
}
```

### 2. Alert routing (where triggered stock alerts are sent)

When an alert **triggers**, it is sent to a channel chosen by **economy** (and optionally ETF/Pairs) and **timeframe**. The timeframe selects which mapping is used:

| Alert timeframe | Mapping used | Where it comes from |
|-----------------|--------------|---------------------|
| **Hourly** (1h, 1hr, hourly) | `channel_mappings_hourly` | Built in `discord_routing.py` from `channel_mappings`; each economy can override with its own `webhook_url` and `channel_name` in `channel_mappings_hourly`. |
| **Daily** (1d, daily, or missing) | `channel_mappings_daily` | Same: built from base `channel_mappings`; daily-specific overrides in `channel_mappings_daily`. |
| **Weekly** (1wk, 1w, weekly) | `channel_mappings_weekly` | Same: built from base `channel_mappings`; weekly-specific overrides in `channel_mappings_weekly`. |

- **Code:** `discord_routing.py` → `_load_config()` builds `channel_mappings_daily`, `channel_mappings_weekly`, and `channel_mappings_hourly` from the base `channel_mappings`. For each economy in the base map:
  - **Daily/Weekly:** If you don’t add a `channel_mappings_daily` or `channel_mappings_weekly` entry for that economy, it **inherits** `webhook_url` (and derived channel_name) from the base. So daily and weekly alerts use the **same webhooks as the old base channels** until you add explicit daily/weekly entries with different webhooks.
  - **Hourly:** Same inheritance; you’ve overridden hourly with your own channels, so hourly has its own webhooks.

**To give daily and weekly alerts their own channels:** In `discord_channels_config`, define `channel_mappings_daily` and/or `channel_mappings_weekly` with the same economy keys (e.g. `General`, `US`, `China`, etc.) and set each entry’s `webhook_url` (and optionally `channel_name`) to the new channel. If an economy is missing from `channel_mappings_daily` or `channel_mappings_weekly`, it keeps using the base mapping (old channel).

---

## Why Alerts Might Not Reach Discord: Checklist

### 1. Alert logic (condition never true)

- **Conditions:** Stored in `alert["conditions"]` (or legacy `condition`). Evaluated with `evaluate_expression_list()` in `src/services/backend.py` (e.g. `Close[-1] > 100`).
- **Check:** Run a single alert manually with `scripts/analysis/test_alert_trigger.py` (or equivalent) and confirm the condition evaluates to true for the current data.
- **Data source:** Stock checker uses **FMP** only. If FMP returns no data or different data, the condition might not trigger.

### 2. Scheduler / job not running

- **Daily/Weekly:** Is `auto_scheduler_v2.py` running? Check `scheduler_status.json` (or document store key `scheduler_status`), lock file `scheduler_v2.lock`, and logs.
- **Hourly:** Is `hourly_data_scheduler.py` running? Check its lock/status and logs.
- **Futures:** Is `futures_scheduler.py` (or the service that runs it) running?
- **Time windows:** Daily jobs are scheduled after market close (exchange-calendars). If the job is scheduled but never fires, check exchange schedule config and cron/trigger times.

### 3. No (or wrong) price data for the check

- **Stocks:** Checker uses **FMP**, not the DB. So:
  - FMP API key must be set and valid.
  - If you expect to evaluate on “just updated” DB data, that’s not what the stock checker does today; it always refetches from FMP. No data → `no_data`, no Discord.
- **Futures:** Checker reads from `continuous_prices`. Ensure `futures_price_updater` (or equivalent) has run so that table has recent data for the symbol.

### 4. Discord routing / webhook

- **Config file:** `discord_channels_config.json` (or document store key `discord_channels_config`). Must have real webhook URLs (not placeholders like `YOUR_GENERAL_WEBHOOK_URL_HERE`). The actual file is not committed (it contains webhook URLs); use **`discord_channels_config.example.json`** in the project root as a template, then copy to `discord_channels_config.json` and fill in webhooks, or configure via the **Discord Management** page (which persists to the document store).
- **Channel selection:** `determine_alert_channel(alert)` uses:
  - `enable_industry_routing` → economy/ETF/Pairs/default.
  - Ticker → economy from metadata (`rbics_economy`). If metadata has no economy for the ticker, it falls back to default channel.
- **Timeframe:** Alert’s `timeframe` (e.g. `daily`, `1d`, `hourly`, `1h`) is normalized and used to choose `channel_mappings` vs `channel_mappings_hourly`. Wrong or missing timeframe can pick a mapping that has no valid webhook.
- **Logs:** Look for “No Discord webhook available”, “Discord webhook not configured”, “Failed to send Discord notification”, and “Alert sent successfully”.

### 5. Alert filtered out before check

- **Action:** Alerts with `action == "off"` are skipped.
- **Already triggered today:** `should_skip_alert()` skips if `last_triggered` is today (by date).
- **Exchange/timeframe:** Only alerts whose exchange is in the job’s exchange list and whose timeframe matches the job (`daily`/`weekly`/`hourly`) are in `relevant_alerts`. If your alert’s `timeframe` is stored as e.g. `"1d"` or `"hourly"`, the filters now treat those as daily/hourly; if you use a different string, it may be excluded.

### 6. Errors during send

- **Rate limiting:** Discord router can use a rate limiter. Check logs for rate-limit or backoff messages.
- **Network/timeouts:** Failed HTTP to webhook → “Failed to send Discord notification”. Check response status and body in logs.

---

## Quick Verification Commands

- **Test one stock alert (condition + Discord):**  
  `python scripts/analysis/test_alert_trigger.py --alert-id <id>` (or as your script supports).

- **Check scheduler status:**  
  `python scripts/analysis/check_scheduler_status.py` (or equivalent).

- **Check Discord routing for an alert:**  
  Use `scripts/analysis/check_alert_routing.py` or the Discord Management page to see which channel and webhook would be used for a given alert.

- **Inspect alert audits:**  
  Query or UI for `alert_audits`: see whether checks ran, whether price data was pulled, and whether conditions were evaluated/triggered.

---

## Summary

- **App:** Schedulers update prices (DB), then run alert checks. Stock checker uses **FMP** for prices; futures checker uses DB. Triggered alerts go through `send_economy_discord_alert` and are routed by economy/timeframe to Discord webhooks.
- **Fixes applied:** `scheduled_price_updater.run_alert_checks` now runs the checker; hourly timeframe is included in scheduler and checker so hourly alerts are actually evaluated.
- **If alerts still don’t reach Discord:** Confirm the condition fires, the right scheduler is running, FMP (or futures DB) has data, Discord config has valid webhooks, and the chosen channel mapping has a valid URL; then check logs for send success/failure and rate limits.

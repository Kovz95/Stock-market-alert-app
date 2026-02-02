# Stock Alert Workflow — Detailed README

This document describes how the stock alert system is **scheduled**, where it **gets data**, how **conditions are calculated**, and how **alerts are sent to Discord**.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Scheduling](#2-scheduling)
3. [Data Sources](#3-data-sources)
4. [Alert Calculation](#4-alert-calculation)
5. [Sending to Discord](#5-sending-to-discord)
6. [End-to-End Flows](#6-end-to-end-flows)
7. [Configuration & Deployment](#7-configuration--deployment)

---

## 1. Overview

The app runs three separate alert pipelines:

| Pipeline | Asset type | Scheduler | Data for alerts | Alert checker |
|----------|------------|-----------|-----------------|---------------|
| **Daily/Weekly stocks** | Stocks, ETFs | `auto_scheduler_v2.py` | FMP API (live) | `StockAlertChecker` |
| **Hourly stocks** | Stocks, ETFs | `hourly_data_scheduler.py` | FMP API (live) | `StockAlertChecker` |
| **Futures** | Futures | `futures_scheduler.py` | Postgres `continuous_prices` | `FuturesAlertChecker` |

Important: **Stock** alert evaluation always uses **live FMP data** (not the database). The database is used to store daily/hourly prices for the app’s own use; alert conditions are evaluated against data fetched from FMP at check time.

---

## 2. Scheduling

### 2.1 Stock daily and weekly — `auto_scheduler_v2.py`

- **Location:** `src/services/auto_scheduler_v2.py`
- **Engine:** APScheduler `BackgroundScheduler` (UTC).

**How it’s scheduled:**

- Exchange closing times come from **`src/config/exchange_schedule_config.py`** (39+ exchanges).
- Closing times are in **UTC**, with a **40-minute delay** after market close for data availability.
- For each exchange, two cron jobs are registered:
  - **Daily job:** Runs at `(close_time)` on the exchange’s trading days (e.g. `mon-fri`).
  - **Weekly job:** Runs at the same time on the exchange’s **last trading day of the week** (e.g. `fri`).
- Asian/Pacific markets that close after midnight ET are treated as the previous US calendar day.

**Job execution:**

1. **Per exchange, per job type (daily or weekly):**
   - `execute_exchange_job(exchange_name, job_type)` runs in a **subprocess** with a configurable timeout (default 900s).
   - **Step 1:** `update_prices_for_exchanges([exchange_name], resample_weekly=...)`  
     → Uses `ScheduledPriceUpdater` / `OptimizedDailyPriceCollector` to fetch daily (and optionally weekly) prices from FMP and write them to the DB.
   - **Step 2:** `run_alert_checks([exchange_name], timeframe_key)`  
     → Filters alerts by exchange and timeframe (`daily` or `weekly`), then runs `StockAlertChecker().check_alerts(...)`.

2. **Additional job:**
   - **Full daily update:** All tickers at **23:00 UTC** via `run_full_daily_update` (writes to DB only; does not run alert checks).

3. **Locking:** Job lock per `(job_type, exchange)` prevents overlapping runs. Heartbeat job updates status periodically.

**Deployment:** Often run as a long-lived process or via `deploy/auto-scheduler.service` (see `deploy/README.md`).

---

### 2.2 Stock hourly — `hourly_data_scheduler.py`

- **Location:** `src/services/hourly_data_scheduler.py`
- **Engine:** APScheduler `BackgroundScheduler` (UTC).

**How it’s scheduled:**

- Exchanges are grouped by “style” (e.g. hour, half, quarter) from `exchange_schedule_config` and calendar helpers.
- Cron jobs run **every hour** at fixed minutes:
  - e.g. `minute=5` for one group, `minute=35` for another, `minute=20` for another.
- Any exchange not in a predefined style runs at `minute=5` (`hourly_misc`).

**Job execution:**

1. **`update_hourly_data(exchange_filter=...)`**
   - Uses **`HourlyPriceCollector`** to fetch hourly data (from FMP) and write to the DB.
   - Then calls **`run_alert_checks(open_exchanges, "hourly")`** (imported from `auto_scheduler_v2`), which:
     - Filters alerts by those exchanges and **hourly** timeframe.
     - Runs **`StockAlertChecker().check_alerts(..., "hourly")`**.

So: **hourly** pipeline = hourly price update to DB + hourly alert check using **FMP** (not DB) for condition evaluation.

---

### 2.3 Futures — `futures_scheduler.py`

- **Location:** `src/services/futures_scheduler.py`
- **Engine:** APScheduler `BackgroundScheduler` (UTC).

**How it’s scheduled:**

- Config (e.g. `futures_scheduler_config.json` or document store) defines **`update_times`** (defaults like `["06:00", "12:00", "16:00", "20:00"]` UTC).
- One cron job per time: `execute_futures_job` at that hour/minute.
- **IB hours:** If `ib_hours` is set (e.g. 05:00–23:00 UTC), the job body checks current time and **skips** execution outside that window (no price update, no alert check).

**Job execution:**

1. **Single combined job** (in a subprocess with timeout):
   - **Step 1:** `run_price_update()`  
     → Updates futures prices (e.g. from Interactive Brokers) and writes to Postgres table **`continuous_prices`**.
   - **Step 2:** Short sleep (e.g. 5s), then **`run_alert_checks()`**  
     → Loads **futures** alerts, runs **`FuturesAlertChecker`**, which reads price data from **`continuous_prices`** (not FMP).

---

## 3. Data Sources

### 3.1 Stocks (daily and weekly)

- **Written to DB:**  
  - **`ScheduledPriceUpdater`** / **`OptimizedDailyPriceCollector`** (in `scheduled_price_updater.py`, `backend_fmp_optimized.py`) fetch from **FMP** and write to **`DailyPriceRepository`** (Postgres).
- **Used for alert evaluation:**  
  - **`StockAlertChecker`** does **not** read from the DB. It uses **`FMPDataFetcher`** (`backend_fmp.py`):
    - Daily: `get_historical_data(ticker, period="1day")`.
    - Weekly: daily from FMP then resampled to weekly in FMP layer.
    - Hourly: `get_historical_data(ticker, period="1hour")`.
  - API key: **`FMP_API_KEY`** (env or default in code).

So: DB is for storing/displaying prices; **alert logic always uses live FMP** at check time. If FMP is down or returns no data, the checker reports `no_data` and does not send Discord.

### 3.2 Stocks (hourly)

- **Written to DB:**  
  - **`HourlyPriceCollector`** fetches hourly data (from FMP) and writes to the same DB (e.g. daily_price / hourly tables via repository).
- **Used for alert evaluation:**  
  - Again **FMP** via **`StockAlertChecker.get_price_data(ticker, "1h")`** — not the DB.

### 3.3 Futures

- **Written to DB:**  
  - Futures price updater (e.g. `futures_price_updater.py`) writes to Postgres table **`continuous_prices`** (symbol, date, open, high, low, close, volume).
- **Used for alert evaluation:**  
  - **`FuturesAlertChecker.get_price_data(symbol)`** reads from **`continuous_prices`** only (no FMP). Optional adjustment (e.g. Panama) can be applied in the checker.

---

## 4. Alert Calculation

### 4.1 Where alerts are stored

- **Repository:** `src/data_access/alert_repository.py`
- **Storage:** Postgres table **`alerts`** (and optional Redis cache).
- **Fields (conceptually):** `alert_id`, `name`, `ticker` (or `ticker1`/`ticker2` for ratio alerts), `conditions`, `combination_logic`, `timeframe`, `exchange`, `action`, `last_triggered`, etc.
- **Listing:** **`list_alerts()`** returns all alerts; schedulers and checkers filter by exchange, timeframe, and `action != "off"`.

### 4.2 Condition format

- Conditions are stored in **`conditions`** (list) or legacy **`condition`** (single string).
- Each condition is a **string expression**, e.g.:
  - `Close[-1] > sma(20)[-1]`
  - `rsi(14)[-1] < 30`
  - `(EWO(...)[-1] > 0) & (EWO(...)[-2] <= 0)`
  - Ichimoku and other indicator expressions.
- **Combination logic** is in **`combination_logic`** (stocks) or **`entry_combination`** (futures), e.g. `AND`, `OR`, or expressions like `1 AND (2 OR 3)`.

### 4.3 How conditions are evaluated

- **Stock checker:**  
  **`StockAlertChecker.evaluate_alert(alert, df)`** (in `src/services/stock_alert_checker.py`):
  1. **`extract_conditions(alert)`** → list of condition strings.
  2. **`evaluate_expression_list(df, conditions, combination, vals)`** from **`src/services/backend.py`**.
- **Futures checker:**  
  **`FuturesAlertChecker.evaluate_alert(alert, df)`** does the same with **`evaluate_expression_list`** and **`entry_combination`**.

**`evaluate_expression_list`** (in `backend.py`):

- Takes the OHLCV **DataFrame** and the list of condition strings.
- **Combination** can be:
  - `"AND"` / `"1"` → all conditions must be True.
  - `"OR"` → any condition True.
  - `"1 AND 2"`, `"(1 AND 2) OR 3"`, etc. for complex logic.
- Each single condition is evaluated by **`evaluate_expression(df, exp, ...)`**.

**`evaluate_expression`** (in `backend.py`):

- Parses and evaluates one expression against the last bar(s) of the DataFrame (e.g. `Close[-1]`, `sma(20)[-1]`).
- Uses:
  - **`simplify_conditions`** and **`apply_function`** for indicator/price lookups (SMA, RSI, EWO, Ichimoku, etc. from `src/utils/indicators.py` and backend).
  - Special handling for EWO cross, Ichimoku, then simple comparisons, then a Python-eval fallback.
- Returns **True** if the condition is met, **False** otherwise.

So: **one** OHLCV DataFrame (from FMP for stocks, or from `continuous_prices` for futures) + list of condition strings + combination logic → single boolean “triggered or not.”

### 4.4 After a trigger (stocks)

- **Message:** **`StockAlertChecker.format_alert_message(alert, df)`** builds the text.
- **Discord:** **`send_economy_discord_alert(alert, message)`** (see below).
- **Persistence:** **`update_alert(alert_id, {"last_triggered": ...})`** in the alert repository so the same alert can be skipped or throttled (e.g. “already triggered today”) depending on app logic.

---

## 5. Sending to Discord

### 5.1 Entry point

- **Stock/futures triggered alert:**  
  **`send_economy_discord_alert(alert, message)`** in **`src/services/discord_routing.py`** (wrapper around **`DiscordEconomyRouter.send_economy_alert(alert, message)`**).

### 5.2 Channel selection — `determine_alert_channel(alert)`

- **Config:** `discord_channels_config.json` (or document store key **`discord_channels_config`**). Contains:
  - **`channel_mappings`** (daily/weekly) and **`channel_mappings_hourly`** (hourly).
  - Each entry: economy/channel name → `channel_name`, `webhook_url`, `description`.
- **Routing rules:**
  1. **Ratio alerts** (`ratio == "Yes"` or `ticker1`/`ticker2`) → **Pairs** channel (from mapping).
  2. **ETF:** If ticker is classified as ETF (e.g. from metadata `asset_type`) → **ETFs** channel.
  3. **Stocks:** Economy from metadata (e.g. **`rbics_economy`**) → channel with that economy name in the mapping.
  4. **Timeframe:** Alert’s **`timeframe`** (e.g. `daily`, `1d`, `hourly`, `1h`) selects **`channel_mappings`** vs **`channel_mappings_hourly`** so hourly alerts can go to different webhooks.
  5. **Fallback:** **`default_channel`** (e.g. General) if no economy/ETF/Pairs match or webhook missing.

### 5.3 Custom channels

- **Config:** `custom_discord_channels.json` (or document store **`custom_discord_channels`**).
- **Logic:** **`get_custom_channels_for_alert(alert)`** checks each custom channel’s **condition** (or keyword like **`price_level`**) against the alert’s condition(s). If it matches, that channel is added to the list of targets.
- **Result:** One alert can be sent to the **economy/default** channel **and** to any **matching custom** channels.

### 5.4 Sending the message

- **Payload:** Discord webhook payload with:
  - **`content`:** Formatted message (optionally with divider/spacing from **`message_spacing`** config).
  - **`username`:** e.g. `"Stock Alert Bot"`.
  - **`avatar_url`:** Optional.
- **HTTP:** **`requests.post(webhook_url, json=payload, timeout=10)`**. Success = status **204** (or 200).
- **Rate limiting:** Optional **`discord_rate_limiter`** in `src/utils/discord_rate_limiter.py` can queue/throttle webhook POSTs to avoid Discord rate limits.
- **Order:**  
  1. Send to the **economy/ETF/Pairs** channel from **determine_alert_channel**.  
  2. Send to each **matching custom** channel.

If the webhook URL is missing or still the placeholder (e.g. `YOUR_GENERAL_WEBHOOK_URL_HERE`), the send is skipped and a warning is logged.

---

## 6. End-to-End Flows

### 6.1 Stock daily (or weekly)

```
auto_scheduler_v2 (cron per exchange)
  → execute_exchange_job(exchange, "daily" | "weekly")
      → update_prices_for_exchanges([exchange], resample_weekly=...)
      → run_alert_checks([exchange], "daily" | "weekly")
          → list_alerts() → filter by exchange + timeframe
          → StockAlertChecker().check_alerts(relevant_alerts, timeframe_key)
              → for each alert:
                  → get_price_data(ticker, timeframe)  [FMP]
                  → evaluate_alert(alert, df)          [backend.evaluate_expression_list]
                  → if triggered:
                      → format_alert_message(alert, df)
                      → send_economy_discord_alert(alert, message)
                      → update_alert(alert_id, { last_triggered })
```

### 6.2 Stock hourly

```
hourly_data_scheduler (cron every hour, per group)
  → update_hourly_data(exchange_filter=...)
      → HourlyPriceCollector: fetch hourly from FMP → write DB
      → run_alert_checks(open_exchanges, "hourly")
          → list_alerts() → filter by exchange + "hourly"
          → StockAlertChecker().check_alerts(relevant_alerts, "hourly")
              → [same as above: FMP get_price_data → evaluate → Discord + update_alert]
```

### 6.3 Futures

```
futures_scheduler (cron at update_times, within IB hours)
  → execute_futures_job()
      → run_price_update()        → write continuous_prices (Postgres)
      → run_alert_checks()       → FuturesAlertChecker
          → load futures alerts
          → for each: get_price_data(symbol) from continuous_prices
          → evaluate_alert(alert, df)
          → if triggered: send to Discord (same routing pattern if used) + update
```

---

## 7. Configuration & Deployment

- **Scheduler config (stock):** Document store key **`scheduler_config`** or `scheduler_config.json`: e.g. **`scheduler_webhook`** (Discord for job start/complete), **`notification_settings`** (timeouts, notifications).
- **Exchange times:** **`src/config/exchange_schedule_config.py`** — `EXCHANGE_SCHEDULES`, `get_exchanges_by_closing_time()`, `get_market_days_for_exchange()`.
- **Futures:** **`futures_scheduler_config.json`** or document store: **`update_times`**, **`ib_hours`**, **`notification_settings`**, etc.
- **Discord:** **`discord_channels_config.json`** (channel mappings, default channel, industry routing), **`custom_discord_channels.json`** (custom channels and conditions).
- **FMP:** **`FMP_API_KEY`** environment variable (or default in `FMPDataFetcher`).
- **Deploy:** Systemd units under **`deploy/`** (e.g. `auto-scheduler.service`, `hourly-data-scheduler.service`, `futures-scheduler.service`, `scheduler-watchdog.service`). See **`deploy/README.md`**.

For **why an alert might not reach Discord** (condition, scheduler, data, webhook, filtering), see **`docs/alert_discord_debug.md`**.

---

*This README reflects the codebase as of the last update. Key modules: `auto_scheduler_v2.py`, `hourly_data_scheduler.py`, `futures_scheduler.py`, `scheduled_price_updater.py`, `stock_alert_checker.py`, `futures_alert_checker.py`, `backend.py`, `discord_routing.py`, `alert_repository.py`, `exchange_schedule_config.py`.*

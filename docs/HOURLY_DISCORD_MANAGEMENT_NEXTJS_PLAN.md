# Migration Plan: Hourly Discord Management → Next.js

This plan covers migrating the Streamlit page **Hourly Discord Management** (`pages/Hourly_Discord_Management.py`) to the Next.js app (`apps/ui/web`).

---

## 1. Current Streamlit Page Summary

### 1.1 Purpose

- Configure **hourly timeframe** Discord webhooks per economy/special channel.
- Copy daily webhook URLs into hourly config in one click.
- Test routing (ticker → economy → channel) and send a test message to the resolved channel.

### 1.2 Features (and data flow)

| Feature | Streamlit implementation | Data / backend |
|--------|----------------------------|----------------|
| **Metrics** | 3 `st.metric` cards | `DiscordEconomyRouter.config`, `get_available_channels(timeframe='hourly')` |
| **Copy Daily → Hourly** | Button; in-memory merge then `save_document` + `clear_cache` | `document_store.save_document("discord_channels_config", config)`, `clear_cache("discord_channels_config")` |
| **Economy channels tab** | Expanders per channel; password webhook input + Save | `config['channel_mappings_hourly']`, `router.update_channel_config(name, webhook, timeframe='hourly')` (writes via `save_document`) |
| **Special channels tab** | Same pattern for ETFs, Pairs, General, Futures, Failed_Price_Updates | Same config and `update_channel_config` |
| **Test ticker** | Text input + "Test Hourly Channel" → `get_stock_economy_classification(ticker)` | `DiscordEconomyRouter.get_stock_economy()` (uses `stock_metadata` / `cleaned_data.csv`) |
| **Send test message** | Button; `requests.post(webhook_url, json=payload)` | Webhook URL from session state; **must stay server-side** so URLs are not exposed to the client |

### 1.3 Backend dependencies (Python)

- **`src.services.discord_routing`**: `DiscordEconomyRouter`, `get_stock_economy_classification`
  - Config: `load_document("discord_channels_config", fallback_path="discord_channels_config.json")` → `channel_mappings_hourly`, `enable_industry_routing`, `log_routing_decisions`
  - Persistence: `save_document("discord_channels_config", config)`, `clear_cache("discord_channels_config")`
- **`src.data_access.document_store`**: `load_document`, `save_document`, `clear_cache` (PostgreSQL `app_documents` + optional Redis)
- **Stock economy lookup**: `fetch_stock_metadata_map()` (e.g. `metadata_repository`) → `rbics_economy` by symbol; fallback `cleaned_data.csv`

---

## 2. Target: Next.js App

### 2.1 Route and sidebar

- **Route**: e.g. `/discord/hourly` (recommended: `apps/ui/web/app/discord/hourly/page.tsx`).
- **Sidebar**: In `app-sidebar.tsx`, set the “Hourly Discord Management” item to `url: "/discord/hourly"` (replace the current `#`).

### 2.2 Page structure (mirror of Streamlit)

1. **Header**
   - Title: “Hourly Discord Channel Management”
   - Short description: configure dedicated Discord webhooks for **hourly** alerts.

2. **Metrics row** (3 cards)
   - Hourly Routing: “Enabled” / “Disabled” from `config.enable_industry_routing`.
   - Configured Channels: “X / Y” from hourly channel list (configured = has webhook and not placeholder).
   - Routing Logs: “Enabled” / “Disabled” from `config.log_routing_decisions`.

3. **Copy Daily → Hourly**
   - Single button: “Copy Daily Webhooks → Hourly”.
   - On success: toast and refetch config/channel list.

4. **Tabs: Economy Channels | Special Channels**
   - **Economy**: All hourly channels whose name is **not** in `['ETFs','Pairs','General','Futures','Failed_Price_Updates']`.
   - **Special**: Only those five.
   - Per channel (in expandable sections, e.g. Accordion or Collapsible):
     - Display name and description.
     - Password-type input for webhook URL (prefilled with current value; never send raw webhook to client if you later add a “masked” display).
     - “Save” button → update that channel’s hourly webhook only.

5. **Test Hourly Routing**
   - Ticker input (placeholder e.g. “AAPL, MSFT, XOM”).
   - “Test Hourly Channel” button → resolve economy and hourly channel (name + whether webhook is configured).
   - Show result: e.g. “Ticker X → Economy Y → Hourly channel Z”.
   - If a webhook is configured: show “Send test message to this channel” button. **Sending must be a server action (or API route)** so the webhook URL is never exposed to the client; server sends the test message and returns success/failure.

### 2.3 UI components (align with existing app)

- Use existing Shadcn components: `Card`, `Tabs`, `Button`, `Input` (type password for webhooks), `Accordion` or `Collapsible`, toast (e.g. Sonner).
- Layout: same pattern as other app pages (e.g. `app/alerts/page.tsx`) with a main content area and consistent spacing.

---

## 3. Backend API Options

The Next.js app needs a backend that:

1. **Read** Discord config (at least `channel_mappings_hourly`, `channel_mappings_daily`, `enable_industry_routing`, `log_routing_decisions`).
2. **Write** Discord config: update one channel’s hourly webhook; optionally “copy daily webhooks into hourly” and persist.
3. **Resolve economy** for a ticker (same logic as `get_stock_economy_classification`).
4. **Send test message** to a given webhook (server-side only).

Config lives in **PostgreSQL** `app_documents` (key `discord_channels_config`). Economy comes from **stock_metadata** (or fallback CSV) in the current Python code.

### Option A: Python REST API (fastest, reuses current logic)

- Add a small **FastAPI** (or Flask) app (e.g. under `apps/discord_api` or `api/`) that:
  - Uses existing `document_store` and `DiscordEconomyRouter` (and `metadata_repository` for economy).
  - Exposes REST endpoints, for example:
    - `GET /discord/config` → full config (or a “hourly” view with channel list and metrics).
    - `POST /discord/config/hourly/copy-daily` → copy daily → hourly, persist, clear cache.
    - `PATCH /discord/config/hourly/channels/:name` → body `{ "webhook_url": "..." }` → update one channel, persist, clear cache.
    - `GET /discord/resolve-economy?ticker=AAPL` → `{ "economy": "Technology", "hourlyChannel": "...", "webhookConfigured": true }` (no webhook URL in response).
    - `POST /discord/test-message` → body `{ "channel_name": "Technology" }` or `{ "webhook_id": "..." }`; server looks up webhook for that hourly channel, sends test message, returns success/error.
- Next.js calls these from **server actions** (or route handlers) so that:
  - Webhook URLs are never sent to the browser.
  - Test message is sent only from the backend.
- **Pros**: Reuses all Python logic and document_store; minimal duplication.  
- **Cons**: Extra Python service to run and deploy; need to secure it (e.g. same network as Next.js, or auth).

### Option B: Go gRPC Discord service (implemented)

- A Go gRPC service **discord_service** (`apps/grpc/discord_service/`) implements:
  - **Reads/writes** `app_documents` for `discord_channels_config` (same JSON shape as Python).
  - **GetHourlyDiscordConfig**: returns metrics and channel list (no webhook URLs sent to client).
  - **ResolveHourlyChannelForTicker**: reads from `stock_metadata` by symbol (ticker, UPPER, ticker+`-US`), returns economy and whether webhook is configured; no webhook URL in response.
  - **UpdateHourlyChannelWebhook**, **CopyDailyToHourly**: update the JSON and persist to `app_documents`.
  - **SendHourlyTestMessage**: accepts hourly channel name; server resolves webhook from config, sends HTTP POST to Discord, returns result.
- Proto: `proto/discord/v1/discord.proto`. Generated Go under `gen/go/discord/v1`, TypeScript under `gen/ts/discord/v1`.
- Next.js uses **server actions** (`actions/discord-hourly-actions.ts`) that call the discord gRPC client (`lib/grpc/channel.ts`). Env: `GRPC_DISCORD_ENDPOINT` (default `localhost:50052`).
- **Implemented**: See `apps/grpc/discord_service/`, `apps/ui/web/app/discord/hourly/`, `apps/ui/web/actions/discord-hourly-actions.ts`, `apps/ui/web/lib/hooks/useHourlyDiscord.ts`.

---

## 4. Implementation Phases

### Phase 1: Backend API (choose A or B)

- [ ] Implement config read (hourly channel list + metrics).
- [ ] Implement “copy daily → hourly” and “update single channel webhook”.
- [ ] Implement “resolve economy for ticker” (and which hourly channel, without exposing webhook).
- [ ] Implement “send test message” (server-side only; input = channel or opaque id, never raw webhook in client).

### Phase 2: Next.js data layer

- [ ] Add server actions (e.g. `apps/ui/web/actions/discord-hourly-actions.ts`):
  - `getHourlyDiscordConfig()`, `copyDailyToHourly()`, `updateHourlyChannelWebhook(channelName, webhookUrl)`, `resolveHourlyChannelForTicker(ticker)`, `sendHourlyTestMessage(channelNameOrId)`.
- [ ] Optionally add a small hook (e.g. `useHourlyDiscordConfig`) that uses React Query and the above actions for loading and mutations (with invalidation after copy/update).

### Phase 3: Next.js UI

- [ ] Add route `app/discord/hourly/page.tsx` (or under `app/settings/discord/` if you prefer).
- [ ] Implement metrics cards, Copy button, two tabs (Economy / Special), expandable channel list with webhook input and Save.
- [ ] Implement test section: ticker input, resolve, display result, “Send test message” calling server action only.
- [ ] Update sidebar: “Hourly Discord Management” → `href="/discord/hourly"`.

### Phase 4: Polish and security

- [ ] Ensure webhook URLs are never in client payloads (only “channel name” or server-owned id for test message).
- [ ] Add loading and error states and toasts for success/failure.
- [ ] Optional: add auth (e.g. Clerk) so only allowed users can change Discord config (reuse same pattern as alert actions if applicable).

---

## 5. File Checklist

| Deliverable | Path / location |
|-------------|------------------|
| Migration plan (this doc) | `docs/HOURLY_DISCORD_MANAGEMENT_NEXTJS_PLAN.md` |
| Proto | `proto/discord/v1/discord.proto` |
| Backend (Option B) | `apps/grpc/discord_service/` (Go: server.go, handler.go, config.go, main.go) |
| DB queries | `database/sql/queries/document_queries.sql`, `stock_metadata_queries.sql` |
| Server actions | `apps/ui/web/actions/discord-hourly-actions.ts` |
| Hook | `apps/ui/web/lib/hooks/useHourlyDiscord.ts` |
| Page | `apps/ui/web/app/discord/hourly/page.tsx` |
| Components | `apps/ui/web/app/discord/hourly/_components/` (HourlyDiscordMetrics, HourlyChannelForm, HourlyTestSection) |
| gRPC channel | `apps/ui/web/lib/grpc/channel.ts` (discordClient, `GRPC_DISCORD_ENDPOINT`) |
| Sidebar | `apps/ui/web/components/app-sidebar.tsx` (Hourly Discord Management → `/discord/hourly`) |

---

## 6. Security Reminders

- **Webhook URLs**: Treat as secrets. Do not return them in API responses to the client; only use server-side for “send test message” and for actual alert routing.
- **Config updates**: Restrict to authenticated (and optionally authorized) users; align with how alert mutations are protected (e.g. Clerk in Next.js).
- **Test message**: Always send from backend (server action or API route); never pass webhook URL to the browser.

---

*This plan can be reused as a template for **Daily** and **Weekly** Discord Management pages by swapping “hourly” for “daily”/“weekly” and the corresponding config keys (`channel_mappings_daily`, `channel_mappings_weekly`).*

# CLAUDE.md — Web UI (apps/ui/web)

AI assistant instructions for the **Kovich Stock Alerts** Next.js web application. Read this entire file before writing any code. It contains everything needed to add new pages, features, alert conditions, gRPC integrations, and navigation entries.

---

## 1. App Overview

This is a **Next.js 16 App Router** application (React 19, TypeScript 5) that provides a management UI for a stock alert system. It communicates exclusively with a **Go gRPC backend** over the address set by the `GRPC_ENDPOINT` environment variable (default `127.0.0.1:8081`).

The app is part of a monorepo. The repo root contains:
- `proto/` — Protobuf definitions for all four gRPC services
- `gen/ts/` — TypeScript types generated from the proto files (do not edit)
- `apps/ui/web/` — this Next.js app

---

## 2. Tech Stack

| Concern | Library |
|---|---|
| Framework | Next.js 16, App Router, standalone output, Turbopack |
| Language | TypeScript 5, React 19 |
| Styling | Tailwind CSS v4, shadcn/ui (`components/ui/`) |
| UI state | Jotai v2 (atoms in `lib/store/`) |
| Server data | TanStack Query v5 + `jotai-tanstack-query` |
| gRPC | `nice-grpc` + `@grpc/grpc-js` (server-side only) |
| Charts | Recharts, lightweight-charts |
| Tables | TanStack Table v8 |
| Toasts | Sonner (`toast.success()`, `toast.error()`) |
| Icons | lucide-react |
| Theming | next-themes (dark/light/system) |
| Drag & drop | @dnd-kit |
| Modals | Vaul (drawer), radix-ui dialogs via shadcn |

---

## 3. Project Structure

```
apps/ui/web/
├── app/                            # Next.js App Router pages
│   ├── layout.tsx                  # Root layout: Providers, AppSidebar, SiteHeader, Toaster
│   ├── page.tsx                    # / → Dashboard
│   ├── globals.css
│   ├── alerts/
│   │   ├── page.tsx                # /alerts → paginated alert list
│   │   ├── _components/            # AlertsTable, AlertsSearchBar, AlertsPagination, etc.
│   │   ├── add/
│   │   │   ├── page.tsx            # /alerts/add → create alert form
│   │   │   └── _components/        # AddAlertForm, ConditionBuilder, ConditionRow, types.ts, constants.ts
│   │   ├── audit/
│   │   │   ├── page.tsx            # /alerts/audit → audit dashboard
│   │   │   └── _components/        # AuditContainer, tabs: Overview, EvaluationLog, FailedData
│   │   ├── history/
│   │   │   └── page.tsx            # /alerts/history → trigger history by ticker
│   │   └── delete/
│   │       ├── page.tsx            # /alerts/delete → bulk delete
│   │       └── _components/        # DeleteAlertsContainer, DeleteAlertsTable, etc.
│   ├── portfolios/
│   │   ├── page.tsx                # /portfolios
│   │   └── _components/            # PortfolioList, PortfolioHoldings, CreatePortfolioDialog, etc.
│   ├── discord/
│   │   ├── hourly/page.tsx         # /discord/hourly
│   │   ├── daily/page.tsx          # /discord/daily
│   │   └── weekly/page.tsx         # /discord/weekly
│   ├── scanner/
│   │   ├── page.tsx                # /scanner → RunScan RPC
│   │   └── _components/            # ScannerConditionSection, ScannerFilters, ScannerResults
│   ├── price-database/
│   │   ├── page.tsx                # /price-database
│   │   └── _components/            # PriceDatabaseFilters, PriceDataTable, PriceChartsSection, etc.
│   ├── database/stock/
│   │   ├── page.tsx                # /database/stock → full stock metadata
│   │   └── _components/            # StockDatabaseTable, StockDatabaseFilters, etc.
│   └── scheduler/
│       ├── page.tsx                # /scheduler → scheduler status & control
│       └── _components/            # SchedulerStatusCard, ExchangeScheduleTable, UpcomingChecks
│
├── actions/                        # Next.js Server Actions ("use server") — THE ONLY place gRPC is called
│   ├── alert-actions.ts            # AlertService: CRUD, search, bulk ops
│   ├── audit-actions.ts            # AlertService: audit logs, dashboard stats, performance metrics
│   ├── dashboard-actions.ts        # Aggregates getDashboardStats + getTopTriggeredAlerts
│   ├── scanner-actions.ts          # PriceService: RunScan
│   ├── price-database-actions.ts   # PriceService: LoadPriceData, GetDatabaseStats, stale scans, UpdatePrices
│   ├── stock-database-actions.ts   # PriceService: GetFullStockMetadata, GetStockMetadataMap
│   ├── scheduler-actions.ts        # SchedulerService: status, start/stop, run job, list tasks
│   ├── portfolio-actions.ts        # AlertService: portfolio CRUD, add/remove stocks
│   ├── alert-history-actions.ts    # AlertService: GetTriggerHistoryByTicker, SearchStocks
│   └── discord-hourly-actions.ts   # DiscordConfigService: hourly/daily/weekly config + webhooks
│
├── lib/
│   ├── grpc/
│   │   └── channel.ts              # All gRPC clients — single source of truth
│   ├── providers.tsx               # QueryClient + ThemeProvider + TooltipProvider
│   ├── store/                      # Jotai atoms (one file per feature)
│   │   ├── alerts.ts
│   │   ├── add-alert.ts
│   │   ├── audit.ts
│   │   ├── delete-alerts.ts
│   │   ├── scanner.ts
│   │   ├── price-database.ts
│   │   └── stock-database.ts
│   ├── hooks/                      # TanStack Query hooks (one file per feature)
│   │   ├── useAlerts.ts
│   │   ├── useAudit.ts
│   │   ├── useDashboard.ts
│   │   ├── useScheduler.ts
│   │   ├── usePriceDatabase.ts
│   │   ├── useStockDatabase.ts
│   │   ├── usePortfolios.ts
│   │   ├── useAlertHistory.ts
│   │   └── useHourlyDiscord.ts
│   ├── price-database-enums.ts
│   └── utils.ts                    # cn() helper (clsx + tailwind-merge)
│
├── components/
│   ├── app-sidebar.tsx             # Navigation — add new pages here
│   ├── site-header.tsx
│   ├── dashboard-section-cards.tsx
│   ├── dashboard-active-alerts.tsx
│   ├── alert-activity-chart.tsx
│   └── ui/                         # shadcn/ui primitives (do not create duplicates)
│       ├── button.tsx, card.tsx, input.tsx, select.tsx, checkbox.tsx
│       ├── dialog.tsx, alert-dialog.tsx, sheet.tsx, drawer.tsx
│       ├── table.tsx, tabs.tsx, badge.tsx, tooltip.tsx
│       ├── combobox.tsx            # Searchable dropdown (custom)
│       ├── field.tsx               # Field, FieldSet, FieldContent, FieldLegend form helpers (custom)
│       └── ... (skeleton, separator, label, textarea, etc.)
│
└── next.config.ts                  # Turbopack root = monorepo root; @gen alias; serverExternalPackages
```

**Rule**: every page route gets a `_components/` subfolder for its own components. Shared components go in `components/`.

---

## 4. gRPC Integration

This is the most important section. Understand it fully before touching any data-fetching code.

### How gRPC works in this app

```
Browser (client component)
  ↓  calls
Next.js Server Action  ["use server"]  (actions/*.ts)
  ↓  calls
gRPC client  (lib/grpc/channel.ts)
  ↓  calls
Go backend  at GRPC_ENDPOINT (default 127.0.0.1:8081)
```

- **`lib/grpc/channel.ts`** is a **server-only module**. It creates one channel and four typed clients at module load time.
- `nice-grpc` and `@grpc/grpc-js` are in `serverExternalPackages` in `next.config.ts` — they are Node.js-native and **must never be imported in `"use client"` code**.
- All gRPC calls happen inside server actions (`"use server"` files in `actions/`).
- Server actions convert Protobuf types to plain serializable objects before returning to the client (e.g. `Timestamp → .toISOString()`).

### lib/grpc/channel.ts (current clients)

```typescript
import { createChannel, createClientFactory } from "nice-grpc";
import { AlertServiceDefinition }     from "../../../../gen/ts/alert/v1/alert";
import { DiscordConfigServiceDefinition } from "../../../../gen/ts/discord/v1/discord";
import { PriceServiceDefinition }     from "../../../../gen/ts/price/v1/price";
import { SchedulerServiceDefinition } from "../../../../gen/ts/scheduler/v1/scheduler";

const channel = createChannel(process.env.GRPC_ENDPOINT || "127.0.0.1:8081");
const clientFactory = createClientFactory();

export const alertClient    = clientFactory.create(AlertServiceDefinition, channel);
export const discordClient  = clientFactory.create(DiscordConfigServiceDefinition, channel);
export const priceClient    = clientFactory.create(PriceServiceDefinition, channel);
export const schedulerClient = clientFactory.create(SchedulerServiceDefinition, channel);
```

To add a new service, import its `*Definition` from `gen/ts/` and add a new `clientFactory.create(...)` export.

### Generated types location

All proto-generated TypeScript is at:
```
gen/ts/
├── alert/v1/alert.ts
├── discord/v1/discord.ts
├── price/v1/price.ts
└── scheduler/v1/scheduler.ts
```

Import from relative paths (e.g. `../../../../gen/ts/alert/v1/alert`) or use the `@gen` Turbopack alias.

---

## 5. The Four gRPC Services — Complete RPC Reference

### 5.1 AlertService (`alertClient`)

Used in: `alert-actions.ts`, `audit-actions.ts`, `dashboard-actions.ts`, `portfolio-actions.ts`, `alert-history-actions.ts`

**Alert CRUD**
| RPC | Request | Response | Notes |
|---|---|---|---|
| `ListAlerts` | `{ pageSize, page, search, exchanges[], timeframes[], countries[], triggeredFilter, conditionSearch }` | `{ alerts[], hasNextPage, totalCount }` | Server-side filtered pagination |
| `SearchAlertsStream` | same filters, no pagination | stream of `{ alerts[], done }` | Server-streaming; used for "select all filtered" |
| `GetAlert` | `{ alertId }` | `{ alert }` | Single alert fetch |
| `CreateAlert` | full alert fields (see §8) | `{ alert }` | |
| `UpdateAlert` | `alertId` + updatable fields | `{ alert }` | |
| `DeleteAlert` | `{ alertId }` | `{}` | |
| `BulkDeleteAlerts` | `{ alertIds[] }` | `{ deletedCount }` | |
| `BulkUpdateLastTriggered` | `{ triggers[{ alertId, lastTriggered }] }` | `{ updatedCount }` | |
| `GetTopTriggeredAlerts` | `{ days, limit }` | `{ alerts[{ alert, triggerCount }], totalCount }` | Dashboard top list |
| `EvaluateExchange` | `{ exchange, timeframe }` | `{ success, message, alertsTotal, alertsTriggered, pricesUpdated, durationSeconds }` | Synchronous evaluation |

**Audit / Monitoring**
| RPC | Notes |
|---|---|
| `GetDashboardStats` | KPI counts: active alerts, triggered today, watched symbols, triggers last 7d (with timeframe breakdown) |
| `GetTriggerCountByDay` | `{ days }` → per-day trigger counts for activity chart |
| `GetAuditSummary` | `{ days, limit }` → per-alert summary rows |
| `GetPerformanceMetrics` | `{ days }` → success rate, cache hit rate, avg execution time |
| `GetAlertHistory` | `{ alertId, limit }` → individual audit rows for one alert |
| `GetFailedPriceData` | `{ days }` → failed price pulls with asset/exchange breakdown |
| `GetAuditLog` | paginated, filterable, sortable access to raw audit rows |
| `ClearAuditData` | deletes all audit rows; returns `{ deletedCount }` |

**History / Search**
| RPC | Notes |
|---|---|
| `GetTriggerHistoryByTicker` | `{ ticker, includeAllEvaluations, limit, daysBack }` → audit rows |
| `SearchStocks` | `{ query, limit }` → `{ results[{ ticker, name, exchange, type, rbicsEconomy }] }` |

**Portfolios**
| RPC | Notes |
|---|---|
| `ListPortfolios` | all portfolios |
| `GetPortfolio` | single portfolio by id |
| `CreatePortfolio` | `{ name, discordWebhook }` |
| `UpdatePortfolio` | `{ portfolioId, name, discordWebhook, enabled }` |
| `DeletePortfolio` | `{ portfolioId }` |
| `AddStocksToPortfolio` | `{ portfolioId, tickers[] }` |
| `RemoveStocksFromPortfolio` | `{ portfolioId, tickers[] }` |

### 5.2 PriceService (`priceClient`)

Used in: `price-database-actions.ts`, `scanner-actions.ts`, `stock-database-actions.ts`

| RPC | Notes |
|---|---|
| `GetStockMetadataMap` | lightweight list for filters/dropdowns |
| `GetFullStockMetadata` | full rows with RBICS, ETF fields, market data |
| `GetDatabaseStats` | record counts and date ranges per timeframe |
| `LoadPriceData` | `{ timeframe, tickers[], startDate, endDate, maxRows, dayFilter }` → OHLCV bars |
| `ScanStaleDaily` | tickers behind on daily data |
| `ScanStaleWeekly` | tickers behind on weekly data |
| `ScanStaleHourly` | tickers behind on hourly data |
| `GetHourlyDataQuality` | stale/gap metrics for hourly data |
| `RunScan` | `{ timeframe, conditions[], combinationLogic, tickers[], symbolFilter, maxTickers, lookbackDays }` → `{ matches[] }` |
| `UpdatePrices` | **server-streaming** progress events; `{ exchanges[], tickers[], timeframe }` |

**Timeframe enum values**: `TIMEFRAME_UNSPECIFIED=0`, `TIMEFRAME_HOURLY=1`, `TIMEFRAME_DAILY=2`, `TIMEFRAME_WEEKLY=3`

**SymbolFilter** (for `RunScan`):
```typescript
{
  assetTypes: string[];      // "Stock", "ETF"
  countries: string[];
  exchanges: string[];
  rbicsEconomy: string[];
  rbicsSector: string[];
  rbicsSubsector: string[];
  rbicsIndustryGroup: string[];
  rbicsIndustry: string[];
  rbicsSubindustry: string[];
}
```

### 5.3 SchedulerService (`schedulerClient`)

Used in: `scheduler-actions.ts`

| RPC | Notes |
|---|---|
| `GetSchedulerStatus` | heartbeat, current job, last run, last error, queue breakdown, paused state |
| `GetExchangeSchedule` | `{ timeframe }` → schedule rows with run times and countdown |
| `StartScheduler` | unpauses the Asynq queue |
| `StopScheduler` | pauses the Asynq queue |
| `RunExchangeJob` | `{ exchange, timeframe }` → enqueues an immediate job |
| `ListQueueTasks` | `{ queue }` → tasks with state/type/exchange/timeframe/nextProcessAt |

### 5.4 DiscordConfigService (`discordClient`)

Used in: `discord-hourly-actions.ts`

All three timeframes (Hourly, Daily, Weekly) share the same response types (`GetHourlyDiscordConfigResponse`, `UpdateHourlyChannelWebhookResponse`, etc.).

| RPC | Notes |
|---|---|
| `GetHourlyDiscordConfig` / `GetDailyDiscordConfig` / `GetWeeklyDiscordConfig` | config + channel list with configured status |
| `UpdateHourlyChannelWebhook` / `UpdateDailyChannelWebhook` / `UpdateWeeklyChannelWebhook` | `{ channelName, webhookUrl }` |
| `CopyDailyToHourly` / `CopyBaseToDaily` / `CopyBaseToWeekly` | copies webhook URLs between timeframes |
| `SendHourlyTestMessage` / `SendDailyTestMessage` / `SendWeeklyTestMessage` | `{ channelName }` → sends test Discord message |
| `ResolveHourlyChannelForTicker` / `ResolveDailyChannelForTicker` / `ResolveWeeklyChannelForTicker` | `{ ticker }` → economy + channel name + configured status |

Channel names are economy keys like `"Technology"`, `"Financials"`, `"ETFs"`, etc.

---

## 6. Data Flow Pattern

### Full request lifecycle

```
1. "use client" component renders
2. Calls custom hook: useFeature() from lib/hooks/useFeature.ts
3. Hook calls useQuery / useMutation from TanStack Query
4. queryFn is a Next.js Server Action (async function from actions/*.ts)
5. Server action imports gRPC client from lib/grpc/channel.ts
6. Server action calls the gRPC RPC method
7. Server action maps Protobuf types → plain serializable objects
8. TanStack Query caches the result, component re-renders
```

### Server action pattern

```typescript
// actions/feature-actions.ts
"use server";

import { featureClient } from "@/lib/grpc/channel";
import type { SomeProtoType } from "../../../../gen/ts/service/v1/service";

export type FeatureData = {
  id: string;
  name: string;
  createdAt: string | null;  // Timestamp → ISO string
};

function toFeatureData(proto: SomeProtoType): FeatureData {
  return {
    id: proto.id,
    name: proto.name,
    createdAt: proto.createdAt?.toISOString() ?? null,
  };
}

export async function getFeature(id: string): Promise<FeatureData | null> {
  try {
    const response = await featureClient.getFeature({ id });
    return response.item ? toFeatureData(response.item) : null;
  } catch (err) {
    if (err instanceof ClientError && err.code === Status.NOT_FOUND) return null;
    throw err;
  }
}
```

### Store + hook pattern

```typescript
// lib/store/feature.ts
"use client";
import { atom } from "jotai";
import { atomWithQuery } from "jotai-tanstack-query";
import { getFeatureList } from "@/actions/feature-actions";

export const FEATURE_KEY = ["feature"] as const;
export const featurePageAtom = atom(1);
export const featureSearchAtom = atom("");

export const featureQueryAtom = atomWithQuery((get) => ({
  queryKey: [...FEATURE_KEY, "list", get(featureSearchAtom), get(featurePageAtom)],
  queryFn: () => getFeatureList(get(featureSearchAtom), get(featurePageAtom)),
}));

// lib/hooks/useFeature.ts
"use client";
import { useAtom, useAtomValue, useSetAtom } from "jotai";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { featureQueryAtom, FEATURE_KEY } from "@/lib/store/feature";
import { createFeature, deleteFeature } from "@/actions/feature-actions";

export function useFeatureList() {
  const [result] = useAtom(featureQueryAtom);
  return { data: result.data, isLoading: result.isPending, error: result.error };
}

export function useCreateFeature() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: createFeature,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: FEATURE_KEY }),
  });
}
```

---

## 7. State Management

### Jotai atoms (`lib/store/`)

Used for **UI state only**: form field values, filter selections, pagination page, UI toggles.

- Every atom file starts with `"use client"`.
- Atoms are global — they do **not** reset automatically when a component unmounts. Reset explicitly in a `useEffect` cleanup or when navigating away if needed.
- Use `useAtomValue(atom)` for read-only, `useSetAtom(atom)` for write-only, `useAtom(atom)` for both.
- Use `useStore()` to read atom values imperatively inside form submit handlers (avoids stale closure).

### TanStack Query

Used for all **server data**: fetching, caching, mutations.

- Default `staleTime`: 30 seconds (set in `lib/providers.tsx`)
- Default `refetchOnWindowFocus`: false
- Query keys: `["feature", ...params]` arrays — be specific enough that different param combinations cache separately
- After mutations, invalidate with `queryClient.invalidateQueries({ queryKey: FEATURE_KEY })`

### `atomWithQuery` (jotai-tanstack-query)

Use when your query params live in Jotai atoms and you want them to be reactive. The query refetches automatically when any atom it reads changes.

```typescript
export const myQueryAtom = atomWithQuery((get) => ({
  queryKey: ["items", get(searchAtom), get(pageAtom)],
  queryFn: () => fetchItems(get(searchAtom), get(pageAtom)),
}));
```

---

## 8. Alert Model — Full Field Reference

```typescript
type AlertData = {
  alertId: string;
  name: string;
  stockName: string;            // display name of the asset
  ticker: string;               // primary ticker symbol
  ticker1: string;              // ratio numerator ticker (when isRatio = true)
  ticker2: string;              // ratio denominator ticker (when isRatio = true)
  combinationLogic: string;     // "AND" | "OR"
  action: string;               // "Buy" | "Sell"
  timeframe: string;            // "daily" | "weekly" | "hourly" | "1d" | "1wk" | etc.
  exchange: string;             // "NASDAQ" | "NYSE" | "LSE" | "" etc.
  country: string;              // "US" | "GB" | "" etc.
  ratio: string;                // "Yes" | "No"
  isRatio: boolean;
  adjustmentMethod: string;
  lastTriggered: string | null; // ISO timestamp
  createdAt: string | null;
  updatedAt: string | null;
  conditions?: unknown;         // google.protobuf.Struct — backend conditions JSON
};
```

**Conditions Struct format** (sent to backend):
```json
{
  "condition_1": {
    "conditions": ["rsi(14)[-1] < 30", "Close[-1] > sma(50)[-1]"],
    "combination_logic": "AND"
  }
}
```

Built by `buildConditionsStruct(entries, combinationLogic)` from `app/alerts/add/_components/types.ts`.

**Advanced alert params** (passed as `google.protobuf.Struct`):
- `dtpParams` — DTP (Dynamic Threshold Percentage) parameters
- `multiTimeframeParams` — multi-timeframe comparison settings
- `mixedTimeframeParams` — mixed timeframe condition settings
- `rawPayload` — arbitrary extra data

---

## 9. Alert Conditions — Complete Reference

All condition logic lives in `app/alerts/add/_components/types.ts`. This file is the **single source of truth** for how the UI represents and serializes conditions to the backend expression format.

### Condition expression syntax (backend format)

The backend evaluates Python-style expressions against OHLCV price DataFrames.

```
# Simple keyword syntax
price_above: 150
price_below: 100

# Function call with index ([-1] = latest bar, [-2] = previous bar)
rsi(14)[-1] < 30
sma(period=20)[-1] < Close[-1]

# Crossover (two conditions joined with "and")
(sma(20)[-1] < Close[-1]) and (sma(20)[-2] >= Close[-2])

# Multi-timeframe suffix (_weekly, _daily, _hourly)
Close_weekly[-1] > sma(50)_weekly[-1]

# Z-score wrap
zscore(rsi(14), lookback=20)[-1] > 2

# Volume
volume_above_average: 1.5x
volume_spike: 2x

# Custom free-form expression
Close[-1] > High[-2] and Volume[-1] > Volume[-2] * 1.5
```

### All condition categories

| Category | Description | Key condition types |
|---|---|---|
| `price` | Raw price comparisons | `price_above`, `price_below`, `price_equals` |
| `moving_average` | MA comparisons & crossovers | `price_above_ma`, `price_below_ma`, `price_cross_above_ma`, `price_cross_below_ma`, `ma_crossover` |
| `rsi` | RSI levels | `rsi_oversold`, `rsi_overbought`, `rsi_level` |
| `macd` | MACD signals | `macd_bullish_crossover`, `macd_bearish_crossover`, `macd_histogram_positive` |
| `bollinger` | Bollinger band breaks | `price_above_upper_band`, `price_below_lower_band` |
| `volume` | Volume spikes | `volume_above_average`, `volume_spike` |
| `ma_slope_curve` | HMA/EMA slope & curvature signals | `slope_positive`, `slope_negative`, `slope_turn_up`, `slope_turn_dn`, `curve_positive`, `curve_negative`, `bend_up`, `bend_dn`, `early_bend_up`, `early_bend_dn` |
| `donchian` | Donchian channel position/breakout | `donchian_price_vs_upper`, `donchian_breakout_upper`, `donchian_width_expanding`, etc. (17 types) |
| `pivot_sr` | Pivot support/resistance | `pivot_sr_near_support`, `pivot_sr_crossover_bullish`, `pivot_sr_broke_strong_support`, etc. (9 types) |
| `ichimoku` | Ichimoku Cloud signals | Price vs cloud, TK cross, cloud color, lagging span (27 types) |
| `trend_magic` | Trend Magic indicator | `tm_bullish`, `tm_bearish`, `tm_buy_signal`, `tm_sell_signal`, `tm_any_cross` |
| `supertrend` | SuperTrend indicator | `st_uptrend`, `st_downtrend`, `st_changed_uptrend`, `st_changed_downtrend` |
| `sar` | Parabolic SAR | `sar_price_above`, `sar_cross_above`, `sar_cross_below` |
| `obv_macd` | OBV MACD momentum | `obv_macd_positive`, `obv_macd_signal_bullish`, `obv_macd_signal_bearish` |
| `harsi` | HARSI oscillator | `harsi_bullish`, `harsi_bearish`, `harsi_flip_buy`, `harsi_flip_sell` |
| `ma_zscore` | MA Z-score spread | `ma_zscore_compare`, `ma_zscore_value` |
| `ewo` | Elliott Wave Oscillator | `ewo_above_zero`, `ewo_below_zero`, `ewo_cross_above_zero`, `ewo_cross_below_zero`, `ewo_compare` |
| `roc` | Rate of Change | `roc_above_zero`, `roc_below_zero`, `roc_cross_above_zero`, `roc_compare` |
| `willr` | Williams %R | `willr_oversold`, `willr_overbought` |
| `cci` | CCI | `cci_compare`, `cci_value` |
| `atr` | ATR volatility | `atr_compare`, `atr_value` |
| `kalman_roc_stoch` | Kalman ROC Stochastic | `krs_uptrend`, `krs_downtrend`, `krs_cross_bullish`, `krs_above_60`, `krs_below_10` |
| `indicator` | Any backend-registered indicator by name | Free-form: `indicatorName(params)[-1] > value` |
| `custom` | Free-form Python expression | Any valid expression string |

### MA types supported

`SMA`, `EMA`, `WMA`, `RMA`, `HMA`, `DEMA`, `TEMA`, `VWMA`, `LSMA`, `FRAMA`, `KAMA`

**FRAMA** requires `framaFc` and `framaSc` params.  
**KAMA** requires `kamaFastEnd` and `kamaSlowEnd` params.

### MA input sources (Task 7 feature)

When `maInputSource` is set, the MA is computed over an indicator instead of price:
`Close` (default), `Open`, `High`, `Low`, `EWO`, `RSI`, `MACD_Line`, `MACD_Signal`, `MACD_Histogram`

### Z-score wrapping

Any condition can be wrapped in `zscore()` by setting `useZScore: true` and `zScoreLookback: number` in `ConditionParams`. The serializer in `conditionEntryToExpressionRaw` handles this automatically via `maybeWrapZScore()`.

---

## 10. How to Add a New Alert Condition (Step by Step)

All changes are in `app/alerts/add/_components/`.

### Step 1 — `types.ts`: add the type

```typescript
// 1a. Add to ConditionCategory union
export type ConditionCategory =
  | ... existing ...
  | "my_new_indicator";

// 1b. Add condition type union
export type MyNewIndicatorConditionType =
  | "my_new_bullish"
  | "my_new_bearish";

// 1c. Add params to ConditionParams interface
export type ConditionParams = {
  ... existing ...
  myNewPeriod?: number;
  myNewThreshold?: number;
};
```

### Step 2 — `types.ts`: add serialization

Inside `conditionEntryToExpressionRaw()`:

```typescript
case "my_new_indicator": {
  const period = params.myNewPeriod ?? 14;
  const threshold = params.myNewThreshold ?? 0;
  const fn = `my_new_indicator(${period})`;
  switch (type) {
    case "my_new_bullish":
      return `${fn}[-1] > ${threshold}`;
    case "my_new_bearish":
      return `${fn}[-1] < ${threshold}`;
    default:
      return "";
  }
}
```

### Step 3 — `types.ts`: add label

Inside `conditionEntryLabel()`:

```typescript
const presetCategories: ConditionCategory[] = [
  ... existing ...,
  "my_new_indicator",  // add here to use default expression display
];
```

### Step 4 — `ConditionBuilder.tsx`: add to category dropdown

```typescript
const CATEGORY_OPTIONS = [
  ... existing ...,
  { value: "my_new_indicator" as ConditionCategory, label: "My New Indicator" },
];
```

### Step 5 — `ConditionBuilder.tsx`: add UI fields

Inside the category render switch, add a block for `"my_new_indicator"` showing `Input` fields for period and threshold, writing to the `params` state.

### Step 6 — `ConditionBuilder.tsx`: add default params

In `getDefaultParams(category)`:

```typescript
case "my_new_indicator":
  return { myNewPeriod: 14, myNewThreshold: 0 };
```

---

## 11. How to Add a New Page (Step by Step)

### Step 1 — Create the page file

```typescript
// app/my-feature/page.tsx
import { MyFeatureContainer } from "./_components/MyFeatureContainer";

export default function MyFeaturePage() {
  return <MyFeatureContainer />;
}
```

### Step 2 — Create the container component

```typescript
// app/my-feature/_components/MyFeatureContainer.tsx
"use client";
import { useMyFeature } from "@/lib/hooks/useMyFeature";

export function MyFeatureContainer() {
  const { data, isLoading } = useMyFeature();
  if (isLoading) return <div>Loading...</div>;
  return <div>{/* render data */}</div>;
}
```

### Step 3 — Create the server action

```typescript
// actions/my-feature-actions.ts
"use server";
import { alertClient } from "@/lib/grpc/channel"; // or priceClient, schedulerClient, discordClient

export type MyFeatureData = { ... };

export async function getMyFeature(): Promise<MyFeatureData> {
  const response = await alertClient.someRpc({});
  return { /* map proto → plain object */ };
}
```

### Step 4 — Create the store

```typescript
// lib/store/my-feature.ts
"use client";
import { atom } from "jotai";
import { atomWithQuery } from "jotai-tanstack-query";
import { getMyFeature } from "@/actions/my-feature-actions";

export const MY_FEATURE_KEY = ["my-feature"] as const;
export const myFeatureFilterAtom = atom("");

export const myFeatureQueryAtom = atomWithQuery((get) => ({
  queryKey: [...MY_FEATURE_KEY, get(myFeatureFilterAtom)],
  queryFn: () => getMyFeature(),
}));
```

### Step 5 — Create the hook

```typescript
// lib/hooks/useMyFeature.ts
"use client";
import { useAtom } from "jotai";
import { myFeatureQueryAtom } from "@/lib/store/my-feature";

export function useMyFeature() {
  const [result] = useAtom(myFeatureQueryAtom);
  return { data: result.data, isLoading: result.isPending, error: result.error };
}
```

### Step 6 — Add to sidebar navigation

In `components/app-sidebar.tsx`, add to the appropriate section (`data.alerts`, `data.discord`, or `data.database`):

```typescript
{
  title: "My Feature",
  url: "/my-feature",
  icon: <MyIcon />,  // from lucide-react
},
```

---

## 12. How to Add a New gRPC Service or RPC

### Using an existing RPC that isn't wired up yet

1. Find the RPC in `gen/ts/[service]/v1/[service].ts`
2. Import it in a server action file and call `[client].[rpcName](request)`
3. Map the response to a plain serializable type

### Adding a new service from an existing proto

1. Check `gen/ts/` for the generated definition export (`*ServiceDefinition`)
2. In `lib/grpc/channel.ts`:
   ```typescript
   import { NewServiceDefinition, type NewServiceClient } from "../../../../gen/ts/new/v1/new";
   export const newClient: NewServiceClient = clientFactory.create(NewServiceDefinition, channel);
   ```
3. Create `actions/new-service-actions.ts`

### Adding a brand-new proto service

1. Write the `.proto` file in `proto/[service]/v1/[service].proto` with package `stockalert.[service].v1`
2. Run the project's protobuf codegen to produce `gen/ts/[service]/v1/[service].ts`
3. Add client to `lib/grpc/channel.ts`
4. Create server actions
5. Create store, hook, page, components

### Server-streaming RPCs

Use `for await` in the server action:

```typescript
export async function streamMyData(): Promise<MyItem[]> {
  const results: MyItem[] = [];
  for await (const chunk of client.myStreamingRpc(request)) {
    results.push(...chunk.items.map(toMyItem));
  }
  return results;
}
```

For progress streaming (like `UpdatePrices`), use a different approach with `ReadableStream` or handle chunks client-side via polling/SSE. See `price-database-actions.ts` for the existing pattern.

---

## 13. Component Patterns

### Always use shadcn/ui primitives from `components/ui/`

```typescript
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardDescription, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "@/components/ui/sheet";
import { Table, TableHeader, TableRow, TableHead, TableBody, TableCell } from "@/components/ui/table";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { Separator } from "@/components/ui/separator";
import { Skeleton } from "@/components/ui/skeleton";
```

**Custom components** (not from shadcn, already exist in `components/ui/`):
- `Combobox` — searchable dropdown with keyboard nav
- `Field`, `FieldSet`, `FieldContent`, `FieldLegend` — form layout wrappers that handle label/content alignment

### Toasts

```typescript
import { toast } from "sonner";
toast.success("Alert created successfully.");
toast.error("Failed to create alert.");
toast.loading("Processing...");
```

### Icons

```typescript
import { BellIcon, PlusIcon, Trash2Icon, ChartBarIcon, ClockIcon } from "lucide-react";
// Use size via className: <BellIcon className="size-4" />
```

### Dark mode

Never use hardcoded colors (`text-black`, `bg-white`, `#fff`). Always use Tailwind semantic tokens:
- `bg-background`, `text-foreground`
- `bg-muted`, `text-muted-foreground`
- `bg-card`, `border-border`
- `text-destructive`, `bg-primary`, `text-primary-foreground`

### Loading states

Use `Skeleton` components that mirror the shape of the content, or `isPending` from TanStack Query:

```typescript
if (isLoading) return <Skeleton className="h-32 w-full" />;
```

### Error states

Show errors with a styled card or inline message:

```typescript
if (error) return (
  <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4">
    <p className="text-sm text-destructive">{error.message}</p>
  </div>
);
```

---

## 14. Sidebar Navigation Reference

File: `components/app-sidebar.tsx`

The sidebar has three nav sections. To add a page, push an item to the appropriate array:

```typescript
const data = {
  alerts: [
    // { title, url, icon: <LucideIcon /> }
    // Current: Alerts, Add Alert, Alert Audit, Alert History, Delete Alerts, Portfolios
  ],
  discord: [
    // Current: Hourly Discord Management, Daily Discord Management, Weekly Discord Management
  ],
  database: [
    // Current: Scanner, Price Database, Stock, Scheduler Status
  ],
};
```

Icon must be a `<LucideIcon />` JSX element (not a component reference):
```typescript
icon: <BellIcon />,   // correct
icon: BellIcon,       // wrong
```

---

## 15. Environment & Dev Commands

```bash
# Start dev server (from apps/ui/web/)
pnpm dev      # or npm run dev

# Build
pnpm build

# Lint
pnpm lint
```

**Environment variables:**

| Variable | Default | Description |
|---|---|---|
| `GRPC_ENDPOINT` | `127.0.0.1:8081` | Go gRPC backend address |

The `GRPC_ENDPOINT` variable is read at **server runtime** in `lib/grpc/channel.ts`. It is not exposed to the browser. Set it in `.env.local` for local development.

---

## 16. Key Conventions (Rules for LLMs)

1. **`"use server"` at the top of every file in `actions/`** — no exceptions.
2. **`"use client"` at the top of every file in `lib/store/` and `lib/hooks/`**.
3. **Never import gRPC clients (`lib/grpc/channel.ts`) in client components or hooks.** They must only be imported in `actions/` files.
4. **Protobuf `Timestamp` fields** must be converted to ISO strings (`.toISOString()`) in server actions before returning to the client.
5. **Protobuf `Struct` fields** can be returned as `unknown` — they are arbitrary JSON objects.
6. **Protobuf `int64` / `uint64`** fields come back as `number` in the generated TS — use `Number(val)` to be safe.
7. **Query cache invalidation**: after any mutation, call `queryClient.invalidateQueries({ queryKey: FEATURE_KEY })`.
8. **Jotai atoms do not reset automatically** on navigation. Reset them in cleanup effects if needed.
9. **Reading atoms imperatively in submit handlers**: use `const store = useStore(); store.get(myAtom)` to avoid stale closures.
10. **All UI colors must use Tailwind semantic tokens** — never hardcode hex colors or `bg-white`/`text-black`.
11. **Page layout**: pages are full-width by default (the sidebar is separate). Use `px-4 lg:px-6` for horizontal padding inside page content.
12. **New conditions only need frontend changes** (unless the backend doesn't support the indicator yet) — the expression string is passed directly to the Go evaluator.
13. **Condition expression strings are Python-like** — use `and`/`or` (lowercase), `==`/`!=`/`>`/`<`/`>=`/`<=`, array indexing `[-1]`, function calls with named params.
14. **`next.config.ts` Turbopack root** is the monorepo root (`../../../`), so paths to `gen/ts/` resolve from there.
15. **The `@/` path alias** maps to `apps/ui/web/` (set by `tsconfig.json`).

---

## 17. Existing Pages Quick Reference

| Route | What it does | Primary gRPC calls |
|---|---|---|
| `/` | Dashboard KPIs + activity chart + top triggered alerts | `GetDashboardStats`, `GetTriggerCountByDay`, `GetTopTriggeredAlerts` |
| `/alerts` | Paginated, filterable alert list | `ListAlerts` |
| `/alerts/add` | Create single/bulk/ratio alerts with condition builder | `CreateAlert`, `SearchStocks`, `GetFullStockMetadata` |
| `/alerts/audit` | Audit overview, evaluation log, failed price data tabs | `GetAuditSummary`, `GetPerformanceMetrics`, `GetAuditLog`, `GetFailedPriceData`, `ClearAuditData` |
| `/alerts/history` | Trigger history lookup by ticker | `GetTriggerHistoryByTicker`, `SearchStocks`, `ListPortfolios` |
| `/alerts/delete` | Bulk-select and delete alerts | `ListAlerts`, `BulkDeleteAlerts`, `SearchAlertsStream` |
| `/portfolios` | Portfolio CRUD + holdings management | All portfolio RPCs |
| `/discord/hourly` | Hourly Discord webhook config | `GetHourlyDiscordConfig`, `UpdateHourlyChannelWebhook`, `SendHourlyTestMessage`, `CopyDailyToHourly` |
| `/discord/daily` | Daily Discord webhook config | Daily equivalents |
| `/discord/weekly` | Weekly Discord webhook config | Weekly equivalents |
| `/scanner` | Stock scanner with condition builder | `RunScan`, `GetFullStockMetadata` |
| `/price-database` | Browse/filter/export OHLCV price data | `LoadPriceData`, `GetDatabaseStats`, `GetStockMetadataMap`, `UpdatePrices` |
| `/database/stock` | Full stock metadata browser with RBICS breakdown | `GetFullStockMetadata` |
| `/scheduler` | Scheduler status, exchange schedule, queue control | `GetSchedulerStatus`, `GetExchangeSchedule`, `StartScheduler`, `StopScheduler`, `RunExchangeJob`, `ListQueueTasks` |

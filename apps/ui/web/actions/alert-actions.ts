"use server";

import { ClientError, Status } from "nice-grpc-common";
import { alertClient } from "@/lib/grpc/channel";
import type {
  Alert as AlertProto,
  CreateAlertRequest,
  UpdateAlertRequest,
} from "../../../../gen/ts/alert/v1/alert";
export type AlertData = {
  alertId: string;
  name: string;
  stockName: string;
  ticker: string;
  ticker1: string;
  ticker2: string;
  combinationLogic: string;
  action: string;
  timeframe: string;
  exchange: string;
  country: string;
  ratio: string;
  isRatio: boolean;
  adjustmentMethod: string;
  lastTriggered: string | null;
  createdAt: string | null;
  updatedAt: string | null;
  /** Optional; used by Alert History for condition filter/display */
  conditions?: unknown;
};

function toAlertData(alert: AlertProto): AlertData {
  return {
    alertId: alert.alertId,
    name: alert.name,
    stockName: alert.stockName,
    ticker: alert.ticker,
    ticker1: alert.ticker1,
    ticker2: alert.ticker2,
    combinationLogic: alert.combinationLogic,
    action: alert.action,
    timeframe: alert.timeframe,
    exchange: alert.exchange,
    country: alert.country,
    ratio: alert.ratio,
    isRatio: alert.isRatio,
    adjustmentMethod: alert.adjustmentMethod,
    lastTriggered: alert.lastTriggered?.toISOString() ?? null,
    createdAt: alert.createdAt?.toISOString() ?? null,
    updatedAt: alert.updatedAt?.toISOString() ?? null,
    conditions: alert.conditions,
  };
}

const DEFAULT_PAGE_SIZE = 20;
const MAX_PAGE_SIZE = 100;

export type ListAlertsResult = {
  alerts: AlertData[];
  totalCount: number;
  hasNextPage: boolean;
};

export type TopTriggeredAlertsResult = ListAlertsResult & {
  /** alertId -> number of times triggered (in the requested window). */
  triggerCountByAlertId: Record<string, number>;
};

export async function listAlertsPaginated(
  page: number = 1,
  pageSize: number = DEFAULT_PAGE_SIZE
): Promise<ListAlertsResult> {
  const size = Math.min(Math.max(1, pageSize), MAX_PAGE_SIZE);
  const pageNum = Math.max(1, page);
  const response = await alertClient.listAlerts({
    pageSize: size,
    page: pageNum,
  });
  return {
    alerts: response.alerts.map(toAlertData),
    totalCount: response.totalCount,
    hasNextPage: response.hasNextPage,
  };
}

/** Top N alerts by count of alert_triggered = true in alert_audits (single query, only existing alerts). */
export async function getTopTriggeredAlerts(
  days: number = 30,
  limit: number = 10
): Promise<TopTriggeredAlertsResult> {
  const response = await alertClient.getTopTriggeredAlerts({
    days: Math.min(90, Math.max(1, days)),
    limit: Math.min(50, Math.max(1, limit)),
  });
  const items = response.alerts ?? [];
  const alerts = items.map((item) => toAlertData(item.alert!)).filter(Boolean);
  const triggerCountByAlertId: Record<string, number> = {};
  for (const item of items) {
    if (item.alert?.alertId != null) {
      triggerCountByAlertId[item.alert.alertId] = Number(item.triggerCount ?? 0);
    }
  }
  const totalCount = response.totalCount ?? 0;
  return {
    alerts,
    totalCount,
    hasNextPage: alerts.length < totalCount,
    triggerCountByAlertId,
  };
}

/** Fetch all alerts (multiple pages); used by Alert History browse tab. */
export async function listAllAlertsForHistory(): Promise<AlertData[]> {
  const all: AlertData[] = [];
  let page = 1;
  while (true) {
    const result = await listAlertsPaginated(page, MAX_PAGE_SIZE);
    all.push(...result.alerts);
    if (!result.hasNextPage || result.alerts.length === 0) break;
    page++;
  }
  return all;
}

export type SearchAlertsFilters = {
  search?: string;
  exchanges?: string[];
  timeframes?: string[];
  countries?: string[];
  triggeredFilter?: string;
  conditionSearch?: string;
};

/** Server-side filtered + paginated alert search. */
export async function searchAlerts(
  filters: SearchAlertsFilters,
  page: number = 1,
  pageSize: number = DEFAULT_PAGE_SIZE
): Promise<ListAlertsResult> {
  const size = Math.min(Math.max(1, pageSize), MAX_PAGE_SIZE);
  const pageNum = Math.max(1, page);

  // Map triggered filter labels to server values
  const triggeredMap: Record<string, string> = {
    "Never": "never",
    "Today": "today",
    "This Week": "this_week",
    "This Month": "this_month",
    "This Year": "this_year",
  };

  const response = await alertClient.listAlerts({
    pageSize: size,
    page: pageNum,
    search: filters.search || "",
    exchanges: filters.exchanges || [],
    timeframes: filters.timeframes || [],
    countries: filters.countries || [],
    triggeredFilter: triggeredMap[filters.triggeredFilter || ""] || filters.triggeredFilter || "",
    conditionSearch: filters.conditionSearch || "",
  });
  return {
    alerts: response.alerts.map(toAlertData),
    totalCount: response.totalCount,
    hasNextPage: response.hasNextPage,
  };
}

/** Server-streaming search: returns all matching alerts in one request (for "Select all filtered"). */
export async function searchAlertsStream(
  filters: SearchAlertsFilters
): Promise<AlertData[]> {
  const triggeredMap: Record<string, string> = {
    "Never": "never",
    "Today": "today",
    "This Week": "this_week",
    "This Month": "this_month",
    "This Year": "this_year",
  };

  const stream = alertClient.searchAlertsStream({
    search: filters.search || "",
    exchanges: filters.exchanges || [],
    timeframes: filters.timeframes || [],
    countries: filters.countries || [],
    triggeredFilter: triggeredMap[filters.triggeredFilter || ""] || filters.triggeredFilter || "",
    conditionSearch: filters.conditionSearch || "",
  });

  const all: AlertData[] = [];
  for await (const chunk of stream) {
    for (const alert of chunk.alerts) {
      all.push(toAlertData(alert));
    }
  }
  return all;
}

export async function getAlert(alertId: string): Promise<AlertData | null> {
  try {
    const response = await alertClient.getAlert({ alertId });
    return response.alert ? toAlertData(response.alert) : null;
  } catch (err) {
    if (err instanceof ClientError && err.code === Status.NOT_FOUND) {
      return null;
    }
    throw err;
  }
}

export type CreateAlertInput = Partial<CreateAlertRequest> & {
  name: string;
  stockName: string;
  ticker: string;
  ticker1: string;
  ticker2: string;
  combinationLogic: string;
  action: string;
  timeframe: string;
  exchange: string;
  country: string;
  ratio: string;
  isRatio: boolean;
  adjustmentMethod: string;
};

export async function createAlert(data: CreateAlertInput): Promise<AlertData | null> {
  const response = await alertClient.createAlert(data as CreateAlertRequest);
  return response.alert ? toAlertData(response.alert) : null;
}

export type BulkCreateResult = {
  created: number;
  failed: number;
  skippedDuplicates: number;
  errors: string[];
};

export async function createAlertsBulk(
  shared: Omit<CreateAlertInput, "name" | "stockName" | "ticker">,
  items: { ticker: string; stockName: string; name?: string }[]
): Promise<BulkCreateResult> {
  const result: BulkCreateResult = {
    created: 0,
    failed: 0,
    skippedDuplicates: 0,
    errors: [],
  };
  for (const item of items) {
    try {
      const name = item.name?.trim() || `${item.stockName || item.ticker} Alert`;
      const stockName = item.stockName?.trim() || item.ticker;
      const alert = await createAlert({
        ...shared,
        name,
        stockName,
        ticker: item.ticker.trim(),
      });
      if (alert) result.created++;
      else result.failed++;
    } catch (e) {
      result.failed++;
      result.errors.push(
        `${item.ticker}: ${e instanceof Error ? e.message : String(e)}`
      );
    }
  }
  return result;
}

export async function updateAlert(
  data: Omit<UpdateAlertRequest, "conditions" | "dtpParams" | "multiTimeframeParams" | "mixedTimeframeParams" | "rawPayload">
): Promise<AlertData | null> {
  const response = await alertClient.updateAlert(data);
  return response.alert ? toAlertData(response.alert) : null;
}

export async function deleteAlert(alertId: string): Promise<void> {
  await alertClient.deleteAlert({ alertId });
}

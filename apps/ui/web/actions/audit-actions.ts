"use server";

import { alertClient } from "@/lib/grpc/channel";
import type {
  GetPerformanceMetricsResponse,
  GetFailedPriceDataResponse,
} from "../../../../gen/ts/alert/v1/alert";

/** Row type for audit summary; defined here to avoid re-export from generated proto (avoids SSR ReferenceError). */
export interface AuditSummaryRow {
  alertId: string;
  ticker: string;
  stockName: string;
  exchange: string;
  timeframe: string;
  action: string;
  evaluationType: string;
  totalChecks: number;
  successfulPricePulls: number;
  successfulEvaluations: number;
  totalTriggers: number;
  avgExecutionTimeMs: number;
  lastCheck?: Date | undefined;
  firstCheck?: Date | undefined;
}

/** Row type for alert history; defined here to avoid re-export from generated proto (avoids SSR ReferenceError). */
export interface AuditHistoryRow {
  id: number;
  timestamp?: Date | undefined;
  alertId: string;
  ticker: string;
  stockName: string;
  exchange: string;
  timeframe: string;
  action: string;
  evaluationType: string;
  priceDataPulled: boolean;
  priceDataSource: string;
  conditionsEvaluated: boolean;
  alertTriggered: boolean;
  triggerReason: string;
  executionTimeMs: number;
  cacheHit: boolean;
  errorMessage: string;
  alertName: string;
}

export type PerformanceMetrics = GetPerformanceMetricsResponse;

export type FailedPriceData = GetFailedPriceDataResponse;

export type DashboardStatsResponse = {
  activeAlerts: number;
  triggeredToday: number;
  watchedSymbols: number;
  triggersLast7d: number;
};

export async function getDashboardStatsFromServer(): Promise<DashboardStatsResponse> {
  const response = await alertClient.getDashboardStats({});
  return {
    activeAlerts: response.activeAlerts ?? 0,
    triggeredToday: response.triggeredToday ?? 0,
    watchedSymbols: response.watchedSymbols ?? 0,
    triggersLast7d: response.triggersLast7d ?? 0,
  };
}

export type TriggerCountByDayRow = { date: string; count: number };

export async function getTriggerCountByDayFromServer(
  days: number = 30
): Promise<TriggerCountByDayRow[]> {
  const clamped = Math.min(90, Math.max(7, days));
  const response = await alertClient.getTriggerCountByDay({ days: clamped });
  return (response.rows ?? []).map((r) => ({
    date: r.date ?? "",
    count: Number(r.count ?? 0),
  }));
}

export async function getAuditSummary(
  days: number = 7,
  limit?: number
): Promise<AuditSummaryRow[]> {
  const daysClamped = Math.min(90, Math.max(1, days));
  const response = await alertClient.getAuditSummary({
    days: daysClamped,
    ...(limit != null && limit > 0 && { limit }),
  });
  return response.rows ?? [];
}

export async function getPerformanceMetrics(
  days: number = 7
): Promise<PerformanceMetrics | null> {
  const daysClamped = Math.min(90, Math.max(1, days));
  const response = await alertClient.getPerformanceMetrics({ days: daysClamped });
  return response;
}

export async function getAlertHistory(
  alertId: string,
  limit: number = 100
): Promise<AuditHistoryRow[]> {
  if (!alertId.trim()) return [];
  const response = await alertClient.getAlertHistory({
    alertId: alertId.trim(),
    limit: Math.min(1000, Math.max(1, limit)),
  });
  return response.rows ?? [];
}

export async function getFailedPriceData(
  days: number = 7
): Promise<FailedPriceData | null> {
  const daysClamped = Math.min(90, Math.max(1, days));
  const response = await alertClient.getFailedPriceData({ days: daysClamped });
  return response;
}

export async function clearAuditData(): Promise<number> {
  const response = await alertClient.clearAuditData({});
  return Number(response.deletedCount ?? 0);
}

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

export async function getAuditSummary(days: number = 7): Promise<AuditSummaryRow[]> {
  const daysClamped = Math.min(90, Math.max(1, days));
  const response = await alertClient.getAuditSummary({ days: daysClamped });
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

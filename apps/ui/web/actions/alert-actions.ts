"use server";

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
  };
}

const DEFAULT_PAGE_SIZE = 20;
const MAX_PAGE_SIZE = 100;

export type ListAlertsResult = {
  alerts: AlertData[];
  totalCount: number;
  hasNextPage: boolean;
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

export async function getAlert(alertId: string): Promise<AlertData | null> {
  const response = await alertClient.getAlert({ alertId });
  return response.alert ? toAlertData(response.alert) : null;
}

export async function createAlert(
  data: Omit<CreateAlertRequest, "conditions" | "dtpParams" | "multiTimeframeParams" | "mixedTimeframeParams" | "rawPayload">
): Promise<AlertData | null> {
  const response = await alertClient.createAlert(data);
  return response.alert ? toAlertData(response.alert) : null;
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

"use server";

import { alertClient } from "@/lib/grpc/channel";
import type {
  StockSearchResult,
  AuditHistoryRow,
  Portfolio,
} from "../../../../gen/ts/alert/v1/alert";

export type { StockSearchResult, Portfolio, AuditHistoryRow };

export type TriggerHistoryOptions = {
  includeAllEvaluations?: boolean;
  limit?: number;
  daysBack?: number;
};

export async function searchStocks(
  query: string,
  limit: number = 20
): Promise<StockSearchResult[]> {
  const q = query?.trim() ?? "";
  if (!q) return [];
  const response = await alertClient.searchStocks({
    query: q,
    limit: Math.min(50, Math.max(1, limit)),
  });
  return response.results ?? [];
}

export async function getTriggerHistoryByTicker(
  ticker: string,
  options: TriggerHistoryOptions = {}
): Promise<AuditHistoryRow[]> {
  const t = ticker?.trim() ?? "";
  if (!t) return [];
  const {
    includeAllEvaluations = false,
    limit = 50,
    daysBack = 0,
  } = options;
  const response = await alertClient.getTriggerHistoryByTicker({
    ticker: t,
    includeAllEvaluations,
    limit: Math.min(500, Math.max(1, limit)),
    daysBack: Math.max(0, daysBack),
  });
  return response.rows ?? [];
}

export async function listPortfolios(): Promise<Portfolio[]> {
  const response = await alertClient.listPortfolios({});
  return response.portfolios ?? [];
}

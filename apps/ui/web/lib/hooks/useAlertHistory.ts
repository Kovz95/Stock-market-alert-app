"use client";

import { useQuery } from "@tanstack/react-query";
import {
  searchStocks,
  getTriggerHistoryByTicker,
  listPortfolios,
} from "@/actions/alert-history-actions";
import { listAllAlertsForHistory } from "@/actions/alert-actions";
import type { TriggerHistoryOptions } from "@/actions/alert-history-actions";

export const ALERT_HISTORY_KEY = ["alert-history"] as const;

export function useStockSearch(query: string, enabled: boolean = true) {
  return useQuery({
    queryKey: [...ALERT_HISTORY_KEY, "stock-search", query],
    queryFn: () => searchStocks(query, 20),
    enabled: enabled && (query?.trim().length ?? 0) >= 2,
  });
}

export function useTriggerHistoryByTicker(
  ticker: string,
  options: TriggerHistoryOptions = {}
) {
  const { includeAllEvaluations = false, limit = 50, daysBack = 0 } = options;
  return useQuery({
    queryKey: [
      ...ALERT_HISTORY_KEY,
      "trigger-history",
      ticker,
      includeAllEvaluations,
      limit,
      daysBack,
    ],
    queryFn: () =>
      getTriggerHistoryByTicker(ticker, {
        includeAllEvaluations,
        limit,
        daysBack,
      }),
    enabled: !!ticker?.trim(),
  });
}

export function usePortfolios() {
  return useQuery({
    queryKey: [...ALERT_HISTORY_KEY, "portfolios"],
    queryFn: listPortfolios,
  });
}

export function useAlertsForHistory() {
  return useQuery({
    queryKey: [...ALERT_HISTORY_KEY, "all-alerts"],
    queryFn: listAllAlertsForHistory,
  });
}

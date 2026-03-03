"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getStockMetadataMap,
  getDatabaseStats,
  loadPriceData,
  updatePrices,
  scanStaleDaily,
  scanStaleWeekly,
  scanStaleHourly,
  getHourlyDataQuality,
  type LoadPriceDataParams,
  type UpdatePricesParams,
  type StockMetadataItem,
  type DatabaseStatsData,
  type PriceRowData,
  type StaleTickerRowData,
  type StaleHourlyRowData,
  type ScanStaleHourlyResult,
  type HourlyDataQualityData,
} from "@/actions/price-database-actions";

export const PRICE_METADATA_KEY = ["price", "metadata"] as const;
export const PRICE_STATS_KEY = ["price", "stats"] as const;
export const PRICE_DATA_KEY = ["price", "data"] as const;
export const STALE_DAILY_KEY = ["price", "stale", "daily"] as const;
export const STALE_WEEKLY_KEY = ["price", "stale", "weekly"] as const;
export const STALE_HOURLY_KEY = ["price", "stale", "hourly"] as const;
export const HOURLY_QUALITY_KEY = ["price", "hourly", "quality"] as const;

export function useStockMetadata() {
  return useQuery({
    queryKey: PRICE_METADATA_KEY,
    queryFn: getStockMetadataMap,
    staleTime: 1000 * 60 * 5, // 5 minutes
  });
}

export function useDatabaseStats() {
  return useQuery({
    queryKey: PRICE_STATS_KEY,
    queryFn: getDatabaseStats,
    staleTime: 1000 * 60 * 2, // 2 minutes
  });
}

export function useLoadPriceData() {
  return useMutation({
    mutationFn: (params: LoadPriceDataParams) => loadPriceData(params),
  });
}

export function useUpdatePrices() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (params: UpdatePricesParams) => updatePrices(params),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: PRICE_STATS_KEY });
    },
  });
}

// Convenience: metadata as map by symbol (for filters/display)
export function useStockMetadataMap(): {
  data: Record<string, StockMetadataItem> | undefined;
  isLoading: boolean;
  error: Error | null;
} {
  const { data: items, isLoading, error } = useStockMetadata();
  const map: Record<string, StockMetadataItem> | undefined = items?.length
    ? Object.fromEntries(items.map((item) => [item.symbol, item]))
    : undefined;
  return { data: map, isLoading, error: error as Error | null };
}

// Stale scan: mutations (user triggers scan)
export function useScanStaleDaily() {
  return useMutation({
    mutationFn: (limit?: number) => scanStaleDaily(limit),
  });
}

export function useScanStaleWeekly() {
  return useMutation({
    mutationFn: (limit?: number) => scanStaleWeekly(limit),
  });
}

export function useScanStaleHourly() {
  return useMutation({
    mutationFn: (limit?: number) => scanStaleHourly(limit),
  });
}

// Hourly data quality: query (can refetch on tab focus)
export function useHourlyDataQuality(enabled = true) {
  return useQuery({
    queryKey: HOURLY_QUALITY_KEY,
    queryFn: getHourlyDataQuality,
    enabled,
    staleTime: 1000 * 60 * 2, // 2 minutes
  });
}

export type {
  StockMetadataItem,
  DatabaseStatsData,
  PriceRowData,
  LoadPriceDataParams,
  UpdatePricesParams,
  StaleTickerRowData,
  StaleHourlyRowData,
  ScanStaleHourlyResult,
  HourlyDataQualityData,
};

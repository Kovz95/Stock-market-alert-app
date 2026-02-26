"use client";

import { useQuery, useMutation } from "@tanstack/react-query";
import {
  getStockMetadataMap,
  getDatabaseStats,
  loadPriceData,
  type LoadPriceDataParams,
  type StockMetadataItem,
  type DatabaseStatsData,
  type PriceRowData,
} from "@/actions/price-database-actions";

export const PRICE_METADATA_KEY = ["price", "metadata"] as const;
export const PRICE_STATS_KEY = ["price", "stats"] as const;
export const PRICE_DATA_KEY = ["price", "data"] as const;

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

export type {
  StockMetadataItem,
  DatabaseStatsData,
  PriceRowData,
  LoadPriceDataParams,
};

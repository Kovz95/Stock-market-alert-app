"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getFullStockMetadata,
  previewDeleteTicker,
  deleteTicker,
  type FullStockMetadataRow,
} from "@/actions/stock-database-actions";
import { ALERTS_KEY } from "@/lib/store/alerts";
import { PORTFOLIOS_KEY } from "@/lib/hooks/usePortfolios";

export const STOCK_DATABASE_METADATA_KEY = ["stock-database", "metadata"] as const;

async function fetchFullStockMetadata(): Promise<FullStockMetadataRow[]> {
  const result = await getFullStockMetadata();
  if ("error" in result) {
    throw new Error(result.error);
  }
  return result.data;
}

export function useFullStockMetadata() {
  return useQuery({
    queryKey: STOCK_DATABASE_METADATA_KEY,
    queryFn: fetchFullStockMetadata,
    staleTime: 1000 * 60 * 5, // 5 minutes
  });
}

export function usePreviewDeleteTicker(ticker: string | null) {
  return useQuery({
    queryKey: ["stock-database", "preview-delete", ticker],
    queryFn: () => previewDeleteTicker(ticker!),
    enabled: ticker !== null,
    staleTime: 0,
  });
}

export function useDeleteTicker() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (ticker: string) => deleteTicker(ticker),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: STOCK_DATABASE_METADATA_KEY });
      queryClient.invalidateQueries({ queryKey: ALERTS_KEY });
      queryClient.invalidateQueries({ queryKey: PORTFOLIOS_KEY });
    },
  });
}

export type { FullStockMetadataRow };

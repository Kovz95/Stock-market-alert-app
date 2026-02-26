"use client";

import { useQuery } from "@tanstack/react-query";
import {
  getFullStockMetadata,
  type FullStockMetadataRow,
} from "@/actions/stock-database-actions";

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

export type { FullStockMetadataRow };

"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getAuditSummary,
  getPerformanceMetrics,
  getAlertHistory,
  getFailedPriceData,
  clearAuditData,
} from "@/actions/audit-actions";

export const AUDIT_KEY = ["audit"] as const;

export function useAuditSummary(days: number) {
  return useQuery({
    queryKey: [...AUDIT_KEY, "summary", days],
    queryFn: () => getAuditSummary(days),
  });
}

export function usePerformanceMetrics(days: number) {
  return useQuery({
    queryKey: [...AUDIT_KEY, "metrics", days],
    queryFn: () => getPerformanceMetrics(days),
  });
}

export function useAlertHistory(alertId: string, limit: number = 100) {
  return useQuery({
    queryKey: [...AUDIT_KEY, "history", alertId, limit],
    queryFn: () => getAlertHistory(alertId, limit),
    enabled: !!alertId?.trim(),
  });
}

export function useFailedPriceData(days: number) {
  return useQuery({
    queryKey: [...AUDIT_KEY, "failed", days],
    queryFn: () => getFailedPriceData(days),
  });
}

export function useClearAuditData() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: clearAuditData,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: AUDIT_KEY });
    },
  });
}

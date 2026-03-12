"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAtom } from "jotai";
import {
  getAlert,
  createAlert,
  createAlertsBulk,
  updateAlert,
  deleteAlert,
  bulkDeleteAlerts,
  searchAlerts,
} from "@/actions/alert-actions";
import type { SearchAlertsFilters } from "@/actions/alert-actions";
import {
  ALERTS_KEY,
  alertsPaginatedQueryAtom,
} from "@/lib/store/alerts";

export { ALERTS_KEY };

/** Thin wrapper around alertsPaginatedQueryAtom; returns same shape as useQuery for drop-in use in the alerts page. */
export function useAlertsPaginated() {
  const [result] = useAtom(alertsPaginatedQueryAtom);
  return {
    data: result.data,
    isLoading: result.isPending,
    error: result.error,
    isError: result.isError,
    isFetching: result.isFetching,
    refetch: result.refetch,
  };
}

export function useAlert(alertId: string) {
  return useQuery({
    queryKey: [...ALERTS_KEY, alertId],
    queryFn: () => getAlert(alertId),
    enabled: !!alertId,
  });
}

export function useCreateAlert() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: createAlert,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ALERTS_KEY });
    },
  });
}

export function useCreateAlertsBulk() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      shared,
      items,
    }: {
      shared: Parameters<typeof createAlertsBulk>[0];
      items: Parameters<typeof createAlertsBulk>[1];
    }) => createAlertsBulk(shared, items),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ALERTS_KEY });
    },
  });
}

export function useUpdateAlert() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: updateAlert,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ALERTS_KEY });
      if (data) {
        queryClient.setQueryData([...ALERTS_KEY, data.alertId], data);
      }
    },
  });
}

const SEARCH_ALERTS_KEY = [...ALERTS_KEY, "search"] as const;

/** Server-side filtered + paginated alert search (e.g. delete page). */
export function useSearchAlerts(
  filters: SearchAlertsFilters,
  page: number,
  pageSize: number
) {
  return useQuery({
    queryKey: [...SEARCH_ALERTS_KEY, filters, page, pageSize],
    queryFn: () => searchAlerts(filters, page, pageSize),
    placeholderData: (prev) => prev,
  });
}

export function useDeleteAlert() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: deleteAlert,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ALERTS_KEY });
    },
  });
}

export function useBulkDeleteAlerts() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (alertIds: string[]) => bulkDeleteAlerts(alertIds),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ALERTS_KEY });
    },
  });
}

"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAtom } from "jotai";
import {
  getAlert,
  createAlert,
  createAlertsBulk,
  updateAlert,
  deleteAlert,
  listAllAlertsForHistory,
} from "@/actions/alert-actions";
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

const ALL_ALERTS_KEY = [...ALERTS_KEY, "all"] as const;

/** Fetch all alerts (all pages) for client-side filtering (e.g. delete page). */
export function useAllAlerts() {
  return useQuery({
    queryKey: ALL_ALERTS_KEY,
    queryFn: listAllAlertsForHistory,
    staleTime: 1000 * 60 * 2, // 2 minutes
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

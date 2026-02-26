"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { useAtomValue } from "jotai";
import {
  listAlertsPaginated,
  getAlert,
  createAlert,
  createAlertsBulk,
  updateAlert,
  deleteAlert,
} from "@/actions/alert-actions";
import { alertsPageAtom, alertsPageSizeAtom } from "@/lib/store/alerts";

export const ALERTS_KEY = ["alerts"] as const;

export function useAlertsPaginated() {
  const page = useAtomValue(alertsPageAtom);
  const pageSize = useAtomValue(alertsPageSizeAtom);
  return useQuery({
    queryKey: [...ALERTS_KEY, "paginated", page, pageSize],
    queryFn: () => listAlertsPaginated(page, pageSize),
  });
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

export function useDeleteAlert() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: deleteAlert,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ALERTS_KEY });
    },
  });
}

"use client";

import { useQuery } from "@tanstack/react-query";
import {
  getDashboardStats,
  getTriggerCountByDay,
  getDashboardAlerts,
} from "@/actions/dashboard-actions";

export const DASHBOARD_KEY = ["dashboard"] as const;

const DASHBOARD_ALERTS_PAGE_SIZE = 10;

export function useDashboardStats() {
  return useQuery({
    queryKey: [...DASHBOARD_KEY, "stats"],
    queryFn: getDashboardStats,
  });
}

export function useTriggerCountByDay(days: number) {
  return useQuery({
    queryKey: [...DASHBOARD_KEY, "trigger-count-by-day", days],
    queryFn: () => getTriggerCountByDay(days),
  });
}

export function useDashboardAlerts() {
  return useQuery({
    queryKey: [...DASHBOARD_KEY, "alerts", 1, DASHBOARD_ALERTS_PAGE_SIZE],
    queryFn: () => getDashboardAlerts(DASHBOARD_ALERTS_PAGE_SIZE),
  });
}

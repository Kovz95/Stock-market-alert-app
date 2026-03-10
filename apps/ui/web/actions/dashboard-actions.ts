"use server";

import { getTopTriggeredAlerts, type AlertData } from "@/actions/alert-actions";
import {
  getDashboardStatsFromServer,
  getTriggerCountByDayFromServer,
} from "@/actions/audit-actions";

export type DashboardStats = {
  activeAlerts: number;
  triggeredToday: number;
  watchedSymbols: number;
  triggersLast7d: number;
  /** Set when one or more backend calls failed; stats are best-effort or zeros. */
  error?: string;
};

/** Stats for the dashboard KPI cards. Uses GetDashboardStats RPC (lightweight). Never throws; returns fallback and optional error. */
export async function getDashboardStats(): Promise<DashboardStats> {
  const fallback: DashboardStats = {
    activeAlerts: 0,
    triggeredToday: 0,
    watchedSymbols: 0,
    triggersLast7d: 0,
  };
  try {
    const data = await getDashboardStatsFromServer();
    return {
      activeAlerts: data.activeAlerts,
      triggeredToday: data.triggeredToday,
      watchedSymbols: data.watchedSymbols,
      triggersLast7d: data.triggersLast7d,
    };
  } catch (err) {
    const message =
      err instanceof Error ? err.message : "Failed to load dashboard stats";
    return { ...fallback, error: message };
  }
}

export type TriggerCountByDay = { date: string; count: number }[];

/** Alert trigger counts by day for the activity chart. Uses GetTriggerCountByDay RPC (same source as dashboard stats). Never throws. */
export async function getTriggerCountByDay(
  days: number = 90
): Promise<TriggerCountByDay> {
  const clampedDays = Math.min(90, Math.max(7, days));
  try {
    return await getTriggerCountByDayFromServer(clampedDays);
  } catch {
    return fillEmptyDays(clampedDays);
  }
}

export type DashboardAlertsResult = {
  alerts: AlertData[];
  totalCount: number;
  hasNextPage: boolean;
  /** alertId -> trigger count (only present for dashboard top-triggered list). */
  triggerCountByAlertId?: Record<string, number>;
  error?: string;
};

/** Top N most-triggered alerts for the dashboard (from alert_audits where alert_triggered = true). Never throws. */
export async function getDashboardAlerts(
  limit: number = 10
): Promise<DashboardAlertsResult> {
  try {
    const result = await getTopTriggeredAlerts(30, limit);
    return {
      alerts: result.alerts,
      totalCount: result.totalCount,
      hasNextPage: result.hasNextPage,
      triggerCountByAlertId: result.triggerCountByAlertId,
    };
  } catch (err) {
    const message =
      err instanceof Error ? err.message : "Failed to load alerts";
    return {
      alerts: [],
      totalCount: 0,
      hasNextPage: false,
      error: message,
    };
  }
}

function fillEmptyDays(days: number): TriggerCountByDay {
  const result: TriggerCountByDay = [];
  const start = new Date();
  start.setDate(start.getDate() - days);
  start.setHours(0, 0, 0, 0);
  for (let d = 0; d < days; d++) {
    const day = new Date(start);
    day.setDate(day.getDate() + d);
    result.push({ date: day.toISOString().slice(0, 10), count: 0 });
  }
  return result;
}

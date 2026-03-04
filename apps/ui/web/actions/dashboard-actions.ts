"use server";

import {
  listAlertsPaginated,
  listAllAlertsForHistory,
} from "@/actions/alert-actions";
import { getAuditSummary, getAlertHistory } from "@/actions/audit-actions";
import type { AlertData } from "@/actions/alert-actions";

export type DashboardStats = {
  activeAlerts: number;
  triggeredToday: number;
  watchedSymbols: number;
  triggersLast7d: number;
  /** Set when one or more backend calls failed; stats are best-effort or zeros. */
  error?: string;
};

/** Stats for the dashboard KPI cards. Never throws; returns fallback and optional error. */
export async function getDashboardStats(): Promise<DashboardStats> {
  const fallback: DashboardStats = {
    activeAlerts: 0,
    triggeredToday: 0,
    watchedSymbols: 0,
    triggersLast7d: 0,
  };
  try {
    const [alertsPage, allAlerts, summary1d, summary7d] = await Promise.all([
      listAlertsPaginated(1, 1),
      listAllAlertsForHistory(),
      getAuditSummary(1),
      getAuditSummary(7),
    ]);

    const uniqueTickers = new Set(
      allAlerts.map((a) => (a.isRatio ? a.ratio : a.ticker)).filter(Boolean)
    );

    const triggeredToday = summary1d.reduce(
      (s, r) => s + (r.totalTriggers ?? 0),
      0
    );
    const triggersLast7d = summary7d.reduce(
      (s, r) => s + (r.totalTriggers ?? 0),
      0
    );

    return {
      activeAlerts: alertsPage.totalCount,
      triggeredToday,
      watchedSymbols: uniqueTickers.size,
      triggersLast7d,
    };
  } catch (err) {
    const message =
      err instanceof Error ? err.message : "Failed to load dashboard stats";
    return { ...fallback, error: message };
  }
}

export type TriggerCountByDay = { date: string; count: number }[];

/** Alert trigger counts by day for the activity chart. Samples up to maxAlerts alerts. Never throws. */
export async function getTriggerCountByDay(
  days: number = 90,
  maxAlerts: number = 30
): Promise<TriggerCountByDay> {
  const clampedDays = Math.min(90, Math.max(7, days));
  try {
    const result = await listAlertsPaginated(1, maxAlerts);
    if (!result.alerts.length) {
      return fillEmptyDays(clampedDays);
    }

    const historyByAlert = await Promise.all(
      result.alerts.map((a) =>
        getAlertHistory(a.alertId, 200).then((rows) =>
          rows.filter((r) => r.alertTriggered && r.timestamp)
        )
      )
    );

    const countByDay: Record<string, number> = {};
    const start = new Date();
    start.setDate(start.getDate() - clampedDays);
    start.setHours(0, 0, 0, 0);

    for (let d = 0; d < clampedDays; d++) {
      const day = new Date(start);
      day.setDate(day.getDate() + d);
      const key = day.toISOString().slice(0, 10);
      countByDay[key] = 0;
    }

    for (const rows of historyByAlert) {
      for (const row of rows) {
        const ts = row.timestamp!;
        const day = new Date(ts);
        day.setHours(0, 0, 0, 0);
        const key = day.toISOString().slice(0, 10);
        if (key in countByDay) countByDay[key]++;
      }
    }

    return Object.entries(countByDay)
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([date, count]) => ({ date, count }));
  } catch {
    return fillEmptyDays(clampedDays);
  }
}

export type DashboardAlertsResult = {
  alerts: AlertData[];
  totalCount: number;
  hasNextPage: boolean;
  error?: string;
};

/** First page of alerts for the dashboard. Never throws. */
export async function getDashboardAlerts(
  pageSize: number = 10
): Promise<DashboardAlertsResult> {
  try {
    const result = await listAlertsPaginated(1, pageSize);
    return {
      alerts: result.alerts,
      totalCount: result.totalCount,
      hasNextPage: result.hasNextPage,
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

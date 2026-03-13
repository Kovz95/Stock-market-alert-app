"use client";

import { useAtomValue } from "jotai";
import { usePerformanceMetrics, useAuditSummary, useTriggerCountByDay } from "@/lib/hooks/useAudit";
import { auditDaysAtom } from "@/lib/store/audit";
import { OverviewKpiCards } from "./OverviewKpiCards";
import { OverviewTriggerChart } from "./OverviewTriggerChart";
import { OverviewSuccessRateChart } from "./OverviewSuccessRateChart";

export function OverviewTab() {
  const days = useAtomValue(auditDaysAtom);
  const { data: metrics, isLoading: metricsLoading } = usePerformanceMetrics(days);
  const { data: summaryRows } = useAuditSummary(days);
  const { data: triggerData } = useTriggerCountByDay(days);

  if (metricsLoading) {
    return <p className="text-muted-foreground text-sm">Loading metrics...</p>;
  }

  if (!metrics) {
    return <p className="text-muted-foreground text-sm">No audit data for the selected period.</p>;
  }

  return (
    <div className="space-y-6">
      <OverviewKpiCards metrics={metrics} />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {triggerData && <OverviewTriggerChart data={triggerData} />}
        {summaryRows && <OverviewSuccessRateChart summaryRows={summaryRows} />}
      </div>
    </div>
  );
}

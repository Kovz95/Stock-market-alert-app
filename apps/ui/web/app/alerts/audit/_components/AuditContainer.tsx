"use client";

import * as React from "react";
import { useAtom, useAtomValue, useSetAtom } from "jotai";
import { useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AUDIT_KEY, useClearAuditData, useAuditSummary } from "@/lib/hooks/useAudit";
import {
  auditActiveTabAtom,
  auditDaysAtom,
  auditAlertIdFilterAtom,
  auditTickerFilterAtom,
  auditEvalTypeFilterAtom,
  auditStatusFilterAtom,
  auditLogPageAtom,
  auditAutoRefreshAtom,
  auditClearDialogOpenAtom,
  type AuditTab,
} from "@/lib/store/audit";
import { AuditFilters } from "./AuditFilters";
import { AuditQuickActions } from "./AuditQuickActions";
import { ClearAuditDialog } from "./ClearAuditDialog";
import { OverviewTab } from "./OverviewTab";
import { EvaluationLogTab } from "./EvaluationLogTab";
import { FailedDataTab } from "./FailedDataTab";
import { formatAuditDateShort } from "./formatAuditDate";
import type { AuditSummaryRow } from "@/actions/audit-actions";

export function AuditContainer() {
  const queryClient = useQueryClient();
  const clearMutation = useClearAuditData();

  const [activeTab, setActiveTab] = useAtom(auditActiveTabAtom);
  const days = useAtomValue(auditDaysAtom);
  const alertId = useAtomValue(auditAlertIdFilterAtom);
  const ticker = useAtomValue(auditTickerFilterAtom);
  const evalType = useAtomValue(auditEvalTypeFilterAtom);
  const statusFilter = useAtomValue(auditStatusFilterAtom);
  const autoRefresh = useAtomValue(auditAutoRefreshAtom);
  const setPage = useSetAtom(auditLogPageAtom);
  const setClearOpen = useSetAtom(auditClearDialogOpenAtom);

  // For CSV export we need the summary data
  const { data: summaryRows } = useAuditSummary(days);

  // Reset page to 1 when any filter changes (debounced)
  React.useEffect(() => {
    const timer = setTimeout(() => {
      setPage(1);
    }, 300);
    return () => clearTimeout(timer);
  }, [days, alertId, ticker, evalType, statusFilter, setPage]);

  // Auto-refresh: invalidate all audit queries every 30s
  React.useEffect(() => {
    if (!autoRefresh) return;
    const interval = setInterval(() => {
      queryClient.invalidateQueries({ queryKey: [...AUDIT_KEY] });
    }, 30_000);
    return () => clearInterval(interval);
  }, [autoRefresh, queryClient]);

  const handleRefresh = () => {
    queryClient.invalidateQueries({ queryKey: [...AUDIT_KEY] });
  };

  const handleClear = async () => {
    try {
      const count = await clearMutation.mutateAsync();
      setClearOpen(false);
      toast.success(`Cleared ${count.toLocaleString()} audit records`);
    } catch {
      toast.error("Failed to clear audit data");
    }
  };

  const handleExportCsv = () => {
    const rows = summaryRows;
    if (!rows?.length) return;
    const headers = [
      "Alert ID", "Ticker", "Stock Name", "Exchange", "Timeframe",
      "Action", "Type", "Checks", "Price Pulls", "Evaluations",
      "Triggers", "Avg Time (ms)", "Last Check", "First Check",
    ];
    const toStr = (d: Date | undefined) => (d ? formatAuditDateShort(d) : "");
    const lines = [
      headers.join(","),
      ...rows.map((r: AuditSummaryRow) =>
        [
          r.alertId,
          r.ticker,
          (r.stockName ?? "").replace(/"/g, '""'),
          r.exchange,
          r.timeframe,
          r.action,
          r.evaluationType,
          r.totalChecks,
          r.successfulPricePulls,
          r.successfulEvaluations,
          r.totalTriggers,
          r.avgExecutionTimeMs?.toFixed(1) ?? "",
          toStr(r.lastCheck),
          toStr(r.firstCheck),
        ].join(",")
      ),
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `alert_audit_summary_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <AuditFilters />
        <AuditQuickActions
          onRefresh={handleRefresh}
          onExportCsv={handleExportCsv}
          hasData={(summaryRows?.length ?? 0) > 0}
        />
      </div>

      <Tabs
        value={activeTab}
        onValueChange={(v) => setActiveTab(v as AuditTab)}
      >
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="evaluation-log">Evaluation Log</TabsTrigger>
          <TabsTrigger value="failed-data">Failed Data</TabsTrigger>
        </TabsList>

        <TabsContent value="overview" className="mt-4">
          <OverviewTab />
        </TabsContent>

        <TabsContent value="evaluation-log" className="mt-4">
          <EvaluationLogTab />
        </TabsContent>

        <TabsContent value="failed-data" className="mt-4">
          <FailedDataTab />
        </TabsContent>
      </Tabs>

      <ClearAuditDialog onConfirm={handleClear} isPending={clearMutation.isPending} />
    </div>
  );
}

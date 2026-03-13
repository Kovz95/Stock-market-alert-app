"use client";

import { useAtomValue } from "jotai";
import { useAuditLog } from "@/lib/hooks/useAudit";
import {
  auditDaysAtom,
  auditAlertIdFilterAtom,
  auditTickerFilterAtom,
  auditEvalTypeFilterAtom,
  auditStatusFilterAtom,
  auditLogPageAtom,
  auditLogPageSizeAtom,
  auditLogSortFieldAtom,
  auditLogSortDirectionAtom,
} from "@/lib/store/audit";
import { EvaluationLogTable } from "./EvaluationLogTable";
import { EvaluationLogPagination } from "./EvaluationLogPagination";
import { AuditDetailSheet } from "./AuditDetailSheet";

const STATUS_MAP: Record<string, string> = {
  All: "",
  Success: "success",
  Error: "error",
  Triggered: "triggered",
  "Not Triggered": "not_triggered",
};

export function EvaluationLogTab() {
  const days = useAtomValue(auditDaysAtom);
  const alertId = useAtomValue(auditAlertIdFilterAtom);
  const ticker = useAtomValue(auditTickerFilterAtom);
  const evalType = useAtomValue(auditEvalTypeFilterAtom);
  const statusFilter = useAtomValue(auditStatusFilterAtom);
  const page = useAtomValue(auditLogPageAtom);
  const pageSize = useAtomValue(auditLogPageSizeAtom);
  const sortField = useAtomValue(auditLogSortFieldAtom);
  const sortDirection = useAtomValue(auditLogSortDirectionAtom);

  const { data, isLoading } = useAuditLog({
    days,
    page,
    pageSize,
    sortField,
    sortDirection,
    alertId: alertId.trim() || undefined,
    ticker: ticker.trim() || undefined,
    evaluationType: evalType !== "All" ? evalType : undefined,
    statusFilter: STATUS_MAP[statusFilter] || undefined,
  });

  return (
    <div className="space-y-4">
      <EvaluationLogTable rows={data?.rows ?? []} isLoading={isLoading} />
      <EvaluationLogPagination
        totalCount={data?.totalCount ?? 0}
        pageSize={pageSize}
      />
      <AuditDetailSheet />
    </div>
  );
}

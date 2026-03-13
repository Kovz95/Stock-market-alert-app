"use client";

import { useAtom, useSetAtom } from "jotai";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { ChevronUpIcon, ChevronDownIcon } from "lucide-react";
import {
  auditLogSortFieldAtom,
  auditLogSortDirectionAtom,
  auditSelectedAlertIdAtom,
  auditDetailSheetOpenAtom,
} from "@/lib/store/audit";
import type { SortDirection } from "@/lib/store/audit";
import { formatAuditDateTime } from "./formatAuditDate";
import type { AuditHistoryRow } from "@/actions/audit-actions";

type SortableColumn = {
  key: string;
  label: string;
  align?: "right";
  sortable?: boolean;
};

const COLUMNS: SortableColumn[] = [
  { key: "timestamp", label: "Timestamp", sortable: true },
  { key: "alert_id", label: "Alert ID", sortable: true },
  { key: "ticker", label: "Ticker", sortable: true },
  { key: "evaluation_type", label: "Type", sortable: true },
  { key: "exchange", label: "Exchange", sortable: true },
  { key: "timeframe", label: "Timeframe", sortable: true },
  { key: "price_data", label: "Price data" },
  { key: "evaluated", label: "Evaluated" },
  { key: "triggered", label: "Triggered" },
  { key: "execution_time_ms", label: "Time (ms)", align: "right", sortable: true },
  { key: "error", label: "Error" },
];

type EvaluationLogTableProps = {
  rows: AuditHistoryRow[];
  isLoading: boolean;
};

export function EvaluationLogTable({ rows, isLoading }: EvaluationLogTableProps) {
  const [sortField, setSortField] = useAtom(auditLogSortFieldAtom);
  const [sortDir, setSortDir] = useAtom(auditLogSortDirectionAtom);
  const setSelectedAlertId = useSetAtom(auditSelectedAlertIdAtom);
  const setDetailOpen = useSetAtom(auditDetailSheetOpenAtom);

  const handleSort = (field: string) => {
    if (sortField === field) {
      setSortDir((d: SortDirection) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortField(field);
      setSortDir("desc");
    }
  };

  const handleRowClick = (alertId: string) => {
    setSelectedAlertId(alertId);
    setDetailOpen(true);
  };

  if (isLoading) {
    return <p className="text-sm text-muted-foreground">Loading evaluation log...</p>;
  }

  if (rows.length === 0) {
    return <p className="text-sm text-muted-foreground">No audit records found for the selected filters.</p>;
  }

  return (
    <div className="border rounded-md overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            {COLUMNS.map((col) => (
              <TableHead
                key={col.key}
                className={`${col.align === "right" ? "text-right" : ""} ${col.sortable ? "cursor-pointer select-none hover:text-foreground" : ""}`}
                onClick={col.sortable ? () => handleSort(col.key) : undefined}
              >
                <span className="inline-flex items-center gap-1">
                  {col.label}
                  {col.sortable && sortField === col.key && (
                    sortDir === "asc"
                      ? <ChevronUpIcon className="size-3" />
                      : <ChevronDownIcon className="size-3" />
                  )}
                </span>
              </TableHead>
            ))}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((r) => (
            <TableRow
              key={r.id}
              className="cursor-pointer hover:bg-muted/50"
              onClick={() => handleRowClick(r.alertId)}
            >
              <TableCell className="text-xs whitespace-nowrap">
                {formatAuditDateTime(r.timestamp)}
              </TableCell>
              <TableCell className="font-mono text-xs">{r.alertId}</TableCell>
              <TableCell>{r.ticker}</TableCell>
              <TableCell>{r.evaluationType}</TableCell>
              <TableCell>{r.exchange || "—"}</TableCell>
              <TableCell>{r.timeframe || "—"}</TableCell>
              <TableCell>{r.priceDataPulled ? "Yes" : "No"}</TableCell>
              <TableCell>{r.conditionsEvaluated ? "Yes" : "No"}</TableCell>
              <TableCell>{r.alertTriggered ? "Yes" : "No"}</TableCell>
              <TableCell className="text-right">
                {r.executionTimeMs != null ? r.executionTimeMs : "—"}
              </TableCell>
              <TableCell className="max-w-[180px] truncate text-destructive" title={r.errorMessage ?? ""}>
                {r.errorMessage || "—"}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

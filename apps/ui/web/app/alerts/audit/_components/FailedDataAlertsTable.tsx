"use client";

import { useState } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { ChevronLeftIcon, ChevronRightIcon } from "lucide-react";
import { formatAuditDateShort } from "./formatAuditDate";

type FailedAlertRow = {
  alertId: string;
  ticker: string;
  assetType: string;
  stockName: string;
  exchange: string;
  timeframe: string;
  failureCount: number;
  lastFailure?: Date;
  firstFailure?: Date;
  avgExecutionTime: number;
};

type FailedDataAlertsTableProps = {
  rows: FailedAlertRow[];
};

const PAGE_SIZE = 20;

export function FailedDataAlertsTable({ rows }: FailedDataAlertsTableProps) {
  const [page, setPage] = useState(1);

  if (!rows?.length) return null;

  const totalPages = Math.max(1, Math.ceil(rows.length / PAGE_SIZE));
  const pageRows = rows.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  return (
    <div>
      <h3 className="text-sm font-medium mb-2">Failed alerts ({rows.length} total)</h3>
      <div className="border rounded-md overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Alert ID</TableHead>
              <TableHead>Ticker</TableHead>
              <TableHead>Asset type</TableHead>
              <TableHead>Stock name</TableHead>
              <TableHead>Exchange</TableHead>
              <TableHead>Timeframe</TableHead>
              <TableHead className="text-right">Failures</TableHead>
              <TableHead>Last failure</TableHead>
              <TableHead>First failure</TableHead>
              <TableHead className="text-right">Avg (ms)</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {pageRows.map((r) => (
              <TableRow key={`${r.alertId}-${r.ticker}`}>
                <TableCell className="font-mono text-xs">{r.alertId}</TableCell>
                <TableCell>{r.ticker}</TableCell>
                <TableCell>{r.assetType ?? "Unknown"}</TableCell>
                <TableCell>{r.stockName ?? "—"}</TableCell>
                <TableCell>{r.exchange ?? "—"}</TableCell>
                <TableCell>{r.timeframe ?? "—"}</TableCell>
                <TableCell className="text-right">{r.failureCount}</TableCell>
                <TableCell className="text-xs whitespace-nowrap">
                  {formatAuditDateShort(r.lastFailure)}
                </TableCell>
                <TableCell className="text-xs whitespace-nowrap">
                  {formatAuditDateShort(r.firstFailure)}
                </TableCell>
                <TableCell className="text-right">
                  {r.avgExecutionTime != null ? r.avgExecutionTime.toFixed(0) : "—"}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
      {totalPages > 1 && (
        <div className="flex items-center justify-between mt-2">
          <p className="text-xs text-muted-foreground">
            Page {page} of {totalPages}
          </p>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              disabled={page <= 1}
              onClick={() => setPage((p) => p - 1)}
            >
              <ChevronLeftIcon className="size-4" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              disabled={page >= totalPages}
              onClick={() => setPage((p) => p + 1)}
            >
              <ChevronRightIcon className="size-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}

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

type ExchangeRow = {
  exchange: string;
  failedAlerts: number;
  failureCount: number;
};

type FailedDataExchangeTableProps = {
  rows: ExchangeRow[];
};

const PAGE_SIZE = 15;

export function FailedDataExchangeTable({ rows }: FailedDataExchangeTableProps) {
  const [page, setPage] = useState(1);

  if (!rows?.length) return null;

  const sorted = [...rows].sort((a, b) => b.failureCount - a.failureCount);
  const totalPages = Math.max(1, Math.ceil(sorted.length / PAGE_SIZE));
  const pageRows = sorted.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  return (
    <div>
      <h3 className="text-sm font-medium mb-2">Exchange breakdown</h3>
      <div className="border rounded-md overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Exchange</TableHead>
              <TableHead className="text-right">Failed alerts</TableHead>
              <TableHead className="text-right">Failures</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {pageRows.map((row) => (
              <TableRow key={row.exchange}>
                <TableCell>{row.exchange}</TableCell>
                <TableCell className="text-right">{row.failedAlerts}</TableCell>
                <TableCell className="text-right">{row.failureCount}</TableCell>
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

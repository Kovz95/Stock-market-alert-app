"use client";

import * as React from "react";
import { Button } from "@/components/ui/button";
import { DataTable } from "@/components/ui/data-table";
import type { ScanMatch } from "../../../../../../gen/ts/price/v1/price";
import { getScannerColumns } from "./scannerResultsColumns";
import { ScannerResultsToolbar } from "./ScannerResultsToolbar";

type ScannerResultsProps = {
  matches: ScanMatch[];
  onDownloadCsv: () => void;
  scanning?: boolean;
  scanProgress?: { batch: number; totalBatches: number } | null;
};

function ScannerResultsComponent({ matches, onDownloadCsv, scanning, scanProgress }: ScannerResultsProps) {
  const hasMatchDates = React.useMemo(() => matches.some((m) => m.matchDate), [matches]);

  const sorted = React.useMemo(() => {
    if (!hasMatchDates) return matches;
    return [...matches].sort((a, b) => {
      if (a.matchDate && b.matchDate) {
        const cmp = b.matchDate.localeCompare(a.matchDate);
        if (cmp !== 0) return cmp;
      }
      return a.ticker.localeCompare(b.ticker);
    });
  }, [matches, hasMatchDates]);

  const columns = React.useMemo(() => getScannerColumns(hasMatchDates), [hasMatchDates]);

  const statusLine =
    scanning && scanProgress
      ? `Scanning batch ${scanProgress.batch}/${scanProgress.totalBatches}…`
      : "Scan complete.";

  if (matches.length === 0) {
    return (
      <div className="space-y-4 rounded-lg border bg-card p-4">
        <h3 className="font-semibold">Scan results (0 matches)</h3>
        <p className="text-sm text-muted-foreground">{statusLine} No symbols matched your conditions.</p>
      </div>
    );
  }

  return (
    <div className="space-y-4 rounded-lg border bg-card p-4">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <h3 className="font-semibold">
          Scan results ({matches.length} matches){" "}
          {scanning && scanProgress && `— batch ${scanProgress.batch}/${scanProgress.totalBatches}`}
        </h3>
        <Button size="sm" variant="outline" onClick={onDownloadCsv}>
          Download CSV
        </Button>
      </div>
      <DataTable
        columns={columns}
        data={sorted}
        initialSorting={hasMatchDates ? [{ id: "matchDate", desc: true }] : []}
        toolbar={(table) => <ScannerResultsToolbar table={table} />}
      />
    </div>
  );
}

export const ScannerResults = React.memo(ScannerResultsComponent);

export function scanMatchesToCsv(matches: ScanMatch[]): string {
  const hasMatchDates = matches.some((m) => m.matchDate);
  const headers = [
    "ticker",
    "name",
    ...(hasMatchDates ? ["matchDate"] : []),
    "exchange",
    "country",
    "assetType",
    "price",
    "rbicsEconomy",
    "rbicsSector",
    "rbicsSubsector",
    "rbicsIndustryGroup",
    "rbicsIndustry",
    "rbicsSubindustry",
  ];
  const rows = matches.map((m) =>
    headers
      .map((h) => {
        const v = (m as unknown as Record<string, unknown>)[h];
        const s = v == null ? "" : String(v);
        return s.includes(",") || s.includes('"') ? `"${s.replace(/"/g, '""')}"` : s;
      })
      .join(",")
  );
  return [headers.join(","), ...rows].join("\r\n");
}

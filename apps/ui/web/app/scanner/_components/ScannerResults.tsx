"use client";

import * as React from "react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { ScanMatch } from "../../../../../../gen/ts/price/v1/price";

type ScannerResultsProps = {
  matches: ScanMatch[];
  onDownloadCsv: () => void;
  scanning?: boolean;
  scanProgress?: { batch: number; totalBatches: number } | null;
};

export function ScannerResults({ matches, onDownloadCsv, scanning, scanProgress }: ScannerResultsProps) {
  const [search, setSearch] = React.useState("");

  const hasMatchDates = React.useMemo(() => matches.some((m) => m.matchDate), [matches]);

  const sorted = React.useMemo(() => {
    if (!hasMatchDates) return matches;
    return [...matches].sort((a, b) => {
      // Sort by date descending, then ticker ascending
      if (a.matchDate && b.matchDate) {
        const cmp = b.matchDate.localeCompare(a.matchDate);
        if (cmp !== 0) return cmp;
      }
      return a.ticker.localeCompare(b.ticker);
    });
  }, [matches, hasMatchDates]);

  const filtered = React.useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return sorted;
    return sorted.filter((m) => {
      const s = [m.ticker, m.name, m.exchange, m.country, m.assetType, m.matchDate].filter(Boolean).join(" ").toLowerCase();
      return s.includes(q);
    });
  }, [sorted, search]);

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
          Scan results ({matches.length} matches) {scanning && scanProgress && `— batch ${scanProgress.batch}/${scanProgress.totalBatches}`}
        </h3>
        <div className="flex items-center gap-2">
          <Input
            placeholder="Search ticker or name..."
            className="h-8 w-56 text-sm"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
          <Button size="sm" variant="outline" onClick={onDownloadCsv}>
            Download CSV
          </Button>
        </div>
      </div>
      <p className="text-xs text-muted-foreground">
        Showing {filtered.length} of {matches.length}
      </p>
      <div className="overflow-x-auto rounded border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Ticker</TableHead>
              <TableHead>Name</TableHead>
              {hasMatchDates && <TableHead>Match Date</TableHead>}
              <TableHead>Exchange</TableHead>
              <TableHead>Country</TableHead>
              <TableHead>Type</TableHead>
              <TableHead className="text-right">Price</TableHead>
              <TableHead>RBICS Sector</TableHead>
              <TableHead>RBICS Industry</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filtered.slice(0, 500).map((m, i) => (
              <TableRow key={m.matchDate ? `${m.ticker}-${m.matchDate}` : `${m.ticker}-${i}`}>
                <TableCell className="font-medium">{m.ticker}</TableCell>
                <TableCell>{m.name}</TableCell>
                {hasMatchDates && <TableCell>{m.matchDate}</TableCell>}
                <TableCell>{m.exchange}</TableCell>
                <TableCell>{m.country}</TableCell>
                <TableCell>{m.assetType}</TableCell>
                <TableCell className="text-right">{typeof m.price === "number" ? m.price.toFixed(2) : m.price}</TableCell>
                <TableCell>{m.rbicsSector}</TableCell>
                <TableCell>{m.rbicsIndustry}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
      {filtered.length > 500 && (
        <p className="text-xs text-muted-foreground">
          Showing first 500 rows. Use CSV download for full data.
        </p>
      )}
    </div>
  );
}

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

"use client";

import * as React from "react";
import type { PriceRowData } from "@/actions/price-database-actions";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { DownloadIcon, FileSpreadsheetIcon } from "lucide-react";

function formatCellTime(iso: string | null): string {
  if (!iso) return "";
  try {
    return new Date(iso).toISOString();
  } catch {
    return "";
  }
}

function escapeCsv(s: string): string {
  if (s.includes('"') || s.includes(",") || s.includes("\n") || s.includes("\r")) {
    return `"${s.replace(/"/g, '""')}"`;
  }
  return s;
}

function downloadCsv(data: PriceRowData[]) {
  const headers = ["ticker", "time", "open", "high", "low", "close", "volume"];
  const rows = data.map((r) =>
    [
      escapeCsv(r.ticker),
      escapeCsv(formatCellTime(r.time)),
      r.open,
      r.high,
      r.low,
      r.close,
      r.volume,
    ].join(",")
  );
  const csv = [headers.join(","), ...rows].join("\r\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `price-data-${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

async function downloadExcel(data: PriceRowData[]) {
  try {
    const XLSX = await import("xlsx");
    const wsData = [
      ["ticker", "time", "open", "high", "low", "close", "volume"],
      ...data.map((r) => [
        r.ticker,
        formatCellTime(r.time),
        r.open,
        r.high,
        r.low,
        r.close,
        r.volume,
      ]),
    ];
    const ws = XLSX.utils.aoa_to_sheet(wsData);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, "Price Data");
    XLSX.writeFile(
      wb,
      `price-data-${new Date().toISOString().slice(0, 10)}.xlsx`
    );
  } catch (e) {
    console.error("Excel export failed:", e);
    throw new Error("Excel export failed. Ensure the xlsx package is installed.");
  }
}

type ExportSectionProps = {
  data: PriceRowData[];
  tickerCount: number;
  dateRange: { min: string | null; max: string | null };
};

export function ExportSection({
  data,
  tickerCount,
  dateRange,
}: ExportSectionProps) {
  const [excelLoading, setExcelLoading] = React.useState(false);
  const [excelError, setExcelError] = React.useState<string | null>(null);

  const handleCsv = () => {
    downloadCsv(data);
  };

  const handleExcel = async () => {
    setExcelError(null);
    setExcelLoading(true);
    try {
      await downloadExcel(data);
    } catch (err) {
      setExcelError(err instanceof Error ? err.message : "Export failed");
    } finally {
      setExcelLoading(false);
    }
  };

  const minStr = dateRange.min
    ? new Date(dateRange.min).toLocaleDateString()
    : "—";
  const maxStr = dateRange.max
    ? new Date(dateRange.max).toLocaleDateString()
    : "—";

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">Export</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <p className="text-muted-foreground text-sm">
          {data.length.toLocaleString()} rows · {tickerCount} ticker(s) ·{" "}
          {minStr} – {maxStr}
        </p>
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" size="sm" onClick={handleCsv}>
            <DownloadIcon className="mr-2 size-4" />
            Download CSV
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleExcel}
            disabled={excelLoading}
          >
            <FileSpreadsheetIcon className="mr-2 size-4" />
            {excelLoading ? "Exporting…" : "Download Excel"}
          </Button>
        </div>
        {excelError && (
          <p className="text-destructive text-sm">{excelError}</p>
        )}
      </CardContent>
    </Card>
  );
}

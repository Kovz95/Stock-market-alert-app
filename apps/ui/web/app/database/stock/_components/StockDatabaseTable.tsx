"use client";

import * as React from "react";
import {
  flexRender,
  getCoreRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
  type ColumnDef,
  type SortingState,
} from "@tanstack/react-table";
import type { FullStockMetadataRow } from "@/actions/stock-database-actions";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import {
  ChevronLeftIcon,
  ChevronRightIcon,
  ChevronsLeftIcon,
  ChevronsRightIcon,
  DownloadIcon,
} from "lucide-react";

function escapeCsv(s: string | number | undefined): string {
  if (s === undefined || s === null) return "";
  const t = String(s);
  if (t.includes('"') || t.includes(",") || t.includes("\n") || t.includes("\r")) {
    return `"${t.replace(/"/g, '""')}"`;
  }
  return t;
}

function downloadCsv(rows: FullStockMetadataRow[], columns: { key: keyof FullStockMetadataRow; label: string }[]) {
  const headers = columns.map((c) => c.label);
  const rowsCsv = rows.map((r) =>
    columns.map((c) => escapeCsv((r[c.key] as string | number | undefined) ?? "")).join(",")
  );
  const csv = [headers.join(","), ...rowsCsv].join("\r\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `stock-database-${new Date().toISOString().slice(0, 10)}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

const baseColumns: ColumnDef<FullStockMetadataRow>[] = [
  { accessorKey: "symbol", header: "Symbol", cell: ({ row }) => <span className="font-medium">{row.original.symbol}</span> },
  { accessorKey: "name", header: "Name" },
  { accessorKey: "assetType", header: "Type" },
  { accessorKey: "exchange", header: "Exchange" },
  { accessorKey: "country", header: "Country" },
  { accessorKey: "isin", header: "ISIN" },
];

const rbicsColumns: ColumnDef<FullStockMetadataRow>[] = [
  { accessorKey: "rbicsEconomy", header: "Economy" },
  { accessorKey: "rbicsSector", header: "Sector" },
  { accessorKey: "rbicsSubsector", header: "Subsector" },
  { accessorKey: "rbicsIndustryGroup", header: "Industry group" },
  { accessorKey: "rbicsIndustry", header: "Industry" },
  { accessorKey: "rbicsSubindustry", header: "Subindustry" },
];

const etfColumns: ColumnDef<FullStockMetadataRow>[] = [
  { accessorKey: "etfIssuer", header: "ETF issuer" },
  { accessorKey: "etfAssetClass", header: "Asset class" },
  { accessorKey: "etfFocus", header: "Focus" },
  { accessorKey: "etfNiche", header: "Niche" },
  {
    accessorKey: "expenseRatio",
    header: () => <div className="text-right">Expense ratio</div>,
    cell: ({ row }) => (
      <div className="text-right tabular-nums">
        {row.original.expenseRatio != null ? `${Number(row.original.expenseRatio).toFixed(3)}` : "—"}
      </div>
    ),
  },
  {
    accessorKey: "aum",
    header: () => <div className="text-right">AUM</div>,
    cell: ({ row }) => (
      <div className="text-right tabular-nums">
        {row.original.aum != null ? `$${Number(row.original.aum).toLocaleString(undefined, { maximumFractionDigits: 0 })}` : "—"}
      </div>
    ),
  },
];

type StockDatabaseTableProps = {
  data: FullStockMetadataRow[];
  assetTypeFilter: "All" | "Stocks" | "ETFs";
  searchTerm: string;
  onSearchChange: (v: string) => void;
};

export function StockDatabaseTable({
  data,
  assetTypeFilter,
  searchTerm,
  onSearchChange,
}: StockDatabaseTableProps) {
  const [sorting, setSorting] = React.useState<SortingState>([{ id: "symbol", desc: false }]);
  const [pagination, setPagination] = React.useState({ pageIndex: 0, pageSize: 50 });

  const columns = React.useMemo<ColumnDef<FullStockMetadataRow>[]>(() => {
    const cols = [...baseColumns];
    if (assetTypeFilter === "All" || assetTypeFilter === "Stocks") cols.push(...rbicsColumns);
    if (assetTypeFilter === "All" || assetTypeFilter === "ETFs") cols.push(...etfColumns);
    return cols;
  }, [assetTypeFilter]);

  const table = useReactTable({
    data,
    columns,
    state: { sorting, pagination },
    onSortingChange: setSorting,
    onPaginationChange: setPagination,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
  });

  const csvColumns = React.useMemo(() => {
    const list: { key: keyof FullStockMetadataRow; label: string }[] = [
      { key: "symbol", label: "Symbol" },
      { key: "name", label: "Name" },
      { key: "assetType", label: "Type" },
      { key: "exchange", label: "Exchange" },
      { key: "country", label: "Country" },
      { key: "isin", label: "ISIN" },
    ];
    if (assetTypeFilter === "All" || assetTypeFilter === "Stocks") {
      list.push(
        { key: "rbicsEconomy", label: "Economy" },
        { key: "rbicsSector", label: "Sector" },
        { key: "rbicsSubsector", label: "Subsector" },
        { key: "rbicsIndustryGroup", label: "Industry group" },
        { key: "rbicsIndustry", label: "Industry" },
        { key: "rbicsSubindustry", label: "Subindustry" }
      );
    }
    if (assetTypeFilter === "All" || assetTypeFilter === "ETFs") {
      list.push(
        { key: "etfIssuer", label: "ETF issuer" },
        { key: "etfAssetClass", label: "Asset class" },
        { key: "etfFocus", label: "Focus" },
        { key: "etfNiche", label: "Niche" },
        { key: "expenseRatio", label: "Expense ratio" },
        { key: "aum", label: "AUM" }
      );
    }
    return list;
  }, [assetTypeFilter]);

  if (data.length === 0) {
    return (
      <div className="rounded-lg border bg-muted/20 py-12 text-center text-muted-foreground">
        No symbols match the selected filters. Try adjusting your criteria.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex-1 min-w-[200px]">
          <Input
            placeholder="Search symbols…"
            value={searchTerm}
            onChange={(e) => onSearchChange(e.target.value)}
            className="max-w-sm"
          />
        </div>
        <Button variant="outline" size="sm" onClick={() => downloadCsv(data, csvColumns)}>
          <DownloadIcon className="mr-2 size-4" />
          Download CSV
        </Button>
      </div>
      <p className="text-muted-foreground text-sm">
        {data.length.toLocaleString()} symbol(s)
      </p>

      <div className="overflow-auto rounded-lg border max-h-[70vh]">
        <Table>
          <TableHeader className="bg-muted/50 sticky top-0 z-10">
            {table.getHeaderGroups().map((hg) => (
              <TableRow key={hg.id}>
                {hg.headers.map((h) => (
                  <TableHead key={h.id}>
                    {flexRender(h.column.columnDef.header, h.getContext())}
                  </TableHead>
                ))}
              </TableRow>
            ))}
          </TableHeader>
          <TableBody>
            {table.getRowModel().rows.map((row) => (
              <TableRow key={row.id}>
                {row.getVisibleCells().map((cell) => (
                  <TableCell key={cell.id} className="py-1 text-sm">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <Label className="text-sm whitespace-nowrap">Rows per page</Label>
          <Select
            value={String(table.getState().pagination.pageSize)}
            onValueChange={(v) => table.setPageSize(Number(v))}
          >
            <SelectTrigger className="w-20">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {[25, 50, 100, 200, 500].map((n) => (
                <SelectItem key={n} value={String(n)}>{n}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <div className="flex items-center gap-2 text-sm">
          <span>
            Page {table.getState().pagination.pageIndex + 1} of {table.getPageCount()}
          </span>
          <Button variant="outline" size="icon" className="size-8" onClick={() => table.setPageIndex(0)} disabled={!table.getCanPreviousPage()}>
            <ChevronsLeftIcon className="size-4" />
          </Button>
          <Button variant="outline" size="icon" className="size-8" onClick={() => table.previousPage()} disabled={!table.getCanPreviousPage()}>
            <ChevronLeftIcon className="size-4" />
          </Button>
          <Button variant="outline" size="icon" className="size-8" onClick={() => table.nextPage()} disabled={!table.getCanNextPage()}>
            <ChevronRightIcon className="size-4" />
          </Button>
          <Button variant="outline" size="icon" className="size-8" onClick={() => table.setPageIndex(table.getPageCount() - 1)} disabled={!table.getCanNextPage()}>
            <ChevronsRightIcon className="size-4" />
          </Button>
        </div>
      </div>
    </div>
  );
}

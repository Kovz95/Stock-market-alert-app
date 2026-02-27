"use client";

import * as React from "react";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  StockDatabaseFilters,
  defaultStockDatabaseFilters,
  applyStockDatabaseFilters,
  type StockDatabaseFiltersState,
} from "@/app/database/stock/_components";
import type { FullStockMetadataRow } from "@/actions/stock-database-actions";

type ScannerFiltersProps = {
  metadata: FullStockMetadataRow[];
  filters: StockDatabaseFiltersState;
  onFiltersChange: (f: StockDatabaseFiltersState) => void;
  portfolioId: string;
  onPortfolioIdChange: (id: string) => void;
  portfolioOptions: { portfolioId: string; name: string; tickers: string[] }[];
};

export function ScannerFilters({
  metadata,
  filters,
  onFiltersChange,
  portfolioId,
  onPortfolioIdChange,
  portfolioOptions,
}: ScannerFiltersProps) {
  const filteredCount = React.useMemo(() => {
    const after = applyStockDatabaseFilters(metadata, filters);
    if (portfolioId === "All") return after.length;
    const p = portfolioOptions.find((o) => o.portfolioId === portfolioId);
    if (!p) return after.length;
    const set = new Set(p.tickers);
    return after.filter((r) => set.has(r.symbol)).length;
  }, [metadata, filters, portfolioId, portfolioOptions]);

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label className="text-xs">Portfolio</Label>
        <Select value={portfolioId} onValueChange={onPortfolioIdChange}>
          <SelectTrigger className="h-8 text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="All">All</SelectItem>
            {portfolioOptions.map((p) => (
              <SelectItem key={p.portfolioId} value={p.portfolioId}>
                {p.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <p className="text-xs text-muted-foreground">
        Symbols matching filters: {filteredCount.toLocaleString()}
      </p>
      <StockDatabaseFilters
        data={metadata}
        filters={filters}
        onFiltersChange={onFiltersChange}
      />
    </div>
  );
}

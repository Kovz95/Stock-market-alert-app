"use client";

import * as React from "react";
import { useAtomValue, useSetAtom } from "jotai";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { StockDatabaseFilters } from "@/app/database/stock/_components";
import type { FullStockMetadataRow } from "@/actions/stock-database-actions";
import { scannerFiltersAtom, scannerPortfolioIdAtom } from "@/lib/store/scanner";

export type PortfolioOption = {
  portfolioId: string;
  name: string;
  tickers: string[];
};

type ScannerFiltersProps = {
  metadata: FullStockMetadataRow[];
  portfolioOptions: PortfolioOption[];
  filteredCount: number;
};

function ScannerFiltersComponent({
  metadata,
  portfolioOptions,
  filteredCount,
}: ScannerFiltersProps) {
  const filters = useAtomValue(scannerFiltersAtom);
  const setFilters = useSetAtom(scannerFiltersAtom);
  const portfolioId = useAtomValue(scannerPortfolioIdAtom);
  const setPortfolioId = useSetAtom(scannerPortfolioIdAtom);

  return (
    <div className="space-y-4">
      <div className="space-y-2">
        <Label className="text-xs">Portfolio</Label>
        <Select value={portfolioId} onValueChange={setPortfolioId}>
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
        onFiltersChange={setFilters}
      />
    </div>
  );
}

export const ScannerFilters = React.memo(ScannerFiltersComponent);

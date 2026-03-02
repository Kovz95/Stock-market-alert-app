"use client";

import * as React from "react";
import type { Portfolio } from "../../../../../../gen/ts/alert/v1/alert";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Checkbox } from "@/components/ui/checkbox";
import { Button } from "@/components/ui/button";
import { useRemoveStocks } from "@/lib/hooks/usePortfolios";
import { toast } from "sonner";

type PortfolioHoldingsProps = {
  portfolio: Portfolio;
};

export function PortfolioHoldings({ portfolio }: PortfolioHoldingsProps) {
  const [selected, setSelected] = React.useState<Set<string>>(new Set());
  const removeMutation = useRemoveStocks();

  // Reset selection when portfolio changes
  React.useEffect(() => {
    setSelected(new Set());
  }, [portfolio.portfolioId]);

  const toggleTicker = (ticker: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(ticker)) {
        next.delete(ticker);
      } else {
        next.add(ticker);
      }
      return next;
    });
  };

  const toggleAll = () => {
    if (selected.size === portfolio.tickers.length) {
      setSelected(new Set());
    } else {
      setSelected(new Set(portfolio.tickers));
    }
  };

  const handleRemove = async () => {
    if (selected.size === 0) return;
    try {
      await removeMutation.mutateAsync({
        portfolioId: portfolio.portfolioId,
        tickers: Array.from(selected),
      });
      toast.success(`Removed ${selected.size} stock(s)`);
      setSelected(new Set());
    } catch (err) {
      toast.error(`Failed to remove stocks: ${err}`);
    }
  };

  const handleExport = () => {
    const data = {
      name: portfolio.name,
      stocks: portfolio.tickers,
      exported_date: new Date().toISOString(),
    };
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `portfolio_${portfolio.name.replace(/\s+/g, "_")}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (portfolio.tickers.length === 0) {
    return (
      <div className="rounded-lg border border-dashed p-8 text-center">
        <p className="text-muted-foreground text-sm">
          This portfolio is empty. Add stocks to get started.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold text-sm">
          Holdings ({portfolio.tickers.length} stocks)
        </h3>
        <div className="flex items-center gap-2">
          {selected.size > 0 && (
            <Button
              size="sm"
              variant="destructive"
              onClick={handleRemove}
              disabled={removeMutation.isPending}
            >
              {removeMutation.isPending
                ? "Removing..."
                : `Remove ${selected.size} selected`}
            </Button>
          )}
          <Button size="sm" variant="outline" onClick={handleExport}>
            Export JSON
          </Button>
        </div>
      </div>
      <div className="rounded-lg border max-h-[400px] overflow-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead className="w-10">
                <Checkbox
                  checked={
                    selected.size === portfolio.tickers.length &&
                    portfolio.tickers.length > 0
                  }
                  onCheckedChange={toggleAll}
                  aria-label="Select all"
                />
              </TableHead>
              <TableHead>Symbol</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {portfolio.tickers.map((ticker) => (
              <TableRow key={ticker}>
                <TableCell>
                  <Checkbox
                    checked={selected.has(ticker)}
                    onCheckedChange={() => toggleTicker(ticker)}
                    aria-label={`Select ${ticker}`}
                  />
                </TableCell>
                <TableCell className="font-mono font-medium">{ticker}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}

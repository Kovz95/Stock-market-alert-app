"use client";

import * as React from "react";
import type { Portfolio } from "../../../../../../gen/ts/alert/v1/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useStockSearch } from "@/lib/hooks/useAlertHistory";
import { useAddStocks } from "@/lib/hooks/usePortfolios";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { XIcon } from "lucide-react";

type AddStocksSectionProps = {
  portfolio: Portfolio;
};

export function AddStocksSection({ portfolio }: AddStocksSectionProps) {
  const [searchQuery, setSearchQuery] = React.useState("");
  const [selectedTickers, setSelectedTickers] = React.useState<string[]>([]);
  const [quickAdd, setQuickAdd] = React.useState("");

  const { data: searchResults } = useStockSearch(searchQuery);
  const addMutation = useAddStocks();

  const existingSet = React.useMemo(
    () => new Set(portfolio.tickers),
    [portfolio.tickers]
  );

  const filteredResults = React.useMemo(() => {
    if (!searchResults) return [];
    return searchResults.filter(
      (r) => !existingSet.has(r.ticker) && !selectedTickers.includes(r.ticker)
    );
  }, [searchResults, existingSet, selectedTickers]);

  const addTicker = (ticker: string) => {
    if (!selectedTickers.includes(ticker) && !existingSet.has(ticker)) {
      setSelectedTickers((prev) => [...prev, ticker]);
    }
    setSearchQuery("");
  };

  const removeTicker = (ticker: string) => {
    setSelectedTickers((prev) => prev.filter((t) => t !== ticker));
  };

  const handleAddToPortfolio = async () => {
    if (selectedTickers.length === 0) return;
    try {
      await addMutation.mutateAsync({
        portfolioId: portfolio.portfolioId,
        tickers: selectedTickers,
      });
      toast.success(`Added ${selectedTickers.length} stock(s)`);
      setSelectedTickers([]);
    } catch (err) {
      toast.error(`Failed to add stocks: ${err}`);
    }
  };

  const handleQuickAdd = async () => {
    if (!quickAdd.trim()) return;
    const tickers = quickAdd
      .split(",")
      .map((t) => t.trim().toUpperCase())
      .filter((t) => t && !existingSet.has(t));

    if (tickers.length === 0) {
      toast.warning("All symbols are already in this portfolio or invalid");
      return;
    }

    try {
      await addMutation.mutateAsync({
        portfolioId: portfolio.portfolioId,
        tickers,
      });
      toast.success(`Added ${tickers.length} stock(s)`);
      setQuickAdd("");
    } catch (err) {
      toast.error(`Failed to add stocks: ${err}`);
    }
  };

  return (
    <div className="space-y-4">
      <h3 className="font-semibold text-sm">Add Stocks</h3>

      {/* Search-based add */}
      <div className="space-y-2">
        <Label className="text-xs">Search stocks</Label>
        <Input
          placeholder="Search by ticker or name..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="h-8"
        />
        {filteredResults.length > 0 && searchQuery.length >= 2 && (
          <div className="rounded-md border max-h-48 overflow-auto">
            {filteredResults.slice(0, 20).map((result) => (
              <button
                key={result.ticker}
                className="flex w-full items-center gap-2 px-3 py-1.5 text-xs hover:bg-accent/50 text-left"
                onClick={() => addTicker(result.ticker)}
              >
                <span className="font-mono font-medium">{result.ticker}</span>
                <span className="text-muted-foreground truncate">
                  {result.name}
                </span>
                <span className="ml-auto text-muted-foreground">
                  {result.exchange}
                </span>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Selected chips */}
      {selectedTickers.length > 0 && (
        <div className="space-y-2">
          <div className="flex flex-wrap gap-1">
            {selectedTickers.map((ticker) => (
              <Badge key={ticker} variant="secondary" className="gap-1 pr-1">
                {ticker}
                <button
                  onClick={() => removeTicker(ticker)}
                  className="ml-0.5 rounded-full hover:bg-foreground/10 p-0.5"
                  aria-label={`Remove ${ticker}`}
                >
                  <XIcon className="size-3" />
                </button>
              </Badge>
            ))}
          </div>
          <Button
            size="sm"
            onClick={handleAddToPortfolio}
            disabled={addMutation.isPending}
          >
            {addMutation.isPending
              ? "Adding..."
              : `Add ${selectedTickers.length} stock(s)`}
          </Button>
        </div>
      )}

      {/* Quick add */}
      <div className="space-y-2 border-t pt-4">
        <Label className="text-xs">Quick add (comma-separated)</Label>
        <div className="flex gap-2">
          <Input
            placeholder="AAPL, MSFT, GOOGL"
            value={quickAdd}
            onChange={(e) => setQuickAdd(e.target.value)}
            className="h-8"
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                handleQuickAdd();
              }
            }}
          />
          <Button
            size="sm"
            variant="secondary"
            onClick={handleQuickAdd}
            disabled={!quickAdd.trim() || addMutation.isPending}
          >
            Add
          </Button>
        </div>
      </div>
    </div>
  );
}

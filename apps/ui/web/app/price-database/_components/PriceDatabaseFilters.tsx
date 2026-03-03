"use client";

import * as React from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { DayFilter, Timeframe } from "@/lib/price-database-enums";
import type { LoadPriceDataParams, StockMetadataItem } from "@/actions/price-database-actions";
import { Loader2Icon } from "lucide-react";

const TIMEFRAME_OPTIONS: { value: number; label: string }[] = [
  { value: Timeframe.HOURLY, label: "Hourly" },
  { value: Timeframe.DAILY, label: "Daily" },
  { value: Timeframe.WEEKLY, label: "Weekly" },
];

const DAY_FILTER_OPTIONS: { value: number; label: string }[] = [
  { value: DayFilter.ALL, label: "All days" },
  { value: DayFilter.WEEKDAYS, label: "Weekdays only" },
  { value: DayFilter.WEEKENDS, label: "Weekends only" },
];

const MAX_ROWS_OPTIONS = [100, 500, 1000, 5000, 10000, 50000];

export type PriceFiltersState = {
  timeframe: number;
  exchanges: string[];
  searchType: "ticker" | "name";
  searchInput: string;
  selectedTickers: string[];
  maxRows: number;
  dayFilter: number;
  startDate: string;
  endDate: string;
};

const defaultFilters = (
  statsMin?: string | null,
  statsMax?: string | null
): PriceFiltersState => ({
  timeframe: Timeframe.DAILY,
  exchanges: [],
  searchType: "ticker",
  searchInput: "",
  selectedTickers: [],
  maxRows: 5000,
  dayFilter: DayFilter.ALL,
  startDate: statsMin ? statsMin.slice(0, 10) : "",
  endDate: statsMax ? statsMax.slice(0, 10) : "",
});

export function getLoadParams(
  filters: PriceFiltersState,
  statsDailyMin?: string | null,
  statsDailyMax?: string | null,
  metadata?: StockMetadataItem[]
): LoadPriceDataParams {
  const startDate = filters.startDate
    ? new Date(filters.startDate)
    : (statsDailyMin ? new Date(statsDailyMin) : undefined);
  const endDate = filters.endDate
    ? new Date(filters.endDate)
    : (statsDailyMax ? new Date(statsDailyMax) : undefined);

  let tickers: string[];
  if (filters.selectedTickers.length > 0) {
    tickers = filters.selectedTickers;
  } else if (filters.exchanges.length > 0 && metadata?.length) {
    tickers = metadata
      .filter((m) => filters.exchanges.includes(m.exchange))
      .map((m) => m.symbol);
  } else {
    tickers = [];
  }

  return {
    timeframe: filters.timeframe,
    tickers,
    startDate: startDate ?? undefined,
    endDate: endDate ?? undefined,
    maxRows: filters.maxRows,
    dayFilter: filters.dayFilter,
  };
}

export { defaultFilters };

type PriceDatabaseFiltersProps = {
  filters: PriceFiltersState;
  onFiltersChange: (f: PriceFiltersState) => void;
  metadata: StockMetadataItem[] | undefined;
  metadataLoading: boolean;
  statsDailyMin: string | null;
  statsDailyMax: string | null;
  onLoad: () => void;
  loadPending: boolean;
};

export function PriceDatabaseFilters({
  filters,
  onFiltersChange,
  metadata,
  metadataLoading,
  statsDailyMin,
  statsDailyMax,
  onLoad,
  loadPending,
}: PriceDatabaseFiltersProps) {
  const uniqueExchanges = React.useMemo(() => {
    if (!metadata) return [];
    const set = new Set(metadata.map((m) => m.exchange).filter(Boolean));
    return Array.from(set).sort();
  }, [metadata]);

  const filteredTickerOptions = React.useMemo(() => {
    if (!metadata) return [];
    let list = metadata;
    if (filters.exchanges.length > 0) {
      list = list.filter((m) => filters.exchanges.includes(m.exchange));
    }
    const q = filters.searchInput.trim().toLowerCase();
    if (q) {
      if (filters.searchType === "ticker") {
        list = list.filter((m) => m.symbol.toLowerCase().includes(q));
      } else {
        list = list.filter((m) => m.name.toLowerCase().includes(q));
      }
    }
    return list.slice(0, 200).map((m) => ({ symbol: m.symbol, name: m.name }));
  }, [metadata, filters.exchanges, filters.searchInput, filters.searchType]);

  const toggleExchange = (ex: string) => {
    const next = filters.exchanges.includes(ex)
      ? filters.exchanges.filter((e) => e !== ex)
      : [...filters.exchanges, ex];
    onFiltersChange({ ...filters, exchanges: next });
  };

  const toggleTicker = (symbol: string) => {
    const next = filters.selectedTickers.includes(symbol)
      ? filters.selectedTickers.filter((t) => t !== symbol)
      : [...filters.selectedTickers, symbol];
    onFiltersChange({ ...filters, selectedTickers: next });
  };

  const syncDatesFromStats = () => {
    onFiltersChange({
      ...filters,
      startDate: statsDailyMin ? statsDailyMin.slice(0, 10) : "",
      endDate: statsDailyMax ? statsDailyMax.slice(0, 10) : "",
    });
  };

  return (
    <div className="space-y-4 rounded-lg border bg-card p-4 text-sm">
      <h3 className="font-semibold">Filters</h3>

      <div className="space-y-2">
        <Label>Timeframe</Label>
        <Select
          value={String(filters.timeframe)}
          onValueChange={(v) =>
            onFiltersChange({
              ...filters,
              timeframe: Number(v),
            })
          }
        >
          <SelectTrigger className="w-full">
            <SelectValue placeholder="Timeframe" />
          </SelectTrigger>
          <SelectContent>
            <SelectGroup>
              {TIMEFRAME_OPTIONS.map((opt) => (
                <SelectItem key={opt.value} value={String(opt.value)}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectGroup>
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <Label>Exchanges</Label>
        <div className="max-h-32 overflow-y-auto rounded border bg-muted/30 p-2 space-y-1">
          {metadataLoading ? (
            <p className="text-muted-foreground text-xs">Loading…</p>
          ) : uniqueExchanges.length === 0 ? (
            <p className="text-muted-foreground text-xs">No exchanges</p>
          ) : (
            uniqueExchanges.map((ex) => (
              <label
                key={ex}
                className="flex items-center gap-2 cursor-pointer hover:bg-muted/50 rounded px-1 py-0.5"
              >
                <input
                  type="checkbox"
                  checked={filters.exchanges.includes(ex)}
                  onChange={() => toggleExchange(ex)}
                  className="rounded border-input"
                />
                <span className="text-xs">{ex}</span>
              </label>
            ))
          )}
        </div>
      </div>

      <div className="space-y-2">
        <Label>Search by</Label>
        <div className="flex gap-2">
          <Select
            value={filters.searchType}
            onValueChange={(v: "ticker" | "name") =>
              onFiltersChange({ ...filters, searchType: v })
            }
          >
            <SelectTrigger className="w-28">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="ticker">Ticker</SelectItem>
              <SelectItem value="name">Name</SelectItem>
            </SelectContent>
          </Select>
          <Input
            placeholder={filters.searchType === "ticker" ? "e.g. AAPL" : "Company name"}
            value={filters.searchInput}
            onChange={(e) =>
              onFiltersChange({ ...filters, searchInput: e.target.value })
            }
            className="flex-1"
          />
        </div>
      </div>

      <div className="space-y-2">
        <Label>Tickers (optional, leave empty for all)</Label>
        <div className="max-h-40 overflow-y-auto rounded border bg-muted/30 p-2 space-y-1">
          {filteredTickerOptions.length === 0 ? (
            <p className="text-muted-foreground text-xs">
              {filters.searchInput.trim() ? "No matches" : "Search or filter by exchange"}
            </p>
          ) : (
            filteredTickerOptions.map(({ symbol, name }) => (
              <label
                key={symbol}
                className="flex items-center gap-2 cursor-pointer hover:bg-muted/50 rounded px-1 py-0.5"
              >
                <input
                  type="checkbox"
                  checked={filters.selectedTickers.includes(symbol)}
                  onChange={() => toggleTicker(symbol)}
                  className="rounded border-input"
                />
                <span className="text-xs font-medium">{symbol}</span>
                {name && (
                  <span className="text-muted-foreground text-xs truncate max-w-[120px]" title={name}>
                    {name}
                  </span>
                )}
              </label>
            ))
          )}
        </div>
      </div>

      <div className="space-y-2">
        <Label>Max rows</Label>
        <Select
          value={String(filters.maxRows)}
          onValueChange={(v) =>
            onFiltersChange({ ...filters, maxRows: Number(v) })
          }
        >
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectGroup>
              {MAX_ROWS_OPTIONS.map((n) => (
                <SelectItem key={n} value={String(n)}>
                  {n.toLocaleString()}
                </SelectItem>
              ))}
            </SelectGroup>
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <Label>Day filter</Label>
        <Select
          value={String(filters.dayFilter)}
          onValueChange={(v) =>
            onFiltersChange({ ...filters, dayFilter: Number(v) })
          }
        >
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectGroup>
              {DAY_FILTER_OPTIONS.map((opt) => (
                <SelectItem key={opt.value} value={String(opt.value)}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectGroup>
          </SelectContent>
        </Select>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div className="space-y-1">
          <Label>Start date</Label>
          <Input
            type="date"
            value={filters.startDate}
            onChange={(e) =>
              onFiltersChange({ ...filters, startDate: e.target.value })
            }
          />
        </div>
        <div className="space-y-1">
          <Label>End date</Label>
          <Input
            type="date"
            value={filters.endDate}
            onChange={(e) =>
              onFiltersChange({ ...filters, endDate: e.target.value })
            }
          />
        </div>
      </div>
      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={syncDatesFromStats}
      >
        Use DB range
      </Button>

      <Button
        className="w-full"
        onClick={onLoad}
        disabled={loadPending}
      >
        {loadPending ? (
          <>
            <Loader2Icon className="mr-2 size-4 animate-spin" />
            Loading…
          </>
        ) : (
          "Load Data"
        )}
      </Button>
    </div>
  );
}

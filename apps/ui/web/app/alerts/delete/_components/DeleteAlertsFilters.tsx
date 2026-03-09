"use client";

import * as React from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { FullStockMetadataRow } from "@/actions/stock-database-actions";
import type { AssetTypeFilter } from "@/app/database/stock/_components/types";
import type { DeleteAlertsFiltersState } from "./types";
import {
  CONDITION_TYPE_OPTIONS,
  TIMEFRAME_OPTIONS,
  TRIGGERED_FILTER_OPTIONS,
} from "./types";
import { Trash2Icon } from "lucide-react";
import { defaultDeleteAlertsFilters } from "./types";

type DeleteAlertsFiltersProps = {
  stockData: FullStockMetadataRow[];
  filters: DeleteAlertsFiltersState;
  onFiltersChange: (f: DeleteAlertsFiltersState) => void;
};

function uniqueSorted(arr: (string | undefined)[]): string[] {
  const set = new Set(arr.filter((x): x is string => !!x));
  return Array.from(set).sort();
}

export function DeleteAlertsFilters({
  stockData,
  filters,
  onFiltersChange,
}: DeleteAlertsFiltersProps) {
  const stocksOnly = React.useMemo(
    () => stockData.filter((r) => r.assetType === "Stock" || (r.rbicsEconomy && !r.etfIssuer)),
    [stockData]
  );
  const etfsOnly = React.useMemo(
    () => stockData.filter((r) => r.assetType === "ETF" || !!r.etfIssuer),
    [stockData]
  );

  const countries = React.useMemo(() => uniqueSorted(stockData.map((r) => r.country)), [stockData]);
  const exchanges = React.useMemo(() => {
    let list = stockData.map((r) => r.exchange);
    if (filters.countries.length > 0) {
      list = stockData.filter((r) => r.country && filters.countries.includes(r.country)).map((r) => r.exchange);
    }
    return uniqueSorted(list);
  }, [stockData, filters.countries]);

  const economies = React.useMemo(() => uniqueSorted(stocksOnly.map((r) => r.rbicsEconomy)), [stocksOnly]);
  const sectors = React.useMemo(() => {
    let list = stocksOnly.map((r) => r.rbicsSector);
    if (filters.economies.length > 0) {
      list = stocksOnly.filter((r) => r.rbicsEconomy && filters.economies.includes(r.rbicsEconomy)).map((r) => r.rbicsSector);
    }
    return uniqueSorted(list);
  }, [stocksOnly, filters.economies]);
  const subsectors = React.useMemo(() => {
    let list = stocksOnly.map((r) => r.rbicsSubsector);
    if (filters.sectors.length > 0) {
      list = stocksOnly.filter((r) => r.rbicsSector && filters.sectors.includes(r.rbicsSector)).map((r) => r.rbicsSubsector);
    }
    return uniqueSorted(list);
  }, [stocksOnly, filters.sectors]);
  const industryGroups = React.useMemo(() => {
    let list = stocksOnly.map((r) => r.rbicsIndustryGroup);
    if (filters.subsectors.length > 0) {
      list = stocksOnly.filter((r) => r.rbicsSubsector && filters.subsectors.includes(r.rbicsSubsector)).map((r) => r.rbicsIndustryGroup);
    }
    return uniqueSorted(list);
  }, [stocksOnly, filters.subsectors]);
  const industries = React.useMemo(() => {
    let list = stocksOnly.map((r) => r.rbicsIndustry);
    if (filters.industryGroups.length > 0) {
      list = stocksOnly.filter((r) => r.rbicsIndustryGroup && filters.industryGroups.includes(r.rbicsIndustryGroup)).map((r) => r.rbicsIndustry);
    }
    return uniqueSorted(list);
  }, [stocksOnly, filters.industryGroups]);
  const subindustries = React.useMemo(() => {
    let list = stocksOnly.map((r) => r.rbicsSubindustry);
    if (filters.industries.length > 0) {
      list = stocksOnly.filter((r) => r.rbicsIndustry && filters.industries.includes(r.rbicsIndustry)).map((r) => r.rbicsSubindustry);
    }
    return uniqueSorted(list);
  }, [stocksOnly, filters.industries]);

  const issuers = React.useMemo(() => uniqueSorted(etfsOnly.map((r) => r.etfIssuer)), [etfsOnly]);
  const assetClasses = React.useMemo(() => {
    let list = etfsOnly.map((r) => r.etfAssetClass);
    if (filters.issuers.length > 0) {
      list = etfsOnly.filter((r) => r.etfIssuer && filters.issuers.includes(r.etfIssuer)).map((r) => r.etfAssetClass);
    }
    return uniqueSorted(list);
  }, [etfsOnly, filters.issuers]);
  const focuses = React.useMemo(() => {
    let list = etfsOnly.map((r) => r.etfFocus);
    if (filters.assetClasses.length > 0) {
      list = etfsOnly.filter((r) => r.etfAssetClass && filters.assetClasses.includes(r.etfAssetClass)).map((r) => r.etfFocus);
    } else if (filters.issuers.length > 0) {
      list = etfsOnly.filter((r) => r.etfIssuer && filters.issuers.includes(r.etfIssuer)).map((r) => r.etfFocus);
    }
    return uniqueSorted(list);
  }, [etfsOnly, filters.assetClasses, filters.issuers]);
  const niches = React.useMemo(() => {
    let list = etfsOnly.map((r) => r.etfNiche);
    if (filters.focuses.length > 0) {
      list = etfsOnly.filter((r) => r.etfFocus && filters.focuses.includes(r.etfFocus)).map((r) => r.etfNiche);
    } else if (filters.assetClasses.length > 0) {
      list = etfsOnly.filter((r) => r.etfAssetClass && filters.assetClasses.includes(r.etfAssetClass)).map((r) => r.etfNiche);
    } else if (filters.issuers.length > 0) {
      list = etfsOnly.filter((r) => r.etfIssuer && filters.issuers.includes(r.etfIssuer)).map((r) => r.etfNiche);
    }
    return uniqueSorted(list);
  }, [etfsOnly, filters.focuses, filters.assetClasses, filters.issuers]);

  const update = (patch: Partial<DeleteAlertsFiltersState>) => {
    onFiltersChange({ ...filters, ...patch });
  };

  const clearAll = () => {
    onFiltersChange({ ...defaultDeleteAlertsFilters });
  };

  const multi = (
    label: string,
    options: string[],
    selected: string[],
    onChange: (v: string[]) => void
  ) => (
    <div className="space-y-1">
      <Label className="text-xs">{label}</Label>
      <Select
        value={selected.length ? selected[0] : "__all__"}
        onValueChange={(v) => {
          if (v === "__all__") onChange([]);
          else onChange([v]);
        }}
      >
        <SelectTrigger className="h-8 w-full text-xs">
          <SelectValue placeholder={`All ${label}`} />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="__all__">All</SelectItem>
          {options.slice(0, 200).map((o) => (
            <SelectItem key={o} value={o}>
              {o}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );

  return (
    <div className="space-y-4 rounded-lg border bg-card p-4 text-sm">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold">Filters</h3>
        <Button variant="ghost" size="sm" onClick={clearAll}>
          <Trash2Icon className="size-4" /> Clear
        </Button>
      </div>

      {/* Alert-specific filters */}
      <div className="space-y-1">
        <Label className="text-xs">Search ticker / company</Label>
        <Input
          className="h-8 text-xs"
          placeholder="e.g. AAPL or Apple"
          value={filters.searchText}
          onChange={(e) => update({ searchText: e.target.value })}
        />
      </div>

      <div className="space-y-1">
        <Label className="text-xs">Alert name</Label>
        <Input
          className="h-8 text-xs"
          placeholder="Search by alert name"
          value={filters.nameSearch}
          onChange={(e) => update({ nameSearch: e.target.value })}
        />
      </div>

      <div className="space-y-1">
        <Label className="text-xs">Condition type</Label>
        <Select
          value={filters.conditionType}
          onValueChange={(v) => update({ conditionType: v })}
        >
          <SelectTrigger className="h-8 w-full text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {CONDITION_TYPE_OPTIONS.map((o) => (
              <SelectItem key={o} value={o}>
                {o === "All" ? "All conditions" : o}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-1">
        <Label className="text-xs">Custom condition search</Label>
        <Input
          className="h-8 text-xs"
          placeholder="e.g. HARSI, RSI"
          value={filters.conditionSearch}
          onChange={(e) => update({ conditionSearch: e.target.value })}
        />
      </div>

      <div className="space-y-1">
        <Label className="text-xs">Timeframe</Label>
        <Select
          value={filters.timeframes.length ? filters.timeframes[0] : "__all__"}
          onValueChange={(v) =>
            update({ timeframes: v === "__all__" ? [] : [v] })
          }
        >
          <SelectTrigger className="h-8 w-full text-xs">
            <SelectValue placeholder="All timeframes" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="__all__">All timeframes</SelectItem>
            {TIMEFRAME_OPTIONS.map((o) => (
              <SelectItem key={o} value={o}>
                {o}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-1">
        <Label className="text-xs">Last triggered</Label>
        <Select
          value={filters.triggeredFilter}
          onValueChange={(v) => update({ triggeredFilter: v })}
        >
          <SelectTrigger className="h-8 w-full text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {TRIGGERED_FILTER_OPTIONS.map((o) => (
              <SelectItem key={o} value={o}>
                {o === "All" ? "All" : o === "Never" ? "Never triggered" : `Triggered ${o.toLowerCase()}`}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Market data filters */}
      <div className="border-t pt-3 mt-3">
        <Label className="text-xs font-medium text-muted-foreground">Market data filters</Label>
      </div>

      <div className="space-y-2">
        <Label className="text-xs">Country</Label>
        <Select
          value={filters.countries.length ? filters.countries[0] : "__all__"}
          onValueChange={(v) =>
            update({ countries: v === "__all__" ? [] : [v] })
          }
        >
          <SelectTrigger className="h-8 w-full text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="__all__">All countries</SelectItem>
            {countries.slice(0, 150).map((c) => (
              <SelectItem key={c} value={c}>{c}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <Label className="text-xs">Exchange</Label>
        <Select
          value={filters.exchanges.length ? filters.exchanges[0] : "__all__"}
          onValueChange={(v) =>
            update({ exchanges: v === "__all__" ? [] : [v] })
          }
        >
          <SelectTrigger className="h-8 w-full text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="__all__">All exchanges</SelectItem>
            {exchanges.slice(0, 150).map((e) => (
              <SelectItem key={e} value={e}>{e}</SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <div className="space-y-2">
        <Label className="text-xs">Asset type</Label>
        <Select
          value={filters.assetType}
          onValueChange={(v) => update({ assetType: v as AssetTypeFilter })}
        >
          <SelectTrigger className="h-8 w-full text-xs">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="All">All</SelectItem>
            <SelectItem value="Stocks">Stocks</SelectItem>
            <SelectItem value="ETFs">ETFs</SelectItem>
          </SelectContent>
        </Select>
      </div>

      {(filters.assetType === "All" || filters.assetType === "Stocks") && (
        <>
          <div className="border-t pt-2 mt-2">
            <Label className="text-xs font-medium text-muted-foreground">Stock (RBICS)</Label>
          </div>
          {multi("Economy", economies, filters.economies, (v) => update({ economies: v }))}
          {multi("Sector", sectors, filters.sectors, (v) => update({ sectors: v }))}
          {multi("Subsector", subsectors, filters.subsectors, (v) => update({ subsectors: v }))}
          {multi("Industry group", industryGroups, filters.industryGroups, (v) => update({ industryGroups: v }))}
          {multi("Industry", industries, filters.industries, (v) => update({ industries: v }))}
          {multi("Subindustry", subindustries, filters.subindustries, (v) => update({ subindustries: v }))}
        </>
      )}

      {(filters.assetType === "All" || filters.assetType === "ETFs") && (
        <>
          <div className="border-t pt-2 mt-2">
            <Label className="text-xs font-medium text-muted-foreground">ETF</Label>
          </div>
          {multi("Issuer", issuers, filters.issuers, (v) => update({ issuers: v }))}
          {multi("Asset class", assetClasses, filters.assetClasses, (v) => update({ assetClasses: v }))}
          {multi("Focus", focuses, filters.focuses, (v) => update({ focuses: v }))}
          {multi("Niche", niches, filters.niches, (v) => update({ niches: v }))}
        </>
      )}
    </div>
  );
}

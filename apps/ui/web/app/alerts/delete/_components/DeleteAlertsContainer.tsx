"use client";

import * as React from "react";
import { useAtomValue, useSetAtom, useStore } from "jotai";
import { toast } from "sonner";
import { useSearchAlerts, useBulkDeleteAlerts } from "@/lib/hooks/useAlerts";
import { useFullStockMetadata } from "@/lib/hooks/useStockDatabase";
import { searchAlertsStream } from "@/actions/alert-actions";
import type { AlertData, SearchAlertsFilters } from "@/actions/alert-actions";
import type { FullStockMetadataRow } from "@/actions/stock-database-actions";
import { applyStockDatabaseFilters } from "@/app/database/stock/_components/types";
import { DeleteAlertsFilters } from "./DeleteAlertsFilters";
import { DeleteAlertsTable } from "./DeleteAlertsTable";
import { DeleteAlertsActionBar } from "./DeleteAlertsActionBar";
import { DeleteConfirmDialog } from "./DeleteConfirmDialog";
import type { DeleteAlertsFiltersState } from "./types";
import { Button } from "@/components/ui/button";
import { ChevronLeftIcon, ChevronRightIcon } from "lucide-react";
import {
  deleteAlertsFiltersAtom,
  deleteAlertsSelectedAtom,
  deleteAlertsPageAtom,
  deleteAlertsConfirmOpenAtom,
  deleteAlertsIsDeletingAtom,
  deleteAlertsIsSelectingAllAtom,
  deleteAlertsProgressAtom,
} from "@/lib/store/delete-alerts";

const PAGE_SIZE = 100;

/**
 * Check if any market-data (client-side) filters are active.
 * These require joining alert tickers to stock metadata, so they must be
 * applied client-side. Country and exchange are NOT included here — they
 * are sent as server-side SQL filters on the alerts table directly.
 */
function hasMarketDataFilters(filters: DeleteAlertsFiltersState): boolean {
  return (
    filters.assetType !== "All" ||
    filters.economies.length > 0 ||
    filters.sectors.length > 0 ||
    filters.subsectors.length > 0 ||
    filters.industryGroups.length > 0 ||
    filters.industries.length > 0 ||
    filters.subindustries.length > 0 ||
    filters.issuers.length > 0 ||
    filters.assetClasses.length > 0 ||
    filters.focuses.length > 0 ||
    filters.niches.length > 0
  );
}

/** Build the server-side filter object from the full filter state. */
function toServerFilters(filters: DeleteAlertsFiltersState): SearchAlertsFilters {
  const conditionSearch =
    filters.conditionType !== "All"
      ? filters.conditionType
      : filters.conditionSearch.trim() || undefined;

  const search = filters.searchText.trim() || filters.nameSearch.trim() || undefined;

  return {
    search,
    exchanges: filters.exchanges.length > 0 ? filters.exchanges : undefined,
    timeframes: filters.timeframes.length > 0 ? filters.timeframes : undefined,
    countries: filters.countries.length > 0 ? filters.countries : undefined,
    triggeredFilter:
      filters.triggeredFilter !== "All" ? filters.triggeredFilter : undefined,
    conditionSearch,
  };
}

function extractConditionText(conditions: unknown): string {
  if (!Array.isArray(conditions)) return "";
  const parts: string[] = [];
  for (const c of conditions) {
    if (c && typeof c === "object" && "conditions" in c) {
      parts.push(String((c as { conditions: string }).conditions));
    }
  }
  return parts.join(" | ");
}

/**
 * Apply client-side post-filters to server results.
 * Handles: market-data filters (RBICS/ETF), and residual text filters
 * that couldn't be fully handled server-side (e.g. nameSearch when
 * searchText is also set, or conditionSearch when conditionType is set).
 */
function applyClientFilters(
  alerts: AlertData[],
  filters: DeleteAlertsFiltersState,
  metadataMap: Map<string, FullStockMetadataRow>
): AlertData[] {
  let result = alerts;

  const nameQ = filters.nameSearch.trim().toLowerCase();
  const searchQ = filters.searchText.trim();
  if (nameQ && searchQ) {
    result = result.filter((a) => a.name.toLowerCase().includes(nameQ));
  }

  const condQ = filters.conditionSearch.trim().toLowerCase();
  if (filters.conditionType !== "All" && condQ) {
    result = result.filter((a) => {
      const text = extractConditionText(a.conditions).toLowerCase();
      return text.includes(condQ);
    });
  }

  if (hasMarketDataFilters(filters) && metadataMap.size > 0) {
    const allMetadata = Array.from(metadataMap.values());
    const filteredMetadata = applyStockDatabaseFilters(allMetadata, filters);
    const allowedSymbols = new Set(
      filteredMetadata.map((r) => r.symbol.toUpperCase())
    );

    result = result.filter((a) => {
      const ticker = a.ticker.toUpperCase();
      if (allowedSymbols.has(ticker)) return true;
      const base = ticker.split("-")[0];
      if (base !== ticker && allowedSymbols.has(base)) return true;
      if (allowedSymbols.has(`${base}-US`)) return true;
      return false;
    });
  }

  return result;
}

export function DeleteAlertsContainer() {
  const store = useStore();
  const bulkDelete = useBulkDeleteAlerts();

  // Stock data needed for client-side market-data filtering
  const { data: stockData } = useFullStockMetadata();

  const filters = useAtomValue(deleteAlertsFiltersAtom);
  const page = useAtomValue(deleteAlertsPageAtom);
  const setPage = useSetAtom(deleteAlertsPageAtom);
  const setSelected = useSetAtom(deleteAlertsSelectedAtom);
  const setConfirmOpen = useSetAtom(deleteAlertsConfirmOpenAtom);
  const setIsDeleting = useSetAtom(deleteAlertsIsDeletingAtom);
  const setIsSelectingAll = useSetAtom(deleteAlertsIsSelectingAllAtom);
  const setProgress = useSetAtom(deleteAlertsProgressAtom);

  // Debounce server filters to avoid spamming on every keystroke
  const [debouncedFilters, setDebouncedFilters] =
    React.useState<DeleteAlertsFiltersState>(filters);

  React.useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedFilters(filters);
      setPage(1);
    }, 300);
    return () => clearTimeout(timer);
  }, [filters, setPage]);

  const serverFilters = React.useMemo(
    () => toServerFilters(debouncedFilters),
    [debouncedFilters]
  );

  // When market-data filters are active we need to fetch a larger page
  // from the server so client-side filtering has enough data.
  const hasClientFilters = hasMarketDataFilters(debouncedFilters);
  const serverPageSize = hasClientFilters ? 100 : PAGE_SIZE;
  const serverPage = hasClientFilters ? 1 : page;

  const {
    data: searchResult,
    isLoading: alertsLoading,
    error: alertsError,
    isFetching,
  } = useSearchAlerts(serverFilters, serverPage, serverPageSize);

  // Build ticker → metadata map
  const metadataMap = React.useMemo(() => {
    const map = new Map<string, FullStockMetadataRow>();
    if (!stockData) return map;
    for (const row of stockData) {
      if (row.symbol) {
        map.set(row.symbol.toUpperCase(), row);
      }
    }
    return map;
  }, [stockData]);

  // Apply client-side market-data filters
  const filteredAlerts = React.useMemo(() => {
    if (!searchResult?.alerts) return [];
    return applyClientFilters(searchResult.alerts, debouncedFilters, metadataMap);
  }, [searchResult, debouncedFilters, metadataMap]);

  // Pagination
  const serverTotalCount = searchResult?.totalCount ?? 0;
  const displayAlerts = hasClientFilters
    ? filteredAlerts.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE)
    : filteredAlerts;
  const totalFiltered = hasClientFilters ? filteredAlerts.length : serverTotalCount;
  const totalPages = Math.max(1, Math.ceil(totalFiltered / PAGE_SIZE));
  const safePage = Math.min(page, totalPages);

  /** Select every alert that matches the current filters (one stream when server-paginated). */
  const selectAllFiltered = React.useCallback(async () => {
    if (totalFiltered <= filteredAlerts.length) {
      setSelected(new Set(filteredAlerts.map((a) => a.alertId)));
      return;
    }
    setIsSelectingAll(true);
    try {
      const alerts = await searchAlertsStream(serverFilters);
      setSelected(new Set(alerts.map((a) => a.alertId)));
    } finally {
      setIsSelectingAll(false);
    }
  }, [serverFilters, totalFiltered, filteredAlerts, setSelected, setIsSelectingAll]);

  const handleDelete = () => {
    const ids = Array.from(store.get(deleteAlertsSelectedAtom));
    setIsDeleting(true);
    setProgress({ completed: 0, total: ids.length });
    bulkDelete.mutate(ids, {
      onSuccess: (deletedCount) => {
        setIsDeleting(false);
        setProgress(null);
        setConfirmOpen(false);
        setSelected(new Set());
        toast.success(`Deleted ${deletedCount} alert${deletedCount !== 1 ? "s" : ""}`);
      },
      onError: (err) => {
        setIsDeleting(false);
        setProgress(null);
        toast.error(err instanceof Error ? err.message : "Failed to delete alerts.");
      },
    });
  };

  if (alertsLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <p className="text-muted-foreground">Loading alerts...</p>
      </div>
    );
  }

  if (alertsError) {
    return (
      <div className="flex items-center justify-center py-12">
        <p className="text-destructive">
          Failed to load alerts: {alertsError.message}
        </p>
      </div>
    );
  }

  return (
    <div className="flex gap-6">
      {/* Sidebar filters */}
      <div className="w-64 shrink-0">
        <DeleteAlertsFilters />
      </div>

      {/* Main content */}
      <div className="flex-1 space-y-4 min-w-0">
        <p className="text-sm text-muted-foreground">
          {totalFiltered} alert{totalFiltered !== 1 ? "s" : ""} found
          {isFetching ? " (loading...)" : ""}
        </p>

        <DeleteAlertsActionBar
          totalFiltered={totalFiltered}
          pageCount={displayAlerts.length}
          onSelectAllFiltered={selectAllFiltered}
        />

        <DeleteAlertsTable alerts={displayAlerts} />

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between">
            <p className="text-sm text-muted-foreground">
              Page {safePage} of {totalPages}
            </p>
            <div className="flex items-center gap-2">
              <Button
                variant="outline"
                size="sm"
                disabled={safePage <= 1}
                onClick={() => setPage((p) => p - 1)}
              >
                <ChevronLeftIcon className="size-4" />
              </Button>
              <Button
                variant="outline"
                size="sm"
                disabled={safePage >= totalPages}
                onClick={() => setPage((p) => p + 1)}
              >
                <ChevronRightIcon className="size-4" />
              </Button>
            </div>
          </div>
        )}

        <DeleteConfirmDialog onConfirm={handleDelete} />
      </div>
    </div>
  );
}

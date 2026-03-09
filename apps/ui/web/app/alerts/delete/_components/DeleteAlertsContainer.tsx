"use client";

import * as React from "react";
import { useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { useAllAlerts, ALERTS_KEY } from "@/lib/hooks/useAlerts";
import { useFullStockMetadata } from "@/lib/hooks/useStockDatabase";
import { deleteAlert } from "@/actions/alert-actions";
import type { AlertData } from "@/actions/alert-actions";
import type { FullStockMetadataRow } from "@/actions/stock-database-actions";
import { applyStockDatabaseFilters } from "@/app/database/stock/_components/types";
import { DeleteAlertsFilters } from "./DeleteAlertsFilters";
import { DeleteAlertsTable } from "./DeleteAlertsTable";
import { DeleteAlertsActionBar } from "./DeleteAlertsActionBar";
import { DeleteConfirmDialog } from "./DeleteConfirmDialog";
import type { DeleteAlertsFiltersState } from "./types";
import { defaultDeleteAlertsFilters } from "./types";
import { Button } from "@/components/ui/button";
import { ChevronLeftIcon, ChevronRightIcon } from "lucide-react";

const PAGE_SIZE = 20;

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

function matchesTriggeredFilter(
  lastTriggered: string | null,
  filter: string
): boolean {
  if (filter === "All") return true;
  if (filter === "Never") return !lastTriggered;
  if (!lastTriggered) return false;

  const date = new Date(lastTriggered);
  const now = new Date();

  switch (filter) {
    case "Today":
      return date.toDateString() === now.toDateString();
    case "This Week": {
      const weekAgo = new Date(now);
      weekAgo.setDate(weekAgo.getDate() - 7);
      return date >= weekAgo;
    }
    case "This Month": {
      return (
        date.getMonth() === now.getMonth() &&
        date.getFullYear() === now.getFullYear()
      );
    }
    case "This Year":
      return date.getFullYear() === now.getFullYear();
    default:
      return true;
  }
}

function applyAlertFilters(
  alerts: AlertData[],
  filters: DeleteAlertsFiltersState,
  metadataMap: Map<string, FullStockMetadataRow>
): AlertData[] {
  let result = alerts;

  // Text search (ticker / company name)
  const searchQ = filters.searchText.trim().toLowerCase();
  if (searchQ) {
    result = result.filter(
      (a) =>
        a.ticker.toLowerCase().includes(searchQ) ||
        a.stockName.toLowerCase().includes(searchQ) ||
        a.name.toLowerCase().includes(searchQ)
    );
  }

  // Alert name search
  const nameQ = filters.nameSearch.trim().toLowerCase();
  if (nameQ) {
    result = result.filter((a) => a.name.toLowerCase().includes(nameQ));
  }

  // Condition type filter
  if (filters.conditionType !== "All") {
    const term = filters.conditionType.toLowerCase();
    result = result.filter((a) => {
      const text = extractConditionText(a.conditions).toLowerCase();
      return text.includes(term);
    });
  }

  // Custom condition search
  const condQ = filters.conditionSearch.trim().toLowerCase();
  if (condQ) {
    result = result.filter((a) => {
      const text = extractConditionText(a.conditions).toLowerCase();
      return text.includes(condQ);
    });
  }

  // Timeframe filter
  if (filters.timeframes.length > 0) {
    result = result.filter((a) => filters.timeframes.includes(a.timeframe));
  }

  // Last triggered filter
  if (filters.triggeredFilter !== "All") {
    result = result.filter((a) =>
      matchesTriggeredFilter(a.lastTriggered, filters.triggeredFilter)
    );
  }

  // Market data filters — check if any are active
  const hasMarketFilters =
    filters.countries.length > 0 ||
    filters.exchanges.length > 0 ||
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
    filters.niches.length > 0;

  if (hasMarketFilters && metadataMap.size > 0) {
    // Build set of tickers that pass the stock database filters
    const allMetadata = Array.from(metadataMap.values());
    const filteredMetadata = applyStockDatabaseFilters(allMetadata, filters);
    const allowedSymbols = new Set(
      filteredMetadata.map((r) => r.symbol.toUpperCase())
    );

    result = result.filter((a) => {
      const ticker = a.ticker.toUpperCase();
      // Try exact, then base (before dash)
      if (allowedSymbols.has(ticker)) return true;
      const base = ticker.split("-")[0];
      if (base !== ticker && allowedSymbols.has(base)) return true;
      // Also try with common suffixes
      if (allowedSymbols.has(`${base}-US`)) return true;
      return false;
    });
  }

  return result;
}

export function DeleteAlertsContainer() {
  const queryClient = useQueryClient();
  const { data: alerts, isLoading: alertsLoading, error: alertsError } = useAllAlerts();
  const { data: stockData, isLoading: stockLoading } = useFullStockMetadata();

  const [filters, setFilters] = React.useState<DeleteAlertsFiltersState>(
    defaultDeleteAlertsFilters
  );
  const [selected, setSelected] = React.useState<Set<string>>(new Set());
  const [page, setPage] = React.useState(1);
  const [confirmOpen, setConfirmOpen] = React.useState(false);
  const [isDeleting, setIsDeleting] = React.useState(false);

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

  // Apply filters
  const filteredAlerts = React.useMemo(() => {
    if (!alerts) return [];
    return applyAlertFilters(alerts, filters, metadataMap);
  }, [alerts, filters, metadataMap]);

  // Paginate
  const totalPages = Math.max(1, Math.ceil(filteredAlerts.length / PAGE_SIZE));
  const safePage = Math.min(page, totalPages);
  const pagedAlerts = React.useMemo(() => {
    const start = (safePage - 1) * PAGE_SIZE;
    return filteredAlerts.slice(start, start + PAGE_SIZE);
  }, [filteredAlerts, safePage]);

  // Reset page when filters change
  React.useEffect(() => {
    setPage(1);
  }, [filters]);

  // Selection handlers
  const toggleOne = (alertId: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(alertId)) next.delete(alertId);
      else next.add(alertId);
      return next;
    });
  };

  const toggleAllOnPage = (checked: boolean) => {
    setSelected((prev) => {
      const next = new Set(prev);
      for (const a of pagedAlerts) {
        if (checked) next.add(a.alertId);
        else next.delete(a.alertId);
      }
      return next;
    });
  };

  const selectAllFiltered = () => {
    setSelected(new Set(filteredAlerts.map((a) => a.alertId)));
  };

  const clearSelection = () => {
    setSelected(new Set());
  };

  // Delete handler
  const handleDelete = async () => {
    setIsDeleting(true);
    let succeeded = 0;
    let failed = 0;

    const ids = Array.from(selected);
    for (const id of ids) {
      try {
        await deleteAlert(id);
        succeeded++;
      } catch {
        failed++;
      }
    }

    setIsDeleting(false);
    setConfirmOpen(false);
    setSelected(new Set());
    queryClient.invalidateQueries({ queryKey: ALERTS_KEY });

    if (failed === 0) {
      toast.success(`Deleted ${succeeded} alert${succeeded !== 1 ? "s" : ""}`);
    } else {
      toast.error(
        `Deleted ${succeeded}, failed ${failed} alert${failed !== 1 ? "s" : ""}`
      );
    }
  };

  if (alertsLoading || stockLoading) {
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

  if (!alerts || alerts.length === 0) {
    return (
      <div className="flex items-center justify-center py-12">
        <p className="text-muted-foreground">No alerts found.</p>
      </div>
    );
  }

  return (
    <div className="flex gap-6">
      {/* Sidebar filters */}
      <div className="w-64 shrink-0">
        <DeleteAlertsFilters
          stockData={stockData ?? []}
          filters={filters}
          onFiltersChange={setFilters}
        />
      </div>

      {/* Main content */}
      <div className="flex-1 space-y-4 min-w-0">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Delete Alerts</h1>
            <p className="text-sm text-muted-foreground mt-1">
              {filteredAlerts.length} of {alerts.length} alerts shown
            </p>
          </div>
        </div>

        <DeleteAlertsActionBar
          selectedCount={selected.size}
          totalFiltered={filteredAlerts.length}
          pageCount={pagedAlerts.length}
          onSelectAllFiltered={selectAllFiltered}
          onClearSelection={clearSelection}
          onDelete={() => setConfirmOpen(true)}
        />

        <DeleteAlertsTable
          alerts={pagedAlerts}
          selected={selected}
          onToggle={toggleOne}
          onToggleAll={toggleAllOnPage}
        />

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

        <DeleteConfirmDialog
          open={confirmOpen}
          onOpenChange={setConfirmOpen}
          count={selected.size}
          isDeleting={isDeleting}
          onConfirm={handleDelete}
        />
      </div>
    </div>
  );
}

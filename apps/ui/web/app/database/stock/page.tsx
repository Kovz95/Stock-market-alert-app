"use client";

import * as React from "react";
import { useAtomValue, useSetAtom } from "jotai";
import { useFullStockMetadata } from "@/lib/hooks/useStockDatabase";
import {
  StockDatabaseFilters,
  StockDatabaseStatsCards,
  StockDatabaseTable,
  RbicsBreakdownSection,
  TopExchangesSection,
  applyStockDatabaseFilters,
  searchRows,
} from "./_components";
import {
  stockDatabaseFiltersAtom,
  stockDatabaseSearchTermAtom,
  stockDatabaseSearchInputAtom,
} from "@/lib/store/stock-database";
import { Skeleton } from "@/components/ui/skeleton";
import type { FullStockMetadataRow } from "@/actions/stock-database-actions";

const SEARCH_DEBOUNCE_MS = 280;

/** Syncs searchInput atom to searchTerm atom with debounce so only the table section re-renders when searchTerm updates. */
function DebouncedSearchTermSync() {
  const searchInput = useAtomValue(stockDatabaseSearchInputAtom);
  const setSearchTerm = useSetAtom(stockDatabaseSearchTermAtom);

  React.useEffect(() => {
    const t = setTimeout(() => setSearchTerm(searchInput), SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(t);
  }, [searchInput, setSearchTerm]);

  return null;
}

/** Sidebar that subscribes to filters atom so only it re-renders on filter change. */
const StockDatabaseSidebar = React.memo(function StockDatabaseSidebar({
  metadata,
}: {
  metadata: FullStockMetadataRow[] | undefined;
}) {
  const filters = useAtomValue(stockDatabaseFiltersAtom);
  const setFilters = useSetAtom(stockDatabaseFiltersAtom);

  if (!metadata) {
    return <Skeleton className="h-64 w-full rounded-lg" />;
  }

  return (
    <StockDatabaseFilters
      data={metadata}
      filters={filters}
      onFiltersChange={setFilters}
    />
  );
});

/** Main content: subscribes to filters (deferred) and searchTerm so filtered list updates don't block filter UI. */
function StockDatabaseMain({
  metadata,
  isLoading,
}: {
  metadata: FullStockMetadataRow[] | null | undefined;
  isLoading: boolean;
}) {
  const filters = useAtomValue(stockDatabaseFiltersAtom);
  const searchTerm = useAtomValue(stockDatabaseSearchTermAtom);
  const deferredFilters = React.useDeferredValue(filters);

  const filtered = React.useMemo(() => {
    if (!metadata) return [];
    const afterFilters = applyStockDatabaseFilters(metadata, deferredFilters);
    return searchRows(afterFilters, searchTerm);
  }, [metadata, deferredFilters, searchTerm]);

  const stats = React.useMemo(() => {
    if (!metadata) return null;
    const stocks = metadata.filter((r) => r.assetType === "Stock" || (r.rbicsEconomy && !r.etfIssuer));
    const etfs = metadata.filter((r) => r.assetType === "ETF" || !!r.etfIssuer);
    const exchanges = new Set(metadata.map((r) => r.exchange).filter(Boolean));
    const countries = new Set(metadata.map((r) => r.country).filter(Boolean));
    return {
      totalSymbols: metadata.length,
      uniqueExchanges: exchanges.size,
      uniqueCountries: countries.size,
      stockCount: stocks.length,
      etfCount: etfs.length,
      stocks,
      etfs,
    };
  }, [metadata]);

  return (
    <main className="min-w-0 space-y-6">
      <section>
        <h2 className="text-lg font-semibold mb-3">Database overview</h2>
        {stats ? (
          <StockDatabaseStatsCards
            totalSymbols={stats.totalSymbols}
            uniqueExchanges={stats.uniqueExchanges}
            uniqueCountries={stats.uniqueCountries}
            stockCount={stats.stockCount}
            etfCount={stats.etfCount}
          />
        ) : isLoading ? (
          <Skeleton className="h-24 w-full rounded-lg" />
        ) : (
          <p className="text-muted-foreground text-sm">No data.</p>
        )}
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        {stats && stats.stocks.length > 0 && (
          <RbicsBreakdownSection stocks={stats.stocks} />
        )}
        {stats && metadata && (
          <TopExchangesSection data={metadata} title="Top exchanges by symbol count" />
        )}
      </section>

      <section>
        <h2 className="text-lg font-semibold mb-3">
          Symbol list ({filtered.length.toLocaleString()} symbols)
        </h2>
        {metadata ? (
          <StockDatabaseTable
            data={filtered}
            assetTypeFilter={filters.assetType}
          />
        ) : isLoading ? (
          <Skeleton className="h-96 w-full rounded-lg" />
        ) : null}
      </section>
    </main>
  );
}

export default function StockDatabasePage() {
  const { data: metadata, isLoading, error } = useFullStockMetadata();

  return (
    <div className="p-6 flex flex-col gap-6">
      <DebouncedSearchTermSync />

      <div>
        <h1 className="text-2xl font-bold">Stock Database</h1>
        <p className="text-muted-foreground">
          Complete database of symbols with industry classifications (auto-updated).
        </p>
      </div>

      {error && (
        <div className="rounded-lg border border-destructive/50 bg-destructive/10 px-4 py-3 text-destructive text-sm">
          <p className="font-medium">Failed to load stock data.</p>
          <p className="mt-1 opacity-90">{error.message}</p>
          <p className="mt-2 text-xs opacity-75">
            Ensure the price service is running (GRPC_ENDPOINT) and the stock_metadata table is populated.
          </p>
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-[280px_1fr]">
        <aside className="lg:min-w-0">
          <StockDatabaseSidebar metadata={metadata ?? undefined} />
        </aside>
        <StockDatabaseMain metadata={metadata} isLoading={isLoading} />
      </div>
    </div>
  );
}

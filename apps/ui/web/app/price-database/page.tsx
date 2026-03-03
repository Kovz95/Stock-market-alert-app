"use client";

import * as React from "react";
import {
  useDatabaseStats,
  useLoadPriceData,
  useUpdatePrices,
  useStockMetadata,
} from "@/lib/hooks/usePriceDatabase";
import { Timeframe } from "@/lib/price-database-enums";
import {
  PriceDatabaseFilters,
  DatabaseStatsCards,
  type TimeframeKind,
  PriceDataTable,
  ExportSection,
  PriceChartsSection,
  AnalysisSection,
  defaultFilters,
  getLoadParams,
} from "./_components";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";

export default function PriceDatabasePage() {
  const { data: metadata, isLoading: metadataLoading } = useStockMetadata();
  const { data: stats } = useDatabaseStats();
  const loadMutation = useLoadPriceData();
  const updateMutation = useUpdatePrices();

  const [filters, setFilters] = React.useState<ReturnType<typeof defaultFilters>>(
    () => defaultFilters(null, null)
  );
  const [loadedData, setLoadedData] = React.useState<
    import("@/actions/price-database-actions").PriceRowData[]
  >([]);

  // Sync date defaults when stats load
  React.useEffect(() => {
    if (!stats) return;
    setFilters((prev) => ({
      ...prev,
      startDate: prev.startDate || (stats.dailyMin?.slice(0, 10) ?? ""),
      endDate: prev.endDate || (stats.dailyMax?.slice(0, 10) ?? ""),
    }));
  }, [stats?.dailyMin, stats?.dailyMax]);

  const handleLoad = React.useCallback(() => {
    const params = getLoadParams(
      filters,
      stats?.dailyMin ?? null,
      stats?.dailyMax ?? null,
      metadata ?? undefined
    );
    loadMutation.mutate(params, {
      onSuccess: (rows) => {
        setLoadedData(rows);
        toast.success(`Loaded ${rows.length.toLocaleString()} rows`);
      },
      onError: (err) => {
        toast.error(err.message ?? "Failed to load price data");
      },
    });
  }, [filters, stats?.dailyMin, stats?.dailyMax, metadata, loadMutation]);

  const timeframeToEnum = (k: TimeframeKind): number => {
    switch (k) {
      case "hourly":
        return Timeframe.HOURLY;
      case "daily":
        return Timeframe.DAILY;
      case "weekly":
        return Timeframe.WEEKLY;
    }
  };

  const handleUpdate = React.useCallback(
    (timeframe: TimeframeKind) => {
      updateMutation.mutate(
        {
          timeframe: timeframeToEnum(timeframe),
          exchanges: filters.exchanges,
          tickers: filters.selectedTickers,
        },
        {
          onSuccess: () => {
            toast.success(`${timeframe} price update complete`);
          },
          onError: (err) => {
            toast.error(err.message ?? "Price update failed");
          },
        }
      );
    },
    [filters.exchanges, filters.selectedTickers, updateMutation]
  );

  const updatePending: Partial<Record<TimeframeKind, boolean>> = React.useMemo(
    () => ({
      hourly:
        updateMutation.isPending &&
        updateMutation.variables?.timeframe === Timeframe.HOURLY,
      daily:
        updateMutation.isPending &&
        updateMutation.variables?.timeframe === Timeframe.DAILY,
      weekly:
        updateMutation.isPending &&
        updateMutation.variables?.timeframe === Timeframe.WEEKLY,
    }),
    [
      updateMutation.isPending,
      updateMutation.variables?.timeframe,
    ]
  );

  const tickerCount = React.useMemo(() => {
    const set = new Set(loadedData.map((r) => r.ticker));
    return set.size;
  }, [loadedData]);

  const dateRange = React.useMemo(() => {
    if (loadedData.length === 0)
      return { min: null as string | null, max: null as string | null };
    const times = loadedData
      .map((r) => r.time)
      .filter((t): t is string => t != null);
    if (times.length === 0) return { min: null, max: null };
    return {
      min: times.reduce((a, b) => (a < b ? a : b)),
      max: times.reduce((a, b) => (a > b ? a : b)),
    };
  }, [loadedData]);

  return (
    <div className="p-6 flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold">Price Database</h1>
        <p className="text-muted-foreground">
          Query and export daily, hourly, and weekly price data from the
          database.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-[280px_1fr]">
        <aside className="lg:min-w-0">
          <PriceDatabaseFilters
            filters={filters}
            onFiltersChange={setFilters}
            metadata={metadata}
            metadataLoading={metadataLoading}
            statsDailyMin={stats?.dailyMin ?? null}
            statsDailyMax={stats?.dailyMax ?? null}
            onLoad={handleLoad}
            loadPending={loadMutation.isPending}
          />
        </aside>

        <main className="min-w-0 space-y-6">
          <section>
            <h2 className="text-lg font-semibold mb-3">Database statistics</h2>
            {stats ? (
              <DatabaseStatsCards
                stats={stats}
                exchanges={filters.exchanges}
                selectedTickers={filters.selectedTickers}
                onUpdate={handleUpdate}
                updatePending={updatePending}
              />
            ) : (
              <p className="text-muted-foreground text-sm">
                Loading database stats…
              </p>
            )}
          </section>

          {loadedData.length > 0 && (
            <Tabs defaultValue="table" className="w-full">
              <TabsList>
                <TabsTrigger value="table">Data table</TabsTrigger>
                <TabsTrigger value="charts">Charts</TabsTrigger>
                <TabsTrigger value="analysis">Analysis</TabsTrigger>
                <TabsTrigger value="export">Export</TabsTrigger>
              </TabsList>
              <TabsContent value="table" className="mt-4">
                <PriceDataTable data={loadedData} />
              </TabsContent>
              <TabsContent value="charts" className="mt-4">
                <PriceChartsSection data={loadedData} />
              </TabsContent>
              <TabsContent value="analysis" className="mt-4">
                <AnalysisSection data={loadedData} />
              </TabsContent>
              <TabsContent value="export" className="mt-4">
                <ExportSection
                  data={loadedData}
                  tickerCount={tickerCount}
                  dateRange={dateRange}
                />
              </TabsContent>
            </Tabs>
          )}

          {loadedData.length === 0 && (
            <p className="text-muted-foreground text-sm">
              Set filters and click <strong>Load Data</strong> to fetch price
              data.
            </p>
          )}
        </main>
      </div>
    </div>
  );
}

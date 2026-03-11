"use client";

import * as React from "react";
import { useAtomValue, useSetAtom } from "jotai";
import { useFullStockMetadata } from "@/lib/hooks/useStockDatabase";
import { usePortfolios } from "@/lib/hooks/useAlertHistory";
import { runScan } from "@/actions/scanner-actions";
import type { ScanMatch } from "../../../../../gen/ts/price/v1/price";
import {
  ScannerFilters,
  ScannerConditionSection,
  ScannerResults,
  scanMatchesToCsv,
} from "./_components";
import {
  applyStockDatabaseFilters,
} from "@/app/database/stock/_components";
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
import { Skeleton } from "@/components/ui/skeleton";
import { Timeframe } from "../../../../../gen/ts/price/v1/price";
import {
  scannerFiltersAtom,
  scannerPortfolioIdAtom,
  scannerConditionsAtom,
  scannerCombinationLogicAtom,
  scannerTimeframeAtom,
  scannerLookbackDaysAtom,
  scannerLookbackInputAtom,
  scannerResultsAtom,
  scannerScanningAtom,
  scannerScanErrorAtom,
  scannerScanProgressAtom,
  scannerPresetNameAtom,
  scannerSelectedPresetAtom,
  scannerPresetsAtom,
  type SavedPreset,
  type ScannerTimeframe,
} from "@/lib/store/scanner";

const MAX_TICKERS = 20000;
/** Chunk size for RunScan to avoid "request too large" (Envoy/gRPC limits). */
const SCAN_BATCH_SIZE = 200;

function chunk<T>(arr: T[], size: number): T[][] {
  const out: T[][] = [];
  for (let i = 0; i < arr.length; i += size) {
    out.push(arr.slice(i, i + size));
  }
  return out;
}

function timeframeToEnum(timeframe: ScannerTimeframe): Timeframe {
  switch (timeframe) {
    case "1h":
      return Timeframe.TIMEFRAME_HOURLY;
    case "1wk":
      return Timeframe.TIMEFRAME_WEEKLY;
    default:
      return Timeframe.TIMEFRAME_DAILY;
  }
}

export default function ScannerPage() {
  const { data: metadata, isLoading: metaLoading, error: metaError } = useFullStockMetadata();
  const { data: portfolios } = usePortfolios();

  const filters = useAtomValue(scannerFiltersAtom);
  const portfolioId = useAtomValue(scannerPortfolioIdAtom);
  const conditions = useAtomValue(scannerConditionsAtom);
  const combinationLogic = useAtomValue(scannerCombinationLogicAtom);
  const timeframe = useAtomValue(scannerTimeframeAtom);
  const lookbackDays = useAtomValue(scannerLookbackDaysAtom);
  const lookbackInput = useAtomValue(scannerLookbackInputAtom);
  const results = useAtomValue(scannerResultsAtom);
  const scanning = useAtomValue(scannerScanningAtom);
  const scanError = useAtomValue(scannerScanErrorAtom);
  const scanProgress = useAtomValue(scannerScanProgressAtom);
  const setFilters = useSetAtom(scannerFiltersAtom);
  const setPortfolioId = useSetAtom(scannerPortfolioIdAtom);
  const setConditions = useSetAtom(scannerConditionsAtom);
  const setCombinationLogic = useSetAtom(scannerCombinationLogicAtom);
  const setTimeframe = useSetAtom(scannerTimeframeAtom);
  const setLookbackDays = useSetAtom(scannerLookbackDaysAtom);
  const setLookbackInput = useSetAtom(scannerLookbackInputAtom);
  const setResults = useSetAtom(scannerResultsAtom);
  const setScanning = useSetAtom(scannerScanningAtom);
  const setScanError = useSetAtom(scannerScanErrorAtom);
  const setScanProgress = useSetAtom(scannerScanProgressAtom);

  const deferredFilters = React.useDeferredValue(filters);
  const deferredPortfolioId = React.useDeferredValue(portfolioId);

  const portfolioOptions = React.useMemo(() => {
    if (!portfolios) return [];
    return portfolios.map((p) => ({
      portfolioId: p.portfolioId,
      name: p.name ?? p.portfolioId,
      tickers: p.tickers ?? [],
    }));
  }, [portfolios]);

  const filteredCount = React.useMemo(() => {
    if (!metadata) return 0;
    let rows = applyStockDatabaseFilters(metadata, deferredFilters);
    if (deferredPortfolioId !== "All") {
      const p = portfolioOptions.find((o) => o.portfolioId === deferredPortfolioId);
      if (p) {
        const set = new Set(p.tickers);
        rows = rows.filter((r) => set.has(r.symbol));
      }
    }
    return Math.min(rows.length, MAX_TICKERS);
  }, [metadata, deferredFilters, deferredPortfolioId, portfolioOptions]);

  const filteredSymbolsForScan = React.useMemo(() => {
    if (!metadata) return [];
    let rows = applyStockDatabaseFilters(metadata, filters);
    if (portfolioId !== "All") {
      const p = portfolioOptions.find((o) => o.portfolioId === portfolioId);
      if (p) {
        const set = new Set(p.tickers);
        rows = rows.filter((r) => set.has(r.symbol));
      }
    }
    return rows.map((r) => r.symbol).slice(0, MAX_TICKERS);
  }, [metadata, filters, portfolioId, portfolioOptions]);

  const handleRunScan = React.useCallback(async () => {
    if (conditions.length === 0) {
      setScanError("Add at least one condition.");
      return;
    }
    setScanning(true);
    setScanError(null);
    setResults([]);
    const batches = chunk(filteredSymbolsForScan, SCAN_BATCH_SIZE);
    setScanProgress({ batch: 0, totalBatches: batches.length });
    const allMatches: ScanMatch[] = [];
    for (let i = 0; i < batches.length; i++) {
      setScanProgress({ batch: i + 1, totalBatches: batches.length });
      const res = await runScan({
        timeframe: timeframeToEnum(timeframe),
        conditions,
        combinationLogic: combinationLogic || "AND",
        tickers: batches[i],
        maxTickers: SCAN_BATCH_SIZE,
        lookbackDays,
      });
      if ("error" in res) {
        setScanError(res.error);
        setResults(allMatches.length > 0 ? allMatches : null);
        setScanning(false);
        setScanProgress(null);
        return;
      }
      allMatches.push(...(res.data ?? []));
      setResults([...allMatches]);
    }
    setScanning(false);
    setScanProgress(null);
    setResults(allMatches);
  }, [
    conditions,
    combinationLogic,
    timeframe,
    lookbackDays,
    filteredSymbolsForScan,
    setScanning,
    setScanError,
    setResults,
    setScanProgress,
  ]);

  const handleDownloadCsv = React.useCallback(() => {
    const currentResults = results;
    if (!currentResults || currentResults.length === 0) return;
    const csv = scanMatchesToCsv(currentResults);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `scan_results_${new Date().toISOString().slice(0, 19).replace(/[-:T]/g, "")}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [results]);

  if (metaLoading || !metadata) {
    return (
      <div className="p-6 space-y-6">
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  return (
    <div className="p-6 flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold">Market Scanner</h1>
        <p className="text-muted-foreground">
          Scan symbols that meet your technical conditions. Use filters to narrow the universe, then run the scan.
        </p>
      </div>

      {metaError && (
        <div className="rounded-lg border border-destructive/50 bg-destructive/10 px-4 py-3 text-destructive text-sm">
          <p className="font-medium">Failed to load stock data.</p>
          <p className="mt-1 opacity-90">{metaError.message}</p>
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-[280px_1fr]">
        <aside className="space-y-4">
          <ScannerFilters
            metadata={metadata}
            portfolioOptions={portfolioOptions}
            filteredCount={filteredCount}
          />
        </aside>

        <main className="space-y-6">
          <ScannerTimeframeAndLookback
            timeframe={timeframe}
            setTimeframe={setTimeframe}
            lookbackInput={lookbackInput}
            setLookbackInput={setLookbackInput}
            lookbackDays={lookbackDays}
            setLookbackDays={setLookbackDays}
          />

          <ScannerConditionSection />

          <div className="flex flex-wrap items-center gap-4">
            <Button
              onClick={handleRunScan}
              disabled={scanning || conditions.length === 0 || filteredSymbolsForScan.length === 0}
            >
              {scanning ? "Scanning…" : "Run Scan"}
            </Button>
            {scanning && scanProgress && (
              <span className="text-sm text-muted-foreground">
                Scanning batch {scanProgress.batch}/{scanProgress.totalBatches} ({filteredSymbolsForScan.length.toLocaleString()} symbols)…
              </span>
            )}
            {scanning && !scanProgress && (
              <span className="text-sm text-muted-foreground">
                Starting scan…
              </span>
            )}
          </div>

          {scanError && (
            <div className="rounded-lg border border-destructive/50 bg-destructive/10 px-4 py-3 text-destructive text-sm">
              {scanError}
            </div>
          )}

          {results !== null && (
            <ScannerResults
              matches={results}
              onDownloadCsv={handleDownloadCsv}
              scanning={scanning}
              scanProgress={scanProgress}
            />
          )}

          <ScannerPresetSection />
        </main>
      </div>
    </div>
  );
}

function ScannerTimeframeAndLookback({
  timeframe,
  setTimeframe,
  lookbackInput,
  setLookbackInput,
  lookbackDays,
  setLookbackDays,
}: {
  timeframe: ScannerTimeframe;
  setTimeframe: (v: ScannerTimeframe) => void;
  lookbackInput: string;
  setLookbackInput: (v: string) => void;
  lookbackDays: number;
  setLookbackDays: (v: number) => void;
}) {
  return (
    <div className="flex flex-wrap items-end gap-4">
      <div className="space-y-2">
        <Label className="text-xs">Timeframe</Label>
        <Select value={timeframe} onValueChange={(v) => setTimeframe(v as ScannerTimeframe)}>
          <SelectTrigger className="h-8 w-32">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="1h">Hourly</SelectItem>
            <SelectItem value="1d">Daily</SelectItem>
            <SelectItem value="1wk">Weekly</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div className="space-y-2">
        <Label className="text-xs">Lookback days</Label>
        <Input
          type="number"
          min={0}
          max={100}
          className="h-7 w-24"
          value={lookbackInput}
          onChange={(e) => {
            const raw = e.target.value;
            setLookbackInput(raw);
            const num = raw === "" ? 0 : Number(raw);
            if (!Number.isNaN(num)) {
              setLookbackDays(Math.max(0, Math.min(100, num)));
            }
          }}
          onBlur={() => setLookbackInput(String(lookbackDays))}
        />
      </div>
    </div>
  );
}

function ScannerPresetSection() {
  const presetName = useAtomValue(scannerPresetNameAtom);
  const setPresetName = useSetAtom(scannerPresetNameAtom);
  const presets = useAtomValue(scannerPresetsAtom);
  const setPresets = useSetAtom(scannerPresetsAtom);
  const selectedPreset = useAtomValue(scannerSelectedPresetAtom);
  const setSelectedPreset = useSetAtom(scannerSelectedPresetAtom);
  const conditions = useAtomValue(scannerConditionsAtom);
  const combinationLogic = useAtomValue(scannerCombinationLogicAtom);
  const timeframe = useAtomValue(scannerTimeframeAtom);
  const filters = useAtomValue(scannerFiltersAtom);
  const portfolioId = useAtomValue(scannerPortfolioIdAtom);
  const lookbackDays = useAtomValue(scannerLookbackDaysAtom);
  const setConditions = useSetAtom(scannerConditionsAtom);
  const setCombinationLogic = useSetAtom(scannerCombinationLogicAtom);
  const setTimeframe = useSetAtom(scannerTimeframeAtom);
  const setFilters = useSetAtom(scannerFiltersAtom);
  const setPortfolioId = useSetAtom(scannerPortfolioIdAtom);
  const setLookbackDays = useSetAtom(scannerLookbackDaysAtom);
  const setLookbackInput = useSetAtom(scannerLookbackInputAtom);

  const handleSavePreset = () => {
    if (!presetName.trim()) return;
    const preset: SavedPreset = {
      name: presetName.trim(),
      conditions: [...conditions],
      combinationLogic,
      timeframe,
      filters: { ...filters },
      portfolioId,
      lookbackDays,
      savedAt: new Date().toISOString(),
    };
    setPresets((prev) => {
      const next = prev.filter((p) => p.name !== preset.name);
      next.push(preset);
      return next;
    });
    setPresetName("");
  };

  const handleLoadPreset = () => {
    const p = presets.find((x) => x.name === selectedPreset);
    if (!p) return;
    setConditions(p.conditions);
    setCombinationLogic(p.combinationLogic);
    setTimeframe(p.timeframe as ScannerTimeframe);
    setFilters(p.filters);
    setPortfolioId(p.portfolioId);
    const days = p.lookbackDays ?? 0;
    setLookbackDays(days);
    setLookbackInput(String(days));
    setSelectedPreset("");
  };

  return (
    <div className="rounded-lg border bg-muted/20 p-4 space-y-4">
      <h3 className="font-semibold">Save / Load preset</h3>
      <div className="flex flex-wrap gap-2 items-end">
        <div className="space-y-1">
          <Label className="text-xs">Preset name</Label>
          <Input
            className="h-8 w-48 text-sm"
            placeholder="Name"
            value={presetName}
            onChange={(e) => setPresetName(e.target.value)}
          />
        </div>
        <Button size="sm" variant="secondary" onClick={handleSavePreset} disabled={!presetName.trim()}>
          Save
        </Button>
      </div>
      {presets.length > 0 && (
        <div className="flex flex-wrap gap-2 items-end">
          <Select value={selectedPreset} onValueChange={setSelectedPreset}>
            <SelectTrigger className="h-8 w-56">
              <SelectValue placeholder="Load preset…" />
            </SelectTrigger>
            <SelectContent>
              {presets.map((p) => (
                <SelectItem key={p.name} value={p.name}>
                  {p.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button size="sm" variant="outline" onClick={handleLoadPreset} disabled={!selectedPreset}>
            Load
          </Button>
        </div>
      )}
    </div>
  );
}

"use client";

import * as React from "react";
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
  defaultStockDatabaseFilters,
  applyStockDatabaseFilters,
  type StockDatabaseFiltersState,
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

const SCANNER_PRESETS_KEY = "scanner_presets";
const MAX_TICKERS = 20000;
/** Chunk size for RunScan to avoid "request too large" (Envoy/gRPC limits). */
const SCAN_BATCH_SIZE = 200;

type SavedPreset = {
  name: string;
  conditions: string[];
  combinationLogic: string;
  timeframe: string;
  filters: StockDatabaseFiltersState;
  portfolioId: string;
  savedAt: string;
};

function loadPresets(): SavedPreset[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(SCANNER_PRESETS_KEY);
    if (!raw) return [];
    return JSON.parse(raw);
  } catch {
    return [];
  }
}

function savePreset(preset: SavedPreset) {
  const list = loadPresets();
  const next = list.filter((p) => p.name !== preset.name);
  next.push(preset);
  localStorage.setItem(SCANNER_PRESETS_KEY, JSON.stringify(next));
}

function chunk<T>(arr: T[], size: number): T[][] {
  const out: T[][] = [];
  for (let i = 0; i < arr.length; i += size) {
    out.push(arr.slice(i, i + size));
  }
  return out;
}

export default function ScannerPage() {
  const { data: metadata, isLoading: metaLoading, error: metaError } = useFullStockMetadata();
  const { data: portfolios } = usePortfolios();

  const [filters, setFilters] = React.useState<StockDatabaseFiltersState>(defaultStockDatabaseFilters);
  const [portfolioId, setPortfolioId] = React.useState("All");
  const [conditions, setConditions] = React.useState<string[]>([]);
  const [combinationLogic, setCombinationLogic] = React.useState("AND");
  const [timeframe, setTimeframe] = React.useState<"1h" | "1d" | "1wk">("1d");
  const [results, setResults] = React.useState<ScanMatch[] | null>(null);
  const [scanning, setScanning] = React.useState(false);
  const [scanError, setScanError] = React.useState<string | null>(null);
  const [scanProgress, setScanProgress] = React.useState<{ batch: number; totalBatches: number } | null>(null);

  const [presetName, setPresetName] = React.useState("");
  const [presets, setPresets] = React.useState<SavedPreset[]>([]);
  const [selectedPreset, setSelectedPreset] = React.useState("");

  React.useEffect(() => {
    setPresets(loadPresets());
  }, []);

  const portfolioOptions = React.useMemo(() => {
    if (!portfolios) return [];
    return portfolios.map((p) => ({
      portfolioId: p.portfolioId,
      name: p.name ?? p.portfolioId,
      tickers: p.tickers ?? [],
    }));
  }, [portfolios]);

  const filteredSymbols = React.useMemo(() => {
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

  const timeframeEnum = (): Timeframe => {
    switch (timeframe) {
      case "1h":
        return Timeframe.TIMEFRAME_HOURLY;
      case "1wk":
        return Timeframe.TIMEFRAME_WEEKLY;
      default:
        return Timeframe.TIMEFRAME_DAILY;
    }
  };

  const handleRunScan = async () => {
    if (conditions.length === 0) {
      setScanError("Add at least one condition.");
      return;
    }
    setScanning(true);
    setScanError(null);
    setResults([]);
    const batches = chunk(filteredSymbols, SCAN_BATCH_SIZE);
    setScanProgress({ batch: 0, totalBatches: batches.length });
    const allMatches: ScanMatch[] = [];
    for (let i = 0; i < batches.length; i++) {
      setScanProgress({ batch: i + 1, totalBatches: batches.length });
      const res = await runScan({
        timeframe: timeframeEnum(),
        conditions,
        combinationLogic: combinationLogic || "AND",
        tickers: batches[i],
        maxTickers: SCAN_BATCH_SIZE,
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
  };

  const handleDownloadCsv = () => {
    if (!results || results.length === 0) return;
    const csv = scanMatchesToCsv(results);
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `scan_results_${new Date().toISOString().slice(0, 19).replace(/[-:T]/g, "")}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleSavePreset = () => {
    if (!presetName.trim()) return;
    savePreset({
      name: presetName.trim(),
      conditions: [...conditions],
      combinationLogic,
      timeframe,
      filters: { ...filters },
      portfolioId,
      savedAt: new Date().toISOString(),
    });
    setPresets(loadPresets());
    setPresetName("");
  };

  const handleLoadPreset = () => {
    const p = presets.find((x) => x.name === selectedPreset);
    if (!p) return;
    setConditions(p.conditions);
    setCombinationLogic(p.combinationLogic);
    setTimeframe(p.timeframe as "1h" | "1d" | "1wk");
    setFilters(p.filters);
    setPortfolioId(p.portfolioId);
    setSelectedPreset("");
  };

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
            filters={filters}
            onFiltersChange={setFilters}
            portfolioId={portfolioId}
            onPortfolioIdChange={setPortfolioId}
            portfolioOptions={portfolioOptions}
          />
        </aside>

        <main className="space-y-6">
          <div className="space-y-2">
            <Label className="text-xs">Timeframe</Label>
            <Select value={timeframe} onValueChange={(v) => setTimeframe(v as "1h" | "1d" | "1wk")}>
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

          <ScannerConditionSection
            conditions={conditions}
            onConditionsChange={setConditions}
            combinationLogic={combinationLogic}
            onCombinationLogicChange={setCombinationLogic}
          />

          <div className="flex flex-wrap items-center gap-4">
            <Button
              onClick={handleRunScan}
              disabled={scanning || conditions.length === 0 || filteredSymbols.length === 0}
            >
              {scanning ? "Scanning…" : "Run Scan"}
            </Button>
            {scanning && scanProgress && (
              <span className="text-sm text-muted-foreground">
                Scanning batch {scanProgress.batch}/{scanProgress.totalBatches} ({filteredSymbols.length.toLocaleString()} symbols)…
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

          <div className="rounded-lg border bg-muted/20 p-4 space-y-4">
            <h3 className="font-semibold">Save / Load preset</h3>
            <div className="flex flex-wrap gap-2 items-end">
              <div className="space-y-1">
                <Label className="text-xs">Preset name</Label>
                <Input
                  className="h-8 w-48"
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
        </main>
      </div>
    </div>
  );
}

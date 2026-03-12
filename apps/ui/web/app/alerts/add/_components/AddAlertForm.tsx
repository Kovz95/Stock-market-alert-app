"use client";

import * as React from "react";
import { useAtomValue, useSetAtom, useStore } from "jotai";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { useCreateAlert, useCreateAlertsBulk } from "@/lib/hooks/useAlerts";
import type { CreateAlertInput } from "@/actions/alert-actions";
import { getSymbolsByFilters } from "@/actions/stock-database-actions";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Field,
  FieldSet,
  FieldContent,
  FieldLegend,
} from "@/components/ui/field";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { AlertBasicFields } from "./AlertBasicFields";
import { AlertTickerSection } from "./AlertTickerSection";
import { AlertConditionsSection } from "./AlertConditionsSection";
import { IndicatorGuide } from "./IndicatorGuide";
import { AlertIndustryFilters } from "./AlertIndustryFilters";
import { AlertEtfFilters } from "./AlertEtfFilters";
import {
  buildConditionsStruct,
  generateAlertNameFromConditions,
} from "./types";
import { TIMEFRAMES } from "./constants";
import {
  addAlertNameAtom,
  addAlertActionAtom,
  addAlertIsRatioAtom,
  addAlertTickerAtom,
  addAlertStockNameAtom,
  addAlertExchangesAtom,
  addAlertCountryAtom,
  addAlertAssetTypeAtom,
  addAlertIndustryFiltersAtom,
  addAlertEtfFiltersAtom,
  addAlertTicker1Atom,
  addAlertTicker2Atom,
  addAlertStockName1Atom,
  addAlertStockName2Atom,
  addAlertAdjustmentMethodAtom,
  addAlertConditionsAtom,
  addAlertCombinationLogicAtom,
  addAlertTimeframeAtom,
  addAlertEnableMultiTimeframeAtom,
  addAlertComparisonTimeframeAtom,
  addAlertEnableMixedTimeframeAtom,
  addAlertBulkModeAtom,
  addAlertBulkTickersTextAtom,
  addAlertApplyToFilteredAtom,
  addAlertFilteredSymbolCountAtom,
  addAlertBulkProgressAtom,
} from "@/lib/store/add-alert";

function parseBulkTickers(text: string): { ticker: string; stockName: string }[] {
  const lines = text
    .split(/[\n,]+/)
    .map((s) => s.trim().toUpperCase())
    .filter(Boolean);
  const seen = new Set<string>();
  return lines
    .filter((t) => {
      if (seen.has(t)) return false;
      seen.add(t);
      return true;
    })
    .map((ticker) => ({ ticker, stockName: ticker }));
}

// ─── BasicSettingsCard ────────────────────────────────────────────────────────
function BasicSettingsCard() {
  const name = useAtomValue(addAlertNameAtom);
  const conditions = useAtomValue(addAlertConditionsAtom);
  const combinationLogic = useAtomValue(addAlertCombinationLogicAtom);

  const namePreview =
    !name.trim() && conditions.length > 0
      ? generateAlertNameFromConditions(conditions, combinationLogic)
      : "";

  return (
    <Card>
      <CardHeader>
        <CardTitle>Basic settings</CardTitle>
        <CardDescription>
          Name, timeframe, exchange, country, and asset type.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <AlertBasicFields />
        {namePreview && (
          <div className="mt-3 rounded-lg border bg-muted/30 p-3">
            <p className="text-xs text-muted-foreground">Auto-generated name preview:</p>
            <p className="text-sm font-medium">{namePreview}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ─── IndustryFiltersCard ──────────────────────────────────────────────────────
function IndustryFiltersCard() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Industry Filters</CardTitle>
        <CardDescription>
          Narrow symbols by RBICS classification (cascading).
        </CardDescription>
      </CardHeader>
      <CardContent>
        <AlertIndustryFilters />
      </CardContent>
    </Card>
  );
}

// ─── EtfFiltersCard ───────────────────────────────────────────────────────────
function EtfFiltersCard() {
  const assetType = useAtomValue(addAlertAssetTypeAtom);
  if (assetType !== "ETFs" && assetType !== "All") return null;

  return (
    <Card>
      <CardHeader>
        <CardTitle>ETF Filters</CardTitle>
        <CardDescription>
          Narrow ETFs by issuer, asset class, focus, and niche.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <AlertEtfFilters />
      </CardContent>
    </Card>
  );
}

// ─── TickerCard ───────────────────────────────────────────────────────────────
function TickerCard() {
  const isRatio = useAtomValue(addAlertIsRatioAtom);
  const applyToFiltered = useAtomValue(addAlertApplyToFilteredAtom);
  const setApplyToFiltered = useSetAtom(addAlertApplyToFilteredAtom);
  const bulkMode = useAtomValue(addAlertBulkModeAtom);
  const setBulkMode = useSetAtom(addAlertBulkModeAtom);
  const bulkTickersText = useAtomValue(addAlertBulkTickersTextAtom);
  const setBulkTickersText = useSetAtom(addAlertBulkTickersTextAtom);
  const filteredSymbolCount = useAtomValue(addAlertFilteredSymbolCountAtom);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Ticker / assets</CardTitle>
        <CardDescription>Single symbol or ratio of two assets.</CardDescription>
      </CardHeader>
      <CardContent>
        <AlertTickerSection />

        {!isRatio && (
          <div className="mt-4 space-y-4">
            <Field>
              <div className="flex items-center gap-2">
                <Checkbox
                  id="apply-to-filtered"
                  checked={applyToFiltered}
                  onCheckedChange={(c) => {
                    setApplyToFiltered(!!c);
                    if (c) setBulkMode(false);
                  }}
                  disabled={filteredSymbolCount === 0}
                />
                <Label htmlFor="apply-to-filtered">
                  Apply to ALL {filteredSymbolCount.toLocaleString()} filtered symbols
                </Label>
              </div>
              {applyToFiltered && filteredSymbolCount > 100 && (
                <p className="mt-2 text-sm text-muted-foreground">
                  Creating alerts for {filteredSymbolCount.toLocaleString()} symbols may take some time.
                </p>
              )}
              {applyToFiltered && filteredSymbolCount > 500 && (
                <p className="mt-1 text-sm text-destructive">
                  WARNING: Creating {filteredSymbolCount.toLocaleString()} alerts may fail. Consider using more specific filters.
                </p>
              )}
            </Field>

            {!applyToFiltered && (
              <Field>
                <div className="flex items-center gap-2">
                  <Checkbox
                    id="bulk-mode"
                    checked={bulkMode}
                    onCheckedChange={(c) => setBulkMode(!!c)}
                  />
                  <Label htmlFor="bulk-mode">
                    Create for multiple tickers (same conditions)
                  </Label>
                </div>
                {bulkMode && (
                  <FieldContent className="mt-2">
                    <Textarea
                      placeholder="Enter tickers, one per line or comma-separated (e.g. AAPL, MSFT, GOOGL)"
                      value={bulkTickersText}
                      onChange={(e) => setBulkTickersText(e.target.value)}
                      rows={4}
                      className="font-mono text-sm"
                    />
                  </FieldContent>
                )}
              </Field>
            )}
          </div>
        )}

        <div className="mt-4 rounded-lg border bg-muted/30 p-3">
          <p className="text-xs text-muted-foreground">
            Smart duplicate detection: allows multiple alerts for the same stock with the same conditions if they have different names.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

// ─── ConditionsCard ───────────────────────────────────────────────────────────
function ConditionsCard() {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Conditions</CardTitle>
        <CardDescription>
          Add one or more conditions; combine with AND or OR.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <AlertConditionsSection />
      </CardContent>
    </Card>
  );
}

// ─── TimeframeOptionsCard ─────────────────────────────────────────────────────
function TimeframeOptionsCard() {
  const timeframe = useAtomValue(addAlertTimeframeAtom);
  const setTimeframe = useSetAtom(addAlertTimeframeAtom);
  const enableMultiTimeframe = useAtomValue(addAlertEnableMultiTimeframeAtom);
  const setEnableMultiTimeframe = useSetAtom(addAlertEnableMultiTimeframeAtom);
  const comparisonTimeframe = useAtomValue(addAlertComparisonTimeframeAtom);
  const setComparisonTimeframe = useSetAtom(addAlertComparisonTimeframeAtom);
  const enableMixedTimeframe = useAtomValue(addAlertEnableMixedTimeframeAtom);
  const setEnableMixedTimeframe = useSetAtom(addAlertEnableMixedTimeframeAtom);

  return (
    <Card>
      <CardHeader>
        <CardTitle>Timeframe Options</CardTitle>
        <CardDescription>
          Multi-timeframe comparison and mixed timeframe conditions.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <Field>
          <div className="flex items-center gap-2">
            <Checkbox
              id="multi-timeframe"
              checked={enableMultiTimeframe}
              onCheckedChange={(c) => setEnableMultiTimeframe(!!c)}
            />
            <Label htmlFor="multi-timeframe">Enable Multi-Timeframe Comparison</Label>
          </div>
        </Field>

        {enableMultiTimeframe && (
          <div className="space-y-3 ml-6">
            <Field>
              <FieldLegend>Primary timeframe</FieldLegend>
              <FieldContent>
                <Select value={timeframe} onValueChange={setTimeframe}>
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {TIMEFRAMES.map((tf) => (
                      <SelectItem key={tf.value} value={tf.value}>
                        {tf.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </FieldContent>
            </Field>
            <Field>
              <FieldLegend>Comparison timeframe</FieldLegend>
              <FieldContent>
                <Select value={comparisonTimeframe} onValueChange={setComparisonTimeframe}>
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {TIMEFRAMES.map((tf) => (
                      <SelectItem key={tf.value} value={tf.value}>
                        {tf.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </FieldContent>
            </Field>
            {timeframe === comparisonTimeframe && (
              <p className="text-sm text-destructive">
                Primary and comparison timeframes are the same. Select different timeframes for meaningful comparison.
              </p>
            )}
            <div className="rounded-lg border bg-muted/30 p-3">
              <p className="text-xs text-muted-foreground">
                Use <code className="text-xs">Close[-1]</code> for primary timeframe and{" "}
                <code className="text-xs">Close_{comparisonTimeframe}[-1]</code> for comparison timeframe in your conditions.
              </p>
            </div>
          </div>
        )}

        <Field>
          <div className="flex items-center gap-2">
            <Checkbox
              id="mixed-timeframe"
              checked={enableMixedTimeframe}
              onCheckedChange={(c) => setEnableMixedTimeframe(!!c)}
            />
            <Label htmlFor="mixed-timeframe">Enable Mixed Timeframe Conditions</Label>
          </div>
        </Field>

        {enableMixedTimeframe && (
          <div className="ml-6 rounded-lg border bg-muted/30 p-3 space-y-2">
            <p className="text-xs text-muted-foreground">
              Mix indicators from different timeframes in your conditions. Examples:
            </p>
            <ul className="text-xs text-muted-foreground list-disc ml-4 space-y-1">
              <li><code>rsi(14)[-1] &lt; 30</code> (daily RSI)</li>
              <li><code>sma(50)_weekly[-1] &gt; sma(200)_weekly[-1]</code> (weekly MA cross)</li>
              <li><code>Close[-1] &gt; sma(20)[-1] AND Close_weekly[-1] &gt; sma(50)_weekly[-1]</code></li>
            </ul>
            <p className="text-xs text-muted-foreground">
              Enter mixed timeframe expressions in the condition builder using the &quot;Custom expression&quot; category.
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

// ─── BulkProgressDisplay ──────────────────────────────────────────────────────
function BulkProgressDisplay() {
  const bulkProgress = useAtomValue(addAlertBulkProgressAtom);
  if (!bulkProgress) return null;
  return (
    <div className="mt-3 rounded-lg border p-4">
      <div className="flex items-center gap-3">
        <span className="text-sm font-medium">
          {bulkProgress.creating ? "Creating alerts..." : "Bulk creation complete"}
        </span>
        {bulkProgress.creating && (
          <Badge variant="secondary">Processing {bulkProgress.total} alerts</Badge>
        )}
      </div>
      {!bulkProgress.creating && (
        <div className="mt-2 flex gap-3">
          {bulkProgress.created > 0 && (
            <Badge variant="default">{bulkProgress.created} created</Badge>
          )}
          {bulkProgress.skipped > 0 && (
            <Badge variant="secondary">{bulkProgress.skipped} skipped</Badge>
          )}
          {bulkProgress.failed > 0 && (
            <Badge variant="destructive">{bulkProgress.failed} failed</Badge>
          )}
        </div>
      )}
    </div>
  );
}

// ─── AddAlertForm ─────────────────────────────────────────────────────────────
export function AddAlertForm() {
  const router = useRouter();
  const store = useStore();
  const createAlert = useCreateAlert();
  const createAlertsBulk = useCreateAlertsBulk();
  const setBulkProgress = useSetAtom(addAlertBulkProgressAtom);
  const bulkMode = useAtomValue(addAlertBulkModeAtom);
  const bulkTickersText = useAtomValue(addAlertBulkTickersTextAtom);

  const isPending = createAlert.isPending || createAlertsBulk.isPending;

  const buildSharedPayload = (): Omit<CreateAlertInput, "name" | "stockName" | "ticker"> => {
    const conditions = store.get(addAlertConditionsAtom);
    const combinationLogic = store.get(addAlertCombinationLogicAtom);
    const action = store.get(addAlertActionAtom);
    const timeframe = store.get(addAlertTimeframeAtom);
    const exchanges = store.get(addAlertExchangesAtom);
    const country = store.get(addAlertCountryAtom);
    const isRatio = store.get(addAlertIsRatioAtom);
    const ticker1 = store.get(addAlertTicker1Atom);
    const ticker2 = store.get(addAlertTicker2Atom);
    const adjustmentMethod = store.get(addAlertAdjustmentMethodAtom);

    const conditionsStruct = buildConditionsStruct(conditions, combinationLogic);
    const validExchanges = exchanges.filter((e) => e !== "All");
    const exchange = validExchanges.length > 0 ? validExchanges[0] : "";

    return {
      ticker1: isRatio ? ticker1 : "",
      ticker2: isRatio ? ticker2 : "",
      combinationLogic,
      action,
      timeframe,
      exchange,
      country: country === "All" ? "" : country,
      ratio: isRatio ? "Yes" : "No",
      isRatio,
      adjustmentMethod: adjustmentMethod || "",
      ...(Object.keys(conditionsStruct).length > 0 && {
        conditions: conditionsStruct as unknown as CreateAlertInput["conditions"],
      }),
    };
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    const isRatio = store.get(addAlertIsRatioAtom);
    const ticker = store.get(addAlertTickerAtom);
    const ticker1 = store.get(addAlertTicker1Atom);
    const ticker2 = store.get(addAlertTicker2Atom);
    const stockName = store.get(addAlertStockNameAtom);
    const stockName1 = store.get(addAlertStockName1Atom);
    const stockName2 = store.get(addAlertStockName2Atom);
    const name = store.get(addAlertNameAtom);
    const applyToFiltered = store.get(addAlertApplyToFilteredAtom);
    const bulkModeVal = store.get(addAlertBulkModeAtom);
    const bulkTickersTextVal = store.get(addAlertBulkTickersTextAtom);
    const exchanges = store.get(addAlertExchangesAtom);
    const country = store.get(addAlertCountryAtom);
    const assetType = store.get(addAlertAssetTypeAtom);
    const industryFilters = store.get(addAlertIndustryFiltersAtom);
    const etfFilters = store.get(addAlertEtfFiltersAtom);

    if (isRatio) {
      if (!ticker1.trim() || !ticker2.trim()) {
        toast.error("Please enter both tickers for the ratio alert.");
        return;
      }
    } else {
      if (!ticker.trim() && !bulkModeVal && !applyToFiltered) {
        toast.error("Please enter a ticker symbol.");
        return;
      }
    }

    const shared = buildSharedPayload();

    if (applyToFiltered && !isRatio) {
      const result = await getSymbolsByFilters({
        exchanges,
        country,
        assetType,
        industry: industryFilters,
        etf:
          assetType !== "Stocks"
            ? {
                etfIssuers: etfFilters.etfIssuers,
                assetClasses: etfFilters.assetClasses,
                etfFocuses: etfFilters.etfFocuses,
                etfNiches: etfFilters.etfNiches,
              }
            : undefined,
      });
      if (result.error || result.symbols.length === 0) {
        toast.error(result.error || "No symbols found matching the filters.");
        return;
      }
      const items = result.symbols.map((sym) => ({
        ticker: sym.symbol,
        stockName: sym.name,
        name: name.trim() ? `${sym.name} - ${name.trim()}` : undefined,
      }));
      setBulkProgress({ creating: true, created: 0, skipped: 0, failed: 0, total: items.length });
      createAlertsBulk.mutate(
        { shared, items },
        {
          onSuccess: (res) => {
            setBulkProgress({
              creating: false,
              created: res.created,
              skipped: items.length - res.created - res.failed,
              failed: res.failed,
              total: items.length,
            });
            if (res.created > 0) toast.success(`Created ${res.created} alert(s).`);
            if (res.failed > 0) toast.error(`Failed to create ${res.failed} alert(s).`);
            if (res.errors.length > 0) res.errors.slice(0, 3).forEach((err) => toast.error(err));
            if (res.created > 0) router.push("/alerts");
          },
          onError: (err) => {
            setBulkProgress(null);
            toast.error(err.message ?? "Bulk create failed.");
          },
        }
      );
      return;
    }

    if (bulkModeVal && bulkTickersTextVal.trim()) {
      const items = parseBulkTickers(bulkTickersTextVal);
      if (items.length === 0) {
        toast.error("Enter at least one ticker (one per line or comma-separated).");
        return;
      }
      const nameTemplate = name.trim() || undefined;
      setBulkProgress({ creating: true, created: 0, skipped: 0, failed: 0, total: items.length });
      createAlertsBulk.mutate(
        {
          shared,
          items: items.map((item) => ({
            ticker: item.ticker,
            stockName: item.stockName,
            name: nameTemplate ? `${item.stockName} - ${nameTemplate}` : undefined,
          })),
        },
        {
          onSuccess: (res) => {
            setBulkProgress({
              creating: false,
              created: res.created,
              skipped: items.length - res.created - res.failed,
              failed: res.failed,
              total: items.length,
            });
            if (res.created > 0) toast.success(`Created ${res.created} alert(s).`);
            if (res.failed > 0) toast.error(`Failed to create ${res.failed} alert(s).`);
            if (res.errors.length > 0) res.errors.slice(0, 3).forEach((err) => toast.error(err));
            if (res.created > 0) router.push("/alerts");
          },
          onError: (err) => {
            setBulkProgress(null);
            toast.error(err.message ?? "Bulk create failed.");
          },
        }
      );
      return;
    }

    if (isRatio) {
      const alertName =
        name.trim() ||
        `${stockName1 || ticker1}/${stockName2 || ticker2} Ratio Alert`;
      const alertStockName = `${stockName1 || ticker1}/${stockName2 || ticker2}`;
      createAlert.mutate(
        {
          ...shared,
          name: alertName,
          stockName: alertStockName,
          ticker: ticker1,
          ticker1,
          ticker2,
        } as CreateAlertInput,
        {
          onSuccess: (data) => {
            if (data) {
              toast.success("Ratio alert created successfully.");
              router.push("/alerts");
            } else toast.error("Failed to create alert.");
          },
          onError: (err) => {
            toast.error(err.message ?? "Failed to create alert.");
          },
        }
      );
    } else {
      const alertName =
        name.trim() ||
        (stockName.trim() ? `${stockName.trim()} Alert` : "New Alert");
      const alertStockName = stockName.trim() || ticker.trim() || "Unknown";
      createAlert.mutate(
        {
          ...shared,
          name: alertName,
          stockName: alertStockName,
          ticker: ticker.trim(),
        } as CreateAlertInput,
        {
          onSuccess: (data) => {
            if (data) {
              toast.success("Alert created successfully.");
              router.push("/alerts");
            } else toast.error("Failed to create alert.");
          },
          onError: (err) => {
            toast.error(err.message ?? "Failed to create alert.");
          },
        }
      );
    }
  };

  return (
    <div className="space-y-8">
      <form onSubmit={handleSubmit}>
        <FieldSet className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <BasicSettingsCard />
          <IndustryFiltersCard />
          <EtfFiltersCard />
          <TickerCard />
          <ConditionsCard />
          <TimeframeOptionsCard />
        </FieldSet>

        <BulkProgressDisplay />

        <div className="flex flex-wrap gap-3 mt-3">
          <Button type="submit" disabled={isPending}>
            {isPending
              ? "Creating..."
              : bulkMode && bulkTickersText.trim()
                ? "Create alerts"
                : "Add alert"}
          </Button>
          <Button
            type="button"
            variant="outline"
            onClick={() => router.push("/alerts")}
          >
            Cancel
          </Button>
        </div>
      </form>
      <IndicatorGuide />
    </div>
  );
}

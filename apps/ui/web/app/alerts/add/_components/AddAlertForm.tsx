"use client";

import { useState } from "react";
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
  emptyIndustryFilters,
  emptyEtfFilters,
  generateAlertNameFromConditions,
  type AddAlertFormState,
  type BulkTickerItem,
} from "./types";
import {
  DEFAULT_TIMEFRAME,
  DEFAULT_COUNTRY,
  TIMEFRAMES,
} from "./constants";

const defaultFormState: AddAlertFormState = {
  name: "",
  action: "Buy",
  isRatio: false,
  ticker: "",
  stockName: "",
  exchanges: [],
  country: DEFAULT_COUNTRY,
  assetType: "All",
  industryFilters: emptyIndustryFilters,
  etfFilters: emptyEtfFilters,
  ticker1: "",
  ticker2: "",
  stockName1: "",
  stockName2: "",
  adjustmentMethod: "",
  conditions: [],
  combinationLogic: "AND",
  timeframe: DEFAULT_TIMEFRAME,
  enableMultiTimeframe: false,
  comparisonTimeframe: "1wk",
  enableMixedTimeframe: false,
};

function parseBulkTickers(text: string): BulkTickerItem[] {
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

export function AddAlertForm() {
  const router = useRouter();
  const createAlert = useCreateAlert();
  const createAlertsBulk = useCreateAlertsBulk();
  const [form, setForm] = useState<AddAlertFormState>(defaultFormState);
  const [bulkMode, setBulkMode] = useState(false);
  const [bulkTickersText, setBulkTickersText] = useState("");
  const [applyToFiltered, setApplyToFiltered] = useState(false);
  const [filteredSymbolCount, setFilteredSymbolCount] = useState(0);
  const [bulkProgress, setBulkProgress] = useState<{
    creating: boolean;
    created: number;
    skipped: number;
    failed: number;
    total: number;
  } | null>(null);

  const isPending = createAlert.isPending || createAlertsBulk.isPending;

  // Alert name preview (Task 6)
  const namePreview =
    !form.name.trim() && form.conditions.length > 0
      ? generateAlertNameFromConditions(form.conditions, form.combinationLogic)
      : "";

  const buildSharedPayload = (): Omit<
    CreateAlertInput,
    "name" | "stockName" | "ticker"
  > => {
    const conditionsStruct = buildConditionsStruct(
      form.conditions,
      form.combinationLogic
    );
    const validExchanges = form.exchanges.filter(e => e !== "All");
    const exchange = validExchanges.length > 0 ? validExchanges[0] : "";

    return {
      ticker1: form.isRatio ? form.ticker1 : "",
      ticker2: form.isRatio ? form.ticker2 : "",
      combinationLogic: form.combinationLogic,
      action: form.action,
      timeframe: form.timeframe,
      exchange,
      country: form.country === "All" ? "" : form.country,
      ratio: form.isRatio ? "Yes" : "No",
      isRatio: form.isRatio,
      adjustmentMethod: form.adjustmentMethod || "",
      ...(Object.keys(conditionsStruct).length > 0 && {
        conditions: conditionsStruct as unknown as CreateAlertInput["conditions"],
      }),
    };
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (form.isRatio) {
      if (!form.ticker1.trim() || !form.ticker2.trim()) {
        toast.error("Please enter both tickers for the ratio alert.");
        return;
      }
    } else {
      if (!form.ticker.trim() && !bulkMode && !applyToFiltered) {
        toast.error("Please enter a ticker symbol.");
        return;
      }
    }

    const shared = buildSharedPayload();

    // Handle "Apply to all filtered symbols" mode
    if (applyToFiltered && !form.isRatio) {
      const result = await getSymbolsByFilters({
        exchanges: form.exchanges,
        country: form.country,
        assetType: form.assetType,
        industry: form.industryFilters,
        etf: form.assetType !== "Stocks" ? {
          etfIssuers: form.etfFilters.etfIssuers,
          assetClasses: form.etfFilters.assetClasses,
          etfFocuses: form.etfFilters.etfFocuses,
          etfNiches: form.etfFilters.etfNiches,
        } : undefined,
      });
      if (result.error || result.symbols.length === 0) {
        toast.error(result.error || "No symbols found matching the filters.");
        return;
      }

      const items = result.symbols.map((sym) => ({
        ticker: sym.symbol,
        stockName: sym.name,
        name: form.name.trim() ? `${sym.name} - ${form.name.trim()}` : undefined,
      }));

      setBulkProgress({ creating: true, created: 0, skipped: 0, failed: 0, total: items.length });

      createAlertsBulk.mutate(
        { shared, items },
        {
          onSuccess: (result) => {
            setBulkProgress({
              creating: false,
              created: result.created,
              skipped: items.length - result.created - result.failed,
              failed: result.failed,
              total: items.length,
            });
            if (result.created > 0)
              toast.success(`Created ${result.created} alert(s).`);
            if (result.failed > 0)
              toast.error(`Failed to create ${result.failed} alert(s).`);
            if (result.errors.length > 0)
              result.errors.slice(0, 3).forEach((err) => toast.error(err));
            if (result.created > 0) router.push("/alerts");
          },
          onError: (err) => {
            setBulkProgress(null);
            toast.error(err.message ?? "Bulk create failed.");
          },
        }
      );
      return;
    }

    if (bulkMode && bulkTickersText.trim()) {
      const items = parseBulkTickers(bulkTickersText);
      if (items.length === 0) {
        toast.error("Enter at least one ticker (one per line or comma-separated).");
        return;
      }
      const nameTemplate = form.name.trim() || undefined;

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
          onSuccess: (result) => {
            setBulkProgress({
              creating: false,
              created: result.created,
              skipped: items.length - result.created - result.failed,
              failed: result.failed,
              total: items.length,
            });
            if (result.created > 0)
              toast.success(`Created ${result.created} alert(s).`);
            if (result.failed > 0)
              toast.error(`Failed to create ${result.failed} alert(s).`);
            if (result.errors.length > 0)
              result.errors.slice(0, 3).forEach((err) => toast.error(err));
            if (result.created > 0) router.push("/alerts");
          },
          onError: (err) => {
            setBulkProgress(null);
            toast.error(err.message ?? "Bulk create failed.");
          },
        }
      );
      return;
    }

    // Single alert
    if (form.isRatio) {
      const name =
        form.name.trim() ||
        `${form.stockName1 || form.ticker1}/${form.stockName2 || form.ticker2} Ratio Alert`;
      const stockName = `${form.stockName1 || form.ticker1}/${form.stockName2 || form.ticker2}`;
      createAlert.mutate(
        {
          ...shared,
          name,
          stockName,
          ticker: form.ticker1,
          ticker1: form.ticker1,
          ticker2: form.ticker2,
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
      const name =
        form.name.trim() ||
        (form.stockName.trim() ? `${form.stockName.trim()} Alert` : "New Alert");
      const stockName = form.stockName.trim() || form.ticker.trim() || "Unknown";
      const ticker = form.ticker.trim();
      createAlert.mutate(
        {
          ...shared,
          name,
          stockName,
          ticker,
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

  const showEtfFilters = form.assetType === "ETFs" || form.assetType === "All";

  return (
    <div className="space-y-8">
      <form onSubmit={handleSubmit}>
        <FieldSet className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <Card>
            <CardHeader>
              <CardTitle>Basic settings</CardTitle>
              <CardDescription>
                Name, timeframe, exchange, country, and asset type.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <AlertBasicFields
                name={form.name}
                onNameChange={(v) => setForm((f) => ({ ...f, name: v }))}
                timeframe={form.timeframe}
                onTimeframeChange={(v) => setForm((f) => ({ ...f, timeframe: v }))}
                exchanges={form.exchanges}
                onExchangesChange={(v) => setForm((f) => ({ ...f, exchanges: v }))}
                country={form.country}
                onCountryChange={(v) => setForm((f) => ({ ...f, country: v }))}
                assetType={form.assetType}
                onAssetTypeChange={(v) => setForm((f) => ({ ...f, assetType: v }))}
                industryFilters={form.industryFilters}
                etfFilters={form.etfFilters}
                onSymbolCountChange={setFilteredSymbolCount}
              />

              {/* Alert name preview (Task 6) */}
              {namePreview && (
                <div className="mt-3 rounded-lg border bg-muted/30 p-3">
                  <p className="text-xs text-muted-foreground">Auto-generated name preview:</p>
                  <p className="text-sm font-medium">{namePreview}</p>
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Industry Filters</CardTitle>
              <CardDescription>
                Narrow symbols by RBICS classification (cascading).
              </CardDescription>
            </CardHeader>
            <CardContent>
              <AlertIndustryFilters
                exchanges={form.exchanges}
                country={form.country}
                filters={form.industryFilters}
                onFiltersChange={(v) =>
                  setForm((f) => ({ ...f, industryFilters: v }))
                }
              />
            </CardContent>
          </Card>

          {/* ETF Filters card (Task 2) */}
          {showEtfFilters && (
            <Card>
              <CardHeader>
                <CardTitle>ETF Filters</CardTitle>
                <CardDescription>
                  Narrow ETFs by issuer, asset class, focus, and niche.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <AlertEtfFilters
                  exchanges={form.exchanges}
                  country={form.country}
                  filters={form.etfFilters}
                  onFiltersChange={(v) =>
                    setForm((f) => ({ ...f, etfFilters: v }))
                  }
                />
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle>Ticker / assets</CardTitle>
              <CardDescription>
                Single symbol or ratio of two assets.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <AlertTickerSection
                isRatio={form.isRatio}
                onRatioChange={(v) => setForm((f) => ({ ...f, isRatio: v }))}
                ticker={form.ticker}
                onTickerChange={(v) => setForm((f) => ({ ...f, ticker: v }))}
                stockName={form.stockName}
                onStockNameChange={(v) =>
                  setForm((f) => ({ ...f, stockName: v }))
                }
                ticker1={form.ticker1}
                onTicker1Change={(v) => setForm((f) => ({ ...f, ticker1: v }))}
                ticker2={form.ticker2}
                onTicker2Change={(v) => setForm((f) => ({ ...f, ticker2: v }))}
                stockName1={form.stockName1}
                onStockName1Change={(v) =>
                  setForm((f) => ({ ...f, stockName1: v }))
                }
                stockName2={form.stockName2}
                onStockName2Change={(v) =>
                  setForm((f) => ({ ...f, stockName2: v }))
                }
                adjustmentMethod={form.adjustmentMethod}
                onAdjustmentMethodChange={(v) =>
                  setForm((f) => ({ ...f, adjustmentMethod: v }))
                }
              />

              {!form.isRatio && (
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

              {/* Duplicate detection info (Task 9) */}
              <div className="mt-4 rounded-lg border bg-muted/30 p-3">
                <p className="text-xs text-muted-foreground">
                  Smart duplicate detection: allows multiple alerts for the same stock with the same conditions if they have different names.
                </p>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Conditions</CardTitle>
              <CardDescription>
                Add one or more conditions; combine with AND or OR.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <AlertConditionsSection
                conditions={form.conditions}
                onConditionsChange={(v) =>
                  setForm((f) => ({ ...f, conditions: v }))
                }
                combinationLogic={form.combinationLogic}
                onCombinationLogicChange={(v) =>
                  setForm((f) => ({ ...f, combinationLogic: v }))
                }
              />
            </CardContent>
          </Card>

          {/* Multi-Timeframe & Mixed Timeframe card (Tasks 3 & 4) */}
          <Card>
            <CardHeader>
              <CardTitle>Timeframe Options</CardTitle>
              <CardDescription>
                Multi-timeframe comparison and mixed timeframe conditions.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Multi-Timeframe Comparison (Task 3) */}
              <Field>
                <div className="flex items-center gap-2">
                  <Checkbox
                    id="multi-timeframe"
                    checked={form.enableMultiTimeframe}
                    onCheckedChange={(c) =>
                      setForm((f) => ({ ...f, enableMultiTimeframe: !!c }))
                    }
                  />
                  <Label htmlFor="multi-timeframe">
                    Enable Multi-Timeframe Comparison
                  </Label>
                </div>
              </Field>

              {form.enableMultiTimeframe && (
                <div className="space-y-3 ml-6">
                  <Field>
                    <FieldLegend>Primary timeframe</FieldLegend>
                    <FieldContent>
                      <Select value={form.timeframe} onValueChange={(v) => setForm((f) => ({ ...f, timeframe: v }))}>
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
                      <Select value={form.comparisonTimeframe} onValueChange={(v) => setForm((f) => ({ ...f, comparisonTimeframe: v }))}>
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
                  {form.timeframe === form.comparisonTimeframe && (
                    <p className="text-sm text-destructive">
                      Primary and comparison timeframes are the same. Select different timeframes for meaningful comparison.
                    </p>
                  )}
                  <div className="rounded-lg border bg-muted/30 p-3">
                    <p className="text-xs text-muted-foreground">
                      Use <code className="text-xs">Close[-1]</code> for primary timeframe and{" "}
                      <code className="text-xs">Close_{form.comparisonTimeframe}[-1]</code> for comparison timeframe in your conditions.
                    </p>
                  </div>
                </div>
              )}

              {/* Mixed Timeframe Conditions (Task 4) */}
              <Field>
                <div className="flex items-center gap-2">
                  <Checkbox
                    id="mixed-timeframe"
                    checked={form.enableMixedTimeframe}
                    onCheckedChange={(c) =>
                      setForm((f) => ({ ...f, enableMixedTimeframe: !!c }))
                    }
                  />
                  <Label htmlFor="mixed-timeframe">
                    Enable Mixed Timeframe Conditions
                  </Label>
                </div>
              </Field>

              {form.enableMixedTimeframe && (
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
        </FieldSet>

        {/* Bulk progress indicator (Task 10) */}
        {bulkProgress && (
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
        )}

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

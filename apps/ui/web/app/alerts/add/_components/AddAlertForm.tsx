"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";
import { useCreateAlert, useCreateAlertsBulk } from "@/lib/hooks/useAlerts";
import type { CreateAlertInput } from "@/actions/alert-actions";
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
  FieldGroup,
  FieldSet,
  FieldLabel,
  FieldContent,
  FieldLegend,
} from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { AlertBasicFields } from "./AlertBasicFields";
import { AlertTickerSection } from "./AlertTickerSection";
import { AlertConditionsSection } from "./AlertConditionsSection";
import { IndicatorGuide } from "./IndicatorGuide";
import {
  buildConditionsStruct,
  type AddAlertFormState,
  type BulkTickerItem,
} from "./types";

const defaultFormState: AddAlertFormState = {
  name: "",
  action: "Buy",
  isRatio: false,
  ticker: "",
  stockName: "",
  exchange: "NYSE",
  country: "US",
  ticker1: "",
  ticker2: "",
  stockName1: "",
  stockName2: "",
  adjustmentMethod: "",
  conditions: [],
  combinationLogic: "AND",
  timeframe: "1D",
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

  const isPending = createAlert.isPending || createAlertsBulk.isPending;

  const buildSharedPayload = (): Omit<
    CreateAlertInput,
    "name" | "stockName" | "ticker"
  > => {
    const conditionsStruct = buildConditionsStruct(
      form.conditions,
      form.combinationLogic
    );
    return {
      ticker1: form.isRatio ? form.ticker1 : "",
      ticker2: form.isRatio ? form.ticker2 : "",
      combinationLogic: form.combinationLogic,
      action: form.action,
      timeframe: form.timeframe,
      exchange: form.exchange,
      country: form.country,
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
      if (!form.ticker.trim() && !bulkMode) {
        toast.error("Please enter a ticker symbol.");
        return;
      }
    }

    const shared = buildSharedPayload();

    if (bulkMode && bulkTickersText.trim()) {
      const items = parseBulkTickers(bulkTickersText);
      if (items.length === 0) {
        toast.error("Enter at least one ticker (one per line or comma-separated).");
        return;
      }
      const nameTemplate = form.name.trim() || undefined;
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
            if (result.created > 0)
              toast.success(`Created ${result.created} alert(s).`);
            if (result.failed > 0)
              toast.error(`Failed to create ${result.failed} alert(s).`);
            if (result.errors.length > 0)
              result.errors.slice(0, 3).forEach((err) => toast.error(err));
            if (result.created > 0) router.push("/alerts");
          },
          onError: (err) => {
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

  return (
    <div className="space-y-8">
      <form onSubmit={handleSubmit}>
        <FieldSet className="flex flex-col gap-8">
          <Card>
            <CardHeader>
              <CardTitle>Basic settings</CardTitle>
              <CardDescription>
                Name, action, timeframe, exchange, and country.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <AlertBasicFields
                name={form.name}
                onNameChange={(v) => setForm((f) => ({ ...f, name: v }))}
                action={form.action}
                onActionChange={(v) => setForm((f) => ({ ...f, action: v }))}
                timeframe={form.timeframe}
                onTimeframeChange={(v) => setForm((f) => ({ ...f, timeframe: v }))}
                exchange={form.exchange}
                onExchangeChange={(v) => setForm((f) => ({ ...f, exchange: v }))}
                country={form.country}
                onCountryChange={(v) => setForm((f) => ({ ...f, country: v }))}
              />
            </CardContent>
          </Card>

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
                <Field className="mt-4">
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

          <div className="flex flex-wrap gap-3">
            <Button type="submit" disabled={isPending}>
              {isPending
                ? "Creating…"
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
        </FieldSet>
      </form>

      <IndicatorGuide />
    </div>
  );
}

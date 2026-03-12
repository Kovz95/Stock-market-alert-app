"use client";

import { useEffect, useState } from "react";
import { useAtomValue, useSetAtom } from "jotai";
import {
  Field,
  FieldGroup,
  FieldLabel,
  FieldContent,
  FieldLegend,
} from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Combobox,
  ComboboxChips,
  ComboboxChip,
  ComboboxChipsInput,
  ComboboxContent,
  ComboboxList,
  ComboboxItem,
  ComboboxEmpty,
  useComboboxAnchor,
} from "@/components/ui/combobox";
import {
  TIMEFRAMES,
  EXCHANGES,
  COUNTRIES,
  COUNTRY_CODE_TO_NAME,
} from "./constants";
import { countSymbolsByFilters } from "@/actions/stock-database-actions";
import { Badge } from "@/components/ui/badge";
import type { AssetType } from "./types";
import {
  addAlertNameAtom,
  addAlertTimeframeAtom,
  addAlertExchangesAtom,
  addAlertCountryAtom,
  addAlertAssetTypeAtom,
  addAlertIndustryFiltersAtom,
  addAlertEtfFiltersAtom,
  addAlertFilteredSymbolCountAtom,
} from "@/lib/store/add-alert";

export function AlertBasicFields() {
  const name = useAtomValue(addAlertNameAtom);
  const setName = useSetAtom(addAlertNameAtom);
  const timeframe = useAtomValue(addAlertTimeframeAtom);
  const setTimeframe = useSetAtom(addAlertTimeframeAtom);
  const exchanges = useAtomValue(addAlertExchangesAtom);
  const setExchanges = useSetAtom(addAlertExchangesAtom);
  const country = useAtomValue(addAlertCountryAtom);
  const setCountry = useSetAtom(addAlertCountryAtom);
  const assetType = useAtomValue(addAlertAssetTypeAtom);
  const setAssetType = useSetAtom(addAlertAssetTypeAtom);
  const industryFilters = useAtomValue(addAlertIndustryFiltersAtom);
  const etfFilters = useAtomValue(addAlertEtfFiltersAtom);
  const setFilteredSymbolCount = useSetAtom(addAlertFilteredSymbolCountAtom);

  const [symbolCount, setSymbolCount] = useState<number | null>(null);
  const [isCountingSymbols, setIsCountingSymbols] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const fetchCount = async () => {
      setIsCountingSymbols(true);
      try {
        const result = await countSymbolsByFilters({
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
        if (!cancelled && !result.error) {
          setSymbolCount(result.count);
          setFilteredSymbolCount(result.count);
        }
      } catch (err) {
        console.error("Failed to count symbols:", err);
      } finally {
        if (!cancelled) setIsCountingSymbols(false);
      }
    };
    fetchCount();
    return () => {
      cancelled = true;
    };
  }, [exchanges, country, assetType, industryFilters, etfFilters, setFilteredSymbolCount]);

  const showSymbolCount = symbolCount !== null;
  const exchangeAnchor = useComboboxAnchor();

  const handleExchangeChange = (value: string | string[] | null) => {
    if (Array.isArray(value)) {
      setExchanges(value);
    } else if (value) {
      setExchanges([value]);
    } else {
      setExchanges([]);
    }
  };

  return (
    <FieldGroup>
      <Field>
        <FieldLabel>Alert name (optional)</FieldLabel>
        <FieldContent>
          <Input
            placeholder="e.g. AAPL Price Alert"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </FieldContent>
      </Field>
      <Field>
        <FieldLegend>Timeframe</FieldLegend>
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
        <FieldLegend>Exchange(s)</FieldLegend>
        <FieldContent>
          <Combobox
            value={exchanges}
            onValueChange={handleExchangeChange}
            items={EXCHANGES.map((ex) => ({ value: ex, label: ex }))}
            multiple
          >
            <ComboboxChips ref={exchangeAnchor}>
              {exchanges.map((ex) => (
                <ComboboxChip key={ex}>{ex}</ComboboxChip>
              ))}
              <ComboboxChipsInput placeholder="Select exchanges..." />
            </ComboboxChips>
            <ComboboxContent anchor={exchangeAnchor.current}>
              <ComboboxList>
                {EXCHANGES.map((ex) => (
                  <ComboboxItem key={ex} value={ex}>
                    {ex}
                  </ComboboxItem>
                ))}
                <ComboboxEmpty>No exchanges found</ComboboxEmpty>
              </ComboboxList>
            </ComboboxContent>
          </Combobox>
        </FieldContent>
      </Field>
      <Field>
        <FieldLegend>Country</FieldLegend>
        <FieldContent>
          <Select value={country} onValueChange={setCountry}>
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {COUNTRIES.map((code) => (
                <SelectItem key={code} value={code}>
                  {COUNTRY_CODE_TO_NAME[code] ?? code} ({code})
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </FieldContent>
      </Field>
      <Field>
        <FieldLegend>Asset type</FieldLegend>
        <FieldContent>
          <Select
            value={assetType}
            onValueChange={(v) => setAssetType(v as AssetType)}
          >
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="All">All</SelectItem>
              <SelectItem value="Stocks">Stocks</SelectItem>
              <SelectItem value="ETFs">ETFs</SelectItem>
            </SelectContent>
          </Select>
        </FieldContent>
      </Field>

      {showSymbolCount && (
        <div className="rounded-lg border bg-muted/50 p-3">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium">Filtered symbols:</span>
            {isCountingSymbols ? (
              <Badge variant="secondary">Loading...</Badge>
            ) : (
              <Badge variant="default" className="font-mono">
                {symbolCount?.toLocaleString()}
              </Badge>
            )}
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            {(() => {
              const validExchanges = exchanges.filter((e) => e !== "All");
              const hasExchangeFilter = validExchanges.length > 0;
              const hasCountryFilter = country !== "All";

              if (hasExchangeFilter && hasCountryFilter) {
                const exchangeText =
                  validExchanges.length === 1
                    ? validExchanges[0]
                    : `${validExchanges.length} exchanges (${validExchanges.join(", ")})`;
                return `Showing symbols from ${exchangeText} in ${COUNTRY_CODE_TO_NAME[country] ?? country}`;
              } else if (hasExchangeFilter) {
                const exchangeText =
                  validExchanges.length === 1
                    ? validExchanges[0]
                    : `${validExchanges.length} exchanges (${validExchanges.join(", ")})`;
                return `Showing symbols from ${exchangeText}`;
              } else if (hasCountryFilter) {
                return `Showing symbols in ${COUNTRY_CODE_TO_NAME[country] ?? country}`;
              }
              return "Showing symbols from all exchanges";
            })()}
          </p>
        </div>
      )}
    </FieldGroup>
  );
}

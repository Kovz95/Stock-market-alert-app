"use client";

import { useEffect, useState } from "react";
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
import type { IndustryFilters, AssetType, EtfFilters } from "./types";

export interface AlertBasicFieldsProps {
  name: string;
  onNameChange: (v: string) => void;
  timeframe: string;
  onTimeframeChange: (v: string) => void;
  exchanges: string[];
  onExchangesChange: (v: string[]) => void;
  country: string;
  onCountryChange: (v: string) => void;
  assetType: AssetType;
  onAssetTypeChange: (v: AssetType) => void;
  industryFilters: IndustryFilters;
  etfFilters: EtfFilters;
  onSymbolCountChange?: (count: number) => void;
}

export function AlertBasicFields({
  name,
  onNameChange,
  timeframe,
  onTimeframeChange,
  exchanges,
  onExchangesChange,
  country,
  onCountryChange,
  assetType,
  onAssetTypeChange,
  industryFilters,
  etfFilters,
  onSymbolCountChange,
}: AlertBasicFieldsProps) {
  const [symbolCount, setSymbolCount] = useState<number | null>(null);
  const [isCountingSymbols, setIsCountingSymbols] = useState(false);

  useEffect(() => {
    const fetchCount = async () => {
      setIsCountingSymbols(true);
      try {
        const result = await countSymbolsByFilters({
          exchanges,
          country,
          assetType,
          industry: industryFilters,
          etf: assetType !== "Stocks" ? {
            etfIssuers: etfFilters.etfIssuers,
            assetClasses: etfFilters.assetClasses,
            etfFocuses: etfFilters.etfFocuses,
            etfNiches: etfFilters.etfNiches,
          } : undefined,
        });
        if (!result.error) {
          setSymbolCount(result.count);
          onSymbolCountChange?.(result.count);
        }
      } catch (err) {
        console.error("Failed to count symbols:", err);
      } finally {
        setIsCountingSymbols(false);
      }
    };

    fetchCount();
  }, [exchanges, country, assetType, industryFilters, etfFilters, onSymbolCountChange]);

  const showSymbolCount = symbolCount !== null;
  const exchangeAnchor = useComboboxAnchor();

  const handleExchangeChange = (value: string | string[] | null) => {
    if (Array.isArray(value)) {
      onExchangesChange(value);
    } else if (value) {
      onExchangesChange([value]);
    } else {
      onExchangesChange([]);
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
            onChange={(e) => onNameChange(e.target.value)}
          />
        </FieldContent>
      </Field>
      <Field>
        <FieldLegend>Timeframe</FieldLegend>
        <FieldContent>
          <Select value={timeframe} onValueChange={onTimeframeChange}>
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
            items={EXCHANGES.map(ex => ({ value: ex, label: ex }))}
            multiple
          >
            <ComboboxChips ref={exchangeAnchor}>
              {exchanges.map((ex) => (
                <ComboboxChip key={ex}>
                  {ex}
                </ComboboxChip>
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
          <Select value={country} onValueChange={onCountryChange}>
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
          <Select value={assetType} onValueChange={(v) => onAssetTypeChange(v as AssetType)}>
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
              const validExchanges = exchanges.filter(e => e !== "All");
              const hasExchangeFilter = validExchanges.length > 0;
              const hasCountryFilter = country !== "All";

              if (hasExchangeFilter && hasCountryFilter) {
                const exchangeText = validExchanges.length === 1
                  ? validExchanges[0]
                  : `${validExchanges.length} exchanges (${validExchanges.join(", ")})`;
                return `Showing symbols from ${exchangeText} in ${COUNTRY_CODE_TO_NAME[country] ?? country}`;
              } else if (hasExchangeFilter) {
                const exchangeText = validExchanges.length === 1
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

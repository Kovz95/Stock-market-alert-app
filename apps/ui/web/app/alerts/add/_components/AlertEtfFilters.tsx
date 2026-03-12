"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import {
  Field,
  FieldGroup,
  FieldLegend,
  FieldContent,
} from "@/components/ui/field";
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
  getEtfFilterOptions,
  type EtfFilterOptions,
  type EtfFilterValues,
} from "@/actions/stock-database-actions";
import type { EtfFilters } from "./types";

export interface AlertEtfFiltersProps {
  exchanges: string[];
  country: string;
  filters: EtfFilters;
  onFiltersChange: (filters: EtfFilters) => void;
}

type EtfFilterLevel = keyof EtfFilters;

const FILTER_LEVELS: { key: EtfFilterLevel; label: string }[] = [
  { key: "etfIssuers", label: "ETF Issuer" },
  { key: "assetClasses", label: "Asset Class" },
  { key: "etfFocuses", label: "ETF Focus" },
  { key: "etfNiches", label: "ETF Niche" },
];

const EMPTY_OPTIONS: EtfFilterOptions = {
  etfIssuers: [],
  assetClasses: [],
  etfFocuses: [],
  etfNiches: [],
};

export function AlertEtfFilters({
  exchanges,
  country,
  filters,
  onFiltersChange,
}: AlertEtfFiltersProps) {
  const [options, setOptions] = useState<EtfFilterOptions>(EMPTY_OPTIONS);
  const [loading, setLoading] = useState(false);
  const fetchIdRef = useRef(0);

  const issuerAnchor = useComboboxAnchor();
  const assetClassAnchor = useComboboxAnchor();
  const focusAnchor = useComboboxAnchor();
  const nicheAnchor = useComboboxAnchor();

  const anchors: Record<EtfFilterLevel, React.RefObject<HTMLDivElement | null>> = {
    etfIssuers: issuerAnchor,
    assetClasses: assetClassAnchor,
    etfFocuses: focusAnchor,
    etfNiches: nicheAnchor,
  };

  const fetchOptions = useCallback(async () => {
    const id = ++fetchIdRef.current;
    setLoading(true);
    try {
      const selected: EtfFilterValues = {
        etfIssuers: filters.etfIssuers,
        assetClasses: filters.assetClasses,
        etfFocuses: filters.etfFocuses,
        etfNiches: filters.etfNiches,
      };
      const result = await getEtfFilterOptions(
        { exchanges, country },
        selected
      );
      if (id === fetchIdRef.current) {
        setOptions(result);
      }
    } catch {
      // keep stale options on error
    } finally {
      if (id === fetchIdRef.current) setLoading(false);
    }
  }, [exchanges, country, filters]);

  useEffect(() => {
    fetchOptions();
  }, [fetchOptions]);

  const handleChange = (level: EtfFilterLevel, value: string | string[] | null) => {
    const newValues = Array.isArray(value) ? value : value ? [value] : [];
    const levelIndex = FILTER_LEVELS.findIndex((l) => l.key === level);

    const updated = { ...filters, [level]: newValues };

    // Clear all child levels when a parent changes
    for (let i = levelIndex + 1; i < FILTER_LEVELS.length; i++) {
      updated[FILTER_LEVELS[i].key] = [];
    }

    onFiltersChange(updated);
  };

  const hasAnyFilter = FILTER_LEVELS.some((l) => filters[l.key].length > 0);

  return (
    <FieldGroup>
      <div className="flex items-center justify-between">
        <FieldLegend>ETF Filters</FieldLegend>
        {hasAnyFilter && (
          <button
            type="button"
            className="text-xs text-muted-foreground hover:text-foreground underline"
            onClick={() =>
              onFiltersChange({
                etfIssuers: [],
                assetClasses: [],
                etfFocuses: [],
                etfNiches: [],
              })
            }
          >
            Clear all
          </button>
        )}
      </div>

      {FILTER_LEVELS.map(({ key, label }) => {
        const levelOptions = options[key] ?? [];
        const selected = filters[key];
        const anchor = anchors[key];

        return (
          <Field key={key}>
            <FieldLegend>{label}</FieldLegend>
            <FieldContent>
              <Combobox
                value={selected}
                onValueChange={(v) => handleChange(key, v)}
                items={levelOptions.map((o) => ({ value: o, label: o }))}
                multiple
              >
                <ComboboxChips ref={anchor}>
                  {selected.map((v) => (
                    <ComboboxChip key={v}>{v}</ComboboxChip>
                  ))}
                  <ComboboxChipsInput
                    placeholder={
                      loading
                        ? "Loading..."
                        : levelOptions.length === 0
                          ? "No options available"
                          : `Select ${label.toLowerCase()}...`
                    }
                  />
                </ComboboxChips>
                <ComboboxContent anchor={anchor.current}>
                  <ComboboxList>
                    {levelOptions.map((o) => (
                      <ComboboxItem key={o} value={o}>
                        {o}
                      </ComboboxItem>
                    ))}
                    <ComboboxEmpty>No {label.toLowerCase()} found</ComboboxEmpty>
                  </ComboboxList>
                </ComboboxContent>
              </Combobox>
            </FieldContent>
          </Field>
        );
      })}
    </FieldGroup>
  );
}

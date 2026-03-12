"use client";

import { useEffect, useState, useCallback, useRef } from "react";
import { useAtomValue, useSetAtom } from "jotai";
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
  getIndustryFilterOptions,
  type IndustryFilterOptions,
  type IndustryFilterValues,
} from "@/actions/stock-database-actions";
import type { IndustryFilters } from "./types";
import {
  addAlertExchangesAtom,
  addAlertCountryAtom,
  addAlertIndustryFiltersAtom,
} from "@/lib/store/add-alert";

type FilterLevel = keyof IndustryFilters;

const FILTER_LEVELS: { key: FilterLevel; label: string }[] = [
  { key: "economies", label: "Economy" },
  { key: "sectors", label: "Sector" },
  { key: "subsectors", label: "Subsector" },
  { key: "industryGroups", label: "Industry Group" },
  { key: "industries", label: "Industry" },
  { key: "subindustries", label: "Subindustry" },
];

const EMPTY_OPTIONS: IndustryFilterOptions = {
  economies: [],
  sectors: [],
  subsectors: [],
  industryGroups: [],
  industries: [],
  subindustries: [],
};

export function AlertIndustryFilters() {
  const exchanges = useAtomValue(addAlertExchangesAtom);
  const country = useAtomValue(addAlertCountryAtom);
  const filters = useAtomValue(addAlertIndustryFiltersAtom);
  const setFilters = useSetAtom(addAlertIndustryFiltersAtom);

  const [options, setOptions] = useState<IndustryFilterOptions>(EMPTY_OPTIONS);
  const [loading, setLoading] = useState(false);
  const fetchIdRef = useRef(0);

  const economyAnchor = useComboboxAnchor();
  const sectorAnchor = useComboboxAnchor();
  const subsectorAnchor = useComboboxAnchor();
  const industryGroupAnchor = useComboboxAnchor();
  const industryAnchor = useComboboxAnchor();
  const subindustryAnchor = useComboboxAnchor();

  const anchors: Record<FilterLevel, React.RefObject<HTMLDivElement | null>> = {
    economies: economyAnchor,
    sectors: sectorAnchor,
    subsectors: subsectorAnchor,
    industryGroups: industryGroupAnchor,
    industries: industryAnchor,
    subindustries: subindustryAnchor,
  };

  const fetchOptions = useCallback(async () => {
    const id = ++fetchIdRef.current;
    setLoading(true);
    try {
      const selected: IndustryFilterValues = {
        economies: filters.economies,
        sectors: filters.sectors,
        subsectors: filters.subsectors,
        industryGroups: filters.industryGroups,
        industries: filters.industries,
        subindustries: filters.subindustries,
      };
      const result = await getIndustryFilterOptions(
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

  const handleChange = (level: FilterLevel, value: string | string[] | null) => {
    const newValues = Array.isArray(value) ? value : value ? [value] : [];
    const levelIndex = FILTER_LEVELS.findIndex((l) => l.key === level);

    const updated = { ...filters, [level]: newValues };

    // Clear all child levels when a parent changes
    for (let i = levelIndex + 1; i < FILTER_LEVELS.length; i++) {
      updated[FILTER_LEVELS[i].key] = [];
    }

    setFilters(updated);
  };

  const hasAnyFilter = FILTER_LEVELS.some((l) => filters[l.key].length > 0);

  return (
    <FieldGroup>
      <div className="flex items-center justify-between">
        <FieldLegend>Industry Filters (RBICS)</FieldLegend>
        {hasAnyFilter && (
          <button
            type="button"
            className="text-xs text-muted-foreground hover:text-foreground underline"
            onClick={() =>
              setFilters({
                economies: [],
                sectors: [],
                subsectors: [],
                industryGroups: [],
                industries: [],
                subindustries: [],
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

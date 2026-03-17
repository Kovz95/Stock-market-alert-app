"use client";

import { useAtomValue, useSetAtom } from "jotai";
import { useCallback, useEffect, useRef, useState } from "react";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  alertsPageAtom,
  alertsSearchAtom,
  alertsTimeframeFilterAtom,
} from "@/lib/store/alerts";

const DEBOUNCE_MS = 300;

const TIMEFRAME_OPTIONS = [
  { value: "all", label: "All Timeframes" },
  { value: "1h", label: "Hourly" },
  { value: "1d", label: "Daily" },
  { value: "1wk", label: "Weekly" },
] as const;

export function AlertsSearchBar() {
  const searchFromStore = useAtomValue(alertsSearchAtom);
  const setSearch = useSetAtom(alertsSearchAtom);
  const setPage = useSetAtom(alertsPageAtom);
  const timeframe = useAtomValue(alertsTimeframeFilterAtom);
  const setTimeframe = useSetAtom(alertsTimeframeFilterAtom);

  const [inputValue, setInputValue] = useState(searchFromStore);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Sync store -> input when store changes (e.g. external clear)
  useEffect(() => {
    setInputValue(searchFromStore);
  }, [searchFromStore]);

  // Debounce input -> store and reset to page 1
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      debounceRef.current = null;
      const trimmed = inputValue.trim();
      if (trimmed !== searchFromStore) {
        setSearch(trimmed);
        setPage(1);
      }
    }, DEBOUNCE_MS);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [inputValue, setSearch, setPage, searchFromStore]);

  const handleClear = useCallback(() => {
    setInputValue("");
    setSearch("");
    setPage(1);
  }, [setSearch, setPage]);

  const handleTimeframeChange = useCallback(
    (value: string) => {
      setTimeframe(value === "all" ? "" : value);
      setPage(1);
    },
    [setTimeframe, setPage]
  );

  return (
    <div className="flex items-center gap-2">
      <Select
        value={timeframe || "all"}
        onValueChange={handleTimeframeChange}
      >
        <SelectTrigger className="w-[150px]">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {TIMEFRAME_OPTIONS.map((opt) => (
            <SelectItem key={opt.value} value={opt.value}>
              {opt.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <Input
        type="search"
        placeholder="Search by name or ticker..."
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        className="max-w-sm"
        aria-label="Search alerts"
      />
      {(inputValue || timeframe) && (
        <button
          type="button"
          onClick={() => {
            handleClear();
            setTimeframe("");
          }}
          className="text-xs text-muted-foreground hover:text-foreground underline"
        >
          Clear
        </button>
      )}
    </div>
  );
}

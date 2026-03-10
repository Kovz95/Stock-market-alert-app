"use client";

import { useAtomValue, useSetAtom } from "jotai";
import { useCallback, useEffect, useRef, useState } from "react";
import { Input } from "@/components/ui/input";
import {
  alertsPageAtom,
  alertsSearchAtom,
} from "@/lib/store/alerts";

const DEBOUNCE_MS = 300;

export function AlertsSearchBar() {
  const searchFromStore = useAtomValue(alertsSearchAtom);
  const setSearch = useSetAtom(alertsSearchAtom);
  const setPage = useSetAtom(alertsPageAtom);

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

  return (
    <div className="flex items-center gap-2">
      <Input
        type="search"
        placeholder="Search by name or ticker..."
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        className="max-w-sm"
        aria-label="Search alerts"
      />
      {inputValue && (
        <button
          type="button"
          onClick={handleClear}
          className="text-xs text-muted-foreground hover:text-foreground underline"
        >
          Clear
        </button>
      )}
    </div>
  );
}

"use client";

import { atom } from "jotai";
import { atomWithStorage } from "jotai/utils";
import {
  defaultStockDatabaseFilters,
  type StockDatabaseFiltersState,
} from "@/app/database/stock/_components";

const STOCK_DATABASE_FILTERS_KEY = "stock-database-filters";

function isStockDatabaseFiltersState(value: unknown): value is StockDatabaseFiltersState {
  if (!value || typeof value !== "object") return false;
  const o = value as Record<string, unknown>;
  return (
    Array.isArray(o.countries) &&
    Array.isArray(o.exchanges) &&
    (o.assetType === "All" || o.assetType === "Stocks" || o.assetType === "ETFs") &&
    Array.isArray(o.economies) &&
    Array.isArray(o.sectors) &&
    Array.isArray(o.subsectors) &&
    Array.isArray(o.industryGroups) &&
    Array.isArray(o.industries) &&
    Array.isArray(o.subindustries) &&
    Array.isArray(o.issuers) &&
    Array.isArray(o.assetClasses) &&
    Array.isArray(o.focuses) &&
    Array.isArray(o.niches)
  );
}

const filtersStorage = {
  getItem: (key: string, initialValue: StockDatabaseFiltersState): StockDatabaseFiltersState => {
    if (typeof window === "undefined") return initialValue;
    try {
      const raw = localStorage.getItem(key);
      if (raw == null) return initialValue;
      const value = JSON.parse(raw) as unknown;
      return isStockDatabaseFiltersState(value) ? value : initialValue;
    } catch {
      return initialValue;
    }
  },
  setItem: (key: string, value: StockDatabaseFiltersState) => {
    if (typeof window === "undefined") return;
    localStorage.setItem(key, JSON.stringify(value));
  },
  removeItem: (key: string) => {
    if (typeof window === "undefined") return;
    localStorage.removeItem(key);
  },
};

/** Persisted filters for the stock database page (sidebar). */
export const stockDatabaseFiltersAtom = atomWithStorage<StockDatabaseFiltersState>(
  STOCK_DATABASE_FILTERS_KEY,
  defaultStockDatabaseFilters,
  filtersStorage,
  { getOnInit: true }
);

/** Immediate search input (table search field). */
export const stockDatabaseSearchInputAtom = atom("");

/** Debounced search term used for filtering the list. */
export const stockDatabaseSearchTermAtom = atom("");

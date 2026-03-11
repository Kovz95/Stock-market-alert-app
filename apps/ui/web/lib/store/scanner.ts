"use client";

import { atom } from "jotai";
import { atomWithStorage } from "jotai/utils";
import type { ScanMatch } from "../../../../../gen/ts/price/v1/price";
import {
  defaultStockDatabaseFilters,
  type StockDatabaseFiltersState,
} from "@/app/database/stock/_components";

const SCANNER_PRESETS_KEY = "scanner_presets";

export type ScannerTimeframe = "1h" | "1d" | "1wk";

export type SavedPreset = {
  name: string;
  conditions: string[];
  combinationLogic: string;
  timeframe: string;
  filters: StockDatabaseFiltersState;
  portfolioId: string;
  lookbackDays: number;
  savedAt: string;
};

function isSavedPresetArray(value: unknown): value is SavedPreset[] {
  return Array.isArray(value) && value.every(
    (item) =>
      item &&
      typeof item === "object" &&
      typeof item.name === "string" &&
      Array.isArray(item.conditions) &&
      typeof item.combinationLogic === "string" &&
      typeof item.timeframe === "string" &&
      typeof item.filters === "object" &&
      typeof item.portfolioId === "string" &&
      typeof item.lookbackDays === "number" &&
      typeof item.savedAt === "string"
  );
}

const validatedPresetsStorage = {
  getItem: (key: string, initialValue: SavedPreset[]): SavedPreset[] => {
    if (typeof window === "undefined") return initialValue;
    try {
      const raw = localStorage.getItem(key);
      if (raw == null) return initialValue;
      const value = JSON.parse(raw) as unknown;
      return isSavedPresetArray(value) ? value : initialValue;
    } catch {
      return initialValue;
    }
  },
  setItem: (key: string, value: SavedPreset[]) => {
    if (typeof window === "undefined") return;
    localStorage.setItem(key, JSON.stringify(value));
  },
  removeItem: (key: string) => {
    if (typeof window === "undefined") return;
    localStorage.removeItem(key);
  },
};

// ─── Filter & portfolio ─────────────────────────────────────────────────────
export const scannerFiltersAtom = atom<StockDatabaseFiltersState>(defaultStockDatabaseFilters);
export const scannerPortfolioIdAtom = atom<string>("All");

// ─── Conditions ──────────────────────────────────────────────────────────────
export const scannerConditionsAtom = atom<string[]>([]);
export const scannerCombinationLogicAtom = atom<string>("AND");

// ─── Scan params ─────────────────────────────────────────────────────────────
export const scannerTimeframeAtom = atom<ScannerTimeframe>("1d");
export const scannerLookbackDaysAtom = atom<number>(0);
export const scannerLookbackInputAtom = atom<string>("0");

// ─── Scan results (ephemeral) ─────────────────────────────────────────────────
export const scannerResultsAtom = atom<ScanMatch[] | null>(null);
export const scannerScanningAtom = atom<boolean>(false);
export const scannerScanErrorAtom = atom<string | null>(null);
export const scannerScanProgressAtom = atom<{
  batch: number;
  totalBatches: number;
} | null>(null);

// ─── Presets ─────────────────────────────────────────────────────────────────
export const scannerPresetNameAtom = atom<string>("");
export const scannerSelectedPresetAtom = atom<string>("");
export const scannerPresetsAtom = atomWithStorage<SavedPreset[]>(
  SCANNER_PRESETS_KEY,
  [],
  validatedPresetsStorage,
  { getOnInit: true }
);

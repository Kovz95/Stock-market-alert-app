"use client";

import { atom } from "jotai";
import {
  emptyIndustryFilters,
  emptyEtfFilters,
  type ConditionEntry,
  type IndustryFilters,
  type EtfFilters,
  type AssetType,
} from "@/app/alerts/add/_components/types";
import { DEFAULT_TIMEFRAME, DEFAULT_COUNTRY } from "@/app/alerts/add/_components/constants";

export type BulkProgress = {
  creating: boolean;
  created: number;
  skipped: number;
  failed: number;
  total: number;
};

// ─── Basic settings ───────────────────────────────────────────────────────────
export const addAlertNameAtom = atom<string>("");
export const addAlertActionAtom = atom<"Buy" | "Sell">("Buy");
export const addAlertTimeframeAtom = atom<string>(DEFAULT_TIMEFRAME);
export const addAlertExchangesAtom = atom<string[]>([]);
export const addAlertCountryAtom = atom<string>(DEFAULT_COUNTRY);
export const addAlertAssetTypeAtom = atom<AssetType>("All");

// ─── Filters ─────────────────────────────────────────────────────────────────
export const addAlertIndustryFiltersAtom = atom<IndustryFilters>(emptyIndustryFilters);
export const addAlertEtfFiltersAtom = atom<EtfFilters>(emptyEtfFilters);

// ─── Ticker / ratio ───────────────────────────────────────────────────────────
export const addAlertIsRatioAtom = atom<boolean>(false);
export const addAlertTickerAtom = atom<string>("");
export const addAlertStockNameAtom = atom<string>("");
export const addAlertTicker1Atom = atom<string>("");
export const addAlertTicker2Atom = atom<string>("");
export const addAlertStockName1Atom = atom<string>("");
export const addAlertStockName2Atom = atom<string>("");
export const addAlertAdjustmentMethodAtom = atom<string>("");

// ─── Conditions ───────────────────────────────────────────────────────────────
export const addAlertConditionsAtom = atom<ConditionEntry[]>([]);
export const addAlertCombinationLogicAtom = atom<"AND" | "OR">("AND");

// ─── Multi / mixed timeframe ──────────────────────────────────────────────────
export const addAlertEnableMultiTimeframeAtom = atom<boolean>(false);
export const addAlertComparisonTimeframeAtom = atom<string>("1wk");
export const addAlertEnableMixedTimeframeAtom = atom<boolean>(false);

// ─── UI-only state ────────────────────────────────────────────────────────────
export const addAlertBulkModeAtom = atom<boolean>(false);
export const addAlertBulkTickersTextAtom = atom<string>("");
export const addAlertApplyToFilteredAtom = atom<boolean>(false);
export const addAlertFilteredSymbolCountAtom = atom<number>(0);
export const addAlertBulkProgressAtom = atom<BulkProgress | null>(null);

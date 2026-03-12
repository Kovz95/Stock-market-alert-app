"use client";

import { atom } from "jotai";
import type { PriceRowData } from "@/actions/price-database-actions";
import {
  defaultFilters,
  type PriceFiltersState,
} from "@/app/price-database/_components";

/** Filter state for the price database page (sidebar). */
export const priceDatabaseFiltersAtom = atom<PriceFiltersState>(
  defaultFilters(null, null)
);

/** Loaded price data (result of Load Data). */
export const priceDatabaseLoadedDataAtom = atom<PriceRowData[]>([]);

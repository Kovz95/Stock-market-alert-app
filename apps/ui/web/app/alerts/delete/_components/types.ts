import type { StockDatabaseFiltersState } from "@/app/database/stock/_components/types";

export type DeleteAlertsFiltersState = StockDatabaseFiltersState & {
  searchText: string;
  nameSearch: string;
  conditionType: string;
  conditionSearch: string;
  timeframes: string[];
  triggeredFilter: string;
};

export const defaultDeleteAlertsFilters: DeleteAlertsFiltersState = {
  countries: [],
  exchanges: [],
  assetType: "All",
  economies: [],
  sectors: [],
  subsectors: [],
  industryGroups: [],
  industries: [],
  subindustries: [],
  issuers: [],
  assetClasses: [],
  focuses: [],
  niches: [],
  searchText: "",
  nameSearch: "",
  conditionType: "All",
  conditionSearch: "",
  timeframes: [],
  triggeredFilter: "All",
};

export const CONDITION_TYPE_OPTIONS = [
  "All",
  "RSI",
  "MACD",
  "SMA",
  "EMA",
  "HMA",
  "BB",
  "ATR",
  "CCI",
  "ROC",
  "Williams %R",
  "Close",
  "Open",
  "High",
  "Low",
  "HARSI",
  "SROCST",
  "Breakout",
  "Slope",
  "Price",
] as const;

export const TIMEFRAME_OPTIONS = [
  "Weekly",
  "Daily",
  "Hourly",
  "15 minutes",
  "5 minutes",
  "1 minute",
] as const;

export const TRIGGERED_FILTER_OPTIONS = [
  "All",
  "Never",
  "Today",
  "This Week",
  "This Month",
  "This Year",
] as const;

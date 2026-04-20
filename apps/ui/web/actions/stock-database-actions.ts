"use server";

import { priceClient } from "@/lib/grpc/channel";
import type { FullStockMetadata, TickerDeletionCounts as ProtoTickerDeletionCounts } from "../../../../gen/ts/price/v1/price";

export type TickerDeletionCounts = {
  stockMetadata: number;
  tickerMetadata: number;
  dailyPrices: number;
  hourlyPrices: number;
  weeklyPrices: number;
  continuousPrices: number;
  dailyMoveStats: number;
  futuresMetadata: number;
  alertsDirect: number;
  alertsRatio: number;
  alertAudits: number;
  portfolioStocks: number;
};

export type PreviewDeleteTickerResult = {
  ticker: string;
  exists: boolean;
  counts: TickerDeletionCounts;
};

export type DeleteTickerResult = {
  success: boolean;
  errorMessage?: string;
  ticker: string;
  counts: TickerDeletionCounts;
};

function zeroCounts(): TickerDeletionCounts {
  return {
    stockMetadata: 0,
    tickerMetadata: 0,
    dailyPrices: 0,
    hourlyPrices: 0,
    weeklyPrices: 0,
    continuousPrices: 0,
    dailyMoveStats: 0,
    futuresMetadata: 0,
    alertsDirect: 0,
    alertsRatio: 0,
    alertAudits: 0,
    portfolioStocks: 0,
  };
}

function toDeletionCounts(proto: ProtoTickerDeletionCounts | undefined): TickerDeletionCounts {
  if (!proto) return zeroCounts();
  return {
    stockMetadata: Number(proto.stockMetadata),
    tickerMetadata: Number(proto.tickerMetadata),
    dailyPrices: Number(proto.dailyPrices),
    hourlyPrices: Number(proto.hourlyPrices),
    weeklyPrices: Number(proto.weeklyPrices),
    continuousPrices: Number(proto.continuousPrices),
    dailyMoveStats: Number(proto.dailyMoveStats),
    futuresMetadata: Number(proto.futuresMetadata),
    alertsDirect: Number(proto.alertsDirect),
    alertsRatio: Number(proto.alertsRatio),
    alertAudits: Number(proto.alertAudits),
    portfolioStocks: Number(proto.portfolioStocks),
  };
}

export async function previewDeleteTicker(ticker: string): Promise<PreviewDeleteTickerResult> {
  const res = await priceClient.previewDeleteTicker({ ticker });
  return {
    ticker: res.ticker,
    exists: res.exists,
    counts: toDeletionCounts(res.counts),
  };
}

export async function deleteTicker(ticker: string): Promise<DeleteTickerResult> {
  const res = await priceClient.deleteTicker({ ticker });
  return {
    success: res.success,
    errorMessage: res.errorMessage || undefined,
    ticker: res.ticker,
    counts: toDeletionCounts(res.counts),
  };
}

/**
 * One row of full stock metadata (table columns + flattened ETF fields from raw_payload).
 * Matches the shape returned by the price service GetFullStockMetadata RPC.
 */
export type FullStockMetadataRow = {
  symbol: string;
  name: string;
  exchange: string;
  country: string;
  isin: string;
  assetType: string;
  rbicsEconomy: string;
  rbicsSector: string;
  rbicsSubsector: string;
  rbicsIndustryGroup: string;
  rbicsIndustry: string;
  rbicsSubindustry: string;
  closingPrice?: number;
  marketValue?: number;
  sales?: number;
  avgDailyVolume?: number;
  dataSource: string;
  lastUpdated?: Date;
  etfIssuer: string;
  etfAssetClass: string;
  etfFocus: string;
  etfNiche: string;
  expenseRatio?: number;
  aum?: number;
};

function mapItem(item: Record<string, unknown>): FullStockMetadataRow {
  const get = (key: string) => item[key] ?? item[key.replace(/([A-Z])/g, "_$1").toLowerCase().replace(/^_/, "")];
  const str = (key: string) => (get(key) != null ? String(get(key)) : "") as string;
  const num = (key: string) => (get(key) != null ? Number(get(key)) : undefined) as number | undefined;
  const date = (key: string) => {
    const v = get(key);
    if (v instanceof Date) return v;
    if (typeof v === "string") return new Date(v);
    return undefined;
  };
  return {
    symbol: str("symbol"),
    name: str("name"),
    exchange: str("exchange"),
    country: str("country"),
    isin: str("isin"),
    assetType: str("assetType") || str("asset_type"),
    rbicsEconomy: str("rbicsEconomy") || str("rbics_economy"),
    rbicsSector: str("rbicsSector") || str("rbics_sector"),
    rbicsSubsector: str("rbicsSubsector") || str("rbics_subsector"),
    rbicsIndustryGroup: str("rbicsIndustryGroup") || str("rbics_industry_group"),
    rbicsIndustry: str("rbicsIndustry") || str("rbics_industry"),
    rbicsSubindustry: str("rbicsSubindustry") || str("rbics_subindustry"),
    closingPrice: num("closingPrice") ?? num("closing_price"),
    marketValue: num("marketValue") ?? num("market_value"),
    sales: num("sales"),
    avgDailyVolume: num("avgDailyVolume") ?? num("avg_daily_volume"),
    dataSource: str("dataSource") || str("data_source"),
    lastUpdated: date("lastUpdated") ?? date("last_updated"),
    etfIssuer: str("etfIssuer") || str("etf_issuer"),
    etfAssetClass: str("etfAssetClass") || str("etf_asset_class"),
    etfFocus: str("etfFocus") || str("etf_focus"),
    etfNiche: str("etfNiche") || str("etf_niche"),
    expenseRatio: num("expenseRatio") ?? num("expense_ratio"),
    aum: num("aum"),
  };
}

export type GetFullStockMetadataResult =
  | { data: FullStockMetadataRow[] }
  | { error: string };

export async function getFullStockMetadata(): Promise<GetFullStockMetadataResult> {
  try {
    const res = await priceClient.getFullStockMetadata({});
    return { data: res.items ?? [] };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return { error: message };
  }
}

export type IndustryFilterValues = {
  economies?: string[];
  sectors?: string[];
  subsectors?: string[];
  industryGroups?: string[];
  industries?: string[];
  subindustries?: string[];
};

export type EtfFilterValues = {
  etfIssuers?: string[];
  assetClasses?: string[];
  etfFocuses?: string[];
  etfNiches?: string[];
};

export type CountSymbolsFilters = {
  exchanges?: string[];
  country?: string;
  assetType?: "All" | "Stocks" | "ETFs";
  industry?: IndustryFilterValues;
  etf?: EtfFilterValues;
};

export type CountSymbolsResult = {
  count: number;
  error?: string;
};

/** Country code → display name. Backend may store either; we match both when filtering. */
const COUNTRY_CODE_TO_NAME: Record<string, string> = {
  US: "United States",
  CA: "Canada",
  UK: "United Kingdom",
  JP: "Japan",
  DE: "Germany",
  FR: "France",
  AU: "Australia",
  CH: "Switzerland",
  NL: "Netherlands",
  IT: "Italy",
  ES: "Spain",
  SE: "Sweden",
  NO: "Norway",
  DK: "Denmark",
  FI: "Finland",
  BE: "Belgium",
  IE: "Ireland",
  PT: "Portugal",
  AT: "Austria",
  PL: "Poland",
  GR: "Greece",
  HU: "Hungary",
  CZ: "Czech Republic",
  TR: "Turkey",
  MX: "Mexico",
  HK: "Hong Kong",
  SG: "Singapore",
  TW: "Taiwan",
  MY: "Malaysia",
  KR: "South Korea",
  IN: "India",
  CN: "China",
  TH: "Thailand",
  ID: "Indonesia",
  PH: "Philippines",
  VN: "Vietnam",
};

function applyBaseFilters(
  items: FullStockMetadata[],
  filters: CountSymbolsFilters
): FullStockMetadata[] {
  let filtered = items;

  if (filters.exchanges && filters.exchanges.length > 0) {
    const validExchanges = filters.exchanges.filter(e => e !== "All");
    if (validExchanges.length > 0) {
      filtered = filtered.filter((item) => validExchanges.includes(item.exchange));
    }
  }

  if (filters.country && filters.country !== "All") {
    const code = filters.country;
    const name = COUNTRY_CODE_TO_NAME[code];
    filtered = filtered.filter((item) =>
      item.country === code || (name != null && item.country === name)
    );
  }

  // Asset type filter
  if (filters.assetType && filters.assetType !== "All") {
    if (filters.assetType === "Stocks") {
      filtered = filtered.filter(
        (item) => !item.assetType || item.assetType.toLowerCase() !== "etf"
      );
    } else if (filters.assetType === "ETFs") {
      filtered = filtered.filter(
        (item) => item.assetType && item.assetType.toLowerCase() === "etf"
      );
    }
  }

  return filtered;
}

function applyEtfFilters(
  items: FullStockMetadata[],
  etf?: EtfFilterValues
): FullStockMetadata[] {
  if (!etf) return items;
  let filtered = items;

  if (etf.etfIssuers && etf.etfIssuers.length > 0) {
    filtered = filtered.filter((item) => etf.etfIssuers!.includes(item.etfIssuer));
  }
  if (etf.assetClasses && etf.assetClasses.length > 0) {
    filtered = filtered.filter((item) => etf.assetClasses!.includes(item.etfAssetClass));
  }
  if (etf.etfFocuses && etf.etfFocuses.length > 0) {
    filtered = filtered.filter((item) => etf.etfFocuses!.includes(item.etfFocus));
  }
  if (etf.etfNiches && etf.etfNiches.length > 0) {
    filtered = filtered.filter((item) => etf.etfNiches!.includes(item.etfNiche));
  }

  return filtered;
}

type RbicsGetter = (item: FullStockMetadata) => string;

const INDUSTRY_FIELD_MAP: [keyof IndustryFilterValues, RbicsGetter][] = [
  ["economies", (i) => i.rbicsEconomy],
  ["sectors", (i) => i.rbicsSector],
  ["subsectors", (i) => i.rbicsSubsector],
  ["industryGroups", (i) => i.rbicsIndustryGroup],
  ["industries", (i) => i.rbicsIndustry],
  ["subindustries", (i) => i.rbicsSubindustry],
];

function applyIndustryFilters(
  items: FullStockMetadata[],
  industry?: IndustryFilterValues
): FullStockMetadata[] {
  if (!industry) return items;
  let filtered = items;

  for (const [key, getter] of INDUSTRY_FIELD_MAP) {
    const vals = industry[key];
    if (vals && vals.length > 0) {
      filtered = filtered.filter((item) => vals.includes(getter(item)));
    }
  }

  return filtered;
}

/**
 * Count symbols that match the given exchanges, country, and industry filters.
 */
export async function countSymbolsByFilters(
  filters: CountSymbolsFilters
): Promise<CountSymbolsResult> {
  try {
    const res = await priceClient.getFullStockMetadata({});
    const items = res.items ?? [];
    const base = applyBaseFilters(items, filters);
    const withIndustry = applyIndustryFilters(base, filters.industry);
    const filtered = applyEtfFilters(withIndustry, filters.etf);
    return { count: filtered.length };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return { count: 0, error: message };
  }
}

export type SymbolInfo = {
  symbol: string;
  name: string;
  exchange: string;
  country: string;
};

export type GetSymbolsByFiltersResult = {
  symbols: SymbolInfo[];
  error?: string;
};

/**
 * Get actual symbols that match the given exchanges, country, and industry filters.
 */
export async function getSymbolsByFilters(
  filters: CountSymbolsFilters
): Promise<GetSymbolsByFiltersResult> {
  try {
    const res = await priceClient.getFullStockMetadata({});
    const items = res.items ?? [];
    const base = applyBaseFilters(items, filters);
    const withIndustry = applyIndustryFilters(base, filters.industry);
    const filtered = applyEtfFilters(withIndustry, filters.etf);

    const symbols: SymbolInfo[] = filtered.map((item) => ({
      symbol: String(item.symbol || item["symbol"] || ""),
      name: String(item.name || item["name"] || ""),
      exchange: String(item.exchange || item["exchange"] || ""),
      country: String(item.country || item["country"] || ""),
    }));

    return { symbols };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return { symbols: [], error: message };
  }
}

/**
 * RBICS filter options returned by getIndustryFilterOptions.
 * Each level contains the distinct values available given the parent selections.
 */
export type IndustryFilterOptions = {
  economies: string[];
  sectors: string[];
  subsectors: string[];
  industryGroups: string[];
  industries: string[];
  subindustries: string[];
  error?: string;
};

/**
 * Get available RBICS industry filter options with cascading logic.
 * Selecting a parent level narrows the available options for child levels.
 */
export async function getIndustryFilterOptions(
  baseFilters: { exchanges?: string[]; country?: string },
  selected: IndustryFilterValues
): Promise<IndustryFilterOptions> {
  try {
    const res = await priceClient.getFullStockMetadata({});
    const allItems = res.items ?? [];
    let items = applyBaseFilters(allItems, baseFilters);

    const unique = (arr: string[]) => [...new Set(arr.filter(Boolean))].sort();

    const economies = unique(items.map(i => i.rbicsEconomy));

    if (selected.economies && selected.economies.length > 0) {
      items = items.filter(i => selected.economies!.includes(i.rbicsEconomy));
    }
    const sectors = unique(items.map(i => i.rbicsSector));

    if (selected.sectors && selected.sectors.length > 0) {
      items = items.filter(i => selected.sectors!.includes(i.rbicsSector));
    }
    const subsectors = unique(items.map(i => i.rbicsSubsector));

    if (selected.subsectors && selected.subsectors.length > 0) {
      items = items.filter(i => selected.subsectors!.includes(i.rbicsSubsector));
    }
    const industryGroups = unique(items.map(i => i.rbicsIndustryGroup));

    if (selected.industryGroups && selected.industryGroups.length > 0) {
      items = items.filter(i => selected.industryGroups!.includes(i.rbicsIndustryGroup));
    }
    const industries = unique(items.map(i => i.rbicsIndustry));

    if (selected.industries && selected.industries.length > 0) {
      items = items.filter(i => selected.industries!.includes(i.rbicsIndustry));
    }
    const subindustries = unique(items.map(i => i.rbicsSubindustry));

    return { economies, sectors, subsectors, industryGroups, industries, subindustries };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return { economies: [], sectors: [], subsectors: [], industryGroups: [], industries: [], subindustries: [], error: message };
  }
}

/**
 * ETF filter options returned by getEtfFilterOptions.
 */
export type EtfFilterOptions = {
  etfIssuers: string[];
  assetClasses: string[];
  etfFocuses: string[];
  etfNiches: string[];
  error?: string;
};

/**
 * Get available ETF filter options with cascading logic.
 */
export async function getEtfFilterOptions(
  baseFilters: { exchanges?: string[]; country?: string },
  selected: EtfFilterValues
): Promise<EtfFilterOptions> {
  try {
    const res = await priceClient.getFullStockMetadata({});
    const allItems = res.items ?? [];
    // Only include ETFs
    let items = applyBaseFilters(allItems, { ...baseFilters, assetType: "ETFs" });

    const unique = (arr: string[]) => [...new Set(arr.filter(Boolean))].sort();

    const etfIssuers = unique(items.map(i => i.etfIssuer));

    if (selected.etfIssuers && selected.etfIssuers.length > 0) {
      items = items.filter(i => selected.etfIssuers!.includes(i.etfIssuer));
    }
    const assetClasses = unique(items.map(i => i.etfAssetClass));

    if (selected.assetClasses && selected.assetClasses.length > 0) {
      items = items.filter(i => selected.assetClasses!.includes(i.etfAssetClass));
    }
    const etfFocuses = unique(items.map(i => i.etfFocus));

    if (selected.etfFocuses && selected.etfFocuses.length > 0) {
      items = items.filter(i => selected.etfFocuses!.includes(i.etfFocus));
    }
    const etfNiches = unique(items.map(i => i.etfNiche));

    return { etfIssuers, assetClasses, etfFocuses, etfNiches };
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    return { etfIssuers: [], assetClasses: [], etfFocuses: [], etfNiches: [], error: message };
  }
}

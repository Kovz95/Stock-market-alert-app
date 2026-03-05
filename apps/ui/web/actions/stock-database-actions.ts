"use server";

import { priceClient } from "@/lib/grpc/channel";

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

export type CountSymbolsFilters = {
  exchanges?: string[];
  country?: string;
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

/**
 * Count symbols that match the given exchanges and country filters.
 * "All" in exchanges array means no filter for exchanges.
 * Country filter matches both code (e.g. "US") and display name (e.g. "United States") since the backend may store either.
 */
export async function countSymbolsByFilters(
  filters: CountSymbolsFilters
): Promise<CountSymbolsResult> {
  try {
    const res = await priceClient.getFullStockMetadata({});
    const items = res.items ?? [];

    let filtered = items;

    // Filter by exchanges if not "All" and array is not empty
    if (filters.exchanges && filters.exchanges.length > 0) {
      const validExchanges = filters.exchanges.filter(e => e !== "All");
      if (validExchanges.length > 0) {
        filtered = filtered.filter((item) => {
          const exchange = item.exchange || item["exchange"] as string;
          return validExchanges.includes(exchange);
        });
      }
    }

    // Filter by country if not "All" (match both code and display name; backend may store either)
    if (filters.country && filters.country !== "All") {
      const code = filters.country;
      const name = COUNTRY_CODE_TO_NAME[code];
      filtered = filtered.filter((item) => {
        const itemCountry = (item.country ?? item["country"] ?? "") as string;
        return itemCountry === code || (name != null && itemCountry === name);
      });
    }

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
 * Get actual symbols that match the given exchanges and country filters.
 */
export async function getSymbolsByFilters(
  filters: CountSymbolsFilters
): Promise<GetSymbolsByFiltersResult> {
  try {
    const res = await priceClient.getFullStockMetadata({});
    const items = res.items ?? [];

    let filtered = items;

    // Filter by exchanges if not "All" and array is not empty
    if (filters.exchanges && filters.exchanges.length > 0) {
      const validExchanges = filters.exchanges.filter(e => e !== "All");
      if (validExchanges.length > 0) {
        filtered = filtered.filter((item) => {
          const exchange = item.exchange || item["exchange"] as string;
          return validExchanges.includes(exchange);
        });
      }
    }

    // Filter by country if not "All" (match both code and display name; backend may store either)
    if (filters.country && filters.country !== "All") {
      const code = filters.country;
      const name = COUNTRY_CODE_TO_NAME[code];
      filtered = filtered.filter((item) => {
        const itemCountry = (item.country ?? item["country"] ?? "") as string;
        return itemCountry === code || (name != null && itemCountry === name);
      });
    }

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

import type { FullStockMetadataRow } from "@/actions/stock-database-actions";

export type AssetTypeFilter = "All" | "Stocks" | "ETFs";

export type StockDatabaseFiltersState = {
  countries: string[];
  exchanges: string[];
  assetType: AssetTypeFilter;
  // Stock (RBICS)
  economies: string[];
  sectors: string[];
  subsectors: string[];
  industryGroups: string[];
  industries: string[];
  subindustries: string[];
  // ETF
  issuers: string[];
  assetClasses: string[];
  focuses: string[];
  niches: string[];
};

export const defaultStockDatabaseFilters: StockDatabaseFiltersState = {
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
};

export function applyStockDatabaseFilters(
  rows: FullStockMetadataRow[],
  filters: StockDatabaseFiltersState
): FullStockMetadataRow[] {
  let out = rows;

  if (filters.countries.length > 0) {
    out = out.filter((r) => r.country && filters.countries.includes(r.country));
  }
  if (filters.exchanges.length > 0) {
    out = out.filter((r) => r.exchange && filters.exchanges.includes(r.exchange));
  }
  if (filters.assetType === "Stocks") {
    out = out.filter((r) => r.assetType === "Stock");
  } else if (filters.assetType === "ETFs") {
    out = out.filter((r) => r.assetType === "ETF" || !!r.etfIssuer);
  }

  if (filters.assetType === "All" || filters.assetType === "Stocks") {
    if (filters.economies.length > 0) {
      out = out.filter((r) => r.rbicsEconomy && filters.economies.includes(r.rbicsEconomy));
    }
    if (filters.sectors.length > 0) {
      out = out.filter((r) => r.rbicsSector && filters.sectors.includes(r.rbicsSector));
    }
    if (filters.subsectors.length > 0) {
      out = out.filter((r) => r.rbicsSubsector && filters.subsectors.includes(r.rbicsSubsector));
    }
    if (filters.industryGroups.length > 0) {
      out = out.filter((r) => r.rbicsIndustryGroup && filters.industryGroups.includes(r.rbicsIndustryGroup));
    }
    if (filters.industries.length > 0) {
      out = out.filter((r) => r.rbicsIndustry && filters.industries.includes(r.rbicsIndustry));
    }
    if (filters.subindustries.length > 0) {
      out = out.filter((r) => r.rbicsSubindustry && filters.subindustries.includes(r.rbicsSubindustry));
    }
  }

  if (filters.assetType === "All" || filters.assetType === "ETFs") {
    if (filters.issuers.length > 0) {
      out = out.filter((r) => r.etfIssuer && filters.issuers.includes(r.etfIssuer));
    }
    if (filters.assetClasses.length > 0) {
      out = out.filter((r) => r.etfAssetClass && filters.assetClasses.includes(r.etfAssetClass));
    }
    if (filters.focuses.length > 0) {
      out = out.filter((r) => r.etfFocus && filters.focuses.includes(r.etfFocus));
    }
    if (filters.niches.length > 0) {
      out = out.filter((r) => r.etfNiche && filters.niches.includes(r.etfNiche));
    }
  }

  return out;
}

export function searchRows(
  rows: FullStockMetadataRow[],
  searchTerm: string
): FullStockMetadataRow[] {
  const q = searchTerm.trim().toLowerCase();
  if (!q) return rows;
  return rows.filter((r) => {
    const s = [
      r.symbol,
      r.name,
      r.exchange,
      r.country,
      r.isin,
      r.assetType,
      r.rbicsEconomy,
      r.rbicsSector,
      r.rbicsSubsector,
      r.rbicsIndustryGroup,
      r.rbicsIndustry,
      r.rbicsSubindustry,
      r.etfIssuer,
      r.etfAssetClass,
      r.etfFocus,
      r.etfNiche,
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
    return s.includes(q);
  });
}

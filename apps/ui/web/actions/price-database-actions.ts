"use server";

import { priceClient } from "@/lib/grpc/channel";
import {
  DayFilter,
  Timeframe,
} from "../../../../gen/ts/price/v1/price";

export type StockMetadataItem = {
  symbol: string;
  name: string;
  exchange: string;
  isin: string;
};

export type DatabaseStatsData = {
  hourlyRecords: number;
  dailyRecords: number;
  weeklyRecords: number;
  hourlyTickers: number;
  dailyTickers: number;
  weeklyTickers: number;
  hourlyMin: string | null;
  hourlyMax: string | null;
  dailyMin: string | null;
  dailyMax: string | null;
  weeklyMin: string | null;
  weeklyMax: string | null;
};

export type PriceRowData = {
  ticker: string;
  time: string | null;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

export async function getStockMetadataMap(): Promise<StockMetadataItem[]> {
  const res = await priceClient.getStockMetadataMap({});
  return (res.items ?? []).map((item) => ({
    symbol: item.symbol,
    name: item.name ?? "",
    exchange: item.exchange ?? "",
    isin: item.isin ?? "",
  }));
}

export async function getDatabaseStats(): Promise<DatabaseStatsData> {
  const res = await priceClient.getDatabaseStats({});
  const s = res.stats;
  if (!s) {
    return {
      hourlyRecords: 0,
      dailyRecords: 0,
      weeklyRecords: 0,
      hourlyTickers: 0,
      dailyTickers: 0,
      weeklyTickers: 0,
      hourlyMin: null,
      hourlyMax: null,
      dailyMin: null,
      dailyMax: null,
      weeklyMin: null,
      weeklyMax: null,
    };
  }
  return {
    hourlyRecords: Number(s.hourlyRecords ?? 0),
    dailyRecords: Number(s.dailyRecords ?? 0),
    weeklyRecords: Number(s.weeklyRecords ?? 0),
    hourlyTickers: Number(s.hourlyTickers ?? 0),
    dailyTickers: Number(s.dailyTickers ?? 0),
    weeklyTickers: Number(s.weeklyTickers ?? 0),
    hourlyMin: s.hourlyMin ? s.hourlyMin.toISOString() : null,
    hourlyMax: s.hourlyMax ? s.hourlyMax.toISOString() : null,
    dailyMin: s.dailyMin ? s.dailyMin.toISOString() : null,
    dailyMax: s.dailyMax ? s.dailyMax.toISOString() : null,
    weeklyMin: s.weeklyMin ? s.weeklyMin.toISOString() : null,
    weeklyMax: s.weeklyMax ? s.weeklyMax.toISOString() : null,
  };
}

export type LoadPriceDataParams = {
  timeframe: number; // Timeframe enum value (e.g. 1=Hourly, 2=Daily, 3=Weekly)
  tickers: string[];
  startDate?: Date | null;
  endDate?: Date | null;
  maxRows: number;
  dayFilter: number; // DayFilter enum value (e.g. 1=All, 2=Weekdays, 3=Weekends)
};

export async function loadPriceData(
  params: LoadPriceDataParams
): Promise<PriceRowData[]> {
  const res = await priceClient.loadPriceData({
    timeframe: params.timeframe as Timeframe,
    tickers: params.tickers ?? [],
    startDate: params.startDate ?? undefined,
    endDate: params.endDate ?? undefined,
    maxRows: params.maxRows || 5000,
    dayFilter: (params.dayFilter ?? DayFilter.DAY_FILTER_ALL) as DayFilter,
  });
  return (res.rows ?? []).map((row) => ({
    ticker: row.ticker,
    time: row.time ? row.time.toISOString() : null,
    open: Number(row.open ?? 0),
    high: Number(row.high ?? 0),
    low: Number(row.low ?? 0),
    close: Number(row.close ?? 0),
    volume: Number(row.volume ?? 0),
  }));
}

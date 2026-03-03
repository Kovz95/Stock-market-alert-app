"use server";

import { Metadata } from "nice-grpc-common";
import { priceClient } from "@/lib/grpc/channel";
import type {
  RunScanRequest,
  RunScanResponse,
  ScanMatch,
  SymbolFilter,
  Timeframe,
} from "../../../../gen/ts/price/v1/price";


export type RunScanResult =
  | { data: ScanMatch[] }
  | { error: string };

const RUN_SCAN_TIMEOUT_MS = 10 * 60 * 1000; // 10 minutes

export async function runScan(request: {
  timeframe: Timeframe;
  conditions: string[];
  combinationLogic: string;
  tickers: string[];
  symbolFilter?: SymbolFilter;
  maxTickers?: number;
  lookbackDays?: number;
}): Promise<RunScanResult> {
  try {
    const req: RunScanRequest = {
      timeframe: request.timeframe,
      conditions: request.conditions.filter(Boolean),
      combinationLogic: request.combinationLogic || "AND",
      tickers: request.tickers ?? [],
      symbolFilter: request.symbolFilter,
      maxTickers: request.maxTickers ?? 20000,
      lookbackDays: request.lookbackDays ?? 0,
    };
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), RUN_SCAN_TIMEOUT_MS);
    const res: RunScanResponse = await priceClient.runScan(req, {
      signal: controller.signal,
      metadata: Metadata({ "grpc-timeout": "600S" }),
    });
    clearTimeout(timeoutId);
    if (res.errorMessage) {
      return { error: res.errorMessage };
    }
    return { data: res.matches ?? [] };
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    return { error: message };
  }
}

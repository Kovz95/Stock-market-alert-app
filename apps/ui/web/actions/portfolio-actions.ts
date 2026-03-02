"use server";

import { alertClient } from "@/lib/grpc/channel";
import type { Portfolio } from "../../../../gen/ts/alert/v1/alert";

export async function listPortfolios(): Promise<Portfolio[]> {
  const response = await alertClient.listPortfolios({});
  return response.portfolios ?? [];
}

export async function getPortfolio(
  portfolioId: string
): Promise<Portfolio | undefined> {
  const response = await alertClient.getPortfolio({ portfolioId });
  return response.portfolio ?? undefined;
}

export async function createPortfolio(
  name: string,
  discordWebhook: string = ""
): Promise<Portfolio | undefined> {
  const response = await alertClient.createPortfolio({
    name: name.trim(),
    discordWebhook: discordWebhook.trim(),
  });
  return response.portfolio ?? undefined;
}

export async function updatePortfolio(params: {
  portfolioId: string;
  name: string;
  discordWebhook: string;
  enabled: boolean;
}): Promise<Portfolio | undefined> {
  const response = await alertClient.updatePortfolio({
    portfolioId: params.portfolioId,
    name: params.name.trim(),
    discordWebhook: params.discordWebhook.trim(),
    enabled: params.enabled,
  });
  return response.portfolio ?? undefined;
}

export async function deletePortfolio(portfolioId: string): Promise<void> {
  await alertClient.deletePortfolio({ portfolioId });
}

export async function addStocksToPortfolio(
  portfolioId: string,
  tickers: string[]
): Promise<Portfolio | undefined> {
  const response = await alertClient.addStocksToPortfolio({
    portfolioId,
    tickers,
  });
  return response.portfolio ?? undefined;
}

export async function removeStocksFromPortfolio(
  portfolioId: string,
  tickers: string[]
): Promise<Portfolio | undefined> {
  const response = await alertClient.removeStocksFromPortfolio({
    portfolioId,
    tickers,
  });
  return response.portfolio ?? undefined;
}

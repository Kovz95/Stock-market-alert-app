"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  listPortfolios,
  getPortfolio,
  createPortfolio,
  updatePortfolio,
  deletePortfolio,
  addStocksToPortfolio,
  removeStocksFromPortfolio,
} from "@/actions/portfolio-actions";

export const PORTFOLIOS_KEY = ["portfolios"] as const;

export function usePortfoliosList() {
  return useQuery({
    queryKey: [...PORTFOLIOS_KEY],
    queryFn: listPortfolios,
  });
}

export function usePortfolio(portfolioId: string) {
  return useQuery({
    queryKey: [...PORTFOLIOS_KEY, portfolioId],
    queryFn: () => getPortfolio(portfolioId),
    enabled: !!portfolioId,
  });
}

export function useCreatePortfolio() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ name, discordWebhook }: { name: string; discordWebhook?: string }) =>
      createPortfolio(name, discordWebhook ?? ""),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: PORTFOLIOS_KEY });
    },
  });
}

export function useUpdatePortfolio() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: updatePortfolio,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: PORTFOLIOS_KEY });
    },
  });
}

export function useDeletePortfolio() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: deletePortfolio,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: PORTFOLIOS_KEY });
    },
  });
}

export function useAddStocks() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ portfolioId, tickers }: { portfolioId: string; tickers: string[] }) =>
      addStocksToPortfolio(portfolioId, tickers),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: PORTFOLIOS_KEY });
    },
  });
}

export function useRemoveStocks() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ portfolioId, tickers }: { portfolioId: string; tickers: string[] }) =>
      removeStocksFromPortfolio(portfolioId, tickers),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: PORTFOLIOS_KEY });
    },
  });
}

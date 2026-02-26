"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getHourlyDiscordConfig,
  copyDailyToHourly,
  updateHourlyChannelWebhook,
  resolveHourlyChannelForTicker,
  sendHourlyTestMessage,
  getDailyDiscordConfig,
  copyBaseToDaily,
  updateDailyChannelWebhook,
  resolveDailyChannelForTicker,
  sendDailyTestMessage,
  getWeeklyDiscordConfig,
  copyBaseToWeekly,
  updateWeeklyChannelWebhook,
  resolveWeeklyChannelForTicker,
  sendWeeklyTestMessage,
  type HourlyDiscordConfig,
  type ResolveHourlyResult,
} from "@/actions/discord-hourly-actions";

export const HOURLY_DISCORD_KEY = ["discord", "hourly"] as const;
export const DAILY_DISCORD_KEY = ["discord", "daily"] as const;
export const WEEKLY_DISCORD_KEY = ["discord", "weekly"] as const;

export function useHourlyDiscordConfig() {
  return useQuery({
    queryKey: HOURLY_DISCORD_KEY,
    queryFn: getHourlyDiscordConfig,
  });
}

export function useCopyDailyToHourly() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: copyDailyToHourly,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: HOURLY_DISCORD_KEY });
    },
  });
}

export function useUpdateHourlyChannelWebhook() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      channelName,
      webhookUrl,
    }: {
      channelName: string;
      webhookUrl: string;
    }) => updateHourlyChannelWebhook(channelName, webhookUrl),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: HOURLY_DISCORD_KEY });
    },
  });
}

export function useResolveHourlyChannelForTicker() {
  return useMutation({
    mutationFn: (ticker: string) => resolveHourlyChannelForTicker(ticker),
  });
}

export function useSendHourlyTestMessage() {
  return useMutation({
    mutationFn: (channelName: string) => sendHourlyTestMessage(channelName),
  });
}

// Daily
export function useDailyDiscordConfig() {
  return useQuery({
    queryKey: DAILY_DISCORD_KEY,
    queryFn: getDailyDiscordConfig,
  });
}

export function useCopyBaseToDaily() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: copyBaseToDaily,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: DAILY_DISCORD_KEY });
    },
  });
}

export function useUpdateDailyChannelWebhook() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      channelName,
      webhookUrl,
    }: {
      channelName: string;
      webhookUrl: string;
    }) => updateDailyChannelWebhook(channelName, webhookUrl),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: DAILY_DISCORD_KEY });
    },
  });
}

export function useResolveDailyChannelForTicker() {
  return useMutation({
    mutationFn: (ticker: string) => resolveDailyChannelForTicker(ticker),
  });
}

export function useSendDailyTestMessage() {
  return useMutation({
    mutationFn: (channelName: string) => sendDailyTestMessage(channelName),
  });
}

// Weekly
export function useWeeklyDiscordConfig() {
  return useQuery({
    queryKey: WEEKLY_DISCORD_KEY,
    queryFn: getWeeklyDiscordConfig,
  });
}

export function useCopyBaseToWeekly() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: copyBaseToWeekly,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: WEEKLY_DISCORD_KEY });
    },
  });
}

export function useUpdateWeeklyChannelWebhook() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      channelName,
      webhookUrl,
    }: {
      channelName: string;
      webhookUrl: string;
    }) => updateWeeklyChannelWebhook(channelName, webhookUrl),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: WEEKLY_DISCORD_KEY });
    },
  });
}

export function useResolveWeeklyChannelForTicker() {
  return useMutation({
    mutationFn: (ticker: string) => resolveWeeklyChannelForTicker(ticker),
  });
}

export function useSendWeeklyTestMessage() {
  return useMutation({
    mutationFn: (channelName: string) => sendWeeklyTestMessage(channelName),
  });
}

export type { HourlyDiscordConfig, ResolveHourlyResult };

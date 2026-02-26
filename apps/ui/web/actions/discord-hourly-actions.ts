"use server";

import { discordClient } from "@/lib/grpc/channel";
import type { GetHourlyDiscordConfigResponse } from "../../../../gen/ts/discord/v1/discord";

/** Local type for hourly channel; matches proto HourlyChannelInfo to avoid runtime re-export in server actions. */
export type HourlyChannelInfo = {
  name: string;
  channelName: string;
  description: string;
  configured: boolean;
};

export type HourlyDiscordConfig = {
  enableIndustryRouting: boolean;
  logRoutingDecisions: boolean;
  channels: HourlyChannelInfo[];
  configuredCount: number;
  totalCount: number;
};

export async function getHourlyDiscordConfig(): Promise<HourlyDiscordConfig> {
  const res: GetHourlyDiscordConfigResponse =
    await discordClient.getHourlyDiscordConfig({});
  return {
    enableIndustryRouting: res.enableIndustryRouting,
    logRoutingDecisions: res.logRoutingDecisions,
    channels: res.channels ?? [],
    configuredCount: res.configuredCount ?? 0,
    totalCount: res.totalCount ?? 0,
  };
}

export async function copyDailyToHourly(): Promise<{
  success: boolean;
  errorMessage?: string;
}> {
  const res = await discordClient.copyDailyToHourly({});
  return {
    success: res.success,
    errorMessage: res.errorMessage || undefined,
  };
}

export async function updateHourlyChannelWebhook(
  channelName: string,
  webhookUrl: string
): Promise<{ success: boolean; errorMessage?: string }> {
  const res = await discordClient.updateHourlyChannelWebhook({
    channelName,
    webhookUrl,
  });
  return {
    success: res.success,
    errorMessage: res.errorMessage || undefined,
  };
}

export type ResolveHourlyResult = {
  economy: string;
  hourlyChannelName: string;
  webhookConfigured: boolean;
};

export async function resolveHourlyChannelForTicker(
  ticker: string
): Promise<ResolveHourlyResult> {
  const res = await discordClient.resolveHourlyChannelForTicker({ ticker });
  return {
    economy: res.economy ?? "",
    hourlyChannelName: res.hourlyChannelName ?? "",
    webhookConfigured: res.webhookConfigured ?? false,
  };
}

export async function sendHourlyTestMessage(channelName: string): Promise<{
  success: boolean;
  errorMessage?: string;
}> {
  const res = await discordClient.sendHourlyTestMessage({ channelName });
  return {
    success: res.success,
    errorMessage: res.errorMessage || undefined,
  };
}

// ---------------------------------------------------------------------------
// Daily Discord (same shape as hourly)
// ---------------------------------------------------------------------------

export async function getDailyDiscordConfig(): Promise<HourlyDiscordConfig> {
  const res = await discordClient.getDailyDiscordConfig({});
  return {
    enableIndustryRouting: res.enableIndustryRouting,
    logRoutingDecisions: res.logRoutingDecisions,
    channels: res.channels ?? [],
    configuredCount: res.configuredCount ?? 0,
    totalCount: res.totalCount ?? 0,
  };
}

export async function copyBaseToDaily(): Promise<{
  success: boolean;
  errorMessage?: string;
}> {
  const res = await discordClient.copyBaseToDaily({});
  return {
    success: res.success,
    errorMessage: res.errorMessage || undefined,
  };
}

export async function updateDailyChannelWebhook(
  channelName: string,
  webhookUrl: string
): Promise<{ success: boolean; errorMessage?: string }> {
  const res = await discordClient.updateDailyChannelWebhook({
    channelName,
    webhookUrl,
  });
  return {
    success: res.success,
    errorMessage: res.errorMessage || undefined,
  };
}

export async function resolveDailyChannelForTicker(
  ticker: string
): Promise<ResolveHourlyResult> {
  const res = await discordClient.resolveDailyChannelForTicker({ ticker });
  return {
    economy: res.economy ?? "",
    hourlyChannelName: res.hourlyChannelName ?? "",
    webhookConfigured: res.webhookConfigured ?? false,
  };
}

export async function sendDailyTestMessage(channelName: string): Promise<{
  success: boolean;
  errorMessage?: string;
}> {
  const res = await discordClient.sendDailyTestMessage({ channelName });
  return {
    success: res.success,
    errorMessage: res.errorMessage || undefined,
  };
}

// ---------------------------------------------------------------------------
// Weekly Discord (same shape as hourly)
// ---------------------------------------------------------------------------

export async function getWeeklyDiscordConfig(): Promise<HourlyDiscordConfig> {
  const res = await discordClient.getWeeklyDiscordConfig({});
  return {
    enableIndustryRouting: res.enableIndustryRouting,
    logRoutingDecisions: res.logRoutingDecisions,
    channels: res.channels ?? [],
    configuredCount: res.configuredCount ?? 0,
    totalCount: res.totalCount ?? 0,
  };
}

export async function copyBaseToWeekly(): Promise<{
  success: boolean;
  errorMessage?: string;
}> {
  const res = await discordClient.copyBaseToWeekly({});
  return {
    success: res.success,
    errorMessage: res.errorMessage || undefined,
  };
}

export async function updateWeeklyChannelWebhook(
  channelName: string,
  webhookUrl: string
): Promise<{ success: boolean; errorMessage?: string }> {
  const res = await discordClient.updateWeeklyChannelWebhook({
    channelName,
    webhookUrl,
  });
  return {
    success: res.success,
    errorMessage: res.errorMessage || undefined,
  };
}

export async function resolveWeeklyChannelForTicker(
  ticker: string
): Promise<ResolveHourlyResult> {
  const res = await discordClient.resolveWeeklyChannelForTicker({ ticker });
  return {
    economy: res.economy ?? "",
    hourlyChannelName: res.hourlyChannelName ?? "",
    webhookConfigured: res.webhookConfigured ?? false,
  };
}

export async function sendWeeklyTestMessage(channelName: string): Promise<{
  success: boolean;
  errorMessage?: string;
}> {
  const res = await discordClient.sendWeeklyTestMessage({ channelName });
  return {
    success: res.success,
    errorMessage: res.errorMessage || undefined,
  };
}

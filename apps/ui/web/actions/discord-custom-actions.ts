"use server";

import { discordClient } from "@/lib/grpc/channel";

export type CustomDiscordChannel = {
  name: string;
  channelName: string;
  description: string;
  webhookUrl: string;
  condition: string;
  enabled: boolean;
  createdAt: string | null;
};

export type MutationResult = {
  success: boolean;
  errorMessage?: string;
  channel?: CustomDiscordChannel;
};

function toCustomChannel(proto: {
  name: string;
  channelName: string;
  description: string;
  webhookUrl: string;
  condition: string;
  enabled: boolean;
  createdAt?: Date | undefined;
}): CustomDiscordChannel {
  return {
    name: proto.name,
    channelName: proto.channelName,
    description: proto.description,
    webhookUrl: proto.webhookUrl,
    condition: proto.condition,
    enabled: proto.enabled,
    createdAt: proto.createdAt ? proto.createdAt.toISOString() : null,
  };
}

export async function listCustomDiscordChannels(): Promise<CustomDiscordChannel[]> {
  const res = await discordClient.listCustomDiscordChannels({});
  return (res.channels ?? []).map(toCustomChannel);
}

export async function createCustomDiscordChannel(input: {
  name: string;
  webhookUrl: string;
  description: string;
  condition: string;
  enabled: boolean;
}): Promise<MutationResult> {
  const res = await discordClient.createCustomDiscordChannel(input);
  return {
    success: res.success,
    errorMessage: res.errorMessage || undefined,
    channel: res.channel ? toCustomChannel(res.channel) : undefined,
  };
}

export async function updateCustomDiscordChannel(
  name: string,
  patch: {
    webhookUrl?: string;
    description?: string;
    condition?: string;
    enabled?: boolean;
  }
): Promise<MutationResult> {
  const res = await discordClient.updateCustomDiscordChannel({ name, ...patch });
  return {
    success: res.success,
    errorMessage: res.errorMessage || undefined,
    channel: res.channel ? toCustomChannel(res.channel) : undefined,
  };
}

export async function deleteCustomDiscordChannel(
  name: string
): Promise<{ success: boolean; errorMessage?: string }> {
  const res = await discordClient.deleteCustomDiscordChannel({ name });
  return {
    success: res.success,
    errorMessage: res.errorMessage || undefined,
  };
}

export async function sendCustomDiscordChannelTestMessage(
  name: string
): Promise<{ success: boolean; errorMessage?: string }> {
  const res = await discordClient.sendCustomDiscordChannelTestMessage({ name });
  return {
    success: res.success,
    errorMessage: res.errorMessage || undefined,
  };
}

"use client";

import { useAtom } from "jotai";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { customDiscordChannelsQueryAtom, CUSTOM_DISCORD_KEY } from "@/lib/store/discord-custom";
import {
  createCustomDiscordChannel,
  updateCustomDiscordChannel,
  deleteCustomDiscordChannel,
  sendCustomDiscordChannelTestMessage,
} from "@/actions/discord-custom-actions";

export function useCustomDiscordChannels() {
  const [result] = useAtom(customDiscordChannelsQueryAtom);
  return {
    data: result.data,
    isLoading: result.isPending,
    error: result.error,
  };
}

export function useCreateCustomDiscordChannel() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: createCustomDiscordChannel,
    onSuccess: (data) => {
      if (data.success) {
        queryClient.invalidateQueries({ queryKey: CUSTOM_DISCORD_KEY });
      }
    },
  });
}

export function useUpdateCustomDiscordChannel() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({
      name,
      patch,
    }: {
      name: string;
      patch: {
        webhookUrl?: string;
        description?: string;
        condition?: string;
        enabled?: boolean;
      };
    }) => updateCustomDiscordChannel(name, patch),
    onSuccess: (data) => {
      if (data.success) {
        queryClient.invalidateQueries({ queryKey: CUSTOM_DISCORD_KEY });
      }
    },
  });
}

export function useDeleteCustomDiscordChannel() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (name: string) => deleteCustomDiscordChannel(name),
    onSuccess: (data) => {
      if (data.success) {
        queryClient.invalidateQueries({ queryKey: CUSTOM_DISCORD_KEY });
      }
    },
  });
}

export function useSendCustomDiscordChannelTestMessage() {
  return useMutation({
    mutationFn: (name: string) => sendCustomDiscordChannelTestMessage(name),
  });
}

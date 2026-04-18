"use client";

import { atomWithQuery } from "jotai-tanstack-query";
import { listCustomDiscordChannels } from "@/actions/discord-custom-actions";

export const CUSTOM_DISCORD_KEY = ["discord", "custom"] as const;

export const customDiscordChannelsQueryAtom = atomWithQuery(() => ({
  queryKey: CUSTOM_DISCORD_KEY,
  queryFn: listCustomDiscordChannels,
}));

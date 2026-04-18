"use client";

import { useCustomDiscordChannels } from "@/lib/hooks/useDiscordCustom";
import { Skeleton } from "@/components/ui/skeleton";
import { CustomChannelCreateForm } from "./CustomChannelCreateForm";
import { CustomChannelListCard } from "./CustomChannelListCard";

export function CustomDiscordChannelsContainer() {
  const { data: channels, isLoading, error } = useCustomDiscordChannels();

  return (
    <div className="space-y-8 px-4 lg:px-6 py-6">
      <div>
        <h1 className="text-2xl font-semibold">Custom Discord Channels</h1>
        <p className="mt-1 text-sm text-muted-foreground">
          Create condition-keyed Discord channels. Alerts whose condition
          matches will be routed to the configured webhook.
        </p>
      </div>

      <CustomChannelCreateForm />

      <div className="space-y-4">
        <h2 className="text-lg font-medium">Existing channels</h2>

        {isLoading && (
          <div className="space-y-3">
            <Skeleton className="h-36 w-full" />
            <Skeleton className="h-36 w-full" />
          </div>
        )}

        {error && (
          <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4">
            <p className="text-sm text-destructive">{(error as Error).message}</p>
          </div>
        )}

        {!isLoading && !error && channels && channels.length === 0 && (
          <p className="text-sm text-muted-foreground">
            No custom channels yet — create one above.
          </p>
        )}

        {!isLoading && !error && channels && channels.length > 0 && (
          <div className="grid gap-4">
            {channels.map((ch) => (
              <CustomChannelListCard key={ch.name} channel={ch} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

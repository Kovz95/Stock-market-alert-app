"use client";

import {
  useDailyDiscordConfig,
  useCopyBaseToDaily,
  useUpdateDailyChannelWebhook,
  useResolveDailyChannelForTicker,
  useSendDailyTestMessage,
} from "@/lib/hooks/useHourlyDiscord";
import { DiscordMetrics, DiscordChannelForm, DiscordTestSection } from "../_components";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { toast } from "sonner";

const SPECIAL_CHANNELS = new Set([
  "ETFs",
  "Pairs",
  "General",
  "Futures",
  "Failed_Price_Updates",
]);

export default function DailyDiscordPage() {
  const { data: config, isLoading, error } = useDailyDiscordConfig();
  const copyBaseToDaily = useCopyBaseToDaily();
  const updateWebhook = useUpdateDailyChannelWebhook();
  const resolve = useResolveDailyChannelForTicker();
  const sendTest = useSendDailyTestMessage();

  async function handleCopyBaseToDaily() {
    const result = await copyBaseToDaily.mutateAsync();
    if (result.success) {
      toast.success("Copied base webhooks into daily configuration.");
    } else {
      toast.error(result.errorMessage ?? "Failed to copy.");
    }
  }

  if (isLoading) {
    return (
      <div className="p-8">
        <p className="text-muted-foreground">Loading daily Discord config…</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8">
        <p className="text-destructive">Error loading config: {error.message}</p>
      </div>
    );
  }

  if (!config) {
    return (
      <div className="p-8">
        <p className="text-muted-foreground">No config loaded.</p>
      </div>
    );
  }

  const economyChannels = config.channels.filter((ch) => !SPECIAL_CHANNELS.has(ch.name));
  const specialChannels = config.channels.filter((ch) => SPECIAL_CHANNELS.has(ch.name));

  return (
    <div className="p-8 max-w-4xl">
      <h1 className="text-2xl font-bold mb-2">Daily Discord Channel Management</h1>
      <p className="text-muted-foreground mb-6">
        Configure dedicated Discord webhooks for <strong>daily</strong> timeframe alerts.
      </p>

      <DiscordMetrics config={config} routingLabel="Daily Routing" />

      <div className="mt-4">
        <Button
          variant="outline"
          onClick={handleCopyBaseToDaily}
          disabled={copyBaseToDaily.isPending}
        >
          {copyBaseToDaily.isPending ? "Copying…" : "Copy Base Webhooks → Daily"}
        </Button>
        <span className="ml-2 text-xs text-muted-foreground">
          Populate daily channels using the base (default) webhook URLs.
        </span>
      </div>

      <Separator className="my-6" />

      <h2 className="text-lg font-semibold mb-4">Configure Daily Channels</h2>
      <p className="text-muted-foreground text-sm mb-4">
        Update the webhook URL for each daily economy or special channel.
      </p>

      <Tabs defaultValue="economy">
        <TabsList>
          <TabsTrigger value="economy">Economy Channels</TabsTrigger>
          <TabsTrigger value="special">Special Channels</TabsTrigger>
        </TabsList>
        <TabsContent value="economy" className="space-y-4 mt-4">
          {economyChannels.length === 0 ? (
            <p className="text-muted-foreground text-sm">No economy channels configured.</p>
          ) : (
            economyChannels.map((ch) => (
              <DiscordChannelForm
                key={ch.name}
                channel={ch}
                prefixIcon="🌍"
                onSave={async (name, url) => updateWebhook.mutateAsync({ channelName: name, webhookUrl: url })}
                isPending={updateWebhook.isPending}
              />
            ))
          )}
        </TabsContent>
        <TabsContent value="special" className="space-y-4 mt-4">
          {specialChannels.length === 0 ? (
            <p className="text-muted-foreground text-sm">No special channels configured.</p>
          ) : (
            specialChannels.map((ch) => (
              <DiscordChannelForm
                key={ch.name}
                channel={ch}
                prefixIcon="⭐"
                onSave={async (name, url) => updateWebhook.mutateAsync({ channelName: name, webhookUrl: url })}
                isPending={updateWebhook.isPending}
              />
            ))
          )}
        </TabsContent>
      </Tabs>

      <Separator className="my-6" />

      <DiscordTestSection
        title="Test Daily Routing"
        description="Enter a ticker to see which daily channel it would use. Use Send test message to post a test to that channel."
        resolveTicker={resolve.mutateAsync}
        sendTestMessage={sendTest.mutateAsync}
        isResolvePending={resolve.isPending}
        isSendPending={sendTest.isPending}
      />
    </div>
  );
}

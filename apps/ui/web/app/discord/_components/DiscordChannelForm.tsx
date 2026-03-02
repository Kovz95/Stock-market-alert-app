"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { HourlyChannelInfo } from "@/actions/discord-hourly-actions";
import { toast } from "sonner";

export function DiscordChannelForm({
  channel,
  prefixIcon,
  onSave,
  isPending,
  saveButtonLabel,
}: {
  channel: HourlyChannelInfo;
  prefixIcon: string;
  onSave: (channelName: string, webhookUrl: string) => Promise<{ success: boolean; errorMessage?: string }>;
  isPending: boolean;
  saveButtonLabel?: string;
}) {
  const [webhookUrl, setWebhookUrl] = useState("");

  async function handleSave() {
    if (!webhookUrl.trim()) {
      toast.warning("Please enter a webhook URL before saving.");
      return;
    }
    try {
      const result = await onSave(channel.name, webhookUrl.trim());
      if (result.success) {
        toast.success(`Updated webhook for ${channel.name}`);
        setWebhookUrl("");
      } else {
        toast.error(result.errorMessage ?? `Failed to update ${channel.name}`);
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : `Failed to update ${channel.name}`);
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-sm font-medium">
          {prefixIcon} {channel.name} — {channel.channelName}
        </CardTitle>
        <CardDescription>
          {channel.description || "No description provided."}
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="space-y-2">
          <Label htmlFor={`webhook-${channel.name}`}>Webhook URL</Label>
          <Input
            id={`webhook-${channel.name}`}
            type="password"
            placeholder="Paste webhook URL to update"
            value={webhookUrl}
            onChange={(e) => setWebhookUrl(e.target.value)}
            className="font-mono text-xs"
          />
        </div>
        <Button
          size="sm"
          onClick={handleSave}
          disabled={isPending || !webhookUrl.trim()}
        >
          {isPending ? "Saving…" : saveButtonLabel ?? `Save ${channel.name}`}
        </Button>
      </CardContent>
    </Card>
  );
}

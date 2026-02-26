"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import type { ResolveHourlyResult } from "@/actions/discord-hourly-actions";
import { toast } from "sonner";

export function DiscordTestSection({
  title,
  description,
  resolveTicker,
  sendTestMessage,
  isResolvePending,
  isSendPending,
}: {
  title: string;
  description: string;
  resolveTicker: (ticker: string) => Promise<ResolveHourlyResult>;
  sendTestMessage: (channelName: string) => Promise<{ success: boolean; errorMessage?: string }>;
  isResolvePending: boolean;
  isSendPending: boolean;
}) {
  const [ticker, setTicker] = useState("");
  const [resolved, setResolved] = useState<ResolveHourlyResult | null>(null);

  async function handleTestChannel() {
    const t = ticker.trim();
    if (!t) return;
    try {
      const result = await resolveTicker(t);
      setResolved(result);
      if (result.economy) {
        toast.success(`${t.toUpperCase()} → ${result.economy}`);
      } else {
        toast.error("Could not determine the economy for that ticker.");
      }
    } catch {
      toast.error("Failed to resolve channel.");
    }
  }

  async function handleSendTestMessage() {
    if (!resolved?.economy) return;
    try {
      const result = await sendTestMessage(resolved.economy);
      if (result.success) {
        toast.success(`Test message sent to ${resolved.hourlyChannelName}.`);
      } else {
        toast.error(result.errorMessage ?? "Failed to send test message.");
      }
    } catch {
      toast.error("Failed to send test message.");
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        <CardDescription>{description}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap items-end gap-2">
          <div className="space-y-2 min-w-[180px]">
            <Label htmlFor="test-ticker">Ticker Symbol</Label>
            <Input
              id="test-ticker"
              placeholder="e.g. AAPL, MSFT, XOM"
              value={ticker}
              onChange={(e) => {
                setTicker(e.target.value);
                setResolved(null);
              }}
            />
          </div>
          <Button
            size="default"
            onClick={handleTestChannel}
            disabled={isResolvePending || !ticker.trim()}
          >
            {isResolvePending ? "Resolving…" : "Test Channel"}
          </Button>
        </div>

        {resolved && (
          <div className="rounded-md border bg-muted/50 p-3 text-sm space-y-2">
            {resolved.economy ? (
              <>
                <p>
                  <strong>Economy:</strong> {resolved.economy}
                </p>
                <p>
                  <strong>Channel:</strong> {resolved.hourlyChannelName}
                </p>
                {!resolved.webhookConfigured && (
                  <p className="text-amber-600 dark:text-amber-500">
                    Channel for this economy is not configured (no webhook URL).
                  </p>
                )}
                {resolved.webhookConfigured && (
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={handleSendTestMessage}
                    disabled={isSendPending}
                  >
                    {isSendPending ? "Sending…" : "Send test message to this channel"}
                  </Button>
                )}
              </>
            ) : (
              <p className="text-muted-foreground">No economy found for this ticker.</p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

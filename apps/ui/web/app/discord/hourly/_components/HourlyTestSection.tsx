"use client";

import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  useResolveHourlyChannelForTicker,
  useSendHourlyTestMessage,
} from "@/lib/hooks/useHourlyDiscord";
import { toast } from "sonner";

export function HourlyTestSection() {
  const [ticker, setTicker] = useState("");
  const [resolved, setResolved] = useState<{
    economy: string;
    hourlyChannelName: string;
    webhookConfigured: boolean;
  } | null>(null);

  const resolve = useResolveHourlyChannelForTicker();
  const sendTest = useSendHourlyTestMessage();

  async function handleTestChannel() {
    const t = ticker.trim();
    if (!t) return;
    try {
      const result = await resolve.mutateAsync(t);
      setResolved({
        economy: result.economy,
        hourlyChannelName: result.hourlyChannelName,
        webhookConfigured: result.webhookConfigured,
      });
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
      const result = await sendTest.mutateAsync(resolved.economy);
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
        <CardTitle>Test Hourly Routing</CardTitle>
        <CardDescription>
          Enter a ticker to see which hourly channel it would use. Use &quot;Send test message&quot; to
          post a test to that channel.
        </CardDescription>
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
            disabled={resolve.isPending || !ticker.trim()}
          >
            {resolve.isPending ? "Resolving…" : "Test Hourly Channel"}
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
                  <strong>Hourly channel:</strong> {resolved.hourlyChannelName}
                </p>
                {!resolved.webhookConfigured && (
                  <p className="text-amber-600 dark:text-amber-500">
                    Hourly channel for this economy is not configured (no webhook URL).
                  </p>
                )}
                {resolved.webhookConfigured && (
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={handleSendTestMessage}
                    disabled={sendTest.isPending}
                  >
                    {sendTest.isPending ? "Sending…" : "Send test message to this channel"}
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

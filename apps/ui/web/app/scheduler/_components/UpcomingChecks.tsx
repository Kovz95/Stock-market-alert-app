"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { useExchangeSchedule } from "@/lib/hooks/useScheduler";
import { useMemo } from "react";
import type { Timeframe } from "@/lib/hooks/useScheduler";

function formatCountdown(totalSeconds: number): string {
  if (totalSeconds <= 0) return "now";
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

export function UpcomingChecks({ timeframe }: { timeframe: Timeframe }) {
  const { data: schedule, isLoading } = useExchangeSchedule(timeframe);
  const timeframeLabel =
    timeframe.charAt(0).toUpperCase() + timeframe.slice(1);

  const upcoming = useMemo(() => {
    if (!schedule) return [];
    return [...schedule]
      .filter((r) => r.timeRemainingSeconds > 0)
      .sort((a, b) => a.timeRemainingSeconds - b.timeRemainingSeconds)
      .slice(0, 5);
  }, [schedule]);

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Upcoming Checks — {timeframeLabel}</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">Loading...</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Upcoming Checks — {timeframeLabel}</CardTitle>
      </CardHeader>
      <CardContent>
        {upcoming.length === 0 ? (
          <p className="text-muted-foreground text-sm">
            No upcoming checks (weekend or all completed).
          </p>
        ) : (
          <div className="space-y-3">
            {upcoming.map((row) => (
              <div
                key={row.symbol}
                className="flex items-center justify-between text-sm"
              >
                <div>
                  <span className="font-medium">{row.exchange}</span>
                  {row.runTimeEt && (
                    <span className="text-muted-foreground ml-2">
                      {timeframe === "hourly" ? `candle ${row.runTimeEt} ET` : `${row.runTimeEt} ET`}
                    </span>
                  )}
                </div>
                <Badge variant="outline">
                  {formatCountdown(row.timeRemainingSeconds)}
                </Badge>
              </div>
            ))}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

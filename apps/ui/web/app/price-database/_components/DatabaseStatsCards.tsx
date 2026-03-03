"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import type { DatabaseStatsData } from "@/actions/price-database-actions";
import { Loader2Icon } from "lucide-react";

function formatDate(iso: string | null): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleDateString();
  } catch {
    return "—";
  }
}

export type TimeframeKind = "hourly" | "daily" | "weekly";

export type DatabaseStatsCardsProps = {
  stats: DatabaseStatsData;
  exchanges: string[];
  selectedTickers: string[];
  onUpdate: (timeframe: TimeframeKind) => void;
  updatePending?: Partial<Record<TimeframeKind, boolean>>;
};

export function DatabaseStatsCards({
  stats,
  exchanges,
  selectedTickers,
  onUpdate,
  updatePending = {},
}: DatabaseStatsCardsProps) {
  const canUpdate = exchanges.length > 0;

  return (
    <div className="grid gap-4 md:grid-cols-3">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Hourly
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <p className="text-2xl font-bold">
            {stats.hourlyRecords.toLocaleString()} records
          </p>
          <p className="text-muted-foreground text-xs">
            {stats.hourlyTickers} tickers · {formatDate(stats.hourlyMin)} –{" "}
            {formatDate(stats.hourlyMax)}
          </p>
          {updatePending.hourly ? (
            <p className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2Icon className="size-4 animate-spin" />
              Updating…
            </p>
          ) : (
            <Button
              variant="outline"
              size="sm"
              disabled={!canUpdate}
              onClick={() => onUpdate("hourly")}
            >
              Update
            </Button>
          )}
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Daily
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <p className="text-2xl font-bold">
            {stats.dailyRecords.toLocaleString()} records
          </p>
          <p className="text-muted-foreground text-xs">
            {stats.dailyTickers} tickers · {formatDate(stats.dailyMin)} –{" "}
            {formatDate(stats.dailyMax)}
          </p>
          {updatePending.daily ? (
            <p className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2Icon className="size-4 animate-spin" />
              Updating…
            </p>
          ) : (
            <Button
              variant="outline"
              size="sm"
              disabled={!canUpdate}
              onClick={() => onUpdate("daily")}
            >
              Update
            </Button>
          )}
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Weekly
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <p className="text-2xl font-bold">
            {stats.weeklyRecords.toLocaleString()} records
          </p>
          <p className="text-muted-foreground text-xs">
            {stats.weeklyTickers} tickers · {formatDate(stats.weeklyMin)} –{" "}
            {formatDate(stats.weeklyMax)}
          </p>
          {updatePending.weekly ? (
            <p className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2Icon className="size-4 animate-spin" />
              Updating…
            </p>
          ) : (
            <Button
              variant="outline"
              size="sm"
              disabled={!canUpdate}
              onClick={() => onUpdate("weekly")}
            >
              Update
            </Button>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

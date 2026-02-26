"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { DatabaseStatsData } from "@/actions/price-database-actions";

function formatDate(iso: string | null): string {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleDateString();
  } catch {
    return "—";
  }
}

export function DatabaseStatsCards({ stats }: { stats: DatabaseStatsData }) {
  return (
    <div className="grid gap-4 md:grid-cols-3">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Hourly
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-2xl font-bold">
            {stats.hourlyRecords.toLocaleString()} records
          </p>
          <p className="text-muted-foreground text-xs">
            {stats.hourlyTickers} tickers · {formatDate(stats.hourlyMin)} –{" "}
            {formatDate(stats.hourlyMax)}
          </p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Daily
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-2xl font-bold">
            {stats.dailyRecords.toLocaleString()} records
          </p>
          <p className="text-muted-foreground text-xs">
            {stats.dailyTickers} tickers · {formatDate(stats.dailyMin)} –{" "}
            {formatDate(stats.dailyMax)}
          </p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Weekly
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-2xl font-bold">
            {stats.weeklyRecords.toLocaleString()} records
          </p>
          <p className="text-muted-foreground text-xs">
            {stats.weeklyTickers} tickers · {formatDate(stats.weeklyMin)} –{" "}
            {formatDate(stats.weeklyMax)}
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

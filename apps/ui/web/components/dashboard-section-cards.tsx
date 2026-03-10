"use client";

import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardAction,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useDashboardStats } from "@/lib/hooks/useDashboard";
import type { DashboardTimeframeBreakdown } from "@/actions/dashboard-actions";
import {
  BellIcon,
  TrendingUpIcon,
  BarChart3Icon,
  LayersIcon,
} from "lucide-react";

function TimeframeBreakdown({ by }: { by: DashboardTimeframeBreakdown }) {
  const parts = [
    by.daily > 0 ? `Daily ${by.daily}` : null,
    by.weekly > 0 ? `Weekly ${by.weekly}` : null,
    by.hourly > 0 ? `Hourly ${by.hourly}` : null,
  ].filter(Boolean);
  if (parts.length === 0) return <span className="text-muted-foreground">Daily 0 · Weekly 0 · Hourly 0</span>;
  return <span className="text-muted-foreground">{parts.join(" · ")}</span>;
}

function StatCard({
  description,
  value,
  footer,
  icon: Icon,
  badge,
  timeframeBreakdown,
}: {
  description: string;
  value: string | number;
  footer: string;
  icon: React.ElementType;
  badge?: React.ReactNode;
  timeframeBreakdown?: DashboardTimeframeBreakdown;
}) {
  return (
    <Card className="@container/card">
      <CardHeader>
        <CardDescription>{description}</CardDescription>
        <CardTitle className="text-2xl font-semibold tabular-nums @[250px]/card:text-3xl">
          {value}
        </CardTitle>
        {badge != null && <CardAction>{badge}</CardAction>}
      </CardHeader>
      <CardFooter className="flex-col items-start gap-1.5 text-sm">
        {timeframeBreakdown != null && (
          <div className="w-full text-xs">
            <TimeframeBreakdown by={timeframeBreakdown} />
          </div>
        )}
        <div className="line-clamp-1 flex gap-2 font-medium">
          <Icon className="size-4 shrink-0" />
          {footer}
        </div>
      </CardFooter>
    </Card>
  );
}

export function DashboardSectionCards() {
  const { data: stats, isLoading, error } = useDashboardStats();

  if (error) {
    return (
      <div className="px-4 lg:px-6 space-y-4">
        <p className="text-sm text-destructive">
          Could not load dashboard:{" "}
          {error instanceof Error ? error.message : "Unknown error"}
        </p>
        <div className="*:data-[slot=card]:from-primary/5 *:data-[slot=card]:to-card dark:*:data-[slot=card]:bg-card grid grid-cols-1 gap-4 @xl/main:grid-cols-2 @5xl/main:grid-cols-4">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i} className="@container/card">
              <CardHeader>
                <CardDescription>—</CardDescription>
                <CardTitle className="text-2xl font-semibold tabular-nums">—</CardTitle>
              </CardHeader>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  if (isLoading || stats == null) {
    return (
      <div className="*:data-[slot=card]:from-primary/5 *:data-[slot=card]:to-card dark:*:data-[slot=card]:bg-card grid grid-cols-1 gap-4 px-4 lg:px-6 @xl/main:grid-cols-2 @5xl/main:grid-cols-4">
        {[1, 2, 3, 4].map((i) => (
          <Card key={i} className="@container/card">
            <CardHeader>
              <CardDescription className="animate-pulse">—</CardDescription>
              <CardTitle className="text-2xl font-semibold tabular-nums @[250px]/card:text-3xl animate-pulse">
                —
              </CardTitle>
            </CardHeader>
            <CardFooter className="flex-col items-start gap-1.5 text-sm text-muted-foreground">
              Loading…
            </CardFooter>
          </Card>
        ))}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {stats.error && (
        <div className="px-4 lg:px-6">
          <p className="text-sm text-amber-600 dark:text-amber-500">
            Backend issue: {stats.error}. Showing best-effort data.
          </p>
        </div>
      )}
      <div className="*:data-[slot=card]:from-primary/5 *:data-[slot=card]:to-card dark:*:data-[slot=card]:bg-card grid grid-cols-1 gap-4 px-4 lg:px-6 *:data-[slot=card]:bg-gradient-to-t *:data-[slot=card]:shadow-xs @xl/main:grid-cols-2 @5xl/main:grid-cols-4">
        <StatCard
          description="Active Alerts"
          value={stats.activeAlerts}
          footer="Alerts currently enabled"
          icon={BellIcon}
          timeframeBreakdown={stats.activeAlertsByTimeframe}
        />
        <StatCard
          description="Triggered Today"
          value={stats.triggeredToday}
          footer="Triggers in the last 24 hours"
          icon={TrendingUpIcon}
          badge={
            stats.triggeredToday > 0 ? (
              <Badge variant="outline">
                <TrendingUpIcon className="size-3.5" />
                {stats.triggeredToday}
              </Badge>
            ) : undefined
          }
          timeframeBreakdown={stats.triggeredTodayByTimeframe}
        />
        <StatCard
          description="Watched Symbols"
          value={stats.watchedSymbols}
          footer="Unique tickers across alerts"
          icon={LayersIcon}
        />
        <StatCard
          description="Triggers (7d)"
          value={stats.triggersLast7d}
          footer="Total triggers in the last 7 days"
          icon={BarChart3Icon}
          badge={
            stats.triggersLast7d > 0 ? (
              <Badge variant="outline">
                <BarChart3Icon className="size-3.5" />
                {stats.triggersLast7d}
              </Badge>
            ) : undefined
          }
          timeframeBreakdown={stats.triggersLast7dByTimeframe}
        />
      </div>
    </div>
  );
}

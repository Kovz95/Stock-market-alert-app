"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { HourlyDiscordConfig } from "@/lib/hooks/useHourlyDiscord";

export function HourlyDiscordMetrics({ config }: { config: HourlyDiscordConfig }) {
  return (
    <div className="grid gap-4 md:grid-cols-3">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Hourly Routing
          </CardTitle>
        </CardHeader>
        <CardContent>
          <span className="text-xl font-semibold">
            {config.enableIndustryRouting ? "Enabled" : "Disabled"}
          </span>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Configured Channels
          </CardTitle>
        </CardHeader>
        <CardContent>
          <span className="text-xl font-semibold">
            {config.configuredCount}/{config.totalCount}
          </span>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Routing Logs
          </CardTitle>
        </CardHeader>
        <CardContent>
          <span className="text-xl font-semibold">
            {config.logRoutingDecisions ? "Enabled" : "Disabled"}
          </span>
        </CardContent>
      </Card>
    </div>
  );
}

"use client";

import { useState } from "react";
import { SchedulerStatusCard } from "./_components/SchedulerStatusCard";
import { TimeInfoBar } from "./_components/TimeInfoBar";
import { UpcomingChecks } from "./_components/UpcomingChecks";
import { ExchangeScheduleTable } from "./_components/ExchangeScheduleTable";
import {
  ToggleGroup,
  ToggleGroupItem,
} from "@/components/ui/toggle-group";
import type { Timeframe } from "@/lib/hooks/useScheduler";

export type { Timeframe };

export default function SchedulerPage() {
  const [timeframe, setTimeframe] = useState<Timeframe>("daily");

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Scheduler Status</h1>
        <ToggleGroup
          type="single"
          value={timeframe}
          onValueChange={(v) => v && setTimeframe(v as Timeframe)}
          variant="outline"
          size="sm"
        >
          <ToggleGroupItem value="hourly">Hourly</ToggleGroupItem>
          <ToggleGroupItem value="daily">Daily</ToggleGroupItem>
          <ToggleGroupItem value="weekly">Weekly</ToggleGroupItem>
        </ToggleGroup>
      </div>

      <TimeInfoBar />

      <div className="grid gap-6 lg:grid-cols-2">
        <SchedulerStatusCard />
        <UpcomingChecks timeframe={timeframe} />
      </div>

      <ExchangeScheduleTable timeframe={timeframe} />
    </div>
  );
}

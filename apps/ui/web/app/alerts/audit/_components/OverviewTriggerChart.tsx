"use client";

import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import { LineChart, Line, XAxis, YAxis, CartesianGrid } from "recharts";
import type { TriggerCountByDayRow } from "@/actions/audit-actions";

const chartConfig = {
  date: { label: "Date" },
  count: { label: "Triggers", color: "hsl(var(--chart-1))" },
} satisfies ChartConfig;

type OverviewTriggerChartProps = {
  data: TriggerCountByDayRow[];
};

export function OverviewTriggerChart({ data }: OverviewTriggerChartProps) {
  if (data.length < 2) return null;

  return (
    <div>
      <h3 className="text-sm font-medium mb-2">Trigger count by day</h3>
      <ChartContainer config={chartConfig} className="h-[260px] w-full">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" tick={{ fontSize: 10 }} />
          <YAxis tick={{ fontSize: 10 }} />
          <ChartTooltip content={<ChartTooltipContent />} />
          <Line
            type="monotone"
            dataKey="count"
            stroke="var(--chart-1)"
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ChartContainer>
    </div>
  );
}

"use client";

import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { PieChart, Pie, Cell } from "recharts";

type AssetBreakdown = {
  assetType: string;
  failedAlerts: number;
  failureCount: number;
};

type FailedDataAssetChartProps = {
  breakdown: AssetBreakdown[];
};

export function FailedDataAssetChart({ breakdown }: FailedDataAssetChartProps) {
  if (!breakdown?.length) return null;

  const pieData = breakdown.map((r) => ({
    name: r.assetType,
    value: r.failedAlerts,
  }));

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div>
        <h3 className="text-sm font-medium mb-2">Failures by asset type</h3>
        <ul className="text-sm text-muted-foreground space-y-1">
          {breakdown.map((row) => (
            <li key={row.assetType}>
              <strong>{row.assetType}</strong>: {row.failedAlerts} alerts,{" "}
              {row.failureCount} failures
            </li>
          ))}
        </ul>
      </div>
      <ChartContainer
        config={{
          name: { label: "Asset type" },
          value: { label: "Failed alerts" },
        }}
        className="h-[200px] w-full"
      >
        <PieChart>
          <Pie
            data={pieData}
            dataKey="value"
            nameKey="name"
            cx="50%"
            cy="50%"
            outerRadius={80}
            label={({ name, value }) => `${name}: ${value}`}
          >
            {pieData.map((_, i) => (
              <Cell key={i} fill={`hsl(var(--chart-${(i % 5) + 1}))`} />
            ))}
          </Pie>
          <ChartTooltip content={<ChartTooltipContent />} />
        </PieChart>
      </ChartContainer>
    </div>
  );
}

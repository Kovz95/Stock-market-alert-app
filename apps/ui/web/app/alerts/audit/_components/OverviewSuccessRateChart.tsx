"use client";

import { useMemo } from "react";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid } from "recharts";
import type { AuditSummaryRow } from "@/actions/audit-actions";

type OverviewSuccessRateChartProps = {
  summaryRows: AuditSummaryRow[];
};

export function OverviewSuccessRateChart({ summaryRows }: OverviewSuccessRateChartProps) {
  const data = useMemo(() => {
    if (!summaryRows?.length) return [];
    const byTf = new Map<string, { total: number; success: number }>();
    for (const r of summaryRows) {
      const tf = r.timeframe || "(empty)";
      const cur = byTf.get(tf) ?? { total: 0, success: 0 };
      cur.total += r.totalChecks ?? 0;
      cur.success += r.successfulEvaluations ?? 0;
      byTf.set(tf, cur);
    }
    return Array.from(byTf.entries()).map(([timeframe, v]) => ({
      timeframe,
      successRate: v.total ? (v.success / v.total) * 100 : 0,
    }));
  }, [summaryRows]);

  if (data.length === 0) return null;

  return (
    <div>
      <h3 className="text-sm font-medium mb-2">Success rate by timeframe</h3>
      <ChartContainer
        config={{
          timeframe: { label: "Timeframe" },
          successRate: { label: "Success rate (%)", color: "hsl(var(--chart-2))" },
        }}
        className="h-[260px] w-full"
      >
        <BarChart data={data} margin={{ left: 12, right: 12 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="timeframe" tick={{ fontSize: 10 }} />
          <YAxis tick={{ fontSize: 10 }} unit="%" />
          <ChartTooltip content={<ChartTooltipContent />} />
          <Bar dataKey="successRate" fill="var(--chart-2)" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ChartContainer>
    </div>
  );
}

"use client";

import * as React from "react";
import type { FullStockMetadataRow } from "@/actions/stock-database-actions";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type TopExchangesSectionProps = {
  data: FullStockMetadataRow[];
  title?: string;
};

export function TopExchangesSection({ data, title = "Top exchanges" }: TopExchangesSectionProps) {
  const counts = React.useMemo(() => {
    const map = new Map<string, number>();
    for (const r of data) {
      const e = r.exchange || "";
      if (!e) continue;
      map.set(e, (map.get(e) ?? 0) + 1);
    }
    return Array.from(map.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 15);
  }, [data]);

  if (data.length === 0 || counts.length === 0) return null;

  const total = data.length;
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <ul className="space-y-1 text-sm">
          {counts.map(([exchange, count]) => {
            const pct = ((count / total) * 100).toFixed(1);
            return (
              <li key={exchange}>
                <span className="font-medium">{exchange}</span>: {count.toLocaleString()} ({pct}%)
              </li>
            );
          })}
        </ul>
      </CardContent>
    </Card>
  );
}

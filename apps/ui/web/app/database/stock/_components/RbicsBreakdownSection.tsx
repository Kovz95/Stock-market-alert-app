"use client";

import * as React from "react";
import type { FullStockMetadataRow } from "@/actions/stock-database-actions";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type RbicsBreakdownSectionProps = {
  stocks: FullStockMetadataRow[];
};

export function RbicsBreakdownSection({ stocks }: RbicsBreakdownSectionProps) {
  const counts = React.useMemo(() => {
    const map = new Map<string, number>();
    for (const r of stocks) {
      const e = r.rbicsEconomy || "";
      if (!e) continue;
      map.set(e, (map.get(e) ?? 0) + 1);
    }
    return Array.from(map.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 20);
  }, [stocks]);

  if (stocks.length === 0 || counts.length === 0) return null;

  const total = stocks.length;
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-base">RBICS economy (top 20)</CardTitle>
      </CardHeader>
      <CardContent>
        <ul className="space-y-1 text-sm">
          {counts.map(([economy, count]) => {
            const pct = ((count / total) * 100).toFixed(1);
            return (
              <li key={economy}>
                <span className="font-medium">{economy}</span>: {count.toLocaleString()} ({pct}%)
              </li>
            );
          })}
        </ul>
      </CardContent>
    </Card>
  );
}

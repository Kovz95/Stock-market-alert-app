"use client";

import { Card, CardContent } from "@/components/ui/card";
import type { FailedPriceData } from "@/actions/audit-actions";

type FailedDataKpiCardsProps = {
  data: FailedPriceData;
};

export function FailedDataKpiCards({ data }: FailedDataKpiCardsProps) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
      <Card size="sm">
        <CardContent className="pt-3">
          <p className="text-muted-foreground text-xs">Failed alerts</p>
          <p className="text-lg font-semibold">
            {data.totalFailedAlerts?.toLocaleString() ?? 0}
          </p>
        </CardContent>
      </Card>
      <Card size="sm">
        <CardContent className="pt-3">
          <p className="text-muted-foreground text-xs">Total failures</p>
          <p className="text-lg font-semibold">
            {data.totalFailures?.toLocaleString() ?? 0}
          </p>
        </CardContent>
      </Card>
      <Card size="sm">
        <CardContent className="pt-3">
          <p className="text-muted-foreground text-xs">Avg failures/alert</p>
          <p className="text-lg font-semibold">
            {(
              (data.totalFailures ?? 0) /
              Math.max(data.totalFailedAlerts ?? 1, 1)
            ).toFixed(1)}
          </p>
        </CardContent>
      </Card>
      <Card size="sm">
        <CardContent className="pt-3">
          <p className="text-muted-foreground text-xs">Failure rate</p>
          <p className="text-lg font-semibold">
            {(data.failureRate ?? 0).toFixed(1)}%
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

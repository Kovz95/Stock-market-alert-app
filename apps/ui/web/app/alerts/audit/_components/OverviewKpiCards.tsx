"use client";

import { Card, CardContent } from "@/components/ui/card";
import {
  AlertTriangleIcon,
  CheckCircleIcon,
  InfoIcon,
} from "lucide-react";
import type { PerformanceMetrics } from "@/actions/audit-actions";

type OverviewKpiCardsProps = {
  metrics: PerformanceMetrics;
};

export function OverviewKpiCards({ metrics }: OverviewKpiCardsProps) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <Card size="sm">
          <CardContent className="pt-3">
            <p className="text-muted-foreground text-xs">Total checks</p>
            <p className="text-lg font-semibold">
              {metrics.totalChecks?.toLocaleString() ?? 0}
            </p>
          </CardContent>
        </Card>
        <Card size="sm">
          <CardContent className="pt-3">
            <p className="text-muted-foreground text-xs">Success rate</p>
            <p className="text-lg font-semibold">
              {(metrics.successRate ?? 0).toFixed(1)}%
            </p>
          </CardContent>
        </Card>
        <Card size="sm">
          <CardContent className="pt-3">
            <p className="text-muted-foreground text-xs">Cache hit rate</p>
            <p className="text-lg font-semibold">
              {(metrics.cacheHitRate ?? 0).toFixed(1)}%
            </p>
          </CardContent>
        </Card>
        <Card size="sm">
          <CardContent className="pt-3">
            <p className="text-muted-foreground text-xs">Avg execution</p>
            <p className="text-lg font-semibold">
              {Math.round(metrics.avgExecutionTimeMs ?? 0)} ms
            </p>
          </CardContent>
        </Card>
      </div>
      {metrics.errorRate != null && (
        <div className="flex items-center gap-2 text-sm">
          {metrics.errorRate > 5 ? (
            <span className="flex items-center gap-1 text-amber-600">
              <AlertTriangleIcon className="size-4" />
              High error rate: {metrics.errorRate.toFixed(1)}%
            </span>
          ) : metrics.errorRate > 1 ? (
            <span className="flex items-center gap-1 text-blue-600">
              <InfoIcon className="size-4" />
              Error rate: {metrics.errorRate.toFixed(1)}%
            </span>
          ) : (
            <span className="flex items-center gap-1 text-green-600">
              <CheckCircleIcon className="size-4" />
              Low error rate: {metrics.errorRate.toFixed(1)}%
            </span>
          )}
        </div>
      )}
    </div>
  );
}

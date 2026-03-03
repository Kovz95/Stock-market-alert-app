"use client";

import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  useSchedulerStatus,
  useQueueTasks,
  useStartScheduler,
  useStopScheduler,
} from "@/lib/hooks/useScheduler";
import type { Timeframe } from "@/lib/hooks/useScheduler";
import { toast } from "sonner";

function formatTimeAgo(iso: string | null): string {
  if (!iso) return "N/A";
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ${mins % 60}m ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

function formatProcessAt(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  return d.toLocaleString(undefined, {
    dateStyle: "short",
    timeStyle: "short",
  });
}

export function SchedulerStatusCard({ timeframe }: { timeframe: Timeframe }) {
  const { data: status, isLoading: statusLoading } = useSchedulerStatus();
  const { data: queueTasks, isLoading: queueLoading } = useQueueTasks();
  const startMutation = useStartScheduler();
  const stopMutation = useStopScheduler();

  const enqueuedForTimeframe = useMemo(() => {
    if (!queueTasks) return [];
    return queueTasks.filter((t) => t.timeframe === timeframe);
  }, [queueTasks, timeframe]);

  async function handleStart() {
    const result = await startMutation.mutateAsync();
    if (result.success) {
      toast.success(result.message);
    } else {
      toast.error(result.message);
    }
  }

  async function handleStop() {
    const result = await stopMutation.mutateAsync();
    if (result.success) {
      toast.success(result.message);
    } else {
      toast.error(result.message);
    }
  }

  const isLoading = statusLoading || !status;
  const timeframeLabel =
    timeframe.charAt(0).toUpperCase() + timeframe.slice(1);

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Scheduler Status</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">Loading...</p>
        </CardContent>
      </Card>
    );
  }

  const statusVariant =
    status.status === "running"
      ? "default"
      : status.status === "error"
        ? "destructive"
        : "secondary";

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle>Scheduler Status</CardTitle>
        <Badge variant={statusVariant}>
          {status.status === "running"
            ? "Running"
            : status.status === "error"
              ? "Error"
              : status.status}
        </Badge>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-3 text-sm">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Heartbeat</span>
            <span>{formatTimeAgo(status.heartbeat)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Queue Size</span>
            <span>{status.queueSize}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Tasks In Progress</span>
            <span>{status.activeWorkers}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Worker Processes</span>
            <span>{status.workerProcesses}</span>
          </div>
        </div>

        <div className="rounded-md bg-muted p-3 text-sm">
          <p className="font-medium mb-2">
            Enqueued {timeframeLabel} Jobs
            {!queueLoading && (
              <span className="text-muted-foreground font-normal ml-2">
                ({enqueuedForTimeframe.length})
              </span>
            )}
          </p>
          {queueLoading ? (
            <p className="text-muted-foreground">Loading queue...</p>
          ) : enqueuedForTimeframe.length === 0 ? (
            <p className="text-muted-foreground">
              {timeframe === "hourly"
                ? "No hourly jobs in the queue. Hourly tasks are enqueued only when an exchange is open (weekday + market hours in its timezone) and are scheduled for ~30 min from each cycle."
                : `No ${timeframe} jobs in the queue.`}
            </p>
          ) : (
            <ul className="space-y-2 max-h-48 overflow-y-auto">
              {enqueuedForTimeframe.map((t) => (
                <li
                  key={t.id}
                  className="flex items-center justify-between gap-2 text-muted-foreground"
                >
                  <span className="font-medium text-foreground truncate">
                    {t.exchange}
                  </span>
                  <div className="flex items-center gap-2 shrink-0">
                    <Badge variant="outline" className="text-xs">
                      {t.state}
                    </Badge>
                    {t.nextProcessAt && (
                      <span className="text-xs">
                        {formatProcessAt(t.nextProcessAt)}
                      </span>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="flex gap-2 pt-2">
          <Button
            size="sm"
            onClick={handleStart}
            disabled={startMutation.isPending}
          >
            {startMutation.isPending ? "Starting..." : "Start"}
          </Button>
          <Button
            size="sm"
            variant="destructive"
            onClick={handleStop}
            disabled={stopMutation.isPending}
          >
            {stopMutation.isPending ? "Stopping..." : "Stop"}
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

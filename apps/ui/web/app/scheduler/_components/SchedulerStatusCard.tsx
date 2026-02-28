"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  useSchedulerStatus,
  useStartScheduler,
  useStopScheduler,
} from "@/lib/hooks/useScheduler";
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

export function SchedulerStatusCard() {
  const { data: status, isLoading } = useSchedulerStatus();
  const startMutation = useStartScheduler();
  const stopMutation = useStopScheduler();

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

  if (isLoading || !status) {
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

        {status.currentJob && (
          <div className="rounded-md bg-muted p-3 text-sm">
            <p className="font-medium">Current Job</p>
            <p className="text-muted-foreground">
              {status.currentJob.exchange} / {status.currentJob.timeframe}
              {status.currentJob.started &&
                ` — started ${formatTimeAgo(status.currentJob.started)}`}
            </p>
          </div>
        )}

        {status.lastRun && (
          <div className="rounded-md bg-muted p-3 text-sm">
            <p className="font-medium">Last Run</p>
            <p className="text-muted-foreground">
              {status.lastRun.exchange} / {status.lastRun.timeframe}
              {status.lastRun.completedAt &&
                ` — ${formatTimeAgo(status.lastRun.completedAt)}`}
            </p>
          </div>
        )}

        {status.lastError && (
          <div className="rounded-md border border-destructive/50 bg-destructive/10 p-3 text-sm">
            <p className="font-medium text-destructive">Last Error</p>
            <p className="text-muted-foreground">
              {status.lastError.exchange} / {status.lastError.timeframe}:{" "}
              {status.lastError.message}
            </p>
          </div>
        )}

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

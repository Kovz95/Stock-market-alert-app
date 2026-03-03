"use server";

import { alertClient, schedulerClient } from "@/lib/grpc/channel";

export type SchedulerStatusData = {
  status: string;
  heartbeat: string | null;
  currentJob: {
    exchange: string;
    timeframe: string;
    started: string | null;
  } | null;
  lastRun: {
    exchange: string;
    timeframe: string;
    completedAt: string | null;
  } | null;
  lastError: {
    exchange: string;
    timeframe: string;
    message: string;
    time: string | null;
  } | null;
  queueSize: number;
  activeWorkers: number;
  workerProcesses: number;
};

export type ExchangeScheduleRow = {
  exchange: string;
  symbol: string;
  region: string;
  runTimeEt: string;
  runTimeUtc: string;
  localClose: string;
  localTz: string;
  timeRemainingSeconds: number;
  lastRunDate: string;
  lastRunStart: string;
  lastRunEnd: string;
};

export type ControlResult = {
  success: boolean;
  message: string;
};

export type QueueTaskData = {
  state: string;
  type: string;
  exchange: string;
  timeframe: string;
  nextProcessAt: string | null;
  id: string;
};

export async function listQueueTasks(queue: string = ""): Promise<QueueTaskData[]> {
  const res = await schedulerClient.listQueueTasks({ queue });
  return (res.tasks ?? []).map((t) => ({
    state: t.state,
    type: t.type,
    exchange: t.exchange,
    timeframe: t.timeframe,
    nextProcessAt: t.nextProcessAt ? t.nextProcessAt.toISOString() : null,
    id: t.id,
  }));
}

export async function getSchedulerStatus(): Promise<SchedulerStatusData> {
  const res = await schedulerClient.getSchedulerStatus({});
  return {
    status: res.status,
    heartbeat: res.heartbeat ? res.heartbeat.toISOString() : null,
    currentJob: res.currentJob
      ? {
          exchange: res.currentJob.exchange,
          timeframe: res.currentJob.timeframe,
          started: res.currentJob.started
            ? res.currentJob.started.toISOString()
            : null,
        }
      : null,
    lastRun: res.lastRun
      ? {
          exchange: res.lastRun.exchange,
          timeframe: res.lastRun.timeframe,
          completedAt: res.lastRun.completedAt
            ? res.lastRun.completedAt.toISOString()
            : null,
        }
      : null,
    lastError: res.lastError
      ? {
          exchange: res.lastError.exchange,
          timeframe: res.lastError.timeframe,
          message: res.lastError.message,
          time: res.lastError.time
            ? res.lastError.time.toISOString()
            : null,
        }
      : null,
    queueSize: res.queueSize,
    activeWorkers: res.activeWorkers,
    workerProcesses: res.workerProcesses,
  };
}

export async function getExchangeSchedule(timeframe: string = "daily"): Promise<ExchangeScheduleRow[]> {
  const res = await schedulerClient.getExchangeSchedule({ timeframe });
  return (res.rows ?? []).map((row) => ({
    exchange: row.exchange,
    symbol: row.symbol,
    region: row.region,
    runTimeEt: row.runTimeEt,
    runTimeUtc: row.runTimeUtc,
    localClose: row.localClose,
    localTz: row.localTz,
    timeRemainingSeconds: Number(row.timeRemainingSeconds ?? 0),
    lastRunDate: row.lastRunDate,
    lastRunStart: row.lastRunStart,
    lastRunEnd: row.lastRunEnd,
  }));
}

export async function startScheduler(): Promise<ControlResult> {
  const res = await schedulerClient.startScheduler({});
  return {
    success: res.success,
    message: res.message,
  };
}

export async function stopScheduler(): Promise<ControlResult> {
  const res = await schedulerClient.stopScheduler({});
  return {
    success: res.success,
    message: res.message,
  };
}

export async function runExchangeJob(
  exchange: string,
  timeframe: string
): Promise<ControlResult> {
  const res = await schedulerClient.runExchangeJob({ exchange, timeframe });
  return {
    success: res.success,
    message: res.message,
  };
}

export type EvaluateExchangeResult = {
  success: boolean;
  message: string;
  alertsTotal: number;
  alertsTriggered: number;
  pricesUpdated: number;
  durationSeconds: number;
};

export async function evaluateExchange(
  exchange: string,
  timeframe: string
): Promise<EvaluateExchangeResult> {
  const res = await alertClient.evaluateExchange({ exchange, timeframe });
  return {
    success: res.success,
    message: res.message,
    alertsTotal: res.alertsTotal,
    alertsTriggered: res.alertsTriggered,
    pricesUpdated: res.pricesUpdated,
    durationSeconds: res.durationSeconds,
  };
}

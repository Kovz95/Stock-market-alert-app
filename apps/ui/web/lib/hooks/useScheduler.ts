"use client";

import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  getSchedulerStatus,
  getExchangeSchedule,
  listQueueTasks,
  startScheduler,
  stopScheduler,
  runExchangeJob,
  evaluateExchange,
  type SchedulerStatusData,
  type ExchangeScheduleRow,
  type QueueTaskData,
} from "@/actions/scheduler-actions";

export const SCHEDULER_STATUS_KEY = ["scheduler", "status"] as const;
export const SCHEDULER_SCHEDULE_KEY = ["scheduler", "schedule"] as const;
export const SCHEDULER_QUEUE_TASKS_KEY = ["scheduler", "queue-tasks"] as const;

export type Timeframe = "daily" | "weekly" | "hourly";

export function useSchedulerStatus() {
  return useQuery<SchedulerStatusData>({
    queryKey: [...SCHEDULER_STATUS_KEY],
    queryFn: getSchedulerStatus,
    refetchInterval: 30_000,
  });
}

export function useExchangeSchedule(timeframe: Timeframe = "daily") {
  return useQuery<ExchangeScheduleRow[]>({
    queryKey: [...SCHEDULER_SCHEDULE_KEY, timeframe],
    queryFn: () => getExchangeSchedule(timeframe),
    refetchInterval: timeframe === "hourly" ? 30_000 : 60_000,
  });
}

export function useQueueTasks(queue: string = "") {
  return useQuery<QueueTaskData[]>({
    queryKey: [...SCHEDULER_QUEUE_TASKS_KEY, queue],
    queryFn: () => listQueueTasks(queue),
    refetchInterval: 30_000,
  });
}

export function useStartScheduler() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: startScheduler,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [...SCHEDULER_STATUS_KEY] });
      queryClient.invalidateQueries({ queryKey: [...SCHEDULER_QUEUE_TASKS_KEY] });
    },
  });
}

export function useStopScheduler() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: stopScheduler,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [...SCHEDULER_STATUS_KEY] });
      queryClient.invalidateQueries({ queryKey: [...SCHEDULER_QUEUE_TASKS_KEY] });
    },
  });
}

export function useRunExchangeJob() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ exchange, timeframe }: { exchange: string; timeframe: string }) =>
      runExchangeJob(exchange, timeframe),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [...SCHEDULER_STATUS_KEY] });
      queryClient.invalidateQueries({ queryKey: [...SCHEDULER_SCHEDULE_KEY] });
      queryClient.invalidateQueries({ queryKey: [...SCHEDULER_QUEUE_TASKS_KEY] });
    },
  });
}

export function useEvaluateExchange() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: ({ exchange, timeframe }: { exchange: string; timeframe: string }) =>
      evaluateExchange(exchange, timeframe),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: [...SCHEDULER_STATUS_KEY] });
      queryClient.invalidateQueries({ queryKey: [...SCHEDULER_SCHEDULE_KEY] });
      queryClient.invalidateQueries({ queryKey: [...SCHEDULER_QUEUE_TASKS_KEY] });
    },
  });
}

export type { SchedulerStatusData, ExchangeScheduleRow, QueueTaskData };

"use client";

import { useMemo, useState } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  useExchangeSchedule,
  useEvaluateExchange,
  type ExchangeScheduleRow,
} from "@/lib/hooks/useScheduler";
import { toast } from "sonner";
import type { Timeframe } from "@/lib/hooks/useScheduler";

function formatCountdown(totalSeconds: number): string {
  if (totalSeconds <= 0) return "done";
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

function formatLastRun(row: ExchangeScheduleRow): string {
  if (!row.lastRunDate) return "—";
  return row.lastRunDate;
}

const REGIONS = ["All", "Asia-Pacific", "Europe", "Americas"] as const;

export function ExchangeScheduleTable({ timeframe }: { timeframe: Timeframe }) {
  const { data: schedule, isLoading, error } = useExchangeSchedule(timeframe);
  const evaluate = useEvaluateExchange();
  const [regionFilter, setRegionFilter] = useState<string>("All");
  const [search, setSearch] = useState("");
  const [runningExchange, setRunningExchange] = useState<string | null>(null);

  const filtered = useMemo(() => {
    if (!schedule) return [];
    let rows = [...schedule];
    if (regionFilter !== "All") {
      rows = rows.filter((r) => r.region === regionFilter);
    }
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      rows = rows.filter(
        (r) =>
          r.exchange.toLowerCase().includes(q) ||
          r.symbol.toLowerCase().includes(q)
      );
    }
    // For hourly: open exchanges first (soonest candle), then closed (alphabetical).
    // For daily/weekly: soonest run first, with 0 (done) at the end.
    if (timeframe === "hourly") {
      rows.sort((a, b) => {
        const aOpen = a.timeRemainingSeconds > 0;
        const bOpen = b.timeRemainingSeconds > 0;
        if (aOpen && !bOpen) return -1;
        if (!aOpen && bOpen) return 1;
        if (aOpen && bOpen) return a.timeRemainingSeconds - b.timeRemainingSeconds;
        return a.exchange.localeCompare(b.exchange);
      });
    } else {
      rows.sort((a, b) => {
        if (a.timeRemainingSeconds === 0 && b.timeRemainingSeconds === 0) return 0;
        if (a.timeRemainingSeconds === 0) return 1;
        if (b.timeRemainingSeconds === 0) return -1;
        return a.timeRemainingSeconds - b.timeRemainingSeconds;
      });
    }
    return rows;
  }, [schedule, regionFilter, search]);

  async function handleRunNow(exchange: string) {
    setRunningExchange(exchange);
    const toastId = toast.loading(`Evaluating ${exchange} (${timeframe})…`);
    try {
      const result = await evaluate.mutateAsync({ exchange, timeframe });
      if (result.success) {
        toast.success(
          `${exchange} complete — ${result.alertsTriggered}/${result.alertsTotal} triggered · ${result.pricesUpdated} prices updated · ${result.durationSeconds.toFixed(1)}s`,
          { id: toastId }
        );
      } else {
        toast.error(result.message, { id: toastId });
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "Evaluation failed", { id: toastId });
    } finally {
      setRunningExchange(null);
    }
  }

  const timeframeLabel =
    timeframe.charAt(0).toUpperCase() + timeframe.slice(1);

  if (isLoading) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Exchange Schedule — {timeframeLabel}</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">Loading...</p>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Exchange Schedule — {timeframeLabel}</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-destructive">
            Failed to load exchange schedule:{" "}
            {error instanceof Error ? error.message : String(error)}
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Exchange Schedule — {timeframeLabel}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-col gap-2 sm:flex-row">
          <Input
            placeholder="Search exchanges..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="sm:max-w-xs"
          />
          <Select value={regionFilter} onValueChange={setRegionFilter}>
            <SelectTrigger className="sm:w-[180px]">
              <SelectValue placeholder="Region" />
            </SelectTrigger>
            <SelectContent>
              {REGIONS.map((r) => (
                <SelectItem key={r} value={r}>
                  {r}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="rounded-md border overflow-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Exchange</TableHead>
                <TableHead>Symbol</TableHead>
                <TableHead>Region</TableHead>
                {timeframe === "hourly" ? (
                  <>
                    <TableHead>Status</TableHead>
                    <TableHead>Next Candle (ET)</TableHead>
                    <TableHead>Next Candle (UTC)</TableHead>
                    <TableHead>Alignment</TableHead>
                  </>
                ) : (
                  <>
                    <TableHead>Run (ET)</TableHead>
                    <TableHead>Run (UTC)</TableHead>
                    <TableHead>Close (ET)</TableHead>
                  </>
                )}
                <TableHead>Local TZ</TableHead>
                <TableHead>Time Remaining</TableHead>
                <TableHead>Last Run</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filtered.length === 0 ? (
                <TableRow>
                  <TableCell colSpan={timeframe === "hourly" ? 11 : 10} className="text-center text-muted-foreground">
                    No exchanges found.
                  </TableCell>
                </TableRow>
              ) : (
                filtered.map((row) => (
                  <TableRow key={row.symbol}>
                    <TableCell className="font-medium">{row.exchange}</TableCell>
                    <TableCell>{row.symbol}</TableCell>
                    <TableCell>{row.region}</TableCell>
                    {timeframe === "hourly" ? (
                      <>
                        <TableCell>
                          {row.timeRemainingSeconds > 0 ? (
                            <Badge variant="default" className="text-xs">Open</Badge>
                          ) : (
                            <Badge variant="secondary" className="text-xs">Closed</Badge>
                          )}
                        </TableCell>
                        <TableCell className="tabular-nums">{row.runTimeEt || "—"}</TableCell>
                        <TableCell className="tabular-nums">{row.runTimeUtc || "—"}</TableCell>
                        <TableCell className="tabular-nums">{row.localClose || "—"}</TableCell>
                      </>
                    ) : (
                      <>
                        <TableCell className="tabular-nums">{row.runTimeEt}</TableCell>
                        <TableCell className="tabular-nums">{row.runTimeUtc}</TableCell>
                        <TableCell className="tabular-nums">{row.localClose}</TableCell>
                      </>
                    )}
                    <TableCell className="text-xs">{row.localTz}</TableCell>
                    <TableCell className="tabular-nums">
                      {timeframe === "hourly" && row.timeRemainingSeconds === 0
                        ? "—"
                        : formatCountdown(row.timeRemainingSeconds)}
                    </TableCell>
                    <TableCell>{formatLastRun(row)}</TableCell>
                    <TableCell className="text-right">
                      <Button
                        size="sm"
                        variant="secondary"
                        disabled={runningExchange !== null}
                        onClick={() => handleRunNow(row.symbol)}
                      >
                        {runningExchange === row.symbol ? "Running…" : "Run now"}
                      </Button>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}

"use client";

import { useState, useMemo } from "react";
import {
  useAuditSummary,
  usePerformanceMetrics,
  useAlertHistory,
  useFailedPriceData,
  useClearAuditData,
} from "@/lib/hooks/useAudit";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart";
import { LineChart, Line, XAxis, YAxis, CartesianGrid } from "recharts";
import { BarChart, Bar } from "recharts";
import { PieChart, Pie, Cell } from "recharts";
import { formatAuditDateTime, formatAuditDateShort } from "./_components/formatAuditDate";
import type { AuditSummaryRow, AuditHistoryRow } from "@/actions/audit-actions";
import {
  RefreshCwIcon,
  Trash2Icon,
  DownloadIcon,
  BarChart3Icon,
  AlertTriangleIcon,
  CheckCircleIcon,
  InfoIcon,
} from "lucide-react";

const EVALUATION_TYPES = ["All", "scheduled", "manual", "test"] as const;
const STATUS_OPTIONS = ["All", "Success", "Error", "Triggered", "Not Triggered"] as const;

function filterSummaryRows(
  rows: AuditSummaryRow[],
  filters: {
    alertId: string;
    ticker: string;
    evaluationType: string;
    status: string;
    maxRows: number;
  }
): AuditSummaryRow[] {
  let out = rows;
  if (filters.alertId.trim()) {
    const q = filters.alertId.toLowerCase();
    out = out.filter(
      (r) =>
        r.alertId?.toLowerCase().includes(q)
    );
  }
  if (filters.ticker.trim()) {
    const q = filters.ticker.toUpperCase();
    out = out.filter(
      (r) => r.ticker?.toUpperCase().includes(q)
    );
  }
  if (filters.evaluationType !== "All") {
    out = out.filter((r) => r.evaluationType === filters.evaluationType);
  }
  if (filters.status !== "All") {
    if (filters.status === "Success")
      out = out.filter((r) => (r.successfulEvaluations ?? 0) > 0);
    else if (filters.status === "Error")
      out = out.filter((r) => (r.totalChecks ?? 0) > (r.successfulEvaluations ?? 0));
    else if (filters.status === "Triggered")
      out = out.filter((r) => (r.totalTriggers ?? 0) > 0);
    else if (filters.status === "Not Triggered")
      out = out.filter((r) => (r.totalTriggers ?? 0) === 0);
  }
  return out.slice(0, filters.maxRows);
}

export default function AlertAuditPage() {
  const [days, setDays] = useState(7);
  const [alertIdFilter, setAlertIdFilter] = useState("");
  const [tickerFilter, setTickerFilter] = useState("");
  const [evaluationType, setEvaluationType] = useState<string>("All");
  const [statusFilter, setStatusFilter] = useState<string>("All");
  const [maxRows, setMaxRows] = useState(500);
  const [searchInResults, setSearchInResults] = useState("");
  const [clearConfirm, setClearConfirm] = useState(false);

  const { data: summaryRows, isLoading: summaryLoading, refetch: refetchSummary } = useAuditSummary(days);
  const { data: metrics, isLoading: metricsLoading, refetch: refetchMetrics } = usePerformanceMetrics(days);
  const { data: historyRows, isLoading: historyLoading } = useAlertHistory(alertIdFilter.trim(), 100);
  const { data: failedData, isLoading: failedLoading, refetch: refetchFailed } = useFailedPriceData(days);
  const clearMutation = useClearAuditData();

  const filteredSummary = useMemo(
    () =>
      filterSummaryRows(summaryRows ?? [], {
        alertId: alertIdFilter,
        ticker: tickerFilter,
        evaluationType,
        status: statusFilter,
        maxRows,
      }),
    [summaryRows, alertIdFilter, tickerFilter, evaluationType, statusFilter, maxRows]
  );

  const searchFilteredSummary = useMemo(() => {
    if (!searchInResults.trim()) return filteredSummary;
    const q = searchInResults.toLowerCase();
    return filteredSummary.filter((r) => {
      const str = [
        r.alertId,
        r.ticker,
        r.stockName,
        r.exchange,
        r.timeframe,
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      return str.includes(q);
    });
  }, [filteredSummary, searchInResults]);

  const refetchAll = () => {
    refetchSummary();
    refetchMetrics();
    refetchFailed();
  };

  const handleClearAudit = async () => {
    if (!clearConfirm) {
      setClearConfirm(true);
      return;
    }
    await clearMutation.mutateAsync();
    setClearConfirm(false);
    refetchAll();
  };

  const exportSummaryCsv = () => {
    const rows = filteredSummary;
    if (rows.length === 0) return;
    const headers = [
      "Alert ID",
      "Ticker",
      "Stock Name",
      "Exchange",
      "Timeframe",
      "Action",
      "Type",
      "Checks",
      "Price Pulls",
      "Evaluations",
      "Triggers",
      "Avg Time (ms)",
      "Last Check",
      "First Check",
    ];
    const toStr = (d: Date | undefined) => (d ? formatAuditDateShort(d) : "");
    const lines = [
      headers.join(","),
      ...rows.map((r) =>
        [
          r.alertId,
          r.ticker,
          (r.stockName ?? "").replace(/"/g, '""'),
          r.exchange,
          r.timeframe,
          r.action,
          r.evaluationType,
          r.totalChecks,
          r.successfulPricePulls,
          r.successfulEvaluations,
          r.totalTriggers,
          r.avgExecutionTimeMs?.toFixed(1) ?? "",
          toStr(r.lastCheck),
          toStr(r.firstCheck),
        ].join(",")
      ),
    ];
    const blob = new Blob([lines.join("\n")], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `alert_audit_summary_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const executionChartConfig = {
    time: { label: "Time" },
    executionTimeMs: { label: "Execution Time (ms)", color: "hsl(var(--chart-1))" },
  } satisfies ChartConfig;

  const executionChartData = useMemo(() => {
    if (!historyRows?.length || historyRows.length < 2) return [];
    return [...(historyRows ?? [])]
      .sort(
        (a, b) =>
          new Date(a.timestamp ?? 0).getTime() - new Date(b.timestamp ?? 0).getTime()
      )
      .filter((r) => r.executionTimeMs != null)
      .map((r) => ({
        time: formatAuditDateTime(r.timestamp),
        executionTimeMs: r.executionTimeMs,
      }));
  }, [historyRows]);

  const successRateByTimeframe = useMemo(() => {
    if (!summaryRows?.length) return [];
    const byTf = new Map<
      string,
      { total: number; success: number }
    >();
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

  const isLoading = summaryLoading || metricsLoading;
  const hasSummary = (summaryRows?.length ?? 0) > 0;

  return (
    <div className="p-6 space-y-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl font-bold flex items-center gap-2">
          <BarChart3Icon className="size-6" />
          Alert Audit Logs & Analytics
        </h1>
        <p className="text-muted-foreground text-sm">
          Tracking of alert evaluations, performance metrics, and system health.
        </p>
      </div>

      {/* Filters */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Filters & options</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-wrap gap-4 items-end">
          <div className="flex flex-col gap-1">
            <label className="text-xs text-muted-foreground">Days to analyze</label>
            <Select
              value={String(days)}
              onValueChange={(v) => setDays(Number(v))}
            >
              <SelectTrigger className="w-[100px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {[1, 7, 14, 30, 60, 90].map((d) => (
                  <SelectItem key={d} value={String(d)}>
                    {d} days
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-muted-foreground">Alert ID</label>
            <Input
              placeholder="Filter by alert ID"
              value={alertIdFilter}
              onChange={(e) => setAlertIdFilter(e.target.value)}
              className="w-[180px]"
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-muted-foreground">Ticker</label>
            <Input
              placeholder="Filter by ticker"
              value={tickerFilter}
              onChange={(e) => setTickerFilter(e.target.value)}
              className="w-[120px]"
            />
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-muted-foreground">Evaluation type</label>
            <Select value={evaluationType} onValueChange={setEvaluationType}>
              <SelectTrigger className="w-[120px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {EVALUATION_TYPES.map((t) => (
                  <SelectItem key={t} value={t}>
                    {t}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-muted-foreground">Status</label>
            <Select value={statusFilter} onValueChange={setStatusFilter}>
              <SelectTrigger className="w-[140px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {STATUS_OPTIONS.map((s) => (
                  <SelectItem key={s} value={s}>
                    {s}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="flex flex-col gap-1">
            <label className="text-xs text-muted-foreground">Max rows</label>
            <Select value={String(maxRows)} onValueChange={(v) => setMaxRows(Number(v))}>
              <SelectTrigger className="w-[100px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {[100, 500, 1000, 2000, 5000].map((n) => (
                  <SelectItem key={n} value={String(n)}>
                    {n}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* Performance overview + quick actions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 space-y-4">
          <h2 className="text-lg font-semibold">Performance overview</h2>
          {isLoading ? (
            <p className="text-muted-foreground text-sm">Loading metrics…</p>
          ) : metrics ? (
            <>
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
            </>
          ) : (
            <p className="text-muted-foreground text-sm">No audit data for the selected period.</p>
          )}
        </div>
        <div className="space-y-3">
          <h2 className="text-lg font-semibold">Quick actions</h2>
          <div className="flex flex-col gap-2">
            <Button variant="outline" size="sm" onClick={() => refetchAll()}>
              <RefreshCwIcon className="size-3.5 mr-1" />
              Refresh data
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={handleClearAudit}
              disabled={clearMutation.isPending}
              className={clearConfirm ? "border-destructive text-destructive" : ""}
            >
              <Trash2Icon className="size-3.5 mr-1" />
              {clearConfirm ? "Click again to clear all" : "Clear all audit data"}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={exportSummaryCsv}
              disabled={!hasSummary}
            >
              <DownloadIcon className="size-3.5 mr-1" />
              Export summary CSV
            </Button>
          </div>
        </div>
      </div>

      {/* Alert evaluation summary table */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Alert evaluation summary</CardTitle>
          <p className="text-muted-foreground text-xs">
            Showing data for the last {days} days. Found {searchFilteredSummary.length} alerts
            matching filters.
          </p>
        </CardHeader>
        <CardContent className="space-y-3">
          {summaryLoading ? (
            <p className="text-sm text-muted-foreground">Loading summary…</p>
          ) : !hasSummary ? (
            <p className="text-sm text-muted-foreground">
              No audit data. Alerts will appear here once evaluated.
            </p>
          ) : (
            <>
              <Input
                placeholder="Search in results (alert ID, ticker, stock name…)"
                value={searchInResults}
                onChange={(e) => setSearchInResults(e.target.value)}
                className="max-w-sm"
              />
              <div className="border rounded-md overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Alert ID</TableHead>
                      <TableHead>Ticker</TableHead>
                      <TableHead>Stock name</TableHead>
                      <TableHead>Exchange</TableHead>
                      <TableHead>Timeframe</TableHead>
                      <TableHead>Action</TableHead>
                      <TableHead>Type</TableHead>
                      <TableHead className="text-right">Checks</TableHead>
                      <TableHead className="text-right">Evaluations</TableHead>
                      <TableHead className="text-right">Triggers</TableHead>
                      <TableHead className="text-right">Avg (ms)</TableHead>
                      <TableHead>Last check</TableHead>
                      <TableHead>First check</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {searchFilteredSummary.map((r, i) => (
                      <TableRow key={`${r.alertId}-${r.ticker}-${r.evaluationType}-${i}`}>
                        <TableCell className="font-mono text-xs">{r.alertId}</TableCell>
                        <TableCell>{r.ticker}</TableCell>
                        <TableCell>{r.stockName ?? "—"}</TableCell>
                        <TableCell>{r.exchange ?? "—"}</TableCell>
                        <TableCell>{r.timeframe ?? "—"}</TableCell>
                        <TableCell>{r.action ?? "—"}</TableCell>
                        <TableCell>{r.evaluationType}</TableCell>
                        <TableCell className="text-right">{r.totalChecks?.toLocaleString() ?? 0}</TableCell>
                        <TableCell className="text-right">{r.successfulEvaluations?.toLocaleString() ?? 0}</TableCell>
                        <TableCell className="text-right">{r.totalTriggers?.toLocaleString() ?? 0}</TableCell>
                        <TableCell className="text-right">
                          {r.avgExecutionTimeMs != null ? r.avgExecutionTimeMs.toFixed(1) : "—"}
                        </TableCell>
                        <TableCell className="text-xs whitespace-nowrap">
                          {formatAuditDateShort(r.lastCheck)}
                        </TableCell>
                        <TableCell className="text-xs whitespace-nowrap">
                          {formatAuditDateShort(r.firstCheck)}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {/* Detailed alert history */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Detailed alert history</CardTitle>
          <p className="text-muted-foreground text-xs">
            {alertIdFilter.trim()
              ? `History for Alert ID: ${alertIdFilter}`
              : "Enter an Alert ID in the filters above to view detailed history."}
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          {!alertIdFilter.trim() ? (
            <p className="text-sm text-muted-foreground">Enter an Alert ID above to view history.</p>
          ) : historyLoading ? (
            <p className="text-sm text-muted-foreground">Loading history…</p>
          ) : !historyRows?.length ? (
            <p className="text-sm text-muted-foreground">No history found for this alert.</p>
          ) : (
            <>
              <div className="border rounded-md overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Timestamp</TableHead>
                      <TableHead>Ticker</TableHead>
                      <TableHead>Type</TableHead>
                      <TableHead>Price data</TableHead>
                      <TableHead>Source</TableHead>
                      <TableHead>Cache</TableHead>
                      <TableHead>Evaluated</TableHead>
                      <TableHead>Triggered</TableHead>
                      <TableHead>Trigger reason</TableHead>
                      <TableHead className="text-right">Time (ms)</TableHead>
                      <TableHead>Error</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {historyRows.map((r: AuditHistoryRow) => (
                      <TableRow key={r.id ?? `${r.timestamp}-${r.ticker}`}>
                        <TableCell className="text-xs whitespace-nowrap">
                          {formatAuditDateTime(r.timestamp)}
                        </TableCell>
                        <TableCell>{r.ticker}</TableCell>
                        <TableCell>{r.evaluationType}</TableCell>
                        <TableCell>{r.priceDataPulled ? "Yes" : "No"}</TableCell>
                        <TableCell>{r.priceDataSource ?? "—"}</TableCell>
                        <TableCell>{r.cacheHit ? "Yes" : "No"}</TableCell>
                        <TableCell>{r.conditionsEvaluated ? "Yes" : "No"}</TableCell>
                        <TableCell>{r.alertTriggered ? "Yes" : "No"}</TableCell>
                        <TableCell className="max-w-[200px] truncate" title={r.triggerReason ?? ""}>
                          {r.triggerReason ?? "—"}
                        </TableCell>
                        <TableCell className="text-right">
                          {r.executionTimeMs != null ? r.executionTimeMs : "—"}
                        </TableCell>
                        <TableCell className="max-w-[180px] truncate text-destructive" title={r.errorMessage ?? ""}>
                          {r.errorMessage || "—"}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
              {executionChartData.length > 1 && (
                <div>
                  <h3 className="text-sm font-medium mb-2">Performance trend (execution time)</h3>
                  <ChartContainer config={executionChartConfig} className="h-[240px] w-full">
                    <LineChart data={executionChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" tick={{ fontSize: 10 }} />
                      <YAxis tick={{ fontSize: 10 }} />
                      <ChartTooltip content={<ChartTooltipContent />} />
                      <Line
                        type="monotone"
                        dataKey="executionTimeMs"
                        stroke="var(--chart-1)"
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ChartContainer>
                </div>
              )}
            </>
          )}
        </CardContent>
      </Card>

      {/* Analytics: success rate by timeframe */}
      {hasSummary && successRateByTimeframe.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Success rate by timeframe</CardTitle>
          </CardHeader>
          <CardContent>
            <ChartContainer
              config={{
                timeframe: { label: "Timeframe" },
                successRate: { label: "Success rate (%)", color: "hsl(var(--chart-2))" },
              }}
              className="h-[260px] w-full"
            >
              <BarChart data={successRateByTimeframe} margin={{ left: 12, right: 12 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timeframe" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10 }} unit="%" />
                <ChartTooltip content={<ChartTooltipContent />} />
                <Bar dataKey="successRate" fill="var(--chart-2)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ChartContainer>
          </CardContent>
        </Card>
      )}

      {/* Failed price data analysis */}
      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <div>
            <CardTitle className="text-base">Failed price data analysis</CardTitle>
            <p className="text-muted-foreground text-xs">
              Alerts that failed to retrieve price data (last {days} days).
            </p>
          </div>
          <Button variant="outline" size="sm" onClick={() => refetchFailed()}>
            <RefreshCwIcon className="size-3.5 mr-1" />
            Refresh failed data
          </Button>
        </CardHeader>
        <CardContent className="space-y-6">
          {failedLoading ? (
            <p className="text-sm text-muted-foreground">Loading failed data…</p>
          ) : !failedData ? (
            <p className="text-sm text-muted-foreground">Unable to load failed price data.</p>
          ) : failedData.rows.length === 0 ? (
            <p className="text-sm text-green-600">No failed price data retrievals in this period.</p>
          ) : (
            <>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <Card size="sm">
                  <CardContent className="pt-3">
                    <p className="text-muted-foreground text-xs">Failed alerts</p>
                    <p className="text-lg font-semibold">
                      {failedData.totalFailedAlerts?.toLocaleString() ?? 0}
                    </p>
                  </CardContent>
                </Card>
                <Card size="sm">
                  <CardContent className="pt-3">
                    <p className="text-muted-foreground text-xs">Total failures</p>
                    <p className="text-lg font-semibold">
                      {failedData.totalFailures?.toLocaleString() ?? 0}
                    </p>
                  </CardContent>
                </Card>
                <Card size="sm">
                  <CardContent className="pt-3">
                    <p className="text-muted-foreground text-xs">Avg failures/alert</p>
                    <p className="text-lg font-semibold">
                      {(
                        (failedData.totalFailures ?? 0) /
                        Math.max(failedData.totalFailedAlerts ?? 1, 1)
                      ).toFixed(1)}
                    </p>
                  </CardContent>
                </Card>
                <Card size="sm">
                  <CardContent className="pt-3">
                    <p className="text-muted-foreground text-xs">Failure rate</p>
                    <p className="text-lg font-semibold">
                      {(failedData.failureRate ?? 0).toFixed(1)}%
                    </p>
                  </CardContent>
                </Card>
              </div>
              {failedData.assetTypeBreakdown?.length > 0 && (
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h3 className="text-sm font-medium mb-2">Failures by asset type</h3>
                    <ul className="text-sm text-muted-foreground space-y-1">
                      {failedData.assetTypeBreakdown.map((row) => (
                        <li key={row.assetType}>
                          <strong>{row.assetType}</strong>: {row.failedAlerts} alerts,{" "}
                          {row.failureCount} failures
                        </li>
                      ))}
                    </ul>
                  </div>
                  <ChartContainer
                    config={{
                      name: { label: "Asset type" },
                      value: { label: "Failed alerts" },
                    }}
                    className="h-[200px] w-full"
                  >
                    <PieChart>
                      <Pie
                        data={(failedData.assetTypeBreakdown ?? []).map((r) => ({
                          name: r.assetType,
                          value: r.failedAlerts,
                        }))}
                        dataKey="value"
                        nameKey="name"
                        cx="50%"
                        cy="50%"
                        outerRadius={80}
                        label={({ name, value }) => `${name}: ${value}`}
                      >
                        {(failedData.assetTypeBreakdown ?? []).map((_, i) => (
                          <Cell
                            key={i}
                            fill={`hsl(var(--chart-${(i % 5) + 1}))`}
                          />
                        ))}
                      </Pie>
                      <ChartTooltip content={<ChartTooltipContent />} />
                    </PieChart>
                  </ChartContainer>
                </div>
              )}
              {failedData.exchangeBreakdown?.length > 0 && (
                <div>
                  <h3 className="text-sm font-medium mb-2">Top exchanges by failure count</h3>
                  <div className="border rounded-md overflow-x-auto">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Exchange</TableHead>
                          <TableHead className="text-right">Failed alerts</TableHead>
                          <TableHead className="text-right">Failures</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {failedData.exchangeBreakdown
                          .slice(0, 15)
                          .map((row) => (
                            <TableRow key={row.exchange}>
                              <TableCell>{row.exchange}</TableCell>
                              <TableCell className="text-right">{row.failedAlerts}</TableCell>
                              <TableCell className="text-right">{row.failureCount}</TableCell>
                            </TableRow>
                          ))}
                      </TableBody>
                    </Table>
                  </div>
                </div>
              )}
              <div>
                <h3 className="text-sm font-medium mb-2">Top 20 alerts with most failures</h3>
                <div className="border rounded-md overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Alert ID</TableHead>
                        <TableHead>Ticker</TableHead>
                        <TableHead>Asset type</TableHead>
                        <TableHead>Stock name</TableHead>
                        <TableHead>Exchange</TableHead>
                        <TableHead>Timeframe</TableHead>
                        <TableHead className="text-right">Failures</TableHead>
                        <TableHead>Last failure</TableHead>
                        <TableHead>First failure</TableHead>
                        <TableHead className="text-right">Avg (ms)</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {failedData.rows.slice(0, 20).map((r) => (
                        <TableRow key={`${r.alertId}-${r.ticker}`}>
                          <TableCell className="font-mono text-xs">{r.alertId}</TableCell>
                          <TableCell>{r.ticker}</TableCell>
                          <TableCell>{r.assetType ?? "Unknown"}</TableCell>
                          <TableCell>{r.stockName ?? "—"}</TableCell>
                          <TableCell>{r.exchange ?? "—"}</TableCell>
                          <TableCell>{r.timeframe ?? "—"}</TableCell>
                          <TableCell className="text-right">{r.failureCount}</TableCell>
                          <TableCell className="text-xs whitespace-nowrap">
                            {formatAuditDateShort(r.lastFailure)}
                          </TableCell>
                          <TableCell className="text-xs whitespace-nowrap">
                            {formatAuditDateShort(r.firstFailure)}
                          </TableCell>
                          <TableCell className="text-right">
                            {r.avgExecutionTime != null
                              ? r.avgExecutionTime.toFixed(0)
                              : "—"}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </div>
            </>
          )}
        </CardContent>
      </Card>

      <p className="text-muted-foreground text-xs border-t pt-4">
        Use the filters to focus on specific alerts, timeframes, or statuses. Export data to CSV for
        external analysis.
      </p>
    </div>
  );
}

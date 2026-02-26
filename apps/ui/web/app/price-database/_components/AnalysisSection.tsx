"use client";

import * as React from "react";
import type { PriceRowData } from "@/actions/price-database-actions";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

function formatNum(n: number): string {
  if (Number.isInteger(n)) return n.toLocaleString();
  return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 });
}

type TickerStats = {
  ticker: string;
  count: number;
  mean: number;
  std: number;
  min: number;
  max: number;
  totalVolume: number;
  dateMin: string | null;
  dateMax: string | null;
};

type RecentChange = {
  ticker: string;
  latestClose: number;
  previousClose: number;
  change: number;
  changePct: number;
};

function computeSummaryByTicker(data: PriceRowData[]): TickerStats[] {
  const byTicker = new Map<string, PriceRowData[]>();
  for (const row of data) {
    const list = byTicker.get(row.ticker) ?? [];
    list.push(row);
    byTicker.set(row.ticker, list);
  }
  const result: TickerStats[] = [];
  for (const [ticker, rows] of byTicker) {
    const closes = rows.map((r) => r.close).filter((c) => Number.isFinite(c));
    const n = closes.length;
    if (n === 0) {
      result.push({
        ticker,
        count: rows.length,
        mean: 0,
        std: 0,
        min: 0,
        max: 0,
        totalVolume: rows.reduce((s, r) => s + r.volume, 0),
        dateMin: rows[0]?.time ?? null,
        dateMax: rows[rows.length - 1]?.time ?? null,
      });
      continue;
    }
    const mean = closes.reduce((a, b) => a + b, 0) / n;
    const variance =
      closes.reduce((s, c) => s + (c - mean) ** 2, 0) / n;
    const std = Math.sqrt(variance);
    const min = Math.min(...closes);
    const max = Math.max(...closes);
    const sorted = [...rows].sort((a, b) => (a.time && b.time ? a.time.localeCompare(b.time) : 0));
    result.push({
      ticker,
      count: n,
      mean,
      std,
      min,
      max,
      totalVolume: rows.reduce((s, r) => s + r.volume, 0),
      dateMin: sorted[0]?.time ?? null,
      dateMax: sorted[sorted.length - 1]?.time ?? null,
    });
  }
  result.sort((a, b) => a.ticker.localeCompare(b.ticker));
  return result;
}

function computeRecentChanges(data: PriceRowData[]): RecentChange[] {
  const byTicker = new Map<string, PriceRowData[]>();
  for (const row of data) {
    const list = byTicker.get(row.ticker) ?? [];
    list.push(row);
    byTicker.set(row.ticker, list);
  }
  const result: RecentChange[] = [];
  for (const [ticker, rows] of byTicker) {
    const sorted = [...rows].sort((a, b) => (a.time && b.time ? a.time.localeCompare(b.time) : 0));
    if (sorted.length < 2) continue;
    const latest = sorted[sorted.length - 1];
    const previous = sorted[sorted.length - 2];
    const latestClose = latest.close;
    const previousClose = previous.close;
    if (!Number.isFinite(latestClose) || !Number.isFinite(previousClose) || previousClose === 0) continue;
    const change = latestClose - previousClose;
    const changePct = (change / previousClose) * 100;
    result.push({ ticker, latestClose, previousClose, change, changePct });
  }
  result.sort((a, b) => Math.abs(b.changePct) - Math.abs(a.changePct));
  return result;
}

type AnalysisSectionProps = {
  data: PriceRowData[];
};

export function AnalysisSection({ data }: AnalysisSectionProps) {
  const summary = React.useMemo(() => computeSummaryByTicker(data), [data]);
  const recentChanges = React.useMemo(() => computeRecentChanges(data), [data]);

  if (data.length === 0) {
    return (
      <div className="rounded-lg border bg-muted/20 py-12 text-center text-muted-foreground">
        No data. Load price data first.
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Summary by ticker</CardTitle>
          <p className="text-muted-foreground text-sm">
            Count, mean, std, min, max (close), total volume, date range
          </p>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto rounded border">
            <Table>
              <TableHeader>
                <TableRow className="bg-muted/50">
                  <TableHead>Ticker</TableHead>
                  <TableHead className="text-right">N</TableHead>
                  <TableHead className="text-right">Mean</TableHead>
                  <TableHead className="text-right">Std</TableHead>
                  <TableHead className="text-right">Min</TableHead>
                  <TableHead className="text-right">Max</TableHead>
                  <TableHead className="text-right">Volume</TableHead>
                  <TableHead>From</TableHead>
                  <TableHead>To</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {summary.map((s) => (
                  <TableRow key={s.ticker}>
                    <TableCell className="font-medium">{s.ticker}</TableCell>
                    <TableCell className="text-right tabular-nums">{s.count}</TableCell>
                    <TableCell className="text-right tabular-nums">{formatNum(s.mean)}</TableCell>
                    <TableCell className="text-right tabular-nums">{formatNum(s.std)}</TableCell>
                    <TableCell className="text-right tabular-nums">{formatNum(s.min)}</TableCell>
                    <TableCell className="text-right tabular-nums">{formatNum(s.max)}</TableCell>
                    <TableCell className="text-right tabular-nums">
                      {s.totalVolume.toLocaleString()}
                    </TableCell>
                    <TableCell className="text-muted-foreground text-xs">
                      {s.dateMin ? new Date(s.dateMin).toLocaleDateString() : "—"}
                    </TableCell>
                    <TableCell className="text-muted-foreground text-xs">
                      {s.dateMax ? new Date(s.dateMax).toLocaleDateString() : "—"}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Recent price changes</CardTitle>
          <p className="text-muted-foreground text-sm">
            Latest bar close vs previous bar close, % change (sorted by |%|)
          </p>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto rounded border">
            <Table>
              <TableHeader>
                <TableRow className="bg-muted/50">
                  <TableHead>Ticker</TableHead>
                  <TableHead className="text-right">Prev close</TableHead>
                  <TableHead className="text-right">Latest close</TableHead>
                  <TableHead className="text-right">Change</TableHead>
                  <TableHead className="text-right">%</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {recentChanges.map((r) => (
                  <TableRow key={r.ticker}>
                    <TableCell className="font-medium">{r.ticker}</TableCell>
                    <TableCell className="text-right tabular-nums">
                      {formatNum(r.previousClose)}
                    </TableCell>
                    <TableCell className="text-right tabular-nums">
                      {formatNum(r.latestClose)}
                    </TableCell>
                    <TableCell
                      className={`text-right tabular-nums ${
                        r.change >= 0 ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"
                      }`}
                    >
                      {r.change >= 0 ? "+" : ""}{formatNum(r.change)}
                    </TableCell>
                    <TableCell
                      className={`text-right tabular-nums ${
                        r.changePct >= 0 ? "text-green-600 dark:text-green-400" : "text-red-600 dark:text-red-400"
                      }`}
                    >
                      {r.changePct >= 0 ? "+" : ""}{r.changePct.toFixed(2)}%
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
          {recentChanges.length === 0 && (
            <p className="text-muted-foreground text-sm py-4">
              Need at least 2 bars per ticker to show recent changes.
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

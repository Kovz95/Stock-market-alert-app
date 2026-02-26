"use client";

import * as React from "react";
import {
  createChart,
  CandlestickSeries,
  HistogramSeries,
  ColorType,
} from "lightweight-charts";
import type { PriceRowData } from "@/actions/price-database-actions";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

function toChartTime(iso: string | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  const y = d.getUTCFullYear();
  const m = String(d.getUTCMonth() + 1).padStart(2, "0");
  const day = String(d.getUTCDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function toChartTimeUTC(iso: string | null): number {
  if (!iso) return 0;
  return Math.floor(new Date(iso).getTime() / 1000) as unknown as import("lightweight-charts").UTCTimestamp;
}

function formatNum(n: number): string {
  if (Number.isInteger(n)) return n.toLocaleString();
  return n.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 4 });
}

/** Resolve a CSS color (e.g. hsl(var(--foreground))) to rgb(r,g,b). Handles lab()/oklch() computed values. */
function resolveCssColor(cssColor: string): string {
  if (typeof document === "undefined") return "rgb(113, 113, 122)";
  const div = document.createElement("div");
  div.style.cssText = `color: ${cssColor}; position: absolute; visibility: hidden;`;
  document.body.appendChild(div);
  const computed = getComputedStyle(div).color;
  document.body.removeChild(div);
  if (!computed) return "rgb(113, 113, 122)";
  if (/^rgb/.test(computed)) return computed;
  const canvas = document.createElement("canvas");
  canvas.width = 1;
  canvas.height = 1;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "rgb(113, 113, 122)";
  ctx.fillStyle = computed;
  ctx.fillRect(0, 0, 1, 1);
  const [r, g, b] = ctx.getImageData(0, 0, 1, 1).data;
  return `rgb(${r}, ${g}, ${b})`;
}

type PriceChartsSectionProps = {
  data: PriceRowData[];
};

export function PriceChartsSection({ data }: PriceChartsSectionProps) {
  const chartRef = React.useRef<HTMLDivElement>(null);
  const chartInstance = React.useRef<ReturnType<typeof createChart> | null>(null);
  const [selectedTicker, setSelectedTicker] = React.useState<string>("");

  const tickers = React.useMemo(() => {
    const set = new Set(data.map((r) => r.ticker));
    return Array.from(set).sort();
  }, [data]);

  const seriesByTicker = React.useMemo(() => {
    const map = new Map<string, PriceRowData[]>();
    for (const row of data) {
      const list = map.get(row.ticker) ?? [];
      list.push(row);
      map.set(row.ticker, list);
    }
    for (const list of map.values()) {
      list.sort((a, b) => (a.time && b.time ? a.time.localeCompare(b.time) : 0));
    }
    return map;
  }, [data]);

  const chartData = React.useMemo(() => {
    if (!selectedTicker) return { candlestick: [], volume: [] };
    const rows = seriesByTicker.get(selectedTicker) ?? [];
    const isIntraday = rows.some((r) => r.time?.includes("T") && r.time.length > 10);
    const candlestick = rows.map((r) => {
      const t = isIntraday ? toChartTimeUTC(r.time) : toChartTime(r.time);
      return {
        time: t as import("lightweight-charts").Time,
        open: r.open,
        high: r.high,
        low: r.low,
        close: r.close,
      };
    });
    const volume = rows.map((r) => {
      const t = isIntraday ? toChartTimeUTC(r.time) : toChartTime(r.time);
      const color = r.close >= r.open ? "rgba(34, 171, 148, 0.5)" : "rgba(255, 82, 82, 0.5)";
      return {
        time: t as import("lightweight-charts").Time,
        value: r.volume,
        color,
      };
    });
    return { candlestick, volume };
  }, [selectedTicker, seriesByTicker]);

  const latestOHLCV = React.useMemo(() => {
    if (!selectedTicker) return null;
    const rows = seriesByTicker.get(selectedTicker) ?? [];
    if (rows.length === 0) return null;
    const last = rows[rows.length - 1];
    return {
      open: last.open,
      high: last.high,
      low: last.low,
      close: last.close,
      volume: last.volume,
    };
  }, [selectedTicker, seriesByTicker]);

  React.useEffect(() => {
    if (!selectedTicker && tickers.length > 0) {
      setSelectedTicker(tickers[0]);
    }
  }, [selectedTicker, tickers]);

  React.useEffect(() => {
    if (!chartRef.current || !selectedTicker || chartData.candlestick.length === 0) return;

    const textColor = resolveCssColor("hsl(var(--foreground))");
    const borderColor = resolveCssColor("hsl(var(--border))");

    const chart = createChart(chartRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor,
      },
      grid: { vertLines: { color: borderColor }, horzLines: { color: borderColor } },
      width: chartRef.current.clientWidth,
      height: 360,
      timeScale: { timeVisible: true, secondsVisible: false },
      rightPriceScale: { borderColor },
    });

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "rgba(34, 171, 148, 1)",
      downColor: "rgba(255, 82, 82, 1)",
      borderDownColor: "rgba(255, 82, 82, 1)",
      borderUpColor: "rgba(34, 171, 148, 1)",
    });
    candleSeries.setData(chartData.candlestick);

    const volSeries = chart.addSeries(HistogramSeries, {
      color: "rgba(100, 116, 139, 0.5)",
      priceFormat: { type: "volume" },
      priceScaleId: "",
    });
    volSeries.priceScale().applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
    volSeries.setData(chartData.volume);

    chart.timeScale().fitContent();
    chartInstance.current = chart;

    const handleResize = () => {
      if (chartRef.current && chartInstance.current) {
        chartInstance.current.applyOptions({ width: chartRef.current.clientWidth });
      }
    };
    const ro = new ResizeObserver(handleResize);
    ro.observe(chartRef.current);

    return () => {
      ro.disconnect();
      chart.remove();
      chartInstance.current = null;
    };
  }, [selectedTicker, chartData.candlestick, chartData.volume]);

  if (data.length === 0) {
    return (
      <div className="rounded-lg border bg-muted/20 py-12 text-center text-muted-foreground">
        No data. Load price data first.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-4">
        <div className="space-y-2">
          <Label>Ticker</Label>
          <Select
            value={selectedTicker || (tickers[0] ?? "")}
            onValueChange={setSelectedTicker}
          >
            <SelectTrigger className="w-32">
              <SelectValue placeholder="Select ticker" />
            </SelectTrigger>
            <SelectContent>
              <SelectGroup>
                {tickers.map((t) => (
                  <SelectItem key={t} value={t}>
                    {t}
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>
        </div>
      </div>

      {latestOHLCV && (
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              Latest bar — {selectedTicker}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-6 text-sm">
              <span><strong>O</strong> {formatNum(latestOHLCV.open)}</span>
              <span><strong>H</strong> {formatNum(latestOHLCV.high)}</span>
              <span><strong>L</strong> {formatNum(latestOHLCV.low)}</span>
              <span><strong>C</strong> {formatNum(latestOHLCV.close)}</span>
              <span><strong>Vol</strong> {latestOHLCV.volume.toLocaleString()}</span>
            </div>
          </CardContent>
        </Card>
      )}

      <div ref={chartRef} className="rounded-lg border bg-card" />
    </div>
  );
}

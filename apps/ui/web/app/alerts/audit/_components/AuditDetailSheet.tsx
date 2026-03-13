"use client";

import { useAtom, useAtomValue } from "jotai";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
} from "@/components/ui/sheet";
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
import { useAlertHistory } from "@/lib/hooks/useAudit";
import {
  auditDetailSheetOpenAtom,
  auditSelectedAlertIdAtom,
} from "@/lib/store/audit";
import { formatAuditDateTime } from "./formatAuditDate";
import { useMemo } from "react";

const executionChartConfig = {
  time: { label: "Time" },
  executionTimeMs: { label: "Execution Time (ms)", color: "hsl(var(--chart-1))" },
} satisfies ChartConfig;

export function AuditDetailSheet() {
  const [open, setOpen] = useAtom(auditDetailSheetOpenAtom);
  const alertId = useAtomValue(auditSelectedAlertIdAtom);
  const { data: historyRows, isLoading } = useAlertHistory(alertId ?? "", 200);

  const executionChartData = useMemo(() => {
    if (!historyRows?.length || historyRows.length < 2) return [];
    return [...historyRows]
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

  return (
    <Sheet open={open} onOpenChange={setOpen}>
      <SheetContent className="w-[600px] sm:max-w-[600px] overflow-y-auto">
        <SheetHeader>
          <SheetTitle>Alert History</SheetTitle>
          <SheetDescription className="font-mono text-xs">
            {alertId ?? "No alert selected"}
          </SheetDescription>
        </SheetHeader>

        <div className="mt-4 space-y-4">
          {isLoading ? (
            <p className="text-sm text-muted-foreground">Loading history...</p>
          ) : !historyRows?.length ? (
            <p className="text-sm text-muted-foreground">No history found for this alert.</p>
          ) : (
            <>
              {executionChartData.length > 1 && (
                <div>
                  <h3 className="text-sm font-medium mb-2">Execution time trend</h3>
                  <ChartContainer config={executionChartConfig} className="h-[200px] w-full">
                    <LineChart data={executionChartData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" tick={{ fontSize: 9 }} />
                      <YAxis tick={{ fontSize: 9 }} />
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

              <div className="border rounded-md overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Timestamp</TableHead>
                      <TableHead>Type</TableHead>
                      <TableHead>Price</TableHead>
                      <TableHead>Cache</TableHead>
                      <TableHead>Triggered</TableHead>
                      <TableHead className="text-right">Time (ms)</TableHead>
                      <TableHead>Error</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {historyRows.map((r) => (
                      <TableRow key={r.id}>
                        <TableCell className="text-xs whitespace-nowrap">
                          {formatAuditDateTime(r.timestamp)}
                        </TableCell>
                        <TableCell>{r.evaluationType}</TableCell>
                        <TableCell>{r.priceDataPulled ? "Yes" : "No"}</TableCell>
                        <TableCell>{r.cacheHit ? "Yes" : "No"}</TableCell>
                        <TableCell>{r.alertTriggered ? "Yes" : "No"}</TableCell>
                        <TableCell className="text-right">
                          {r.executionTimeMs != null ? r.executionTimeMs : "—"}
                        </TableCell>
                        <TableCell className="text-destructive text-xs break-all">
                          {r.errorMessage || "—"}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </>
          )}
        </div>
      </SheetContent>
    </Sheet>
  );
}

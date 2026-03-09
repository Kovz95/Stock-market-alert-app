"use client";

import type { AlertData } from "@/actions/alert-actions";
import { Checkbox } from "@/components/ui/checkbox";
import { formatAlertDate } from "@/app/alerts/_components/formatDate";

function extractConditionText(conditions: unknown): string {
  if (!Array.isArray(conditions)) return "";
  const parts: string[] = [];
  for (const c of conditions) {
    if (c && typeof c === "object" && "conditions" in c) {
      parts.push(String((c as { conditions: string }).conditions));
    }
  }
  return parts.join(" | ");
}

type DeleteAlertsTableProps = {
  alerts: AlertData[];
  selected: Set<string>;
  onToggle: (alertId: string) => void;
  onToggleAll: (checked: boolean) => void;
};

export function DeleteAlertsTable({
  alerts,
  selected,
  onToggle,
  onToggleAll,
}: DeleteAlertsTableProps) {
  const allSelected = alerts.length > 0 && alerts.every((a) => selected.has(a.alertId));
  const someSelected = alerts.some((a) => selected.has(a.alertId)) && !allSelected;

  return (
    <div className="border rounded-lg overflow-hidden">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b bg-muted/50">
            <th className="p-3 w-10">
              <Checkbox
                checked={allSelected ? true : someSelected ? "indeterminate" : false}
                onCheckedChange={(checked) => onToggleAll(!!checked)}
                aria-label="Select all on page"
              />
            </th>
            <th className="text-left p-3 font-medium">Name</th>
            <th className="text-left p-3 font-medium">Ticker</th>
            <th className="text-left p-3 font-medium">Timeframe</th>
            <th className="text-left p-3 font-medium">Exchange</th>
            <th className="text-left p-3 font-medium">Action</th>
            <th className="text-left p-3 font-medium">Conditions</th>
            <th className="text-left p-3 font-medium">Last Triggered</th>
          </tr>
        </thead>
        <tbody>
          {alerts.length === 0 ? (
            <tr>
              <td colSpan={8} className="p-6 text-center text-muted-foreground">
                No alerts match the current filters.
              </td>
            </tr>
          ) : (
            alerts.map((alert) => (
              <tr
                key={alert.alertId}
                className="border-b hover:bg-muted/25 cursor-pointer"
                onClick={() => onToggle(alert.alertId)}
              >
                <td className="p-3">
                  <Checkbox
                    checked={selected.has(alert.alertId)}
                    onCheckedChange={() => onToggle(alert.alertId)}
                    onClick={(e) => e.stopPropagation()}
                    aria-label={`Select ${alert.name}`}
                  />
                </td>
                <td className="p-3 font-medium">{alert.name}</td>
                <td className="p-3 font-mono text-xs">
                  {alert.isRatio ? alert.ratio : alert.ticker}
                </td>
                <td className="p-3">{alert.timeframe || "\u2014"}</td>
                <td className="p-3">{alert.exchange || "\u2014"}</td>
                <td className="p-3">{alert.action || "\u2014"}</td>
                <td className="p-3 text-xs max-w-xs truncate">
                  {extractConditionText(alert.conditions) || "\u2014"}
                </td>
                <td className="p-3 text-xs">{formatAlertDate(alert.lastTriggered)}</td>
              </tr>
            ))
          )}
        </tbody>
      </table>
    </div>
  );
}

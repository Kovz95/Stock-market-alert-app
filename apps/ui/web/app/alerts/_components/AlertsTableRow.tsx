"use client";

import type { AlertData } from "@/actions/alert-actions";
import { formatAlertDate } from "./formatDate";

export function AlertsTableRow({
  alert,
  triggerCount,
}: {
  alert: AlertData;
  triggerCount?: number;
}) {
  return (
    <tr className="border-b hover:bg-muted/25">
      <td className="p-3 font-medium">{alert.name}</td>
      <td className="p-3 font-mono text-xs">
        {alert.isRatio ? alert.ratio : alert.ticker}
      </td>
      <td className="p-3">{alert.timeframe || "—"}</td>
      <td className="p-3">{alert.exchange || "—"}</td>
      <td className="p-3">{alert.action || "—"}</td>
      {triggerCount !== undefined && (
        <td className="p-3 text-right tabular-nums">
          {(triggerCount ?? 0).toLocaleString()}
        </td>
      )}
      <td className="p-3 text-xs">{formatAlertDate(alert.lastTriggered)}</td>
      <td className="p-3 text-xs">{formatAlertDate(alert.updatedAt)}</td>
    </tr>
  );
}

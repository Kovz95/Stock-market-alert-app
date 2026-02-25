"use client";

import type { AlertData } from "@/actions/alert-actions";
import { formatAlertDate } from "./formatDate";

export function AlertsTableRow({ alert }: { alert: AlertData }) {
  return (
    <tr className="border-b hover:bg-muted/25">
      <td className="p-3 font-medium">{alert.name}</td>
      <td className="p-3 font-mono text-xs">
        {alert.isRatio ? alert.ratio : alert.ticker}
      </td>
      <td className="p-3">{alert.timeframe || "—"}</td>
      <td className="p-3">{alert.exchange || "—"}</td>
      <td className="p-3">{alert.action || "—"}</td>
      <td className="p-3 text-xs">{formatAlertDate(alert.lastTriggered)}</td>
      <td className="p-3 text-xs">{formatAlertDate(alert.updatedAt)}</td>
    </tr>
  );
}

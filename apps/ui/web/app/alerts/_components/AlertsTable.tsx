"use client";

import type { AlertData } from "@/actions/alert-actions";
import { AlertsTableRow } from "./AlertsTableRow";

export function AlertsTable({ alerts }: { alerts: AlertData[] }) {
  return (
    <div className="border rounded-lg overflow-hidden">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b bg-muted/50">
            <th className="text-left p-3 font-medium">Name</th>
            <th className="text-left p-3 font-medium">Ticker</th>
            <th className="text-left p-3 font-medium">Timeframe</th>
            <th className="text-left p-3 font-medium">Exchange</th>
            <th className="text-left p-3 font-medium">Action</th>
            <th className="text-left p-3 font-medium">Last Triggered</th>
            <th className="text-left p-3 font-medium">Updated</th>
          </tr>
        </thead>
        <tbody>
          {alerts.map((alert) => (
            <AlertsTableRow key={alert.alertId} alert={alert} />
          ))}
        </tbody>
      </table>
    </div>
  );
}

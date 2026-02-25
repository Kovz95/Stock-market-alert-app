"use client";

import { useAlerts } from "@/lib/hooks/useAlerts";

function formatDate(dateStr: string | null): string {
  if (!dateStr) return "—";
  return new Date(dateStr).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function AlertsPage() {
  const { data: alerts, isLoading, error } = useAlerts();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-muted-foreground">Loading alerts...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-destructive">
          Failed to load alerts: {error.message}
        </div>
      </div>
    );
  }

  if (!alerts || alerts.length === 0) {
    return (
      <div className="p-8">
        <h1 className="text-2xl font-bold mb-4">Alerts</h1>
        <p className="text-muted-foreground">No alerts configured yet.</p>
      </div>
    );
  }

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-6">Alerts</h1>
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
              <tr key={alert.alertId} className="border-b hover:bg-muted/25">
                <td className="p-3 font-medium">{alert.name}</td>
                <td className="p-3 font-mono text-xs">
                  {alert.isRatio ? alert.ratio : alert.ticker}
                </td>
                <td className="p-3">{alert.timeframe || "—"}</td>
                <td className="p-3">{alert.exchange || "—"}</td>
                <td className="p-3">{alert.action || "—"}</td>
                <td className="p-3 text-xs">
                  {formatDate(alert.lastTriggered)}
                </td>
                <td className="p-3 text-xs">{formatDate(alert.updatedAt)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-4 text-xs text-muted-foreground">
        {alerts.length} alert{alerts.length !== 1 ? "s" : ""}
      </p>
    </div>
  );
}

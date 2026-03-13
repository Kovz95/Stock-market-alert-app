"use client";

import { BarChart3Icon } from "lucide-react";
import { AuditContainer } from "./_components/AuditContainer";

export default function AlertAuditPage() {
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
      <AuditContainer />
    </div>
  );
}

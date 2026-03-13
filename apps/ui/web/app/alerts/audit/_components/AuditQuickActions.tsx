"use client";

import { useAtom, useSetAtom, useAtomValue } from "jotai";
import { Button } from "@/components/ui/button";
import {
  RefreshCwIcon,
  Trash2Icon,
  DownloadIcon,
  TimerIcon,
} from "lucide-react";
import {
  auditAutoRefreshAtom,
  auditClearDialogOpenAtom,
} from "@/lib/store/audit";

type AuditQuickActionsProps = {
  onRefresh: () => void;
  onExportCsv: () => void;
  hasData: boolean;
};

export function AuditQuickActions({ onRefresh, onExportCsv, hasData }: AuditQuickActionsProps) {
  const [autoRefresh, setAutoRefresh] = useAtom(auditAutoRefreshAtom);
  const setClearOpen = useSetAtom(auditClearDialogOpenAtom);

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Button variant="outline" size="sm" onClick={onRefresh}>
        <RefreshCwIcon className="size-3.5 mr-1" />
        Refresh
      </Button>
      <Button
        variant={autoRefresh ? "default" : "outline"}
        size="sm"
        onClick={() => setAutoRefresh(!autoRefresh)}
      >
        <TimerIcon className="size-3.5 mr-1" />
        {autoRefresh ? "Auto-refresh on" : "Auto-refresh"}
      </Button>
      <Button
        variant="outline"
        size="sm"
        onClick={() => setClearOpen(true)}
      >
        <Trash2Icon className="size-3.5 mr-1" />
        Clear audit data
      </Button>
      <Button
        variant="outline"
        size="sm"
        onClick={onExportCsv}
        disabled={!hasData}
      >
        <DownloadIcon className="size-3.5 mr-1" />
        Export CSV
      </Button>
    </div>
  );
}

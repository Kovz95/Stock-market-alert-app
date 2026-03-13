"use client";

import { useAtom } from "jotai";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  auditDaysAtom,
  auditAlertIdFilterAtom,
  auditTickerFilterAtom,
  auditEvalTypeFilterAtom,
  auditStatusFilterAtom,
} from "@/lib/store/audit";

const DAYS_OPTIONS = [1, 7, 14, 30, 60, 90] as const;
const EVALUATION_TYPES = ["All", "scheduled", "manual", "test"] as const;
const STATUS_OPTIONS = ["All", "Success", "Error", "Triggered", "Not Triggered"] as const;

export function AuditFilters() {
  const [days, setDays] = useAtom(auditDaysAtom);
  const [alertId, setAlertId] = useAtom(auditAlertIdFilterAtom);
  const [ticker, setTicker] = useAtom(auditTickerFilterAtom);
  const [evalType, setEvalType] = useAtom(auditEvalTypeFilterAtom);
  const [status, setStatus] = useAtom(auditStatusFilterAtom);

  return (
    <div className="flex flex-wrap gap-4 items-end">
      <div className="flex flex-col gap-1">
        <label className="text-xs text-muted-foreground">Days</label>
        <Select value={String(days)} onValueChange={(v) => setDays(Number(v))}>
          <SelectTrigger className="w-[100px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {DAYS_OPTIONS.map((d) => (
              <SelectItem key={d} value={String(d)}>
                {d} days
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="flex flex-col gap-1">
        <label className="text-xs text-muted-foreground">Alert ID</label>
        <Input
          placeholder="Filter by alert ID"
          value={alertId}
          onChange={(e) => setAlertId(e.target.value)}
          className="w-[180px]"
        />
      </div>
      <div className="flex flex-col gap-1">
        <label className="text-xs text-muted-foreground">Ticker</label>
        <Input
          placeholder="Filter by ticker"
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          className="w-[120px]"
        />
      </div>
      <div className="flex flex-col gap-1">
        <label className="text-xs text-muted-foreground">Evaluation type</label>
        <Select value={evalType} onValueChange={setEvalType}>
          <SelectTrigger className="w-[120px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {EVALUATION_TYPES.map((t) => (
              <SelectItem key={t} value={t}>
                {t}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
      <div className="flex flex-col gap-1">
        <label className="text-xs text-muted-foreground">Status</label>
        <Select value={status} onValueChange={setStatus}>
          <SelectTrigger className="w-[140px]">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            {STATUS_OPTIONS.map((s) => (
              <SelectItem key={s} value={s}>
                {s}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}

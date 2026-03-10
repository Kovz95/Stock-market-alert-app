"use client";

import { useState } from "react";
import { ChevronDownIcon } from "lucide-react";
import type { AlertData } from "@/actions/alert-actions";
import { formatAlertDate } from "./formatDate";

/** Extract condition expression strings from backend shapes: condition_1.conditions, or { conditions: [...] }, or array of { conditions: string }. */
function getConditionStrings(conditions: unknown): string[] {
  if (conditions == null) return [];
  if (Array.isArray(conditions)) {
    const parts: string[] = [];
    for (const c of conditions) {
      if (c && typeof c === "object" && "conditions" in c) {
        const v = (c as { conditions: unknown }).conditions;
        if (Array.isArray(v)) parts.push(...v.map(String));
        else parts.push(String(v));
      }
    }
    return parts;
  }
  if (typeof conditions === "object") {
    const obj = conditions as Record<string, unknown>;
    // Top-level "conditions" key (array of strings or array of items)
    if (Array.isArray(obj.conditions)) {
      return (obj.conditions as unknown[]).map((x) =>
        typeof x === "object" && x != null && "conditions" in x
          ? String((x as { conditions: string }).conditions)
          : String(x)
      );
    }
    // condition_1, condition_2, ... with .conditions array
    const out: string[] = [];
    for (const key of Object.keys(obj)) {
      const block = obj[key];
      if (block && typeof block === "object" && "conditions" in block) {
        const cond = (block as { conditions: unknown }).conditions;
        if (Array.isArray(cond)) out.push(...cond.map(String));
        else out.push(String(cond));
      }
    }
    return out;
  }
  return [];
}

function formatConditions(conditions: unknown): string {
  const parts = getConditionStrings(conditions);
  if (parts.length === 0) {
    if (conditions != null && typeof conditions === "object") return JSON.stringify(conditions, null, 2);
    return "—";
  }
  return parts.join(" | ");
}

export function AlertsTableRow({
  alert,
  triggerCount,
  columnCount,
}: {
  alert: AlertData;
  triggerCount?: number;
  columnCount: number;
}) {
  const [expanded, setExpanded] = useState(false);
  const conditionStrings = getConditionStrings(alert.conditions);
  const hasConditions = conditionStrings.length > 0;

  return (
    <>
      <tr
        role="button"
        tabIndex={0}
        className="border-b hover:bg-muted/25 cursor-pointer select-none"
        onClick={() => setExpanded((e) => !e)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            setExpanded((e) => !e);
          }
        }}
        aria-expanded={expanded}
      >
        <td className="p-3 w-9 align-middle">
          <ChevronDownIcon
            className={`size-4 text-muted-foreground transition-transform ${expanded ? "rotate-180" : ""}`}
            aria-hidden
          />
        </td>
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
      {expanded && (
        <tr className="border-b bg-muted/20">
          <td colSpan={columnCount} className="p-0">
            <div className="px-3 py-3 pl-12">
              <div className="rounded-md border bg-background p-3 text-sm">
                <div className="font-medium text-muted-foreground mb-1.5">
                  Conditions
                  {alert.combinationLogic && (
                    <span className="ml-2 font-normal">
                      (combine with {alert.combinationLogic})
                    </span>
                  )}
                </div>
                {hasConditions ? (
                  <pre className="whitespace-pre-wrap break-words font-mono text-xs overflow-x-auto">
                    {formatConditions(alert.conditions)}
                  </pre>
                ) : alert.conditions === undefined ? (
                  <p className="text-muted-foreground text-xs">
                    Conditions not returned by server (list endpoint may omit them).
                  </p>
                ) : (
                  <p className="text-muted-foreground text-xs">No conditions</p>
                )}
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

"use client";

import * as React from "react";
import { useAtomValue, useSetAtom } from "jotai";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ConditionBuilder } from "@/app/alerts/add/_components/ConditionBuilder";
import { conditionEntryToExpression } from "@/app/alerts/add/_components/types";
import type { ConditionEntry } from "@/app/alerts/add/_components/types";
import { scannerConditionsAtom, scannerCombinationLogicAtom } from "@/lib/store/scanner";

export function ScannerConditionSection() {
  const conditions = useAtomValue(scannerConditionsAtom);
  const setConditions = useSetAtom(scannerConditionsAtom);
  const combinationLogic = useAtomValue(scannerCombinationLogicAtom);
  const setCombinationLogic = useSetAtom(scannerCombinationLogicAtom);

  const handleAddEntry = React.useCallback((entry: ConditionEntry) => {
    const expr = conditionEntryToExpression(entry);
    if (expr) {
      setConditions((prev) => [...prev, expr]);
    }
  }, [setConditions]);

  const removeAt = (i: number) => {
    setConditions((prev) => prev.filter((_, j) => j !== i));
  };

  return (
    <div className="space-y-4 rounded-lg border bg-card p-4">
      <h3 className="font-semibold">Build conditions</h3>
      <ConditionBuilder onAdd={handleAddEntry} />
      {conditions.length > 0 && (
        <>
          <div className="space-y-2">
            <Label className="text-xs">Current conditions</Label>
            <ul className="space-y-1">
              {conditions.map((c, i) => (
                <li key={i} className="flex items-center gap-2 text-sm">
                  <code className="flex-1 rounded bg-muted px-2 py-1 text-xs">
                    {i + 1}. {c}
                  </code>
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="h-7 w-7 p-0"
                    onClick={() => removeAt(i)}
                  >
                    ×
                  </Button>
                </li>
              ))}
            </ul>
          </div>
          {conditions.length > 1 && (
            <div className="space-y-2">
              <Label className="text-xs">Combine with</Label>
              <Select
                value={combinationLogic === "AND" ? "AND" : combinationLogic === "OR" ? "OR" : "custom"}
                onValueChange={(v) => {
                  if (v === "AND") setCombinationLogic("AND");
                  else if (v === "OR") setCombinationLogic("OR");
                  else setCombinationLogic(combinationLogic || "1 AND 2");
                }}
              >
                <SelectTrigger className="h-8 w-48">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="AND">All (AND)</SelectItem>
                  <SelectItem value="OR">Any (OR)</SelectItem>
                  <SelectItem value="custom">Custom expression</SelectItem>
                </SelectContent>
              </Select>
              {combinationLogic !== "AND" && combinationLogic !== "OR" && (
                <Input
                  className="h-8 w-full max-w-md font-mono text-xs"
                  placeholder="e.g. 1 AND (2 OR 3)"
                  value={combinationLogic}
                  onChange={(e) => setCombinationLogic(e.target.value)}
                />
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}

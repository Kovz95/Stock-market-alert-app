"use client";

import { useState } from "react";
import { Label } from "@/components/ui/label";
import { ConditionBuilder } from "@/app/alerts/add/_components/ConditionBuilder";
import { conditionEntryToExpression } from "@/app/alerts/add/_components/types";
import type { ConditionEntry } from "@/app/alerts/add/_components/types";

interface Props {
  value: string;
  onChange: (condition: string) => void;
}

export function CustomChannelConditionInput({ value, onChange }: Props) {
  const [mode, setMode] = useState<"specific" | "price_level">(
    value === "price_level" ? "price_level" : "specific"
  );

  function handleModeChange(next: "specific" | "price_level") {
    setMode(next);
    if (next === "price_level") {
      onChange("price_level");
    } else {
      onChange("");
    }
  }

  function handleAdd(entry: ConditionEntry) {
    const expr = conditionEntryToExpression(entry);
    onChange(expr);
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-6">
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="radio"
            name="condition-mode"
            value="specific"
            checked={mode === "specific"}
            onChange={() => handleModeChange("specific")}
            className="accent-primary"
          />
          <Label className="cursor-pointer">Specific condition</Label>
        </label>
        <label className="flex items-center gap-2 cursor-pointer">
          <input
            type="radio"
            name="condition-mode"
            value="price_level"
            checked={mode === "price_level"}
            onChange={() => handleModeChange("price_level")}
            className="accent-primary"
          />
          <Label className="cursor-pointer">Any price level</Label>
        </label>
      </div>

      {mode === "specific" && (
        <div className="rounded-lg border border-border p-4">
          <ConditionBuilder onAdd={handleAdd} />
          {value && value !== "price_level" && (
            <p className="mt-3 text-sm text-muted-foreground">
              Selected:{" "}
              <code className="rounded bg-muted px-1.5 py-0.5 text-xs font-mono">
                {value}
              </code>
            </p>
          )}
          {!value && (
            <p className="mt-3 text-xs text-muted-foreground">
              Build a condition above, then click &quot;Add condition&quot; to set it.
            </p>
          )}
        </div>
      )}

      {mode === "price_level" && (
        <p className="text-sm text-muted-foreground">
          This channel will receive alerts for any price-level condition (e.g.{" "}
          <code className="rounded bg-muted px-1 py-0.5 text-xs font-mono">
            Close[-1] &lt; 150
          </code>
          ).
        </p>
      )}
    </div>
  );
}

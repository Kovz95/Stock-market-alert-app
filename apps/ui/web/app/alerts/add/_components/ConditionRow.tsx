"use client";

import { Button } from "@/components/ui/button";
import { conditionEntryLabel, type ConditionEntry } from "./types";
import { Trash2Icon } from "lucide-react";

export interface ConditionRowProps {
  entry: ConditionEntry;
  onRemove: () => void;
}

export function ConditionRow({ entry, onRemove }: ConditionRowProps) {
  const label = conditionEntryLabel(entry);
  return (
    <div className="flex items-center justify-between gap-2 rounded-md border bg-muted/30 px-3 py-2 text-sm">
      <code className="flex-1 truncate text-xs" title={label}>
        {label || "Empty condition"}
      </code>
      <Button
        type="button"
        variant="ghost"
        size="icon-xs"
        aria-label="Remove condition"
        onClick={onRemove}
      >
        <Trash2Icon className="size-3.5" />
      </Button>
    </div>
  );
}

"use client";

import { Button } from "@/components/ui/button";
import { Trash2Icon } from "lucide-react";

type DeleteAlertsActionBarProps = {
  selectedCount: number;
  totalFiltered: number;
  pageCount: number;
  onSelectAllFiltered: () => void;
  onClearSelection: () => void;
  onDelete: () => void;
};

export function DeleteAlertsActionBar({
  selectedCount,
  totalFiltered,
  pageCount,
  onSelectAllFiltered,
  onClearSelection,
  onDelete,
}: DeleteAlertsActionBarProps) {
  if (selectedCount === 0) return null;

  return (
    <div className="flex items-center gap-3 rounded-lg border bg-muted/50 px-4 py-3">
      <span className="text-sm font-medium">
        {selectedCount} of {totalFiltered} alert{totalFiltered !== 1 ? "s" : ""} selected
      </span>

      <div className="flex items-center gap-2 ml-auto">
        {selectedCount < totalFiltered && (
          <Button variant="outline" size="sm" onClick={onSelectAllFiltered}>
            Select all {totalFiltered} filtered
          </Button>
        )}
        <Button variant="outline" size="sm" onClick={onClearSelection}>
          Clear selection
        </Button>
        <Button variant="destructive" size="sm" onClick={onDelete}>
          <Trash2Icon className="size-4 mr-1" />
          Delete {selectedCount} selected
        </Button>
      </div>
    </div>
  );
}

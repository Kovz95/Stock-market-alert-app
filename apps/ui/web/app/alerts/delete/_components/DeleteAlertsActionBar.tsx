"use client";

import { useAtomValue, useSetAtom } from "jotai";
import { Button } from "@/components/ui/button";
import { Trash2Icon } from "lucide-react";
import {
  deleteAlertsSelectedAtom,
  deleteAlertsIsSelectingAllAtom,
  deleteAlertsConfirmOpenAtom,
} from "@/lib/store/delete-alerts";

type DeleteAlertsActionBarProps = {
  totalFiltered: number;
  pageCount: number;
  onSelectAllFiltered: () => void | Promise<void>;
};

export function DeleteAlertsActionBar({
  totalFiltered,
  pageCount,
  onSelectAllFiltered,
}: DeleteAlertsActionBarProps) {
  const selected = useAtomValue(deleteAlertsSelectedAtom);
  const selectedCount = selected.size;
  const isSelectingAll = useAtomValue(deleteAlertsIsSelectingAllAtom);
  const setSelected = useSetAtom(deleteAlertsSelectedAtom);
  const setConfirmOpen = useSetAtom(deleteAlertsConfirmOpenAtom);

  if (totalFiltered === 0) return null;

  return (
    <div className="flex items-center gap-3 rounded-lg border bg-muted/50 px-4 py-3">
      <span className="text-sm font-medium">
        {selectedCount} of {totalFiltered} alert{totalFiltered !== 1 ? "s" : ""} selected
      </span>

      <div className="flex items-center gap-2 ml-auto">
        {selectedCount < totalFiltered && (
          <Button
            variant="outline"
            size="sm"
            disabled={isSelectingAll}
            onClick={() => void onSelectAllFiltered()}
          >
            {isSelectingAll ? "Selecting all..." : `Select all ${totalFiltered} filtered`}
          </Button>
        )}
        <Button
          variant="outline"
          size="sm"
          onClick={() => setSelected(new Set())}
          disabled={selectedCount === 0}
        >
          Clear selection
        </Button>
        <Button
          variant="destructive"
          size="sm"
          onClick={() => setConfirmOpen(true)}
          disabled={selectedCount === 0}
        >
          <Trash2Icon className="size-4 mr-1" />
          Delete {selectedCount} selected
        </Button>
      </div>
    </div>
  );
}

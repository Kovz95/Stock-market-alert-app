"use client";

import { useAtomValue, useSetAtom } from "jotai";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  deleteAlertsConfirmOpenAtom,
  deleteAlertsIsDeletingAtom,
  deleteAlertsSelectedAtom,
  deleteAlertsProgressAtom,
} from "@/lib/store/delete-alerts";

type DeleteConfirmDialogProps = {
  onConfirm: () => void;
};

export function DeleteConfirmDialog({ onConfirm }: DeleteConfirmDialogProps) {
  const open = useAtomValue(deleteAlertsConfirmOpenAtom);
  const setOpen = useSetAtom(deleteAlertsConfirmOpenAtom);
  const isDeleting = useAtomValue(deleteAlertsIsDeletingAtom);
  const selected = useAtomValue(deleteAlertsSelectedAtom);
  const progress = useAtomValue(deleteAlertsProgressAtom);
  const count = selected.size;

  const progressPct = progress
    ? Math.round((progress.completed / progress.total) * 100)
    : 0;

  return (
    <AlertDialog open={open} onOpenChange={setOpen}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Delete {count} alert{count !== 1 ? "s" : ""}?</AlertDialogTitle>
          <AlertDialogDescription>
            This action cannot be undone. {count} alert{count !== 1 ? "s" : ""} will be
            permanently deleted.
          </AlertDialogDescription>
        </AlertDialogHeader>

        {isDeleting && progress && (
          <div className="space-y-2">
            <div className="flex justify-between text-sm text-muted-foreground">
              <span>Deleting alerts...</span>
              <span>{progress.completed} / {progress.total}</span>
            </div>
            <div className="h-2 w-full rounded-full bg-muted overflow-hidden">
              <div
                className="h-full bg-primary transition-all duration-200"
                style={{ width: `${progressPct}%` }}
              />
            </div>
          </div>
        )}

        <AlertDialogFooter>
          <AlertDialogCancel disabled={isDeleting}>Cancel</AlertDialogCancel>
          <AlertDialogAction
            variant="destructive"
            disabled={isDeleting}
            onClick={(e) => {
              e.preventDefault();
              onConfirm();
            }}
          >
            {isDeleting
              ? `Deleting... ${progressPct}%`
              : `Delete ${count} alert${count !== 1 ? "s" : ""}`}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

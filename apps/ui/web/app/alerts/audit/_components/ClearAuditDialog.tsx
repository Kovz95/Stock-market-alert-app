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
import { auditClearDialogOpenAtom } from "@/lib/store/audit";

type ClearAuditDialogProps = {
  onConfirm: () => void;
  isPending: boolean;
};

export function ClearAuditDialog({ onConfirm, isPending }: ClearAuditDialogProps) {
  const open = useAtomValue(auditClearDialogOpenAtom);
  const setOpen = useSetAtom(auditClearDialogOpenAtom);

  return (
    <AlertDialog open={open} onOpenChange={setOpen}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Clear all audit data?</AlertDialogTitle>
          <AlertDialogDescription>
            This action cannot be undone. All audit logs and history will be permanently deleted.
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel disabled={isPending}>Cancel</AlertDialogCancel>
          <AlertDialogAction
            variant="destructive"
            disabled={isPending}
            onClick={(e) => {
              e.preventDefault();
              onConfirm();
            }}
          >
            {isPending ? "Clearing..." : "Clear all data"}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

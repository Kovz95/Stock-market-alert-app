"use client";

import * as React from "react";
import { Loader2Icon } from "lucide-react";
import { toast } from "sonner";
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { usePreviewDeleteTicker, useDeleteTicker } from "@/lib/hooks/useStockDatabase";
import type { TickerDeletionCounts } from "@/actions/stock-database-actions";

type Props = {
  ticker: string | null;
  onClose: () => void;
};

const fmt = new Intl.NumberFormat("en-US");

type CountRow = { label: string; value: number };

function buildCountRows(counts: TickerDeletionCounts): CountRow[] {
  const rows: CountRow[] = [
    { label: "stock metadata rows", value: counts.stockMetadata },
    { label: "ticker metadata rows", value: counts.tickerMetadata },
    { label: "daily price bars", value: counts.dailyPrices },
    { label: "hourly price bars", value: counts.hourlyPrices },
    { label: "weekly price bars", value: counts.weeklyPrices },
    { label: "continuous price bars", value: counts.continuousPrices },
    { label: "daily move stats rows", value: counts.dailyMoveStats },
    { label: "futures metadata rows", value: counts.futuresMetadata },
    { label: "direct alerts (ticker = this symbol)", value: counts.alertsDirect },
    { label: "ratio alerts (this ticker is ticker1 or ticker2)", value: counts.alertsRatio },
    { label: "alert audit rows", value: counts.alertAudits },
    { label: "portfolio holdings", value: counts.portfolioStocks },
  ];
  return rows.filter((r) => r.value > 0);
}

function allZero(counts: TickerDeletionCounts): boolean {
  return Object.values(counts).every((v) => v === 0);
}

export function DeleteTickerDialog({ ticker, onClose }: Props) {
  const open = ticker !== null;
  const { data: preview, isLoading: previewLoading, error: previewError } = usePreviewDeleteTicker(ticker);
  const deleteMutation = useDeleteTicker();

  const isEmpty = preview && !preview.exists && allZero(preview.counts);
  const canDelete = !isEmpty && !previewLoading && !previewError;

  function handleDelete() {
    if (!ticker) return;
    deleteMutation.mutate(ticker, {
      onSuccess: (result) => {
        if (!result.success) {
          toast.error(result.errorMessage || "Failed to delete ticker");
          return;
        }
        const c = result.counts;
        const totalAlerts = c.alertsDirect + c.alertsRatio;
        const totalBars = c.dailyPrices + c.hourlyPrices + c.weeklyPrices + c.continuousPrices;
        toast.success(
          `Removed ${ticker}. Deleted ${fmt.format(totalAlerts)} alert${totalAlerts !== 1 ? "s" : ""}, ${fmt.format(totalBars)} price bar${totalBars !== 1 ? "s" : ""}, ${fmt.format(c.alertAudits)} audit row${c.alertAudits !== 1 ? "s" : ""}.`
        );
        onClose();
      },
      onError: (err) => {
        toast.error(err instanceof Error ? err.message : "Failed to delete ticker");
      },
    });
  }

  const isPending = deleteMutation.isPending;

  return (
    <AlertDialog open={open} onOpenChange={(isOpen) => { if (!isOpen) onClose(); }}>
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>Remove {ticker}?</AlertDialogTitle>
          <AlertDialogDescription asChild>
            <div className="space-y-3">
              <p>
                This permanently removes <strong>{ticker}</strong> from every table in the
                database. This action cannot be undone.
              </p>
              {previewLoading && (
                <div className="space-y-2">
                  <Skeleton className="h-4 w-3/4" />
                  <Skeleton className="h-4 w-1/2" />
                  <Skeleton className="h-4 w-2/3" />
                </div>
              )}
              {previewError && (
                <p className="text-sm text-destructive">
                  Failed to load preview: {previewError instanceof Error ? previewError.message : String(previewError)}
                </p>
              )}
              {preview && isEmpty && (
                <p className="text-sm text-muted-foreground">
                  Nothing to delete — this ticker isn&apos;t in the database.
                </p>
              )}
              {preview && !isEmpty && (
                <div>
                  <p className="text-sm font-medium">The following will be deleted:</p>
                  <ul className="mt-1 list-disc pl-5 space-y-0.5 text-sm text-muted-foreground">
                    {buildCountRows(preview.counts).map((row) => (
                      <li key={row.label}>
                        <span className="font-medium text-foreground">{fmt.format(row.value)}</span>{" "}
                        {row.label}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel disabled={isPending}>Cancel</AlertDialogCancel>
          <Button
            variant="destructive"
            disabled={!canDelete || isPending}
            onClick={handleDelete}
          >
            {isPending && <Loader2Icon className="mr-2 size-4 animate-spin" />}
            Delete permanently
          </Button>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}

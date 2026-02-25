"use client";

import { useAtomValue, useSetAtom } from "jotai";
import { Button } from "@/components/ui/button";
import { alertsPageAtom, alertsPageSizeAtom } from "@/lib/store/alerts";

type AlertsPaginationProps = {
  totalCount: number;
  hasNextPage: boolean;
};

export function AlertsPagination({
  totalCount,
  hasNextPage,
}: AlertsPaginationProps) {
  const page = useAtomValue(alertsPageAtom);
  const pageSize = useAtomValue(alertsPageSizeAtom);
  const setPage = useSetAtom(alertsPageAtom);
  const setPageSize = useSetAtom(alertsPageSizeAtom);

  const start = totalCount === 0 ? 0 : (page - 1) * pageSize + 1;
  const end = Math.min(page * pageSize, totalCount);
  const totalPages = Math.max(1, Math.ceil(totalCount / pageSize));

  return (
    <div className="mt-4 flex flex-wrap items-center justify-between gap-4">
      <p className="text-xs text-muted-foreground">
        {totalCount === 0
          ? "No alerts"
          : `Showing ${start}–${end} of ${totalCount} alert${totalCount !== 1 ? "s" : ""}`}
      </p>
      <div className="flex items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          disabled={page <= 1}
          onClick={() => setPage((p) => Math.max(1, p - 1))}
        >
          Previous
        </Button>
        <span className="text-xs text-muted-foreground px-2">
          Page {page} of {totalPages}
        </span>
        <Button
          variant="outline"
          size="sm"
          disabled={!hasNextPage}
          onClick={() => setPage((p) => p + 1)}
        >
          Next
        </Button>
        <label className="flex items-center gap-1.5 text-xs text-muted-foreground">
          Per page:
          <select
            className="rounded border bg-background px-1.5 py-0.5 text-xs"
            value={pageSize}
            onChange={(e) => {
              setPageSize(Number(e.target.value));
              setPage(1);
            }}
          >
            {[10, 20, 50, 100].map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
      </div>
    </div>
  );
}

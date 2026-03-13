"use client";

import { useAtom } from "jotai";
import { Button } from "@/components/ui/button";
import { ChevronLeftIcon, ChevronRightIcon } from "lucide-react";
import { auditLogPageAtom } from "@/lib/store/audit";

type EvaluationLogPaginationProps = {
  totalCount: number;
  pageSize: number;
};

export function EvaluationLogPagination({ totalCount, pageSize }: EvaluationLogPaginationProps) {
  const [page, setPage] = useAtom(auditLogPageAtom);
  const totalPages = Math.max(1, Math.ceil(totalCount / pageSize));

  if (totalPages <= 1) return null;

  return (
    <div className="flex items-center justify-between">
      <p className="text-sm text-muted-foreground">
        Page {page} of {totalPages} ({totalCount.toLocaleString()} rows)
      </p>
      <div className="flex items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          disabled={page <= 1}
          onClick={() => setPage((p) => p - 1)}
        >
          <ChevronLeftIcon className="size-4" />
        </Button>
        <Button
          variant="outline"
          size="sm"
          disabled={page >= totalPages}
          onClick={() => setPage((p) => p + 1)}
        >
          <ChevronRightIcon className="size-4" />
        </Button>
      </div>
    </div>
  );
}

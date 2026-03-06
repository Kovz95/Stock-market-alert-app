"use client";

import { listAlertsPaginated } from "@/actions/alert-actions";
import { atom } from "jotai";
import { atomWithQuery } from "jotai-tanstack-query";

export const ALERTS_KEY = ["alerts"] as const;
export const DEFAULT_ALERTS_PAGE_SIZE = 20;

/** 1-based current page for the alerts list */
export const alertsPageAtom = atom(1);

/** Page size for the alerts list */
export const alertsPageSizeAtom = atom(DEFAULT_ALERTS_PAGE_SIZE);

/** Paginated alerts query as a Jotai atom; key depends on alertsPageAtom and alertsPageSizeAtom. */
export const alertsPaginatedQueryAtom = atomWithQuery((get) => {
  const page = get(alertsPageAtom);
  const pageSize = get(alertsPageSizeAtom);
  return {
    queryKey: [...ALERTS_KEY, "paginated", page, pageSize],
    queryFn: () => listAlertsPaginated(page, pageSize),
  };
});

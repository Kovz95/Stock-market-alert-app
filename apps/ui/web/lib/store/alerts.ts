"use client";

import { searchAlerts } from "@/actions/alert-actions";
import { atom } from "jotai";
import { atomWithQuery } from "jotai-tanstack-query";

export const ALERTS_KEY = ["alerts"] as const;
export const DEFAULT_ALERTS_PAGE_SIZE = 20;

/** 1-based current page for the alerts list */
export const alertsPageAtom = atom(1);

/** Page size for the alerts list */
export const alertsPageSizeAtom = atom(DEFAULT_ALERTS_PAGE_SIZE);

/** Server-side search query (name, ticker, etc.). Empty string = no filter. */
export const alertsSearchAtom = atom("");

/** Paginated alerts query; server-side search + pagination. */
export const alertsPaginatedQueryAtom = atomWithQuery((get) => {
  const page = get(alertsPageAtom);
  const pageSize = get(alertsPageSizeAtom);
  const search = get(alertsSearchAtom);
  return {
    queryKey: [...ALERTS_KEY, "paginated", search, page, pageSize],
    queryFn: () =>
      searchAlerts(
        { search: search.trim() },
        page,
        pageSize
      ),
  };
});

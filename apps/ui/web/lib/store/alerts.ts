"use client";

import { atom } from "jotai";

export const DEFAULT_ALERTS_PAGE_SIZE = 20;

/** 1-based current page for the alerts list */
export const alertsPageAtom = atom(1);

/** Page size for the alerts list */
export const alertsPageSizeAtom = atom(DEFAULT_ALERTS_PAGE_SIZE);

"use client";

import { atom } from "jotai";

export type AuditTab = "overview" | "evaluation-log" | "failed-data";
export type SortDirection = "asc" | "desc";

export const auditActiveTabAtom = atom<AuditTab>("overview");
export const auditDaysAtom = atom<number>(7);
export const auditAlertIdFilterAtom = atom<string>("");
export const auditTickerFilterAtom = atom<string>("");
export const auditEvalTypeFilterAtom = atom<string>("All");
export const auditStatusFilterAtom = atom<string>("All");
export const auditLogPageAtom = atom<number>(1);
export const auditLogPageSizeAtom = atom<number>(50);
export const auditLogSortFieldAtom = atom<string>("timestamp");
export const auditLogSortDirectionAtom = atom<SortDirection>("desc");
export const auditSelectedAlertIdAtom = atom<string | null>(null);
export const auditDetailSheetOpenAtom = atom<boolean>(false);
export const auditClearDialogOpenAtom = atom<boolean>(false);
export const auditAutoRefreshAtom = atom<boolean>(false);

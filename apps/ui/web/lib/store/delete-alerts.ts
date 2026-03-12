"use client";

import { atom } from "jotai";
import type { DeleteAlertsFiltersState } from "@/app/alerts/delete/_components/types";
import { defaultDeleteAlertsFilters } from "@/app/alerts/delete/_components/types";

export type DeleteProgress = {
  completed: number;
  total: number;
};

export const deleteAlertsFiltersAtom = atom<DeleteAlertsFiltersState>(defaultDeleteAlertsFilters);
export const deleteAlertsSelectedAtom = atom<Set<string>>(new Set<string>());
export const deleteAlertsPageAtom = atom<number>(1);
export const deleteAlertsConfirmOpenAtom = atom<boolean>(false);
export const deleteAlertsIsDeletingAtom = atom<boolean>(false);
export const deleteAlertsIsSelectingAllAtom = atom<boolean>(false);
export const deleteAlertsProgressAtom = atom<DeleteProgress | null>(null);

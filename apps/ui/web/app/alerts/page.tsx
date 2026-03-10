"use client";

import { useAtomValue } from "jotai";
import { useAlertsPaginated } from "@/lib/hooks/useAlerts";
import { alertsSearchAtom } from "@/lib/store/alerts";
import { AlertsEmpty } from "./_components/AlertsEmpty";
import { AlertsError } from "./_components/AlertsError";
import { AlertsLoading } from "./_components/AlertsLoading";
import { AlertsPagination } from "./_components/AlertsPagination";
import { AlertsSearchBar } from "./_components/AlertsSearchBar";
import { AlertsTable } from "./_components/AlertsTable";

export default function AlertsPage() {
  const searchQuery = useAtomValue(alertsSearchAtom);
  const { data, isLoading, error } = useAlertsPaginated();

  return (
    <div className="p-8">
      <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
        <h1 className="text-2xl font-bold">Alerts</h1>
        <AlertsSearchBar />
      </div>
      {isLoading && <AlertsLoading />}
      {!isLoading && error && <AlertsError message={error.message} />}
      {!isLoading && !error && (!data || data.totalCount === 0) && (
        <AlertsEmpty searchQuery={searchQuery.trim() || undefined} />
      )}
      {!isLoading && !error && data && data.totalCount > 0 && (
        <>
          <AlertsTable alerts={data.alerts} />
          <AlertsPagination
            totalCount={data.totalCount}
            hasNextPage={data.hasNextPage}
          />
        </>
      )}
    </div>
  );
}

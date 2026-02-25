"use client";

import { useAlertsPaginated } from "@/lib/hooks/useAlerts";
import { AlertsEmpty } from "./_components/AlertsEmpty";
import { AlertsError } from "./_components/AlertsError";
import { AlertsLoading } from "./_components/AlertsLoading";
import { AlertsPagination } from "./_components/AlertsPagination";
import { AlertsTable } from "./_components/AlertsTable";

export default function AlertsPage() {
  const { data, isLoading, error } = useAlertsPaginated();

  if (isLoading) {
    return <AlertsLoading />;
  }

  if (error) {
    return <AlertsError message={error.message} />;
  }

  if (!data || data.totalCount === 0) {
    return <AlertsEmpty />;
  }

  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-6">Alerts</h1>
      <AlertsTable alerts={data.alerts} />
      <AlertsPagination
        totalCount={data.totalCount}
        hasNextPage={data.hasNextPage}
      />
    </div>
  );
}

"use client";

import Link from "next/link";
import { useDashboardAlerts } from "@/lib/hooks/useDashboard";
import { Button } from "@/components/ui/button";
import { AlertsTable } from "@/app/alerts/_components/AlertsTable";
import { AlertsLoading } from "@/app/alerts/_components/AlertsLoading";
import { ChevronRightIcon } from "lucide-react";

export function DashboardActiveAlerts() {
  const { data, isLoading, error } = useDashboardAlerts();

  if (isLoading) {
    return (
      <div className="px-4 lg:px-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">Active alerts</h2>
        </div>
        <AlertsLoading />
      </div>
    );
  }

  if (error) {
    return (
      <div className="px-4 lg:px-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">Active alerts</h2>
        </div>
        <p className="text-sm text-destructive">
          {error instanceof Error ? error.message : "Failed to load alerts."}
        </p>
      </div>
    );
  }

  if (!data) {
    return null;
  }

  if (data.error) {
    return (
      <div className="px-4 lg:px-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">Active alerts</h2>
        </div>
        <p className="text-sm text-amber-600 dark:text-amber-500">
          {data.error}
        </p>
      </div>
    );
  }

  if (data.alerts.length === 0) {
    return (
      <div className="px-4 lg:px-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">Active alerts</h2>
        </div>
        <p className="text-muted-foreground text-sm py-6 border rounded-lg text-center">
          No alerts configured yet.{" "}
          <Link href="/alerts/add" className="text-primary underline">
            Add an alert
          </Link>
        </p>
      </div>
    );
  }

  return (
    <div className="px-4 lg:px-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Active alerts</h2>
        <Button variant="ghost" size="sm" asChild>
          <Link href="/alerts">
            View all
            <ChevronRightIcon className="size-4 ml-1" />
          </Link>
        </Button>
      </div>
      <AlertsTable alerts={data.alerts} />
      {data.totalCount > data.alerts.length && (
        <p className="text-muted-foreground text-xs mt-2">
          Showing {data.alerts.length} of {data.totalCount} alerts.{" "}
          <Link href="/alerts" className="text-primary underline">
            View all
          </Link>
        </p>
      )}
    </div>
  );
}

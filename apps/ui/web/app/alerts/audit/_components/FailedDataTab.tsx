"use client";

import { useAtomValue } from "jotai";
import { useFailedPriceData } from "@/lib/hooks/useAudit";
import { auditDaysAtom } from "@/lib/store/audit";
import { FailedDataKpiCards } from "./FailedDataKpiCards";
import { FailedDataAssetChart } from "./FailedDataAssetChart";
import { FailedDataExchangeTable } from "./FailedDataExchangeTable";
import { FailedDataAlertsTable } from "./FailedDataAlertsTable";

export function FailedDataTab() {
  const days = useAtomValue(auditDaysAtom);
  const { data: failedData, isLoading } = useFailedPriceData(days);

  if (isLoading) {
    return <p className="text-sm text-muted-foreground">Loading failed data...</p>;
  }

  if (!failedData) {
    return <p className="text-sm text-muted-foreground">Unable to load failed price data.</p>;
  }

  if (failedData.rows.length === 0) {
    return <p className="text-sm text-green-600">No failed price data retrievals in this period.</p>;
  }

  return (
    <div className="space-y-6">
      <FailedDataKpiCards data={failedData} />
      <FailedDataAssetChart breakdown={failedData.assetTypeBreakdown ?? []} />
      <FailedDataExchangeTable rows={failedData.exchangeBreakdown ?? []} />
      <FailedDataAlertsTable rows={failedData.rows ?? []} />
    </div>
  );
}

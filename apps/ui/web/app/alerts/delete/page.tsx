"use client";

import { DeleteAlertsContainer } from "./_components/DeleteAlertsContainer";

export default function DeleteAlertsPage() {
  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-2">Delete Alerts</h1>
      <p className="text-muted-foreground mb-6">
        Search and filter your alerts, then select and delete them in bulk.
      </p>
      <DeleteAlertsContainer />
    </div>
  );
}

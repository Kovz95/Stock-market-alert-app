"use client";

import { AddAlertForm } from "./_components/AddAlertForm";

export default function AddAlertPage() {
  return (
    <div className="p-8">
      <h1 className="text-2xl font-bold mb-2">Add Alert</h1>
      <p className="text-muted-foreground mb-6">
        Create a new stock alert. Set name (optional), ticker or ratio pair,
        conditions, and optionally create the same alert for multiple tickers.
      </p>
      <AddAlertForm />
    </div>
  );
}

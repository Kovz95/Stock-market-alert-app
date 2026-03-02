"use client";

import * as React from "react";
import { usePortfoliosList } from "@/lib/hooks/usePortfolios";
import { Skeleton } from "@/components/ui/skeleton";
import { Button } from "@/components/ui/button";
import { PlusIcon } from "lucide-react";
import {
  PortfolioList,
  CreatePortfolioDialog,
  PortfolioSettings,
  PortfolioHoldings,
  AddStocksSection,
  PortfolioSummaryTable,
} from "./_components";

export default function PortfoliosPage() {
  const { data: portfolios, isLoading, error } = usePortfoliosList();
  const [selectedId, setSelectedId] = React.useState<string | null>(null);
  const [createOpen, setCreateOpen] = React.useState(false);

  const selectedPortfolio = React.useMemo(() => {
    if (!portfolios || !selectedId) return null;
    return portfolios.find((p) => p.portfolioId === selectedId) ?? null;
  }, [portfolios, selectedId]);

  // Auto-select first portfolio when data loads
  React.useEffect(() => {
    if (portfolios && portfolios.length > 0 && !selectedId) {
      setSelectedId(portfolios[0].portfolioId);
    }
  }, [portfolios, selectedId]);

  if (isLoading) {
    return (
      <div className="p-6 space-y-6">
        <Skeleton className="h-8 w-48" />
        <div className="grid gap-6 lg:grid-cols-[280px_1fr]">
          <Skeleton className="h-64 w-full" />
          <Skeleton className="h-64 w-full" />
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Portfolios</h1>
          <p className="text-muted-foreground">
            Manage portfolios with individual Discord alert channels.
          </p>
        </div>
        <Button onClick={() => setCreateOpen(true)}>
          <PlusIcon className="mr-1 size-4" />
          New Portfolio
        </Button>
      </div>

      {error && (
        <div className="rounded-lg border border-destructive/50 bg-destructive/10 px-4 py-3 text-destructive text-sm">
          Failed to load portfolios: {error.message}
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-[280px_1fr]">
        {/* Sidebar: Portfolio list */}
        <aside className="space-y-4">
          <PortfolioList
            portfolios={portfolios ?? []}
            selectedId={selectedId}
            onSelect={setSelectedId}
          />
        </aside>

        {/* Main content */}
        <main className="space-y-6">
          {selectedPortfolio ? (
            <>
              <div className="flex items-center gap-3">
                <h2 className="text-xl font-semibold">{selectedPortfolio.name}</h2>
                {selectedPortfolio.discordWebhook ? (
                  <span className="text-xs text-green-600 bg-green-50 dark:bg-green-950 px-2 py-0.5 rounded">
                    Webhook configured
                  </span>
                ) : (
                  <span className="text-xs text-yellow-600 bg-yellow-50 dark:bg-yellow-950 px-2 py-0.5 rounded">
                    No webhook
                  </span>
                )}
              </div>

              <div className="grid gap-6 xl:grid-cols-[1fr_300px]">
                <div className="space-y-6">
                  <PortfolioHoldings portfolio={selectedPortfolio} />
                </div>
                <div className="space-y-6">
                  <AddStocksSection portfolio={selectedPortfolio} />
                  <PortfolioSettings
                    portfolio={selectedPortfolio}
                    onDeleted={() => setSelectedId(null)}
                  />
                </div>
              </div>
            </>
          ) : (
            <div className="rounded-lg border bg-muted/20 p-8 text-center">
              <p className="text-muted-foreground">
                {portfolios && portfolios.length > 0
                  ? "Select a portfolio from the list."
                  : "Create your first portfolio to get started."}
              </p>
            </div>
          )}
        </main>
      </div>

      {/* Summary table */}
      {portfolios && portfolios.length > 0 && (
        <PortfolioSummaryTable portfolios={portfolios} />
      )}

      {/* How it works info */}
      <div className="rounded-lg border bg-muted/10 p-4 text-sm text-muted-foreground">
        <p className="font-medium text-foreground mb-1">How Portfolio Alerts Work</p>
        <ul className="list-disc pl-5 space-y-1 text-xs">
          <li>Each portfolio can have its own Discord webhook URL (different channel)</li>
          <li>When an alert triggers for a stock in a portfolio, it&apos;s sent to that portfolio&apos;s Discord channel</li>
          <li>You can have the same stock in multiple portfolios with different webhooks</li>
          <li>Portfolio alerts have a [PORTFOLIO: name] prefix to identify which portfolio triggered</li>
        </ul>
      </div>

      <CreatePortfolioDialog
        open={createOpen}
        onOpenChange={setCreateOpen}
        onCreated={(id) => setSelectedId(id)}
      />
    </div>
  );
}

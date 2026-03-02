"use client";

import * as React from "react";
import type { Portfolio } from "../../../../../../gen/ts/alert/v1/alert";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

type PortfolioListProps = {
  portfolios: Portfolio[];
  selectedId: string | null;
  onSelect: (id: string) => void;
};

export function PortfolioList({
  portfolios,
  selectedId,
  onSelect,
}: PortfolioListProps) {
  if (portfolios.length === 0) {
    return (
      <p className="text-muted-foreground text-sm">
        No portfolios yet. Create one to get started.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      {portfolios.map((p) => (
        <Card
          key={p.portfolioId}
          className={cn(
            "cursor-pointer transition-colors hover:bg-accent/50",
            selectedId === p.portfolioId && "border-primary bg-accent/30"
          )}
          onClick={() => onSelect(p.portfolioId)}
        >
          <CardContent className="p-3">
            <div className="flex items-center justify-between">
              <span className="font-medium text-sm">{p.name}</span>
              <Badge variant={p.enabled ? "default" : "secondary"}>
                {p.enabled ? "Active" : "Disabled"}
              </Badge>
            </div>
            <div className="mt-1 flex items-center gap-3 text-xs text-muted-foreground">
              <span>{p.tickers.length} stocks</span>
              {p.discordWebhook ? (
                <span className="text-green-600">Webhook set</span>
              ) : (
                <span className="text-yellow-600">No webhook</span>
              )}
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

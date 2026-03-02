"use client";

import type { Portfolio } from "../../../../../../gen/ts/alert/v1/alert";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";

type PortfolioSummaryTableProps = {
  portfolios: Portfolio[];
};

export function PortfolioSummaryTable({ portfolios }: PortfolioSummaryTableProps) {
  if (portfolios.length === 0) return null;

  return (
    <div className="space-y-2">
      <h3 className="font-semibold text-sm">All Portfolios Summary</h3>
      <div className="rounded-lg border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead className="text-right">Stocks</TableHead>
              <TableHead>Discord</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Last Updated</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {portfolios.map((p) => (
              <TableRow key={p.portfolioId}>
                <TableCell className="font-medium">{p.name}</TableCell>
                <TableCell className="text-right">{p.tickers.length}</TableCell>
                <TableCell>
                  {p.discordWebhook ? (
                    <Badge variant="default" className="text-xs">Set</Badge>
                  ) : (
                    <Badge variant="secondary" className="text-xs">None</Badge>
                  )}
                </TableCell>
                <TableCell>
                  <Badge variant={p.enabled ? "default" : "secondary"}>
                    {p.enabled ? "Enabled" : "Disabled"}
                  </Badge>
                </TableCell>
                <TableCell className="text-muted-foreground">
                  {p.lastUpdated ? p.lastUpdated.slice(0, 10) : "-"}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}

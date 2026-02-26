"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

type StockDatabaseStatsCardsProps = {
  totalSymbols: number;
  uniqueExchanges: number;
  uniqueCountries: number;
  stockCount: number;
  etfCount: number;
};

export function StockDatabaseStatsCards({
  totalSymbols,
  uniqueExchanges,
  uniqueCountries,
  stockCount,
  etfCount,
}: StockDatabaseStatsCardsProps) {
  return (
    <div className="grid gap-4 md:grid-cols-5">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Total symbols
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-2xl font-bold">{totalSymbols.toLocaleString()}</p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Exchanges
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-2xl font-bold">{uniqueExchanges.toLocaleString()}</p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Countries
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-2xl font-bold">{uniqueCountries.toLocaleString()}</p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            Stocks
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-2xl font-bold">{stockCount.toLocaleString()}</p>
        </CardContent>
      </Card>
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm font-medium text-muted-foreground">
            ETFs
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-2xl font-bold">{etfCount.toLocaleString()}</p>
        </CardContent>
      </Card>
    </div>
  );
}

"use client";

import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export function IndicatorGuide() {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium">
          Indicator guide & condition syntax
        </CardTitle>
      </CardHeader>
      <CardContent className="text-muted-foreground text-xs space-y-3">
        <details className="group">
          <summary className="cursor-pointer list-none font-medium text-foreground">
            Price conditions
          </summary>
          <ul className="mt-2 list-disc pl-4 space-y-1">
            <li>
              <code>price_above: 150</code> – current price above value
            </li>
            <li>
              <code>price_below: 100</code> – current price below value
            </li>
            <li>
              <code>price_equals: 125.50</code> – current price equals value
            </li>
          </ul>
        </details>
        <details className="group">
          <summary className="cursor-pointer list-none font-medium text-foreground">
            Moving averages
          </summary>
          <ul className="mt-2 list-disc pl-4 space-y-1">
            <li>Price above/below MA(period) – SMA or EMA</li>
            <li>MA crossover: fast period &gt; slow period</li>
          </ul>
        </details>
        <details className="group">
          <summary className="cursor-pointer list-none font-medium text-foreground">
            RSI
          </summary>
          <ul className="mt-2 list-disc pl-4 space-y-1">
            <li>Oversold: RSI(period)[-1] &lt; 30</li>
            <li>Overbought: RSI(period)[-1] &gt; 70</li>
          </ul>
        </details>
        <details className="group">
          <summary className="cursor-pointer list-none font-medium text-foreground">
            MACD
          </summary>
          <ul className="mt-2 list-disc pl-4 space-y-1">
            <li>Bullish/bearish crossover, histogram positive</li>
          </ul>
        </details>
        <details className="group">
          <summary className="cursor-pointer list-none font-medium text-foreground">
            Bollinger Bands &amp; volume
          </summary>
          <ul className="mt-2 list-disc pl-4 space-y-1">
            <li>Price above upper / below lower band</li>
            <li>Volume above average (e.g. 1.5x)</li>
          </ul>
        </details>
      </CardContent>
    </Card>
  );
}

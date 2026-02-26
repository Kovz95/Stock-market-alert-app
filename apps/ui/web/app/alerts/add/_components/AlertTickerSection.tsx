"use client";

import {
  Field,
  FieldGroup,
  FieldLabel,
  FieldContent,
  FieldLegend,
} from "@/components/ui/field";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const ADJUSTMENT_METHODS = ["None", "split", "dividend"] as const;

export interface AlertTickerSectionProps {
  isRatio: boolean;
  onRatioChange: (v: boolean) => void;
  ticker: string;
  onTickerChange: (v: string) => void;
  stockName: string;
  onStockNameChange: (v: string) => void;
  ticker1: string;
  onTicker1Change: (v: string) => void;
  ticker2: string;
  onTicker2Change: (v: string) => void;
  stockName1: string;
  onStockName1Change: (v: string) => void;
  stockName2: string;
  onStockName2Change: (v: string) => void;
  adjustmentMethod: string;
  onAdjustmentMethodChange: (v: string) => void;
}

export function AlertTickerSection({
  isRatio,
  onRatioChange,
  ticker,
  onTickerChange,
  stockName,
  onStockNameChange,
  ticker1,
  onTicker1Change,
  ticker2,
  onTicker2Change,
  stockName1,
  onStockName1Change,
  stockName2,
  onStockName2Change,
  adjustmentMethod,
  onAdjustmentMethodChange,
}: AlertTickerSectionProps) {
  return (
    <FieldGroup>
      <Field>
        <FieldLegend>Ratio of 2 assets?</FieldLegend>
        <FieldContent>
          <Select
            value={isRatio ? "Yes" : "No"}
            onValueChange={(v) => onRatioChange(v === "Yes")}
          >
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="No">No</SelectItem>
              <SelectItem value="Yes">Yes</SelectItem>
            </SelectContent>
          </Select>
        </FieldContent>
      </Field>

      {!isRatio ? (
        <>
          <Field>
            <FieldLabel>Stock / asset name</FieldLabel>
            <FieldContent>
              <Input
                placeholder="e.g. Apple Inc."
                value={stockName}
                onChange={(e) => onStockNameChange(e.target.value)}
              />
            </FieldContent>
          </Field>
          <Field>
            <FieldLabel>Ticker symbol *</FieldLabel>
            <FieldContent>
              <Input
                placeholder="e.g. AAPL"
                value={ticker}
                onChange={(e) => onTickerChange(e.target.value)}
              />
            </FieldContent>
          </Field>
        </>
      ) : (
        <>
          <Field>
            <FieldLabel>First asset name</FieldLabel>
            <FieldContent>
              <Input
                placeholder="e.g. SPY"
                value={stockName1}
                onChange={(e) => onStockName1Change(e.target.value)}
              />
            </FieldContent>
          </Field>
          <Field>
            <FieldLabel>First ticker *</FieldLabel>
            <FieldContent>
              <Input
                placeholder="e.g. SPY"
                value={ticker1}
                onChange={(e) => onTicker1Change(e.target.value)}
              />
            </FieldContent>
          </Field>
          <Field>
            <FieldLabel>Second asset name</FieldLabel>
            <FieldContent>
              <Input
                placeholder="e.g. QQQ"
                value={stockName2}
                onChange={(e) => onStockName2Change(e.target.value)}
              />
            </FieldContent>
          </Field>
          <Field>
            <FieldLabel>Second ticker *</FieldLabel>
            <FieldContent>
              <Input
                placeholder="e.g. QQQ"
                value={ticker2}
                onChange={(e) => onTicker2Change(e.target.value)}
              />
            </FieldContent>
          </Field>
          <Field>
            <FieldLegend>Adjustment method (futures/ratio)</FieldLegend>
            <FieldContent>
              <Select
                value={adjustmentMethod || "None"}
                onValueChange={onAdjustmentMethodChange}
              >
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {ADJUSTMENT_METHODS.map((m) => (
                    <SelectItem key={m} value={m}>
                      {m}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </FieldContent>
          </Field>
        </>
      )}
    </FieldGroup>
  );
}

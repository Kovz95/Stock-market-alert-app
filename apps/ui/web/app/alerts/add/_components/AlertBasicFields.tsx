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

const TIMEFRAMES = ["1D", "1W", "1M"] as const;
const EXCHANGES = ["NYSE", "NASDAQ", "US"] as const;
const COUNTRIES = ["US", "CA"] as const;

export interface AlertBasicFieldsProps {
  name: string;
  onNameChange: (v: string) => void;
  action: "Buy" | "Sell";
  onActionChange: (v: "Buy" | "Sell") => void;
  timeframe: string;
  onTimeframeChange: (v: string) => void;
  exchange: string;
  onExchangeChange: (v: string) => void;
  country: string;
  onCountryChange: (v: string) => void;
}

export function AlertBasicFields({
  name,
  onNameChange,
  action,
  onActionChange,
  timeframe,
  onTimeframeChange,
  exchange,
  onExchangeChange,
  country,
  onCountryChange,
}: AlertBasicFieldsProps) {
  return (
    <FieldGroup>
      <Field>
        <FieldLabel>Alert name (optional)</FieldLabel>
        <FieldContent>
          <Input
            placeholder="e.g. AAPL Price Alert"
            value={name}
            onChange={(e) => onNameChange(e.target.value)}
          />
        </FieldContent>
      </Field>
      <Field>
        <FieldLegend>Action</FieldLegend>
        <FieldContent>
          <Select value={action} onValueChange={(v) => onActionChange(v as "Buy" | "Sell")}>
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="Buy">Buy</SelectItem>
              <SelectItem value="Sell">Sell</SelectItem>
            </SelectContent>
          </Select>
        </FieldContent>
      </Field>
      <Field>
        <FieldLegend>Timeframe</FieldLegend>
        <FieldContent>
          <Select value={timeframe} onValueChange={onTimeframeChange}>
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {TIMEFRAMES.map((tf) => (
                <SelectItem key={tf} value={tf}>
                  {tf}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </FieldContent>
      </Field>
      <Field>
        <FieldLegend>Exchange</FieldLegend>
        <FieldContent>
          <Select value={exchange} onValueChange={onExchangeChange}>
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {EXCHANGES.map((ex) => (
                <SelectItem key={ex} value={ex}>
                  {ex}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </FieldContent>
      </Field>
      <Field>
        <FieldLegend>Country</FieldLegend>
        <FieldContent>
          <Select value={country} onValueChange={onCountryChange}>
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {COUNTRIES.map((c) => (
                <SelectItem key={c} value={c}>
                  {c}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </FieldContent>
      </Field>
    </FieldGroup>
  );
}

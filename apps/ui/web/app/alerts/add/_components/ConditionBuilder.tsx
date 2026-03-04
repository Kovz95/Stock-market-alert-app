"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
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
import {
  type ConditionEntry,
  type ConditionCategory,
  type ConditionParams,
  type PriceConditionType,
  type MAConditionType,
  type RSIConditionType,
  type MACDConditionType,
  type BBConditionType,
  type VolumeConditionType,
  type MASlopeCurveConditionType,
} from "./types";

const CATEGORY_OPTIONS: { value: ConditionCategory; label: string }[] = [
  { value: "price", label: "Price" },
  { value: "moving_average", label: "Moving Average" },
  { value: "rsi", label: "RSI" },
  { value: "macd", label: "MACD" },
  { value: "bollinger", label: "Bollinger Bands" },
  { value: "volume", label: "Volume" },
  { value: "ma_slope_curve", label: "MA Slope + Curvature" },
  { value: "custom", label: "Custom expression" },
];

const PRICE_TYPES: { value: PriceConditionType; label: string }[] = [
  { value: "price_above", label: "Price above" },
  { value: "price_below", label: "Price below" },
  { value: "price_equals", label: "Price equals" },
];

const MA_TYPES: { value: MAConditionType; label: string }[] = [
  { value: "price_above_ma", label: "Price above MA" },
  { value: "price_below_ma", label: "Price below MA" },
  { value: "ma_crossover", label: "MA crossover" },
];

const RSI_TYPES: { value: RSIConditionType; label: string }[] = [
  { value: "rsi_oversold", label: "RSI oversold" },
  { value: "rsi_overbought", label: "RSI overbought" },
  { value: "rsi_level", label: "RSI at level" },
];

const MACD_TYPES: { value: MACDConditionType; label: string }[] = [
  { value: "macd_bullish_crossover", label: "MACD bullish crossover" },
  { value: "macd_bearish_crossover", label: "MACD bearish crossover" },
  { value: "macd_histogram_positive", label: "MACD histogram positive" },
];

const BB_TYPES: { value: BBConditionType; label: string }[] = [
  { value: "price_above_upper_band", label: "Price above upper band" },
  { value: "price_below_lower_band", label: "Price below lower band" },
];

const VOLUME_TYPES: { value: VolumeConditionType; label: string }[] = [
  { value: "volume_above_average", label: "Volume above average" },
  { value: "volume_spike", label: "Volume spike" },
];

const MA_SLOPE_CURVE_TYPES: { value: MASlopeCurveConditionType; label: string }[] = [
  { value: "slope_positive", label: "Slope > 0" },
  { value: "slope_negative", label: "Slope < 0" },
  { value: "slope_turn_up", label: "Slope turn up (pulse)" },
  { value: "slope_turn_dn", label: "Slope turn down (pulse)" },
  { value: "curve_positive", label: "Curvature > 0" },
  { value: "curve_negative", label: "Curvature < 0" },
  { value: "bend_up", label: "Curvature bend up (pulse)" },
  { value: "bend_dn", label: "Curvature bend down (pulse)" },
  { value: "early_bend_up", label: "Early bend up (pulse)" },
  { value: "early_bend_dn", label: "Early bend down (pulse)" },
];

function generateId(): string {
  return `cond_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
}

/** Resolve params with the same defaults the UI displays, so "Add condition" saves what the user sees. */
function resolveParamsForAdd(
  category: ConditionCategory,
  type: string,
  params: ConditionParams
): ConditionParams {
  const p = { ...params };
  if (category === "rsi") {
    p.rsiPeriod = p.rsiPeriod ?? 14;
    if (type === "rsi_oversold") p.oversoldLevel = p.oversoldLevel ?? 30;
    if (type === "rsi_overbought") p.overboughtLevel = p.overboughtLevel ?? 70;
    if (type === "rsi_level") {
      p.rsiLevel = p.rsiLevel ?? 50;
      p.rsiLevelOperator = p.rsiLevelOperator ?? ">";
    }
  }
  if (category === "moving_average") {
    if (type === "price_above_ma" || type === "price_below_ma") {
      p.maPeriod = p.maPeriod ?? 20;
      p.maType = p.maType ?? "SMA";
    }
    if (type === "ma_crossover") {
      p.fastPeriod = p.fastPeriod ?? 10;
      p.slowPeriod = p.slowPeriod ?? 20;
    }
  }
  if (category === "volume") {
    p.volumeMultiplier = p.volumeMultiplier ?? 1.5;
  }
  if (category === "bollinger") {
    p.bbPeriod = p.bbPeriod ?? 20;
    p.bbStd = p.bbStd ?? 2;
  }
  if (category === "ma_slope_curve") {
    p.maLen = p.maLen ?? 200;
    p.maType = p.maType ?? "HMA";
    p.slopeLookback = p.slopeLookback ?? 3;
    p.smoothType = p.smoothType ?? "EMA";
    p.smoothLen = p.smoothLen ?? 2;
    p.normMode = p.normMode ?? "ATR";
    p.atrLen = p.atrLen ?? 14;
    p.slopeThr = p.slopeThr ?? 0;
    p.curveThr = p.curveThr ?? 0;
  }
  return p;
}

export interface ConditionBuilderProps {
  onAdd: (entry: ConditionEntry) => void;
}

export function ConditionBuilder({ onAdd }: ConditionBuilderProps) {
  const [category, setCategory] = useState<ConditionCategory>("price");
  const [type, setType] = useState<string>("price_above");
  const [params, setParams] = useState<ConditionParams>({ priceValue: 100 });
  const isMaSlopeCurve = category === "ma_slope_curve";

  const handleAdd = () => {
    const resolved = resolveParamsForAdd(category, type, params);
    const entry: ConditionEntry = {
      id: generateId(),
      category,
      type,
      params: resolved,
    };
    onAdd(entry);
    setParams({ priceValue: 100 });
    setType("price_above");
    setCategory("price");
  };

  const typeOptions =
    category === "price"
      ? PRICE_TYPES
      : category === "moving_average"
        ? MA_TYPES
        : category === "rsi"
          ? RSI_TYPES
          : category === "macd"
            ? MACD_TYPES
            : category === "bollinger"
              ? BB_TYPES
              : category === "volume"
                ? VOLUME_TYPES
                : category === "ma_slope_curve"
                  ? MA_SLOPE_CURVE_TYPES
                  : [];

  const showTypeSelect = typeOptions.length > 0;

  return (
    <div className="space-y-4 rounded-lg border bg-muted/20 p-4">
      <FieldLegend>Add condition</FieldLegend>
      <FieldGroup>
        <Field>
          <FieldLabel>Category</FieldLabel>
          <FieldContent>
            <Select
              value={category}
              onValueChange={(v) => {
                setCategory(v as ConditionCategory);
                if (v === "price") setType("price_above");
                else if (v === "moving_average") setType("price_above_ma");
                else if (v === "rsi") setType("rsi_oversold");
                else if (v === "macd") setType("macd_bullish_crossover");
                else if (v === "bollinger") setType("price_above_upper_band");
                else if (v === "volume") setType("volume_above_average");
                else if (v === "ma_slope_curve") setType("slope_positive");
              }}
            >
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {CATEGORY_OPTIONS.map((opt) => (
                  <SelectItem key={opt.value} value={opt.value}>
                    {opt.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </FieldContent>
        </Field>

        {showTypeSelect && (
          <Field>
            <FieldLabel>Condition type</FieldLabel>
            <FieldContent>
              <Select
              value={type}
              onValueChange={(v) => {
                setType(v);
                if (v === "rsi_level" && (params.rsiLevel == null || params.rsiLevelOperator == null))
                  setParams({
                    ...params,
                    rsiLevel: params.rsiLevel ?? 50,
                    rsiLevelOperator: params.rsiLevelOperator ?? ">",
                  });
              }}
            >
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {typeOptions.map((opt) => (
                    <SelectItem key={opt.value} value={opt.value}>
                      {opt.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </FieldContent>
          </Field>
        )}

        {category === "price" && (
          <Field>
            <FieldLabel>Price value</FieldLabel>
            <FieldContent>
              <Input
                type="number"
                step="0.01"
                placeholder="e.g. 150"
                value={params.priceValue ?? ""}
                onChange={(e) =>
                  setParams({
                    ...params,
                    priceValue: e.target.value ? Number(e.target.value) : undefined,
                  })
                }
              />
            </FieldContent>
          </Field>
        )}

        {(category === "moving_average" &&
          (type === "price_above_ma" || type === "price_below_ma")) && (
          <>
            <Field>
              <FieldLabel>MA type</FieldLabel>
              <FieldContent>
                <Select
                  value={params.maType ?? "SMA"}
                  onValueChange={(v) => setParams({ ...params, maType: v })}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="SMA">SMA</SelectItem>
                    <SelectItem value="EMA">EMA</SelectItem>
                  </SelectContent>
                </Select>
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>Period</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  value={params.maPeriod ?? 20}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      maPeriod: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
          </>
        )}

        {category === "moving_average" && type === "ma_crossover" && (
          <>
            <Field>
              <FieldLabel>Fast period</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  value={params.fastPeriod ?? 10}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      fastPeriod: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>Slow period</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  value={params.slowPeriod ?? 20}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      slowPeriod: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
          </>
        )}

        {category === "rsi" && (
          <>
            <Field>
              <FieldLabel>RSI period</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  value={params.rsiPeriod ?? 14}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      rsiPeriod: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            {(type === "rsi_oversold" || type === "rsi_overbought") && (
              <Field>
                <FieldLabel>{type === "rsi_oversold" ? "Oversold level" : "Overbought level"}</FieldLabel>
                <FieldContent>
                  <Input
                    type="number"
                    min={0}
                    max={100}
                    value={
                      type === "rsi_oversold"
                        ? (params.oversoldLevel ?? 30)
                        : (params.overboughtLevel ?? 70)
                    }
                    onChange={(e) => {
                      const n = e.target.value ? Number(e.target.value) : undefined;
                      if (type === "rsi_oversold")
                        setParams({ ...params, oversoldLevel: n });
                      else setParams({ ...params, overboughtLevel: n });
                    }}
                  />
                </FieldContent>
              </Field>
            )}
            {type === "rsi_level" && (
              <>
                <Field>
                  <FieldLabel>Comparison</FieldLabel>
                  <FieldContent>
                    <Select
                      value={params.rsiLevelOperator ?? ">"}
                      onValueChange={(v) =>
                        setParams({ ...params, rsiLevelOperator: v })
                      }
                    >
                      <SelectTrigger className="w-full">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value=">">{">"} (above)</SelectItem>
                        <SelectItem value="<">{"<"} (below)</SelectItem>
                        <SelectItem value=">=">{"≥"} (at or above)</SelectItem>
                        <SelectItem value="<=">{"≤"} (at or below)</SelectItem>
                      </SelectContent>
                    </Select>
                  </FieldContent>
                </Field>
                <Field>
                  <FieldLabel>RSI level (0–100)</FieldLabel>
                  <FieldContent>
                    <Input
                      type="number"
                      min={0}
                      max={100}
                      value={params.rsiLevel ?? 50}
                      onChange={(e) =>
                        setParams({
                          ...params,
                          rsiLevel: e.target.value
                            ? Number(e.target.value)
                            : undefined,
                        })
                      }
                    />
                  </FieldContent>
                </Field>
              </>
            )}
          </>
        )}

        {category === "volume" && (
          <Field>
            <FieldLabel>Volume multiplier (e.g. 1.5 for 1.5x)</FieldLabel>
            <FieldContent>
              <Input
                type="number"
                step="0.1"
                min={0}
                value={params.volumeMultiplier ?? 1.5}
                onChange={(e) =>
                  setParams({
                    ...params,
                    volumeMultiplier: e.target.value ? Number(e.target.value) : undefined,
                  })
                }
              />
            </FieldContent>
          </Field>
        )}

        {category === "bollinger" && (
          <>
            <Field>
              <FieldLabel>Period</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  value={params.bbPeriod ?? 20}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      bbPeriod: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>Std dev</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  step="0.1"
                  value={params.bbStd ?? 2}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      bbStd: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
          </>
        )}

        {isMaSlopeCurve && (
          <>
            <Field>
              <FieldLabel>MA type</FieldLabel>
              <FieldContent>
                <Select
                  value={params.maType ?? "HMA"}
                  onValueChange={(v) => setParams({ ...params, maType: v })}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="HMA">HMA</SelectItem>
                    <SelectItem value="EMA">EMA</SelectItem>
                    <SelectItem value="SMA">SMA</SelectItem>
                    <SelectItem value="WMA">WMA</SelectItem>
                    <SelectItem value="RMA">RMA</SelectItem>
                  </SelectContent>
                </Select>
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>MA length</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  value={params.maLen ?? 200}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      maLen: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>Slope lookback (bars)</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  value={params.slopeLookback ?? 3}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      slopeLookback: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>Smoothing</FieldLabel>
              <FieldContent>
                <Select
                  value={params.smoothType ?? "EMA"}
                  onValueChange={(v) => setParams({ ...params, smoothType: v })}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="None">None</SelectItem>
                    <SelectItem value="EMA">EMA</SelectItem>
                    <SelectItem value="SMA">SMA</SelectItem>
                    <SelectItem value="RMA">RMA</SelectItem>
                  </SelectContent>
                </Select>
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>Smooth length</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  value={params.smoothLen ?? 2}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      smoothLen: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>Normalize by</FieldLabel>
              <FieldContent>
                <Select
                  value={params.normMode ?? "ATR"}
                  onValueChange={(v) => setParams({ ...params, normMode: v })}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="None">None</SelectItem>
                    <SelectItem value="ATR">ATR</SelectItem>
                    <SelectItem value="Percent">Percent</SelectItem>
                  </SelectContent>
                </Select>
              </FieldContent>
            </Field>
            {(params.normMode ?? "ATR") === "ATR" && (
              <Field>
                <FieldLabel>ATR length</FieldLabel>
                <FieldContent>
                  <Input
                    type="number"
                    min={1}
                    value={params.atrLen ?? 14}
                    onChange={(e) =>
                      setParams({
                        ...params,
                        atrLen: e.target.value ? Number(e.target.value) : undefined,
                      })
                    }
                  />
                </FieldContent>
              </Field>
            )}
            <Field>
              <FieldLabel>Slope threshold (optional)</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  step="0.0001"
                  value={params.slopeThr ?? 0}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      slopeThr: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>Curvature threshold (optional)</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  step="0.0001"
                  value={params.curveThr ?? 0}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      curveThr: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
          </>
        )}

        {category === "custom" && (
          <Field>
            <FieldLabel>Expression</FieldLabel>
            <FieldContent>
              <Input
                placeholder="e.g. close[-1] > 100"
                value={params.customExpression ?? ""}
                onChange={(e) =>
                  setParams({ ...params, customExpression: e.target.value })
                }
              />
            </FieldContent>
          </Field>
        )}
      </FieldGroup>
      <Button type="button" onClick={handleAdd} size="sm">
        Add condition
      </Button>
    </div>
  );
}

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
  type DonchianConditionType,
  type PivotSRConditionType,
  type IchimokuConditionType,
  type TrendMagicConditionType,
  type SupertrendConditionType,
  type SARConditionType,
  type OBVMACDConditionType,
  type HARSIConditionType,
  type MAZScoreConditionType,
  type EWOConditionType,
  type ROCConditionType,
  type WillRConditionType,
  type CCIConditionType,
  type ATRConditionType,
  type KalmanROCConditionType,
} from "./types";
import { INDICATOR_NAMES } from "./indicatorList";

const CATEGORY_OPTIONS: { value: ConditionCategory; label: string }[] = [
  { value: "price", label: "Price" },
  { value: "moving_average", label: "Moving Average" },
  { value: "rsi", label: "RSI" },
  { value: "macd", label: "MACD" },
  { value: "bollinger", label: "Bollinger Bands" },
  { value: "volume", label: "Volume" },
  { value: "donchian", label: "Donchian Channels" },
  { value: "pivot_sr", label: "Pivot S/R" },
  { value: "ichimoku", label: "Ichimoku Cloud" },
  { value: "trend_magic", label: "Trend Magic" },
  { value: "supertrend", label: "SuperTrend" },
  { value: "sar", label: "SAR" },
  { value: "obv_macd", label: "OBV MACD" },
  { value: "harsi", label: "HARSI" },
  { value: "ma_zscore", label: "MA Z-Score" },
  { value: "ewo", label: "EWO" },
  { value: "roc", label: "ROC" },
  { value: "willr", label: "Williams %R" },
  { value: "cci", label: "CCI" },
  { value: "atr", label: "ATR" },
  { value: "kalman_roc_stoch", label: "Kalman ROC Stoch" },
  { value: "ma_slope_curve", label: "MA Slope + Curvature" },
  { value: "indicator", label: "Any indicator" },
  { value: "custom", label: "Custom expression" },
];

const INDICATOR_OPERATORS: { value: string; label: string }[] = [
  { value: ">", label: "> (greater than)" },
  { value: "<", label: "< (less than)" },
  { value: ">=", label: "≥ (at least)" },
  { value: "<=", label: "≤ (at most)" },
  { value: "==", label: "= (equals)" },
  { value: "!=", label: "≠ (not equals)" },
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

const DONCHIAN_TYPES: { value: DonchianConditionType; label: string; group: string }[] = [
  // Channel Lines
  { value: "donchian_upper_value", label: "Upper band value", group: "Channel Lines" },
  { value: "donchian_lower_value", label: "Lower band value", group: "Channel Lines" },
  { value: "donchian_basis_value", label: "Basis (middle) value", group: "Channel Lines" },
  { value: "donchian_price_vs_upper", label: "Price above upper band", group: "Channel Lines" },
  { value: "donchian_price_vs_lower", label: "Price below lower band", group: "Channel Lines" },
  { value: "donchian_price_vs_basis", label: "Price above basis", group: "Channel Lines" },
  // Channel Breakout
  { value: "donchian_breakout_upper", label: "Upper band breakout", group: "Channel Breakout" },
  { value: "donchian_breakout_lower", label: "Lower band breakout", group: "Channel Breakout" },
  { value: "donchian_basis_cross_up", label: "Basis cross up", group: "Channel Breakout" },
  { value: "donchian_basis_cross_down", label: "Basis cross down", group: "Channel Breakout" },
  // Channel Position
  { value: "donchian_position_value", label: "Position value (0–1)", group: "Channel Position" },
  { value: "donchian_position_above", label: "Near upper band (>0.8)", group: "Channel Position" },
  { value: "donchian_position_below", label: "Near lower band (<0.2)", group: "Channel Position" },
  { value: "donchian_position_near_middle", label: "Near middle (0.4–0.6)", group: "Channel Position" },
  // Channel Width
  { value: "donchian_width_value", label: "Width value", group: "Channel Width" },
  { value: "donchian_width_expanding", label: "Width expanding", group: "Channel Width" },
  { value: "donchian_width_contracting", label: "Width contracting", group: "Channel Width" },
];
const PIVOT_SR_TYPES: { value: PivotSRConditionType; label: string }[] = [
  { value: "pivot_sr_bullish", label: "Bullish (breakout)" },
  { value: "pivot_sr_bearish", label: "Bearish (breakdown)" },
  { value: "pivot_sr_near_support", label: "Near support" },
  { value: "pivot_sr_near_resistance", label: "Near resistance" },
  { value: "pivot_sr_crossover_bullish", label: "Bullish crossover" },
  { value: "pivot_sr_crossover_bearish", label: "Bearish crossover" },
];
const ICHIMOKU_TYPES: { value: IchimokuConditionType; label: string; group: string }[] = [
  // Price vs Cloud
  { value: "ichimoku_price_above_cloud", label: "Price above cloud", group: "Price vs Cloud" },
  { value: "ichimoku_price_below_cloud", label: "Price below cloud", group: "Price vs Cloud" },
  { value: "ichimoku_price_in_cloud", label: "Price in cloud", group: "Price vs Cloud" },
  { value: "ichimoku_price_entered_cloud_above", label: "Entered cloud (from above)", group: "Price vs Cloud" },
  { value: "ichimoku_price_entered_cloud_below", label: "Entered cloud (from below)", group: "Price vs Cloud" },
  { value: "ichimoku_price_entered_cloud_any", label: "Entered cloud (any direction)", group: "Price vs Cloud" },
  { value: "ichimoku_price_crossed_above_cloud", label: "Crossed above cloud", group: "Price vs Cloud" },
  { value: "ichimoku_price_crossed_below_cloud", label: "Crossed below cloud", group: "Price vs Cloud" },
  // Line Crossovers
  { value: "ichimoku_tk_cross_bull", label: "TK cross bullish", group: "Line Crossovers" },
  { value: "ichimoku_tk_cross_bear", label: "TK cross bearish", group: "Line Crossovers" },
  { value: "ichimoku_price_cross_above_conversion", label: "Price crosses above conversion", group: "Line Crossovers" },
  { value: "ichimoku_price_cross_below_conversion", label: "Price crosses below conversion", group: "Line Crossovers" },
  { value: "ichimoku_price_cross_above_base", label: "Price crosses above base", group: "Line Crossovers" },
  { value: "ichimoku_price_cross_below_base", label: "Price crosses below base", group: "Line Crossovers" },
  // Cloud Color
  { value: "ichimoku_cloud_bullish", label: "Bullish cloud (green)", group: "Cloud Color" },
  { value: "ichimoku_cloud_bearish", label: "Bearish cloud (red)", group: "Cloud Color" },
  { value: "ichimoku_cloud_turned_bullish", label: "Cloud turned bullish", group: "Cloud Color" },
  { value: "ichimoku_cloud_turned_bearish", label: "Cloud turned bearish", group: "Cloud Color" },
  // Individual Lines
  { value: "ichimoku_price_above_conversion", label: "Price above conversion line", group: "Individual Lines" },
  { value: "ichimoku_price_below_conversion", label: "Price below conversion line", group: "Individual Lines" },
  { value: "ichimoku_price_above_base", label: "Price above base line", group: "Individual Lines" },
  { value: "ichimoku_price_below_base", label: "Price below base line", group: "Individual Lines" },
  { value: "ichimoku_conversion_above_base", label: "Conversion above base", group: "Individual Lines" },
  { value: "ichimoku_conversion_below_base", label: "Conversion below base", group: "Individual Lines" },
  // Lagging Span
  { value: "ichimoku_lagging_above_price", label: "Lagging span above price", group: "Lagging Span" },
  { value: "ichimoku_lagging_below_price", label: "Lagging span below price", group: "Lagging Span" },
  { value: "ichimoku_lagging_crossed_above", label: "Lagging span crossed above", group: "Lagging Span" },
  { value: "ichimoku_lagging_crossed_below", label: "Lagging span crossed below", group: "Lagging Span" },
];
const TREND_MAGIC_TYPES: { value: TrendMagicConditionType; label: string }[] = [
  { value: "trend_magic_bullish", label: "Bullish" },
  { value: "trend_magic_bearish", label: "Bearish" },
];
const SUPERTREND_TYPES: { value: SupertrendConditionType; label: string }[] = [
  { value: "supertrend_uptrend", label: "Uptrend" },
  { value: "supertrend_downtrend", label: "Downtrend" },
];
const SAR_TYPES: { value: SARConditionType; label: string }[] = [
  { value: "sar_value", label: "SAR value" },
  { value: "sar_price_above", label: "Price above SAR" },
  { value: "sar_price_below", label: "Price below SAR" },
  { value: "sar_cross_above", label: "Price crossed above SAR" },
  { value: "sar_cross_below", label: "Price crossed below SAR" },
];
const OBV_MACD_TYPES: { value: OBVMACDConditionType; label: string }[] = [
  { value: "obv_macd_positive", label: "OBV MACD > 0" },
  { value: "obv_macd_above_signal", label: "OBV MACD above signal" },
];
const HARSI_TYPES: { value: HARSIConditionType; label: string }[] = [
  { value: "harsi_bullish", label: "Bullish" },
  { value: "harsi_bearish", label: "Bearish" },
];
const MA_ZSCORE_TYPES: { value: MAZScoreConditionType; label: string }[] = [
  { value: "ma_zscore_above", label: "Z-Score above threshold" },
  { value: "ma_zscore_below", label: "Z-Score below -threshold" },
];
const EWO_TYPES: { value: EWOConditionType; label: string }[] = [
  { value: "ewo_positive", label: "EWO > 0" },
  { value: "ewo_negative", label: "EWO < 0" },
];
const ROC_TYPES: { value: ROCConditionType; label: string }[] = [
  { value: "roc_above", label: "ROC above level" },
  { value: "roc_below", label: "ROC below level" },
];
const WILLR_TYPES: { value: WillRConditionType; label: string }[] = [
  { value: "willr_oversold", label: "Oversold (< -80)" },
  { value: "willr_overbought", label: "Overbought (> -20)" },
];
const CCI_TYPES: { value: CCIConditionType; label: string }[] = [
  { value: "cci_above", label: "CCI > 100" },
  { value: "cci_below", label: "CCI < -100" },
];
const ATR_TYPES: { value: ATRConditionType; label: string }[] = [
  { value: "atr_above", label: "ATR above value" },
  { value: "atr_below", label: "ATR below value" },
];
const KALMAN_ROC_STOCH_TYPES: { value: KalmanROCConditionType; label: string }[] = [
  { value: "kalman_roc_stoch_positive", label: "Kalman ROC Stoch > 0" },
  { value: "kalman_roc_stoch_signal_bullish", label: "Signal bullish" },
  { value: "kalman_roc_stoch_crossover_bullish", label: "Bullish crossover" },
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
  if (category === "indicator") {
    p.indicatorOperator = p.indicatorOperator ?? ">";
    p.indicatorValue = p.indicatorValue ?? 0;
  }
  if (category === "donchian") {
    p.donchianLength = p.donchianLength ?? 20;
    p.donchianOffset = p.donchianOffset ?? 0;
  }
  if (category === "ichimoku") {
    p.ichConversion = p.ichConversion ?? 9;
    p.ichBase = p.ichBase ?? 26;
    p.ichSpanB = p.ichSpanB ?? 52;
    p.ichDisplacement = p.ichDisplacement ?? 26;
  }
  if (category === "pivot_sr") {
    p.pivotLeftBars = p.pivotLeftBars ?? 5;
    p.pivotRightBars = p.pivotRightBars ?? 5;
  }
  if (category === "supertrend") {
    p.supertrendPeriod = p.supertrendPeriod ?? 10;
    p.supertrendMultiplier = p.supertrendMultiplier ?? 3;
  }
  if (category === "sar") {
    p.sarAcceleration = p.sarAcceleration ?? 0.02;
    p.sarMaxAcceleration = p.sarMaxAcceleration ?? 0.2;
  }
  if (category === "harsi") p.harsiPeriod = p.harsiPeriod ?? 14;
  if (category === "ma_zscore") {
    p.maZScoreThreshold = p.maZScoreThreshold ?? 2;
    p.maZScoreMaLength = p.maZScoreMaLength ?? 20;
  }
  if (category === "ewo") {
    p.ewoSma1Length = p.ewoSma1Length ?? 5;
    p.ewoSma2Length = p.ewoSma2Length ?? 35;
  }
  if (category === "roc") {
    p.rocPeriod = p.rocPeriod ?? 14;
    p.rocLevel = p.rocLevel ?? 0;
  }
  if (category === "willr") p.willrPeriod = p.willrPeriod ?? 14;
  if (category === "cci") p.cciPeriod = p.cciPeriod ?? 20;
  if (category === "atr") {
    p.atrPeriod = p.atrPeriod ?? 14;
    p.atrValue = p.atrValue ?? 2;
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
                : category === "donchian"
                  ? DONCHIAN_TYPES
                  : category === "pivot_sr"
                    ? PIVOT_SR_TYPES
                    : category === "ichimoku"
                      ? ICHIMOKU_TYPES
                      : category === "trend_magic"
                        ? TREND_MAGIC_TYPES
                        : category === "supertrend"
                          ? SUPERTREND_TYPES
                          : category === "sar"
                            ? SAR_TYPES
                            : category === "obv_macd"
                              ? OBV_MACD_TYPES
                              : category === "harsi"
                                ? HARSI_TYPES
                                : category === "ma_zscore"
                                  ? MA_ZSCORE_TYPES
                                  : category === "ewo"
                                    ? EWO_TYPES
                                    : category === "roc"
                                      ? ROC_TYPES
                                      : category === "willr"
                                        ? WILLR_TYPES
                                        : category === "cci"
                                          ? CCI_TYPES
                                          : category === "atr"
                                            ? ATR_TYPES
                                            : category === "kalman_roc_stoch"
                                              ? KALMAN_ROC_STOCH_TYPES
                                              : category === "ma_slope_curve"
                                                ? MA_SLOPE_CURVE_TYPES
                                                : category === "indicator"
                                                  ? []
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
                else if (v === "donchian") setType("donchian_upper_value");
                else if (v === "pivot_sr") setType("pivot_sr_bullish");
                else if (v === "ichimoku") setType("ichimoku_price_above_cloud");
                else if (v === "trend_magic") setType("trend_magic_bullish");
                else if (v === "supertrend") setType("supertrend_uptrend");
                else if (v === "sar") setType("sar_price_above");
                else if (v === "obv_macd") setType("obv_macd_positive");
                else if (v === "harsi") setType("harsi_bullish");
                else if (v === "ma_zscore") setType("ma_zscore_above");
                else if (v === "ewo") setType("ewo_positive");
                else if (v === "roc") setType("roc_above");
                else if (v === "willr") setType("willr_oversold");
                else if (v === "cci") setType("cci_above");
                else if (v === "atr") setType("atr_above");
                else if (v === "kalman_roc_stoch") setType("kalman_roc_stoch_positive");
                else if (v === "ma_slope_curve") setType("slope_positive");
                else if (v === "indicator") setType("indicator_compare");
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

        {category === "donchian" && (
          <>
            <Field>
              <FieldLabel>Channel length</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={5}
                  max={200}
                  value={params.donchianLength ?? 20}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      donchianLength: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>Offset</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={-50}
                  max={50}
                  value={params.donchianOffset ?? 0}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      donchianOffset: e.target.value !== "" ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
          </>
        )}

        {category === "ichimoku" && (
          <>
            <Field>
              <FieldLabel>Conversion line period</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  max={100}
                  value={params.ichConversion ?? 9}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      ichConversion: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>Base line period</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  max={100}
                  value={params.ichBase ?? 26}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      ichBase: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>Span B period</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  max={200}
                  value={params.ichSpanB ?? 52}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      ichSpanB: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>Displacement</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  max={100}
                  value={params.ichDisplacement ?? 26}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      ichDisplacement: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
          </>
        )}

        {category === "pivot_sr" && (
          <>
            <Field>
              <FieldLabel>Left bars</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  value={params.pivotLeftBars ?? 5}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      pivotLeftBars: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>Right bars</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  value={params.pivotRightBars ?? 5}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      pivotRightBars: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
          </>
        )}

        {category === "supertrend" && (
          <>
            <Field>
              <FieldLabel>Period</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  value={params.supertrendPeriod ?? 10}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      supertrendPeriod: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>Multiplier</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  step="0.1"
                  min={0.1}
                  value={params.supertrendMultiplier ?? 3}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      supertrendMultiplier: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
          </>
        )}

        {category === "sar" && (
          <>
            <Field>
              <FieldLabel>Acceleration</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  step="0.01"
                  value={params.sarAcceleration ?? 0.02}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      sarAcceleration: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>Max acceleration</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  step="0.01"
                  value={params.sarMaxAcceleration ?? 0.2}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      sarMaxAcceleration: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
          </>
        )}

        {category === "harsi" && (
          <Field>
            <FieldLabel>Period</FieldLabel>
            <FieldContent>
              <Input
                type="number"
                min={1}
                value={params.harsiPeriod ?? 14}
                onChange={(e) =>
                  setParams({
                    ...params,
                    harsiPeriod: e.target.value ? Number(e.target.value) : undefined,
                  })
                }
              />
            </FieldContent>
          </Field>
        )}

        {category === "ma_zscore" && (
          <>
            <Field>
              <FieldLabel>Z-Score threshold</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  step="0.1"
                  value={params.maZScoreThreshold ?? 2}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      maZScoreThreshold: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>MA length</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  value={params.maZScoreMaLength ?? 20}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      maZScoreMaLength: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
          </>
        )}

        {category === "ewo" && (
          <>
            <Field>
              <FieldLabel>SMA 1 length</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  value={params.ewoSma1Length ?? 5}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      ewoSma1Length: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>SMA 2 length</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  value={params.ewoSma2Length ?? 35}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      ewoSma2Length: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
          </>
        )}

        {category === "roc" && (
          <>
            <Field>
              <FieldLabel>Period</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  value={params.rocPeriod ?? 14}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      rocPeriod: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>Level (e.g. 0)</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  value={params.rocLevel ?? 0}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      rocLevel: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
          </>
        )}

        {category === "willr" && (
          <Field>
            <FieldLabel>Period</FieldLabel>
            <FieldContent>
              <Input
                type="number"
                min={1}
                value={params.willrPeriod ?? 14}
                onChange={(e) =>
                  setParams({
                    ...params,
                    willrPeriod: e.target.value ? Number(e.target.value) : undefined,
                  })
                }
              />
            </FieldContent>
          </Field>
        )}

        {category === "cci" && (
          <Field>
            <FieldLabel>Period</FieldLabel>
            <FieldContent>
              <Input
                type="number"
                min={1}
                value={params.cciPeriod ?? 20}
                onChange={(e) =>
                  setParams({
                    ...params,
                    cciPeriod: e.target.value ? Number(e.target.value) : undefined,
                  })
                }
              />
            </FieldContent>
          </Field>
        )}

        {category === "atr" && (
          <>
            <Field>
              <FieldLabel>Period</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  value={params.atrPeriod ?? 14}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      atrPeriod: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>ATR value (e.g. 2)</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  step="0.01"
                  value={params.atrValue ?? 2}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      atrValue: e.target.value ? Number(e.target.value) : undefined,
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

        {category === "indicator" && (
          <>
            <Field>
              <FieldLabel>Indicator</FieldLabel>
              <FieldContent>
                <Select
                  value={params.indicatorName ?? ""}
                  onValueChange={(v) =>
                    setParams({ ...params, indicatorName: v })
                  }
                >
                  <SelectTrigger className="w-full">
                    <SelectValue placeholder="Select indicator" />
                  </SelectTrigger>
                  <SelectContent>
                    {INDICATOR_NAMES.map((name) => (
                      <SelectItem key={name} value={name}>
                        {name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>Parameters (optional)</FieldLabel>
              <FieldContent>
                <Input
                  placeholder="e.g. 14 or 20, 2 or timeperiod=14"
                  value={params.indicatorParams ?? ""}
                  onChange={(e) =>
                    setParams({ ...params, indicatorParams: e.target.value })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>Compare</FieldLabel>
              <FieldContent>
                <Select
                  value={params.indicatorOperator ?? ">"}
                  onValueChange={(v) =>
                    setParams({ ...params, indicatorOperator: v })
                  }
                >
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {INDICATOR_OPERATORS.map((opt) => (
                      <SelectItem key={opt.value} value={opt.value}>
                        {opt.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>Value</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  step="any"
                  placeholder="e.g. 0 or 30"
                  value={
                    params.indicatorValue !== undefined &&
                    params.indicatorValue !== null
                      ? params.indicatorValue
                      : ""
                  }
                  onChange={(e) =>
                    setParams({
                      ...params,
                      indicatorValue: e.target.value
                        ? Number(e.target.value)
                        : undefined,
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
      <Button
        type="button"
        onClick={handleAdd}
        size="sm"
        disabled={
          category === "indicator" &&
          !params.indicatorName?.trim()
        }
      >
        Add condition
      </Button>
    </div>
  );
}

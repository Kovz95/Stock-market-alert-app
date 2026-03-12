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
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
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
  { value: "pivot_sr_any_signal", label: "Any signal" },
  { value: "pivot_sr_near_support", label: "Near support" },
  { value: "pivot_sr_near_resistance", label: "Near resistance" },
  { value: "pivot_sr_near_any", label: "Near any level" },
  { value: "pivot_sr_crossover_bullish", label: "Bullish crossover" },
  { value: "pivot_sr_crossover_bearish", label: "Bearish crossover" },
  { value: "pivot_sr_any_crossover", label: "Any crossover" },
  { value: "pivot_sr_broke_strong_resistance", label: "Broke strong resistance (3+)" },
  { value: "pivot_sr_broke_strong_support", label: "Broke strong support (3+)" },
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
const TREND_MAGIC_TYPES: { value: TrendMagicConditionType; label: string; group: string }[] = [
  // Trend Direction
  { value: "tm_bullish", label: "Bullish (CCI >= 0)", group: "Trend Direction" },
  { value: "tm_bearish", label: "Bearish (CCI < 0)", group: "Trend Direction" },
  // Price vs Trend Magic
  { value: "tm_price_above", label: "Price above Trend Magic", group: "Price vs Trend Magic" },
  { value: "tm_price_below", label: "Price below Trend Magic", group: "Price vs Trend Magic" },
  { value: "tm_price_crossed", label: "Price crossed Trend Magic", group: "Price vs Trend Magic" },
  // Trend Crossover
  { value: "tm_buy_signal", label: "Buy signal (Low crosses above)", group: "Trend Crossover" },
  { value: "tm_sell_signal", label: "Sell signal (High crosses below)", group: "Trend Crossover" },
  { value: "tm_any_cross", label: "Any cross", group: "Trend Crossover" },
];
const SUPERTREND_TYPES: { value: SupertrendConditionType; label: string; group: string }[] = [
  // Trend Direction
  { value: "st_uptrend", label: "Uptrend", group: "Trend Direction" },
  { value: "st_downtrend", label: "Downtrend", group: "Trend Direction" },
  // Price vs SuperTrend
  { value: "st_price_above", label: "Price above SuperTrend", group: "Price vs SuperTrend" },
  { value: "st_price_below", label: "Price below SuperTrend", group: "Price vs SuperTrend" },
  // Trend Change
  { value: "st_changed_uptrend", label: "Changed to Uptrend (Buy)", group: "Trend Change" },
  { value: "st_changed_downtrend", label: "Changed to Downtrend (Sell)", group: "Trend Change" },
  { value: "st_any_change", label: "Any change", group: "Trend Change" },
];
const SAR_TYPES: { value: SARConditionType; label: string }[] = [
  { value: "sar_value", label: "SAR value" },
  { value: "sar_price_above", label: "Price above SAR" },
  { value: "sar_price_below", label: "Price below SAR" },
  { value: "sar_cross_above", label: "Price crossed above SAR" },
  { value: "sar_cross_below", label: "Price crossed below SAR" },
];
const OBV_MACD_TYPES: { value: OBVMACDConditionType; label: string; group: string }[] = [
  // Value
  { value: "obv_macd_value", label: "Raw value", group: "Value" },
  { value: "obv_macd_positive", label: "OBV MACD > 0", group: "Value" },
  { value: "obv_macd_negative", label: "OBV MACD < 0", group: "Value" },
  // Signal Direction
  { value: "obv_macd_signal_bullish", label: "Signal bullish", group: "Signal Direction" },
  { value: "obv_macd_signal_bearish", label: "Signal bearish", group: "Signal Direction" },
];
const HARSI_TYPES: { value: HARSIConditionType; label: string; group: string }[] = [
  // Value
  { value: "harsi_value", label: "Raw value", group: "Value" },
  { value: "harsi_bullish", label: "HARSI > 0 (Bullish)", group: "Value" },
  { value: "harsi_bearish", label: "HARSI < 0 (Bearish)", group: "Value" },
  // Flip
  { value: "harsi_flip_buy", label: "Red to Green (Buy)", group: "Flip" },
  { value: "harsi_flip_sell", label: "Green to Red (Sell)", group: "Flip" },
  { value: "harsi_flip_any", label: "Any flip", group: "Flip" },
];
const MA_ZSCORE_TYPES: { value: MAZScoreConditionType; label: string }[] = [
  { value: "ma_zscore_compare", label: "Z-Score vs threshold" },
  { value: "ma_zscore_value", label: "Raw value" },
];
const EWO_TYPES: { value: EWOConditionType; label: string; group: string }[] = [
  // Levels
  { value: "ewo_above_zero", label: "Above zero", group: "Levels" },
  { value: "ewo_below_zero", label: "Below zero", group: "Levels" },
  { value: "ewo_cross_above_zero", label: "Crossover above zero", group: "Levels" },
  { value: "ewo_cross_below_zero", label: "Crossover below zero", group: "Levels" },
  // Value
  { value: "ewo_compare", label: "Custom comparison", group: "Value" },
  { value: "ewo_value", label: "Raw value", group: "Value" },
];
const ROC_TYPES: { value: ROCConditionType; label: string; group: string }[] = [
  // Levels
  { value: "roc_above_zero", label: "Above zero", group: "Levels" },
  { value: "roc_below_zero", label: "Below zero", group: "Levels" },
  { value: "roc_cross_above_zero", label: "Crossover above zero", group: "Levels" },
  { value: "roc_cross_below_zero", label: "Crossover below zero", group: "Levels" },
  // Value
  { value: "roc_compare", label: "Custom comparison", group: "Value" },
  { value: "roc_value", label: "Raw value", group: "Value" },
];
const WILLR_TYPES: { value: WillRConditionType; label: string }[] = [
  { value: "willr_oversold", label: "Oversold (< -80)" },
  { value: "willr_overbought", label: "Overbought (> -20)" },
];
const CCI_TYPES: { value: CCIConditionType; label: string }[] = [
  { value: "cci_compare", label: "CCI vs level" },
  { value: "cci_value", label: "Raw value" },
];
const ATR_TYPES: { value: ATRConditionType; label: string }[] = [
  { value: "atr_compare", label: "ATR vs value" },
  { value: "atr_value", label: "Raw value" },
];
const KALMAN_ROC_STOCH_TYPES: { value: KalmanROCConditionType; label: string; group: string }[] = [
  // Direction
  { value: "krs_uptrend", label: "Uptrend (White)", group: "Direction" },
  { value: "krs_downtrend", label: "Downtrend (Blue)", group: "Direction" },
  // Crossovers
  { value: "krs_cross_bullish", label: "Bullish crossover (Buy)", group: "Crossovers" },
  { value: "krs_cross_bearish", label: "Bearish crossunder (Sell)", group: "Crossovers" },
  { value: "krs_cross_any", label: "Any cross", group: "Crossovers" },
  // Levels
  { value: "krs_above_60", label: "Above 60 (Overbought)", group: "Levels" },
  { value: "krs_below_10", label: "Below 10 (Oversold)", group: "Levels" },
  { value: "krs_above_50", label: "Above 50", group: "Levels" },
  { value: "krs_below_50", label: "Below 50", group: "Levels" },
  { value: "krs_between_10_60", label: "Between 10 and 60", group: "Levels" },
  // Value
  { value: "krs_value", label: "Raw value", group: "Value" },
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
    p.pivotProximity = p.pivotProximity ?? 1.0;
    p.pivotBuffer = p.pivotBuffer ?? 0.5;
  }
  if (category === "trend_magic") {
    p.tmCciPeriod = p.tmCciPeriod ?? 20;
    p.tmAtrMult = p.tmAtrMult ?? 1.0;
    p.tmAtrPeriod = p.tmAtrPeriod ?? 5;
  }
  if (category === "supertrend") {
    p.supertrendPeriod = p.supertrendPeriod ?? 10;
    p.supertrendMultiplier = p.supertrendMultiplier ?? 3;
    p.supertrendUseHl2 = p.supertrendUseHl2 ?? true;
    p.supertrendUseAtr = p.supertrendUseAtr ?? true;
  }
  if (category === "obv_macd") {
    p.obvWindowLen = p.obvWindowLen ?? 28;
    p.obvVLen = p.obvVLen ?? 14;
    p.obvLen = p.obvLen ?? 1;
    p.obvMaType = p.obvMaType ?? "DEMA";
    p.obvMaLen = p.obvMaLen ?? 9;
    p.obvSlowLen = p.obvSlowLen ?? 26;
    p.obvSlopeLen = p.obvSlopeLen ?? 2;
    p.obvP = p.obvP ?? 1.0;
  }
  if (category === "sar") {
    p.sarAcceleration = p.sarAcceleration ?? 0.02;
    p.sarMaxAcceleration = p.sarMaxAcceleration ?? 0.2;
  }
  if (category === "harsi") {
    p.harsiPeriod = p.harsiPeriod ?? 14;
    p.harsiSmoothing = p.harsiSmoothing ?? 1;
  }
  if (category === "ma_zscore") {
    p.maZScoreThreshold = p.maZScoreThreshold ?? 2;
    p.maZScoreMaLength = p.maZScoreMaLength ?? 20;
    p.maZScoreMaType = p.maZScoreMaType ?? "SMA";
    p.maZScoreMeanWindow = p.maZScoreMeanWindow ?? (p.maZScoreMaLength ?? 20);
    p.maZScoreStdWindow = p.maZScoreStdWindow ?? (p.maZScoreMeanWindow ?? p.maZScoreMaLength ?? 20);
    p.maZScorePriceCol = p.maZScorePriceCol ?? "Close";
    p.maZScoreUsePercent = p.maZScoreUsePercent ?? true;
    p.maZScoreOperator = p.maZScoreOperator ?? ">";
  }
  if (category === "ewo") {
    p.ewoSma1Length = p.ewoSma1Length ?? 5;
    p.ewoSma2Length = p.ewoSma2Length ?? 35;
    p.ewoSource = p.ewoSource ?? "Close";
    p.ewoUsePercent = p.ewoUsePercent ?? true;
    p.ewoOperator = p.ewoOperator ?? ">";
    p.ewoValue = p.ewoValue ?? 0;
  }
  if (category === "roc") {
    p.rocPeriod = p.rocPeriod ?? 12;
    p.rocLevel = p.rocLevel ?? 0;
    p.rocOperator = p.rocOperator ?? ">";
  }
  if (category === "willr") p.willrPeriod = p.willrPeriod ?? 14;
  if (category === "cci") {
    p.cciPeriod = p.cciPeriod ?? 20;
    p.cciOperator = p.cciOperator ?? ">";
    p.cciLevel = p.cciLevel ?? 100;
  }
  if (category === "atr") {
    p.atrPeriod = p.atrPeriod ?? 14;
    p.atrValue = p.atrValue ?? 2;
    p.atrOperator = p.atrOperator ?? ">";
  }
  if (category === "kalman_roc_stoch") {
    p.krsMaType = p.krsMaType ?? "TEMA";
    p.krsSmoothLen = p.krsSmoothLen ?? 12;
    p.krsLsmaOff = p.krsLsmaOff ?? 0;
    p.krsKalSrc = p.krsKalSrc ?? "Close";
    p.krsSharp = p.krsSharp ?? 25.0;
    p.krsKPeriod = p.krsKPeriod ?? 1.0;
    p.krsRocLen = p.krsRocLen ?? 9;
    p.krsStochLen = p.krsStochLen ?? 14;
    p.krsSmoothK = p.krsSmoothK ?? 1;
    p.krsSmoothD = p.krsSmoothD ?? 3;
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
                else if (v === "pivot_sr") setType("pivot_sr_any_signal");
                else if (v === "ichimoku") setType("ichimoku_price_above_cloud");
                else if (v === "trend_magic") setType("tm_bullish");
                else if (v === "supertrend") setType("st_uptrend");
                else if (v === "sar") setType("sar_price_above");
                else if (v === "obv_macd") setType("obv_macd_signal_bullish");
                else if (v === "harsi") setType("harsi_bullish");
                else if (v === "ma_zscore") setType("ma_zscore_compare");
                else if (v === "ewo") setType("ewo_above_zero");
                else if (v === "roc") setType("roc_above_zero");
                else if (v === "willr") setType("willr_oversold");
                else if (v === "cci") setType("cci_compare");
                else if (v === "atr") setType("atr_compare");
                else if (v === "kalman_roc_stoch") setType("krs_uptrend");
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
                    <SelectItem value="FRAMA">FRAMA</SelectItem>
                    <SelectItem value="KAMA">KAMA</SelectItem>
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
            {/* FRAMA-specific params (Task 8) */}
            {(params.maType === "FRAMA") && (
              <>
                <Field>
                  <FieldLabel>FC (fast constant)</FieldLabel>
                  <FieldContent>
                    <Input
                      type="number"
                      min={1}
                      max={300}
                      value={params.framaFc ?? 1}
                      onChange={(e) =>
                        setParams({
                          ...params,
                          framaFc: e.target.value ? Number(e.target.value) : undefined,
                        })
                      }
                    />
                  </FieldContent>
                </Field>
                <Field>
                  <FieldLabel>SC (slow constant)</FieldLabel>
                  <FieldContent>
                    <Input
                      type="number"
                      min={1}
                      max={500}
                      value={params.framaSc ?? 198}
                      onChange={(e) =>
                        setParams({
                          ...params,
                          framaSc: e.target.value ? Number(e.target.value) : undefined,
                        })
                      }
                    />
                  </FieldContent>
                </Field>
              </>
            )}
            {/* KAMA-specific params (Task 8) */}
            {(params.maType === "KAMA") && (
              <>
                <Field>
                  <FieldLabel>Fast end</FieldLabel>
                  <FieldContent>
                    <Input
                      type="number"
                      step="0.001"
                      min={0.001}
                      max={1}
                      value={params.kamaFastEnd ?? 0.666}
                      onChange={(e) =>
                        setParams({
                          ...params,
                          kamaFastEnd: e.target.value ? Number(e.target.value) : undefined,
                        })
                      }
                    />
                  </FieldContent>
                </Field>
                <Field>
                  <FieldLabel>Slow end</FieldLabel>
                  <FieldContent>
                    <Input
                      type="number"
                      step="0.0001"
                      min={0.001}
                      max={1}
                      value={params.kamaSlowEnd ?? 0.0645}
                      onChange={(e) =>
                        setParams({
                          ...params,
                          kamaSlowEnd: e.target.value ? Number(e.target.value) : undefined,
                        })
                      }
                    />
                  </FieldContent>
                </Field>
              </>
            )}
            {/* MA Input Source (Task 7) */}
            <Field>
              <FieldLabel>Input source</FieldLabel>
              <FieldContent>
                <Select
                  value={params.maInputSource ?? "Close"}
                  onValueChange={(v) => setParams({ ...params, maInputSource: v })}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Close">Close</SelectItem>
                    <SelectItem value="Open">Open</SelectItem>
                    <SelectItem value="High">High</SelectItem>
                    <SelectItem value="Low">Low</SelectItem>
                    <SelectItem value="RSI">RSI</SelectItem>
                    <SelectItem value="EWO">EWO</SelectItem>
                    <SelectItem value="MACD_Line">MACD (Line)</SelectItem>
                    <SelectItem value="MACD_Signal">MACD (Signal)</SelectItem>
                    <SelectItem value="MACD_Histogram">MACD (Histogram)</SelectItem>
                  </SelectContent>
                </Select>
              </FieldContent>
            </Field>
            {/* RSI input params */}
            {params.maInputSource === "RSI" && (
              <Field>
                <FieldLabel>RSI period</FieldLabel>
                <FieldContent>
                  <Input
                    type="number"
                    min={2}
                    max={100}
                    value={params.maInputRsiPeriod ?? 14}
                    onChange={(e) =>
                      setParams({
                        ...params,
                        maInputRsiPeriod: e.target.value ? Number(e.target.value) : undefined,
                      })
                    }
                  />
                </FieldContent>
              </Field>
            )}
            {/* EWO input params */}
            {params.maInputSource === "EWO" && (
              <>
                <Field>
                  <FieldLabel>EWO fast SMA</FieldLabel>
                  <FieldContent>
                    <Input
                      type="number"
                      min={1}
                      max={100}
                      value={params.maInputEwoSma1 ?? 5}
                      onChange={(e) =>
                        setParams({
                          ...params,
                          maInputEwoSma1: e.target.value ? Number(e.target.value) : undefined,
                        })
                      }
                    />
                  </FieldContent>
                </Field>
                <Field>
                  <FieldLabel>EWO slow SMA</FieldLabel>
                  <FieldContent>
                    <Input
                      type="number"
                      min={1}
                      max={200}
                      value={params.maInputEwoSma2 ?? 35}
                      onChange={(e) =>
                        setParams({
                          ...params,
                          maInputEwoSma2: e.target.value ? Number(e.target.value) : undefined,
                        })
                      }
                    />
                  </FieldContent>
                </Field>
              </>
            )}
            {/* MACD input params */}
            {(params.maInputSource === "MACD_Line" ||
              params.maInputSource === "MACD_Signal" ||
              params.maInputSource === "MACD_Histogram") && (
              <>
                <Field>
                  <FieldLabel>MACD fast</FieldLabel>
                  <FieldContent>
                    <Input
                      type="number"
                      min={1}
                      max={100}
                      value={params.maInputMacdFast ?? 12}
                      onChange={(e) =>
                        setParams({
                          ...params,
                          maInputMacdFast: e.target.value ? Number(e.target.value) : undefined,
                        })
                      }
                    />
                  </FieldContent>
                </Field>
                <Field>
                  <FieldLabel>MACD slow</FieldLabel>
                  <FieldContent>
                    <Input
                      type="number"
                      min={1}
                      max={200}
                      value={params.maInputMacdSlow ?? 26}
                      onChange={(e) =>
                        setParams({
                          ...params,
                          maInputMacdSlow: e.target.value ? Number(e.target.value) : undefined,
                        })
                      }
                    />
                  </FieldContent>
                </Field>
                <Field>
                  <FieldLabel>MACD signal</FieldLabel>
                  <FieldContent>
                    <Input
                      type="number"
                      min={1}
                      max={100}
                      value={params.maInputMacdSignal ?? 9}
                      onChange={(e) =>
                        setParams({
                          ...params,
                          maInputMacdSignal: e.target.value ? Number(e.target.value) : undefined,
                        })
                      }
                    />
                  </FieldContent>
                </Field>
              </>
            )}
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
                  min={2}
                  max={120}
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
                  min={2}
                  max={120}
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
            <Field>
              <FieldLabel>Proximity threshold (%)</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  step="0.1"
                  min={0.1}
                  max={5}
                  value={params.pivotProximity ?? 1.0}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      pivotProximity: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>Buffer between levels (%)</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  step="0.1"
                  min={0.1}
                  max={5}
                  value={params.pivotBuffer ?? 0.5}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      pivotBuffer: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
          </>
        )}

        {category === "trend_magic" && (
          <>
            <Field>
              <FieldLabel>CCI period</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={5}
                  max={100}
                  value={params.tmCciPeriod ?? 20}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      tmCciPeriod: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>ATR multiplier</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  step="0.1"
                  min={0.1}
                  max={5}
                  value={params.tmAtrMult ?? 1.0}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      tmAtrMult: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>ATR period</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  max={50}
                  value={params.tmAtrPeriod ?? 5}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      tmAtrPeriod: e.target.value ? Number(e.target.value) : undefined,
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
              <FieldLabel>ATR period</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  max={100}
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
              <FieldLabel>ATR multiplier</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  step="0.1"
                  min={0.1}
                  max={10}
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
            <Field>
              <FieldContent>
                <div className="flex items-center gap-2">
                  <Checkbox
                    id="st-use-hl2"
                    checked={params.supertrendUseHl2 ?? true}
                    onCheckedChange={(checked) =>
                      setParams({ ...params, supertrendUseHl2: !!checked })
                    }
                  />
                  <Label htmlFor="st-use-hl2">Use (H+L)/2 as source</Label>
                </div>
              </FieldContent>
            </Field>
            <Field>
              <FieldContent>
                <div className="flex items-center gap-2">
                  <Checkbox
                    id="st-use-atr"
                    checked={params.supertrendUseAtr ?? true}
                    onCheckedChange={(checked) =>
                      setParams({ ...params, supertrendUseAtr: !!checked })
                    }
                  />
                  <Label htmlFor="st-use-atr">Use built-in ATR</Label>
                </div>
              </FieldContent>
            </Field>
          </>
        )}

        {category === "obv_macd" && (
          <>
            <Field>
              <FieldLabel>Window length</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={5}
                  max={100}
                  value={params.obvWindowLen ?? 28}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      obvWindowLen: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>Volume smoothing</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={5}
                  max={50}
                  value={params.obvVLen ?? 14}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      obvVLen: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>OBV EMA length</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  max={50}
                  value={params.obvLen ?? 1}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      obvLen: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>MA type</FieldLabel>
              <FieldContent>
                <Select
                  value={params.obvMaType ?? "DEMA"}
                  onValueChange={(v) => setParams({ ...params, obvMaType: v })}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {["DEMA", "EMA", "TEMA", "TDEMA", "TTEMA", "AVG", "THMA", "ZLEMA", "ZLDEMA", "ZLTEMA", "DZLEMA", "TZLEMA", "LLEMA", "NMA"].map((ma) => (
                      <SelectItem key={ma} value={ma}>{ma}</SelectItem>
                    ))}
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
                  max={100}
                  value={params.obvMaLen ?? 9}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      obvMaLen: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>MACD slow length</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={10}
                  max={100}
                  value={params.obvSlowLen ?? 26}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      obvSlowLen: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>Slope length</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  max={20}
                  value={params.obvSlopeLen ?? 2}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      obvSlopeLen: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            {(type === "obv_macd_signal_bullish" || type === "obv_macd_signal_bearish") && (
              <Field>
                <FieldLabel>Channel sensitivity (p)</FieldLabel>
                <FieldContent>
                  <Input
                    type="number"
                    step="0.1"
                    min={0.1}
                    max={10}
                    value={params.obvP ?? 1.0}
                    onChange={(e) =>
                      setParams({
                        ...params,
                        obvP: e.target.value ? Number(e.target.value) : undefined,
                      })
                    }
                  />
                </FieldContent>
              </Field>
            )}
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
          <>
            <Field>
              <FieldLabel>Period</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={5}
                  max={100}
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
            <Field>
              <FieldLabel>Smoothing</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  max={20}
                  value={params.harsiSmoothing ?? 1}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      harsiSmoothing: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
          </>
        )}

        {category === "ma_zscore" && (
          <>
            <Field>
              <FieldLabel>MA type</FieldLabel>
              <FieldContent>
                <Select
                  value={params.maZScoreMaType ?? "SMA"}
                  onValueChange={(v) => setParams({ ...params, maZScoreMaType: v })}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {["SMA", "EMA", "HMA"].map((ma) => (
                      <SelectItem key={ma} value={ma}>{ma}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>MA period</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  max={500}
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
            <Field>
              <FieldLabel>Spread mean window</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  max={500}
                  value={params.maZScoreMeanWindow ?? (params.maZScoreMaLength ?? 20)}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      maZScoreMeanWindow: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>Spread std dev window</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  max={500}
                  value={params.maZScoreStdWindow ?? (params.maZScoreMeanWindow ?? params.maZScoreMaLength ?? 20)}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      maZScoreStdWindow: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>Price column</FieldLabel>
              <FieldContent>
                <Select
                  value={params.maZScorePriceCol ?? "Close"}
                  onValueChange={(v) => setParams({ ...params, maZScorePriceCol: v })}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {["Close", "Open", "High", "Low"].map((col) => (
                      <SelectItem key={col} value={col}>{col}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </FieldContent>
            </Field>
            <Field>
              <FieldContent>
                <div className="flex items-center gap-2">
                  <Checkbox
                    id="zs-use-percent"
                    checked={params.maZScoreUsePercent ?? true}
                    onCheckedChange={(checked) =>
                      setParams({ ...params, maZScoreUsePercent: !!checked })
                    }
                  />
                  <Label htmlFor="zs-use-percent">Use percent spread</Label>
                </div>
              </FieldContent>
            </Field>
            {type === "ma_zscore_compare" && (
              <>
                <Field>
                  <FieldLabel>Condition</FieldLabel>
                  <FieldContent>
                    <Select
                      value={params.maZScoreOperator ?? ">"}
                      onValueChange={(v) => setParams({ ...params, maZScoreOperator: v })}
                    >
                      <SelectTrigger className="w-full">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value=">">{">"} (above)</SelectItem>
                        <SelectItem value=">=">{">="} (at or above)</SelectItem>
                        <SelectItem value="<">{"<"} (below)</SelectItem>
                        <SelectItem value="<=">{"<="} (at or below)</SelectItem>
                      </SelectContent>
                    </Select>
                  </FieldContent>
                </Field>
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
              </>
            )}
          </>
        )}

        {category === "ewo" && (
          <>
            <Field>
              <FieldLabel>Fast SMA period</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  max={100}
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
              <FieldLabel>Slow SMA period</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  max={200}
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
            <Field>
              <FieldLabel>Source</FieldLabel>
              <FieldContent>
                <Select
                  value={params.ewoSource ?? "Close"}
                  onValueChange={(v) => setParams({ ...params, ewoSource: v })}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {["Close", "Open", "High", "Low"].map((src) => (
                      <SelectItem key={src} value={src}>{src}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </FieldContent>
            </Field>
            <Field>
              <FieldContent>
                <div className="flex items-center gap-2">
                  <Checkbox
                    id="ewo-use-percent"
                    checked={params.ewoUsePercent ?? true}
                    onCheckedChange={(checked) =>
                      setParams({ ...params, ewoUsePercent: !!checked })
                    }
                  />
                  <Label htmlFor="ewo-use-percent">Show as percentage of price</Label>
                </div>
              </FieldContent>
            </Field>
            {type === "ewo_compare" && (
              <>
                <Field>
                  <FieldLabel>Operator</FieldLabel>
                  <FieldContent>
                    <Select
                      value={params.ewoOperator ?? ">"}
                      onValueChange={(v) => setParams({ ...params, ewoOperator: v })}
                    >
                      <SelectTrigger className="w-full">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value=">">{">"} (above)</SelectItem>
                        <SelectItem value="<">{"<"} (below)</SelectItem>
                        <SelectItem value=">=">{">="} (at or above)</SelectItem>
                        <SelectItem value="<=">{"<="} (at or below)</SelectItem>
                        <SelectItem value="==">{"=="} (equal)</SelectItem>
                      </SelectContent>
                    </Select>
                  </FieldContent>
                </Field>
                <Field>
                  <FieldLabel>Value</FieldLabel>
                  <FieldContent>
                    <Input
                      type="number"
                      step="0.1"
                      value={params.ewoValue ?? 0}
                      onChange={(e) =>
                        setParams({
                          ...params,
                          ewoValue: e.target.value ? Number(e.target.value) : undefined,
                        })
                      }
                    />
                  </FieldContent>
                </Field>
              </>
            )}
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
                  max={100}
                  value={params.rocPeriod ?? 12}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      rocPeriod: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            {type === "roc_compare" && (
              <>
                <Field>
                  <FieldLabel>Operator</FieldLabel>
                  <FieldContent>
                    <Select
                      value={params.rocOperator ?? ">"}
                      onValueChange={(v) => setParams({ ...params, rocOperator: v })}
                    >
                      <SelectTrigger className="w-full">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value=">">{">"} (above)</SelectItem>
                        <SelectItem value="<">{"<"} (below)</SelectItem>
                        <SelectItem value=">=">{">="} (at or above)</SelectItem>
                        <SelectItem value="<=">{"<="} (at or below)</SelectItem>
                      </SelectContent>
                    </Select>
                  </FieldContent>
                </Field>
                <Field>
                  <FieldLabel>Value</FieldLabel>
                  <FieldContent>
                    <Input
                      type="number"
                      step="0.1"
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
          <>
            <Field>
              <FieldLabel>Period</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={5}
                  max={100}
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
            {type === "cci_compare" && (
              <>
                <Field>
                  <FieldLabel>Operator</FieldLabel>
                  <FieldContent>
                    <Select
                      value={params.cciOperator ?? ">"}
                      onValueChange={(v) => setParams({ ...params, cciOperator: v })}
                    >
                      <SelectTrigger className="w-full">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value=">">{">"} (above)</SelectItem>
                        <SelectItem value="<">{"<"} (below)</SelectItem>
                        <SelectItem value=">=">{">="} (at or above)</SelectItem>
                        <SelectItem value="<=">{"<="} (at or below)</SelectItem>
                      </SelectContent>
                    </Select>
                  </FieldContent>
                </Field>
                <Field>
                  <FieldLabel>Level</FieldLabel>
                  <FieldContent>
                    <Input
                      type="number"
                      step="1"
                      value={params.cciLevel ?? 100}
                      onChange={(e) =>
                        setParams({
                          ...params,
                          cciLevel: e.target.value ? Number(e.target.value) : undefined,
                        })
                      }
                    />
                  </FieldContent>
                </Field>
              </>
            )}
          </>
        )}

        {category === "atr" && (
          <>
            <Field>
              <FieldLabel>Period</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  max={100}
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
            {type === "atr_compare" && (
              <>
                <Field>
                  <FieldLabel>Operator</FieldLabel>
                  <FieldContent>
                    <Select
                      value={params.atrOperator ?? ">"}
                      onValueChange={(v) => setParams({ ...params, atrOperator: v })}
                    >
                      <SelectTrigger className="w-full">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value=">">{">"} (above)</SelectItem>
                        <SelectItem value="<">{"<"} (below)</SelectItem>
                        <SelectItem value=">=">{">="} (at or above)</SelectItem>
                        <SelectItem value="<=">{"<="} (at or below)</SelectItem>
                      </SelectContent>
                    </Select>
                  </FieldContent>
                </Field>
                <Field>
                  <FieldLabel>Value</FieldLabel>
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
          </>
        )}

        {category === "kalman_roc_stoch" && (
          <>
            <Field>
              <FieldLabel>MA type</FieldLabel>
              <FieldContent>
                <Select
                  value={params.krsMaType ?? "TEMA"}
                  onValueChange={(v) => setParams({ ...params, krsMaType: v })}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {["TEMA", "EMA", "DEMA", "WMA", "VWMA", "SMA", "SMMA", "HMA", "LSMA", "PEMA"].map((ma) => (
                      <SelectItem key={ma} value={ma}>{ma}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>Smoothing length</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  max={50}
                  value={params.krsSmoothLen ?? 12}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      krsSmoothLen: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            {(params.krsMaType ?? "TEMA") === "LSMA" && (
              <Field>
                <FieldLabel>LSMA offset</FieldLabel>
                <FieldContent>
                  <Input
                    type="number"
                    min={0}
                    max={10}
                    value={params.krsLsmaOff ?? 0}
                    onChange={(e) =>
                      setParams({
                        ...params,
                        krsLsmaOff: e.target.value ? Number(e.target.value) : undefined,
                      })
                    }
                  />
                </FieldContent>
              </Field>
            )}
            <Field>
              <FieldLabel>Kalman source</FieldLabel>
              <FieldContent>
                <Select
                  value={params.krsKalSrc ?? "Close"}
                  onValueChange={(v) => setParams({ ...params, krsKalSrc: v })}
                >
                  <SelectTrigger className="w-full">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {["Close", "Open", "High", "Low", "HL2", "HLC3", "OHLC4"].map((src) => (
                      <SelectItem key={src} value={src}>{src}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>Sharpness</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  step="0.1"
                  min={1}
                  max={100}
                  value={params.krsSharp ?? 25.0}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      krsSharp: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>Filter period</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  step="0.1"
                  min={0.1}
                  max={10}
                  value={params.krsKPeriod ?? 1.0}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      krsKPeriod: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>ROC length</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  max={50}
                  value={params.krsRocLen ?? 9}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      krsRocLen: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>Stoch %K length</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  max={50}
                  value={params.krsStochLen ?? 14}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      krsStochLen: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>%K smooth</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  max={20}
                  value={params.krsSmoothK ?? 1}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      krsSmoothK: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            </Field>
            <Field>
              <FieldLabel>%D smooth</FieldLabel>
              <FieldContent>
                <Input
                  type="number"
                  min={1}
                  max={20}
                  value={params.krsSmoothD ?? 3}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      krsSmoothD: e.target.value ? Number(e.target.value) : undefined,
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

        {/* Z-Score transformation toggle (Task 5) */}
        {category !== "price" && category !== "custom" && (
          <Field>
            <FieldContent>
              <div className="flex items-center gap-2">
                <Checkbox
                  id="use-zscore"
                  checked={params.useZScore ?? false}
                  onCheckedChange={(checked) =>
                    setParams({ ...params, useZScore: !!checked })
                  }
                />
                <Label htmlFor="use-zscore">Transform to Z-score (rolling)</Label>
              </div>
            </FieldContent>
            {params.useZScore && (
              <FieldContent className="mt-2">
                <FieldLabel>Lookback period</FieldLabel>
                <Input
                  type="number"
                  min={5}
                  max={500}
                  value={params.zScoreLookback ?? 20}
                  onChange={(e) =>
                    setParams({
                      ...params,
                      zScoreLookback: e.target.value ? Number(e.target.value) : undefined,
                    })
                  }
                />
              </FieldContent>
            )}
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

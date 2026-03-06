/**
 * Form state and condition types for the Add Alert flow.
 * Maps to CreateAlertRequest and backend conditions format.
 */

export type ConditionCategory =
  | "price"
  | "moving_average"
  | "rsi"
  | "macd"
  | "bollinger"
  | "volume"
  | "ma_slope_curve"
  | "indicator"
  | "custom";

export type PriceConditionType = "price_above" | "price_below" | "price_equals";
export type MAConditionType =
  | "price_above_ma"
  | "price_below_ma"
  | "ma_crossover";
export type RSIConditionType = "rsi_oversold" | "rsi_overbought" | "rsi_level";
export type MACDConditionType =
  | "macd_bullish_crossover"
  | "macd_bearish_crossover"
  | "macd_histogram_positive";
export type BBConditionType =
  | "price_above_upper_band"
  | "price_below_lower_band";
export type VolumeConditionType =
  | "volume_above_average"
  | "volume_spike";

export type MASlopeCurveConditionType =
  | "slope_positive"
  | "slope_negative"
  | "slope_turn_up"
  | "slope_turn_dn"
  | "curve_positive"
  | "curve_negative"
  | "bend_up"
  | "bend_dn"
  | "early_bend_up"
  | "early_bend_dn";

export type ConditionParams = {
  // Price
  priceValue?: number;
  // MA (moving_average: SMA etc.; ma_slope_curve: HMA, EMA, SMA, WMA, RMA)
  maType?: string;
  maPeriod?: number;
  fastPeriod?: number;
  slowPeriod?: number;
  // RSI
  rsiPeriod?: number;
  rsiLevel?: number;
  rsiLevelOperator?: string; // ">", "<", ">=", "<=" for rsi_level
  oversoldLevel?: number;
  overboughtLevel?: number;
  // MACD
  macdFast?: number;
  macdSlow?: number;
  macdSignal?: number;
  // Bollinger
  bbPeriod?: number;
  bbStd?: number;
  // Volume
  volumeMultiplier?: number;
  volumeFraction?: number;
  // MA Slope + Curvature
  maLen?: number;
  slopeLookback?: number;
  smoothType?: string; // None, EMA, SMA, RMA
  smoothLen?: number;
  normMode?: string; // None, ATR, Percent
  atrLen?: number;
  slopeThr?: number;
  curveThr?: number;
  // Indicator (any registered indicator: name, optional params, operator, value)
  indicatorName?: string;
  indicatorParams?: string;
  indicatorOperator?: string;
  indicatorValue?: number;
  // Custom
  customExpression?: string;
};

export interface ConditionEntry {
  id: string;
  category: ConditionCategory;
  type: string;
  params: ConditionParams;
}

/**
 * Serialize a condition entry to the backend condition string format
 * (e.g. "price_above: 150", "rsi(14)[-1] < 30").
 */
export function conditionEntryToExpression(entry: ConditionEntry): string {
  const { category, type, params } = entry;
  switch (category) {
    case "price":
      if (type === "price_above" && params.priceValue != null)
        return `price_above: ${params.priceValue}`;
      if (type === "price_below" && params.priceValue != null)
        return `price_below: ${params.priceValue}`;
      if (type === "price_equals" && params.priceValue != null)
        return `price_equals: ${params.priceValue}`;
      break;
    case "moving_average":
      if (type === "price_above_ma" && params.maPeriod != null)
        return `price_above_ma: ${params.maPeriod} (${params.maType ?? "SMA"})`;
      if (type === "price_below_ma" && params.maPeriod != null)
        return `price_below_ma: ${params.maPeriod} (${params.maType ?? "SMA"})`;
      if (
        type === "ma_crossover" &&
        params.fastPeriod != null &&
        params.slowPeriod != null
      )
        return `ma_crossover: ${params.fastPeriod} > ${params.slowPeriod}`;
      break;
    case "rsi":
      if (type === "rsi_oversold" && params.rsiPeriod != null)
        return `rsi(${params.rsiPeriod})[-1] < ${params.oversoldLevel ?? 30}`;
      if (type === "rsi_overbought" && params.rsiPeriod != null)
        return `rsi(${params.rsiPeriod})[-1] > ${params.overboughtLevel ?? 70}`;
      if (type === "rsi_level" && params.rsiPeriod != null && params.rsiLevel != null)
        return `rsi(${params.rsiPeriod})[-1] ${params.rsiLevelOperator ?? ">"} ${params.rsiLevel}`;
      break;
    case "macd":
      if (type === "macd_bullish_crossover")
        return "macd_bullish_crossover";
      if (type === "macd_bearish_crossover")
        return "macd_bearish_crossover";
      if (type === "macd_histogram_positive")
        return "macd_histogram_positive";
      break;
    case "bollinger":
      if (type === "price_above_upper_band" && params.bbPeriod != null && params.bbStd != null)
        return `close[-1] > bbands(${params.bbPeriod}, ${params.bbStd}, type='upper')[-1]`;
      if (type === "price_below_lower_band" && params.bbPeriod != null && params.bbStd != null)
        return `close[-1] < bbands(${params.bbPeriod}, ${params.bbStd}, type='lower')[-1]`;
      break;
    case "volume":
      if (type === "volume_above_average" && params.volumeMultiplier != null)
        return `volume_above_average: ${params.volumeMultiplier}x`;
      if (type === "volume_spike" && params.volumeMultiplier != null)
        return `volume_spike: ${params.volumeMultiplier}x`;
      break;
    case "indicator": {
      const name = params.indicatorName?.trim();
      const op = params.indicatorOperator ?? ">";
      const val = params.indicatorValue ?? 0;
      if (!name) return "";
      const paramStr = params.indicatorParams?.trim();
      const args = paramStr ? paramStr : "";
      const expr = args ? `${name}(${args})[-1] ${op} ${val}` : `${name}()[-1] ${op} ${val}`;
      return expr;
    }
    case "ma_slope_curve": {
      const maLen = params.maLen ?? 200;
      const slopeLookback = params.slopeLookback ?? 3;
      const maType = params.maType ?? "HMA";
      const smoothType = params.smoothType ?? "EMA";
      const smoothLen = params.smoothLen ?? 2;
      const normMode = params.normMode ?? "ATR";
      const atrLen = params.atrLen ?? 14;
      const slopeThr = params.slopeThr ?? 0;
      const curveThr = params.curveThr ?? 0;
      const baseParams = `ma_len=${maLen}, slope_lookback=${slopeLookback}, ma_type='${maType}', smooth_type='${smoothType}', smooth_len=${smoothLen}, norm_mode='${normMode}', atr_len=${atrLen}, slope_thr=${slopeThr}, curve_thr=${curveThr}`;
      const shortParams = `ma_len=${maLen}, slope_lookback=${slopeLookback}, ma_type='${maType}'`;
      switch (type) {
        case "slope_positive":
          return `ma_slope_curve_slope(${baseParams})[-1] > 0`;
        case "slope_negative":
          return `ma_slope_curve_slope(${baseParams})[-1] < 0`;
        case "slope_turn_up":
          return `ma_slope_curve_turn_up(${shortParams})[-1] == 1`;
        case "slope_turn_dn":
          return `ma_slope_curve_turn_dn(${shortParams})[-1] == 1`;
        case "curve_positive":
          return `ma_slope_curve_curve(${baseParams})[-1] > 0`;
        case "curve_negative":
          return `ma_slope_curve_curve(${baseParams})[-1] < 0`;
        case "bend_up":
          return `ma_slope_curve_bend_up(${shortParams})[-1] == 1`;
        case "bend_dn":
          return `ma_slope_curve_bend_dn(${shortParams})[-1] == 1`;
        case "early_bend_up":
          return `ma_slope_curve_early_up(${shortParams})[-1] == 1`;
        case "early_bend_dn":
          return `ma_slope_curve_early_dn(${shortParams})[-1] == 1`;
        default:
          return "";
      }
    }
    case "custom":
      if (params.customExpression) return params.customExpression;
      break;
  }
  return "";
}

/**
 * Build backend conditions struct: { condition_1: { conditions: string[], combination_logic } }
 */
export function buildConditionsStruct(
  entries: ConditionEntry[],
  combinationLogic: "AND" | "OR"
): { [key: string]: { conditions: string[]; combination_logic: string } } {
  const expressions = entries
    .map(conditionEntryToExpression)
    .filter((s) => s.length > 0);
  if (expressions.length === 0) return {};
  return {
    condition_1: {
      conditions: expressions,
      combination_logic: combinationLogic,
    },
  };
}

/**
 * Human-readable label for a condition entry (for ConditionRow display).
 */
export function conditionEntryLabel(entry: ConditionEntry): string {
  const expr = conditionEntryToExpression(entry);
  if (expr) return expr;
  if (entry.category === "rsi" && entry.type === "rsi_level")
    return "RSI at level (incomplete)";
  if (entry.category === "bollinger")
    return `Bollinger ${entry.type === "price_above_upper_band" ? "upper" : "lower"} band (incomplete)`;
  if (entry.category === "ma_slope_curve")
    return `MA Slope+Curvature: ${entry.type}`;
  if (entry.category === "indicator") {
    const expr = conditionEntryToExpression(entry);
    return expr || "Indicator (incomplete)";
  }
  return `${entry.category} – ${entry.type}`;
}

export interface AddAlertFormState {
  name: string;
  action: "Buy" | "Sell";
  isRatio: boolean;
  ticker: string;
  stockName: string;
  exchanges: string[]; // Changed from single exchange to array
  country: string;
  ticker1: string;
  ticker2: string;
  stockName1: string;
  stockName2: string;
  adjustmentMethod: string;
  conditions: ConditionEntry[];
  combinationLogic: "AND" | "OR";
  timeframe: string;
}

/** One item for bulk create: same conditions, different ticker. */
export interface BulkTickerItem {
  ticker: string;
  stockName: string;
}

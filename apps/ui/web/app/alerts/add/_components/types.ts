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
  | "donchian"
  | "pivot_sr"
  | "ichimoku"
  | "trend_magic"
  | "supertrend"
  | "sar"
  | "obv_macd"
  | "harsi"
  | "ma_zscore"
  | "ewo"
  | "roc"
  | "willr"
  | "cci"
  | "atr"
  | "kalman_roc_stoch"
  | "slow_stoch"
  | "indicator"
  | "custom";

export type PriceConditionType = "price_above" | "price_below" | "price_equals";
export type MAConditionType =
  | "price_above_ma"
  | "price_below_ma"
  | "price_cross_above_ma"
  | "price_cross_below_ma"
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

export type DonchianConditionType =
  // Channel Lines
  | "donchian_upper_value"
  | "donchian_lower_value"
  | "donchian_basis_value"
  | "donchian_price_vs_upper"
  | "donchian_price_vs_lower"
  | "donchian_price_vs_basis"
  // Channel Breakout
  | "donchian_breakout_upper"
  | "donchian_breakout_lower"
  | "donchian_basis_cross_up"
  | "donchian_basis_cross_down"
  // Channel Position
  | "donchian_position_value"
  | "donchian_position_above"
  | "donchian_position_below"
  | "donchian_position_near_middle"
  // Channel Width
  | "donchian_width_value"
  | "donchian_width_expanding"
  | "donchian_width_contracting";
export type PivotSRConditionType =
  | "pivot_sr_any_signal"
  | "pivot_sr_near_support"
  | "pivot_sr_near_resistance"
  | "pivot_sr_near_any"
  | "pivot_sr_crossover_bullish"
  | "pivot_sr_crossover_bearish"
  | "pivot_sr_any_crossover"
  | "pivot_sr_broke_strong_support"
  | "pivot_sr_broke_strong_resistance";
export type IchimokuConditionType =
  // Price vs Cloud
  | "ichimoku_price_above_cloud"
  | "ichimoku_price_below_cloud"
  | "ichimoku_price_in_cloud"
  | "ichimoku_price_entered_cloud_above"
  | "ichimoku_price_entered_cloud_below"
  | "ichimoku_price_entered_cloud_any"
  | "ichimoku_price_crossed_above_cloud"
  | "ichimoku_price_crossed_below_cloud"
  // Line Crossovers
  | "ichimoku_tk_cross_bull"
  | "ichimoku_tk_cross_bear"
  | "ichimoku_price_cross_above_conversion"
  | "ichimoku_price_cross_below_conversion"
  | "ichimoku_price_cross_above_base"
  | "ichimoku_price_cross_below_base"
  // Cloud Color
  | "ichimoku_cloud_bullish"
  | "ichimoku_cloud_bearish"
  | "ichimoku_cloud_turned_bullish"
  | "ichimoku_cloud_turned_bearish"
  // Individual Lines
  | "ichimoku_price_above_conversion"
  | "ichimoku_price_below_conversion"
  | "ichimoku_price_above_base"
  | "ichimoku_price_below_base"
  | "ichimoku_conversion_above_base"
  | "ichimoku_conversion_below_base"
  // Lagging Span
  | "ichimoku_lagging_above_price"
  | "ichimoku_lagging_below_price"
  | "ichimoku_lagging_crossed_above"
  | "ichimoku_lagging_crossed_below";
export type TrendMagicConditionType =
  // Trend Direction
  | "tm_bullish"
  | "tm_bearish"
  // Price vs Trend Magic
  | "tm_price_above"
  | "tm_price_below"
  | "tm_price_crossed"
  // Trend Crossover
  | "tm_buy_signal"
  | "tm_sell_signal"
  | "tm_any_cross";
export type SupertrendConditionType =
  // Trend Direction
  | "st_uptrend"
  | "st_downtrend"
  // Price vs SuperTrend
  | "st_price_above"
  | "st_price_below"
  // Trend Change
  | "st_changed_uptrend"
  | "st_changed_downtrend"
  | "st_any_change";
export type SARConditionType =
  | "sar_value"
  | "sar_price_above"
  | "sar_price_below"
  | "sar_cross_above"
  | "sar_cross_below";
export type OBVMACDConditionType =
  // Value
  | "obv_macd_value"
  | "obv_macd_positive"
  | "obv_macd_negative"
  // Signal Direction
  | "obv_macd_signal_bullish"
  | "obv_macd_signal_bearish";
export type HARSIConditionType =
  // HARSI Value
  | "harsi_value"
  | "harsi_bullish"
  | "harsi_bearish"
  // HARSI Flip
  | "harsi_flip_buy"
  | "harsi_flip_sell"
  | "harsi_flip_any";
export type MAZScoreConditionType =
  | "ma_zscore_compare"
  | "ma_zscore_value";
export type EWOConditionType =
  // Levels
  | "ewo_above_zero"
  | "ewo_below_zero"
  | "ewo_cross_above_zero"
  | "ewo_cross_below_zero"
  // Value
  | "ewo_compare"
  | "ewo_value";
export type ROCConditionType =
  // Levels
  | "roc_above_zero"
  | "roc_below_zero"
  | "roc_cross_above_zero"
  | "roc_cross_below_zero"
  // Value
  | "roc_compare"
  | "roc_value";
export type WillRConditionType = "willr_oversold" | "willr_overbought";
export type CCIConditionType =
  | "cci_compare"
  | "cci_value";
export type ATRConditionType =
  | "atr_compare"
  | "atr_value";
export type SlowStochConditionType =
  // K line levels
  | "slow_stoch_k_oversold"
  | "slow_stoch_k_overbought"
  // D line levels
  | "slow_stoch_d_oversold"
  | "slow_stoch_d_overbought"
  // K/D crossovers
  | "slow_stoch_k_cross_above_d"
  | "slow_stoch_k_cross_below_d"
  // Custom comparisons
  | "slow_stoch_k_compare"
  | "slow_stoch_d_compare";

export type KalmanROCConditionType =
  // Direction
  | "krs_uptrend"
  | "krs_downtrend"
  // Crossovers
  | "krs_cross_bullish"
  | "krs_cross_bearish"
  | "krs_cross_any"
  // Levels
  | "krs_above_60"
  | "krs_below_10"
  | "krs_above_50"
  | "krs_below_50"
  | "krs_between_10_60"
  // Value
  | "krs_value";

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
  // Trend Magic
  tmCciPeriod?: number;
  tmAtrMult?: number;
  tmAtrPeriod?: number;
  // Donchian
  donchianLength?: number;
  donchianOffset?: number;
  // Ichimoku
  ichConversion?: number;
  ichBase?: number;
  ichSpanB?: number;
  ichDisplacement?: number;
  // Pivot S/R
  pivotLeftBars?: number;
  pivotRightBars?: number;
  pivotProximity?: number;
  pivotBuffer?: number;
  // Supertrend
  supertrendPeriod?: number;
  supertrendMultiplier?: number;
  supertrendUseHl2?: boolean;
  supertrendUseAtr?: boolean;
  // OBV MACD
  obvWindowLen?: number;
  obvVLen?: number;
  obvLen?: number;
  obvMaType?: string;
  obvMaLen?: number;
  obvSlowLen?: number;
  obvSlopeLen?: number;
  obvP?: number;
  // SAR
  sarAcceleration?: number;
  sarMaxAcceleration?: number;
  // HARSI
  harsiPeriod?: number;
  harsiSmoothing?: number;
  // MA Z-Score
  maZScoreThreshold?: number;
  maZScoreMaLength?: number;
  maZScoreMaType?: string;
  maZScoreMeanWindow?: number;
  maZScoreStdWindow?: number;
  maZScorePriceCol?: string;
  maZScoreUsePercent?: boolean;
  maZScoreOperator?: string;
  // EWO
  ewoSma1Length?: number;
  ewoSma2Length?: number;
  ewoSource?: string;
  ewoUsePercent?: boolean;
  ewoOperator?: string;
  ewoValue?: number;
  // ROC / Williams %R / CCI / ATR
  rocPeriod?: number;
  rocLevel?: number;
  rocOperator?: string;
  willrPeriod?: number;
  cciPeriod?: number;
  cciOperator?: string;
  cciLevel?: number;
  atrPeriod?: number;
  atrValue?: number;
  atrOperator?: string;
  // Slow Stochastic
  slowStochSmoothK?: number;
  slowStochSmoothD?: number;
  slowStochOperator?: string;
  slowStochLevel?: number;
  // Kalman ROC Stoch
  krsMaType?: string;
  krsSmoothLen?: number;
  krsLsmaOff?: number;
  krsKalSrc?: string;
  krsSharp?: number;
  krsKPeriod?: number;
  krsRocLen?: number;
  krsStochLen?: number;
  krsSmoothK?: number;
  krsSmoothD?: number;
  // Z-Score transformation (Task 5)
  useZScore?: boolean;
  zScoreLookback?: number;
  // MA input source (Task 7)
  maInputSource?: string; // "Close" | "Open" | "High" | "Low" | "EWO" | "RSI" | "MACD_Line" | "MACD_Signal" | "MACD_Histogram"
  maInputRsiPeriod?: number;
  maInputEwoSma1?: number;
  maInputEwoSma2?: number;
  maInputMacdFast?: number;
  maInputMacdSlow?: number;
  maInputMacdSignal?: number;
  // FRAMA params (Task 8)
  framaFc?: number;
  framaSc?: number;
  // KAMA params (Task 8)
  kamaFastEnd?: number;
  kamaSlowEnd?: number;
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
/**
 * Build the inner expression for MA input source (Task 7).
 * Returns empty string for standard price columns (Close, Open, etc.).
 */
function buildMaInputExpr(source: string, params: ConditionParams): string {
  switch (source) {
    case "RSI":
      return `rsi(${params.maInputRsiPeriod ?? 14})`;
    case "EWO":
      return `ewo(sma1_length=${params.maInputEwoSma1 ?? 5}, sma2_length=${params.maInputEwoSma2 ?? 35})`;
    case "MACD_Line":
      return `macd_line(fast=${params.maInputMacdFast ?? 12}, slow=${params.maInputMacdSlow ?? 26}, signal=${params.maInputMacdSignal ?? 9})`;
    case "MACD_Signal":
      return `macd_signal(fast=${params.maInputMacdFast ?? 12}, slow=${params.maInputMacdSlow ?? 26}, signal=${params.maInputMacdSignal ?? 9})`;
    case "MACD_Histogram":
      return `macd_histogram(fast=${params.maInputMacdFast ?? 12}, slow=${params.maInputMacdSlow ?? 26}, signal=${params.maInputMacdSignal ?? 9})`;
    default:
      // Close, Open, High, Low - handled by default input param
      return "";
  }
}

/**
 * Wrap a simple indicator expression with zscore() if the z-score flag is set.
 * Skips wrapping for boolean expressions (containing "and", "or", "==", "!=").
 */
function maybeWrapZScore(expr: string, params: ConditionParams): string {
  if (!params.useZScore || !expr) return expr;
  const lookback = params.zScoreLookback ?? 20;
  // Don't wrap boolean / comparison / multi-part expressions
  if (/\b(and|or)\b/i.test(expr) || /[!=]=/.test(expr)) return expr;
  // Match pattern like `indicator(...)[-1] > value`
  const m = expr.match(/^(.+?\))\[-1\]\s*([><=!]+)\s*(.+)$/);
  if (m) {
    return `zscore(${m[1]}, lookback=${lookback})[-1] ${m[2]} ${m[3]}`;
  }
  // Match pattern like `indicator(...)[-1]` (value only)
  const m2 = expr.match(/^(.+?\))\[-1\]$/);
  if (m2) {
    return `zscore(${m2[1]}, lookback=${lookback})[-1]`;
  }
  return expr;
}

export function conditionEntryToExpression(entry: ConditionEntry): string {
  const raw = conditionEntryToExpressionRaw(entry);
  return maybeWrapZScore(raw, entry.params);
}

function conditionEntryToExpressionRaw(entry: ConditionEntry): string {
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
    case "moving_average": {
      const maType = params.maType ?? "SMA";
      const period = params.maPeriod ?? 20;
      // Build input source string for MA (Task 7)
      const inputSrc = params.maInputSource ?? "Close";
      const inputExpr = buildMaInputExpr(inputSrc, params);
      if (type === "price_above_ma" && params.maPeriod != null) {
        if (inputExpr) {
          return `${maType.toLowerCase()}(period=${period}, input=${inputExpr})[-1] < Close[-1]`;
        }
        // FRAMA/KAMA special params (Task 8)
        if (maType === "FRAMA") {
          const fc = params.framaFc ?? 1;
          const sc = params.framaSc ?? 198;
          return `FRAMA(df, length=${period}, FC=${fc}, SC=${sc})[-1] < Close[-1]`;
        }
        if (maType === "KAMA") {
          const fe = params.kamaFastEnd ?? 0.666;
          const se = params.kamaSlowEnd ?? 0.0645;
          return `KAMA(df, length=${period}, fast_end=${fe}, slow_end=${se})[-1] < Close[-1]`;
        }
        return `price_above_ma: ${period} (${maType})`;
      }
      if (type === "price_below_ma" && params.maPeriod != null) {
        if (inputExpr) {
          return `${maType.toLowerCase()}(period=${period}, input=${inputExpr})[-1] > Close[-1]`;
        }
        if (maType === "FRAMA") {
          const fc = params.framaFc ?? 1;
          const sc = params.framaSc ?? 198;
          return `FRAMA(df, length=${period}, FC=${fc}, SC=${sc})[-1] > Close[-1]`;
        }
        if (maType === "KAMA") {
          const fe = params.kamaFastEnd ?? 0.666;
          const se = params.kamaSlowEnd ?? 0.0645;
          return `KAMA(df, length=${period}, fast_end=${fe}, slow_end=${se})[-1] > Close[-1]`;
        }
        return `price_below_ma: ${period} (${maType})`;
      }
      if (type === "price_cross_above_ma" && params.maPeriod != null) {
        if (inputExpr) {
          const maFn = `${maType.toLowerCase()}(period=${period}, input=${inputExpr})`;
          return `(${maFn}[-1] < Close[-1]) and (${maFn}[-2] >= Close[-2])`;
        }
        if (maType === "FRAMA") {
          const fc = params.framaFc ?? 1;
          const sc = params.framaSc ?? 198;
          const maFn = `FRAMA(df, length=${period}, FC=${fc}, SC=${sc})`;
          return `(${maFn}[-1] < Close[-1]) and (${maFn}[-2] >= Close[-2])`;
        }
        if (maType === "KAMA") {
          const fe = params.kamaFastEnd ?? 0.666;
          const se = params.kamaSlowEnd ?? 0.0645;
          const maFn = `KAMA(df, length=${period}, fast_end=${fe}, slow_end=${se})`;
          return `(${maFn}[-1] < Close[-1]) and (${maFn}[-2] >= Close[-2])`;
        }
        return `price_cross_above_ma: ${period} (${maType})`;
      }
      if (type === "price_cross_below_ma" && params.maPeriod != null) {
        if (inputExpr) {
          const maFn = `${maType.toLowerCase()}(period=${period}, input=${inputExpr})`;
          return `(${maFn}[-1] > Close[-1]) and (${maFn}[-2] <= Close[-2])`;
        }
        if (maType === "FRAMA") {
          const fc = params.framaFc ?? 1;
          const sc = params.framaSc ?? 198;
          const maFn = `FRAMA(df, length=${period}, FC=${fc}, SC=${sc})`;
          return `(${maFn}[-1] > Close[-1]) and (${maFn}[-2] <= Close[-2])`;
        }
        if (maType === "KAMA") {
          const fe = params.kamaFastEnd ?? 0.666;
          const se = params.kamaSlowEnd ?? 0.0645;
          const maFn = `KAMA(df, length=${period}, fast_end=${fe}, slow_end=${se})`;
          return `(${maFn}[-1] > Close[-1]) and (${maFn}[-2] <= Close[-2])`;
        }
        return `price_cross_below_ma: ${period} (${maType})`;
      }
      if (
        type === "ma_crossover" &&
        params.fastPeriod != null &&
        params.slowPeriod != null
      )
        return `ma_crossover: ${params.fastPeriod} > ${params.slowPeriod}`;
      break;
    }
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
    case "donchian": {
      const len = params.donchianLength ?? 20;
      const off = params.donchianOffset ?? 0;
      const dArgs = off !== 0 ? `${len}, ${off}` : `${len}`;
      switch (type) {
        // Channel Lines
        case "donchian_upper_value":
          return `donchian_upper(${dArgs})[-1]`;
        case "donchian_lower_value":
          return `donchian_lower(${dArgs})[-1]`;
        case "donchian_basis_value":
          return `donchian_basis(${dArgs})[-1]`;
        case "donchian_price_vs_upper":
          return `Close[-1] > donchian_upper(${dArgs})[-1]`;
        case "donchian_price_vs_lower":
          return `Close[-1] < donchian_lower(${dArgs})[-1]`;
        case "donchian_price_vs_basis":
          return `Close[-1] > donchian_basis(${dArgs})[-1]`;
        // Channel Breakout
        case "donchian_breakout_upper":
          return `(Close[-1] > donchian_upper(${dArgs})[-1]) and (Close[-2] <= donchian_upper(${dArgs})[-2])`;
        case "donchian_breakout_lower":
          return `(Close[-1] < donchian_lower(${dArgs})[-1]) and (Close[-2] >= donchian_lower(${dArgs})[-2])`;
        case "donchian_basis_cross_up":
          return `(Close[-1] > donchian_basis(${dArgs})[-1]) and (Close[-2] <= donchian_basis(${dArgs})[-2])`;
        case "donchian_basis_cross_down":
          return `(Close[-1] < donchian_basis(${dArgs})[-1]) and (Close[-2] >= donchian_basis(${dArgs})[-2])`;
        // Channel Position
        case "donchian_position_value":
          return `donchian_position(${dArgs})[-1]`;
        case "donchian_position_above":
          return `donchian_position(${dArgs})[-1] >= 0.8`;
        case "donchian_position_below":
          return `donchian_position(${dArgs})[-1] <= 0.2`;
        case "donchian_position_near_middle":
          return `(donchian_position(${dArgs})[-1] > 0.4) and (donchian_position(${dArgs})[-1] < 0.6)`;
        // Channel Width
        case "donchian_width_value":
          return `donchian_width(${dArgs})[-1]`;
        case "donchian_width_expanding":
          return `donchian_width(${dArgs})[-1] > donchian_width(${dArgs})[-2]`;
        case "donchian_width_contracting":
          return `donchian_width(${dArgs})[-1] < donchian_width(${dArgs})[-2]`;
        default:
          return "";
      }
    }
    case "pivot_sr": {
      const left = params.pivotLeftBars ?? 5;
      const right = params.pivotRightBars ?? 5;
      const prox = params.pivotProximity ?? 1.0;
      const buf = params.pivotBuffer ?? 0.5;
      const pivotArgs = `left_bars=${left}, right_bars=${right}, proximity_threshold=${prox}, buffer_percent=${buf}`;
      const fn = `pivot_sr(${pivotArgs})`;
      switch (type) {
        case "pivot_sr_any_signal":
          return `${fn}[-1] != 0`;
        case "pivot_sr_near_support":
          return `${fn}[-1] == 1`;
        case "pivot_sr_near_resistance":
          return `${fn}[-1] == -1`;
        case "pivot_sr_near_any":
          return `abs(${fn}[-1]) == 1`;
        case "pivot_sr_crossover_bullish":
          return `${fn}[-1] == 2`;
        case "pivot_sr_crossover_bearish":
          return `${fn}[-1] == -2`;
        case "pivot_sr_any_crossover":
          return `abs(${fn}[-1]) == 2`;
        case "pivot_sr_broke_strong_support":
          return `${fn}[-1] == -3`;
        case "pivot_sr_broke_strong_resistance":
          return `${fn}[-1] == 3`;
        default:
          return "";
      }
    }
    case "ichimoku": {
      const conv = params.ichConversion ?? 9;
      const base = params.ichBase ?? 26;
      const spanB = params.ichSpanB ?? 52;
      const disp = params.ichDisplacement ?? 26;
      const cloudArgs = `conversion_periods=${conv}, base_periods=${base}, span_b_periods=${spanB}, displacement=${disp}, visual=True`;
      const cloudTop = `ichimoku_cloud_top(${cloudArgs})`;
      const cloudBot = `ichimoku_cloud_bottom(${cloudArgs})`;
      const cloudSig = `ichimoku_cloud_signal(${cloudArgs})`;
      const convLine = `ichimoku_conversion(periods=${conv})`;
      const baseLine = `ichimoku_base(periods=${base})`;
      const lagging = `ichimoku_lagging(displacement=${disp}, visual=True)`;
      switch (type) {
        // Price vs Cloud
        case "ichimoku_price_above_cloud":
          return `Close[-1] > ${cloudTop}[-1]`;
        case "ichimoku_price_below_cloud":
          return `Close[-1] < ${cloudBot}[-1]`;
        case "ichimoku_price_in_cloud":
          return `(Close[-1] <= ${cloudTop}[-1]) and (Close[-1] >= ${cloudBot}[-1])`;
        case "ichimoku_price_entered_cloud_above":
          return `(Close[-2] > ${cloudTop}[-2]) and (Close[-1] <= ${cloudTop}[-1]) and (Close[-1] >= ${cloudBot}[-1])`;
        case "ichimoku_price_entered_cloud_below":
          return `(Close[-2] < ${cloudBot}[-2]) and (Close[-1] >= ${cloudBot}[-1]) and (Close[-1] <= ${cloudTop}[-1])`;
        case "ichimoku_price_entered_cloud_any":
          return `((Close[-2] > ${cloudTop}[-2]) or (Close[-2] < ${cloudBot}[-2])) and (Close[-1] <= ${cloudTop}[-1]) and (Close[-1] >= ${cloudBot}[-1])`;
        case "ichimoku_price_crossed_above_cloud":
          return `(Close[-1] > ${cloudTop}[-1]) and (Close[-2] <= ${cloudTop}[-2])`;
        case "ichimoku_price_crossed_below_cloud":
          return `(Close[-1] < ${cloudBot}[-1]) and (Close[-2] >= ${cloudBot}[-2])`;
        // Line Crossovers
        case "ichimoku_tk_cross_bull":
          return `(${convLine}[-1] > ${baseLine}[-1]) and (${convLine}[-2] <= ${baseLine}[-2])`;
        case "ichimoku_tk_cross_bear":
          return `(${convLine}[-1] < ${baseLine}[-1]) and (${convLine}[-2] >= ${baseLine}[-2])`;
        case "ichimoku_price_cross_above_conversion":
          return `(Close[-1] > ${convLine}[-1]) and (Close[-2] <= ${convLine}[-2])`;
        case "ichimoku_price_cross_below_conversion":
          return `(Close[-1] < ${convLine}[-1]) and (Close[-2] >= ${convLine}[-2])`;
        case "ichimoku_price_cross_above_base":
          return `(Close[-1] > ${baseLine}[-1]) and (Close[-2] <= ${baseLine}[-2])`;
        case "ichimoku_price_cross_below_base":
          return `(Close[-1] < ${baseLine}[-1]) and (Close[-2] >= ${baseLine}[-2])`;
        // Cloud Color
        case "ichimoku_cloud_bullish":
          return `${cloudSig}[-1] == 1`;
        case "ichimoku_cloud_bearish":
          return `${cloudSig}[-1] == -1`;
        case "ichimoku_cloud_turned_bullish":
          return `(${cloudSig}[-1] == 1) and (${cloudSig}[-2] != 1)`;
        case "ichimoku_cloud_turned_bearish":
          return `(${cloudSig}[-1] == -1) and (${cloudSig}[-2] != -1)`;
        // Individual Lines
        case "ichimoku_price_above_conversion":
          return `Close[-1] > ${convLine}[-1]`;
        case "ichimoku_price_below_conversion":
          return `Close[-1] < ${convLine}[-1]`;
        case "ichimoku_price_above_base":
          return `Close[-1] > ${baseLine}[-1]`;
        case "ichimoku_price_below_base":
          return `Close[-1] < ${baseLine}[-1]`;
        case "ichimoku_conversion_above_base":
          return `${convLine}[-1] > ${baseLine}[-1]`;
        case "ichimoku_conversion_below_base":
          return `${convLine}[-1] < ${baseLine}[-1]`;
        // Lagging Span
        case "ichimoku_lagging_above_price":
          return `${lagging}[-1] > Close[-${disp + 1}]`;
        case "ichimoku_lagging_below_price":
          return `${lagging}[-1] < Close[-${disp + 1}]`;
        case "ichimoku_lagging_crossed_above":
          return `(${lagging}[-1] > Close[-${disp + 1}]) and (${lagging}[-2] <= Close[-${disp + 2}])`;
        case "ichimoku_lagging_crossed_below":
          return `(${lagging}[-1] < Close[-${disp + 1}]) and (${lagging}[-2] >= Close[-${disp + 2}])`;
        default:
          return "";
      }
    }
    case "trend_magic": {
      const cciP = params.tmCciPeriod ?? 20;
      const atrM = params.tmAtrMult ?? 1.0;
      const atrP = params.tmAtrPeriod ?? 5;
      const tmArgs = `cci_period=${cciP}, atr_multiplier=${atrM}, atr_period=${atrP}`;
      const tm = `trend_magic(${tmArgs})`;
      const tmSig = `trend_magic_signal(${tmArgs})`;
      switch (type) {
        // Trend Direction
        case "tm_bullish":
          return `${tmSig}[-1] == 1`;
        case "tm_bearish":
          return `${tmSig}[-1] == -1`;
        // Price vs Trend Magic
        case "tm_price_above":
          return `Close[-1] > ${tm}[-1]`;
        case "tm_price_below":
          return `Close[-1] < ${tm}[-1]`;
        case "tm_price_crossed":
          return `(Close[-1] > ${tm}[-1] and Close[-2] <= ${tm}[-2]) or (Close[-1] < ${tm}[-1] and Close[-2] >= ${tm}[-2])`;
        // Trend Crossover
        case "tm_buy_signal":
          return `Low[-1] > ${tm}[-1] and Low[-2] <= ${tm}[-2]`;
        case "tm_sell_signal":
          return `High[-1] < ${tm}[-1] and High[-2] >= ${tm}[-2]`;
        case "tm_any_cross":
          return `((Close[-1] > ${tm}[-1]) != (Close[-2] > ${tm}[-2]))`;
        default:
          return "";
      }
    }
    case "supertrend": {
      const period = params.supertrendPeriod ?? 10;
      const mult = params.supertrendMultiplier ?? 3;
      const useHl2 = params.supertrendUseHl2 ?? true;
      const useAtr = params.supertrendUseAtr ?? true;
      const stArgs = `period=${period}, multiplier=${mult}, use_hl2=${useHl2 ? "True" : "False"}, use_builtin_atr=${useAtr ? "True" : "False"}`;
      const st = `supertrend(${stArgs})`;
      const stUpper = `supertrend_upper(${stArgs})`;
      const stLower = `supertrend_lower(${stArgs})`;
      switch (type) {
        // Trend Direction
        case "st_uptrend":
          return `${st}[-1] == 1`;
        case "st_downtrend":
          return `${st}[-1] == -1`;
        // Price vs SuperTrend
        case "st_price_above":
          return `Close[-1] > ${stUpper}[-1]`;
        case "st_price_below":
          return `Close[-1] < ${stLower}[-1]`;
        // Trend Change
        case "st_changed_uptrend":
          return `${st}[-1] == 1 and ${st}[-2] == -1`;
        case "st_changed_downtrend":
          return `${st}[-1] == -1 and ${st}[-2] == 1`;
        case "st_any_change":
          return `${st}[-1] != ${st}[-2]`;
        default:
          return "";
      }
    }
    case "sar": {
      const acc = params.sarAcceleration ?? 0.02;
      const maxAcc = params.sarMaxAcceleration ?? 0.2;
      const sarFn = `sar(${acc}, ${maxAcc})`;
      switch (type) {
        case "sar_value":
          return `${sarFn}[-1]`;
        case "sar_price_above":
          return `Close[-1] > ${sarFn}[-1]`;
        case "sar_price_below":
          return `Close[-1] < ${sarFn}[-1]`;
        case "sar_cross_above":
          return `(Close[-1] > ${sarFn}[-1]) and (Close[-2] <= ${sarFn}[-2])`;
        case "sar_cross_below":
          return `(Close[-1] < ${sarFn}[-1]) and (Close[-2] >= ${sarFn}[-2])`;
        default:
          return "";
      }
    }
    case "obv_macd": {
      const wLen = params.obvWindowLen ?? 28;
      const vLen = params.obvVLen ?? 14;
      const oLen = params.obvLen ?? 1;
      const maT = params.obvMaType ?? "DEMA";
      const maL = params.obvMaLen ?? 9;
      const sLen = params.obvSlowLen ?? 26;
      const slLen = params.obvSlopeLen ?? 2;
      const pVal = params.obvP ?? 1.0;
      const obvArgs = `window_len=${wLen}, v_len=${vLen}, obv_len=${oLen}, ma_type='${maT}', ma_len=${maL}, slow_len=${sLen}, slope_len=${slLen}`;
      const obv = `obv_macd(${obvArgs})`;
      const obvSig = `obv_macd_signal(${obvArgs}, p=${pVal})`;
      switch (type) {
        // Value
        case "obv_macd_value":
          return `${obv}[-1]`;
        case "obv_macd_positive":
          return `${obv}[-1] > 0`;
        case "obv_macd_negative":
          return `${obv}[-1] < 0`;
        // Signal Direction
        case "obv_macd_signal_bullish":
          return `${obvSig}[-1] == 1`;
        case "obv_macd_signal_bearish":
          return `${obvSig}[-1] == -1`;
        default:
          return "";
      }
    }
    case "harsi": {
      const period = params.harsiPeriod ?? 14;
      const smooth = params.harsiSmoothing ?? 1;
      const hArgs = `period=${period}, smoothing=${smooth}`;
      const h = `harsi(${hArgs})`;
      const hFlip = `harsi_flip(${hArgs})`;
      switch (type) {
        // Value
        case "harsi_value":
          return `${h}[-1]`;
        case "harsi_bullish":
          return `${h}[-1] > 0`;
        case "harsi_bearish":
          return `${h}[-1] < 0`;
        // Flip
        case "harsi_flip_buy":
          return `${hFlip}[-1] == 2`;
        case "harsi_flip_sell":
          return `${hFlip}[-1] == 1`;
        case "harsi_flip_any":
          return `${hFlip}[-1] > 0`;
        default:
          return "";
      }
    }
    case "ma_zscore": {
      const threshold = params.maZScoreThreshold ?? 2;
      const maLen = params.maZScoreMaLength ?? 20;
      const maT = params.maZScoreMaType ?? "SMA";
      const meanW = params.maZScoreMeanWindow ?? maLen;
      const stdW = params.maZScoreStdWindow ?? meanW;
      const priceCol = params.maZScorePriceCol ?? "Close";
      const usePct = params.maZScoreUsePercent ?? true;
      const op = params.maZScoreOperator ?? ">";
      const zsArgs = `price_col='${priceCol}', ma_length=${maLen}, spread_mean_window=${meanW}, spread_std_window=${stdW}, ma_type='${maT}', use_percent=${usePct ? "True" : "False"}, output='zscore'`;
      const zs = `ma_spread_zscore(${zsArgs})`;
      switch (type) {
        case "ma_zscore_compare":
          return `${zs}[-1] ${op} ${threshold}`;
        case "ma_zscore_value":
          return `${zs}[-1]`;
        default:
          return "";
      }
    }
    case "ewo": {
      const sma1 = params.ewoSma1Length ?? 5;
      const sma2 = params.ewoSma2Length ?? 35;
      const src = params.ewoSource ?? "Close";
      const usePct = params.ewoUsePercent ?? true;
      const ewoArgs = `sma1_length=${sma1}, sma2_length=${sma2}, source='${src}', use_percent=${usePct ? "True" : "False"}`;
      const ewo = `ewo(${ewoArgs})`;
      switch (type) {
        // Levels
        case "ewo_above_zero":
          return `${ewo}[-1] > 0`;
        case "ewo_below_zero":
          return `${ewo}[-1] < 0`;
        case "ewo_cross_above_zero":
          return `(${ewo}[-1] > 0) and (${ewo}[-2] <= 0)`;
        case "ewo_cross_below_zero":
          return `(${ewo}[-1] < 0) and (${ewo}[-2] >= 0)`;
        // Value
        case "ewo_compare": {
          const op = params.ewoOperator ?? ">";
          const val = params.ewoValue ?? 0;
          return `${ewo}[-1] ${op} ${val}`;
        }
        case "ewo_value":
          return `${ewo}[-1]`;
        default:
          return "";
      }
    }
    case "roc": {
      const period = params.rocPeriod ?? 12;
      const r = `roc(${period})`;
      switch (type) {
        // Levels
        case "roc_above_zero":
          return `${r}[-1] > 0`;
        case "roc_below_zero":
          return `${r}[-1] < 0`;
        case "roc_cross_above_zero":
          return `(${r}[-1] > 0) and (${r}[-2] <= 0)`;
        case "roc_cross_below_zero":
          return `(${r}[-1] < 0) and (${r}[-2] >= 0)`;
        // Value
        case "roc_compare": {
          const op = params.rocOperator ?? ">";
          const level = params.rocLevel ?? 0;
          return `${r}[-1] ${op} ${level}`;
        }
        case "roc_value":
          return `${r}[-1]`;
        default:
          return "";
      }
    }
    case "willr": {
      const period = params.willrPeriod ?? 14;
      switch (type) {
        case "willr_oversold":
          return `willr(${period})[-1] < -80`;
        case "willr_overbought":
          return `willr(${period})[-1] > -20`;
        default:
          return "";
      }
    }
    case "cci": {
      const period = params.cciPeriod ?? 20;
      const c = `cci(${period})`;
      switch (type) {
        case "cci_compare": {
          const op = params.cciOperator ?? ">";
          const level = params.cciLevel ?? 100;
          return `${c}[-1] ${op} ${level}`;
        }
        case "cci_value":
          return `${c}[-1]`;
        default:
          return "";
      }
    }
    case "atr": {
      const period = params.atrPeriod ?? 14;
      const a = `atr(${period})`;
      switch (type) {
        case "atr_compare": {
          const op = params.atrOperator ?? ">";
          const val = params.atrValue ?? 2;
          return `${a}[-1] ${op} ${val}`;
        }
        case "atr_value":
          return `${a}[-1]`;
        default:
          return "";
      }
    }
    case "slow_stoch": {
      const smoothK = params.slowStochSmoothK ?? 14;
      const smoothD = params.slowStochSmoothD ?? 3;
      const sArgs = `smooth_k=${smoothK}, smooth_d=${smoothD}`;
      const kFn = `slow_stoch_k(${sArgs})`;
      const dFn = `slow_stoch_d(${sArgs})`;
      switch (type) {
        case "slow_stoch_k_oversold":
          return `${kFn}[-1] < 20`;
        case "slow_stoch_k_overbought":
          return `${kFn}[-1] > 80`;
        case "slow_stoch_d_oversold":
          return `${dFn}[-1] < 20`;
        case "slow_stoch_d_overbought":
          return `${dFn}[-1] > 80`;
        case "slow_stoch_k_cross_above_d":
          return `(${kFn}[-1] > ${dFn}[-1]) and (${kFn}[-2] <= ${dFn}[-2])`;
        case "slow_stoch_k_cross_below_d":
          return `(${kFn}[-1] < ${dFn}[-1]) and (${kFn}[-2] >= ${dFn}[-2])`;
        case "slow_stoch_k_compare": {
          const op = params.slowStochOperator ?? "<";
          const lvl = params.slowStochLevel ?? 20;
          return `${kFn}[-1] ${op} ${lvl}`;
        }
        case "slow_stoch_d_compare": {
          const op = params.slowStochOperator ?? "<";
          const lvl = params.slowStochLevel ?? 20;
          return `${dFn}[-1] ${op} ${lvl}`;
        }
        default:
          return "";
      }
    }
    case "kalman_roc_stoch": {
      const maType = params.krsMaType ?? "TEMA";
      const smoothLen = params.krsSmoothLen ?? 12;
      const lsmaOff = params.krsLsmaOff ?? 0;
      const kalSrc = params.krsKalSrc ?? "Close";
      const sharp = params.krsSharp ?? 25.0;
      const kPeriod = params.krsKPeriod ?? 1.0;
      const rocLen = params.krsRocLen ?? 9;
      const stochLen = params.krsStochLen ?? 14;
      const smoothK = params.krsSmoothK ?? 1;
      const smoothD = params.krsSmoothD ?? 3;
      const krsArgs = `ma_type='${maType}', lsma_off=${lsmaOff}, smooth_len=${smoothLen}, kal_src='${kalSrc}', sharp=${sharp}, k_period=${kPeriod}, roc_len=${rocLen}, stoch_len=${stochLen}, smooth_k=${smoothK}, smooth_d=${smoothD}`;
      const krs = `kalman_roc_stoch(${krsArgs})`;
      const krsSig = `kalman_roc_stoch_signal(${krsArgs})`;
      const krsCross = `kalman_roc_stoch_crossover(${krsArgs})`;
      switch (type) {
        // Direction
        case "krs_uptrend":
          return `${krsSig}[-1] == 1`;
        case "krs_downtrend":
          return `${krsSig}[-1] == -1`;
        // Crossovers
        case "krs_cross_bullish":
          return `${krsCross}[-1] == 1`;
        case "krs_cross_bearish":
          return `${krsCross}[-1] == -1`;
        case "krs_cross_any":
          return `${krsCross}[-1] != 0`;
        // Levels
        case "krs_above_60":
          return `${krs}[-1] > 60`;
        case "krs_below_10":
          return `${krs}[-1] < 10`;
        case "krs_above_50":
          return `${krs}[-1] > 50`;
        case "krs_below_50":
          return `${krs}[-1] < 50`;
        case "krs_between_10_60":
          return `(${krs}[-1] > 10) and (${krs}[-1] < 60)`;
        // Value
        case "krs_value":
          return `${krs}[-1]`;
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
  const presetCategories: ConditionCategory[] = [
    "donchian", "pivot_sr", "ichimoku", "trend_magic", "supertrend", "sar",
    "obv_macd", "harsi", "ma_zscore", "ewo", "roc", "willr", "cci", "atr", "kalman_roc_stoch", "slow_stoch",
  ];
  if (presetCategories.includes(entry.category)) {
    const expr = conditionEntryToExpression(entry);
    return expr || `${entry.category}: ${entry.type}`;
  }
  return `${entry.category} – ${entry.type}`;
}

export interface IndustryFilters {
  economies: string[];
  sectors: string[];
  subsectors: string[];
  industryGroups: string[];
  industries: string[];
  subindustries: string[];
}

export const emptyIndustryFilters: IndustryFilters = {
  economies: [],
  sectors: [],
  subsectors: [],
  industryGroups: [],
  industries: [],
  subindustries: [],
};

export interface EtfFilters {
  etfIssuers: string[];
  assetClasses: string[];
  etfFocuses: string[];
  etfNiches: string[];
}

export const emptyEtfFilters: EtfFilters = {
  etfIssuers: [],
  assetClasses: [],
  etfFocuses: [],
  etfNiches: [],
};

export type AssetType = "All" | "Stocks" | "ETFs";

export interface AddAlertFormState {
  name: string;
  action: "Buy" | "Sell";
  isRatio: boolean;
  ticker: string;
  stockName: string;
  exchanges: string[];
  country: string;
  assetType: AssetType;
  industryFilters: IndustryFilters;
  etfFilters: EtfFilters;
  ticker1: string;
  ticker2: string;
  stockName1: string;
  stockName2: string;
  adjustmentMethod: string;
  conditions: ConditionEntry[];
  combinationLogic: "AND" | "OR";
  timeframe: string;
  // Multi-timeframe (Task 3)
  enableMultiTimeframe: boolean;
  comparisonTimeframe: string;
  // Mixed timeframe (Task 4)
  enableMixedTimeframe: boolean;
}

/** One item for bulk create: same conditions, different ticker. */
export interface BulkTickerItem {
  ticker: string;
  stockName: string;
}

/**
 * Generate an alert name from conditions (Task 6).
 * Extracts indicator names from condition expressions.
 */
export function generateAlertNameFromConditions(
  entries: ConditionEntry[],
  combinationLogic: "AND" | "OR"
): string {
  if (entries.length === 0) return "";
  const names = entries.map((e) => {
    const { category, type, params } = e;
    switch (category) {
      case "price":
        return type.replace(/_/g, " ");
      case "moving_average":
        return `${params.maType ?? "SMA"}(${params.maPeriod ?? 20})`;
      case "rsi":
        return `RSI(${params.rsiPeriod ?? 14})`;
      case "macd":
        return "MACD";
      case "bollinger":
        return "BBands";
      case "volume":
        return "Volume";
      case "indicator":
        return params.indicatorName ?? "indicator";
      case "custom":
        return "Custom";
      default:
        return category.replace(/_/g, " ");
    }
  });
  const joined = names.join(combinationLogic === "AND" ? " + " : " | ");
  return joined.length > 60 ? joined.slice(0, 57) + "..." : joined;
}

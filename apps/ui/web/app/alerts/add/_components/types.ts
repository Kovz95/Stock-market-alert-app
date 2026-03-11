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
  | "pivot_sr_bullish"
  | "pivot_sr_bearish"
  | "pivot_sr_near_support"
  | "pivot_sr_near_resistance"
  | "pivot_sr_crossover_bullish"
  | "pivot_sr_crossover_bearish";
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
export type TrendMagicConditionType = "trend_magic_bullish" | "trend_magic_bearish";
export type SupertrendConditionType = "supertrend_uptrend" | "supertrend_downtrend";
export type SARConditionType =
  | "sar_value"
  | "sar_price_above"
  | "sar_price_below"
  | "sar_cross_above"
  | "sar_cross_below";
export type OBVMACDConditionType = "obv_macd_positive" | "obv_macd_above_signal";
export type HARSIConditionType = "harsi_bullish" | "harsi_bearish";
export type MAZScoreConditionType = "ma_zscore_above" | "ma_zscore_below";
export type EWOConditionType = "ewo_positive" | "ewo_negative";
export type ROCConditionType = "roc_above" | "roc_below";
export type WillRConditionType = "willr_oversold" | "willr_overbought";
export type CCIConditionType = "cci_above" | "cci_below";
export type ATRConditionType = "atr_above" | "atr_below";
export type KalmanROCConditionType =
  | "kalman_roc_stoch_positive"
  | "kalman_roc_stoch_signal_bullish"
  | "kalman_roc_stoch_crossover_bullish";

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
  // Supertrend
  supertrendPeriod?: number;
  supertrendMultiplier?: number;
  // SAR
  sarAcceleration?: number;
  sarMaxAcceleration?: number;
  // HARSI
  harsiPeriod?: number;
  // MA Z-Score
  maZScoreThreshold?: number;
  maZScoreMaLength?: number;
  // EWO
  ewoSma1Length?: number;
  ewoSma2Length?: number;
  // ROC / Williams %R / CCI / ATR
  rocPeriod?: number;
  rocLevel?: number;
  willrPeriod?: number;
  cciPeriod?: number;
  atrPeriod?: number;
  atrValue?: number;
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
      const p = left !== 5 || right !== 5 ? `left_bars=${left}, right_bars=${right}` : "";
      const fn = p ? `pivot_sr(${p})` : "pivot_sr()";
      switch (type) {
        case "pivot_sr_bullish":
          return `${fn}[-1] >= 2`;
        case "pivot_sr_bearish":
          return `${fn}[-1] <= -2`;
        case "pivot_sr_near_support":
          return "pivot_sr_proximity()[-1] == 1";
        case "pivot_sr_near_resistance":
          return "pivot_sr_proximity()[-1] == -1";
        case "pivot_sr_crossover_bullish":
          return "pivot_sr_crossover()[-1] == 1";
        case "pivot_sr_crossover_bearish":
          return "pivot_sr_crossover()[-1] == -1";
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
      switch (type) {
        case "trend_magic_bullish":
          return "trend_magic_signal()[-1] == 1";
        case "trend_magic_bearish":
          return "trend_magic_signal()[-1] == -1";
        default:
          return "";
      }
    }
    case "supertrend": {
      const period = params.supertrendPeriod ?? 10;
      const mult = params.supertrendMultiplier ?? 3;
      switch (type) {
        case "supertrend_uptrend":
          return `supertrend(${period}, ${mult})[-1] == 1`;
        case "supertrend_downtrend":
          return `supertrend(${period}, ${mult})[-1] == -1`;
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
      switch (type) {
        case "obv_macd_positive":
          return "obv_macd()[-1] > 0";
        case "obv_macd_above_signal":
          return "obv_macd()[-1] > obv_macd_signal()[-1]";
        default:
          return "";
      }
    }
    case "harsi": {
      const period = params.harsiPeriod ?? 14;
      switch (type) {
        case "harsi_bullish":
          return `harsi(${period})[-1] > 0`;
        case "harsi_bearish":
          return `harsi(${period})[-1] < 0`;
        default:
          return "";
      }
    }
    case "ma_zscore": {
      const threshold = params.maZScoreThreshold ?? 2;
      const maLen = params.maZScoreMaLength ?? 20;
      switch (type) {
        case "ma_zscore_above":
          return `ma_spread_zscore(ma_length=${maLen})[-1] > ${threshold}`;
        case "ma_zscore_below":
          return `ma_spread_zscore(ma_length=${maLen})[-1] < -${threshold}`;
        default:
          return "";
      }
    }
    case "ewo": {
      const sma1 = params.ewoSma1Length ?? 5;
      const sma2 = params.ewoSma2Length ?? 35;
      switch (type) {
        case "ewo_positive":
          return `EWO(sma1_length=${sma1}, sma2_length=${sma2})[-1] > 0`;
        case "ewo_negative":
          return `EWO(sma1_length=${sma1}, sma2_length=${sma2})[-1] < 0`;
        default:
          return "";
      }
    }
    case "roc": {
      const period = params.rocPeriod ?? 14;
      const level = params.rocLevel ?? 0;
      switch (type) {
        case "roc_above":
          return `roc(${period})[-1] > ${level}`;
        case "roc_below":
          return `roc(${period})[-1] < ${level}`;
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
      switch (type) {
        case "cci_above":
          return `cci(${period})[-1] > 100`;
        case "cci_below":
          return `cci(${period})[-1] < -100`;
        default:
          return "";
      }
    }
    case "atr": {
      const period = params.atrPeriod ?? 14;
      const val = params.atrValue ?? 2;
      switch (type) {
        case "atr_above":
          return `atr(${period})[-1] > ${val}`;
        case "atr_below":
          return `atr(${period})[-1] < ${val}`;
        default:
          return "";
      }
    }
    case "kalman_roc_stoch": {
      switch (type) {
        case "kalman_roc_stoch_positive":
          return "kalman_roc_stoch()[-1] > 0";
        case "kalman_roc_stoch_signal_bullish":
          return "kalman_roc_stoch_signal()[-1] == 1";
        case "kalman_roc_stoch_crossover_bullish":
          return "kalman_roc_stoch_crossover()[-1] == 1";
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
    "obv_macd", "harsi", "ma_zscore", "ewo", "roc", "willr", "cci", "atr", "kalman_roc_stoch",
  ];
  if (presetCategories.includes(entry.category)) {
    const expr = conditionEntryToExpression(entry);
    return expr || `${entry.category}: ${entry.type}`;
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

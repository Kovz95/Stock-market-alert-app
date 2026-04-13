package indicator

import (
	"fmt"
	"strings"
)

// IndicatorFunc computes an indicator from OHLCV data and returns a float64
// series. The params map holds parsed keyword arguments from the alert
// expression (e.g. {"timeperiod": 14, "input": "Close"}).
//
// The returned slice must have the same length as the OHLCV input. Leading
// values that cannot be computed should be math.NaN().
type IndicatorFunc func(data *OHLCV, params map[string]interface{}) ([]float64, error)

// Registry maps lowercase indicator names to their compute functions.
type Registry struct {
	funcs map[string]IndicatorFunc
}

// NewRegistry creates an empty Registry.
func NewRegistry() *Registry {
	return &Registry{funcs: make(map[string]IndicatorFunc)}
}

// Register adds an indicator function under the given name (stored lowercase).
func (r *Registry) Register(name string, fn IndicatorFunc) {
	r.funcs[strings.ToLower(name)] = fn
}

// Get looks up an indicator by name (case-insensitive).
func (r *Registry) Get(name string) (IndicatorFunc, bool) {
	fn, ok := r.funcs[strings.ToLower(name)]
	return fn, ok
}

// MustGet looks up an indicator by name and panics if not found.
func (r *Registry) MustGet(name string) IndicatorFunc {
	fn, ok := r.Get(name)
	if !ok {
		panic(fmt.Sprintf("indicator %q not registered", name))
	}
	return fn
}

// Names returns all registered indicator names (lowercase, sorted).
func (r *Registry) Names() []string {
	names := make([]string, 0, len(r.funcs))
	for k := range r.funcs {
		names = append(names, k)
	}
	return names
}

// NewDefaultRegistry creates a Registry pre-populated with all available indicators.
func NewDefaultRegistry() *Registry {
	r := NewRegistry()

	// Basic indicators (go-talib wrappers)
	r.Register("sma", SMA)
	r.Register("ema", EMA)
	r.Register("rsi", RSI)
	r.Register("volume_ratio", VolumeRatio)
	r.Register("roc", ROC)
	r.Register("atr", ATR)
	r.Register("cci", CCI)
	r.Register("willr", WILLR)

	// Additional TA-Lib (talib_ext.go)
	r.Register("adx", ADX)
	r.Register("stoch_k", StochK)
	r.Register("stoch_d", StochD)
	r.Register("slow_stoch_k", SlowStochK)
	r.Register("slow_stoch_d", SlowStochD)
	r.Register("stoch_rsi_k", StochRsiK)
	r.Register("stoch_rsi_d", StochRsiD)
	r.Register("obv", OBV)
	r.Register("mfi", MFI)
	r.Register("mom", Mom)
	r.Register("ad", Ad)
	// Medium priority: trend / volatility / slope
	r.Register("plus_di", PlusDI)
	r.Register("minus_di", MinusDI)
	r.Register("natr", Natr)
	r.Register("linear_reg_slope", LinearRegSlope)
	r.Register("linear_reg", LinearReg)
	r.Register("stddev", StdDev)
	r.Register("aroon_osc", AroonOsc)
	r.Register("cmo", Cmo)

	// Multi-output indicators
	r.Register("macd", MACD)
	r.Register("macd_line", MACDLine)
	r.Register("macd_signal", MACDSignal)
	r.Register("macd_histogram", MACDHistogram)
	r.Register("bbands", BBANDS)
	r.Register("sar", SAR)

	// Advanced indicators
	r.Register("hma", HMA)
	r.Register("frama", FRAMA)
	r.Register("kama", KAMA)
	r.Register("ewo", EWO)
	r.Register("ma_spread_zscore", MASpreadZscore)
	r.Register("zscore", ZScore)
	r.Register("harsi_flip", HARSIFlip)

	// Slope indicators
	r.Register("slope_sma", SlopeSMA)
	r.Register("slope_ema", SlopeEMA)
	r.Register("slope_hma", SlopeHMA)

	// Supertrend
	r.Register("supertrend", Supertrend)
	r.Register("supertrend_upper", SupertrendUpper)
	r.Register("supertrend_lower", SupertrendLower)

	// Ichimoku
	r.Register("ichimoku_conversion", IchimokuConversion)
	r.Register("ichimoku_base", IchimokuBase)
	r.Register("ichimoku_span_a", IchimokuSpanA)
	r.Register("ichimoku_span_b", IchimokuSpanB)
	r.Register("ichimoku_lagging", IchimokuLagging)
	r.Register("ichimoku_cloud_top", IchimokuCloudTop)
	r.Register("ichimoku_cloud_bottom", IchimokuCloudBottom)
	r.Register("ichimoku_cloud_signal", IchimokuCloudSignal)

	// Donchian
	r.Register("donchian_upper", DonchianUpper)
	r.Register("donchian_lower", DonchianLower)
	r.Register("donchian_basis", DonchianBasis)
	r.Register("donchian_width", DonchianWidth)
	r.Register("donchian_position", DonchianPosition)

	// Trend Magic
	r.Register("trend_magic", TrendMagic)
	r.Register("trend_magic_signal", TrendMagicSignal)

	// Kalman
	r.Register("kalman_roc_stoch", KalmanROCStoch)
	r.Register("kalman_roc_stoch_signal", KalmanROCStochSignal)
	r.Register("kalman_roc_stoch_crossover", KalmanROCStochCrossover)

	// OBV MACD
	r.Register("obv_macd", OBVMACD)
	r.Register("obv_macd_signal", OBVMACDSignal)

	// Pivot S/R
	r.Register("pivot_sr", PivotSR)
	r.Register("pivot_sr_crossover", PivotSRCrossover)
	r.Register("pivot_sr_proximity", PivotSRProximity)

	// Custom (e.g. from TradingView / Pine Script — add more in indicator/custom.go)
	r.Register("my_smoothed_rsi", MySmoothedRSI)

	// MA Slope + Curvature
	r.Register("ma_slope_curve_ma", MaSlopeCurveMA)
	r.Register("ma_slope_curve_slope", MaSlopeCurveSlope)
	r.Register("ma_slope_curve_curve", MaSlopeCurveCurve)
	r.Register("ma_slope_curve_turn_up", MaSlopeCurveTurnUp)
	r.Register("ma_slope_curve_turn_dn", MaSlopeCurveTurnDn)
	r.Register("ma_slope_curve_bend_up", MaSlopeCurveBendUp)
	r.Register("ma_slope_curve_bend_dn", MaSlopeCurveBendDn)
	r.Register("ma_slope_curve_early_up", MaSlopeCurveEarlyUp)
	r.Register("ma_slope_curve_early_dn", MaSlopeCurveEarlyDn)

	return r
}

// --- Parameter helpers ---

// paramInt extracts an int parameter with a default value.
func paramInt(params map[string]interface{}, key string, def int) int {
	if v, ok := params[key]; ok {
		switch val := v.(type) {
		case int:
			return val
		case float64:
			return int(val)
		case int64:
			return int(val)
		}
	}
	return def
}

// paramFloat extracts a float64 parameter with a default value.
func paramFloat(params map[string]interface{}, key string, def float64) float64 {
	if v, ok := params[key]; ok {
		switch val := v.(type) {
		case float64:
			return val
		case int:
			return float64(val)
		case int64:
			return float64(val)
		}
	}
	return def
}

// paramString extracts a string parameter with a default value.
func paramString(params map[string]interface{}, key string, def string) string {
	if v, ok := params[key]; ok {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return def
}

// paramBool extracts a bool parameter with a default value.
func paramBool(params map[string]interface{}, key string, def bool) bool {
	if v, ok := params[key]; ok {
		if b, ok := v.(bool); ok {
			return b
		}
	}
	return def
}

// resolveInput returns the price series to use based on the "input" parameter.
// If a pre-computed series was injected under "_computed_input" (from a nested
// indicator expression like sma(period=20, input=rsi(14))), that is returned first.
// Otherwise "input" is treated as a column name (Close, Open, High, Low, Volume).
func resolveInput(data *OHLCV, params map[string]interface{}) []float64 {
	if computed, ok := params["_computed_input"]; ok {
		if s, ok := computed.([]float64); ok {
			return s
		}
	}
	input := paramString(params, "input", "Close")
	col, err := data.Column(input)
	if err != nil {
		return data.Close
	}
	return col
}

package indicator

import (
	"math"
)

// computeSupertrend is the core Supertrend algorithm returning trend, upper band, lower band.
func computeSupertrend(data *OHLCV, period int, multiplier float64) (trend, up, dn []float64) {
	n := data.Len()
	high := data.High
	low := data.Low
	close := data.Close

	// Source = HL2
	src := make([]float64, n)
	for i := range src {
		src[i] = (high[i] + low[i]) / 2
	}

	// True Range -> ATR (SMA of TR)
	tr := TrueRange(high, low, close)
	atr := RollingMean(tr, period)

	// Basic bands
	up = make([]float64, n)
	dn = make([]float64, n)
	for i := range up {
		up[i] = src[i] - multiplier*atr[i]
		dn[i] = src[i] + multiplier*atr[i]
	}

	// Trend
	trend = make([]float64, n)
	trend[0] = 1

	for i := 1; i < n; i++ {
		// Update upper band (support in uptrend)
		if close[i-1] > up[i-1] && !math.IsNaN(up[i-1]) {
			up[i] = math.Max(up[i], up[i-1])
		}

		// Update lower band (resistance in downtrend)
		if close[i-1] < dn[i-1] && !math.IsNaN(dn[i-1]) {
			dn[i] = math.Min(dn[i], dn[i-1])
		}

		// Update trend direction
		if trend[i-1] == -1 && close[i] > dn[i-1] {
			trend[i] = 1
		} else if trend[i-1] == 1 && close[i] < up[i-1] {
			trend[i] = -1
		} else {
			trend[i] = trend[i-1]
		}
	}

	return trend, up, dn
}

// Supertrend computes Supertrend direction (1 for uptrend, -1 for downtrend).
// Params: period (int, default 10), multiplier (float64, default 3.0).
func Supertrend(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "period", 10)
	multiplier := paramFloat(params, "multiplier", 3.0)
	trend, _, _ := computeSupertrend(data, period, multiplier)
	return trend, nil
}

// SupertrendUpper returns the upper band (support line) only during uptrend, NaN otherwise.
// Params: period (int, default 10), multiplier (float64, default 3.0).
func SupertrendUpper(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "period", 10)
	multiplier := paramFloat(params, "multiplier", 3.0)
	trend, up, _ := computeSupertrend(data, period, multiplier)

	out := make([]float64, len(trend))
	for i := range out {
		if trend[i] == 1 {
			out[i] = up[i]
		} else {
			out[i] = math.NaN()
		}
	}
	return out, nil
}

// SupertrendLower returns the lower band (resistance line) only during downtrend, NaN otherwise.
// Params: period (int, default 10), multiplier (float64, default 3.0).
func SupertrendLower(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "period", 10)
	multiplier := paramFloat(params, "multiplier", 3.0)
	trend, _, dn := computeSupertrend(data, period, multiplier)

	out := make([]float64, len(trend))
	for i := range out {
		if trend[i] == -1 {
			out[i] = dn[i]
		} else {
			out[i] = math.NaN()
		}
	}
	return out, nil
}

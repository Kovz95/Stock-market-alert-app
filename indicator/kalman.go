package indicator

import (
	"math"
	"strings"

	"github.com/markcheno/go-talib"
)

// KalmanROCStoch computes the Kalman Smoothed ROC & Stochastic blended indicator.
// Params: ma_type (string, default "TEMA"), smooth_len (int, default 12),
//
//	kal_src (string, default "Close"), sharp (float64, default 25.0),
//	k_period (float64, default 1.0), roc_len (int, default 9),
//	stoch_len (int, default 14), smooth_k (int, default 1), smooth_d (int, default 3).
func KalmanROCStoch(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	maType := strings.ToUpper(paramString(params, "ma_type", "TEMA"))
	smoothLen := paramInt(params, "smooth_len", 12)
	kalSrc := paramString(params, "kal_src", "Close")
	sharp := paramFloat(params, "sharp", 25.0)
	kPeriod := paramFloat(params, "k_period", 1.0)
	rocLen := paramInt(params, "roc_len", 9)
	stochLen := paramInt(params, "stoch_len", 14)
	smoothK := paramInt(params, "smooth_k", 1)
	smoothD := paramInt(params, "smooth_d", 3)

	src, _ := data.Column(kalSrc)
	if src == nil {
		src = data.Close
	}

	n := data.Len()

	// Kalman Filter
	kfilt := make([]float64, n)
	vel := make([]float64, n)
	kfilt[0] = src[0]

	for i := 1; i < n; i++ {
		dist := src[i] - kfilt[i-1]
		err := kfilt[i-1] + dist*math.Sqrt(sharp*kPeriod/100)
		vel[i] = vel[i-1] + dist*(kPeriod/100)
		kfilt[i] = err + vel[i]
	}

	// ROC on Kalman filtered values
	roc := make([]float64, n)
	for i := 0; i < n; i++ {
		if i < rocLen || kfilt[i-rocLen] == 0 {
			roc[i] = math.NaN()
		} else {
			roc[i] = 100 * (kfilt[i] - kfilt[i-rocLen]) / kfilt[i-rocLen]
		}
	}

	// Stochastic (using go-talib Min/Max/Sma)
	lowestLow := talib.Min(data.Low, stochLen)
	highestHigh := talib.Max(data.High, stochLen)
	kRaw := make([]float64, n)
	for i := range kRaw {
		denom := highestHigh[i] - lowestLow[i]
		if denom != 0 && !math.IsNaN(denom) {
			kRaw[i] = 100 * (data.Close[i] - lowestLow[i]) / denom
		} else {
			kRaw[i] = math.NaN()
		}
	}
	kSma := talib.Sma(kRaw, smoothK)
	dSma := talib.Sma(kSma, smoothD)

	// Blend ROC and Stochastic D
	blendRaw := make([]float64, n)
	for i := range blendRaw {
		blendRaw[i] = (roc[i] + dSma[i]) / 2
	}

	// Apply MA smoothing
	return applyMASmoothing(blendRaw, maType, smoothLen), nil
}

// applyMASmoothing applies the selected MA type to the input series.
func applyMASmoothing(data []float64, maType string, length int) []float64 {
	switch maType {
	case "SMA":
		return talib.Sma(data, length)
	case "EMA":
		return talib.Ema(data, length)
	case "DEMA":
		return dema(data, length)
	case "TEMA":
		return tema(data, length)
	case "WMA":
		return WMA(data, length)
	case "HMA":
		return computeHMA(data, length)
	default:
		return talib.Sma(data, length)
	}
}

// dema computes Double Exponential Moving Average.
func dema(data []float64, length int) []float64 {
	ma1 := talib.Ema(data, length)
	ma2 := talib.Ema(ma1, length)
	out := make([]float64, len(data))
	for i := range out {
		out[i] = 2*ma1[i] - ma2[i]
	}
	return out
}

// tema computes Triple Exponential Moving Average.
func tema(data []float64, length int) []float64 {
	ma1 := talib.Ema(data, length)
	ma2 := talib.Ema(ma1, length)
	ma3 := talib.Ema(ma2, length)
	out := make([]float64, len(data))
	for i := range out {
		out[i] = 3*(ma1[i]-ma2[i]) + ma3[i]
	}
	return out
}

// KalmanROCStochSignal returns direction signal: 1 uptrend, -1 downtrend, 0 neutral.
func KalmanROCStochSignal(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	blend, err := KalmanROCStoch(data, params)
	if err != nil {
		return nil, err
	}

	n := len(blend)
	out := make([]float64, n)
	for i := 1; i < n; i++ {
		if math.IsNaN(blend[i]) || math.IsNaN(blend[i-1]) {
			continue
		}
		if blend[i] > blend[i-1] {
			out[i] = 1
		} else if blend[i] < blend[i-1] {
			out[i] = -1
		}
	}
	return out, nil
}

// KalmanROCStochCrossover returns crossover signals: 1 bullish, -1 bearish, 0 none.
func KalmanROCStochCrossover(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	blend, err := KalmanROCStoch(data, params)
	if err != nil {
		return nil, err
	}

	n := len(blend)
	out := make([]float64, n)
	for i := 2; i < n; i++ {
		if math.IsNaN(blend[i]) || math.IsNaN(blend[i-1]) || math.IsNaN(blend[i-2]) {
			continue
		}
		// Bullish crossover: current > prev AND prev <= prev2
		if blend[i] > blend[i-1] && blend[i-1] <= blend[i-2] {
			out[i] = 1
		}
		// Bearish crossunder: current < prev AND prev >= prev2
		if blend[i] < blend[i-1] && blend[i-1] >= blend[i-2] {
			out[i] = -1
		}
	}
	return out, nil
}

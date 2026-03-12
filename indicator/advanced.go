package indicator

import (
	"math"

	"github.com/markcheno/go-talib"
)

// HMA computes Hull Moving Average.
// Params: timeperiod (int, required), input (string, default "Close").
func HMA(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "timeperiod", 20)
	input := resolveInput(data, params)
	return computeHMA(input, period), nil
}

// computeHMA is the core HMA algorithm, usable by other indicators.
func computeHMA(prices []float64, period int) []float64 {
	halfPeriod := period / 2
	sqrtPeriod := int(math.Sqrt(float64(period)))

	wmaHalf := talib.Wma(prices, halfPeriod)
	wmaFull := talib.Wma(prices, period)

	// 2 * WMA(half) - WMA(full)
	delta := make([]float64, len(prices))
	for i := range delta {
		delta[i] = 2*wmaHalf[i] - wmaFull[i]
	}

	return talib.Wma(delta, sqrtPeriod)
}

// FRAMA computes Fractal Adaptive Moving Average.
// Params: length (int, default 16), FC (int, default 1), SC (int, default 198),
//
//	price_type (string, default "HL2").
func FRAMA(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	length := paramInt(params, "length", 16)
	fc := paramFloat(params, "FC", 1)
	sc := paramFloat(params, "SC", 198)
	priceType := paramString(params, "price_type", "HL2")

	var price []float64
	switch priceType {
	case "HL2":
		price, _ = data.Column("HL2")
	case "Close", "Open", "High", "Low":
		price, _ = data.Column(priceType)
	default:
		price, _ = data.Column("HL2")
	}

	n := data.Len()
	out := make([]float64, n)
	for i := range out {
		out[i] = math.NaN()
	}

	len1 := length / 2
	w := math.Log(2 / (sc + 1))

	for i := length; i < n; i++ {
		// First half period
		h1 := math.Inf(-1)
		l1 := math.Inf(1)
		for j := i - len1; j < i; j++ {
			if data.High[j] > h1 {
				h1 = data.High[j]
			}
			if data.Low[j] < l1 {
				l1 = data.Low[j]
			}
		}
		n1 := 0.0
		if len1 > 0 {
			n1 = (h1 - l1) / float64(len1)
		}

		// Second half period
		h2 := math.Inf(-1)
		l2 := math.Inf(1)
		for j := i - length; j < i-len1; j++ {
			if data.High[j] > h2 {
				h2 = data.High[j]
			}
			if data.Low[j] < l2 {
				l2 = data.Low[j]
			}
		}
		n2 := 0.0
		if len1 > 0 {
			n2 = (h2 - l2) / float64(len1)
		}

		// Full period
		h3 := math.Inf(-1)
		l3 := math.Inf(1)
		for j := i - length; j < i; j++ {
			if data.High[j] > h3 {
				h3 = data.High[j]
			}
			if data.Low[j] < l3 {
				l3 = data.Low[j]
			}
		}
		n3 := 0.0
		if length > 0 {
			n3 = (h3 - l3) / float64(length)
		}

		// Fractal dimension
		var dimen float64
		if n1 > 0 && n2 > 0 && n3 > 0 {
			dimen = (math.Log(n1+n2) - math.Log(n3)) / math.Log(2)
		} else {
			if i > length && !math.IsNaN(out[i-1]) {
				dimen = out[i-1]
			} else {
				dimen = 1
			}
		}

		// Alpha calculation
		alpha1 := math.Exp(w * (dimen - 1))
		oldAlpha := alpha1
		if oldAlpha > 1 {
			oldAlpha = 1
		} else if oldAlpha < 0.01 {
			oldAlpha = 0.01
		}

		oldN := (2 - oldAlpha) / oldAlpha
		nVal := ((sc-fc)*(oldN-1))/(sc-1) + fc
		alpha := 2 / (nVal + 1)
		minAlpha := 2 / (sc + 1)
		if alpha < minAlpha {
			alpha = minAlpha
		} else if alpha > 1 {
			alpha = 1
		}

		if i == length {
			out[i] = price[i]
		} else {
			out[i] = (1-alpha)*out[i-1] + alpha*price[i]
		}
	}

	return out, nil
}

// KAMA computes Kaufman Adaptive Moving Average.
// Params: length (int, default 21), price_type (string, default "Close"),
//
//	fast_end (float64, default 0.666), slow_end (float64, default 0.0645).
func KAMA(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	length := paramInt(params, "length", 21)
	priceType := paramString(params, "price_type", "Close")
	fastEnd := paramFloat(params, "fast_end", 0.666)
	slowEnd := paramFloat(params, "slow_end", 0.0645)

	var price []float64
	switch priceType {
	case "HL2":
		price, _ = data.Column("HL2")
	case "Close", "Open", "High", "Low":
		price, _ = data.Column(priceType)
	default:
		price = data.Close
	}

	n := data.Len()
	kama := make([]float64, n)
	for i := range kama {
		kama[i] = math.NaN()
	}

	for i := length; i < n; i++ {
		signal := math.Abs(price[i] - price[i-length])

		noise := 0.0
		for j := i - length + 1; j <= i; j++ {
			noise += math.Abs(price[j] - price[j-1])
		}

		er := 0.0
		if noise != 0 {
			er = signal / noise
		}

		smooth := math.Pow(er*(fastEnd-slowEnd)+slowEnd, 2)

		if i == length {
			// Initialize with SMA
			sum := 0.0
			for j := i - length + 1; j <= i; j++ {
				sum += price[j]
			}
			kama[i] = sum / float64(length)
		} else {
			kama[i] = kama[i-1] + smooth*(price[i]-kama[i-1])
		}
	}

	return kama, nil
}

// EWO computes Elliott Wave Oscillator.
// Params: sma1_length (int, default 5), sma2_length (int, default 35),
//
//	source (string, default "Close"), use_percent (bool, default true).
func EWO(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	sma1Len := paramInt(params, "sma1_length", 5)
	sma2Len := paramInt(params, "sma2_length", 35)
	source := paramString(params, "source", "Close")
	usePercent := paramBool(params, "use_percent", true)

	src, err := data.Column(source)
	if err != nil {
		src = data.Close
	}

	sma1 := talib.Sma(src, sma1Len)
	sma2 := talib.Sma(src, sma2Len)

	n := len(src)
	out := make([]float64, n)
	for i := range out {
		diff := sma1[i] - sma2[i]
		if usePercent && src[i] != 0 {
			out[i] = (diff / src[i]) * 100
		} else {
			out[i] = diff
		}
	}

	return out, nil
}

// MASpreadZscore computes the z-score of the spread between price and a moving average.
// Params: price_col (string, default "Close"), ma_length (int, default 20),
//
//	spread_mean_window (int), spread_std_window (int), ma_type (string, default "SMA"),
//	use_percent (bool, default false).
//
// Returns the zscore series.
func MASpreadZscore(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	priceCol := paramString(params, "price_col", "Close")
	maLength := paramInt(params, "ma_length", 20)
	maType := paramString(params, "ma_type", "SMA")
	usePercent := paramBool(params, "use_percent", false)
	spreadMeanWindow := paramInt(params, "spread_mean_window", 0)
	spreadStdWindow := paramInt(params, "spread_std_window", 0)

	if spreadMeanWindow == 0 {
		spreadMeanWindow = maLength
	}
	if spreadStdWindow == 0 {
		spreadStdWindow = spreadMeanWindow
	}

	price, err := data.Column(priceCol)
	if err != nil {
		return nil, err
	}

	var ma []float64
	switch maType {
	case "EMA":
		ma = talib.Ema(price, maLength)
	case "HMA":
		ma = computeHMA(price, maLength)
	default:
		ma = talib.Sma(price, maLength)
	}

	n := len(price)
	spread := make([]float64, n)
	for i := range spread {
		if usePercent {
			if ma[i] != 0 {
				spread[i] = (price[i] - ma[i]) / ma[i] * 100
			} else {
				spread[i] = math.NaN()
			}
		} else {
			spread[i] = price[i] - ma[i]
		}
	}

	spreadMean := talib.Sma(spread, spreadMeanWindow)
	spreadStd := talib.StdDev(spread, spreadStdWindow, 1.0)

	zscore := make([]float64, n)
	for i := range zscore {
		if spreadStd[i] != 0 && !math.IsNaN(spreadStd[i]) {
			zscore[i] = (spread[i] - spreadMean[i]) / spreadStd[i]
		} else {
			zscore[i] = math.NaN()
		}
	}

	return zscore, nil
}

// HARSIFlip computes the Heikin-Ashi RSI Flip indicator.
// Returns: 0 = no change, 1 = green to red (sell), 2 = red to green (buy).
// Params: timeperiod (int, default 14), smoothing (int, default 1).
func HARSIFlip(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "timeperiod", 14)
	smoothing := paramInt(params, "smoothing", 1)

	n := data.Len()

	// Calculate RSI for Close, High, Low then subtract 50
	cRSI := talib.Rsi(data.Close, period)
	hRSI := talib.Rsi(data.High, period)
	lRSI := talib.Rsi(data.Low, period)

	for i := range cRSI {
		cRSI[i] -= 50
		hRSI[i] -= 50
		lRSI[i] -= 50
	}

	// oRSI = previous close RSI (shifted)
	oRSI := Shift(cRSI, 1)
	if n > 0 {
		oRSI[0] = cRSI[0]
	}

	// hRSImax = max(hRSI, lRSI), lRSImin = min(hRSI, lRSI)
	hRSImax := make([]float64, n)
	lRSImin := make([]float64, n)
	for i := range hRSImax {
		hRSImax[i] = math.Max(hRSI[i], lRSI[i])
		lRSImin[i] = math.Min(hRSI[i], lRSI[i])
	}

	// HA Close = (oRSI + hRSImax + lRSImin + cRSI) / 4
	closeHA := make([]float64, n)
	for i := range closeHA {
		closeHA[i] = (oRSI[i] + hRSImax[i] + lRSImin[i] + cRSI[i]) / 4
	}

	// HA Open (recursive)
	openHA := make([]float64, n)
	if n > 0 {
		openHA[0] = (oRSI[0] + cRSI[0]) / 2
		sm := float64(smoothing)
		for i := 1; i < n; i++ {
			openHA[i] = (openHA[i-1]*sm + closeHA[i-1]) / (sm + 1)
		}
	}

	// Colors: green if close > open, else red
	colors := make([]int, n) // 0=red, 1=green
	for i := range colors {
		if closeHA[i] > openHA[i] {
			colors[i] = 1
		}
	}

	// Transitions
	out := make([]float64, n)
	for i := 1; i < n; i++ {
		if colors[i-1] == 1 && colors[i] == 0 {
			out[i] = 1 // green to red
		} else if colors[i-1] == 0 && colors[i] == 1 {
			out[i] = 2 // red to green
		}
	}

	return out, nil
}

// ZScore computes a rolling z-score of the input series.
// The input series is typically a nested indicator expression (e.g. zscore(rsi(14), lookback=20))
// which the evaluator resolves into a pre-computed series passed as "_computed_input".
// Falls back to a price column (default "Close") when no nested series is provided.
// Params: lookback (int, default 20).
func ZScore(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	lookback := paramInt(params, "lookback", 20)
	if lookback < 2 {
		lookback = 2
	}
	input := resolveInput(data, params)
	n := len(input)
	out := make([]float64, n)
	for i := range out {
		out[i] = math.NaN()
	}
	for i := lookback - 1; i < n; i++ {
		window := input[i-lookback+1 : i+1]
		// Compute mean
		sum := 0.0
		valid := 0
		for _, v := range window {
			if !math.IsNaN(v) {
				sum += v
				valid++
			}
		}
		if valid < 2 {
			continue
		}
		mean := sum / float64(valid)
		// Compute population std dev
		variance := 0.0
		for _, v := range window {
			if !math.IsNaN(v) {
				d := v - mean
				variance += d * d
			}
		}
		std := math.Sqrt(variance / float64(valid))
		if std == 0 {
			out[i] = 0
		} else {
			out[i] = (input[i] - mean) / std
		}
	}
	return out, nil
}

// SlopeSMA computes the gradient of SMA.
// Params: timeperiod (int, required), input (string, default "Close").
func SlopeSMA(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "timeperiod", 20)
	input := resolveInput(data, params)
	sma := talib.Sma(input, period)
	return Gradient(sma), nil
}

// SlopeEMA computes the gradient of EMA.
// Params: timeperiod (int, required), input (string, default "Close").
func SlopeEMA(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "timeperiod", 20)
	input := resolveInput(data, params)
	ema := talib.Ema(input, period)
	return Gradient(ema), nil
}

// SlopeHMA computes the gradient of HMA.
// Params: timeperiod (int, required), input (string, default "Close").
func SlopeHMA(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "timeperiod", 20)
	input := resolveInput(data, params)
	hma := computeHMA(input, period)
	return Gradient(hma), nil
}

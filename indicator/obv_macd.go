package indicator

import (
	"math"
	"strings"

	"github.com/markcheno/go-talib"
)

// OBVMACD computes OBV-based MACD with slope calculation.
// Params: window_len (int, default 28), v_len (int, default 14), obv_len (int, default 1),
//
//	ma_type (string, default "DEMA"), ma_len (int, default 9),
//	slow_len (int, default 26), slope_len (int, default 2).
func OBVMACD(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	windowLen := paramInt(params, "window_len", 28)
	vLen := paramInt(params, "v_len", 14)
	obvLen := paramInt(params, "obv_len", 1)
	maType := strings.ToUpper(paramString(params, "ma_type", "DEMA"))
	maLen := paramInt(params, "ma_len", 9)
	slowLen := paramInt(params, "slow_len", 26)
	slopeLen := paramInt(params, "slope_len", 2)

	n := data.Len()

	// Calculate OBV
	closeChange := make([]float64, n)
	for i := 1; i < n; i++ {
		closeChange[i] = data.Close[i] - data.Close[i-1]
	}
	sign := make([]float64, n)
	for i := range sign {
		if closeChange[i] > 0 {
			sign[i] = 1
		} else if closeChange[i] < 0 {
			sign[i] = -1
		}
	}

	vol := data.Volume
	if len(vol) == 0 {
		vol = make([]float64, n)
	}

	v := make([]float64, n)
	v[0] = sign[0] * vol[0]
	for i := 1; i < n; i++ {
		v[i] = v[i-1] + sign[i]*vol[i]
	}

	// Price spread std
	hl := make([]float64, n)
	for i := range hl {
		hl[i] = data.High[i] - data.Low[i]
	}
	priceSpread := RollingStd(hl, windowLen)

	// Smooth OBV
	smooth := talib.Sma(v, vLen)

	// Volume spread
	vDiff := make([]float64, n)
	for i := range vDiff {
		vDiff[i] = v[i] - smooth[i]
	}
	vSpread := RollingStd(vDiff, windowLen)

	// Shadow calculation
	shadow := make([]float64, n)
	for i := range shadow {
		if vSpread[i] != 0 && !math.IsNaN(vSpread[i]) {
			shadow[i] = (v[i] - smooth[i]) / vSpread[i] * priceSpread[i]
		}
	}

	// Out = High + shadow (if positive) or Low + shadow
	out := make([]float64, n)
	for i := range out {
		if shadow[i] > 0 {
			out[i] = data.High[i] + shadow[i]
		} else {
			out[i] = data.Low[i] + shadow[i]
		}
	}

	// OBV EMA
	obvEma := talib.Ema(out, obvLen)

	// Apply selected MA type
	var ma []float64
	switch maType {
	case "EMA":
		ma = talib.Ema(obvEma, maLen)
	case "DEMA":
		ma = dema(obvEma, maLen)
	case "TEMA":
		ma = tema(obvEma, maLen)
	case "TDEMA":
		ma = tdema(obvEma, maLen)
	case "TTEMA":
		ma = ttema(obvEma, maLen)
	case "ZLEMA":
		ma = zlema(obvEma, maLen)
	case "ZLDEMA":
		ma = zldema(obvEma, maLen)
	case "ZLTEMA":
		ma = zltema(obvEma, maLen)
	case "DZLEMA":
		ma = dzlema(obvEma, maLen)
	case "TZLEMA":
		ma = tzlema(obvEma, maLen)
	case "LLEMA":
		ma = llema(obvEma, maLen)
	case "NMA":
		ma = nma(obvEma, maLen, 26)
	case "AVG":
		t1 := ttema(obvEma, maLen)
		t2 := tdema(obvEma, maLen)
		ma = make([]float64, n)
		for i := range ma {
			ma[i] = (t1[i] + t2[i]) / 2
		}
	default:
		ma = talib.Ema(obvEma, maLen)
	}

	// MACD = ma - slow EMA of Close
	slowMA := talib.Ema(data.Close, slowLen)
	macd := make([]float64, n)
	for i := range macd {
		macd[i] = ma[i] - slowMA[i]
	}

	// Slope (linear regression)
	return calcSlope(macd, slopeLen), nil
}

// OBV MACD helper MA functions
func tdema(src []float64, length int) []float64 {
	ma1 := dema(src, length)
	ma2 := dema(ma1, length)
	ma3 := dema(ma2, length)
	out := make([]float64, len(src))
	for i := range out {
		out[i] = 3*(ma1[i]-ma2[i]) + ma3[i]
	}
	return out
}

func ttema(src []float64, length int) []float64 {
	ma1 := tema(src, length)
	ma2 := tema(ma1, length)
	ma3 := tema(ma2, length)
	out := make([]float64, len(src))
	for i := range out {
		out[i] = 3*(ma1[i]-ma2[i]) + ma3[i]
	}
	return out
}

func zlema(src []float64, length int) []float64 {
	lag := (length - 1) / 2
	shifted := Shift(src, lag)
	zlsrc := make([]float64, len(src))
	for i := range zlsrc {
		zlsrc[i] = src[i] + (src[i] - shifted[i])
	}
	return talib.Ema(zlsrc, length)
}

func zldema(src []float64, length int) []float64 {
	lag := (length - 1) / 2
	shifted := Shift(src, lag)
	zlsrc := make([]float64, len(src))
	for i := range zlsrc {
		zlsrc[i] = src[i] + (src[i] - shifted[i])
	}
	return dema(zlsrc, length)
}

func zltema(src []float64, length int) []float64 {
	lag := (length - 1) / 2
	shifted := Shift(src, lag)
	zlsrc := make([]float64, len(src))
	for i := range zlsrc {
		zlsrc[i] = src[i] + (src[i] - shifted[i])
	}
	return tema(zlsrc, length)
}

func dzlema(src []float64, length int) []float64 {
	ma1 := zlema(src, length)
	ma2 := zlema(ma1, length)
	out := make([]float64, len(src))
	for i := range out {
		out[i] = 2*ma1[i] - ma2[i]
	}
	return out
}

func tzlema(src []float64, length int) []float64 {
	ma1 := zlema(src, length)
	ma2 := zlema(ma1, length)
	ma3 := zlema(ma2, length)
	out := make([]float64, len(src))
	for i := range out {
		out[i] = 3*(ma1[i]-ma2[i]) + ma3[i]
	}
	return out
}

func llema(src []float64, length int) []float64 {
	n := len(src)
	srcNew := make([]float64, n)
	for i := 2; i < n; i++ {
		srcNew[i] = 0.25*src[i] + 0.5*src[i-1] + 0.25*src[i-2]
	}
	if n > 0 {
		srcNew[0] = src[0]
	}
	if n > 1 {
		srcNew[1] = src[1]
	}
	return talib.Ema(srcNew, length)
}

func nma(src []float64, length1, length2 int) []float64 {
	lambdaVal := float64(length1) / float64(length2)
	alpha := lambdaVal * (float64(length1) - 1) / (float64(length1) - lambdaVal)
	ma1 := talib.Ema(src, length1)
	ma2 := talib.Ema(ma1, length2)
	out := make([]float64, len(src))
	for i := range out {
		out[i] = (1+alpha)*ma1[i] - alpha*ma2[i]
	}
	return out
}

// calcSlope computes linear regression slope line.
func calcSlope(src []float64, length int) []float64 {
	n := len(src)
	out := make([]float64, n)
	for i := range out {
		out[i] = math.NaN()
	}

	for i := length; i < n; i++ {
		sumX := 0.0
		sumY := 0.0
		sumXSqr := 0.0
		sumXY := 0.0
		for j := 0; j < length; j++ {
			x := float64(j + 1)
			y := src[i-length+1+j]
			sumX += x
			sumY += y
			sumXSqr += x * x
			sumXY += x * y
		}
		l := float64(length)
		slopeVal := (l*sumXY - sumX*sumY) / (l*sumXSqr - sumX*sumX)
		average := sumY / l
		intercept := average - slopeVal*sumX/l + slopeVal
		out[i] = intercept + slopeVal*l
	}
	return out
}

// OBVMACDSignal returns trend channel signal: 1 bullish, -1 bearish, 0 neutral.
// Params: same as OBVMACD plus p (float64, default 1).
func OBVMACDSignal(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	src, err := OBVMACD(data, params)
	if err != nil {
		return nil, err
	}

	p := paramFloat(params, "p", 1.0)
	n := len(src)

	b := make([]float64, n)
	dev := make([]float64, n)
	oc := make([]float64, n)

	for i := 1; i < n; i++ {
		if math.IsNaN(src[i]) {
			b[i] = b[i-1]
			dev[i] = dev[i-1]
			oc[i] = oc[i-1]
			continue
		}

		// Calculate adaptive threshold
		sum := 0.0
		count := 0
		for j := 0; j <= i; j++ {
			if !math.IsNaN(src[j]) {
				sum += math.Abs(src[j] - b[i-1])
				count++
			}
		}
		a := 0.0
		if count > 0 {
			a = sum / float64(count) * p
		}

		if src[i] > b[i-1]+a {
			b[i] = src[i]
		} else if src[i] < b[i-1]-a {
			b[i] = src[i]
		} else {
			b[i] = b[i-1]
		}

		if b[i] != b[i-1] {
			dev[i] = a
		} else {
			dev[i] = dev[i-1]
		}

		if b[i] > b[i-1] {
			oc[i] = 1
		} else if b[i] < b[i-1] {
			oc[i] = -1
		} else {
			oc[i] = oc[i-1]
		}
	}

	return oc, nil
}

package indicator

import (
	"math"

	"github.com/markcheno/go-talib"
)

// maSlopeCurveParams holds shared parameters for all ma_slope_curve_* indicators.
type maSlopeCurveParams struct {
	maLen        int
	maType       string
	slopeLB      int
	smoothType   string
	smoothLen    int
	normMode     string
	atrLen       int
	slopeThr     float64
	curveThr     float64
}

func parseMaSlopeCurveParams(params map[string]interface{}) maSlopeCurveParams {
	return maSlopeCurveParams{
		maLen:      paramInt(params, "ma_len", paramInt(params, "timeperiod", 200)),
		maType:     paramString(params, "ma_type", "HMA"),
		slopeLB:    paramInt(params, "slope_lookback", 3),
		smoothType: paramString(params, "smooth_type", "EMA"),
		smoothLen:  paramInt(params, "smooth_len", 2),
		normMode:   paramString(params, "norm_mode", "ATR"),
		atrLen:     paramInt(params, "atr_len", 14),
		slopeThr:   math.Abs(paramFloat(params, "slope_thr", 0)),
		curveThr:   math.Abs(paramFloat(params, "curve_thr", 0)),
	}
}

// fMa returns MA of src by type (HMA, EMA, SMA, WMA, RMA).
func fMa(src []float64, maType string, maLen int) []float64 {
	switch maType {
	case "SMA":
		return talib.Sma(src, maLen)
	case "EMA":
		return talib.Ema(src, maLen)
	case "HMA":
		return computeHMA(src, maLen)
	case "WMA":
		return talib.Wma(src, maLen)
	case "RMA":
		return RMA(src, maLen)
	default:
		return computeHMA(src, maLen)
	}
}

// fSmooth smooths x by type (None, EMA, SMA, RMA).
// All variants are NaN-aware: leading NaN values in x do not poison the
// running sum/state. Output is NaN wherever the window still contains NaN.
func fSmooth(x []float64, smoothType string, smoothLen int) []float64 {
	switch smoothType {
	case "None", "":
		return x
	case "EMA":
		return nanAwareEWM(x, smoothLen)
	case "SMA":
		return nanAwareSMA(x, smoothLen)
	case "RMA":
		return nanAwareRMA(x, smoothLen)
	default:
		return nanAwareEWM(x, smoothLen)
	}
}

// nanAwareSMA computes SMA(period) without propagating NaN through the running
// sum. Any window that contains at least one NaN produces NaN; once all values
// in the window are valid, the standard per-window mean is returned.
func nanAwareSMA(x []float64, period int) []float64 {
	n := len(x)
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		if i < period-1 {
			out[i] = math.NaN()
			continue
		}
		sum := 0.0
		bad := false
		for j := i - period + 1; j <= i; j++ {
			if math.IsNaN(x[j]) {
				bad = true
				break
			}
			sum += x[j]
		}
		if bad {
			out[i] = math.NaN()
		} else {
			out[i] = sum / float64(period)
		}
	}
	return out
}

// nanAwareEWM is like EWM but seeds itself from the first non-NaN value rather
// than from index 0, so leading NaN values do not poison the recursive state.
func nanAwareEWM(x []float64, span int) []float64 {
	n := len(x)
	out := make([]float64, n)
	for i := range out {
		out[i] = math.NaN()
	}
	alpha := 2.0 / (float64(span) + 1.0)
	started := false
	prev := 0.0
	for i := 0; i < n; i++ {
		if math.IsNaN(x[i]) {
			continue
		}
		if !started {
			prev = x[i]
			out[i] = prev
			started = true
		} else {
			prev = alpha*x[i] + (1-alpha)*prev
			out[i] = prev
		}
	}
	return out
}

// nanAwareRMA is like RMA but seeds itself from the first non-NaN value.
func nanAwareRMA(x []float64, length int) []float64 {
	n := len(x)
	out := make([]float64, n)
	for i := range out {
		out[i] = math.NaN()
	}
	if length <= 0 {
		return out
	}
	alpha := 1.0 / float64(length)
	started := false
	prev := 0.0
	for i := 0; i < n; i++ {
		if math.IsNaN(x[i]) {
			continue
		}
		if !started {
			prev = x[i]
			out[i] = prev
			started = true
		} else {
			prev = alpha*x[i] + (1-alpha)*prev
			out[i] = prev
		}
	}
	return out
}

// maSlopeCurveCalc computes MA, normalized slope, normalized curvature, and signal pulses.
// All slices have length n = len(close). Leading NaNs where not enough data.
func maSlopeCurveCalc(data *OHLCV, params map[string]interface{}) (
	ma, slope, curve []float64,
	turnUp, turnDn, bendUp, bendDn, earlyUp, earlyDn []float64,
) {
	p := parseMaSlopeCurveParams(params)
	src := resolveInput(data, params)
	n := len(src)
	ma = make([]float64, n)
	slope = make([]float64, n)
	curve = make([]float64, n)
	turnUp = make([]float64, n)
	turnDn = make([]float64, n)
	bendUp = make([]float64, n)
	bendDn = make([]float64, n)
	earlyUp = make([]float64, n)
	earlyDn = make([]float64, n)

	for i := 0; i < n; i++ {
		ma[i] = math.NaN()
		slope[i] = math.NaN()
		curve[i] = math.NaN()
	}

	m := fMa(src, p.maType, p.maLen)
	slopeRaw := make([]float64, n)
	for i := range slopeRaw {
		slopeRaw[i] = math.NaN()
	}
	for i := p.slopeLB; i < n; i++ {
		diff := m[i] - m[i-p.slopeLB]
		if math.IsNaN(diff) {
			continue
		}
		slopeRaw[i] = diff / float64(p.slopeLB)
	}
	slopeSm := fSmooth(slopeRaw, p.smoothType, p.smoothLen)

	curveRaw := make([]float64, n)
	for i := range curveRaw {
		curveRaw[i] = math.NaN()
	}
	for i := 1; i < n; i++ {
		if math.IsNaN(slopeSm[i]) || math.IsNaN(slopeSm[i-1]) {
			continue
		}
		curveRaw[i] = slopeSm[i] - slopeSm[i-1]
	}
	curveSm := fSmooth(curveRaw, p.smoothType, p.smoothLen)

	var atr []float64
	if p.normMode == "ATR" {
		atr = talib.Atr(data.High, data.Low, data.Close, p.atrLen)
	}

	for i := 0; i < n; i++ {
		ma[i] = m[i]
		sN := slopeSm[i]
		cN := curveSm[i]
		if p.normMode == "ATR" && i < len(atr) && atr[i] > 0 && !math.IsNaN(atr[i]) {
			sN = slopeSm[i] / atr[i]
			cN = curveSm[i] / atr[i]
		} else if p.normMode == "Percent" && i < len(m) && m[i] != 0 && !math.IsNaN(m[i]) {
			sN = (slopeSm[i] / m[i]) * 100
			cN = (curveSm[i] / m[i]) * 100
		}
		slope[i] = sN
		curve[i] = cN

		sThr := p.slopeThr
		cThr := p.curveThr
		if cThr == 0 {
			cThr = 1e-10
		}
		if sThr == 0 {
			sThr = 1e-10
		}

		// One-bar pulses
		turnUp[i] = 0
		turnDn[i] = 0
		bendUp[i] = 0
		bendDn[i] = 0
		earlyUp[i] = 0
		earlyDn[i] = 0
		if i == 0 {
			continue
		}
		prevS := slope[i-1]
		prevC := curve[i-1]
		if math.IsNaN(prevS) {
			prevS = sN
		}
		if math.IsNaN(prevC) {
			prevC = cN
		}
		if !math.IsNaN(sN) && sN > sThr && prevS <= sThr {
			turnUp[i] = 1
		}
		if !math.IsNaN(sN) && sN < -sThr && prevS >= -sThr {
			turnDn[i] = 1
		}
		if !math.IsNaN(cN) && cN > cThr && prevC <= cThr {
			bendUp[i] = 1
			if !math.IsNaN(sN) && sN <= sThr {
				earlyUp[i] = 1
			}
		}
		if !math.IsNaN(cN) && cN < -cThr && prevC >= -cThr {
			bendDn[i] = 1
			if !math.IsNaN(sN) && sN >= -sThr {
				earlyDn[i] = 1
			}
		}
	}
	return ma, slope, curve, turnUp, turnDn, bendUp, bendDn, earlyUp, earlyDn
}

// MaSlopeCurveMA returns the MA line. Params: ma_len/timeperiod, ma_type, input, slope_lookback, smooth_type, smooth_len, norm_mode, atr_len, slope_thr, curve_thr.
func MaSlopeCurveMA(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	ma, _, _, _, _, _, _, _, _ := maSlopeCurveCalc(data, params)
	return ma, nil
}

// MaSlopeCurveSlope returns the normalized slope series.
func MaSlopeCurveSlope(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	_, slope, _, _, _, _, _, _, _ := maSlopeCurveCalc(data, params)
	return slope, nil
}

// MaSlopeCurveCurve returns the normalized curvature series.
func MaSlopeCurveCurve(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	_, _, curve, _, _, _, _, _, _ := maSlopeCurveCalc(data, params)
	return curve, nil
}

// MaSlopeCurveTurnUp returns 1 on slope turn-up pulse bars, 0 otherwise.
func MaSlopeCurveTurnUp(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	_, _, _, turnUp, _, _, _, _, _ := maSlopeCurveCalc(data, params)
	return turnUp, nil
}

// MaSlopeCurveTurnDn returns 1 on slope turn-down pulse bars, 0 otherwise.
func MaSlopeCurveTurnDn(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	_, _, _, _, turnDn, _, _, _, _ := maSlopeCurveCalc(data, params)
	return turnDn, nil
}

// MaSlopeCurveBendUp returns 1 on curvature bend-up pulse bars, 0 otherwise.
func MaSlopeCurveBendUp(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	_, _, _, _, _, bendUp, _, _, _ := maSlopeCurveCalc(data, params)
	return bendUp, nil
}

// MaSlopeCurveBendDn returns 1 on curvature bend-down pulse bars, 0 otherwise.
func MaSlopeCurveBendDn(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	_, _, _, _, _, _, bendDn, _, _ := maSlopeCurveCalc(data, params)
	return bendDn, nil
}

// MaSlopeCurveEarlyUp returns 1 on early bend-up pulse bars, 0 otherwise.
func MaSlopeCurveEarlyUp(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	_, _, _, _, _, _, _, earlyUp, _ := maSlopeCurveCalc(data, params)
	return earlyUp, nil
}

// MaSlopeCurveEarlyDn returns 1 on early bend-down pulse bars, 0 otherwise.
func MaSlopeCurveEarlyDn(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	_, _, _, _, _, _, _, _, earlyDn := maSlopeCurveCalc(data, params)
	return earlyDn, nil
}

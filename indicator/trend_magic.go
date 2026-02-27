package indicator

import (
	"math"
)

// computeCCI computes CCI manually (matching Python implementation).
func computeCCI(data *OHLCV, period int, priceType string) []float64 {
	var price []float64
	switch priceType {
	case "HLC":
		price = make([]float64, data.Len())
		for i := range price {
			price[i] = (data.High[i] + data.Low[i] + data.Close[i]) / 3
		}
	default:
		price = data.Close
	}

	n := len(price)
	sma := RollingMean(price, period)

	// Mean deviation
	meanDev := make([]float64, n)
	for i := 0; i < n; i++ {
		if i < period-1 {
			meanDev[i] = math.NaN()
			continue
		}
		sum := 0.0
		for j := i - period + 1; j <= i; j++ {
			sum += math.Abs(price[j] - sma[i])
		}
		meanDev[i] = sum / float64(period)
	}

	cci := make([]float64, n)
	for i := range cci {
		if meanDev[i] != 0 && !math.IsNaN(meanDev[i]) {
			cci[i] = (price[i] - sma[i]) / (0.015 * meanDev[i])
		} else {
			cci[i] = math.NaN()
		}
	}
	return cci
}

// TrendMagic computes the Trend Magic line.
// Params: cci_period (int, default 20), atr_multiplier (float64, default 1.0),
//
//	atr_period (int, default 5), price_type (string, default "Close").
func TrendMagic(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	cciPeriod := paramInt(params, "cci_period", 20)
	atrMul := paramFloat(params, "atr_multiplier", 1.0)
	atrPeriod := paramInt(params, "atr_period", 5)
	priceType := paramString(params, "price_type", "Close")

	n := data.Len()

	tr := TrueRange(data.High, data.Low, data.Close)
	atr := RollingMean(tr, atrPeriod)
	cci := computeCCI(data, cciPeriod, priceType)

	// Upper = Low - ATR * multiplier, Lower = High + ATR * multiplier
	upT := make([]float64, n)
	downT := make([]float64, n)
	for i := range upT {
		upT[i] = data.Low[i] - atr[i]*atrMul
		downT[i] = data.High[i] + atr[i]*atrMul
	}

	magic := make([]float64, n)
	for i := 0; i < n; i++ {
		if i == 0 {
			if !math.IsNaN(cci[i]) && cci[i] >= 0 {
				magic[i] = upT[i]
			} else {
				magic[i] = downT[i]
			}
			continue
		}

		if math.IsNaN(cci[i]) {
			magic[i] = magic[i-1]
			continue
		}

		if cci[i] >= 0 {
			if !math.IsNaN(magic[i-1]) && upT[i] < magic[i-1] {
				magic[i] = magic[i-1]
			} else {
				magic[i] = upT[i]
			}
		} else {
			if !math.IsNaN(magic[i-1]) && downT[i] > magic[i-1] {
				magic[i] = magic[i-1]
			} else {
				magic[i] = downT[i]
			}
		}
	}

	return magic, nil
}

// TrendMagicSignal returns 1 for bullish (CCI >= 0), -1 for bearish.
// Params: cci_period (int, default 20), price_type (string, default "Close").
func TrendMagicSignal(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	cciPeriod := paramInt(params, "cci_period", 20)
	priceType := paramString(params, "price_type", "Close")

	cci := computeCCI(data, cciPeriod, priceType)

	out := make([]float64, len(cci))
	for i := range out {
		if math.IsNaN(cci[i]) {
			out[i] = 0
		} else if cci[i] >= 0 {
			out[i] = 1
		} else {
			out[i] = -1
		}
	}
	return out, nil
}

package indicator

import (
	"math"
)

// PivotSR computes Pivot Support/Resistance signals.
// Returns: 3 strong bullish, 2 bullish, 1 near support,
// 0 no signal, -1 near resistance, -2 bearish, -3 strong bearish.
// Params: left_bars (int, default 5), right_bars (int, default 5),
//
//	proximity_threshold (float64, default 1.0), buffer_percent (float64, default 0.5).
func PivotSR(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	leftBars := paramInt(params, "left_bars", 5)
	rightBars := paramInt(params, "right_bars", 5)
	proxThreshold := paramFloat(params, "proximity_threshold", 1.0)
	bufferPct := paramFloat(params, "buffer_percent", 0.5)

	n := data.Len()
	signals := make([]float64, n)

	// Find pivot highs and lows
	pivotHighs := findPivotHighs(data.High, leftBars, rightBars)
	pivotLows := findPivotLows(data.Low, leftBars, rightBars)

	// Build resistance levels from pivot highs
	resistance := buildLevels(pivotHighs, bufferPct)
	// Build support levels from pivot lows
	support := buildLevels(pivotLows, bufferPct)

	// Check last 20 bars for proximity and crossover
	lookback := 20
	if lookback > n {
		lookback = n
	}

	for i := 0; i < lookback; i++ {
		idx := n - 1 - i
		if idx < 1 {
			break
		}
		currentPrice := data.Close[idx]
		prevPrice := data.Close[idx-1]
		signal := 0.0

		// Check crossovers
		for _, level := range support {
			if prevPrice > level.price && currentPrice <= level.price {
				if level.touches >= 3 {
					signal = -3
				} else {
					signal = -2
				}
			} else if prevPrice < level.price && currentPrice >= level.price {
				signal = 2
			}
		}
		for _, level := range resistance {
			if prevPrice < level.price && currentPrice >= level.price {
				if level.touches >= 3 {
					signal = 3
				} else {
					signal = 2
				}
			} else if prevPrice > level.price && currentPrice <= level.price {
				signal = -2
			}
		}

		// If no crossover, check proximity
		if signal == 0 {
			for _, level := range support {
				distPct := math.Abs((currentPrice - level.price) / level.price * 100)
				if distPct <= proxThreshold {
					if signal < 1 {
						signal = 1
					}
				}
			}
			for _, level := range resistance {
				distPct := math.Abs((currentPrice - level.price) / level.price * 100)
				if distPct <= proxThreshold {
					if signal > -1 {
						signal = -1
					}
				}
			}
		}

		signals[idx] = signal
	}

	return signals, nil
}

// PivotSRCrossover returns 1 for bullish crossover, -1 for bearish, 0 otherwise.
func PivotSRCrossover(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	leftBars := paramInt(params, "left_bars", 5)
	rightBars := paramInt(params, "right_bars", 5)
	bufferPct := paramFloat(params, "buffer_percent", 0.5)

	n := data.Len()
	signals := make([]float64, n)

	pivotHighs := findPivotHighs(data.High, leftBars, rightBars)
	pivotLows := findPivotLows(data.Low, leftBars, rightBars)
	resistance := buildLevels(pivotHighs, bufferPct)
	support := buildLevels(pivotLows, bufferPct)
	allLevels := append(support, resistance...)

	lookback := 5
	if lookback > n {
		lookback = n
	}

	for i := 0; i < lookback; i++ {
		idx := n - 1 - i
		if idx < 1 {
			break
		}
		currentPrice := data.Close[idx]
		prevPrice := data.Close[idx-1]
		signal := 0.0

		for _, level := range allLevels {
			if prevPrice < level.price && currentPrice >= level.price {
				signal = 1
			} else if prevPrice > level.price && currentPrice <= level.price {
				signal = -1
			}
		}
		signals[idx] = signal
	}

	return signals, nil
}

// PivotSRProximity returns 1 near support, -1 near resistance, 0 otherwise.
func PivotSRProximity(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	leftBars := paramInt(params, "left_bars", 5)
	rightBars := paramInt(params, "right_bars", 5)
	proxThreshold := paramFloat(params, "proximity_threshold", 1.0)
	bufferPct := paramFloat(params, "buffer_percent", 0.5)

	n := data.Len()
	signals := make([]float64, n)

	pivotHighs := findPivotHighs(data.High, leftBars, rightBars)
	pivotLows := findPivotLows(data.Low, leftBars, rightBars)
	resistance := buildLevels(pivotHighs, bufferPct)
	support := buildLevels(pivotLows, bufferPct)

	lookback := 5
	if lookback > n {
		lookback = n
	}

	for i := 0; i < lookback; i++ {
		idx := n - 1 - i
		if idx < 0 {
			break
		}
		currentPrice := data.Close[idx]
		signal := 0.0

		for _, level := range support {
			distPct := math.Abs((currentPrice - level.price) / level.price * 100)
			if distPct <= proxThreshold {
				signal = 1
			}
		}
		for _, level := range resistance {
			distPct := math.Abs((currentPrice - level.price) / level.price * 100)
			if distPct <= proxThreshold {
				signal = -1
			}
		}
		signals[idx] = signal
	}

	return signals, nil
}

// Internal types and helpers

type pivotPoint struct {
	index int
	price float64
}

type priceLevel struct {
	price   float64
	touches int
}

func findPivotHighs(high []float64, leftBars, rightBars int) []pivotPoint {
	var pivots []pivotPoint
	n := len(high)
	for i := leftBars; i < n-rightBars; i++ {
		isPivot := true
		val := high[i]
		for j := i - leftBars; j < i; j++ {
			if high[j] >= val {
				isPivot = false
				break
			}
		}
		if isPivot {
			for j := i + 1; j <= i+rightBars; j++ {
				if high[j] >= val {
					isPivot = false
					break
				}
			}
		}
		if isPivot {
			pivots = append(pivots, pivotPoint{index: i, price: val})
		}
	}
	return pivots
}

func findPivotLows(low []float64, leftBars, rightBars int) []pivotPoint {
	var pivots []pivotPoint
	n := len(low)
	for i := leftBars; i < n-rightBars; i++ {
		isPivot := true
		val := low[i]
		for j := i - leftBars; j < i; j++ {
			if low[j] <= val {
				isPivot = false
				break
			}
		}
		if isPivot {
			for j := i + 1; j <= i+rightBars; j++ {
				if low[j] <= val {
					isPivot = false
					break
				}
			}
		}
		if isPivot {
			pivots = append(pivots, pivotPoint{index: i, price: val})
		}
	}
	return pivots
}

func buildLevels(pivots []pivotPoint, bufferPct float64) []priceLevel {
	var levels []priceLevel
	for _, p := range pivots {
		found := false
		for i := range levels {
			pctDiff := math.Abs((p.price - levels[i].price) / levels[i].price * 100)
			if pctDiff < bufferPct {
				levels[i].touches++
				found = true
				break
			}
		}
		if !found {
			levels = append(levels, priceLevel{price: p.price, touches: 1})
		}
	}
	return levels
}

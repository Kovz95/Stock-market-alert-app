package indicator

import (
	"fmt"

	"github.com/markcheno/go-talib"
)

// MACD computes Moving Average Convergence Divergence.
// Params: fast_period (int, default 12), slow_period (int, default 26),
//
//	signal_period (int, default 9), type (string: "line", "signal", or "histogram", default "line").
func MACD(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	fastPeriod := paramInt(params, "fast_period", 12)
	slowPeriod := paramInt(params, "slow_period", 26)
	signalPeriod := paramInt(params, "signal_period", 9)
	outputType := paramString(params, "type", "line")

	macdLine, macdSignal, macdHist := talib.Macd(data.Close, fastPeriod, slowPeriod, signalPeriod)

	switch outputType {
	case "line":
		return macdLine, nil
	case "signal":
		return macdSignal, nil
	case "histogram":
		return macdHist, nil
	default:
		return nil, fmt.Errorf("MACD type must be 'line', 'signal', or 'histogram', got %q", outputType)
	}
}

// macdParams extracts the MACD period params, supporting both the legacy names
// (fast_period, slow_period, signal_period) and the shorter names the frontend
// emits (fast, slow, signal).
func macdParams(params map[string]interface{}) (fast, slow, signal int) {
	fast = paramInt(params, "fast", 12)
	if v := paramInt(params, "fast_period", 0); v > 0 {
		fast = v
	}
	slow = paramInt(params, "slow", 26)
	if v := paramInt(params, "slow_period", 0); v > 0 {
		slow = v
	}
	signal = paramInt(params, "signal", 9)
	if v := paramInt(params, "signal_period", 0); v > 0 {
		signal = v
	}
	return
}

// MACDLine returns only the MACD line (fast EMA − slow EMA).
// Params: fast (int, default 12), slow (int, default 26), signal (int, default 9).
// Also accepts fast_period / slow_period / signal_period for compatibility.
func MACDLine(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	fast, slow, sig := macdParams(params)
	line, _, _ := talib.Macd(data.Close, fast, slow, sig)
	return line, nil
}

// MACDSignal returns only the MACD signal line.
// Params: fast (int, default 12), slow (int, default 26), signal (int, default 9).
func MACDSignal(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	fast, slow, sig := macdParams(params)
	_, signal, _ := talib.Macd(data.Close, fast, slow, sig)
	return signal, nil
}

// MACDHistogram returns only the MACD histogram (line − signal).
// Params: fast (int, default 12), slow (int, default 26), signal (int, default 9).
func MACDHistogram(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	fast, slow, sig := macdParams(params)
	_, _, hist := talib.Macd(data.Close, fast, slow, sig)
	return hist, nil
}

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

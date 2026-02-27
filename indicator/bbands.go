package indicator

import (
	"fmt"

	"github.com/markcheno/go-talib"
)

// BBANDS computes Bollinger Bands.
// Params: timeperiod (int, default 20), std_dev (float64, default 2.0),
//
//	type (string: "upper", "middle", "lower", default "upper").
func BBANDS(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "timeperiod", 20)
	stdDev := paramFloat(params, "std_dev", 2.0)
	outputType := paramString(params, "type", "upper")

	upper, middle, lower := talib.BBands(data.Close, period, stdDev, stdDev, talib.SMA)

	switch outputType {
	case "upper":
		return upper, nil
	case "middle":
		return middle, nil
	case "lower":
		return lower, nil
	default:
		return nil, fmt.Errorf("BBANDS type must be 'upper', 'middle', or 'lower', got %q", outputType)
	}
}

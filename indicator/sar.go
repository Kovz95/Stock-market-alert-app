package indicator

import (
	"github.com/markcheno/go-talib"
)

// SAR computes Parabolic SAR.
// Params: acceleration (float64, default 0.02), max_acceleration (float64, default 0.2).
func SAR(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	accel := paramFloat(params, "acceleration", 0.02)
	maxAccel := paramFloat(params, "max_acceleration", 0.2)
	return talib.Sar(data.High, data.Low, accel, maxAccel), nil
}

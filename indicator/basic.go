package indicator

import (
	"math"

	"github.com/markcheno/go-talib"
)

// SMA computes Simple Moving Average.
// Params: timeperiod (int, required), input (string, default "Close").
func SMA(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "timeperiod", 20)
	input := resolveInput(data, params)
	// When input is a computed nested series (e.g. sma(input=ewo(...))), it may
	// contain NaN from indicator warmup. talib.Sma uses a running sum that is
	// permanently poisoned by NaN, so fall back to the per-window nanAwareSMA.
	if _, ok := params["_computed_input"]; ok {
		return nanAwareSMA(input, period), nil
	}
	return talib.Sma(input, period), nil
}

// EMA computes Exponential Moving Average.
// Params: timeperiod (int, required), input (string, default "Close").
func EMA(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "timeperiod", 20)
	input := resolveInput(data, params)
	return talib.Ema(input, period), nil
}

// RSI computes Relative Strength Index.
// Params: timeperiod (int, default 14), input (string, default "Close").
func RSI(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "timeperiod", 14)
	input := resolveInput(data, params)
	return talib.Rsi(input, period), nil
}

// ROC computes Rate of Change.
// Params: timeperiod (int, default 10), input (string, default "Close").
func ROC(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "timeperiod", 10)
	input := resolveInput(data, params)
	return talib.Roc(input, period), nil
}

// ATR computes Average True Range.
// Params: timeperiod (int, default 14).
func ATR(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "timeperiod", 14)
	return talib.Atr(data.High, data.Low, data.Close, period), nil
}

// CCI computes Commodity Channel Index using go-talib.
// Params: timeperiod (int, default 20).
func CCI(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "timeperiod", 20)
	return talib.Cci(data.High, data.Low, data.Close, period), nil
}

// WILLR computes Williams %R.
// Params: timeperiod (int, default 14).
func WILLR(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "timeperiod", 14)
	return talib.WillR(data.High, data.Low, data.Close, period), nil
}

// VolumeRatio returns volume / SMA(volume, period) for each bar.
// Params: timeperiod (int, default 20). Used for volume_above_average / volume_spike conditions.
func VolumeRatio(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "timeperiod", 20)
	volSma := talib.Sma(data.Volume, period)
	n := len(data.Volume)
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		if i < period-1 || volSma[i] == 0 || volSma[i] != volSma[i] {
			out[i] = math.NaN()
			continue
		}
		out[i] = data.Volume[i] / volSma[i]
	}
	return out, nil
}

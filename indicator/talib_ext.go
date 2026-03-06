// Package indicator: additional go-talib wrappers (ADX, Stoch, StochRSI, OBV, MFI, Mom, Ad).
// See docs/TALIB_INDICATORS_ROADMAP.md for the full list of TA-Lib functions to expose.

package indicator

import (
	"github.com/markcheno/go-talib"
)

// ADX computes Average Directional Index (trend strength, 0–100).
// Params: timeperiod (int, default 14).
func ADX(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "timeperiod", 14)
	return talib.Adx(data.High, data.Low, data.Close, period), nil
}

// StochK returns the Stochastic %K line.
// Params: fast_k_period (int, default 5), slow_k_period (int, default 3), slow_d_period (int, default 3).
func StochK(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	fastK := paramInt(params, "fast_k_period", 5)
	slowK := paramInt(params, "slow_k_period", 3)
	slowD := paramInt(params, "slow_d_period", 3)
	k, _ := talib.Stoch(data.High, data.Low, data.Close, fastK, slowK, talib.SMA, slowD, talib.SMA)
	return k, nil
}

// StochD returns the Stochastic %D line (signal).
// Params: fast_k_period (int, default 5), slow_k_period (int, default 3), slow_d_period (int, default 3).
func StochD(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	fastK := paramInt(params, "fast_k_period", 5)
	slowK := paramInt(params, "slow_k_period", 3)
	slowD := paramInt(params, "slow_d_period", 3)
	_, d := talib.Stoch(data.High, data.Low, data.Close, fastK, slowK, talib.SMA, slowD, talib.SMA)
	return d, nil
}

// StochRsiK returns the Stochastic RSI %K line.
// Params: timeperiod (int, default 14), fast_k_period (int, default 5), fast_d_period (int, default 3).
func StochRsiK(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "timeperiod", 14)
	fastK := paramInt(params, "fast_k_period", 5)
	fastD := paramInt(params, "fast_d_period", 3)
	k, _ := talib.StochRsi(data.Close, period, fastK, fastD, talib.SMA)
	return k, nil
}

// StochRsiD returns the Stochastic RSI %D line.
// Params: timeperiod (int, default 14), fast_k_period (int, default 5), fast_d_period (int, default 3).
func StochRsiD(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "timeperiod", 14)
	fastK := paramInt(params, "fast_k_period", 5)
	fastD := paramInt(params, "fast_d_period", 3)
	_, d := talib.StochRsi(data.Close, period, fastK, fastD, talib.SMA)
	return d, nil
}

// OBV computes On Balance Volume.
// No params (uses close and volume).
func OBV(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	return talib.Obv(data.Close, data.Volume), nil
}

// MFI computes Money Flow Index (volume-weighted RSI-like, 0–100).
// Params: timeperiod (int, default 14).
func MFI(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "timeperiod", 14)
	return talib.Mfi(data.High, data.Low, data.Close, data.Volume, period), nil
}

// Mom computes Momentum (close - close[n]).
// Params: timeperiod (int, default 10).
func Mom(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "timeperiod", 10)
	input := resolveInput(data, params)
	return talib.Mom(input, period), nil
}

// Ad computes Accumulation/Distribution line.
// No params.
func Ad(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	return talib.Ad(data.High, data.Low, data.Close, data.Volume), nil
}

// --- Medium priority: trend / volatility / slope ---

// PlusDI returns the Plus Directional Indicator (+DI, part of ADX system).
// Params: timeperiod (int, default 14).
func PlusDI(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "timeperiod", 14)
	return talib.PlusDI(data.High, data.Low, data.Close, period), nil
}

// MinusDI returns the Minus Directional Indicator (-DI, part of ADX system).
// Params: timeperiod (int, default 14).
func MinusDI(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "timeperiod", 14)
	return talib.MinusDI(data.High, data.Low, data.Close, period), nil
}

// Natr returns the Normalized Average True Range (ATR as percentage of close).
// Params: timeperiod (int, default 14).
func Natr(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "timeperiod", 14)
	return talib.Natr(data.High, data.Low, data.Close, period), nil
}

// LinearRegSlope returns the slope of the linear regression line over the period.
// Params: timeperiod (int, default 14), input (string, default "Close").
func LinearRegSlope(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "timeperiod", 14)
	input := resolveInput(data, params)
	return talib.LinearRegSlope(input, period), nil
}

// LinearReg returns the linear regression value over the period.
// Params: timeperiod (int, default 14), input (string, default "Close").
func LinearReg(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "timeperiod", 14)
	input := resolveInput(data, params)
	return talib.LinearReg(input, period), nil
}

// StdDev returns the rolling standard deviation over the period.
// Params: timeperiod (int, default 20), nb_dev (float64, default 1.0), input (string, default "Close").
func StdDev(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "timeperiod", 20)
	nbDev := paramFloat(params, "nb_dev", 1.0)
	input := resolveInput(data, params)
	return talib.StdDev(input, period, nbDev), nil
}

// AroonOsc returns the Aroon Oscillator (Aroon Up - Aroon Down).
// Params: timeperiod (int, default 14).
func AroonOsc(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "timeperiod", 14)
	return talib.AroonOsc(data.High, data.Low, period), nil
}

// Cmo returns the Chande Momentum Oscillator.
// Params: timeperiod (int, default 14), input (string, default "Close").
func Cmo(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "timeperiod", 14)
	input := resolveInput(data, params)
	return talib.Cmo(input, period), nil
}

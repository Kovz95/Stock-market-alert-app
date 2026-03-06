// Package indicator: custom indicators (e.g. ported from TradingView / Pine Script).
// Add your own IndicatorFunc implementations here and register them in registry.go
// (or via a custom registry in your app). See docs/CUSTOM_INDICATORS_FROM_TRADINGVIEW.md.

package indicator

import (
	"github.com/markcheno/go-talib"
)

// MySmoothedRSI computes EMA(RSI(source, length), smooth_len).
// Example port from Pine: RSI then EMA smoothing.
// Params: timeperiod (int, default 14), smooth (int, default 7), input (string, default "Close").
func MySmoothedRSI(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	period := paramInt(params, "timeperiod", 14)
	smooth := paramInt(params, "smooth", 7)
	input := resolveInput(data, params)

	rsi := talib.Rsi(input, period)
	smoothed := talib.Ema(rsi, smooth)
	return smoothed, nil
}

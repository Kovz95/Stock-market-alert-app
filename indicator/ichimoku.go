package indicator

import (
	"math"
)

// donchianMid computes (highest high + lowest low) / 2 over the given period.
func donchianMid(high, low []float64, period int) []float64 {
	hh := RollingMax(high, period)
	ll := RollingMin(low, period)
	out := make([]float64, len(high))
	for i := range out {
		out[i] = (hh[i] + ll[i]) / 2
	}
	return out
}

// IchimokuConversion computes the Conversion Line (Tenkan-sen).
// Params: periods (int, default 9).
func IchimokuConversion(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	periods := paramInt(params, "periods", 9)
	return donchianMid(data.High, data.Low, periods), nil
}

// IchimokuBase computes the Base Line (Kijun-sen).
// Params: periods (int, default 26).
func IchimokuBase(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	periods := paramInt(params, "periods", 26)
	return donchianMid(data.High, data.Low, periods), nil
}

// IchimokuSpanA computes Leading Span A (Senkou Span A).
// Unshifted by default (for scanning). Set visual=true for traditional forward displacement.
// Params: conversion_periods (int, default 9), base_periods (int, default 26),
//
//	displacement (int, default 26), visual (bool, default false).
func IchimokuSpanA(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	convPeriods := paramInt(params, "conversion_periods", 9)
	basePeriods := paramInt(params, "base_periods", 26)
	displacement := paramInt(params, "displacement", 26)
	visual := paramBool(params, "visual", false)

	conversion := donchianMid(data.High, data.Low, convPeriods)
	base := donchianMid(data.High, data.Low, basePeriods)

	n := len(conversion)
	spanA := make([]float64, n)
	for i := range spanA {
		spanA[i] = (conversion[i] + base[i]) / 2
	}

	if visual {
		return Shift(spanA, displacement), nil
	}
	return spanA, nil
}

// IchimokuSpanB computes Leading Span B (Senkou Span B).
// Params: periods (int, default 52), displacement (int, default 26), visual (bool, default false).
func IchimokuSpanB(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	periods := paramInt(params, "periods", 52)
	displacement := paramInt(params, "displacement", 26)
	visual := paramBool(params, "visual", false)

	spanB := donchianMid(data.High, data.Low, periods)

	if visual {
		return Shift(spanB, displacement), nil
	}
	return spanB, nil
}

// IchimokuLagging computes the Lagging Span (Chikou Span).
// Params: displacement (int, default 26), visual (bool, default false).
func IchimokuLagging(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	displacement := paramInt(params, "displacement", 26)
	visual := paramBool(params, "visual", false)

	if visual {
		return Shift(data.Close, -displacement), nil
	}
	// Unshifted: just Close
	out := make([]float64, len(data.Close))
	copy(out, data.Close)
	return out, nil
}

// IchimokuCloudTop returns the max of Span A and Span B.
// Params: conversion_periods (int, default 9), base_periods (int, default 26),
//
//	span_b_periods (int, default 52), displacement (int, default 26), visual (bool, default false).
func IchimokuCloudTop(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	spanA, _ := IchimokuSpanA(data, params)
	spanBParams := map[string]interface{}{
		"periods":      paramInt(params, "span_b_periods", 52),
		"displacement": paramInt(params, "displacement", 26),
		"visual":       paramBool(params, "visual", false),
	}
	spanB, _ := IchimokuSpanB(data, spanBParams)

	out := make([]float64, len(spanA))
	for i := range out {
		out[i] = math.Max(spanA[i], spanB[i])
	}
	return out, nil
}

// IchimokuCloudBottom returns the min of Span A and Span B.
// Params: same as IchimokuCloudTop.
func IchimokuCloudBottom(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	spanA, _ := IchimokuSpanA(data, params)
	spanBParams := map[string]interface{}{
		"periods":      paramInt(params, "span_b_periods", 52),
		"displacement": paramInt(params, "displacement", 26),
		"visual":       paramBool(params, "visual", false),
	}
	spanB, _ := IchimokuSpanB(data, spanBParams)

	out := make([]float64, len(spanA))
	for i := range out {
		out[i] = math.Min(spanA[i], spanB[i])
	}
	return out, nil
}

// IchimokuCloudSignal returns 1 for bullish cloud (Span A > Span B),
// -1 for bearish, 0 for neutral.
func IchimokuCloudSignal(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	spanA, _ := IchimokuSpanA(data, params)
	spanBParams := map[string]interface{}{
		"periods":      paramInt(params, "span_b_periods", 52),
		"displacement": paramInt(params, "displacement", 26),
		"visual":       paramBool(params, "visual", false),
	}
	spanB, _ := IchimokuSpanB(data, spanBParams)

	out := make([]float64, len(spanA))
	for i := range out {
		if spanA[i] > spanB[i] {
			out[i] = 1
		} else if spanA[i] < spanB[i] {
			out[i] = -1
		}
	}
	return out, nil
}

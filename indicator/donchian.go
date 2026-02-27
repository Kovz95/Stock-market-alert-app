package indicator

import (
	"math"
)

// DonchianUpper returns the highest high over the period.
// Params: length (int, default 20), offset (int, default 0).
func DonchianUpper(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	length := paramInt(params, "length", 20)
	offset := paramInt(params, "offset", 0)

	upper := RollingMax(data.High, length)
	if offset != 0 {
		upper = Shift(upper, offset)
	}
	return upper, nil
}

// DonchianLower returns the lowest low over the period.
// Params: length (int, default 20), offset (int, default 0).
func DonchianLower(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	length := paramInt(params, "length", 20)
	offset := paramInt(params, "offset", 0)

	lower := RollingMin(data.Low, length)
	if offset != 0 {
		lower = Shift(lower, offset)
	}
	return lower, nil
}

// DonchianBasis returns the midline (upper + lower) / 2.
// Params: length (int, default 20), offset (int, default 0).
func DonchianBasis(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	length := paramInt(params, "length", 20)
	offset := paramInt(params, "offset", 0)

	upper := RollingMax(data.High, length)
	lower := RollingMin(data.Low, length)

	basis := make([]float64, len(upper))
	for i := range basis {
		basis[i] = (upper[i] + lower[i]) / 2
	}
	if offset != 0 {
		basis = Shift(basis, offset)
	}
	return basis, nil
}

// DonchianWidth returns the channel width (upper - lower).
// Params: length (int, default 20).
func DonchianWidth(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	length := paramInt(params, "length", 20)

	upper := RollingMax(data.High, length)
	lower := RollingMin(data.Low, length)

	width := make([]float64, len(upper))
	for i := range width {
		width[i] = upper[i] - lower[i]
	}
	return width, nil
}

// DonchianPosition returns price position within the channel (0 to 1).
// Params: length (int, default 20).
func DonchianPosition(data *OHLCV, params map[string]interface{}) ([]float64, error) {
	length := paramInt(params, "length", 20)

	upper := RollingMax(data.High, length)
	lower := RollingMin(data.Low, length)

	pos := make([]float64, len(upper))
	for i := range pos {
		w := upper[i] - lower[i]
		if math.IsNaN(w) || w == 0 {
			pos[i] = 0.5
		} else {
			pos[i] = (data.Close[i] - lower[i]) / w
		}
	}
	return pos, nil
}

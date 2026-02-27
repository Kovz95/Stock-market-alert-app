package indicator

import (
	"fmt"
	"math"
)

// OHLCV represents a collection of Open, High, Low, Close, Volume price series.
// All slices must have the same length. Index 0 is the oldest bar.
type OHLCV struct {
	Open   []float64
	High   []float64
	Low    []float64
	Close  []float64
	Volume []float64
}

// Len returns the number of bars in the OHLCV data.
func (o *OHLCV) Len() int {
	return len(o.Close)
}

// Column returns a named price series. Accepted names (case-insensitive during
// registry lookup, but here exact): "Open", "High", "Low", "Close", "Volume",
// "HL2", "HLC3", "OHLC4".
func (o *OHLCV) Column(name string) ([]float64, error) {
	switch name {
	case "Open":
		return o.Open, nil
	case "High":
		return o.High, nil
	case "Low":
		return o.Low, nil
	case "Close":
		return o.Close, nil
	case "Volume":
		return o.Volume, nil
	case "HL2":
		return o.hl2(), nil
	case "HLC3":
		return o.hlc3(), nil
	case "OHLC4":
		return o.ohlc4(), nil
	default:
		return nil, fmt.Errorf("unknown column %q", name)
	}
}

func (o *OHLCV) hl2() []float64 {
	out := make([]float64, o.Len())
	for i := range out {
		out[i] = (o.High[i] + o.Low[i]) / 2
	}
	return out
}

func (o *OHLCV) hlc3() []float64 {
	out := make([]float64, o.Len())
	for i := range out {
		out[i] = (o.High[i] + o.Low[i] + o.Close[i]) / 3
	}
	return out
}

func (o *OHLCV) ohlc4() []float64 {
	out := make([]float64, o.Len())
	for i := range out {
		out[i] = (o.Open[i] + o.High[i] + o.Low[i] + o.Close[i]) / 4
	}
	return out
}

// Validate checks that all slices have the same length and are non-empty.
func (o *OHLCV) Validate() error {
	n := len(o.Close)
	if n == 0 {
		return fmt.Errorf("OHLCV data is empty")
	}
	if len(o.Open) != n || len(o.High) != n || len(o.Low) != n {
		return fmt.Errorf("OHLCV slices have mismatched lengths")
	}
	if len(o.Volume) != 0 && len(o.Volume) != n {
		return fmt.Errorf("Volume slice length %d != Close length %d", len(o.Volume), n)
	}
	return nil
}

// Series wraps a float64 slice and supports Python-style negative indexing.
// Index -1 refers to the last element, -2 to the second-to-last, etc.
type Series struct {
	Data []float64
}

// NewSeries creates a Series from the given data.
func NewSeries(data []float64) Series {
	return Series{Data: data}
}

// Len returns the length of the series.
func (s Series) Len() int {
	return len(s.Data)
}

// At returns the value at the given index. Negative indices are supported:
// -1 is the last element, -2 the second-to-last, etc.
// Returns NaN if the index is out of bounds.
func (s Series) At(index int) float64 {
	i := s.resolveIndex(index)
	if i < 0 || i >= len(s.Data) {
		return math.NaN()
	}
	return s.Data[i]
}

// resolveIndex converts a possibly-negative index to a non-negative one.
func (s Series) resolveIndex(index int) int {
	if index < 0 {
		return len(s.Data) + index
	}
	return index
}

// NaN returns a slice of n NaN values.
func NaN(n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = math.NaN()
	}
	return out
}

// Shift returns a new slice shifted forward by n positions, filling with NaN.
// Equivalent to pandas Series.shift(n).
func Shift(data []float64, n int) []float64 {
	out := make([]float64, len(data))
	if n >= 0 {
		for i := 0; i < n && i < len(data); i++ {
			out[i] = math.NaN()
		}
		for i := n; i < len(data); i++ {
			out[i] = data[i-n]
		}
	} else {
		abs := -n
		for i := 0; i < len(data)-abs; i++ {
			out[i] = data[i+abs]
		}
		for i := len(data) - abs; i < len(data); i++ {
			out[i] = math.NaN()
		}
	}
	return out
}

// RollingMax returns the rolling maximum over a window.
func RollingMax(data []float64, window int) []float64 {
	n := len(data)
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		if i < window-1 {
			out[i] = math.NaN()
			continue
		}
		mx := math.Inf(-1)
		for j := i - window + 1; j <= i; j++ {
			if data[j] > mx {
				mx = data[j]
			}
		}
		out[i] = mx
	}
	return out
}

// RollingMin returns the rolling minimum over a window.
func RollingMin(data []float64, window int) []float64 {
	n := len(data)
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		if i < window-1 {
			out[i] = math.NaN()
			continue
		}
		mn := math.Inf(1)
		for j := i - window + 1; j <= i; j++ {
			if data[j] < mn {
				mn = data[j]
			}
		}
		out[i] = mn
	}
	return out
}

// RollingMean returns the rolling mean over a window.
func RollingMean(data []float64, window int) []float64 {
	n := len(data)
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		if i < window-1 {
			out[i] = math.NaN()
			continue
		}
		sum := 0.0
		for j := i - window + 1; j <= i; j++ {
			sum += data[j]
		}
		out[i] = sum / float64(window)
	}
	return out
}

// RollingStd returns the rolling population standard deviation over a window.
func RollingStd(data []float64, window int) []float64 {
	n := len(data)
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		if i < window-1 {
			out[i] = math.NaN()
			continue
		}
		sum := 0.0
		for j := i - window + 1; j <= i; j++ {
			sum += data[j]
		}
		mean := sum / float64(window)
		variance := 0.0
		for j := i - window + 1; j <= i; j++ {
			d := data[j] - mean
			variance += d * d
		}
		out[i] = math.Sqrt(variance / float64(window))
	}
	return out
}

// TrueRange computes the True Range series from high, low, close.
func TrueRange(high, low, close []float64) []float64 {
	n := len(close)
	tr := make([]float64, n)
	tr[0] = high[0] - low[0]
	for i := 1; i < n; i++ {
		hl := high[i] - low[i]
		hc := math.Abs(high[i] - close[i-1])
		lc := math.Abs(low[i] - close[i-1])
		tr[i] = math.Max(hl, math.Max(hc, lc))
	}
	return tr
}

// CumSum returns the cumulative sum of the input.
func CumSum(data []float64) []float64 {
	out := make([]float64, len(data))
	sum := 0.0
	for i, v := range data {
		sum += v
		out[i] = sum
	}
	return out
}

// EWM computes an exponential weighted mean (EMA) with the given span.
// adjust=false mode (recursive), matching pandas ewm(span=span, adjust=False).mean().
func EWM(data []float64, span int) []float64 {
	n := len(data)
	if n == 0 {
		return nil
	}
	alpha := 2.0 / (float64(span) + 1.0)
	out := make([]float64, n)
	out[0] = data[0]
	for i := 1; i < n; i++ {
		out[i] = alpha*data[i] + (1-alpha)*out[i-1]
	}
	return out
}

// WMA computes a Weighted Moving Average.
func WMA(data []float64, period int) []float64 {
	n := len(data)
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		if i < period-1 {
			out[i] = math.NaN()
			continue
		}
		wsum := 0.0
		denom := 0.0
		for j := 0; j < period; j++ {
			w := float64(j + 1)
			wsum += data[i-period+1+j] * w
			denom += w
		}
		out[i] = wsum / denom
	}
	return out
}

// Gradient computes the numerical gradient of data (numpy.gradient equivalent).
func Gradient(data []float64) []float64 {
	n := len(data)
	if n == 0 {
		return nil
	}
	out := make([]float64, n)
	if n == 1 {
		out[0] = 0
		return out
	}
	out[0] = data[1] - data[0]
	out[n-1] = data[n-1] - data[n-2]
	for i := 1; i < n-1; i++ {
		out[i] = (data[i+1] - data[i-1]) / 2
	}
	return out
}

package indicator

import (
	"math"
	"testing"
)

// testOHLCV creates sample OHLCV data for testing.
func testOHLCV(n int) *OHLCV {
	open := make([]float64, n)
	high := make([]float64, n)
	low := make([]float64, n)
	close := make([]float64, n)
	volume := make([]float64, n)

	// Generate simple trending data
	for i := 0; i < n; i++ {
		base := 100.0 + float64(i)*0.5
		open[i] = base
		high[i] = base + 2.0
		low[i] = base - 1.0
		close[i] = base + 1.0
		volume[i] = 1000000 + float64(i)*10000
	}
	return &OHLCV{Open: open, High: high, Low: low, Close: close, Volume: volume}
}

func TestOHLCVValidate(t *testing.T) {
	data := testOHLCV(50)
	if err := data.Validate(); err != nil {
		t.Fatalf("valid OHLCV should not error: %v", err)
	}

	empty := &OHLCV{}
	if err := empty.Validate(); err == nil {
		t.Fatal("empty OHLCV should error")
	}
}

func TestOHLCVColumn(t *testing.T) {
	data := testOHLCV(10)
	col, err := data.Column("Close")
	if err != nil {
		t.Fatal(err)
	}
	if len(col) != 10 {
		t.Fatalf("expected 10 values, got %d", len(col))
	}

	hl2, err := data.Column("HL2")
	if err != nil {
		t.Fatal(err)
	}
	expected := (data.High[0] + data.Low[0]) / 2
	if hl2[0] != expected {
		t.Fatalf("HL2[0] = %f, expected %f", hl2[0], expected)
	}

	_, err = data.Column("Invalid")
	if err == nil {
		t.Fatal("should error on invalid column")
	}
}

func TestSeriesAt(t *testing.T) {
	s := NewSeries([]float64{10, 20, 30, 40, 50})

	if v := s.At(0); v != 10 {
		t.Fatalf("At(0) = %f, want 10", v)
	}
	if v := s.At(-1); v != 50 {
		t.Fatalf("At(-1) = %f, want 50", v)
	}
	if v := s.At(-2); v != 40 {
		t.Fatalf("At(-2) = %f, want 40", v)
	}
	if v := s.At(10); !math.IsNaN(v) {
		t.Fatalf("At(10) = %f, want NaN", v)
	}
	if v := s.At(-10); !math.IsNaN(v) {
		t.Fatalf("At(-10) = %f, want NaN", v)
	}
}

func TestShift(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5}

	shifted := Shift(data, 2)
	if !math.IsNaN(shifted[0]) || !math.IsNaN(shifted[1]) {
		t.Fatal("first 2 values should be NaN")
	}
	if shifted[2] != 1 || shifted[3] != 2 || shifted[4] != 3 {
		t.Fatal("shifted values incorrect")
	}

	negShifted := Shift(data, -1)
	if negShifted[0] != 2 || negShifted[1] != 3 {
		t.Fatal("negative shift values incorrect")
	}
	if !math.IsNaN(negShifted[4]) {
		t.Fatal("last value should be NaN for negative shift")
	}
}

func TestRollingMaxMin(t *testing.T) {
	data := []float64{5, 3, 8, 1, 9, 2}

	mx := RollingMax(data, 3)
	if math.IsNaN(mx[0]) == false || math.IsNaN(mx[1]) == false {
		// indices 0,1 should be NaN (window not filled)
	}
	if mx[2] != 8 {
		t.Fatalf("RollingMax[2] = %f, want 8", mx[2])
	}

	mn := RollingMin(data, 3)
	if mn[2] != 3 {
		t.Fatalf("RollingMin[2] = %f, want 3", mn[2])
	}
}

func TestSMA(t *testing.T) {
	data := testOHLCV(50)
	result, err := SMA(data, map[string]interface{}{"timeperiod": 10})
	if err != nil {
		t.Fatal(err)
	}
	if len(result) != 50 {
		t.Fatalf("SMA length = %d, want 50", len(result))
	}
	// Last value should be valid
	if math.IsNaN(result[49]) {
		t.Fatal("SMA last value should not be NaN")
	}
}

func TestEMA(t *testing.T) {
	data := testOHLCV(50)
	result, err := EMA(data, map[string]interface{}{"timeperiod": 10})
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(result[49]) {
		t.Fatal("EMA last value should not be NaN")
	}
}

func TestRSI(t *testing.T) {
	data := testOHLCV(50)
	result, err := RSI(data, map[string]interface{}{"timeperiod": 14})
	if err != nil {
		t.Fatal(err)
	}
	// RSI should be between 0 and 100 for valid values
	last := result[49]
	if !math.IsNaN(last) && (last < 0 || last > 100) {
		t.Fatalf("RSI out of range: %f", last)
	}
}

func TestMACDLine(t *testing.T) {
	data := testOHLCV(50)
	result, err := MACD(data, map[string]interface{}{
		"fast_period":   12,
		"slow_period":   26,
		"signal_period": 9,
		"type":          "line",
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(result) != 50 {
		t.Fatalf("MACD length = %d, want 50", len(result))
	}
}

func TestBBANDS(t *testing.T) {
	data := testOHLCV(50)
	upper, err := BBANDS(data, map[string]interface{}{"timeperiod": 20, "std_dev": 2.0, "type": "upper"})
	if err != nil {
		t.Fatal(err)
	}
	lower, err := BBANDS(data, map[string]interface{}{"timeperiod": 20, "std_dev": 2.0, "type": "lower"})
	if err != nil {
		t.Fatal(err)
	}
	// Upper should be >= lower at valid indices
	if !math.IsNaN(upper[49]) && !math.IsNaN(lower[49]) && upper[49] < lower[49] {
		t.Fatalf("upper %f < lower %f", upper[49], lower[49])
	}
}

func TestSupertrend(t *testing.T) {
	data := testOHLCV(100)
	result, err := Supertrend(data, map[string]interface{}{"period": 10, "multiplier": 3.0})
	if err != nil {
		t.Fatal(err)
	}
	// Trend values should be 1 or -1
	for i, v := range result {
		if v != 1 && v != -1 {
			t.Fatalf("Supertrend[%d] = %f, want 1 or -1", i, v)
		}
	}
}

func TestIchimokuConversion(t *testing.T) {
	data := testOHLCV(50)
	result, err := IchimokuConversion(data, map[string]interface{}{"periods": 9})
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(result[49]) {
		t.Fatal("Ichimoku conversion last value should not be NaN")
	}
}

func TestDonchianUpper(t *testing.T) {
	data := testOHLCV(50)
	result, err := DonchianUpper(data, map[string]interface{}{"length": 20})
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(result[49]) {
		t.Fatal("Donchian upper last value should not be NaN")
	}
}

// TestDonchianAllFunctions verifies all five Donchian indicator functions used by
// the UI ConditionBuilder (donchian_upper, donchian_lower, donchian_basis,
// donchian_width, donchian_position) with params matching expr parser output.
func TestDonchianAllFunctions(t *testing.T) {
	data := testOHLCV(100)

	t.Run("upper_length_only", func(t *testing.T) {
		out, err := DonchianUpper(data, map[string]interface{}{"length": 20})
		if err != nil {
			t.Fatal(err)
		}
		if len(out) != data.Len() || math.IsNaN(out[99]) {
			t.Fatalf("DonchianUpper(20): len=%d or last NaN", len(out))
		}
	})
	t.Run("upper_length_and_offset", func(t *testing.T) {
		out, err := DonchianUpper(data, map[string]interface{}{"length": 20, "offset": 1})
		if err != nil {
			t.Fatal(err)
		}
		if len(out) != data.Len() {
			t.Fatalf("DonchianUpper(20,1): len=%d", len(out))
		}
	})
	t.Run("lower_length_only", func(t *testing.T) {
		out, err := DonchianLower(data, map[string]interface{}{"length": 20})
		if err != nil {
			t.Fatal(err)
		}
		if len(out) != data.Len() || math.IsNaN(out[99]) {
			t.Fatalf("DonchianLower(20): len=%d or last NaN", len(out))
		}
	})
	t.Run("lower_length_and_offset", func(t *testing.T) {
		out, err := DonchianLower(data, map[string]interface{}{"length": 20, "offset": 0})
		if err != nil {
			t.Fatal(err)
		}
		if len(out) != data.Len() {
			t.Fatalf("DonchianLower(20,0): len=%d", len(out))
		}
	})
	t.Run("basis_length_only", func(t *testing.T) {
		out, err := DonchianBasis(data, map[string]interface{}{"length": 20})
		if err != nil {
			t.Fatal(err)
		}
		if len(out) != data.Len() || math.IsNaN(out[99]) {
			t.Fatalf("DonchianBasis(20): len=%d or last NaN", len(out))
		}
	})
	t.Run("basis_length_and_offset", func(t *testing.T) {
		out, err := DonchianBasis(data, map[string]interface{}{"length": 14, "offset": 2})
		if err != nil {
			t.Fatal(err)
		}
		if len(out) != data.Len() {
			t.Fatalf("DonchianBasis(14,2): len=%d", len(out))
		}
	})
	t.Run("width_length_only", func(t *testing.T) {
		out, err := DonchianWidth(data, map[string]interface{}{"length": 20})
		if err != nil {
			t.Fatal(err)
		}
		if len(out) != data.Len() || math.IsNaN(out[99]) {
			t.Fatalf("DonchianWidth(20): len=%d or last NaN", len(out))
		}
		// Width should be non-negative (upper - lower)
		if out[99] < 0 {
			t.Errorf("DonchianWidth[99] = %f, want >= 0", out[99])
		}
	})
	t.Run("position_length_only", func(t *testing.T) {
		out, err := DonchianPosition(data, map[string]interface{}{"length": 20})
		if err != nil {
			t.Fatal(err)
		}
		if len(out) != data.Len() || math.IsNaN(out[99]) {
			t.Fatalf("DonchianPosition(20): len=%d or last NaN", len(out))
		}
		// Position should be in [0, 1] (or 0.5 when width is 0)
		if out[99] < 0 || out[99] > 1 {
			t.Errorf("DonchianPosition[99] = %f, want in [0,1]", out[99])
		}
	})
}

// TestPivotSRAllFunctions verifies all three Pivot S/R indicators with params
// matching what the UI sends (keyword: left_bars, right_bars, proximity_threshold, buffer_percent).
func TestPivotSRAllFunctions(t *testing.T) {
	data := testOHLCV(100)
	uiParams := map[string]interface{}{
		"left_bars":            5,
		"right_bars":            5,
		"proximity_threshold": 1.0,
		"buffer_percent":      0.5,
	}

	t.Run("pivot_sr_ui_params", func(t *testing.T) {
		out, err := PivotSR(data, uiParams)
		if err != nil {
			t.Fatal(err)
		}
		if len(out) != data.Len() {
			t.Fatalf("PivotSR len=%d, want %d", len(out), data.Len())
		}
		// Signal values: -3, -2, -1, 0, 1, 2, 3
		for i, v := range out {
			if !math.IsNaN(v) && (v < -3 || v > 3) {
				t.Errorf("PivotSR[%d] = %f, want in [-3,3]", i, v)
			}
		}
	})
	t.Run("pivot_sr_defaults", func(t *testing.T) {
		out, err := PivotSR(data, map[string]interface{}{})
		if err != nil {
			t.Fatal(err)
		}
		if len(out) != data.Len() {
			t.Fatalf("PivotSR() len=%d", len(out))
		}
	})
	t.Run("pivot_sr_crossover_ui_params", func(t *testing.T) {
		params := map[string]interface{}{
			"left_bars":      5,
			"right_bars":    5,
			"buffer_percent": 0.5,
		}
		out, err := PivotSRCrossover(data, params)
		if err != nil {
			t.Fatal(err)
		}
		if len(out) != data.Len() {
			t.Fatalf("PivotSRCrossover len=%d", len(out))
		}
		// -1, 0, 1
		for i, v := range out {
			if !math.IsNaN(v) && v != -1 && v != 0 && v != 1 {
				t.Errorf("PivotSRCrossover[%d] = %f, want -1, 0, or 1", i, v)
			}
		}
	})
	t.Run("pivot_sr_proximity_ui_params", func(t *testing.T) {
		out, err := PivotSRProximity(data, uiParams)
		if err != nil {
			t.Fatal(err)
		}
		if len(out) != data.Len() {
			t.Fatalf("PivotSRProximity len=%d", len(out))
		}
		for i, v := range out {
			if !math.IsNaN(v) && v != -1 && v != 0 && v != 1 {
				t.Errorf("PivotSRProximity[%d] = %f, want -1, 0, or 1", i, v)
			}
		}
	})
}

func TestHMA(t *testing.T) {
	data := testOHLCV(50)
	result, err := HMA(data, map[string]interface{}{"timeperiod": 14})
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(result[49]) {
		t.Fatal("HMA last value should not be NaN")
	}
}

func TestRegistry(t *testing.T) {
	r := NewDefaultRegistry()
	names := r.Names()
	if len(names) == 0 {
		t.Fatal("registry should have indicators")
	}

	fn, ok := r.Get("sma")
	if !ok {
		t.Fatal("SMA should be registered")
	}

	data := testOHLCV(50)
	result, err := fn(data, map[string]interface{}{"timeperiod": 10})
	if err != nil {
		t.Fatal(err)
	}
	if len(result) != 50 {
		t.Fatalf("SMA via registry length = %d, want 50", len(result))
	}

	// Case-insensitive lookup
	_, ok = r.Get("SMA")
	if !ok {
		t.Fatal("registry lookup should be case-insensitive")
	}

	_, ok = r.Get("nonexistent")
	if ok {
		t.Fatal("nonexistent indicator should not be found")
	}
}

func TestGradient(t *testing.T) {
	data := []float64{1, 2, 4, 7, 11}
	grad := Gradient(data)
	// First derivative: [1, 1.5, 2.5, 3.5, 4]
	if grad[0] != 1 {
		t.Fatalf("Gradient[0] = %f, want 1", grad[0])
	}
	if grad[2] != 2.5 {
		t.Fatalf("Gradient[2] = %f, want 2.5", grad[2])
	}
}

func TestWMA(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5}
	result := WMA(data, 3)
	// WMA(3) at index 2: (1*1 + 2*2 + 3*3) / (1+2+3) = 14/6 = 2.333...
	expected := 14.0 / 6.0
	if math.Abs(result[2]-expected) > 1e-10 {
		t.Fatalf("WMA[2] = %f, want %f", result[2], expected)
	}
}

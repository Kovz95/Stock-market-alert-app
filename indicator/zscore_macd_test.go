package indicator

import (
	"math"
	"testing"
)

// =============================================================================
// ZScore Tests
// =============================================================================

func TestZScoreLength(t *testing.T) {
	data := testOHLCV(50)
	result, err := ZScore(data, map[string]interface{}{"lookback": 20})
	if err != nil {
		t.Fatal(err)
	}
	if len(result) != 50 {
		t.Fatalf("ZScore length = %d, want 50", len(result))
	}
}

func TestZScoreLeadingNaN(t *testing.T) {
	// First lookback-1 values must be NaN (window not yet full).
	lookback := 15
	data := testOHLCV(50)
	result, err := ZScore(data, map[string]interface{}{"lookback": lookback})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < lookback-1; i++ {
		if !math.IsNaN(result[i]) {
			t.Errorf("ZScore[%d] = %f, want NaN (lookback=%d)", i, result[i], lookback)
		}
	}
	// First fully-computed value must not be NaN.
	if math.IsNaN(result[lookback-1]) {
		t.Errorf("ZScore[%d] should not be NaN", lookback-1)
	}
}

func TestZScoreLastValueNotNaN(t *testing.T) {
	data := testOHLCV(100)
	result, err := ZScore(data, map[string]interface{}{"lookback": 20})
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(result[99]) {
		t.Fatal("ZScore last value should not be NaN with sufficient data")
	}
}

func TestZScoreConstantSeries(t *testing.T) {
	// All values identical → std dev = 0 → z-score = 0 for every valid bar.
	n := 40
	series := make([]float64, n)
	for i := range series {
		series[i] = 42.0
	}
	data := &OHLCV{Open: series, High: series, Low: series, Close: series, Volume: series}
	result, err := ZScore(data, map[string]interface{}{"lookback": 10})
	if err != nil {
		t.Fatal(err)
	}
	for i := 9; i < n; i++ {
		if math.IsNaN(result[i]) || result[i] != 0 {
			t.Errorf("ZScore[%d] = %v, want 0 for constant series", i, result[i])
		}
	}
}

func TestZScoreMathAccuracy(t *testing.T) {
	// Known series [1, 2, 3, 4, 5] with lookback=3.
	// Each window is a 3-element arithmetic sequence with spacing 1:
	//   window mean = middle element, population std = sqrt(2/3)
	//   z-score of last element = 1 / sqrt(2/3) = sqrt(3/2) ≈ 1.22474487
	series := []float64{1, 2, 3, 4, 5}
	data := &OHLCV{Open: series, High: series, Low: series, Close: series, Volume: series}
	result, err := ZScore(data, map[string]interface{}{"lookback": 3})
	if err != nil {
		t.Fatal(err)
	}
	if len(result) != 5 {
		t.Fatalf("length = %d, want 5", len(result))
	}
	// Indices 0 and 1 must be NaN.
	if !math.IsNaN(result[0]) || !math.IsNaN(result[1]) {
		t.Errorf("ZScore[0]=%f ZScore[1]=%f — both should be NaN", result[0], result[1])
	}
	expected := math.Sqrt(3.0 / 2.0)
	const tol = 1e-9
	for _, i := range []int{2, 3, 4} {
		if math.Abs(result[i]-expected) > tol {
			t.Errorf("ZScore[%d] = %.10f, want %.10f", i, result[i], expected)
		}
	}
}

func TestZScoreValueAtMeanIsZero(t *testing.T) {
	// Window [2, 1, 1.5]: mean=1.5, last value=1.5 → z-score=0.
	series := []float64{1, 2, 1, 2, 1.5}
	data := &OHLCV{Open: series, High: series, Low: series, Close: series, Volume: series}
	result, err := ZScore(data, map[string]interface{}{"lookback": 3})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(result[4]) > 1e-9 {
		t.Errorf("ZScore[4] = %f, want 0 (value equals window mean)", result[4])
	}
}

func TestZScoreNegativeForBelowMean(t *testing.T) {
	// Decreasing series: last value is always below its window mean → z < 0.
	series := []float64{5, 4, 3, 2, 1}
	data := &OHLCV{Open: series, High: series, Low: series, Close: series, Volume: series}
	result, err := ZScore(data, map[string]interface{}{"lookback": 3})
	if err != nil {
		t.Fatal(err)
	}
	expected := -math.Sqrt(3.0 / 2.0)
	const tol = 1e-9
	for _, i := range []int{2, 3, 4} {
		if math.Abs(result[i]-expected) > tol {
			t.Errorf("ZScore[%d] = %.10f, want %.10f", i, result[i], expected)
		}
	}
}

func TestZScorePrecomputedInput(t *testing.T) {
	// Use a constant custom series (std=0 -> z-score=0 everywhere) to distinguish
	// from the linearly-increasing Close, which has a non-zero z-score.
	n := 40
	constant := make([]float64, n)
	for i := range constant {
		constant[i] = 50.0
	}
	data := testOHLCV(n)
	params := map[string]interface{}{
		"_computed_input": constant,
		"lookback":        10,
	}
	result, err := ZScore(data, params)
	if err != nil {
		t.Fatal(err)
	}
	if len(result) != n {
		t.Fatalf("length = %d, want %d", len(result), n)
	}
	for i := 0; i < 9; i++ {
		if !math.IsNaN(result[i]) {
			t.Errorf("ZScore[%d] = %f, want NaN", i, result[i])
		}
	}
	for i := 9; i < n; i++ {
		if result[i] != 0 {
			t.Errorf("ZScore[%d] = %f, want 0 for constant _computed_input", i, result[i])
		}
	}
	closeResult, _ := ZScore(data, map[string]interface{}{"lookback": 10})
	if closeResult[n-1] == 0 {
		t.Fatal("z-score of linearly-increasing Close should be non-zero")
	}
}

func TestZScoreInRegistryAndCallable(t *testing.T) {
	r := NewDefaultRegistry()
	fn, ok := r.Get("zscore")
	if !ok {
		t.Fatal("zscore not found in default registry")
	}
	data := testOHLCV(50)
	series, err := fn(data, map[string]interface{}{"lookback": 10})
	if err != nil {
		t.Fatal(err)
	}
	if len(series) != 50 {
		t.Fatalf("zscore series length = %d, want 50", len(series))
	}
}

// =============================================================================
// MACDLine / MACDSignal / MACDHistogram Tests
// =============================================================================

func TestMACDLineLength(t *testing.T) {
	data := testOHLCV(100)
	result, err := MACDLine(data, map[string]interface{}{"fast": 12, "slow": 26, "signal": 9})
	if err != nil {
		t.Fatal(err)
	}
	if len(result) != 100 {
		t.Fatalf("MACDLine length = %d, want 100", len(result))
	}
}

func TestMACDLineLastValueNotNaN(t *testing.T) {
	data := testOHLCV(100)
	result, err := MACDLine(data, map[string]interface{}{"fast": 12, "slow": 26, "signal": 9})
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(result[99]) {
		t.Fatal("MACDLine last value should not be NaN for 100 bars")
	}
}

func TestMACDSignalLength(t *testing.T) {
	data := testOHLCV(100)
	result, err := MACDSignal(data, map[string]interface{}{"fast": 12, "slow": 26, "signal": 9})
	if err != nil {
		t.Fatal(err)
	}
	if len(result) != 100 {
		t.Fatalf("MACDSignal length = %d, want 100", len(result))
	}
}

func TestMACDSignalLastValueNotNaN(t *testing.T) {
	data := testOHLCV(100)
	result, err := MACDSignal(data, map[string]interface{}{"fast": 12, "slow": 26, "signal": 9})
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(result[99]) {
		t.Fatal("MACDSignal last value should not be NaN")
	}
}

func TestMACDHistogramLength(t *testing.T) {
	data := testOHLCV(100)
	result, err := MACDHistogram(data, map[string]interface{}{"fast": 12, "slow": 26, "signal": 9})
	if err != nil {
		t.Fatal(err)
	}
	if len(result) != 100 {
		t.Fatalf("MACDHistogram length = %d, want 100", len(result))
	}
}

func TestMACDHistogramLastValueNotNaN(t *testing.T) {
	data := testOHLCV(100)
	result, err := MACDHistogram(data, map[string]interface{}{"fast": 12, "slow": 26, "signal": 9})
	if err != nil {
		t.Fatal(err)
	}
	if math.IsNaN(result[99]) {
		t.Fatal("MACDHistogram last value should not be NaN")
	}
}

func TestMACDComponentsConsistency(t *testing.T) {
	// histogram[i] must equal line[i] - signal[i] (talib guarantee).
	data := testOHLCV(100)
	params := map[string]interface{}{"fast": 12, "slow": 26, "signal": 9}
	line, _ := MACDLine(data, params)
	sig, _ := MACDSignal(data, params)
	hist, _ := MACDHistogram(data, params)
	const tol = 1e-9
	for i := 33; i < 100; i++ { // skip early NaN region
		if math.IsNaN(line[i]) || math.IsNaN(sig[i]) || math.IsNaN(hist[i]) {
			continue
		}
		want := line[i] - sig[i]
		if math.Abs(hist[i]-want) > tol {
			t.Errorf("histogram[%d]=%f, line[%d]-signal[%d]=%f (diff=%g)",
				i, hist[i], i, i, want, hist[i]-want)
		}
	}
}

func TestMACDComponentParamAliasesShortVsLong(t *testing.T) {
	// fast/slow/signal must give the same result as fast_period/slow_period/signal_period.
	data := testOHLCV(100)
	short := map[string]interface{}{"fast": 12, "slow": 26, "signal": 9}
	long := map[string]interface{}{"fast_period": 12, "slow_period": 26, "signal_period": 9}
	lineShort, err := MACDLine(data, short)
	if err != nil {
		t.Fatal(err)
	}
	lineLong, err := MACDLine(data, long)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(lineShort[99]-lineLong[99]) > 1e-9 {
		t.Errorf("short params gave %f, long params gave %f — should be equal",
			lineShort[99], lineLong[99])
	}
}

func TestMACDComponentDefaultParams(t *testing.T) {
	// Empty param map must use defaults (12/26/9), same as explicit params.
	data := testOHLCV(100)
	lineDefault, err := MACDLine(data, map[string]interface{}{})
	if err != nil {
		t.Fatal(err)
	}
	lineExplicit, _ := MACDLine(data, map[string]interface{}{"fast": 12, "slow": 26, "signal": 9})
	if math.Abs(lineDefault[99]-lineExplicit[99]) > 1e-9 {
		t.Errorf("default params=%f, explicit params=%f — should be equal",
			lineDefault[99], lineExplicit[99])
	}
}

func TestMACDComponentsInRegistry(t *testing.T) {
	r := NewDefaultRegistry()
	for _, name := range []string{"macd_line", "macd_signal", "macd_histogram"} {
		if _, ok := r.Get(name); !ok {
			t.Errorf("indicator %q not found in default registry", name)
		}
	}
}

func TestMACDComponentsCallableViaRegistry(t *testing.T) {
	r := NewDefaultRegistry()
	data := testOHLCV(100)
	params := map[string]interface{}{"fast": 12, "slow": 26, "signal": 9}
	for _, name := range []string{"macd_line", "macd_signal", "macd_histogram"} {
		fn, _ := r.Get(name)
		series, err := fn(data, params)
		if err != nil {
			t.Errorf("registry call %q: %v", name, err)
			continue
		}
		if len(series) != 100 {
			t.Errorf("registry call %q: series length %d, want 100", name, len(series))
		}
	}
}

func TestMACDLineMatchesMACDWithTypeLine(t *testing.T) {
	// MACDLine must give the same result as the existing MACD(type=line).
	data := testOHLCV(100)
	params := map[string]interface{}{"fast": 12, "slow": 26, "signal": 9}
	newLine, _ := MACDLine(data, params)
	oldLine, _ := MACD(data, map[string]interface{}{
		"fast_period":   12,
		"slow_period":   26,
		"signal_period": 9,
		"type":          "line",
	})
	const tol = 1e-9
	for i := 25; i < 100; i++ {
		if math.IsNaN(newLine[i]) || math.IsNaN(oldLine[i]) {
			continue
		}
		if math.Abs(newLine[i]-oldLine[i]) > tol {
			t.Errorf("MACDLine[%d]=%f differs from MACD(type=line)[%d]=%f",
				i, newLine[i], i, oldLine[i])
		}
	}
}

// =============================================================================
// resolveInput Tests
// =============================================================================

func TestResolveInputPrecomputedSeries(t *testing.T) {
	// _computed_input must take priority over the "input" column name.
	data := testOHLCV(10)
	custom := make([]float64, 10)
	for i := range custom {
		custom[i] = float64(i) * 100 // 0, 100, 200, …, 900
	}
	params := map[string]interface{}{
		"_computed_input": custom,
		"input":           "Open", // should be ignored
	}
	result := resolveInput(data, params)
	if len(result) != 10 {
		t.Fatalf("resolveInput length = %d, want 10", len(result))
	}
	// Spot-check: result[5] should be 500 (from custom), not data.Open[5].
	if result[5] != 500 {
		t.Errorf("resolveInput[5] = %f, want 500 (from _computed_input)", result[5])
	}
	// Confirm data.Open[5] differs so the test is meaningful.
	if data.Open[5] == 500 {
		t.Skip("data.Open[5] coincidentally equals 500 — test inconclusive")
	}
}

func TestResolveInputFallbackToColumnName(t *testing.T) {
	// Without _computed_input, "input" column name is used.
	data := testOHLCV(10)
	params := map[string]interface{}{"input": "High"}
	result := resolveInput(data, params)
	for i := range result {
		if result[i] != data.High[i] {
			t.Errorf("resolveInput[%d] = %f, want High[%d]=%f", i, result[i], i, data.High[i])
		}
	}
}

func TestResolveInputDefaultsToClose(t *testing.T) {
	// Empty params → default to Close column.
	data := testOHLCV(10)
	result := resolveInput(data, map[string]interface{}{})
	for i := range result {
		if result[i] != data.Close[i] {
			t.Errorf("resolveInput[%d] = %f, want Close[%d]=%f", i, result[i], i, data.Close[i])
		}
	}
}

func TestResolveInputInvalidColumnFallsBackToClose(t *testing.T) {
	// An unrecognised column name → falls back to Close.
	data := testOHLCV(10)
	result := resolveInput(data, map[string]interface{}{"input": "NotAColumn"})
	for i := range result {
		if result[i] != data.Close[i] {
			t.Errorf("resolveInput[%d] = %f, want Close fallback = %f", i, result[i], data.Close[i])
		}
	}
}

func TestResolveInputPrecomputedTakesPriorityOverDefault(t *testing.T) {
	// Even when no "input" param is given, _computed_input is still respected.
	data := testOHLCV(10)
	custom := []float64{9, 8, 7, 6, 5, 4, 3, 2, 1, 0}
	result := resolveInput(data, map[string]interface{}{"_computed_input": custom})
	for i := range result {
		if result[i] != custom[i] {
			t.Errorf("resolveInput[%d] = %f, want %f (from _computed_input)", i, result[i], custom[i])
		}
	}
}
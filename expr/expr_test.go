package expr

import (
	"math"
	"testing"

	"stockalert/indicator"
)

// testOHLCV creates sample OHLCV data for testing.
func testOHLCV(n int) *indicator.OHLCV {
	open := make([]float64, n)
	high := make([]float64, n)
	low := make([]float64, n)
	close := make([]float64, n)
	volume := make([]float64, n)

	for i := 0; i < n; i++ {
		base := 100.0 + float64(i)*0.5
		open[i] = base
		high[i] = base + 2.0
		low[i] = base - 1.0
		close[i] = base + 1.0
		volume[i] = 1000000 + float64(i)*10000
	}
	return &indicator.OHLCV{Open: open, High: high, Low: low, Close: close, Volume: volume}
}

// =============================================================================
// Tokenizer Tests
// =============================================================================

func TestTokenizeSimple(t *testing.T) {
	tokens, err := Tokenize("Close[-1]")
	if err != nil {
		t.Fatal(err)
	}
	// Close [ -1 ] EOF
	expected := []TokenType{TokenIdent, TokenLBracket, TokenNumber, TokenRBracket, TokenEOF}
	if len(tokens) != len(expected) {
		t.Fatalf("expected %d tokens, got %d: %v", len(expected), len(tokens), tokens)
	}
	for i, tt := range expected {
		if tokens[i].Type != tt {
			t.Errorf("token %d: expected type %d, got %d (%q)", i, tt, tokens[i].Type, tokens[i].Value)
		}
	}
}

func TestTokenizeComparison(t *testing.T) {
	tokens, err := Tokenize("rsi(14)[-1] < 30")
	if err != nil {
		t.Fatal(err)
	}
	// rsi ( 14 ) [ -1 ] < 30 EOF = 10 tokens
	if len(tokens) != 10 {
		t.Fatalf("expected 10 tokens, got %d: %v", len(tokens), tokens)
	}
	if tokens[8].Type != TokenNumber || tokens[8].Value != "30" {
		t.Errorf("expected number 30, got %v", tokens[8])
	}
}

func TestTokenizeKwargs(t *testing.T) {
	tokens, err := Tokenize("EWO(sma1_length=5, sma2_length=35)[-1]")
	if err != nil {
		t.Fatal(err)
	}
	// Should have: EWO ( sma1_length = 5 , sma2_length = 35 ) [ -1 ] EOF
	if len(tokens) < 10 {
		t.Fatalf("expected at least 10 tokens, got %d: %v", len(tokens), tokens)
	}
}

func TestTokenizeOperators(t *testing.T) {
	tests := []struct {
		input string
		op    string
	}{
		{"a > b", ">"},
		{"a < b", "<"},
		{"a >= b", ">="},
		{"a <= b", "<="},
		{"a == b", "=="},
		{"a != b", "!="},
	}
	for _, tt := range tests {
		tokens, err := Tokenize(tt.input)
		if err != nil {
			t.Errorf("%q: %v", tt.input, err)
			continue
		}
		found := false
		for _, tok := range tokens {
			if tok.Type == TokenOp && tok.Value == tt.op {
				found = true
				break
			}
		}
		if !found {
			t.Errorf("%q: did not find operator %q in tokens %v", tt.input, tt.op, tokens)
		}
	}
}

func TestTokenizeQuotedString(t *testing.T) {
	tokens, err := Tokenize("BBANDS(20, 2.0, type='upper')")
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, tok := range tokens {
		if tok.Type == TokenString && tok.Value == "upper" {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected to find string token 'upper', got: %v", tokens)
	}
}

// =============================================================================
// Parser Tests - ParseOperand
// =============================================================================

func TestParseOperandNumeric(t *testing.T) {
	tests := []struct {
		input string
		num   float64
	}{
		{"150", 150},
		{"-2.5", -2.5},
		{"0", 0},
		{"3.14", 3.14},
	}
	for _, tt := range tests {
		op, err := ParseOperand(tt.input)
		if err != nil {
			t.Errorf("ParseOperand(%q): %v", tt.input, err)
			continue
		}
		if !op.IsNum {
			t.Errorf("ParseOperand(%q): expected IsNum=true", tt.input)
		}
		if op.Number != tt.num {
			t.Errorf("ParseOperand(%q): expected %f, got %f", tt.input, tt.num, op.Number)
		}
	}
}

func TestParseOperandColumn(t *testing.T) {
	tests := []struct {
		input     string
		indicator string
		specifier int
	}{
		{"Close", "close", -1},
		{"Close[-1]", "close", -1},
		{"Close[-2]", "close", -2},
		{"Open[-1]", "open", -1},
		{"High", "high", -1},
		{"Volume", "volume", -1},
	}
	for _, tt := range tests {
		op, err := ParseOperand(tt.input)
		if err != nil {
			t.Errorf("ParseOperand(%q): %v", tt.input, err)
			continue
		}
		if op.IsNum {
			t.Errorf("ParseOperand(%q): expected IsNum=false", tt.input)
		}
		if op.Indicator != tt.indicator {
			t.Errorf("ParseOperand(%q): expected indicator %q, got %q", tt.input, tt.indicator, op.Indicator)
		}
		if op.Specifier != tt.specifier {
			t.Errorf("ParseOperand(%q): expected specifier %d, got %d", tt.input, tt.specifier, op.Specifier)
		}
	}
}

func TestParseOperandIndicator(t *testing.T) {
	tests := []struct {
		input     string
		indicator string
		specifier int
		paramKey  string
		paramVal  interface{}
	}{
		{"rsi(14)", "rsi", -1, "timeperiod", 14},
		{"rsi(14)[-1]", "rsi", -1, "timeperiod", 14},
		{"rsi(14)[-2]", "rsi", -2, "timeperiod", 14},
		{"sma(20)", "sma", -1, "timeperiod", 20},
		{"ema(50)[-1]", "ema", -1, "timeperiod", 50},
	}
	for _, tt := range tests {
		op, err := ParseOperand(tt.input)
		if err != nil {
			t.Errorf("ParseOperand(%q): %v", tt.input, err)
			continue
		}
		if op.Indicator != tt.indicator {
			t.Errorf("ParseOperand(%q): expected indicator %q, got %q", tt.input, tt.indicator, op.Indicator)
		}
		if op.Specifier != tt.specifier {
			t.Errorf("ParseOperand(%q): expected specifier %d, got %d", tt.input, tt.specifier, op.Specifier)
		}
		if val, ok := op.Params[tt.paramKey]; !ok || val != tt.paramVal {
			t.Errorf("ParseOperand(%q): expected param %s=%v, got %v", tt.input, tt.paramKey, tt.paramVal, op.Params)
		}
	}
}

func TestParseOperandKwargs(t *testing.T) {
	op, err := ParseOperand("EWO(sma1_length=5, sma2_length=35)")
	if err != nil {
		t.Fatal(err)
	}
	if op.Indicator != "ewo" {
		t.Fatalf("expected indicator ewo, got %q", op.Indicator)
	}
	if v, ok := op.Params["sma1_length"]; !ok || v != 5 {
		t.Errorf("expected sma1_length=5, got %v", op.Params)
	}
	if v, ok := op.Params["sma2_length"]; !ok || v != 35 {
		t.Errorf("expected sma2_length=35, got %v", op.Params)
	}
}

func TestParseOperandMixedParams(t *testing.T) {
	op, err := ParseOperand("BBANDS(20, 2.0, type='upper')")
	if err != nil {
		t.Fatal(err)
	}
	if op.Indicator != "bbands" {
		t.Fatalf("expected indicator bbands, got %q", op.Indicator)
	}
	if v, ok := op.Params["timeperiod"]; !ok || v != 20 {
		t.Errorf("expected timeperiod=20, got %v", op.Params)
	}
	if v, ok := op.Params["std_dev"]; !ok || v != 2.0 {
		t.Errorf("expected std_dev=2.0, got %v", op.Params)
	}
	if v, ok := op.Params["type"]; !ok || v != "upper" {
		t.Errorf("expected type=upper, got %v", op.Params)
	}
}

func TestParseOperandMACD(t *testing.T) {
	op, err := ParseOperand("MACD(12, 26, 9)")
	if err != nil {
		t.Fatal(err)
	}
	if op.Indicator != "macd" {
		t.Fatalf("expected indicator macd, got %q", op.Indicator)
	}
	if v := op.Params["fast_period"]; v != 12 {
		t.Errorf("expected fast_period=12, got %v", v)
	}
	if v := op.Params["slow_period"]; v != 26 {
		t.Errorf("expected slow_period=26, got %v", v)
	}
	if v := op.Params["signal_period"]; v != 9 {
		t.Errorf("expected signal_period=9, got %v", v)
	}
}

func TestParseOperandSAR(t *testing.T) {
	op, err := ParseOperand("SAR(0.02, 0.2)")
	if err != nil {
		t.Fatal(err)
	}
	if op.Indicator != "sar" {
		t.Fatalf("expected indicator sar, got %q", op.Indicator)
	}
	if v := op.Params["acceleration"]; v != 0.02 {
		t.Errorf("expected acceleration=0.02, got %v", v)
	}
	if v := op.Params["max_acceleration"]; v != 0.2 {
		t.Errorf("expected max_acceleration=0.2, got %v", v)
	}
}

// =============================================================================
// Parser Tests - ParseCondition
// =============================================================================

func TestParseConditionSimple(t *testing.T) {
	tests := []struct {
		input string
		op    string
	}{
		{"Close[-1] > 150", ">"},
		{"rsi(14)[-1] < 30", "<"},
		{"Close[-1] >= sma(20)[-1]", ">="},
		{"Close[-1] <= 200", "<="},
		{"supertrend(10, 3.0)[-1] == 1", "=="},
		{"Close[-1] != 0", "!="},
	}
	for _, tt := range tests {
		comp, err := ParseCondition(tt.input)
		if err != nil {
			t.Errorf("ParseCondition(%q): %v", tt.input, err)
			continue
		}
		if comp.Op != tt.op {
			t.Errorf("ParseCondition(%q): expected op %q, got %q", tt.input, tt.op, comp.Op)
		}
		if comp.Left == nil || comp.Right == nil {
			t.Errorf("ParseCondition(%q): left or right is nil", tt.input)
		}
	}
}

func TestParseConditionRSI(t *testing.T) {
	comp, err := ParseCondition("rsi(14)[-1] < 30")
	if err != nil {
		t.Fatal(err)
	}
	if comp.Left.Indicator != "rsi" {
		t.Errorf("expected left indicator rsi, got %q", comp.Left.Indicator)
	}
	if comp.Left.Params["timeperiod"] != 14 {
		t.Errorf("expected left timeperiod=14, got %v", comp.Left.Params)
	}
	if comp.Right.Number != 30 {
		t.Errorf("expected right number 30, got %f", comp.Right.Number)
	}
}

// =============================================================================
// Evaluator Tests
// =============================================================================

func TestEvalConditionCloseGtValue(t *testing.T) {
	data := testOHLCV(50)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	// Close[-1] should be 100 + 49*0.5 + 1 = 125.5
	// Test: Close[-1] > 100
	result, err := eval.EvalCondition(data, "Close[-1] > 100", nil)
	if err != nil {
		t.Fatal(err)
	}
	if !result {
		t.Fatal("expected Close[-1] > 100 to be true")
	}

	// Test: Close[-1] > 200
	result, err = eval.EvalCondition(data, "Close[-1] > 200", nil)
	if err != nil {
		t.Fatal(err)
	}
	if result {
		t.Fatal("expected Close[-1] > 200 to be false")
	}
}

func TestEvalConditionCloseLtValue(t *testing.T) {
	data := testOHLCV(50)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	result, err := eval.EvalCondition(data, "Close[-1] < 200", nil)
	if err != nil {
		t.Fatal(err)
	}
	if !result {
		t.Fatal("expected Close[-1] < 200 to be true")
	}
}

func TestExpandCatalogCondition(t *testing.T) {
	tests := []struct {
		in   string
		want string
	}{
		// Price
		{"price_above: 0", "close[-1] > 0"},
		{"price_above: 150", "close[-1] > 150"},
		{"price_below: 200", "close[-1] < 200"},
		{"price_equals: 100", "close[-1] == 100"},
		// Moving average
		{"price_above_ma: 20 (SMA)", "close[-1] > sma(20)[-1]"},
		{"price_below_ma: 50 (EMA)", "close[-1] < ema(50)[-1]"},
		{"ma_crossover: 10 > 20", "sma(10)[-1] > sma(20)[-1]"},
		{"ma_crossover: 10 > 20 (SMA)", "sma(10)[-1] > sma(20)[-1]"},
		// RSI
		{"rsi_oversold: 30", "rsi(14)[-1] < 30"},
		{"rsi_overbought: 70", "rsi(14)[-1] > 70"},
		// MACD (no value)
		{"macd_bullish_crossover", "macd(12, 26, 9, type=line)[-1] > macd(12, 26, 9, type=signal)[-1]"},
		{"macd_histogram_positive", "macd(12, 26, 9, type=histogram)[-1] > 0"},
		// Bollinger
		{"price_above_upper_band", "close[-1] > bbands(20, 2.0, type='upper')[-1]"},
		{"price_below_lower_band", "close[-1] < bbands(20, 2.0, type='lower')[-1]"},
		// Volume
		{"volume_above_average: 1.5x", "volume_ratio(20)[-1] > 1.5"},
		{"volume_spike: 2x", "volume_ratio(20)[-1] > 2"},
		// Not catalog
		{"close[-1] > 0", ""},
		{"", ""},
		{"nonsense", ""},
	}
	for _, tt := range tests {
		got := ExpandCatalogCondition(tt.in)
		if got != tt.want {
			t.Errorf("ExpandCatalogCondition(%q) = %q, want %q", tt.in, got, tt.want)
		}
	}
}

func TestEvalConditionCatalogPriceAbove(t *testing.T) {
	data := testOHLCV(50) // Close[-1] = 125.5
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	// Catalog form "price_above: 0" should expand to close[-1] > 0 and match every ticker with positive close
	result, err := eval.EvalCondition(data, "price_above: 0", nil)
	if err != nil {
		t.Fatal(err)
	}
	if !result {
		t.Fatal("expected price_above: 0 (close 125.5) to be true")
	}

	result, err = eval.EvalCondition(data, "price_above: 200", nil)
	if err != nil {
		t.Fatal(err)
	}
	if result {
		t.Fatal("expected price_above: 200 (close 125.5) to be false")
	}
}

func TestEvalConditionCatalogMAAndBollinger(t *testing.T) {
	data := testOHLCV(100) // uptrend: close[-1] > sma(20)[-1] typically
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	// price_above_ma: 20 (SMA) -> close > sma(20)
	result, err := eval.EvalCondition(data, "price_above_ma: 20 (SMA)", nil)
	if err != nil {
		t.Fatal(err)
	}
	if !result {
		t.Fatal("expected price_above_ma: 20 (SMA) in uptrend to be true")
	}

	// price_below_lower_band: close < lower band (in strong uptrend usually false)
	result, err = eval.EvalCondition(data, "price_below_lower_band", nil)
	if err != nil {
		t.Fatal(err)
	}
	// Just ensure no error; result depends on data
	_ = result
}

func TestEvalConditionCatalogPriceCrossAboveMA(t *testing.T) {
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	// Build data where close crosses above EMA(20) on the last bar:
	// bars 0..98 are flat below the EMA, bar 99 jumps above it.
	n := 100
	open := make([]float64, n)
	high := make([]float64, n)
	low := make([]float64, n)
	close := make([]float64, n)
	volume := make([]float64, n)
	for i := 0; i < n; i++ {
		open[i] = 90.0
		high[i] = 91.0
		low[i] = 89.0
		close[i] = 90.0
		volume[i] = 1000000
	}
	// Last bar jumps to 120 — well above any 20-period EMA of 90
	close[n-1] = 120.0
	high[n-1] = 121.0
	crossData := &indicator.OHLCV{Open: open, High: high, Low: low, Close: close, Volume: volume}

	// price_cross_above_ma: 20 (EMA) — the exact condition the user reported
	result, err := eval.EvalCondition(crossData, "price_cross_above_ma: 20 (EMA)", nil)
	if err != nil {
		t.Fatalf("price_cross_above_ma: 20 (EMA) returned error: %v", err)
	}
	if !result {
		t.Fatal("expected price_cross_above_ma: 20 (EMA) to be true after cross-up")
	}

	// price_cross_below_ma: 20 (EMA) should be false in the same scenario
	result, err = eval.EvalCondition(crossData, "price_cross_below_ma: 20 (EMA)", nil)
	if err != nil {
		t.Fatalf("price_cross_below_ma: 20 (EMA) returned error: %v", err)
	}
	if result {
		t.Fatal("expected price_cross_below_ma: 20 (EMA) to be false after cross-up")
	}

	// Also verify SMA variant works
	_, err = eval.EvalCondition(crossData, "price_cross_above_ma: 20 (SMA)", nil)
	if err != nil {
		t.Fatalf("price_cross_above_ma: 20 (SMA) returned error: %v", err)
	}
}

func TestEvalConditionCatalogVolume(t *testing.T) {
	data := testOHLCV(50)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	// volume_above_average: 0.5x -> volume_ratio(20)[-1] > 0.5 (should be true for normal volume)
	result, err := eval.EvalCondition(data, "volume_above_average: 0.5x", nil)
	if err != nil {
		t.Fatal(err)
	}
	if !result {
		t.Fatal("expected volume_ratio > 0.5 for test data")
	}
}

func TestEvalConditionSMAComparison(t *testing.T) {
	data := testOHLCV(50)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	// In an uptrend, Close should be above SMA
	result, err := eval.EvalCondition(data, "Close[-1] > sma(10)[-1]", nil)
	if err != nil {
		t.Fatal(err)
	}
	if !result {
		t.Fatal("expected Close[-1] > sma(10)[-1] to be true in uptrend")
	}
}

func TestEvalConditionRSI(t *testing.T) {
	data := testOHLCV(50)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	// RSI in steady uptrend should be high (> 50)
	result, err := eval.EvalCondition(data, "rsi(14)[-1] > 50", nil)
	if err != nil {
		t.Fatal(err)
	}
	// In a monotonic uptrend, RSI should be very high
	if !result {
		t.Fatal("expected RSI > 50 in steady uptrend")
	}
}

func TestEvalConditionSupertrend(t *testing.T) {
	data := testOHLCV(100)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	// Supertrend in uptrend should be 1
	result, err := eval.EvalCondition(data, "supertrend(10, 3.0)[-1] == 1", nil)
	if err != nil {
		t.Fatal(err)
	}
	if !result {
		t.Fatal("expected supertrend == 1 in steady uptrend")
	}
}

func TestEvalConditionEWO(t *testing.T) {
	data := testOHLCV(100)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	// EWO with default params
	result, err := eval.EvalCondition(data, "EWO(sma1_length=5, sma2_length=35)[-1] > 0", nil)
	if err != nil {
		t.Fatal(err)
	}
	// In uptrend, fast SMA > slow SMA, so EWO > 0
	if !result {
		t.Fatal("expected EWO > 0 in uptrend")
	}
}

func TestEvalConditionBracketIndex(t *testing.T) {
	data := testOHLCV(50)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	// Close[-1] should be greater than Close[-2] in uptrend
	result, err := eval.EvalCondition(data, "Close[-1] > Close[-2]", nil)
	if err != nil {
		t.Fatal(err)
	}
	if !result {
		t.Fatal("expected Close[-1] > Close[-2] in uptrend")
	}
}

func TestEvalConditionEquality(t *testing.T) {
	data := testOHLCV(50)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	// Close[-1] != 0 should always be true for our test data
	result, err := eval.EvalCondition(data, "Close[-1] != 0", nil)
	if err != nil {
		t.Fatal(err)
	}
	if !result {
		t.Fatal("expected Close[-1] != 0")
	}
}

// =============================================================================
// EvalConditionList Tests
// =============================================================================

func TestEvalConditionListAND(t *testing.T) {
	data := testOHLCV(50)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	conditions := []string{
		"Close[-1] > 100",
		"Close[-1] < 200",
	}

	result, err := eval.EvalConditionList(data, conditions, "AND", nil)
	if err != nil {
		t.Fatal(err)
	}
	if !result {
		t.Fatal("expected AND of two true conditions to be true")
	}
}

func TestEvalConditionListOR(t *testing.T) {
	data := testOHLCV(50)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	conditions := []string{
		"Close[-1] > 200", // false
		"Close[-1] < 200", // true
	}

	result, err := eval.EvalConditionList(data, conditions, "OR", nil)
	if err != nil {
		t.Fatal(err)
	}
	if !result {
		t.Fatal("expected OR with one true condition to be true")
	}
}

func TestEvalConditionListComplex(t *testing.T) {
	data := testOHLCV(50)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	conditions := []string{
		"Close[-1] > 100",  // true (condition 1)
		"Close[-1] > 200",  // false (condition 2)
		"Close[-1] < 200",  // true (condition 3)
	}

	// "1 AND (2 OR 3)" = true AND (false OR true) = true AND true = true
	result, err := eval.EvalConditionList(data, conditions, "1 AND (2 OR 3)", nil)
	if err != nil {
		t.Fatal(err)
	}
	if !result {
		t.Fatal("expected '1 AND (2 OR 3)' to be true")
	}

	// "1 AND 2" = true AND false = false
	result, err = eval.EvalConditionList(data, conditions, "1 AND 2", nil)
	if err != nil {
		t.Fatal(err)
	}
	if result {
		t.Fatal("expected '1 AND 2' to be false")
	}
}

func TestEvalConditionListDefault(t *testing.T) {
	data := testOHLCV(50)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	// Empty combination defaults to AND
	conditions := []string{
		"Close[-1] > 100",
		"Close[-1] < 200",
	}

	result, err := eval.EvalConditionList(data, conditions, "", nil)
	if err != nil {
		t.Fatal(err)
	}
	if !result {
		t.Fatal("expected default (AND) to be true")
	}

	// Combination "1" should also default to AND
	result, err = eval.EvalConditionList(data, conditions, "1", nil)
	if err != nil {
		t.Fatal(err)
	}
	if !result {
		t.Fatal("expected combination '1' (AND) to be true")
	}
}

// =============================================================================
// Boolean Expression Tests
// =============================================================================

func TestParseBoolExpr(t *testing.T) {
	tests := []struct {
		expr   string
		result bool
	}{
		{"true", true},
		{"false", false},
		{"true AND true", true},
		{"true AND false", false},
		{"true OR false", true},
		{"false OR false", false},
		{"NOT false", true},
		{"NOT true", false},
		{"(true OR false) AND true", true},
		{"true AND (false OR true)", true},
		{"(true AND false) OR true", true},
		{"NOT (true AND false)", true},
	}
	for _, tt := range tests {
		result, err := parseBoolExpr(tt.expr)
		if err != nil {
			t.Errorf("parseBoolExpr(%q): %v", tt.expr, err)
			continue
		}
		if result != tt.result {
			t.Errorf("parseBoolExpr(%q): expected %v, got %v", tt.expr, tt.result, result)
		}
	}
}

// =============================================================================
// Edge Case Tests
// =============================================================================

func TestParseOperandEmpty(t *testing.T) {
	_, err := ParseOperand("")
	if err == nil {
		t.Fatal("expected error for empty operand")
	}
}

func TestParseConditionNoOperator(t *testing.T) {
	_, err := ParseCondition("Close[-1]")
	if err == nil {
		t.Fatal("expected error for condition without operator")
	}
}

func TestEvalConditionNaN(t *testing.T) {
	// With very short data, some indicators will produce NaN or panic
	data := testOHLCV(5)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	// SMA(20) on 5 bars will be NaN or error (go-talib panics on short data)
	result, _ := eval.EvalCondition(data, "sma(20)[-1] > 100", nil)
	// Either NaN comparison returns false, or error means not-triggered — both should be false
	if result {
		t.Fatal("expected NaN/error comparison to be false")
	}
}

func TestResolveOperandColumn(t *testing.T) {
	data := testOHLCV(10)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	op := &Operand{Indicator: "close", Specifier: -1}
	val, err := eval.resolveOperand(data, op, nil)
	if err != nil {
		t.Fatal(err)
	}
	expected := data.Close[len(data.Close)-1]
	if val != expected {
		t.Fatalf("expected %f, got %f", expected, val)
	}
}

func TestResolveOperandNumeric(t *testing.T) {
	data := testOHLCV(10)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	op := &Operand{IsNum: true, Number: 42.5}
	val, err := eval.resolveOperand(data, op, nil)
	if err != nil {
		t.Fatal(err)
	}
	if val != 42.5 {
		t.Fatalf("expected 42.5, got %f", val)
	}
}

func TestResolveOperandOutOfBounds(t *testing.T) {
	data := testOHLCV(5)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	// Access index that's too far back
	op := &Operand{Indicator: "close", Specifier: -100}
	val, err := eval.resolveOperand(data, op, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !math.IsNaN(val) {
		t.Fatalf("expected NaN for out-of-bounds, got %f", val)
	}
}

// TestEvalIndicatorAliases ensures alias names (bb, psar, harsi) resolve to canonical indicators.
func TestEvalIndicatorAliases(t *testing.T) {
	data := testOHLCV(50)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	// bb -> bbands
	_, err := eval.EvalCondition(data, "bb(20, 2)[-1] > 0", nil)
	if err != nil {
		t.Errorf("alias bb: %v", err)
	}
	// psar -> sar
	_, err = eval.EvalCondition(data, "psar(0.02, 0.2)[-1] > 0", nil)
	if err != nil {
		t.Errorf("alias psar: %v", err)
	}
	// harsi -> harsi_flip
	_, err = eval.EvalCondition(data, "harsi(14)[-1] >= 0", nil)
	if err != nil {
		t.Errorf("alias harsi: %v", err)
	}
}

// TestEvalAdvancedIndicatorConditions ensures key advanced indicators used by the
// scanner can be parsed and evaluated from string conditions without error.
func TestEvalAdvancedIndicatorConditions(t *testing.T) {
	data := testOHLCV(260)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	conditions := []string{
		// Donchian Channels
		"Close[-1] > donchian_upper(20)[-1]",
		"Close[-1] < donchian_lower(20)[-1]",
		"donchian_position(20)[-1] >= 0",
		// Pivot S/R
		"pivot_sr()[-1] >= -3",
		"pivot_sr_proximity()[-1] <= 1",
		"pivot_sr_crossover()[-1] == 0",
		// Kalman ROC Stoch
		"kalman_roc_stoch()[-1] == kalman_roc_stoch()[-1]",
		"kalman_roc_stoch_signal()[-1] >= -1",
		"kalman_roc_stoch_crossover()[-1] <= 1",
		// Ichimoku Cloud
		"ichimoku_cloud_top()[-1] >= ichimoku_cloud_bottom()[-1]",
		"ichimoku_cloud_signal()[-1] >= -1",
		// Trend Magic
		"trend_magic_signal()[-1] >= -1",
		// Supertrend and SAR
		"supertrend(10, 3.0)[-1] == supertrend(10, 3.0)[-1]",
		"sar(0.02, 0.2)[-1] == sar(0.02, 0.2)[-1]",
		// OBV MACD
		"obv_macd()[-1] == obv_macd()[-1]",
		"obv_macd_signal()[-1] == obv_macd_signal()[-1]",
		// MA spread z-score
		"ma_spread_zscore()[-1] == ma_spread_zscore()[-1]",
		// Other core indicators explicitly requested
		"roc(14)[-1] == roc(14)[-1]",
		"willr(14)[-1] == willr(14)[-1]",
		"cci(20)[-1] == cci(20)[-1]",
		"atr(14)[-1] == atr(14)[-1]",
	}

	for _, cond := range conditions {
		if _, err := eval.EvalCondition(data, cond, nil); err != nil {
			t.Errorf("EvalCondition(%q): %v", cond, err)
		}
	}
}

// TestDonchianConditionExpressions verifies that every Donchian condition type
// built by the UI (ConditionBuilder / conditionEntryToExpression) that is a
// single comparison can be parsed and evaluated by the Go expr and indicator
// layers. Note: the UI also emits compound conditions like "(A) and (B)" for
// breakouts; those require EvalConditionList with two separate conditions and
// AND logic, or future support for compound parsing.
func TestDonchianConditionExpressions(t *testing.T) {
	data := testOHLCV(100)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	// Single-comparison expressions only (ParseCondition supports one comparison per string).
	// Compound forms (breakout, basis_cross_*, position_near_middle) are covered by
	// splitting into two conditions and using EvalConditionList with AND.
	tests := []struct {
		name string
		cond string
	}{
		// Channel Lines — price vs upper/lower/basis
		{"price_vs_upper", "Close[-1] > donchian_upper(20)[-1]"},
		{"price_vs_upper_offset", "Close[-1] > donchian_upper(20, 0)[-1]"},
		{"price_vs_lower", "Close[-1] < donchian_lower(20)[-1]"},
		{"price_vs_lower_offset", "Close[-1] < donchian_lower(20, 0)[-1]"},
		{"price_vs_basis", "Close[-1] > donchian_basis(20)[-1]"},
		{"price_vs_basis_offset", "Close[-1] > donchian_basis(20, 0)[-1]"},
		// Channel Position
		{"position_above", "donchian_position(20)[-1] >= 0.8"},
		{"position_below", "donchian_position(20)[-1] <= 0.2"},
		// Channel Width
		{"width_expanding", "donchian_width(20)[-1] > donchian_width(20)[-2]"},
		{"width_contracting", "donchian_width(20)[-1] < donchian_width(20)[-2]"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			comp, err := ParseCondition(tt.cond)
			if err != nil {
				t.Fatalf("ParseCondition(%q): %v", tt.cond, err)
			}
			if comp == nil {
				t.Fatal("ParseCondition returned nil comparison")
			}
			_, err = eval.EvalCondition(data, tt.cond, nil)
			if err != nil {
				t.Errorf("EvalCondition(%q): %v", tt.cond, err)
			}
		})
	}
}

// TestDonchianCompoundViaConditionList verifies that compound Donchian conditions
// (e.g. breakout, basis cross, position near middle) can be evaluated by sending
// two separate conditions and AND combination logic, matching how the backend
// should handle UI-emitted compound expressions when split.
func TestDonchianCompoundViaConditionList(t *testing.T) {
	data := testOHLCV(100)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	// Breakout upper: (Close[-1] > upper[-1]) and (Close[-2] <= upper[-2])
	conds := []string{
		"Close[-1] > donchian_upper(20)[-1]",
		"Close[-2] <= donchian_upper(20)[-2]",
	}
	_, err := eval.EvalConditionList(data, conds, "AND", nil)
	if err != nil {
		t.Errorf("EvalConditionList(breakout_upper): %v", err)
	}

	// Position near middle: (position > 0.4) and (position < 0.6)
	conds2 := []string{
		"donchian_position(20)[-1] > 0.4",
		"donchian_position(20)[-1] < 0.6",
	}
	_, err = eval.EvalConditionList(data, conds2, "AND", nil)
	if err != nil {
		t.Errorf("EvalConditionList(position_near_middle): %v", err)
	}
}

// TestDonchianOperandParsing verifies that Donchian indicator operands (including
// value-only forms like donchian_upper(20)[-1]) parse correctly and that
// length/offset are passed to the indicator. Covers 1-arg and 2-arg forms.
func TestDonchianOperandParsing(t *testing.T) {
	data := testOHLCV(100)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	operands := []string{
		"donchian_upper(20)[-1]",
		"donchian_upper(20, 0)[-1]",
		"donchian_upper(14, 1)[-1]",
		"donchian_lower(20)[-1]",
		"donchian_lower(20, 0)[-1]",
		"donchian_basis(20)[-1]",
		"donchian_basis(20, 0)[-1]",
		"donchian_position(20)[-1]",
		"donchian_position(20, 0)[-1]",
		"donchian_width(20)[-1]",
		"donchian_width(20, 0)[-1]",
	}

	for _, input := range operands {
		op, err := ParseOperand(input)
		if err != nil {
			t.Errorf("ParseOperand(%q): %v", input, err)
			continue
		}
		_, err = eval.computeSeries(data, op, nil)
		if err != nil {
			t.Errorf("computeSeries(%q): %v", input, err)
		}
	}
}

// TestDonchianOffsetParamRemapping verifies that the parser maps the second
// positional argument to "offset" for donchian_upper, donchian_lower, donchian_basis.
func TestDonchianOffsetParamRemapping(t *testing.T) {
	op, err := ParseOperand("donchian_upper(20, 3)[-1]")
	if err != nil {
		t.Fatalf("ParseOperand: %v", err)
	}
	if op.Indicator != "donchian_upper" {
		t.Errorf("indicator = %q, want donchian_upper", op.Indicator)
	}
	if v, ok := op.Params["length"]; !ok {
		t.Error("missing param length")
	} else if n, ok := v.(int); !ok || n != 20 {
		t.Errorf("length = %v (%T), want 20", v, v)
	}
	if v, ok := op.Params["offset"]; !ok {
		t.Error("missing param offset (second positional should be remapped)")
	} else {
		switch n := v.(type) {
		case int:
			if n != 3 {
				t.Errorf("offset = %d, want 3", n)
			}
		case float64:
			if n != 3 {
				t.Errorf("offset = %g, want 3", n)
			}
		default:
			t.Errorf("offset = %v (%T), want 3", v, v)
		}
	}
}

// TestPivotSRConditionExpressions verifies that every Pivot S/R condition type
// built by the UI (conditionEntryToExpression) that is a single comparison can be
// parsed and evaluated. The UI uses keyword args: left_bars, right_bars,
// proximity_threshold, buffer_percent. Note: pivot_sr_near_any and
// pivot_sr_any_crossover use abs(pivot_sr(...)[-1]); the expr parser does not
// support abs() yet, so those are evaluated by splitting into two conditions
// (e.g. pivot_sr()[-1] == 1 or pivot_sr()[-1] == -1) with OR.
func TestPivotSRConditionExpressions(t *testing.T) {
	data := testOHLCV(100)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	// Same keyword args the UI emits (types.ts pivot_sr case).
	kw := "left_bars=5, right_bars=5, proximity_threshold=1.0, buffer_percent=0.5"
	fn := "pivot_sr(" + kw + ")"

	tests := []struct {
		name string
		cond string
	}{
		{"any_signal", fn + "[-1] != 0"},
		{"near_support", fn + "[-1] == 1"},
		{"near_resistance", fn + "[-1] == -1"},
		{"crossover_bullish", fn + "[-1] == 2"},
		{"crossover_bearish", fn + "[-1] == -2"},
		{"broke_strong_support", fn + "[-1] == -3"},
		{"broke_strong_resistance", fn + "[-1] == 3"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			comp, err := ParseCondition(tt.cond)
			if err != nil {
				t.Fatalf("ParseCondition(%q): %v", tt.cond, err)
			}
			if comp == nil {
				t.Fatal("ParseCondition returned nil comparison")
			}
			_, err = eval.EvalCondition(data, tt.cond, nil)
			if err != nil {
				t.Errorf("EvalCondition(%q): %v", tt.cond, err)
			}
		})
	}
}

// TestPivotSROperandParsing verifies that pivot_sr with keyword args (as emitted
// by the UI) parses and that computeSeries runs without error.
func TestPivotSROperandParsing(t *testing.T) {
	data := testOHLCV(100)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	operands := []string{
		"pivot_sr(left_bars=5, right_bars=5, proximity_threshold=1.0, buffer_percent=0.5)[-1]",
		"pivot_sr(left_bars=3, right_bars=3, proximity_threshold=0.5, buffer_percent=0.25)[-1]",
		"pivot_sr()[-1]",
		"pivot_sr_crossover(left_bars=5, right_bars=5, buffer_percent=0.5)[-1]",
		"pivot_sr_proximity(left_bars=5, right_bars=5, proximity_threshold=1.0, buffer_percent=0.5)[-1]",
	}

	for _, input := range operands {
		op, err := ParseOperand(input)
		if err != nil {
			t.Errorf("ParseOperand(%q): %v", input, err)
			continue
		}
		_, err = eval.computeSeries(data, op, nil)
		if err != nil {
			t.Errorf("computeSeries(%q): %v", input, err)
		}
	}
}

// TestPivotSRKeywordParamsPassThrough verifies that keyword params for pivot_sr
// are passed through to the indicator (no remapping; UI sends left_bars, etc.).
func TestPivotSRKeywordParamsPassThrough(t *testing.T) {
	op, err := ParseOperand("pivot_sr(left_bars=7, right_bars=9, proximity_threshold=1.5, buffer_percent=0.6)[-1]")
	if err != nil {
		t.Fatalf("ParseOperand: %v", err)
	}
	if op.Indicator != "pivot_sr" {
		t.Errorf("indicator = %q, want pivot_sr", op.Indicator)
	}
	want := map[string]float64{
		"left_bars":            7,
		"right_bars":            9,
		"proximity_threshold": 1.5,
		"buffer_percent":      0.6,
	}
	for key, wantVal := range want {
		v, ok := op.Params[key]
		if !ok {
			t.Errorf("missing param %q", key)
			continue
		}
		var got float64
		switch x := v.(type) {
		case int:
			got = float64(x)
		case float64:
			got = x
		default:
			t.Errorf("param %q = %v (%T)", key, v, v)
			continue
		}
		if got != wantVal {
			t.Errorf("param %q = %g, want %g", key, got, wantVal)
		}
	}
}

// TestDiagMaSlopeCurveSlope verifies the exact user-reported condition parses
// and evaluates without error, and that the params survive remapping correctly.
func TestDiagMaSlopeCurveSlope(t *testing.T) {
	const cond = "ma_slope_curve_slope(ma_len=200, slope_lookback=3, ma_type='HMA', smooth_type='SMA', smooth_len=2, norm_mode='ATR', atr_len=14, slope_thr=8, curve_thr=0)[-1] > 0"

	// 1. Parse operand – check every named param survives remapping.
	op, err := ParseOperand("ma_slope_curve_slope(ma_len=200, slope_lookback=3, ma_type='HMA', smooth_type='SMA', smooth_len=2, norm_mode='ATR', atr_len=14, slope_thr=8, curve_thr=0)[-1]")
	if err != nil {
		t.Fatalf("ParseOperand: %v", err)
	}
	paramChecks := map[string]interface{}{
		"ma_len":         200,
		"slope_lookback": 3,
		"ma_type":        "HMA",
		"smooth_type":    "SMA",
		"smooth_len":     2,
		"norm_mode":      "ATR",
		"atr_len":        14,
		"slope_thr":      8,
		"curve_thr":      0,
	}
	for k, want := range paramChecks {
		got, ok := op.Params[k]
		if !ok {
			t.Errorf("param %q missing after remapPositionalParams", k)
			continue
		}
		if got != want {
			t.Errorf("param %q = %v (%T), want %v (%T)", k, got, got, want, want)
		}
	}
	if op.Specifier != -1 {
		t.Errorf("specifier = %d, want -1", op.Specifier)
	}

	// 2. Evaluate with enough bars – must not error and must not be NaN.
	data := testOHLCV(300)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	series, err := eval.computeSeries(data, op, nil)
	if err != nil {
		t.Fatalf("computeSeries: %v", err)
	}
	last := series[len(series)-1]
	if math.IsNaN(last) {
		t.Fatalf("ma_slope_curve_slope[-1] is NaN with 300 bars – insufficient warmup compensation")
	}

	// 3. Full EvalCondition must not return an error.
	_, evalErr := eval.EvalCondition(data, cond, nil)
	if evalErr != nil {
		t.Fatalf("EvalCondition returned error: %v", evalErr)
	}
}

// TestEvalAllIndicators ensures every registered indicator can be evaluated with
// default params and sufficient OHLCV data (no panic, no "unknown indicator").
func TestEvalAllIndicators(t *testing.T) {
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)
	// Use enough bars for long lookbacks (e.g. ma_slope_curve_ma default ma_len 200)
	data := testOHLCV(260)

	names := reg.Names()
	if len(names) == 0 {
		t.Fatal("registry has no indicators")
	}

	for _, name := range names {
		op := &Operand{
			Indicator: name,
			Params:    make(map[string]interface{}),
			Specifier: -1,
		}
		series, err := eval.computeSeries(data, op, nil)
		if err != nil {
			t.Errorf("indicator %q: computeSeries failed: %v", name, err)
			continue
		}
		if len(series) != data.Len() {
			t.Errorf("indicator %q: series length %d, want %d", name, len(series), data.Len())
		}
	}
}

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

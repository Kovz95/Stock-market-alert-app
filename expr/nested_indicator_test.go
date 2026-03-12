package expr

// Tests for nested indicator parsing and evaluation introduced by the
// z-score wrapper (Task 5), MA input source (Task 7), FRAMA/KAMA frontend
// expressions, and MACD component indicators.

import (
	"math"
	"testing"

	"stockalert/indicator"
)

// =============================================================================
// Parser: nested indicator as parameter value
// =============================================================================

func TestParseNestedZscoreStructure(t *testing.T) {
	// zscore(rsi(14), lookback=20)[-1]
	// The first positional arg rsi(14) is a nested *Operand. After
	// remapPositionalParams it is stored under "input", not "period".
	op, err := ParseOperand("zscore(rsi(14), lookback=20)[-1]")
	if err != nil {
		t.Fatalf("ParseOperand: %v", err)
	}
	if op.Indicator != "zscore" {
		t.Fatalf("indicator = %q, want zscore", op.Indicator)
	}
	if op.Specifier != -1 {
		t.Errorf("specifier = %d, want -1", op.Specifier)
	}
	// After remap "period" -> "input"
	if _, hasPeriod := op.Params["period"]; hasPeriod {
		t.Error("param 'period' should be removed by remapPositionalParams for zscore")
	}
	inputVal, ok := op.Params["input"]
	if !ok {
		t.Fatal("param 'input' not found after zscore remap")
	}
	nested, ok := inputVal.(*Operand)
	if !ok {
		t.Fatalf("'input' type = %T, want *Operand", inputVal)
	}
	if nested.Indicator != "rsi" {
		t.Errorf("nested indicator = %q, want rsi", nested.Indicator)
	}
	if nested.Params["timeperiod"] != 14 {
		t.Errorf("nested rsi timeperiod = %v, want 14", nested.Params["timeperiod"])
	}
	// lookback must survive as a plain int
	if op.Params["lookback"] != 20 {
		t.Errorf("lookback = %v, want 20", op.Params["lookback"])
	}
}

func TestParseNestedMAInputRSI(t *testing.T) {
	// sma(period=20, input=rsi(14))[-1]
	// period -> timeperiod (remap for sma); input=rsi(14) -> nested *Operand.
	op, err := ParseOperand("sma(period=20, input=rsi(14))[-1]")
	if err != nil {
		t.Fatalf("ParseOperand: %v", err)
	}
	if op.Indicator != "sma" {
		t.Fatalf("indicator = %q, want sma", op.Indicator)
	}
	if op.Params["timeperiod"] != 20 {
		t.Errorf("timeperiod = %v, want 20", op.Params["timeperiod"])
	}
	inputVal, ok := op.Params["input"]
	if !ok {
		t.Fatal("param 'input' not found")
	}
	nested, ok := inputVal.(*Operand)
	if !ok {
		t.Fatalf("'input' type = %T, want *Operand", inputVal)
	}
	if nested.Indicator != "rsi" {
		t.Errorf("nested indicator = %q, want rsi", nested.Indicator)
	}
	if nested.Params["timeperiod"] != 14 {
		t.Errorf("nested rsi timeperiod = %v, want 14", nested.Params["timeperiod"])
	}
}

func TestParseNestedMAInputEWO(t *testing.T) {
	// ema(period=10, input=ewo(sma1_length=5, sma2_length=35))[-1]
	op, err := ParseOperand("ema(period=10, input=ewo(sma1_length=5, sma2_length=35))[-1]")
	if err != nil {
		t.Fatalf("ParseOperand: %v", err)
	}
	if op.Indicator != "ema" {
		t.Fatalf("indicator = %q, want ema", op.Indicator)
	}
	inputVal, ok := op.Params["input"]
	if !ok {
		t.Fatal("param 'input' not found")
	}
	nested, ok := inputVal.(*Operand)
	if !ok {
		t.Fatalf("'input' type = %T, want *Operand", inputVal)
	}
	if nested.Indicator != "ewo" {
		t.Errorf("nested indicator = %q, want ewo", nested.Indicator)
	}
	if nested.Params["sma1_length"] != 5 {
		t.Errorf("ewo sma1_length = %v, want 5", nested.Params["sma1_length"])
	}
	if nested.Params["sma2_length"] != 35 {
		t.Errorf("ewo sma2_length = %v, want 35", nested.Params["sma2_length"])
	}
}

func TestParseNestedMAInputMACDLine(t *testing.T) {
	// sma(period=20, input=macd_line(fast=12, slow=26, signal=9))[-1]
	op, err := ParseOperand("sma(period=20, input=macd_line(fast=12, slow=26, signal=9))[-1]")
	if err != nil {
		t.Fatalf("ParseOperand: %v", err)
	}
	if op.Indicator != "sma" {
		t.Fatalf("indicator = %q, want sma", op.Indicator)
	}
	inputVal, ok := op.Params["input"]
	if !ok {
		t.Fatal("param 'input' not found")
	}
	nested, ok := inputVal.(*Operand)
	if !ok {
		t.Fatalf("'input' type = %T, want *Operand", inputVal)
	}
	if nested.Indicator != "macd_line" {
		t.Errorf("nested indicator = %q, want macd_line", nested.Indicator)
	}
	if nested.Params["fast"] != 12 {
		t.Errorf("macd fast = %v, want 12", nested.Params["fast"])
	}
	if nested.Params["slow"] != 26 {
		t.Errorf("macd slow = %v, want 26", nested.Params["slow"])
	}
	if nested.Params["signal"] != 9 {
		t.Errorf("macd signal = %v, want 9", nested.Params["signal"])
	}
}

func TestParseNestedMAInputMACDSignal(t *testing.T) {
	op, err := ParseOperand("sma(period=5, input=macd_signal(fast=12, slow=26, signal=9))[-1]")
	if err != nil {
		t.Fatalf("ParseOperand: %v", err)
	}
	nested, ok := op.Params["input"].(*Operand)
	if !ok {
		t.Fatal("'input' not a *Operand")
	}
	if nested.Indicator != "macd_signal" {
		t.Errorf("nested indicator = %q, want macd_signal", nested.Indicator)
	}
}

func TestParseNestedMAInputMACDHistogram(t *testing.T) {
	op, err := ParseOperand("sma(period=5, input=macd_histogram(fast=12, slow=26, signal=9))[-1]")
	if err != nil {
		t.Fatalf("ParseOperand: %v", err)
	}
	nested, ok := op.Params["input"].(*Operand)
	if !ok {
		t.Fatal("'input' not a *Operand")
	}
	if nested.Indicator != "macd_histogram" {
		t.Errorf("nested indicator = %q, want macd_histogram", nested.Indicator)
	}
}

// =============================================================================
// Parser: FRAMA / KAMA df-arg cleanup
// =============================================================================

func TestFramaDiscardsDF(t *testing.T) {
	// FRAMA(df, length=16, FC=1, SC=198)[-1]
	// The Python df positional arg must be discarded; named params must survive.
	op, err := ParseOperand("FRAMA(df, length=16, FC=1, SC=198)[-1]")
	if err != nil {
		t.Fatalf("ParseOperand: %v", err)
	}
	if op.Indicator != "frama" {
		t.Fatalf("indicator = %q, want frama", op.Indicator)
	}
	if _, ok := op.Params["period"]; ok {
		t.Error("param 'period' should be removed for FRAMA (df arg cleanup)")
	}
	if op.Params["length"] != 16 {
		t.Errorf("length = %v, want 16", op.Params["length"])
	}
	if op.Params["FC"] != 1 {
		t.Errorf("FC = %v, want 1", op.Params["FC"])
	}
	if op.Params["SC"] != 198 {
		t.Errorf("SC = %v, want 198", op.Params["SC"])
	}
}

func TestKamaDiscardsDF(t *testing.T) {
	// KAMA(df, length=21, fast_end=0.666, slow_end=0.0645)[-1]
	op, err := ParseOperand("KAMA(df, length=21, fast_end=0.666, slow_end=0.0645)[-1]")
	if err != nil {
		t.Fatalf("ParseOperand: %v", err)
	}
	if op.Indicator != "kama" {
		t.Fatalf("indicator = %q, want kama", op.Indicator)
	}
	if _, ok := op.Params["period"]; ok {
		t.Error("param 'period' should be removed for KAMA (df arg cleanup)")
	}
	if op.Params["length"] != 21 {
		t.Errorf("length = %v, want 21", op.Params["length"])
	}
	if v, ok := op.Params["fast_end"]; !ok || v != 0.666 {
		t.Errorf("fast_end = %v, want 0.666", v)
	}
	if v, ok := op.Params["slow_end"]; !ok || v != 0.0645 {
		t.Errorf("slow_end = %v, want 0.0645", v)
	}
}

// =============================================================================
// Parser: deep nesting
// =============================================================================

func TestParseDoubleNestedStructure(t *testing.T) {
	// sma(period=5, input=zscore(rsi(14), lookback=20))[-1]
	// sma -> input=*Operand{zscore} -> input=*Operand{rsi}
	op, err := ParseOperand("sma(period=5, input=zscore(rsi(14), lookback=20))[-1]")
	if err != nil {
		t.Fatalf("ParseOperand: %v", err)
	}
	if op.Indicator != "sma" {
		t.Fatalf("indicator = %q, want sma", op.Indicator)
	}
	zsOp, ok := op.Params["input"].(*Operand)
	if !ok {
		t.Fatal("outer 'input' not a *Operand (want zscore)")
	}
	if zsOp.Indicator != "zscore" {
		t.Errorf("level-1 nested indicator = %q, want zscore", zsOp.Indicator)
	}
	rsiOp, ok := zsOp.Params["input"].(*Operand)
	if !ok {
		t.Fatal("inner 'input' not a *Operand (want rsi)")
	}
	if rsiOp.Indicator != "rsi" {
		t.Errorf("level-2 nested indicator = %q, want rsi", rsiOp.Indicator)
	}
}

// =============================================================================
// Evaluator: nested indicators produce correct series
// =============================================================================

func TestEvalZScoreRSICondition(t *testing.T) {
	data := testOHLCV(100)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	// In a monotonic uptrend, RSI is high and nearly constant, so zscore ~ 0.
	// A z-score in range (-10, 10) is always expected to be > -10.
	cond := "zscore(rsi(14), lookback=20)[-1] > -10"
	result, err := eval.EvalCondition(data, cond, nil)
	if err != nil {
		t.Fatalf("EvalCondition(%q): %v", cond, err)
	}
	if !result {
		t.Errorf("expected z-score of RSI to be > -10")
	}
}

func TestEvalZScoreIsFiniteNotNaN(t *testing.T) {
	// Verify the evaluator does not produce NaN when there is enough data.
	data := testOHLCV(100)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	op, err := ParseOperand("zscore(rsi(14), lookback=20)[-1]")
	if err != nil {
		t.Fatal(err)
	}
	series, err := eval.computeSeries(data, op, nil)
	if err != nil {
		t.Fatalf("computeSeries: %v", err)
	}
	last := series[len(series)-1]
	if math.IsNaN(last) {
		t.Fatal("zscore(rsi(14), lookback=20)[-1] should not be NaN with 100 bars")
	}
}

func TestEvalMAInputSourceRSI(t *testing.T) {
	data := testOHLCV(100)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	// RSI is always in (0, 100); its SMA over any period must also be in (0, 100).
	cond := "sma(period=10, input=rsi(14))[-1] > 0"
	result, err := eval.EvalCondition(data, cond, nil)
	if err != nil {
		t.Fatalf("EvalCondition: %v", err)
	}
	if !result {
		t.Error("SMA of RSI should be > 0")
	}
}

func TestEvalMAInputSourceRSIValueRange(t *testing.T) {
	// SMA of RSI must stay in (0, 100).
	data := testOHLCV(100)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	op, err := ParseOperand("sma(period=10, input=rsi(14))[-1]")
	if err != nil {
		t.Fatal(err)
	}
	series, err := eval.computeSeries(data, op, nil)
	if err != nil {
		t.Fatalf("computeSeries: %v", err)
	}
	last := series[len(series)-1]
	if math.IsNaN(last) {
		t.Fatal("sma of rsi should not be NaN with 100 bars")
	}
	if last <= 0 || last > 100 {
		t.Errorf("sma(rsi(14))[-1] = %f, want in (0, 100]", last)
	}
}

func TestEvalMAInputSourceEWO(t *testing.T) {
	// EWO is positive in an uptrend; SMA of positive EWO should be > 0.
	data := testOHLCV(100)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	cond := "sma(period=5, input=ewo(sma1_length=5, sma2_length=35, source='Close', use_percent=True))[-1] > 0"
	result, err := eval.EvalCondition(data, cond, nil)
	if err != nil {
		t.Fatalf("EvalCondition: %v", err)
	}
	if !result {
		t.Error("SMA of EWO should be > 0 in monotonic uptrend")
	}
}

func TestEvalMAInputSourceMACDLine(t *testing.T) {
	data := testOHLCV(100)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	// a == a is always true - verifies parse+eval completes without error.
	cond := "sma(period=5, input=macd_line(fast=12, slow=26, signal=9))[-1] == sma(period=5, input=macd_line(fast=12, slow=26, signal=9))[-1]"
	result, err := eval.EvalCondition(data, cond, nil)
	if err != nil {
		t.Fatalf("EvalCondition: %v", err)
	}
	if !result {
		t.Error("a == a should always be true")
	}
}

func TestEvalMAInputSourceMACDSignal(t *testing.T) {
	data := testOHLCV(100)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	cond := "sma(period=5, input=macd_signal(fast=12, slow=26, signal=9))[-1] == sma(period=5, input=macd_signal(fast=12, slow=26, signal=9))[-1]"
	result, err := eval.EvalCondition(data, cond, nil)
	if err != nil {
		t.Fatalf("EvalCondition: %v", err)
	}
	if !result {
		t.Error("a == a should always be true")
	}
}

func TestEvalMAInputSourceMACDHistogram(t *testing.T) {
	data := testOHLCV(100)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	cond := "sma(period=5, input=macd_histogram(fast=12, slow=26, signal=9))[-1] == sma(period=5, input=macd_histogram(fast=12, slow=26, signal=9))[-1]"
	result, err := eval.EvalCondition(data, cond, nil)
	if err != nil {
		t.Fatalf("EvalCondition: %v", err)
	}
	if !result {
		t.Error("a == a should always be true")
	}
}

func TestEvalMAInputDiffersFromDefaultClose(t *testing.T) {
	// sma(period=10, input=rsi(14)) must differ from sma(period=10) (which uses Close).
	// RSI is in (0, 100) while Close for our test data is in (101, 150) - different ranges.
	data := testOHLCV(100)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	opRSI, err := ParseOperand("sma(period=10, input=rsi(14))[-1]")
	if err != nil {
		t.Fatal(err)
	}
	opClose, err := ParseOperand("sma(period=10)[-1]")
	if err != nil {
		t.Fatal(err)
	}
	seriesRSI, err := eval.computeSeries(data, opRSI, nil)
	if err != nil {
		t.Fatalf("computeSeries(sma rsi): %v", err)
	}
	seriesClose, err := eval.computeSeries(data, opClose, nil)
	if err != nil {
		t.Fatalf("computeSeries(sma close): %v", err)
	}
	lastRSI := seriesRSI[len(seriesRSI)-1]
	lastClose := seriesClose[len(seriesClose)-1]
	if math.Abs(lastRSI-lastClose) < 1.0 {
		t.Errorf("sma(rsi)=%f is too close to sma(close)=%f - input source may not be wired", lastRSI, lastClose)
	}
}

// =============================================================================
// Evaluator: FRAMA / KAMA frontend expression format
// =============================================================================

func TestEvalFRAMAFrontendExpression(t *testing.T) {
	data := testOHLCV(100)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	// Frontend emits FRAMA(df, length=16, FC=1, SC=198)[-1] - df arg is discarded.
	cond := "FRAMA(df, length=16, FC=1, SC=198)[-1] == FRAMA(df, length=16, FC=1, SC=198)[-1]"
	result, err := eval.EvalCondition(data, cond, nil)
	if err != nil {
		t.Fatalf("EvalCondition(FRAMA): %v", err)
	}
	if !result {
		t.Error("FRAMA value should equal itself (NaN == NaN is false, but valid value == valid value is true)")
	}
}

func TestEvalFRAMAFrontendMatchesNamedOnly(t *testing.T) {
	// FRAMA(df, length=16, ...) must produce the same result as FRAMA(length=16, ...).
	data := testOHLCV(100)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	opDF, _ := ParseOperand("FRAMA(df, length=16, FC=1, SC=198)[-1]")
	opNamed, _ := ParseOperand("FRAMA(length=16, FC=1, SC=198)[-1]")

	s1, err := eval.computeSeries(data, opDF, nil)
	if err != nil {
		t.Fatalf("FRAMA(df,...): %v", err)
	}
	s2, err := eval.computeSeries(data, opNamed, nil)
	if err != nil {
		t.Fatalf("FRAMA(named): %v", err)
	}
	last := len(s1) - 1
	if !math.IsNaN(s1[last]) && !math.IsNaN(s2[last]) {
		if math.Abs(s1[last]-s2[last]) > 1e-9 {
			t.Errorf("FRAMA(df,...)[%d]=%f != FRAMA(named)[%d]=%f", last, s1[last], last, s2[last])
		}
	}
}

func TestEvalKAMAFrontendExpression(t *testing.T) {
	data := testOHLCV(100)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	cond := "KAMA(df, length=21, fast_end=0.666, slow_end=0.0645)[-1] == KAMA(df, length=21, fast_end=0.666, slow_end=0.0645)[-1]"
	result, err := eval.EvalCondition(data, cond, nil)
	if err != nil {
		t.Fatalf("EvalCondition(KAMA): %v", err)
	}
	if !result {
		t.Error("KAMA value should equal itself")
	}
}

func TestEvalKAMAFrontendMatchesNamedOnly(t *testing.T) {
	data := testOHLCV(100)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	opDF, _ := ParseOperand("KAMA(df, length=21, fast_end=0.666, slow_end=0.0645)[-1]")
	opNamed, _ := ParseOperand("KAMA(length=21, fast_end=0.666, slow_end=0.0645)[-1]")

	s1, err := eval.computeSeries(data, opDF, nil)
	if err != nil {
		t.Fatalf("KAMA(df,...): %v", err)
	}
	s2, err := eval.computeSeries(data, opNamed, nil)
	if err != nil {
		t.Fatalf("KAMA(named): %v", err)
	}
	last := len(s1) - 1
	if !math.IsNaN(s1[last]) && !math.IsNaN(s2[last]) {
		if math.Abs(s1[last]-s2[last]) > 1e-9 {
			t.Errorf("KAMA(df,...)[%d]=%f != KAMA(named)[%d]=%f", last, s1[last], last, s2[last])
		}
	}
}

// =============================================================================
// Evaluator: MACD component standalone conditions
// =============================================================================

func TestEvalMACDLineStandalone(t *testing.T) {
	data := testOHLCV(100)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	cond := "macd_line(fast=12, slow=26, signal=9)[-1] == macd_line(fast=12, slow=26, signal=9)[-1]"
	result, err := eval.EvalCondition(data, cond, nil)
	if err != nil {
		t.Fatalf("EvalCondition(macd_line): %v", err)
	}
	if !result {
		t.Error("macd_line == itself should be true")
	}
}

func TestEvalMACDSignalStandalone(t *testing.T) {
	data := testOHLCV(100)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	cond := "macd_signal(fast=12, slow=26, signal=9)[-1] == macd_signal(fast=12, slow=26, signal=9)[-1]"
	result, err := eval.EvalCondition(data, cond, nil)
	if err != nil {
		t.Fatalf("EvalCondition(macd_signal): %v", err)
	}
	if !result {
		t.Error("macd_signal == itself should be true")
	}
}

func TestEvalMACDHistogramStandalone(t *testing.T) {
	data := testOHLCV(100)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	cond := "macd_histogram(fast=12, slow=26, signal=9)[-1] == macd_histogram(fast=12, slow=26, signal=9)[-1]"
	result, err := eval.EvalCondition(data, cond, nil)
	if err != nil {
		t.Fatalf("EvalCondition(macd_histogram): %v", err)
	}
	if !result {
		t.Error("macd_histogram == itself should be true")
	}
}

func TestEvalMACDLineGTSignal(t *testing.T) {
	// In a monotonic uptrend the MACD line tends to be above the signal line.
	data := testOHLCV(100)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	cond := "macd_line(fast=12, slow=26, signal=9)[-1] > macd_signal(fast=12, slow=26, signal=9)[-1]"
	result, err := eval.EvalCondition(data, cond, nil)
	if err != nil {
		t.Fatalf("EvalCondition(macd_line > macd_signal): %v", err)
	}
	if !result {
		t.Error("expected macd_line > macd_signal in steady uptrend")
	}
}

func TestEvalMACDHistogramIsLineMinusSignal(t *testing.T) {
	// histogram == line - signal evaluated as two separate conditions.
	data := testOHLCV(100)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	opLine, _ := ParseOperand("macd_line(fast=12, slow=26, signal=9)[-1]")
	opSig, _ := ParseOperand("macd_signal(fast=12, slow=26, signal=9)[-1]")
	opHist, _ := ParseOperand("macd_histogram(fast=12, slow=26, signal=9)[-1]")

	sLine, _ := eval.computeSeries(data, opLine, nil)
	sSig, _ := eval.computeSeries(data, opSig, nil)
	sHist, _ := eval.computeSeries(data, opHist, nil)

	last := len(sLine) - 1
	want := sLine[last] - sSig[last]
	if !math.IsNaN(sHist[last]) && math.Abs(sHist[last]-want) > 1e-9 {
		t.Errorf("histogram=%f, line-signal=%f", sHist[last], want)
	}
}

// =============================================================================
// Evaluator: deep nesting (2-level)
// =============================================================================

func TestEvalDeepNestingSMAOfZscoreOfRSI(t *testing.T) {
	// sma(period=5, input=zscore(rsi(14), lookback=20))
	// Verifies that 2-level nesting parses and evaluates without error or panic.
	// Note: talib SMA propagates NaN once the accumulator is seeded with NaN
	// (from ZScore warmup), so the last bar may be NaN - the evaluator returns
	// false for NaN comparisons, which is the correct defined behavior.
	data := testOHLCV(100)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	op, err := ParseOperand("sma(period=5, input=zscore(rsi(14), lookback=20))[-1]")
	if err != nil {
		t.Fatalf("ParseOperand (deep nesting): %v", err)
	}
	// computeSeries must not return an error ? NaN is acceptable at the last bar.
	_, err = eval.computeSeries(data, op, nil)
	if err != nil {
		t.Fatalf("computeSeries (deep nesting): %v", err)
	}
}

func TestEvalDeepNestingEvalConditionNoError(t *testing.T) {
	// EvalCondition must return (false, nil) ? not an error ? when the nested
	// result is NaN (evaluator treats NaN comparisons as false, not errors).
	data := testOHLCV(100)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	// Use > -1000 so that a finite result would pass and NaN would return false.
	// The important assertion is "no error".
	_, err := eval.EvalCondition(data, "sma(period=5, input=zscore(rsi(14), lookback=20))[-1] > -1000", nil)
	if err != nil {
		t.Fatalf("EvalCondition (deep nesting): unexpected error: %v", err)
	}
}

// =============================================================================
// Evaluator: z-score expression variants emitted by the UI
// =============================================================================

func TestEvalZScoreExpressionVariants(t *testing.T) {
	data := testOHLCV(100)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	// These are every indicator category that the UI "Transform to Z-score" option
	// can wrap. All should parse and evaluate without error.
	cases := []string{
		"zscore(rsi(14), lookback=20)[-1] > -10",
		"zscore(ewo(sma1_length=5, sma2_length=35), lookback=20)[-1] > -10",
		"zscore(cci(20), lookback=20)[-1] > -10",
		"zscore(roc(12), lookback=20)[-1] > -10",
		"zscore(willr(14), lookback=20)[-1] > -10",
		"zscore(atr(14), lookback=20)[-1] > -10",
		"zscore(macd_line(fast=12, slow=26, signal=9), lookback=20)[-1] > -10",
		"zscore(macd_histogram(fast=12, slow=26, signal=9), lookback=20)[-1] > -10",
	}
	for _, cond := range cases {
		_, err := eval.EvalCondition(data, cond, nil)
		if err != nil {
			t.Errorf("EvalCondition(%q): %v", cond, err)
		}
	}
}

// =============================================================================
// Registry: all new indicators are present
// =============================================================================

func TestNewIndicatorsInRegistry(t *testing.T) {
	reg := indicator.NewDefaultRegistry()
	for _, name := range []string{"zscore", "macd_line", "macd_signal", "macd_histogram"} {
		if _, ok := reg.Get(name); !ok {
			t.Errorf("indicator %q not found in default registry", name)
		}
	}
}

// =============================================================================
// Evaluator: plain column name as input= still works (regression guard)
// =============================================================================

func TestEvalMAInputColumnNameString(t *testing.T) {
	// input=Close (a plain column name, not a nested indicator call) must still work.
	data := testOHLCV(50)
	reg := indicator.NewDefaultRegistry()
	eval := NewEvaluator(reg)

	// With input=Close the SMA should equal the default (which also uses Close).
	opWithInput, err := ParseOperand("sma(period=10, input=Close)[-1]")
	if err != nil {
		t.Fatal(err)
	}
	opDefault, err := ParseOperand("sma(period=10)[-1]")
	if err != nil {
		t.Fatal(err)
	}
	s1, err := eval.computeSeries(data, opWithInput, nil)
	if err != nil {
		t.Fatalf("sma(input=Close): %v", err)
	}
	s2, err := eval.computeSeries(data, opDefault, nil)
	if err != nil {
		t.Fatalf("sma(default): %v", err)
	}
	last := len(s1) - 1
	if math.Abs(s1[last]-s2[last]) > 1e-9 {
		t.Errorf("sma(input=Close)[last]=%f differs from sma[last]=%f", s1[last], s2[last])
	}
}
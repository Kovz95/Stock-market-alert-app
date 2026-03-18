package expr

import (
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"

	"stockalert/indicator"
)

// Evaluator evaluates parsed expressions against OHLCV data.
type Evaluator struct {
	registry *indicator.Registry
}

// NewEvaluator creates an Evaluator with the given indicator registry.
func NewEvaluator(registry *indicator.Registry) *Evaluator {
	return &Evaluator{registry: registry}
}

// EvalCondition evaluates a single condition string against OHLCV data.
// Returns true if the condition is met, false otherwise.
//
// Supports both expression form ("close[-1] > 0", "RSI(14)[-1] < 30") and
// catalog form ("price_above: 0", "price_below: 100") from the scanner/alerts UI.
// Also handles compound boolean expressions with "and"/"or" such as those
// produced by catalog expansions like price_cross_above_ma.
func (e *Evaluator) EvalCondition(data *indicator.OHLCV, condition string, ctx map[string]interface{}) (bool, error) {
	if expanded := ExpandCatalogCondition(condition); expanded != "" {
		condition = expanded
	}

	// Handle compound "and"/"or" expressions (e.g. "(A) and (B)") before
	// attempting single-comparison parsing, since ParseCondition only handles
	// one comparison operator at the top level.
	if result, handled, err := e.tryEvalCompound(data, condition, ctx); handled {
		return result, err
	}

	comp, err := ParseCondition(condition)
	if err != nil {
		return false, fmt.Errorf("parse condition: %w", err)
	}
	return e.evalComparison(data, comp, ctx)
}

// tryEvalCompound attempts to evaluate a compound boolean expression like
// "(A) and (B)" or "(A) or (B)". Returns (result, true, err) when a top-level
// "and"/"or" keyword is found, or (false, false, nil) when none is present.
func (e *Evaluator) tryEvalCompound(data *indicator.OHLCV, condition string, ctx map[string]interface{}) (bool, bool, error) {
	op, left, right := splitOnBoolKeyword(condition)
	if op == "" {
		return false, false, nil
	}

	leftResult, err := e.EvalCondition(data, left, ctx)
	if err != nil {
		return false, true, fmt.Errorf("left side of %s: %w", op, err)
	}

	// Short-circuit evaluation
	if op == "and" && !leftResult {
		return false, true, nil
	}
	if op == "or" && leftResult {
		return true, true, nil
	}

	rightResult, err := e.EvalCondition(data, right, ctx)
	if err != nil {
		return false, true, fmt.Errorf("right side of %s: %w", op, err)
	}

	if op == "and" {
		return leftResult && rightResult, true, nil
	}
	return leftResult || rightResult, true, nil
}

// splitOnBoolKeyword scans s for the first " and " or " or " keyword at
// paren/bracket depth 0. Returns the keyword and the trimmed left/right
// sub-expressions with their outer parentheses stripped, or ("", "", "")
// if no top-level boolean keyword is found.
func splitOnBoolKeyword(s string) (op, left, right string) {
	depth := 0
	lower := strings.ToLower(s)
	for i := 0; i < len(s); i++ {
		ch := s[i]
		if ch == '(' || ch == '[' {
			depth++
		} else if ch == ')' || ch == ']' {
			depth--
		} else if depth == 0 {
			if strings.HasPrefix(lower[i:], " and ") {
				l := stripOuterParens(strings.TrimSpace(s[:i]))
				r := stripOuterParens(strings.TrimSpace(s[i+5:]))
				return "and", l, r
			}
			if strings.HasPrefix(lower[i:], " or ") {
				l := stripOuterParens(strings.TrimSpace(s[:i]))
				r := stripOuterParens(strings.TrimSpace(s[i+4:]))
				return "or", l, r
			}
			// Support "&" as an alias for "and" and "|" as an alias for "or"
			if ch == '&' {
				l := stripOuterParens(strings.TrimSpace(s[:i]))
				r := stripOuterParens(strings.TrimSpace(s[i+1:]))
				return "and", l, r
			}
			if ch == '|' {
				l := stripOuterParens(strings.TrimSpace(s[:i]))
				r := stripOuterParens(strings.TrimSpace(s[i+1:]))
				return "or", l, r
			}
		}
	}
	return "", "", ""
}

// stripOuterParens removes one layer of surrounding parentheses from s if they
// are balanced and wrap the entire expression.
func stripOuterParens(s string) string {
	s = strings.TrimSpace(s)
	if len(s) < 2 || s[0] != '(' || s[len(s)-1] != ')' {
		return s
	}
	inner := s[1 : len(s)-1]
	depth := 0
	for _, ch := range inner {
		if ch == '(' {
			depth++
		} else if ch == ')' {
			depth--
			if depth < 0 {
				return s
			}
		}
	}
	if depth != 0 {
		return s
	}
	return inner
}

// EvalConditionList evaluates a list of conditions with combination logic.
//
// Combination examples:
//   - "" or "AND" — all must be true
//   - "OR" — any must be true
//   - "1 AND 2" — conditions 1 and 2
//   - "1 AND (2 OR 3)" — complex boolean
func (e *Evaluator) EvalConditionList(
	data *indicator.OHLCV,
	conditions []string,
	combination string,
	ctx map[string]interface{},
) (bool, error) {
	if len(conditions) == 0 {
		return false, nil
	}

	// Evaluate each condition
	results := make([]bool, len(conditions))
	for i, cond := range conditions {
		result, err := e.EvalCondition(data, cond, ctx)
		if err != nil {
			// Failed conditions evaluate to false (matching Python behavior)
			results[i] = false
		} else {
			results[i] = result
		}
	}

	// Single condition
	if len(results) == 1 {
		return results[0], nil
	}

	return combineBoolResults(results, combination), nil
}

// evalComparison evaluates a single comparison against OHLCV data.
func (e *Evaluator) evalComparison(data *indicator.OHLCV, comp *Comparison, ctx map[string]interface{}) (bool, error) {
	val1, err := e.resolveOperand(data, comp.Left, ctx)
	if err != nil {
		return false, fmt.Errorf("resolve left operand: %w", err)
	}

	val2, err := e.resolveOperand(data, comp.Right, ctx)
	if err != nil {
		return false, fmt.Errorf("resolve right operand: %w", err)
	}

	// NaN values result in false comparison (matching Python behavior)
	if math.IsNaN(val1) || math.IsNaN(val2) {
		return false, nil
	}

	return compareValues(val1, comp.Op, val2)
}

// resolveOperand computes the scalar value for an operand.
func (e *Evaluator) resolveOperand(data *indicator.OHLCV, op *Operand, ctx map[string]interface{}) (float64, error) {
	if op.IsNum {
		return op.Number, nil
	}

	// Check if it's a price column
	series, err := e.computeSeries(data, op, ctx)
	if err != nil {
		return math.NaN(), err
	}

	// Apply specifier to get scalar value
	s := indicator.NewSeries(series)
	val := s.At(op.Specifier)
	if math.IsNaN(val) {
		return math.NaN(), nil
	}
	return val, nil
}

// indicatorAliases maps alternative names to the canonical registry name.
var indicatorAliases = map[string]string{
	"bb":    "bbands",
	"psar":  "sar",
	"harsi": "harsi_flip",
}

// resolveIndicatorName returns the canonical indicator name for registry lookup.
func resolveIndicatorName(name string) string {
	if canonical, ok := indicatorAliases[name]; ok {
		return canonical
	}
	return name
}

// computeSeries computes the indicator or column series for an operand.
func (e *Evaluator) computeSeries(data *indicator.OHLCV, op *Operand, ctx map[string]interface{}) ([]float64, error) {
	name := resolveIndicatorName(op.Indicator)

	// Check if it's a price column
	switch name {
	case "close", "open", "high", "low", "volume":
		col, err := data.Column(capitalize(name))
		if err != nil {
			return nil, err
		}
		return col, nil
	}

	// Look up indicator in registry
	fn, ok := e.registry.Get(name)
	if !ok {
		return nil, fmt.Errorf("unknown indicator %q", name)
	}

	// Build params map - merge operand params with context
	params := make(map[string]interface{})
	for k, v := range op.Params {
		params[k] = v
	}
	if ctx != nil {
		// Only pass through specific context keys like "ticker"
		if ticker, ok := ctx["ticker"]; ok {
			params["ticker"] = ticker
		}
	}

	// Resolve any nested *Operand param values to computed series.
	// e.g. zscore(rsi(14), lookback=20) stores rsi operand under "input";
	// sma(period=20, input=rsi(14)) stores rsi operand under "input".
	for key, val := range params {
		if nestedOp, ok := val.(*Operand); ok {
			nestedSeries, err := e.computeSeries(data, nestedOp, ctx)
			if err != nil {
				return nil, fmt.Errorf("nested indicator param %q: %w", key, err)
			}
			params["_computed_"+key] = nestedSeries
			delete(params, key)
		}
	}

	// Catch panics from go-talib when data is too short for the period
	var (
		series  []float64
		callErr error
	)
	func() {
		defer func() {
			if r := recover(); r != nil {
				callErr = fmt.Errorf("indicator %q panicked (data may be too short): %v", name, r)
			}
		}()
		series, callErr = fn(data, params)
	}()
	if callErr != nil {
		return nil, callErr
	}
	return series, nil
}

// compareValues performs the comparison operation.
func compareValues(a float64, op string, b float64) (bool, error) {
	switch op {
	case ">":
		return a > b, nil
	case "<":
		return a < b, nil
	case ">=":
		return a >= b, nil
	case "<=":
		return a <= b, nil
	case "==":
		return a == b, nil
	case "!=":
		return a != b, nil
	default:
		return false, fmt.Errorf("unknown operator %q", op)
	}
}

// combineBoolResults applies combination logic to a list of boolean results.
func combineBoolResults(results []bool, combination string) bool {
	combination = strings.TrimSpace(combination)

	// Normalize
	if combination == "" || combination == "1" {
		combination = "AND"
	}

	upper := strings.ToUpper(combination)

	// Simple AND/OR
	if upper == "AND" {
		for _, r := range results {
			if !r {
				return false
			}
		}
		return true
	}
	if upper == "OR" {
		for _, r := range results {
			if r {
				return true
			}
		}
		return false
	}

	// Complex expression like "1 AND (2 OR 3)"
	return evalBoolExpr(results, combination)
}

// evalBoolExpr evaluates a boolean expression like "1 AND (2 OR 3)"
// where numbers refer to 1-based condition indices.
func evalBoolExpr(results []bool, expr string) bool {
	// Replace condition numbers with their boolean values
	// Use word boundaries to avoid "10" matching "1"
	replaced := expr
	for i := len(results); i >= 1; i-- {
		re := regexp.MustCompile(`\b` + strconv.Itoa(i) + `\b`)
		val := "false"
		if results[i-1] {
			val = "true"
		}
		replaced = re.ReplaceAllString(replaced, val)
	}

	// Parse and evaluate the boolean expression
	result, err := parseBoolExpr(replaced)
	if err != nil {
		// Fallback to AND
		for _, r := range results {
			if !r {
				return false
			}
		}
		return true
	}
	return result
}

// parseBoolExpr evaluates a simple boolean expression containing true/false, AND, OR, NOT, and parentheses.
func parseBoolExpr(expr string) (bool, error) {
	expr = strings.TrimSpace(expr)
	tokens := tokenizeBoolExpr(expr)
	bp := &boolParser{tokens: tokens, pos: 0}
	result, err := bp.parseOr()
	if err != nil {
		return false, err
	}
	return result, nil
}

type boolParser struct {
	tokens []string
	pos    int
}

func (bp *boolParser) peek() string {
	if bp.pos < len(bp.tokens) {
		return bp.tokens[bp.pos]
	}
	return ""
}

func (bp *boolParser) advance() string {
	t := bp.peek()
	bp.pos++
	return t
}

func (bp *boolParser) parseOr() (bool, error) {
	left, err := bp.parseAnd()
	if err != nil {
		return false, err
	}
	for strings.ToUpper(bp.peek()) == "OR" {
		bp.advance()
		right, err := bp.parseAnd()
		if err != nil {
			return false, err
		}
		left = left || right
	}
	return left, nil
}

func (bp *boolParser) parseAnd() (bool, error) {
	left, err := bp.parseNot()
	if err != nil {
		return false, err
	}
	for strings.ToUpper(bp.peek()) == "AND" {
		bp.advance()
		right, err := bp.parseNot()
		if err != nil {
			return false, err
		}
		left = left && right
	}
	return left, nil
}

func (bp *boolParser) parseNot() (bool, error) {
	if strings.ToUpper(bp.peek()) == "NOT" {
		bp.advance()
		val, err := bp.parsePrimary()
		if err != nil {
			return false, err
		}
		return !val, nil
	}
	return bp.parsePrimary()
}

func (bp *boolParser) parsePrimary() (bool, error) {
	t := bp.peek()
	tUpper := strings.ToUpper(t)

	if tUpper == "TRUE" {
		bp.advance()
		return true, nil
	}
	if tUpper == "FALSE" {
		bp.advance()
		return false, nil
	}
	if t == "(" {
		bp.advance()
		result, err := bp.parseOr()
		if err != nil {
			return false, err
		}
		if bp.peek() == ")" {
			bp.advance()
		}
		return result, nil
	}
	return false, fmt.Errorf("unexpected token in boolean expression: %q", t)
}

// tokenizeBoolExpr splits a boolean expression into tokens.
func tokenizeBoolExpr(expr string) []string {
	var tokens []string
	expr = strings.TrimSpace(expr)
	i := 0
	for i < len(expr) {
		ch := expr[i]
		if ch == ' ' || ch == '\t' {
			i++
			continue
		}
		if ch == '(' || ch == ')' {
			tokens = append(tokens, string(ch))
			i++
			continue
		}
		// Read word
		j := i
		for j < len(expr) && expr[j] != ' ' && expr[j] != '(' && expr[j] != ')' {
			j++
		}
		tokens = append(tokens, expr[i:j])
		i = j
	}
	return tokens
}

// capitalize returns the string with the first letter uppercased.
func capitalize(s string) string {
	if s == "" {
		return s
	}
	return strings.ToUpper(s[:1]) + s[1:]
}

package expr

import "fmt"

// Operand represents one side of a comparison: either a numeric literal
// or an indicator reference with optional parameters and bracket index.
//
// Examples:
//   - Numeric: {IsNum: true, Number: 30}
//   - Column:  {Indicator: "Close", Specifier: -1}
//   - Indicator: {Indicator: "rsi", Params: {"timeperiod": 14}, Specifier: -1}
type Operand struct {
	// IsNum is true when this operand is a numeric literal.
	IsNum bool

	// Number holds the numeric value when IsNum is true.
	Number float64

	// Indicator is the indicator or column name (lowercase).
	// Empty when IsNum is true.
	Indicator string

	// Params holds parsed keyword arguments from the expression.
	// For "rsi(14)" this is {"period": 14}.
	// For "EWO(sma1_length=5, sma2_length=35)" this is {"sma1_length": 5, "sma2_length": 35}.
	Params map[string]interface{}

	// Specifier is the bracket index (e.g., -1 from "[-1]").
	// Default is -1 (last bar). Set explicitly from the expression.
	Specifier int
}

// String returns a human-readable representation of the operand.
func (o *Operand) String() string {
	if o.IsNum {
		return fmt.Sprintf("%.4f", o.Number)
	}
	s := o.Indicator
	if len(o.Params) > 0 {
		s += fmt.Sprintf("(%v)", o.Params)
	}
	s += fmt.Sprintf("[%d]", o.Specifier)
	return s
}

// Comparison represents a single binary comparison: left op right.
//
// Example: "RSI(14)[-1] < 30" becomes:
//
//	Comparison{Left: rsi_operand, Op: "<", Right: num_30_operand}
type Comparison struct {
	Left  *Operand
	Op    string // One of: ">", "<", ">=", "<=", "==", "!="
	Right *Operand
}

// String returns a human-readable representation.
func (c *Comparison) String() string {
	return fmt.Sprintf("%s %s %s", c.Left, c.Op, c.Right)
}

// CombinedCondition represents a list of comparisons with combination logic.
//
// The Combination field describes how to combine the boolean results:
//   - "" or "AND" — all conditions must be true
//   - "OR" — any condition must be true
//   - "1 AND (2 OR 3)" — complex boolean expression referencing conditions by 1-based index
type CombinedCondition struct {
	Conditions  []*Comparison
	Combination string
}

// ComparisonOps is the set of valid comparison operators, ordered so that
// multi-character operators are checked before single-character ones.
var ComparisonOps = []string{">=", "<=", "==", "!=", ">", "<"}

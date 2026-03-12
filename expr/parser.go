package expr

import (
	"fmt"
	"strconv"
	"strings"
)

// ParseOperand parses an operand expression string into an Operand struct.
// This is the Go equivalent of Python's ind_to_dict().
//
// Examples:
//   - "150"           → {IsNum: true, Number: 150}
//   - "-2.5"          → {IsNum: true, Number: -2.5}
//   - "Close"         → {Indicator: "close", Specifier: -1}
//   - "Close[-1]"     → {Indicator: "close", Specifier: -1}
//   - "rsi(14)"       → {Indicator: "rsi", Params: {"period": 14}, Specifier: -1}
//   - "rsi(14)[-2]"   → {Indicator: "rsi", Params: {"period": 14}, Specifier: -2}
//   - "EWO(sma1_length=5, sma2_length=35)[-1]" → {Indicator: "ewo", Params: {...}, Specifier: -1}
func ParseOperand(input string) (*Operand, error) {
	input = strings.TrimSpace(input)
	if input == "" {
		return nil, fmt.Errorf("empty operand")
	}

	tokens, err := Tokenize(input)
	if err != nil {
		return nil, fmt.Errorf("tokenize operand %q: %w", input, err)
	}

	p := &parser{tokens: tokens, pos: 0}
	op, err := p.parseOperand()
	if err != nil {
		return nil, fmt.Errorf("parse operand %q: %w", input, err)
	}
	return op, nil
}

// ParseCondition parses a condition string like "RSI(14)[-1] < 30"
// into a Comparison struct. This is the Go equivalent of simplify_conditions().
func ParseCondition(input string) (*Comparison, error) {
	input = strings.TrimSpace(input)
	if input == "" {
		return nil, fmt.Errorf("empty condition")
	}

	// Find the comparison operator outside of parentheses and brackets
	op, leftStr, rightStr := splitOnOperator(input)
	if op == "" {
		return nil, fmt.Errorf("no comparison operator found in %q", input)
	}

	left, err := ParseOperand(leftStr)
	if err != nil {
		return nil, fmt.Errorf("parse left side: %w", err)
	}

	right, err := ParseOperand(rightStr)
	if err != nil {
		return nil, fmt.Errorf("parse right side: %w", err)
	}

	return &Comparison{Left: left, Op: op, Right: right}, nil
}

// ParseConditionList parses a list of condition strings and a combination
// expression into a CombinedCondition.
func ParseConditionList(conditions []string, combination string) (*CombinedCondition, error) {
	if len(conditions) == 0 {
		return nil, fmt.Errorf("no conditions provided")
	}

	parsed := make([]*Comparison, 0, len(conditions))
	for i, cond := range conditions {
		c, err := ParseCondition(cond)
		if err != nil {
			return nil, fmt.Errorf("condition %d (%q): %w", i+1, cond, err)
		}
		parsed = append(parsed, c)
	}

	return &CombinedCondition{
		Conditions:  parsed,
		Combination: combination,
	}, nil
}

// splitOnOperator finds the comparison operator at the top level (outside
// parens and brackets) and splits the expression into left, operator, right.
func splitOnOperator(expr string) (op, left, right string) {
	parenDepth := 0
	bracketDepth := 0

	for _, checkOp := range ComparisonOps {
		for i := 0; i <= len(expr)-len(checkOp); i++ {
			ch := expr[i]
			if ch == '(' {
				parenDepth++
			} else if ch == ')' {
				parenDepth--
			} else if ch == '[' {
				bracketDepth++
			} else if ch == ']' {
				bracketDepth--
			} else if parenDepth == 0 && bracketDepth == 0 {
				if expr[i:i+len(checkOp)] == checkOp {
					// Make sure we're not matching part of a larger operator
					// e.g., don't match ">" in ">="
					if len(checkOp) == 1 && i+1 < len(expr) && expr[i+1] == '=' {
						continue
					}
					// Don't match "=" in "==" or keyword "="
					if checkOp == "=" {
						continue
					}
					return checkOp,
						strings.TrimSpace(expr[:i]),
						strings.TrimSpace(expr[i+len(checkOp):])
				}
			}
		}
		// Reset depths for next operator search
		parenDepth = 0
		bracketDepth = 0
	}

	return "", "", ""
}

// parser is a recursive descent parser over a token stream.
type parser struct {
	tokens []Token
	pos    int
}

func (p *parser) peek() Token {
	if p.pos < len(p.tokens) {
		return p.tokens[p.pos]
	}
	return Token{Type: TokenEOF}
}

func (p *parser) advance() Token {
	t := p.peek()
	if p.pos < len(p.tokens) {
		p.pos++
	}
	return t
}

func (p *parser) expect(tt TokenType) (Token, error) {
	t := p.advance()
	if t.Type != tt {
		return t, fmt.Errorf("expected token type %d, got %d (%q) at pos %d", tt, t.Type, t.Value, t.Pos)
	}
	return t, nil
}

// parseOperand parses an operand from the current position.
func (p *parser) parseOperand() (*Operand, error) {
	t := p.peek()

	switch t.Type {
	case TokenNumber:
		return p.parseNumber()
	case TokenIdent:
		return p.parseIndicator()
	case TokenEOF:
		return nil, fmt.Errorf("unexpected end of input")
	default:
		// Try to handle negative number
		if t.Type == TokenOp && t.Value == "-" {
			return p.parseNumber()
		}
		return nil, fmt.Errorf("unexpected token %q at position %d", t.Value, t.Pos)
	}
}

// parseNumber parses a numeric literal.
func (p *parser) parseNumber() (*Operand, error) {
	t := p.advance()
	val, err := strconv.ParseFloat(t.Value, 64)
	if err != nil {
		return nil, fmt.Errorf("invalid number %q: %w", t.Value, err)
	}
	return &Operand{
		IsNum:     true,
		Number:    val,
		Specifier: -1,
	}, nil
}

// parseIndicator parses an indicator reference, possibly with params and bracket index.
func (p *parser) parseIndicator() (*Operand, error) {
	nameToken := p.advance()
	name := nameToken.Value

	op := &Operand{
		Indicator: strings.ToLower(name),
		Params:    make(map[string]interface{}),
		Specifier: -1,
	}

	// Check for parenthesized parameters: name(...)
	if p.peek().Type == TokenLParen {
		p.advance() // consume '('
		if err := p.parseParams(op); err != nil {
			return nil, err
		}
	}

	// Check for bracket specifier: [n]
	if p.peek().Type == TokenLBracket {
		p.advance() // consume '['
		numToken, err := p.expect(TokenNumber)
		if err != nil {
			return nil, fmt.Errorf("expected number in bracket index: %w", err)
		}
		idx, err := strconv.Atoi(numToken.Value)
		if err != nil {
			return nil, fmt.Errorf("invalid bracket index %q: %w", numToken.Value, err)
		}
		op.Specifier = idx
		if _, err := p.expect(TokenRBracket); err != nil {
			return nil, fmt.Errorf("expected ']': %w", err)
		}
	}

	return op, nil
}

// parseParams parses the parameter list inside parentheses.
// Handles: (14), (14, "Close"), (sma1_length=5, sma2_length=35), (20, 2.0, type='upper')
func (p *parser) parseParams(op *Operand) error {
	// Handle empty params
	if p.peek().Type == TokenRParen {
		p.advance()
		return nil
	}

	positionalIdx := 0
	for {
		// Check if this is a key=value pair
		if p.peek().Type == TokenIdent && p.lookAhead(1).Type == TokenEquals {
			// Key = Value
			key := p.advance().Value // ident
			p.advance()              // =
			val, err := p.parseParamValue()
			if err != nil {
				return fmt.Errorf("parameter %q: %w", key, err)
			}
			op.Params[key] = val
		} else {
			// Positional parameter
			val, err := p.parseParamValue()
			if err != nil {
				return fmt.Errorf("positional param %d: %w", positionalIdx, err)
			}
			// Map first positional to "period", second depends on indicator
			if positionalIdx == 0 {
				op.Params["period"] = val
			} else {
				op.Params[fmt.Sprintf("_pos_%d", positionalIdx)] = val
			}
			positionalIdx++
		}

		// Check for comma or closing paren
		if p.peek().Type == TokenComma {
			p.advance() // consume ','
			continue
		}
		break
	}

	if p.peek().Type == TokenRParen {
		p.advance()
	}

	// Remap positional params for specific indicators
	remapPositionalParams(op)

	return nil
}

// parseParamValue parses a single parameter value (number, string, identifier,
// or nested indicator call like rsi(14) or ewo(sma1_length=5, sma2_length=35)).
func (p *parser) parseParamValue() (interface{}, error) {
	t := p.peek()
	switch t.Type {
	case TokenNumber:
		p.advance()
		return parseNumericValue(t.Value)
	case TokenString:
		p.advance()
		return t.Value, nil
	case TokenIdent:
		// Check if this is a nested indicator call like rsi(14) or ewo(sma1_length=5)
		if p.lookAhead(1).Type == TokenLParen {
			return p.parseIndicator()
		}
		p.advance()
		// Check for boolean literals
		lower := strings.ToLower(t.Value)
		if lower == "true" {
			return true, nil
		}
		if lower == "false" {
			return false, nil
		}
		return t.Value, nil
	default:
		return nil, fmt.Errorf("unexpected token %q in parameter value", t.Value)
	}
}

func (p *parser) lookAhead(offset int) Token {
	idx := p.pos + offset
	if idx < len(p.tokens) {
		return p.tokens[idx]
	}
	return Token{Type: TokenEOF}
}

// parseNumericValue converts a string to int or float64.
func parseNumericValue(s string) (interface{}, error) {
	if !strings.Contains(s, ".") {
		if v, err := strconv.Atoi(s); err == nil {
			return v, nil
		}
	}
	if v, err := strconv.ParseFloat(s, 64); err == nil {
		return v, nil
	}
	return nil, fmt.Errorf("not a number: %q", s)
}

// remapPositionalParams converts positional params to named params for specific indicators.
func remapPositionalParams(op *Operand) {
	ind := op.Indicator

	// Map "period" to indicator-specific param names
	if period, ok := op.Params["period"]; ok {
		switch ind {
		case "sma", "ema", "hma", "rsi", "roc", "atr", "cci", "willr", "volume_ratio", "my_smoothed_rsi", "adx", "mfi", "mom", "stoch_rsi_k", "stoch_rsi_d", "plus_di", "minus_di", "natr", "linear_reg_slope", "linear_reg", "aroon_osc", "cmo", "stddev":
			op.Params["timeperiod"] = period
			delete(op.Params, "period")
		}
		if strings.HasPrefix(ind, "ma_slope_curve") {
			op.Params["ma_len"] = period
			delete(op.Params, "period")
		}
		if ind == "donchian_upper" || ind == "donchian_lower" || ind == "donchian_basis" || ind == "donchian_width" || ind == "donchian_position" {
			op.Params["length"] = period
			delete(op.Params, "period")
		}
	}

	// Handle multi-positional indicators
	switch ind {
	case "macd":
		if v, ok := op.Params["period"]; ok {
			op.Params["fast_period"] = v
			delete(op.Params, "period")
		}
		if v, ok := op.Params["_pos_1"]; ok {
			op.Params["slow_period"] = v
			delete(op.Params, "_pos_1")
		}
		if v, ok := op.Params["_pos_2"]; ok {
			op.Params["signal_period"] = v
			delete(op.Params, "_pos_2")
		}
	case "bbands", "bb":
		if v, ok := op.Params["period"]; ok {
			op.Params["timeperiod"] = v
			delete(op.Params, "period")
		}
		if v, ok := op.Params["_pos_1"]; ok {
			op.Params["std_dev"] = v
			delete(op.Params, "_pos_1")
		}
		if v, ok := op.Params["_pos_2"]; ok {
			op.Params["type"] = v
			delete(op.Params, "_pos_2")
		}
	case "sar", "psar":
		if v, ok := op.Params["period"]; ok {
			op.Params["acceleration"] = v
			delete(op.Params, "period")
		}
		if v, ok := op.Params["_pos_1"]; ok {
			op.Params["max_acceleration"] = v
			delete(op.Params, "_pos_1")
		}
	case "supertrend", "supertrend_upper", "supertrend_lower":
		if v, ok := op.Params["_pos_1"]; ok {
			op.Params["multiplier"] = v
			delete(op.Params, "_pos_1")
		}
	case "harsi_flip", "harsi":
		if v, ok := op.Params["period"]; ok {
			op.Params["timeperiod"] = v
			delete(op.Params, "period")
		}
		if v, ok := op.Params["_pos_1"]; ok {
			op.Params["smoothing"] = v
			delete(op.Params, "_pos_1")
		}
	case "stoch_k", "stoch_d":
		if v, ok := op.Params["period"]; ok {
			op.Params["fast_k_period"] = v
			delete(op.Params, "period")
		}
		if v, ok := op.Params["_pos_1"]; ok {
			op.Params["slow_k_period"] = v
			delete(op.Params, "_pos_1")
		}
		if v, ok := op.Params["_pos_2"]; ok {
			op.Params["slow_d_period"] = v
			delete(op.Params, "_pos_2")
		}
	case "stddev":
		if v, ok := op.Params["_pos_1"]; ok {
			op.Params["nb_dev"] = v
			delete(op.Params, "_pos_1")
		}
	case "donchian_upper", "donchian_lower", "donchian_basis":
		if v, ok := op.Params["_pos_1"]; ok {
			op.Params["offset"] = v
			delete(op.Params, "_pos_1")
		}
	case "frama", "kama":
		// Discard the Python 'df' dataframe positional arg that the frontend emits
		delete(op.Params, "period")
	case "zscore":
		// First positional arg is the inner indicator (already parsed as *Operand);
		// remap it to the "input" key so resolveInput can find it.
		if v, ok := op.Params["period"]; ok {
			op.Params["input"] = v
			delete(op.Params, "period")
		}
	}

	// Clean up remaining positional params
	for k := range op.Params {
		if strings.HasPrefix(k, "_pos_") {
			delete(op.Params, k)
		}
	}
}

package expr

import (
	"fmt"
	"strconv"
	"strings"
	"unicode"
)

// TokenType identifies the kind of token.
type TokenType int

const (
	TokenIdent     TokenType = iota // Identifier: Close, RSI, sma, etc.
	TokenNumber                     // Numeric literal: 30, -2.5, 150
	TokenLParen                     // (
	TokenRParen                     // )
	TokenLBracket                   // [
	TokenRBracket                   // ]
	TokenComma                      // ,
	TokenEquals                     // = (for keyword args)
	TokenOp                         // Comparison operator: >, <, >=, <=, ==, !=
	TokenString                     // Quoted string: 'Close', "HL2"
	TokenDot                        // . (for method-style access, if any)
	TokenEOF                        // End of input
)

// Token represents a single lexical token.
type Token struct {
	Type  TokenType
	Value string
	Pos   int // byte position in original input
}

// String returns a debug representation.
func (t Token) String() string {
	return fmt.Sprintf("{%d %q @%d}", t.Type, t.Value, t.Pos)
}

// Tokenize converts an expression string into a slice of tokens.
//
// Examples of tokenizable expressions:
//   - "Close[-1]"
//   - "rsi(14)[-1] < 30"
//   - "EWO(sma1_length=5, sma2_length=35)[-1] > 0"
//   - "BBANDS(20, 2.0, type='upper')[-1]"
func Tokenize(input string) ([]Token, error) {
	var tokens []Token
	i := 0
	n := len(input)

	for i < n {
		ch := input[i]

		// Skip whitespace
		if unicode.IsSpace(rune(ch)) {
			i++
			continue
		}

		// Single character tokens
		switch ch {
		case '(':
			tokens = append(tokens, Token{Type: TokenLParen, Value: "(", Pos: i})
			i++
			continue
		case ')':
			tokens = append(tokens, Token{Type: TokenRParen, Value: ")", Pos: i})
			i++
			continue
		case '[':
			tokens = append(tokens, Token{Type: TokenLBracket, Value: "[", Pos: i})
			i++
			continue
		case ']':
			tokens = append(tokens, Token{Type: TokenRBracket, Value: "]", Pos: i})
			i++
			continue
		case ',':
			tokens = append(tokens, Token{Type: TokenComma, Value: ",", Pos: i})
			i++
			continue
		case '.':
			tokens = append(tokens, Token{Type: TokenDot, Value: ".", Pos: i})
			i++
			continue
		}

		// Comparison operators (must check >= <= == != before > < =)
		if ch == '>' || ch == '<' || ch == '!' || ch == '=' {
			op, length := scanOperator(input, i)
			if op != "" {
				tokens = append(tokens, Token{Type: TokenOp, Value: op, Pos: i})
				i += length
				continue
			}
			// Single '=' is for keyword args
			if ch == '=' {
				tokens = append(tokens, Token{Type: TokenEquals, Value: "=", Pos: i})
				i++
				continue
			}
		}

		// Quoted strings
		if ch == '\'' || ch == '"' {
			str, length, err := scanString(input, i)
			if err != nil {
				return nil, err
			}
			tokens = append(tokens, Token{Type: TokenString, Value: str, Pos: i})
			i += length
			continue
		}

		// Numbers (including negative numbers)
		if isDigitStart(input, i, tokens) {
			num, length := scanNumber(input, i)
			tokens = append(tokens, Token{Type: TokenNumber, Value: num, Pos: i})
			i += length
			continue
		}

		// Identifiers (letters, digits, underscores)
		if isIdentStart(ch) {
			ident, length := scanIdent(input, i)
			tokens = append(tokens, Token{Type: TokenIdent, Value: ident, Pos: i})
			i += length
			continue
		}

		// Negative sign before number
		if ch == '-' {
			// Check if next char is a digit (negative number)
			if i+1 < n && (input[i+1] >= '0' && input[i+1] <= '9') {
				num, length := scanNumber(input, i)
				tokens = append(tokens, Token{Type: TokenNumber, Value: num, Pos: i})
				i += length
				continue
			}
		}

		return nil, fmt.Errorf("unexpected character %q at position %d", ch, i)
	}

	tokens = append(tokens, Token{Type: TokenEOF, Value: "", Pos: n})
	return tokens, nil
}

// scanOperator checks for multi-character comparison operators.
func scanOperator(input string, pos int) (string, int) {
	if pos+1 < len(input) {
		two := input[pos : pos+2]
		switch two {
		case ">=", "<=", "==", "!=":
			return two, 2
		}
	}
	ch := input[pos]
	if ch == '>' || ch == '<' {
		return string(ch), 1
	}
	return "", 0
}

// scanString scans a quoted string starting at pos.
func scanString(input string, pos int) (string, int, error) {
	quote := input[pos]
	i := pos + 1
	for i < len(input) {
		if input[i] == byte(quote) {
			// Return the content without quotes
			return input[pos+1 : i], i - pos + 1, nil
		}
		if input[i] == '\\' && i+1 < len(input) {
			i += 2 // skip escape
			continue
		}
		i++
	}
	return "", 0, fmt.Errorf("unterminated string starting at position %d", pos)
}

// scanNumber scans a numeric literal (int or float, possibly negative).
func scanNumber(input string, pos int) (string, int) {
	i := pos
	if i < len(input) && input[i] == '-' {
		i++
	}
	for i < len(input) && input[i] >= '0' && input[i] <= '9' {
		i++
	}
	if i < len(input) && input[i] == '.' {
		i++
		for i < len(input) && input[i] >= '0' && input[i] <= '9' {
			i++
		}
	}
	return input[pos:i], i - pos
}

// scanIdent scans an identifier (letters, digits, underscores).
func scanIdent(input string, pos int) (string, int) {
	i := pos
	for i < len(input) && isIdentChar(input[i]) {
		i++
	}
	return input[pos:i], i - pos
}

// isDigitStart checks if position starts a number (not a negative identifier).
func isDigitStart(input string, pos int, prevTokens []Token) bool {
	ch := input[pos]
	if ch >= '0' && ch <= '9' {
		return true
	}
	// Negative number: '-' followed by digit, and previous token is operator/bracket/comma/start
	if ch == '-' && pos+1 < len(input) && input[pos+1] >= '0' && input[pos+1] <= '9' {
		if len(prevTokens) == 0 {
			return true
		}
		last := prevTokens[len(prevTokens)-1]
		// Negative number after operator, open bracket, comma, or equals
		return last.Type == TokenOp || last.Type == TokenLBracket ||
			last.Type == TokenComma || last.Type == TokenEquals ||
			last.Type == TokenLParen
	}
	return false
}

func isIdentStart(ch byte) bool {
	return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch == '_'
}

func isIdentChar(ch byte) bool {
	return isIdentStart(ch) || (ch >= '0' && ch <= '9')
}

// ParseNumber parses a string as a number (int or float64).
func ParseNumber(s string) (interface{}, error) {
	s = strings.TrimSpace(s)
	// Try int first
	if !strings.Contains(s, ".") {
		if v, err := strconv.ParseInt(s, 10, 64); err == nil {
			return int(v), nil
		}
	}
	// Try float
	if v, err := strconv.ParseFloat(s, 64); err == nil {
		return v, nil
	}
	return nil, fmt.Errorf("not a number: %q", s)
}

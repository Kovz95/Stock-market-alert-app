package alert

import (
	"encoding/json"
	"fmt"
	"strings"

	"stockalert/expr"
	"stockalert/indicator"
)

// Evaluator wraps expr.Evaluator to provide alert-specific evaluation logic.
// It handles extracting conditions from the alert JSON, normalizing timeframes,
// and evaluating conditions against OHLCV price data.
type Evaluator struct {
	exprEval *expr.Evaluator
}

// NewEvaluator creates an Evaluator with the given indicator registry.
func NewEvaluator(registry *indicator.Registry) *Evaluator {
	return &Evaluator{
		exprEval: expr.NewEvaluator(registry),
	}
}

// EvaluateAlert evaluates an alert's conditions against the given OHLCV data.
// conditions is the raw JSON from the alerts.conditions column.
// combinationLogic is the combination expression (e.g., "AND", "1 AND (2 OR 3)").
// ticker is passed through to the expression evaluator context.
//
// Returns true if the alert is triggered, false otherwise.
func (e *Evaluator) EvaluateAlert(
	data *indicator.OHLCV,
	conditions []byte,
	combinationLogic string,
	ticker string,
) (bool, error) {
	condStrs, err := ExtractConditions(conditions)
	if err != nil {
		return false, fmt.Errorf("extract conditions: %w", err)
	}
	if len(condStrs) == 0 {
		return false, nil
	}

	ctx := map[string]interface{}{
		"ticker": ticker,
	}

	return e.exprEval.EvalConditionList(data, condStrs, combinationLogic, ctx)
}

// ExtractConditions parses the conditions JSON column into a flat list of
// condition strings. The JSON may be:
//   - A JSON array of strings: ["RSI(14)[-1] < 30", "Close[-1] > 100"]
//   - A JSON array of objects with a "conditions" key: [{"conditions": "RSI(14)[-1] < 30"}]
//   - A JSON array of arrays: [["RSI(14)[-1] < 30"]]
//   - A JSON string (single condition): "RSI(14)[-1] < 30"
func ExtractConditions(raw []byte) ([]string, error) {
	if len(raw) == 0 {
		return nil, nil
	}

	raw = trimBytes(raw)
	if len(raw) == 0 {
		return nil, nil
	}

	// Try as JSON array first (most common)
	if raw[0] == '[' {
		var items []json.RawMessage
		if err := json.Unmarshal(raw, &items); err != nil {
			return nil, fmt.Errorf("unmarshal conditions array: %w", err)
		}

		var result []string
		for _, item := range items {
			item = trimBytes(item)
			if len(item) == 0 {
				continue
			}

			switch item[0] {
			case '"':
				// String element
				var s string
				if err := json.Unmarshal(item, &s); err == nil && s != "" {
					result = append(result, s)
				}

			case '{':
				// Object with "conditions" key
				var obj map[string]interface{}
				if err := json.Unmarshal(item, &obj); err == nil {
					if cond, ok := obj["conditions"]; ok {
						if s, ok := cond.(string); ok && s != "" {
							result = append(result, s)
						}
					}
				}

			case '[':
				// Nested array - take first string
				var arr []interface{}
				if err := json.Unmarshal(item, &arr); err == nil && len(arr) > 0 {
					if s, ok := arr[0].(string); ok && s != "" {
						result = append(result, s)
					}
				}
			}
		}
		return result, nil
	}

	// Try as single JSON string
	if raw[0] == '"' {
		var s string
		if err := json.Unmarshal(raw, &s); err == nil && s != "" {
			return []string{s}, nil
		}
	}

	return nil, fmt.Errorf("unsupported conditions format: %s", string(raw[:min(len(raw), 50)]))
}

// NormalizeTimeframe normalizes timeframe strings to one of: "daily", "weekly", "hourly".
func NormalizeTimeframe(tf string) string {
	switch strings.ToLower(strings.TrimSpace(tf)) {
	case "1wk", "1w", "weekly", "week":
		return "weekly"
	case "1h", "1hr", "hourly", "hour":
		return "hourly"
	default:
		return "daily"
	}
}

func trimBytes(b []byte) []byte {
	s := strings.TrimSpace(string(b))
	return []byte(s)
}

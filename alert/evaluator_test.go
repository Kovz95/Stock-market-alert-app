package alert

import (
	"encoding/json"
	"testing"
)

func TestExtractConditions_condition1Format(t *testing.T) {
	// Web Add Alert and Streamlit send: { "condition_1": { "conditions": ["..."], "combination_logic": "AND" } }
	payload := map[string]interface{}{
		"condition_1": map[string]interface{}{
			"conditions":         []string{"price_above: 150", "rsi(14)[-1] < 30"},
			"combination_logic": "AND",
		},
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}

	got, err := ExtractConditions(raw)
	if err != nil {
		t.Fatalf("ExtractConditions: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 conditions, got %d", len(got))
	}
	if got[0] != "price_above: 150" {
		t.Errorf("conditions[0] = %q, want price_above: 150", got[0])
	}
	if got[1] != "rsi(14)[-1] < 30" {
		t.Errorf("conditions[1] = %q, want rsi(14)[-1] < 30", got[1])
	}
}

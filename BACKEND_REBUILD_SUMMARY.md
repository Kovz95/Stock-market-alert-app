# Backend.py Rebuild Summary

## Overview
The original `backend.py` source code was missing. I rebuilt the missing functions by analyzing the codebase usage patterns and context. The module now works independently without requiring the legacy bytecode file.

## Rebuilt Functions

### 1. `simplify_conditions(cond)`
**Purpose**: Parse a condition string into a structured dictionary for evaluation.

**Input**: Condition string like `"Close[-1] > sma(20)[-1]"` or `"rsi(14)[-1] < 30"`

**Output**: Dictionary with structure:
```python
{
    "ind1": parsed_left_side,    # Left side indicator/value
    "ind2": parsed_right_side,   # Right side indicator/value
    "comparison": operator        # ">", "<", ">=", "<=", "==", "!="
}
```

**Usage Example**:
```python
result = simplify_conditions("Close[-1] > 150")
# Returns: {
#     "ind1": {"isNum": False, "ind": "Close", "specifier": -1, ...},
#     "ind2": {"isNum": True, "number": 150, ...},
#     "comparison": ">"
# }
```

---

### 2. `indicator_calculation(df, ind, vals=None, debug_mode=False)`
**Purpose**: Calculate an indicator value from a parsed indicator dictionary.

**Parameters**:
- `df`: DataFrame with price data (OHLC + Volume)
- `ind`: Indicator dictionary from `ind_to_dict()` or `simplify_conditions()`
- `vals`: Optional values dictionary (legacy parameter)
- `debug_mode`: Enable debug output

**Returns**: Indicator value (scalar or Series)

**Usage Example**:
```python
ind = ind_to_dict("sma(20)[-1]")
value = indicator_calculation(df, ind)
# Returns the SMA(20) value at index -1
```

---

### 3. `evaluate_expression_list(df, exps, combination="1")`
**Purpose**: Evaluate multiple expressions with combination logic.

**Parameters**:
- `df`: DataFrame with price data
- `exps`: List of condition strings
- `combination`: How to combine conditions:
  - `"AND"` or `"1"` - all conditions must be True
  - `"OR"` - any condition must be True
  - `"1 AND 2"` - conditions 1 and 2 must be True
  - `"(1 AND 2) OR 3"` - complex logic expressions
  - Custom expressions with parentheses

**Returns**: `bool` - True if combination logic evaluates to True

**Usage Examples**:
```python
# Simple AND logic
result = evaluate_expression_list(
    df,
    ["Close[-1] > 100", "Close[-1] < 200"],
    "AND"
)

# Complex logic
result = evaluate_expression_list(
    df,
    ["Close[-1] > sma(20)[-1]", "rsi(14)[-1] < 30", "Volume[-1] > 1000000"],
    "(1 AND 2) OR 3"
)
```

---

## Enhanced Existing Functions

### 4. `ind_to_dict(ind, debug_mode=False)`
**Enhanced**: Added fallback implementation when legacy bytecode is unavailable.

**Purpose**: Parse indicator expression strings into structured dictionaries.

**Handles**:
- Numeric values: `"150"`, `"-2.0"`
- Price columns: `"Close[-1]"`, `"Open[-2]"`
- Indicators with parameters: `"rsi(14)[-1]"`, `"sma(20)[-1]"`
- Complex indicators: `"ICHIMOKU_CONVERSION(periods=9)[-1]"`

**Returns**: Dictionary with indicator metadata

---

### 5. `apply_function(df, ind, vals=None, debug_mode=False)`
**Enhanced**: Added fallback implementations for common indicators when legacy code is unavailable.

**Supports**:
- Price columns: Close, Open, High, Low, Volume
- Moving Averages: SMA, EMA, HMA
- Oscillators: RSI, CCI, Williams %R, ROC
- Others: ATR, MACD, Bollinger Bands
- Custom indicators: EWO, MA_SPREAD_ZSCORE, Ichimoku components

---

### 6. `evaluate_expression(df, exp, debug_mode=False)`
**Enhanced**: Added fallback for simple binary comparisons.

**Purpose**: Evaluate a conditional expression against a DataFrame.

**Handles**:
- Simple comparisons: `"Close[-1] > 150"`
- Indicator comparisons: `"Close[-1] > sma(20)[-1]"`
- Complex expressions with `&` and `|`: `"(Close[-1] > 100) & (rsi(14)[-1] < 70)"`
- EWO zero-cross patterns
- Ichimoku cloud expressions

---

## Helper Functions Added

### `_apply_function_fallback(df, ind, vals, debug_mode)`
Fallback implementation for common indicators when legacy code is unavailable.

### `_evaluate_simple_comparison(df, exp, debug_mode)`
Evaluate simple binary comparison expressions as a fallback.

---

## Backward Compatibility

The rebuilt `backend.py` maintains backward compatibility:

1. **With Legacy Bytecode**: If the `backend_legacy.cpython-311.pyc` file exists, it will be loaded and used
2. **Without Legacy Bytecode**: Falls back to the rebuilt implementations seamlessly

The bootstrap mechanism was updated to be optional:
```python
def _load_legacy_code() -> CodeType | None:
    if not _LEGACY_BYTECODE.exists():
        return None  # Gracefully handle missing bytecode
    # ... load bytecode ...
```

---

## Testing

All functions were tested and verified to work correctly:
- ✅ Function imports
- ✅ Indicator parsing (`ind_to_dict`)
- ✅ Condition parsing (`simplify_conditions`)
- ✅ Indicator calculation (`indicator_calculation`)
- ✅ Single expression evaluation (`evaluate_expression`)
- ✅ Multiple expression evaluation with logic (`evaluate_expression_list`)
- ✅ Complex combination logic (AND, OR, parentheses)

---

## Usage in Codebase

The rebuilt functions are used in:
- `pages/Scanner.py` - Market scanner functionality
- Alert processing systems
- Technical analysis condition evaluation

---

## Dependencies

Required libraries:
- `pandas` - DataFrame operations
- `numpy` - Numerical operations
- `talib` - Technical analysis indicators
- `indicators_lib` - Custom indicator implementations (EMA, SMA, HMA, EWO, MA_SPREAD_ZSCORE, BBANDS, Ichimoku, etc.)
- `re` - Regular expression parsing

---

## Notes

1. The fallback implementations cover the most commonly used indicators
2. For advanced/uncommon indicators, the legacy bytecode is still preferred if available
3. All parsing functions are robust and handle edge cases gracefully
4. The module now works independently without requiring the legacy bytecode file

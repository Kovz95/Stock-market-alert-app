package expr

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// parseMAValue extracts period and optional MA type from strings like "20", "20 (SMA)", "20 (EMA)".
func parseMAValue(s string) (period int, maType string) {
	maType = "SMA"
	s = strings.TrimSpace(s)
	if idx := strings.Index(s, " ("); idx > 0 {
		typePart := s[idx+2:]
		typePart = strings.TrimSuffix(typePart, ")")
		typePart = strings.TrimSpace(typePart)
		if typePart != "" {
			maType = strings.ToUpper(typePart)
		}
		s = strings.TrimSpace(s[:idx])
	}
	p, _ := strconv.Atoi(s)
	if p <= 0 {
		p = 20
	}
	return p, maType
}

// parseMACrossoverValue extracts fast, slow and optional type from "10 > 20" or "10 > 20 (SMA)".
func parseMACrossoverValue(s string) (fast, slow int, maType string) {
	maType = "SMA"
	s = strings.TrimSpace(s)
	if idx := strings.Index(s, " ("); idx > 0 {
		typePart := s[idx+2:]
		typePart = strings.TrimSuffix(typePart, ")")
		typePart = strings.TrimSpace(typePart)
		if typePart != "" {
			maType = strings.ToUpper(typePart)
		}
		s = strings.TrimSpace(s[:idx])
	}
	// "10 > 20" or "10>20"
	parts := regexp.MustCompile(`\s*>\s*`).Split(s, 2)
	if len(parts) != 2 {
		return 10, 20, maType
	}
	f, _ := strconv.Atoi(strings.TrimSpace(parts[0]))
	sl, _ := strconv.Atoi(strings.TrimSpace(parts[1]))
	if f <= 0 {
		f = 10
	}
	if sl <= 0 {
		sl = 20
	}
	return f, sl, maType
}

// maIndicator returns the indicator name for the MA type (sma, ema, hma). Default sma.
func maIndicator(maType string) string {
	switch maType {
	case "EMA":
		return "ema"
	case "HMA":
		return "hma"
	default:
		return "sma"
	}
}

// ExpandCatalogCondition converts UI catalog-style condition strings into
// expression form that ParseCondition understands. Returns the expanded
// expression, or empty string if the condition is not a catalog form.
//
// Supported catalog forms (aligned with Python scanner/alerts):
//   - price: price_above / price_below / price_equals with numeric value
//   - MA: price_above_ma / price_below_ma (period and optional SMA/EMA/HMA), ma_crossover (fast > slow)
//   - RSI: rsi_oversold / rsi_overbought with level (default period 14)
//   - MACD: macd_bullish_crossover, macd_bearish_crossover, macd_histogram_positive (default 12,26,9)
//   - Bollinger: price_above_upper_band, price_below_lower_band (default 20, 2)
//   - Volume: volume_above_average / volume_spike with multiplier (e.g. 1.5x, default period 20)
func ExpandCatalogCondition(condition string) string {
	condition = strings.TrimSpace(condition)
	if condition == "" {
		return ""
	}

	colon := strings.Index(condition, ":")
	var key, valueStr string
	if colon <= 0 {
		key = strings.ToLower(condition)
		valueStr = ""
	} else {
		key = strings.TrimSpace(strings.ToLower(condition[:colon]))
		valueStr = strings.TrimSpace(condition[colon+1:])
	}

	switch key {
	// --- Price ---
	case "price_above":
		if v, err := strconv.ParseFloat(strings.TrimSpace(valueStr), 64); err == nil {
			return fmt.Sprintf("close[-1] > %g", v)
		}
		return ""
	case "price_below":
		if v, err := strconv.ParseFloat(strings.TrimSpace(valueStr), 64); err == nil {
			return fmt.Sprintf("close[-1] < %g", v)
		}
		return ""
	case "price_equals":
		if v, err := strconv.ParseFloat(strings.TrimSpace(valueStr), 64); err == nil {
			return fmt.Sprintf("close[-1] == %g", v)
		}
		return ""

	// --- Moving Average ---
	case "price_above_ma":
		period, maType := parseMAValue(valueStr)
		ind := maIndicator(maType)
		return fmt.Sprintf("close[-1] > %s(%d)[-1]", ind, period)
	case "price_below_ma":
		period, maType := parseMAValue(valueStr)
		ind := maIndicator(maType)
		return fmt.Sprintf("close[-1] < %s(%d)[-1]", ind, period)
	case "ma_crossover":
		fast, slow, maType := parseMACrossoverValue(valueStr)
		ind := maIndicator(maType)
		return fmt.Sprintf("%s(%d)[-1] > %s(%d)[-1]", ind, fast, ind, slow)
	case "ma_crossunder":
		fast, slow, maType := parseMACrossoverValue(valueStr)
		ind := maIndicator(maType)
		return fmt.Sprintf("%s(%d)[-1] < %s(%d)[-1]", ind, fast, ind, slow)
	case "price_cross_above_ma":
		period, maType := parseMAValue(valueStr)
		ind := maIndicator(maType)
		return fmt.Sprintf("(close[-1] > %s(%d)[-1]) and (close[-2] <= %s(%d)[-2])", ind, period, ind, period)
	case "price_cross_below_ma":
		period, maType := parseMAValue(valueStr)
		ind := maIndicator(maType)
		return fmt.Sprintf("(close[-1] < %s(%d)[-1]) and (close[-2] >= %s(%d)[-2])", ind, period, ind, period)

	// --- RSI (default period 14) ---
	case "rsi_oversold":
		level, period := 30, 14
		if valueStr != "" {
			numStr := valueStr
			if idx := strings.Index(valueStr, " ("); idx > 0 {
				numStr = strings.TrimSpace(valueStr[:idx])
				paren := valueStr[idx+2:]
				paren = strings.TrimSuffix(paren, ")")
				if p, err := strconv.Atoi(strings.TrimSpace(paren)); err == nil && p > 0 {
					period = p
				}
			}
			if v, err := strconv.Atoi(strings.TrimSpace(numStr)); err == nil && v >= 0 && v <= 100 {
				level = v
			}
		}
		return fmt.Sprintf("rsi(%d)[-1] < %d", period, level)
	case "rsi_overbought":
		level, period := 70, 14
		if valueStr != "" {
			numStr := strings.TrimSpace(valueStr)
			if idx := strings.Index(numStr, " ("); idx > 0 {
				numStr = strings.TrimSpace(numStr[:idx])
			}
			if v, err := strconv.Atoi(numStr); err == nil && v >= 0 && v <= 100 {
				level = v
			}
		}
		return fmt.Sprintf("rsi(%d)[-1] > %d", period, level)

	// --- MACD (default 12, 26, 9) ---
	case "macd_bullish_crossover":
		return "macd(12, 26, 9, type=line)[-1] > macd(12, 26, 9, type=signal)[-1]"
	case "macd_bearish_crossover":
		return "macd(12, 26, 9, type=line)[-1] < macd(12, 26, 9, type=signal)[-1]"
	case "macd_histogram_positive":
		return "macd(12, 26, 9, type=histogram)[-1] > 0"

	// --- Bollinger (default 20, 2) ---
	case "price_above_upper_band":
		return "close[-1] > bbands(20, 2.0, type='upper')[-1]"
	case "price_below_lower_band":
		return "close[-1] < bbands(20, 2.0, type='lower')[-1]"

	// --- Volume (default 20-period average) ---
	case "volume_above_average", "volume_spike":
		mult := 1.5
		s := strings.TrimSpace(strings.TrimSuffix(strings.ToLower(valueStr), "x"))
		if s != "" {
			if v, err := strconv.ParseFloat(s, 64); err == nil && v > 0 {
				mult = v
			}
		}
		return fmt.Sprintf("volume_ratio(20)[-1] > %g", mult)
	case "volume_below_average":
		mult := 0.5
		s := strings.TrimSpace(strings.TrimSuffix(strings.ToLower(valueStr), "x"))
		if s != "" {
			if v, err := strconv.ParseFloat(s, 64); err == nil && v > 0 && v < 1 {
				mult = v
			}
		}
		return fmt.Sprintf("volume_ratio(20)[-1] < %g", mult)

	// --- Slow Stochastic level crosses (default smooth_k=14, smooth_d=3) ---
	case "slow_stoch_k_cross_above_oversold":
		level := 20
		if v, err := strconv.Atoi(strings.TrimSpace(valueStr)); err == nil && v >= 0 && v <= 100 {
			level = v
		}
		return fmt.Sprintf(
			"(slow_stoch_k(smooth_k=14, smooth_d=3)[-1] > %d) and (slow_stoch_k(smooth_k=14, smooth_d=3)[-2] <= %d)",
			level, level,
		)
	case "slow_stoch_k_cross_below_oversold":
		level := 20
		if v, err := strconv.Atoi(strings.TrimSpace(valueStr)); err == nil && v >= 0 && v <= 100 {
			level = v
		}
		return fmt.Sprintf(
			"(slow_stoch_k(smooth_k=14, smooth_d=3)[-1] < %d) and (slow_stoch_k(smooth_k=14, smooth_d=3)[-2] >= %d)",
			level, level,
		)
	case "slow_stoch_k_cross_above_overbought":
		level := 80
		if v, err := strconv.Atoi(strings.TrimSpace(valueStr)); err == nil && v >= 0 && v <= 100 {
			level = v
		}
		return fmt.Sprintf(
			"(slow_stoch_k(smooth_k=14, smooth_d=3)[-1] > %d) and (slow_stoch_k(smooth_k=14, smooth_d=3)[-2] <= %d)",
			level, level,
		)
	case "slow_stoch_k_cross_below_overbought":
		level := 80
		if v, err := strconv.Atoi(strings.TrimSpace(valueStr)); err == nil && v >= 0 && v <= 100 {
			level = v
		}
		return fmt.Sprintf(
			"(slow_stoch_k(smooth_k=14, smooth_d=3)[-1] < %d) and (slow_stoch_k(smooth_k=14, smooth_d=3)[-2] >= %d)",
			level, level,
		)
	case "slow_stoch_d_cross_above_oversold":
		level := 20
		if v, err := strconv.Atoi(strings.TrimSpace(valueStr)); err == nil && v >= 0 && v <= 100 {
			level = v
		}
		return fmt.Sprintf(
			"(slow_stoch_d(smooth_k=14, smooth_d=3)[-1] > %d) and (slow_stoch_d(smooth_k=14, smooth_d=3)[-2] <= %d)",
			level, level,
		)
	case "slow_stoch_d_cross_below_oversold":
		level := 20
		if v, err := strconv.Atoi(strings.TrimSpace(valueStr)); err == nil && v >= 0 && v <= 100 {
			level = v
		}
		return fmt.Sprintf(
			"(slow_stoch_d(smooth_k=14, smooth_d=3)[-1] < %d) and (slow_stoch_d(smooth_k=14, smooth_d=3)[-2] >= %d)",
			level, level,
		)
	case "slow_stoch_d_cross_above_overbought":
		level := 80
		if v, err := strconv.Atoi(strings.TrimSpace(valueStr)); err == nil && v >= 0 && v <= 100 {
			level = v
		}
		return fmt.Sprintf(
			"(slow_stoch_d(smooth_k=14, smooth_d=3)[-1] > %d) and (slow_stoch_d(smooth_k=14, smooth_d=3)[-2] <= %d)",
			level, level,
		)
	case "slow_stoch_d_cross_below_overbought":
		level := 80
		if v, err := strconv.Atoi(strings.TrimSpace(valueStr)); err == nil && v >= 0 && v <= 100 {
			level = v
		}
		return fmt.Sprintf(
			"(slow_stoch_d(smooth_k=14, smooth_d=3)[-1] < %d) and (slow_stoch_d(smooth_k=14, smooth_d=3)[-2] >= %d)",
			level, level,
		)
	}

	return ""
}

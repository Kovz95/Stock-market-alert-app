package price

// MockFMP returns fixed data for integration tests.
type MockFMP struct {
	Daily  map[string][]DailyRow   // ticker -> rows
	Hourly map[string][]HourlyRow  // ticker -> rows
}

// FetchDaily returns mock daily rows for the ticker, or nil if not set.
func (m *MockFMP) FetchDaily(ticker string, limit int) ([]DailyRow, error) {
	if m.Daily == nil {
		return nil, nil
	}
	rows, ok := m.Daily[ticker]
	if !ok || len(rows) == 0 {
		return nil, nil
	}
	if limit > 0 && len(rows) > limit {
		rows = rows[:limit]
	}
	return rows, nil
}

// FetchHourly returns mock hourly rows for the ticker, or nil if not set.
func (m *MockFMP) FetchHourly(ticker string) ([]HourlyRow, error) {
	if m.Hourly == nil {
		return nil, nil
	}
	rows, ok := m.Hourly[ticker]
	if !ok {
		return nil, nil
	}
	return rows, nil
}

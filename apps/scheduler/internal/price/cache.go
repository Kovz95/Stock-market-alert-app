package price

import "sync"

// Cache holds in-memory state for a single scheduler run (e.g. tickers already updated)
// to avoid duplicate work. The alert checker maintains its own OHLCV cache for evaluation.
type Cache struct {
	mu      sync.Mutex
	updated map[string]bool // ticker -> true if updated this run
}

// NewCache creates an empty run cache.
func NewCache() *Cache {
	return &Cache{updated: make(map[string]bool)}
}

// MarkUpdated records that the ticker was updated this run.
func (c *Cache) MarkUpdated(ticker string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.updated[ticker] = true
}

// Count returns the number of tickers marked updated.
func (c *Cache) Count() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.updated)
}

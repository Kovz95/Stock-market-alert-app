package discord

import (
	"context"
	"log/slog"
	"strings"
	"sync"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"

	db "stockalert/database/generated"
)

// DefaultResolverTTL is the cache TTL for PortfolioResolver and CustomChannelResolver,
// matching Python's _CONFIG_CACHE_TTL of 300 seconds.
const DefaultResolverTTL = 5 * time.Minute

// PortfolioQueriesDB is the minimal DB interface needed by PortfolioResolver.
// *db.Queries satisfies it; tests can inject a fake.
type PortfolioQueriesDB interface {
	ListPortfoliosForFanout(ctx context.Context) ([]db.ListPortfoliosForFanoutRow, error)
	ListPortfolioStocks(ctx context.Context) ([]db.PortfolioStock, error)
}

// PortfolioResolver resolves Discord webhook URLs for portfolios containing a given ticker.
// Results are cached in memory with a configurable TTL; a sync.Mutex serializes reloads.
type PortfolioResolver struct {
	queries  PortfolioQueriesDB
	ttl      time.Duration
	logger   *slog.Logger
	mu       sync.Mutex
	loadedAt time.Time
	byTicker map[string][]string // uppercase ticker -> webhook URLs
}

// NewPortfolioResolver creates a PortfolioResolver backed by the given connection pool.
// The cache is populated lazily on the first ResolveWebhooks call.
func NewPortfolioResolver(pool *pgxpool.Pool, ttl time.Duration, logger *slog.Logger) *PortfolioResolver {
	return NewPortfolioResolverFromQueries(db.New(pool), ttl, logger)
}

// NewPortfolioResolverFromQueries creates a PortfolioResolver using a custom query
// implementation. Useful for testing or dependency injection.
func NewPortfolioResolverFromQueries(q PortfolioQueriesDB, ttl time.Duration, logger *slog.Logger) *PortfolioResolver {
	return &PortfolioResolver{
		queries: q,
		ttl:     ttl,
		logger:  logger,
	}
}

// ResolveWebhooks returns the Discord webhook URLs for every enabled portfolio that contains
// ticker (uppercase-insensitive). Exchange suffixes (e.g. ".L") are preserved verbatim.
// Returns a partial list plus a non-nil error if the DB load failed; callers should log and
// continue with whatever URLs were returned.
func (r *PortfolioResolver) ResolveWebhooks(ctx context.Context, ticker string) ([]string, error) {
	upper := strings.ToUpper(ticker)

	r.mu.Lock()
	defer r.mu.Unlock()

	var loadErr error
	if r.loadedAt.IsZero() || time.Since(r.loadedAt) >= r.ttl {
		loadErr = r.loadLocked(ctx)
	}

	if r.byTicker == nil {
		return nil, loadErr
	}

	webhooks := r.byTicker[upper]
	if len(webhooks) == 0 {
		return nil, loadErr
	}
	// Return a copy so callers cannot mutate the cache.
	out := make([]string, len(webhooks))
	copy(out, webhooks)
	return out, loadErr
}

// loadLocked rebuilds the byTicker cache from the database. Must be called with mu held.
func (r *PortfolioResolver) loadLocked(ctx context.Context) error {
	start := time.Now()

	portfolios, err := r.queries.ListPortfoliosForFanout(ctx)
	if err != nil {
		r.logger.Warn("portfolio_resolver: failed to load portfolios",
			"error", err,
			"duration_ms", time.Since(start).Milliseconds(),
		)
		// Update loadedAt so we don't hammer the DB on every call during an outage.
		r.loadedAt = time.Now()
		return err
	}

	// Build portfolio_id -> webhook mapping.
	webhookByID := make(map[string]string, len(portfolios))
	for _, p := range portfolios {
		if p.DiscordWebhook.Valid && p.DiscordWebhook.String != "" {
			webhookByID[p.ID] = p.DiscordWebhook.String
		}
	}

	stocks, err := r.queries.ListPortfolioStocks(ctx)
	if err != nil {
		r.logger.Warn("portfolio_resolver: failed to load portfolio stocks",
			"error", err,
			"duration_ms", time.Since(start).Milliseconds(),
		)
		r.loadedAt = time.Now()
		return err
	}

	byTicker := make(map[string][]string)
	for _, s := range stocks {
		webhook, ok := webhookByID[s.PortfolioID]
		if !ok {
			continue
		}
		upper := strings.ToUpper(s.Ticker)
		byTicker[upper] = append(byTicker[upper], webhook)
	}

	r.byTicker = byTicker
	r.loadedAt = time.Now()
	r.logger.Debug("portfolio_resolver: loaded",
		"portfolios", len(portfolios),
		"stocks", len(stocks),
		"duration_ms", time.Since(start).Milliseconds(),
	)
	return nil
}

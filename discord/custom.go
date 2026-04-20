package discord

import (
	"context"
	"encoding/json"
	"log/slog"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"

	db "stockalert/database/generated"
)

// priceLevelRE matches price-level conditions: (close|open|high|low)[-N]? <op> $?<number>.
// This is a line-for-line port of is_price_level_condition from discord_routing.py.
var priceLevelRE = regexp.MustCompile(`(?i)(close|open|high|low)(\[-?\d+\])?\s*([<>=!]+)\s*\$?\d+\.?\d*`)

// These regexes compile once for normalizeConditionString.
var (
	opSpaceRE      = regexp.MustCompile(`\s*([<>=!]+)\s*`)
	bracketSpaceRE = regexp.MustCompile(`\s*([(),\[\]])\s*`)
	wsRunRE        = regexp.MustCompile(`\s+`)
)

// normalizeConditionString strips spaces around operators/brackets, collapses whitespace
// runs to a single space, then lower-cases and trims. Port of normalize_condition() in
// src/services/discord_routing.py:check_custom_channel_condition.
func normalizeConditionString(s string) string {
	s = opSpaceRE.ReplaceAllString(s, "$1")
	s = bracketSpaceRE.ReplaceAllString(s, "$1")
	s = wsRunRE.ReplaceAllString(s, " ")
	return strings.ToLower(strings.TrimSpace(s))
}

// CustomChannelQueriesDB is the minimal DB interface needed by CustomChannelResolver.
// *db.Queries satisfies it; tests can inject a fake.
type CustomChannelQueriesDB interface {
	GetAppDocument(ctx context.Context, documentKey string) (db.AppDocument, error)
}

// cachedCustomChannel holds a pre-processed custom channel entry ready for matching.
type cachedCustomChannel struct {
	webhook             string
	normalizedCondition string
	isPriceLevel        bool
}

// customChannelEntry is the JSON shape of a single entry in the custom_discord_channels doc.
// Fields must match the Python / gRPC schema exactly.
type customChannelEntry struct {
	WebhookURL    string `json:"webhook_url"`
	Condition     string `json:"condition"`
	ConditionType string `json:"condition_type"` // only present in legacy entries
	Enabled       bool   `json:"enabled"`
}

// CustomChannelResolver resolves Discord webhook URLs for custom channels whose condition
// matches any of the given alert conditions. Results are cached with a configurable TTL.
type CustomChannelResolver struct {
	queries  CustomChannelQueriesDB
	ttl      time.Duration
	logger   *slog.Logger
	mu       sync.Mutex
	loadedAt time.Time
	channels []cachedCustomChannel
}

// NewCustomChannelResolver creates a CustomChannelResolver backed by the given pool.
// The cache is populated lazily on the first ResolveWebhooks call.
func NewCustomChannelResolver(pool *pgxpool.Pool, ttl time.Duration, logger *slog.Logger) *CustomChannelResolver {
	return NewCustomChannelResolverFromQueries(db.New(pool), ttl, logger)
}

// NewCustomChannelResolverFromQueries creates a CustomChannelResolver using a custom query
// implementation. Useful for testing or dependency injection.
func NewCustomChannelResolverFromQueries(q CustomChannelQueriesDB, ttl time.Duration, logger *slog.Logger) *CustomChannelResolver {
	return &CustomChannelResolver{
		queries: q,
		ttl:     ttl,
		logger:  logger,
	}
}

// ResolveWebhooks returns the webhook URLs of every enabled custom channel whose condition
// matches any string in conditions (after normalization), or whose condition is the special
// keyword "price_level" and any condition matches the price-level regex.
//
// Duplicates in the returned slice are allowed; the caller (onTriggered) deduplicates.
func (r *CustomChannelResolver) ResolveWebhooks(ctx context.Context, conditions []string) ([]string, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	var loadErr error
	if r.loadedAt.IsZero() || time.Since(r.loadedAt) >= r.ttl {
		loadErr = r.loadLocked(ctx)
	}

	if len(r.channels) == 0 {
		return nil, loadErr
	}

	// Normalize alert conditions once.
	normalized := make([]string, len(conditions))
	for i, c := range conditions {
		normalized[i] = normalizeConditionString(c)
	}

	var out []string
	for _, ch := range r.channels {
		if ch.isPriceLevel {
			// Match if any alert condition matches the price-level pattern.
			for _, raw := range conditions {
				if priceLevelRE.MatchString(raw) {
					out = append(out, ch.webhook)
					break
				}
			}
		} else {
			// Exact byte-equal match on normalized strings.
			for _, n := range normalized {
				if n == ch.normalizedCondition {
					out = append(out, ch.webhook)
					break
				}
			}
		}
	}
	return out, loadErr
}

// loadLocked rebuilds the channels cache from the database. Must be called with mu held.
func (r *CustomChannelResolver) loadLocked(ctx context.Context) error {
	start := time.Now()

	doc, err := r.queries.GetAppDocument(ctx, "custom_discord_channels")
	if err != nil {
		if err == pgx.ErrNoRows {
			// Document does not exist yet — treat as empty, not an error.
			r.channels = nil
			r.loadedAt = time.Now()
			r.logger.Debug("custom_channel_resolver: no document found")
			return nil
		}
		r.logger.Warn("custom_channel_resolver: failed to load document",
			"error", err,
			"duration_ms", time.Since(start).Milliseconds(),
		)
		r.loadedAt = time.Now()
		return err
	}

	// The document is a JSON object keyed by channel name.
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(doc.Payload, &raw); err != nil {
		r.logger.Warn("custom_channel_resolver: failed to parse document",
			"error", err,
			"duration_ms", time.Since(start).Milliseconds(),
		)
		r.loadedAt = time.Now()
		return err
	}

	var channels []cachedCustomChannel
	for _, entryBytes := range raw {
		var entry customChannelEntry
		if err := json.Unmarshal(entryBytes, &entry); err != nil {
			r.logger.Warn("custom_channel_resolver: skipping unparseable entry", "error", err)
			continue
		}

		if !entry.Enabled {
			continue
		}
		// Legacy schema: condition_type is set but condition is blank — skip.
		if entry.ConditionType != "" && entry.Condition == "" {
			continue
		}
		if entry.Condition == "" || entry.WebhookURL == "" {
			continue
		}

		norm := normalizeConditionString(entry.Condition)
		channels = append(channels, cachedCustomChannel{
			webhook:             entry.WebhookURL,
			normalizedCondition: norm,
			isPriceLevel:        norm == "price_level",
		})
	}

	r.channels = channels
	r.loadedAt = time.Now()
	r.logger.Debug("custom_channel_resolver: loaded",
		"channels", len(channels),
		"duration_ms", time.Since(start).Milliseconds(),
	)
	return nil
}

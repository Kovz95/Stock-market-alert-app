package discord

import (
	"context"
	"encoding/json"
	"log/slog"
	"testing"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgtype"

	db "stockalert/database/generated"
)

// fakeCustomChannelQueries is a test double for CustomChannelQueriesDB.
type fakeCustomChannelQueries struct {
	payload   map[string]interface{}
	loadCount int
	missing   bool // simulate pgx.ErrNoRows
}

func (f *fakeCustomChannelQueries) GetAppDocument(_ context.Context, key string) (db.AppDocument, error) {
	f.loadCount++
	if f.missing {
		return db.AppDocument{}, pgx.ErrNoRows
	}
	b, err := json.Marshal(f.payload)
	if err != nil {
		return db.AppDocument{}, err
	}
	return db.AppDocument{
		DocumentKey: key,
		Payload:     b,
		SourcePath:  pgtype.Text{},
	}, nil
}

func newTestCustomResolver(q CustomChannelQueriesDB, ttl time.Duration) *CustomChannelResolver {
	return NewCustomChannelResolverFromQueries(q, ttl, slog.Default())
}

func channelEntry(webhook, condition string, enabled bool) map[string]interface{} {
	return map[string]interface{}{
		"webhook_url": webhook,
		"condition":   condition,
		"enabled":     enabled,
		"channel_name": "#test",
	}
}

func legacyEntry(webhook string) map[string]interface{} {
	return map[string]interface{}{
		"webhook_url":    webhook,
		"condition_type": "price",
		"enabled":        true,
		// no "condition" key — legacy schema
	}
}

func TestCustomChannelResolver_Empty(t *testing.T) {
	fake := &fakeCustomChannelQueries{payload: map[string]interface{}{}}
	r := newTestCustomResolver(fake, time.Hour)

	urls, err := r.ResolveWebhooks(context.Background(), []string{"close > 100"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(urls) != 0 {
		t.Errorf("expected empty, got %v", urls)
	}
}

func TestCustomChannelResolver_MissingDocument(t *testing.T) {
	fake := &fakeCustomChannelQueries{missing: true}
	r := newTestCustomResolver(fake, time.Hour)

	urls, err := r.ResolveWebhooks(context.Background(), []string{"close > 100"})
	if err != nil {
		t.Fatalf("ErrNoRows should not propagate as error: %v", err)
	}
	if len(urls) != 0 {
		t.Errorf("expected empty, got %v", urls)
	}
}

func TestCustomChannelResolver_Enabled(t *testing.T) {
	fake := &fakeCustomChannelQueries{
		payload: map[string]interface{}{
			"ch1": channelEntry("W1", "close > 100", true),
		},
	}
	r := newTestCustomResolver(fake, time.Hour)

	urls, err := r.ResolveWebhooks(context.Background(), []string{"close > 100"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(urls) != 1 || urls[0] != "W1" {
		t.Errorf("expected [W1], got %v", urls)
	}
}

func TestCustomChannelResolver_Disabled(t *testing.T) {
	fake := &fakeCustomChannelQueries{
		payload: map[string]interface{}{
			"ch1": channelEntry("W1", "close > 100", false),
		},
	}
	r := newTestCustomResolver(fake, time.Hour)

	urls, err := r.ResolveWebhooks(context.Background(), []string{"close > 100"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(urls) != 0 {
		t.Errorf("disabled channel should not match, got %v", urls)
	}
}

func TestCustomChannelResolver_LegacySchemaSkipped(t *testing.T) {
	fake := &fakeCustomChannelQueries{
		payload: map[string]interface{}{
			"ch1": legacyEntry("W1"),
		},
	}
	r := newTestCustomResolver(fake, time.Hour)

	// Even if the alert condition would otherwise match, legacy entries are skipped.
	urls, err := r.ResolveWebhooks(context.Background(), []string{"close > 100"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(urls) != 0 {
		t.Errorf("legacy entry should be skipped, got %v", urls)
	}
}

func TestCustomChannelResolver_NormalizationWhitespace(t *testing.T) {
	fake := &fakeCustomChannelQueries{
		payload: map[string]interface{}{
			// Channel condition has extra spaces.
			"ch1": channelEntry("W1", "close  >  100", true),
		},
	}
	r := newTestCustomResolver(fake, time.Hour)

	urls, err := r.ResolveWebhooks(context.Background(), []string{"close > 100"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(urls) != 1 || urls[0] != "W1" {
		t.Errorf("expected [W1] after normalization, got %v", urls)
	}
}

func TestCustomChannelResolver_NormalizationOperators(t *testing.T) {
	fake := &fakeCustomChannelQueries{
		payload: map[string]interface{}{
			"ch1": channelEntry("W1", "close>100", true),
			"ch2": channelEntry("W2", "a ( b )", true),
		},
	}
	r := newTestCustomResolver(fake, time.Hour)

	urls1, _ := r.ResolveWebhooks(context.Background(), []string{"close > 100"})
	if len(urls1) != 1 || urls1[0] != "W1" {
		t.Errorf("operator normalization: expected [W1], got %v", urls1)
	}

	// Reset TTL to force reload-free check (already loaded).
	urls2, _ := r.ResolveWebhooks(context.Background(), []string{"a(b)"})
	if len(urls2) != 1 || urls2[0] != "W2" {
		t.Errorf("bracket normalization: expected [W2], got %v", urls2)
	}
}

func TestCustomChannelResolver_NormalizationCase(t *testing.T) {
	fake := &fakeCustomChannelQueries{
		payload: map[string]interface{}{
			"ch1": channelEntry("W1", "CLOSE > 100", true),
		},
	}
	r := newTestCustomResolver(fake, time.Hour)

	urls, err := r.ResolveWebhooks(context.Background(), []string{"close > 100"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(urls) != 1 || urls[0] != "W1" {
		t.Errorf("case normalization: expected [W1], got %v", urls)
	}
}

func TestCustomChannelResolver_PriceLevelMatch(t *testing.T) {
	fake := &fakeCustomChannelQueries{
		payload: map[string]interface{}{
			"ch1": channelEntry("W1", "price_level", true),
		},
	}
	r := newTestCustomResolver(fake, time.Hour)

	// close > $100 — should match.
	urls, _ := r.ResolveWebhooks(context.Background(), []string{"close > $100"})
	if len(urls) != 1 || urls[0] != "W1" {
		t.Errorf("price_level: close > $100 should match, got %v", urls)
	}

	// close[-1] >= 50.5 — should match.
	urls2, _ := r.ResolveWebhooks(context.Background(), []string{"close[-1] >= 50.5"})
	if len(urls2) != 1 || urls2[0] != "W1" {
		t.Errorf("price_level: close[-1] >= 50.5 should match, got %v", urls2)
	}

	// rsi > 30 — should NOT match.
	urls3, _ := r.ResolveWebhooks(context.Background(), []string{"rsi > 30"})
	if len(urls3) != 0 {
		t.Errorf("price_level: rsi > 30 should not match, got %v", urls3)
	}
}

func TestCustomChannelResolver_PriceLevelCaseInsensitive(t *testing.T) {
	// Channel condition stored as uppercase "PRICE_LEVEL" — after normalization it
	// becomes "price_level" and the special keyword still triggers.
	fake := &fakeCustomChannelQueries{
		payload: map[string]interface{}{
			"ch1": channelEntry("W1", "PRICE_LEVEL", true),
		},
	}
	r := newTestCustomResolver(fake, time.Hour)

	urls, _ := r.ResolveWebhooks(context.Background(), []string{"CLOSE > $100"})
	if len(urls) != 1 || urls[0] != "W1" {
		t.Errorf("price_level case insensitive: expected [W1], got %v", urls)
	}
}

func TestCustomChannelResolver_ExactMatchNotPartial(t *testing.T) {
	fake := &fakeCustomChannelQueries{
		payload: map[string]interface{}{
			"ch1": channelEntry("W1", "close > 100", true),
		},
	}
	r := newTestCustomResolver(fake, time.Hour)

	// "close > 1000" is NOT the same as "close > 100" after normalization.
	urls, err := r.ResolveWebhooks(context.Background(), []string{"close > 1000"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(urls) != 0 {
		t.Errorf("partial match should not match, got %v", urls)
	}
}

func TestCustomChannelResolver_CacheHit(t *testing.T) {
	fake := &fakeCustomChannelQueries{
		payload: map[string]interface{}{
			"ch1": channelEntry("W1", "close > 100", true),
		},
	}
	r := newTestCustomResolver(fake, time.Hour)

	_, _ = r.ResolveWebhooks(context.Background(), []string{"close > 100"})
	countAfterFirst := fake.loadCount

	_, _ = r.ResolveWebhooks(context.Background(), []string{"close > 100"})
	if fake.loadCount != countAfterFirst {
		t.Errorf("expected no reload within TTL, loadCount went from %d to %d", countAfterFirst, fake.loadCount)
	}
}

func TestCustomChannelResolver_CacheReload(t *testing.T) {
	fake := &fakeCustomChannelQueries{
		payload: map[string]interface{}{
			"ch1": channelEntry("W1", "close > 100", true),
		},
	}
	r := newTestCustomResolver(fake, 1) // TTL = 1 nanosecond

	_, _ = r.ResolveWebhooks(context.Background(), []string{"close > 100"})
	countAfterFirst := fake.loadCount

	time.Sleep(2 * time.Nanosecond)
	_, _ = r.ResolveWebhooks(context.Background(), []string{"close > 100"})
	if fake.loadCount <= countAfterFirst {
		t.Errorf("expected reload after TTL expired, loadCount stayed at %d", fake.loadCount)
	}
}

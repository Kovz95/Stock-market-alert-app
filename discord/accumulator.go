package discord

import (
	"log"
	"sync"
)

// AccumFlusher is the minimal interface for queuing and flushing Discord embeds.
// *Accumulator satisfies it; tests can inject a recording fake.
type AccumFlusher interface {
	Add(webhookURL string, embed Embed)
	FlushAll() (sent int, errors int)
}

// Accumulator collects Discord embeds grouped by webhook URL during alert checks.
// Nothing is sent until FlushAll() is called; then embeds are sent in batches of
// up to 10 per webhook (Discord's per-message limit), with rate limiting between
// sends to the same webhook.
type Accumulator struct {
	mu       sync.Mutex
	notifier *Notifier
	embeds   map[string][]Embed // webhook URL -> embeds
}

// NewAccumulator creates an Accumulator backed by the given Notifier.
func NewAccumulator(notifier *Notifier) *Accumulator {
	return &Accumulator{
		notifier: notifier,
		embeds:   make(map[string][]Embed),
	}
}

// Add queues an embed for later sending to the given webhook URL.
// Thread-safe.
func (a *Accumulator) Add(webhookURL string, embed Embed) {
	if webhookURL == "" {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.embeds[webhookURL] = append(a.embeds[webhookURL], embed)
}

// AddMulti queues multiple embeds for the same webhook URL.
func (a *Accumulator) AddMulti(webhookURL string, embeds []Embed) {
	if webhookURL == "" || len(embeds) == 0 {
		return
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	a.embeds[webhookURL] = append(a.embeds[webhookURL], embeds...)
}

// Len returns the total number of queued embeds across all webhook URLs.
func (a *Accumulator) Len() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	total := 0
	for _, v := range a.embeds {
		total += len(v)
	}
	return total
}

// FlushAll sends all accumulated embeds in batches of 10 per webhook URL.
// Returns the number of embeds successfully sent and any errors encountered.
func (a *Accumulator) FlushAll() (sent int, errors int) {
	a.mu.Lock()
	// Take ownership of current embeds and reset.
	toSend := a.embeds
	a.embeds = make(map[string][]Embed)
	a.mu.Unlock()

	for webhookURL, embeds := range toSend {
		// Send in chunks of 10
		for i := 0; i < len(embeds); i += 10 {
			end := i + 10
			if end > len(embeds) {
				end = len(embeds)
			}
			chunk := embeds[i:end]

			if err := a.notifier.SendEmbeds(webhookURL, chunk); err != nil {
				log.Printf("discord/accumulator: failed to send %d embeds: %v", len(chunk), err)
				errors += len(chunk)
			} else {
				sent += len(chunk)
			}
		}
	}

	if sent > 0 || errors > 0 {
		log.Printf("discord/accumulator: flushed %d embeds (%d errors)", sent, errors)
	}
	return sent, errors
}

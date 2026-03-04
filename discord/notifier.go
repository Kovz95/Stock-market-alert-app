package discord

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Notifier sends messages to Discord webhooks with rate limiting.
type Notifier struct {
	client *http.Client

	// Rate limiter: simple token-bucket per webhook URL.
	mu       sync.Mutex
	lastSend map[string]time.Time

	// MinInterval between sends to the same webhook URL (per webhook).
	// Default 2s keeps under Discord's ~30 requests/min per webhook.
	MinInterval time.Duration
}

// NewNotifier creates a Notifier with sensible defaults.
// MinInterval can be overridden via DISCORD_WEBHOOK_INTERVAL_SEC (e.g. "2" or "2.5").
func NewNotifier() *Notifier {
	interval := 2 * time.Second
	if s := os.Getenv("DISCORD_WEBHOOK_INTERVAL_SEC"); s != "" {
		if sec, err := strconv.ParseFloat(strings.TrimSpace(s), 64); err == nil && sec > 0 {
			interval = time.Duration(sec * float64(time.Second))
		}
	}
	return &Notifier{
		client: &http.Client{
			Timeout: 10 * time.Second,
		},
		lastSend:    make(map[string]time.Time),
		MinInterval: interval,
	}
}

// SendEnabled returns true if Discord sending is enabled via environment.
func SendEnabled() bool {
	v := os.Getenv("DISCORD_SEND_ENABLED")
	return strings.EqualFold(v, "true") || v == "1"
}

// EnvironmentTag returns the environment prefix (e.g., "[PROD] " or "[DEV] ").
// Empty string if DISCORD_ENVIRONMENT is not set.
func EnvironmentTag() string {
	env := os.Getenv("DISCORD_ENVIRONMENT")
	if env == "" {
		return ""
	}
	return fmt.Sprintf("[%s] ", strings.ToUpper(env))
}

// SendMessage sends a plain text message to a Discord webhook.
func (n *Notifier) SendMessage(webhookURL, message string) error {
	if !SendEnabled() {
		log.Println("discord: send disabled, skipping message")
		return nil
	}
	if webhookURL == "" || strings.Contains(webhookURL, "YOUR_") {
		return fmt.Errorf("discord: webhook URL not configured")
	}

	n.rateLimit(webhookURL)

	tagged := EnvironmentTag() + message

	payload := map[string]string{
		"content": tagged,
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("discord: marshal payload: %w", err)
	}

	resp, err := n.client.Post(webhookURL, "application/json", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("discord: post webhook: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusNoContent {
		return fmt.Errorf("discord: webhook returned HTTP %d", resp.StatusCode)
	}
	return nil
}

// SendEmbeds sends a batch of embeds (max 10) to a Discord webhook.
// On 429 (rate limit), it waits for Retry-After then retries up to maxRetries.
func (n *Notifier) SendEmbeds(webhookURL string, embeds []Embed) error {
	if !SendEnabled() {
		log.Printf("discord: send disabled, skipping %d embeds", len(embeds))
		return nil
	}
	if webhookURL == "" || strings.Contains(webhookURL, "YOUR_") {
		return fmt.Errorf("discord: webhook URL not configured")
	}
	if len(embeds) == 0 {
		return nil
	}

	// Discord enforces max 10 embeds per message
	if len(embeds) > 10 {
		embeds = embeds[:10]
	}

	n.rateLimit(webhookURL)

	payload := map[string]interface{}{
		"embeds":   embeds,
		"username": "Stock Alert Bot",
	}
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("discord: marshal embeds payload: %w", err)
	}

	const maxRetries = 3
	var lastErr error
	for attempt := 0; attempt < maxRetries; attempt++ {
		resp, err := n.client.Post(webhookURL, "application/json", bytes.NewReader(body))
		if err != nil {
			lastErr = fmt.Errorf("discord: post webhook: %w", err)
			continue
		}

		if resp.StatusCode == http.StatusTooManyRequests {
			retryAfter := n.parseRetryAfter(resp)
			resp.Body.Close()
			if retryAfter > 0 && attempt+1 < maxRetries {
				log.Printf("discord: rate limited (429), waiting %.1fs before retry", retryAfter.Seconds())
				time.Sleep(retryAfter)
				continue
			}
			lastErr = fmt.Errorf("discord: webhook rate limited (429) after %d retries", attempt+1)
			break
		}

		if resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusNoContent {
			resp.Body.Close()
			return nil
		}
		_ = resp.Body.Close()
		lastErr = fmt.Errorf("discord: webhook returned HTTP %d", resp.StatusCode)
		break
	}
	return lastErr
}

// parseRetryAfter reads Retry-After header or JSON retry_after from 429 response.
func (n *Notifier) parseRetryAfter(resp *http.Response) time.Duration {
	if s := resp.Header.Get("Retry-After"); s != "" {
		if sec, err := strconv.ParseFloat(strings.TrimSpace(s), 64); err == nil && sec > 0 {
			return time.Duration(sec * float64(time.Second))
		}
	}
	// Fallback: parse JSON body for retry_after
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return 2 * time.Second
	}
	var v struct {
		RetryAfter float64 `json:"retry_after"`
	}
	if json.Unmarshal(raw, &v) == nil && v.RetryAfter > 0 {
		return time.Duration(v.RetryAfter * float64(time.Second))
	}
	return 2 * time.Second
}

// rateLimit blocks until enough time has passed since the last send
// to the given webhook URL.
func (n *Notifier) rateLimit(webhookURL string) {
	n.mu.Lock()
	defer n.mu.Unlock()

	if last, ok := n.lastSend[webhookURL]; ok {
		elapsed := time.Since(last)
		if elapsed < n.MinInterval {
			time.Sleep(n.MinInterval - elapsed)
		}
	}
	n.lastSend[webhookURL] = time.Now()
}

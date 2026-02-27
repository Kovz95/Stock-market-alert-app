package discord

import (
	"bytes"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
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

	// MinInterval between sends to the same webhook URL.
	MinInterval time.Duration
}

// NewNotifier creates a Notifier with sensible defaults.
func NewNotifier() *Notifier {
	return &Notifier{
		client: &http.Client{
			Timeout: 10 * time.Second,
		},
		lastSend:    make(map[string]time.Time),
		MinInterval: 500 * time.Millisecond, // Discord allows ~30 messages/min per webhook
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

package discord

import (
	"fmt"
	"log"
	"time"
)

// PriceStats holds price update statistics for status notifications.
type PriceStats struct {
	Updated int
	Total   int
	Failed  int
	Skipped int
}

// AlertStats holds alert check statistics for status notifications.
type AlertStats struct {
	Total        int
	Triggered    int
	NotTriggered int
	Skipped      int
	NoData       int
	Errors       int
}

// StatusNotifier sends scheduler status messages (start, complete, error, skipped)
// to a designated Discord webhook. Mirrors Python's BaseSchedulerDiscord.
type StatusNotifier struct {
	notifier   *Notifier
	webhookURL string
	jobLabel   string // "Daily", "Weekly", "Hourly"
	timeframe  string // "1d", "1wk", "1h"
}

// NewStatusNotifier creates a StatusNotifier for the given job type.
func NewStatusNotifier(notifier *Notifier, webhookURL, jobLabel, timeframe string) *StatusNotifier {
	return &StatusNotifier{
		notifier:   notifier,
		webhookURL: webhookURL,
		jobLabel:   jobLabel,
		timeframe:  timeframe,
	}
}

// NotifyStart sends a job-started message.
func (s *StatusNotifier) NotifyStart(runTime time.Time, exchange string) {
	msg := fmt.Sprintf(
		"✅ **%s Alert Check Started**\n• Run Time (EST): %s\n• Timeframe: %s\n• Exchange: %s",
		s.jobLabel,
		FormatESTTime(runTime),
		s.timeframe,
		exchange,
	)
	if err := s.notifier.SendMessage(s.webhookURL, msg); err != nil {
		log.Printf("discord/status: notify_start failed: %v", err)
	}
}

// NotifyComplete sends a job-completed message with stats.
func (s *StatusNotifier) NotifyComplete(
	runTime time.Time,
	durationSec float64,
	exchange string,
	priceStats *PriceStats,
	alertStats *AlertStats,
) {
	msg := fmt.Sprintf(
		"✅ **%s Alert Check Complete**\n• Run Time (EST): %s\n• Duration: %s\n• Timeframe: %s\n• Exchange: %s",
		s.jobLabel,
		FormatESTTime(runTime),
		FormatDuration(durationSec),
		s.timeframe,
		exchange,
	)

	if priceStats != nil {
		msg += fmt.Sprintf(
			"\n• Price Update: %d/%d updated | failed %d | skipped %d",
			priceStats.Updated, priceStats.Total, priceStats.Failed, priceStats.Skipped,
		)
	}

	if alertStats != nil {
		msg += fmt.Sprintf(
			"\n• Alerts: total %d | triggered %d | not_triggered %d | skipped %d | no_data %d | errors %d",
			alertStats.Total, alertStats.Triggered, alertStats.NotTriggered,
			alertStats.Skipped, alertStats.NoData, alertStats.Errors,
		)
	}

	if err := s.notifier.SendMessage(s.webhookURL, msg); err != nil {
		log.Printf("discord/status: notify_complete failed: %v", err)
	}
}

// NotifySkipped sends a job-skipped message.
func (s *StatusNotifier) NotifySkipped(runTime time.Time, reason string) {
	msg := fmt.Sprintf(
		"⚪ **%s Alert Check Skipped**\n• Run Time (EST): %s\n• Reason: %s",
		s.jobLabel,
		FormatESTTime(runTime),
		reason,
	)
	if err := s.notifier.SendMessage(s.webhookURL, msg); err != nil {
		log.Printf("discord/status: notify_skipped failed: %v", err)
	}
}

// NotifyError sends a job-error message.
func (s *StatusNotifier) NotifyError(runTime time.Time, errMsg string) {
	msg := fmt.Sprintf(
		"❌ **%s Scheduler Error**\n• Run Time (EST): %s\n• Error: %s",
		s.jobLabel,
		FormatESTTime(runTime),
		errMsg,
	)
	if err := s.notifier.SendMessage(s.webhookURL, msg); err != nil {
		log.Printf("discord/status: notify_error failed: %v", err)
	}
}

// NotifySchedulerStart sends a scheduler-online message.
func (s *StatusNotifier) NotifySchedulerStart(scheduleInfo string) {
	msg := fmt.Sprintf(
		"✅ **%s Scheduler Online**\n• Schedules: %s\n• Timestamp (EST): %s",
		s.jobLabel,
		scheduleInfo,
		FormatESTTime(time.Now().UTC()),
	)
	if err := s.notifier.SendMessage(s.webhookURL, msg); err != nil {
		log.Printf("discord/status: notify_scheduler_start failed: %v", err)
	}
}

// NotifySchedulerStop sends a scheduler-stopped message.
func (s *StatusNotifier) NotifySchedulerStop() {
	msg := fmt.Sprintf(
		"⏹️ **%s Scheduler Stopped**\n• Timestamp (EST): %s",
		s.jobLabel,
		FormatESTTime(time.Now().UTC()),
	)
	if err := s.notifier.SendMessage(s.webhookURL, msg); err != nil {
		log.Printf("discord/status: notify_scheduler_stop failed: %v", err)
	}
}

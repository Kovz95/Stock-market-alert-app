---
# Questions: Discord Webhook Visibility (06) — Round 1
---

## Resolved

| # | Question | Resolution |
|---|----------|------------|
| Q1 | Should the webhook URL be shown in full or masked? | Masked by default: display last 20 chars preceded by `…`. Include a copy-to-clipboard button so users can verify/copy without revealing the full token in the UI. |
| Q2 | What does "show when it has changed" mean — a timestamp or a diff? | Show a "Last updated" timestamp (`updatedAt`). No diff/history log is required. |
| Q3 | For standard channels (hourly/daily/weekly), does the proto currently return the webhook URL? | No. `HourlyChannelInfo` (shared by all three timeframes) only returns `configured: bool`, not the URL or timestamp. Both fields must be added to the proto and populated by the backend. |
| Q4 | For custom channels, is the webhook URL already returned? | `webhook_url` is already a field on `CustomDiscordChannel` in the proto. `updated_at` is not present; it must be added. Need to confirm the `CustomChannelListCard` actually renders it (task: read before editing). |
| Q5 | What value should `updated_at` have for channels set before this feature? | Zero/null. The backend returns a zero `time.Time` which maps to `null` on the frontend; the UI omits the "Last updated" label when null. |
| Q6 | Does `ChannelEntry` (the Go struct backing all three standard timeframes) need `updated_at`? | Yes. Add `UpdatedAt time.Time` to `ChannelEntry` in `apps/grpc/discord_service/config.go`. Set it to `time.Now()` in every `Update*ChannelWebhook` handler. |
| Q7 | Should `CopyDailyToHourly` / `CopyBaseToDaily` / `CopyBaseToWeekly` also set `updated_at`? | Yes — a copy is a webhook change; set `time.Now()` on each copied entry. |

## Open

_None._

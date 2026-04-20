---
# Spec: Discord Webhook Visibility (06)
---

## Goal

The Discord configuration pages (hourly, daily, weekly, custom) currently hide whether a channel's webhook URL has been set and when it was last changed. Users must update a channel to discover what URL is saved, and there is no indication when that URL was last modified. This spec adds webhook URL display (masked, with a copy button) and a "last updated" timestamp to every channel form and card in the four Discord config pages.

## Scope

### In scope

- Add `webhook_url: string` and `updated_at: google.protobuf.Timestamp` to `HourlyChannelInfo` proto message (shared by hourly, daily, and weekly config responses).
- Add `updated_at: google.protobuf.Timestamp` to `CustomDiscordChannel` proto message (`webhook_url` already exists).
- Add `UpdatedAt time.Time` to the Go `ChannelEntry` struct; persist it when any `Update*ChannelWebhook` RPC or `Copy*` RPC writes a webhook URL.
- Populate `webhook_url` and `updated_at` in all three `Get*DiscordConfig` RPC responses.
- Set `updated_at` in `UpdateCustomDiscordChannel` when `webhook_url` is patched.
- Update the TypeScript `HourlyChannelInfo` type in `discord-hourly-actions.ts` to include `webhookUrl: string | null` and `updatedAt: string | null`.
- Update `DiscordChannelForm` to show the current webhook URL (masked, with a copy button) and "Last updated" date when the channel has a URL set.
- Update `CustomChannelListCard` to show the current webhook URL and "Last updated" date.

### Out of scope

- Full change history / audit log of previous webhook URLs.
- Un-masking / reveal button for the full URL (copy button is sufficient).
- Webhook URL display on any page other than the four Discord config pages.
- Changes to the test-message section or ticker-resolve section.
- Proto changes to `UpdateCustomDiscordChannelRequest` / response messages.

## Source excerpts

- `apps/grpc/discord_service/config.go` — `ChannelEntry` struct; `DiscordChannelsConfig` struct.
- `apps/grpc/discord_service/handler.go` — `GetHourlyDiscordConfig`, `UpdateHourlyChannelWebhook`, `CopyDailyToHourly`.
- `apps/grpc/discord_service/handler_daily_weekly.go` — daily/weekly equivalents.
- `apps/grpc/discord_service/custom_channels.go` — `UpdateCustomDiscordChannel`, `loadCustomChannels`.
- `proto/discord/v1/discord.proto` — `HourlyChannelInfo`, `CustomDiscordChannel` messages.
- `apps/ui/web/actions/discord-hourly-actions.ts` — `HourlyChannelInfo` TypeScript type and all three timeframe server actions.
- `apps/ui/web/app/discord/_components/DiscordChannelForm.tsx` — current form; takes `channel: HourlyChannelInfo`.
- `apps/ui/web/app/discord/custom/_components/CustomChannelListCard.tsx` — custom channel display card.

## Acceptance criteria

1. **Proto**
   - `HourlyChannelInfo` message has fields `string webhook_url = 5` and `google.protobuf.Timestamp updated_at = 6`.
   - `CustomDiscordChannel` message has field `google.protobuf.Timestamp updated_at = 8`.
   - Generated TypeScript in `gen/ts/discord/v1/discord.ts` reflects both new fields.

2. **Backend — standard channels**
   - `ChannelEntry` Go struct has `UpdatedAt time.Time` serialised as `updated_at` in JSON.
   - `UpdateHourlyChannelWebhook`, `UpdateDailyChannelWebhook`, `UpdateWeeklyChannelWebhook` each set `entry.UpdatedAt = time.Now()` before saving.
   - `CopyDailyToHourly`, `CopyBaseToDaily`, `CopyBaseToWeekly` each set `UpdatedAt = time.Now()` on every copied entry.
   - `GetHourlyDiscordConfig`, `GetDailyDiscordConfig`, `GetWeeklyDiscordConfig` each populate `WebhookUrl` and `UpdatedAt` in every `HourlyChannelInfo` item returned.

3. **Backend — custom channels**
   - `UpdateCustomDiscordChannel` sets `entry.UpdatedAt = time.Now()` when `webhook_url` is included in the patch.
   - `ListCustomDiscordChannels` populates `UpdatedAt` on every returned `CustomDiscordChannel`.

4. **Frontend types**
   - `HourlyChannelInfo` TypeScript type (in `discord-hourly-actions.ts`) includes `webhookUrl: string | null` and `updatedAt: string | null`.
   - Mappers (`toHourlyChannelInfo` or equivalent) convert `proto.webhookUrl` → `string | null` and `proto.updatedAt?.toISOString() ?? null`.
   - `CustomDiscordChannel` TypeScript type in `discord-custom-actions.ts` includes `updatedAt: string | null`.

5. **Frontend UI — DiscordChannelForm**
   - When `channel.webhookUrl` is non-empty: a read-only row shows the masked URL (`…` + last 20 chars of the URL) with a clipboard copy button.
   - When `channel.updatedAt` is non-null: a "Last updated: {formatted date}" line appears below the masked URL.
   - When `channel.webhookUrl` is empty: no "Current webhook" section is shown; the save input placeholder reads "Paste webhook URL to configure".
   - Copying the full URL to the clipboard via the copy button succeeds silently (no modal); a `toast.success` confirms the copy.

6. **Frontend UI — CustomChannelListCard**
   - When `channel.webhookUrl` is non-empty: masked URL row with copy button is visible.
   - When `channel.updatedAt` is non-null: "Last updated" line is visible.
   - When both are absent: no webhook status section is shown.

## Conventions

- Reuse the existing `isPlaceholderWebhook(url string)` helper in `config.go` to treat placeholder URLs as "not configured" — do **not** return placeholder URLs in `webhook_url`.
- `webhook_url` in the proto response should be an empty string `""` when no real URL is set (consistent with existing `configured: bool` semantics).
- Use `navigator.clipboard.writeText` for the copy button; fall back to a `toast.error` if the Clipboard API is unavailable.
- Masking format: `…` + `url.slice(-20)`. Applied on the frontend, not the backend (the full URL is returned from the backend).
- All new UI tokens must use Tailwind semantic classes (`text-muted-foreground`, `bg-muted`, etc.) — no hardcoded colors.
- The proto field numbering must not conflict with existing fields: `HourlyChannelInfo` currently has fields 1–4; add at 5 and 6. `CustomDiscordChannel` currently has fields 1–7; add `updated_at` at 8.

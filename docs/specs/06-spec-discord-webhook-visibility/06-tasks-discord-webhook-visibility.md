---
# Tasks: Discord Webhook Visibility (06)
---

## Task 01 — Update proto: add webhook_url and updated_at fields

- In `proto/discord/v1/discord.proto`, add to `HourlyChannelInfo`:
  ```protobuf
  string webhook_url = 5;
  google.protobuf.Timestamp updated_at = 6;
  ```
- In the same file, add to `CustomDiscordChannel`:
  ```protobuf
  google.protobuf.Timestamp updated_at = 8;
  ```
- Regenerate TypeScript from proto (run whatever buf/protoc codegen script the project uses) so `gen/ts/discord/v1/discord.ts` reflects both changes.

**Proof:** 06-proofs/06-task-01-proofs.md

---

## Task 02 — Backend: track updated_at in ChannelEntry and standard channel handlers

- In `apps/grpc/discord_service/config.go`, add `UpdatedAt time.Time \`json:"updated_at,omitempty"\`` to `ChannelEntry`.
- In `apps/grpc/discord_service/handler.go`:
  - `UpdateHourlyChannelWebhook`: set `entry.UpdatedAt = time.Now()` before calling `saveConfig`.
  - `CopyDailyToHourly`: set `entry.UpdatedAt = time.Now()` on each entry copied.
  - `GetHourlyDiscordConfig`: populate `WebhookUrl` and `UpdatedAt` (via `timestamppb.New(entry.UpdatedAt)` — omit if zero) on every `HourlyChannelInfo` item. Do not return placeholder URLs (use `isPlaceholderWebhook`).
- Repeat the same three changes in `apps/grpc/discord_service/handler_daily_weekly.go` for daily and weekly.

**Proof:** 06-proofs/06-task-02-proofs.md

---

## Task 03 — Backend: track updated_at in custom channel handler

- In `apps/grpc/discord_service/custom_channels.go`:
  - Add `UpdatedAt time.Time \`json:"updated_at,omitempty"\`` to the custom channel entry struct (or wherever the struct is defined — may live in `config.go` or inline).
  - `UpdateCustomDiscordChannel`: when the patch includes a `webhook_url`, set `entry.UpdatedAt = time.Now()`.
  - `ListCustomDiscordChannels` / `CreateCustomDiscordChannel`: populate `UpdatedAt` on returned proto messages.

**Proof:** 06-proofs/06-task-03-proofs.md

---

## Task 04 — Frontend: update server action types for standard channels

- In `apps/ui/web/actions/discord-hourly-actions.ts`:
  - Add `webhookUrl: string | null` and `updatedAt: string | null` to the `HourlyChannelInfo` TypeScript type.
  - In the mapper function (e.g. `toHourlyChannelInfo`), set:
    ```ts
    webhookUrl: ch.webhookUrl || null,
    updatedAt: ch.updatedAt?.toDate().toISOString() ?? null,
    ```
  - This file handles all three timeframes — one edit covers hourly, daily, and weekly.

**Proof:** 06-proofs/06-task-04-proofs.md

---

## Task 05 — Frontend: update server action types for custom channels

- In `apps/ui/web/actions/discord-custom-actions.ts`:
  - Add `updatedAt: string | null` to the `CustomDiscordChannel` TypeScript type.
  - In the mapper, set `updatedAt: ch.updatedAt?.toDate().toISOString() ?? null`.

**Proof:** 06-proofs/06-task-05-proofs.md

---

## Task 06 — Frontend UI: update DiscordChannelForm

- Read `apps/ui/web/app/discord/_components/DiscordChannelForm.tsx` before editing.
- Add a "Current webhook" section rendered only when `channel.webhookUrl` is non-empty:
  ```
  Label: "Current webhook"
  Row: <span class="font-mono text-xs text-muted-foreground">…{channel.webhookUrl.slice(-20)}</span>
       <CopyButton fullUrl={channel.webhookUrl} />
  If updatedAt: <p class="text-xs text-muted-foreground">Last updated: {format(updatedAt)}</p>
  ```
- The copy button calls `navigator.clipboard.writeText(fullUrl)` and fires `toast.success("Webhook URL copied.")` on success, `toast.error(...)` on failure.
- Use a relative date format for `updatedAt` (e.g. `new Intl.DateTimeFormat` or the existing date util already in the project).
- No changes to the existing save input or button behavior.

**Proof:** 06-proofs/06-task-06-proofs.md

---

## Task 07 — Frontend UI: update CustomChannelListCard

- Read `apps/ui/web/app/discord/custom/_components/CustomChannelListCard.tsx` before editing.
- Add a webhook visibility section analogous to Task 06:
  - Show masked URL with copy button when `channel.webhookUrl` is non-empty.
  - Show "Last updated: …" when `channel.updatedAt` is non-null.

**Proof:** 06-proofs/06-task-07-proofs.md

---

## Task 08 — Validate and capture proof artifacts

- Run `pnpm typecheck` from `apps/ui/web/` and confirm exit 0.
- Run `pnpm lint` from `apps/ui/web/` and confirm exit 0.
- Build the Go gRPC service and confirm it compiles cleanly.
- Start the dev server and manually verify each Discord config page:
  - A channel with a webhook shows the masked URL and copy button.
  - Copy button copies the full URL to clipboard.
  - A channel updated after this deploy shows a "Last updated" date.
  - A channel with no webhook shows no "Current webhook" section.
- Capture command output as proof artifacts.

**Proof:** 06-proofs/06-task-08-proofs.md

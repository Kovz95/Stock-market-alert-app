---
# Validation: Discord Webhook Visibility (06)
---

## Automated verification

From `apps/ui/web/`:

```bash
# TypeScript must compile cleanly
pnpm typecheck

# Lint must pass
pnpm lint
```

**Expected:** Both commands exit 0 with no errors or warnings related to the new fields.

From repo root (Go build):

```bash
go build ./apps/grpc/...
```

**Expected:** Exits 0; no compilation errors in `discord_service/`.

Proto field check:

```bash
grep -n "webhook_url\|updated_at" proto/discord/v1/discord.proto
```

**Expected:** Lines showing `string webhook_url = 5;` and `google.protobuf.Timestamp updated_at = 6;` inside `HourlyChannelInfo`, and `google.protobuf.Timestamp updated_at = 8;` inside `CustomDiscordChannel`.

Generated TS check:

```bash
grep -n "webhookUrl\|updatedAt" gen/ts/discord/v1/discord.ts
```

**Expected:** Both field names appear in the generated file.

Frontend type check:

```bash
grep -n "webhookUrl\|updatedAt" apps/ui/web/actions/discord-hourly-actions.ts
grep -n "updatedAt" apps/ui/web/actions/discord-custom-actions.ts
```

**Expected:** Both files export the new fields on their respective types.

## Manual checks

1. Open `/discord/hourly` (or daily/weekly). A channel known to have a webhook configured shows a masked URL row (`…<last-20-chars>`) and a copy button in its form card.
2. Click the copy button — clipboard receives the full URL (verify by pasting into a text editor).
3. A channel with no webhook set shows no "Current webhook" section and the input placeholder reads "Paste webhook URL to configure".
4. Save a new webhook for a channel. Reload the page. "Last updated" appears with today's date.
5. Open `/discord/custom`. A custom channel with a webhook shows the same masked URL row and "Last updated" label.

## Traceability

- Feature spec: `06-spec-discord-webhook-visibility.md`
- Task breakdown: `06-tasks-discord-webhook-visibility.md`
- Questions and decisions: `06-questions-1-discord-webhook-visibility.md`
- Per-task evidence: `06-proofs/06-task-NN-proofs.md`
- Upstream specs: spec 03 (`03-spec-custom-discord-condition-channels`)

## Definition of done

- [ ] AC 1: `HourlyChannelInfo` and `CustomDiscordChannel` proto messages have new fields with correct field numbers; generated TS updated.
- [ ] AC 2: `ChannelEntry.UpdatedAt` set in all three standard-channel update handlers and all three copy handlers; `GetConfig` RPCs populate `webhook_url` and `updated_at`.
- [ ] AC 3: `UpdateCustomDiscordChannel` sets `updated_at` when webhook is patched; `ListCustomDiscordChannels` populates the field.
- [ ] AC 4: `HourlyChannelInfo` TS type includes `webhookUrl` and `updatedAt`; mapper converts correctly.
- [ ] AC 5: `CustomDiscordChannel` TS type includes `updatedAt`; mapper converts correctly.
- [ ] AC 6: `DiscordChannelForm` renders masked URL + copy button + last-updated label per spec.
- [ ] AC 7: `CustomChannelListCard` renders masked URL + copy button + last-updated label per spec.
- [ ] Proof artifacts contain real command output, not placeholders.

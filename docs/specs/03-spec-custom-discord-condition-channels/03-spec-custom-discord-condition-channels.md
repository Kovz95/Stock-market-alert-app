# Spec: Custom Discord Condition Channels (03)

## Goal

Restore the ability to create, list, toggle, test, and delete **custom Discord channels keyed on an alert condition** in the new Next.js UI at `apps/ui/web/app/discord/`. The old Streamlit page (`pages/Discord_Management.py`, "Custom Discord Channels" tab) let users say "send every alert whose condition is `RSI(14)[-1] < 30` (or any price level) to this webhook." That management surface is missing from the React UI, even though the Python router (`src/services/discord_routing.py`) still dispatches to custom channels when `custom_discord_channels.json` contains entries. This spec re-adds the management surface in the new UI and adds matching gRPC RPCs so it works against the Go `DiscordConfigService`.

## Scope

### In scope

- **Proto** (`proto/discord/v1/discord.proto`): add five RPCs to `DiscordConfigService` for managing custom channels:
  - `ListCustomDiscordChannels` → returns all entries.
  - `CreateCustomDiscordChannel` → `{ name, webhook_url, description, condition, enabled }`.
  - `UpdateCustomDiscordChannel` → `{ name, webhook_url?, description?, condition?, enabled? }` (partial update via field-presence).
  - `DeleteCustomDiscordChannel` → `{ name }`.
  - `SendCustomDiscordChannelTestMessage` → `{ name }`.
- **Go service** (`apps/grpc/discord_service/`): implement the five RPCs. Persist through the same `app_documents.custom_discord_channels` document the Python router reads (see R10 in questions) so Python and Go stay in sync. Validate the `condition` string with the shared `expr` package on create/update and reject parse failures with a clear error. The special keyword `price_level` is always accepted (no parse attempt).
- **Next.js actions** (`apps/ui/web/actions/discord-custom-actions.ts`, new): thin server-action wrappers around the five RPCs, returning plain serializable types. Timestamps → ISO strings.
- **gRPC client wiring**: no change needed — `discordClient` in `lib/grpc/channel.ts` already targets `DiscordConfigService`; the new RPCs appear automatically once proto is regenerated.
- **Jotai store + TanStack hooks** (`apps/ui/web/lib/store/discord-custom.ts`, `apps/ui/web/lib/hooks/useDiscordCustom.ts`, new): `atomWithQuery` for the list; mutations for create/update/delete/test with query-key invalidation.
- **Page** (`apps/ui/web/app/discord/custom/page.tsx`, new): renders an "Add custom channel" form plus a list of existing channels, each with enable/disable toggle, test button, and delete button. Reuse `DiscordChannelForm` styling and existing shadcn primitives.
- **Condition input**: reuse `ConditionBuilder` from `app/alerts/add/_components/` with a single-row configuration. Add a top-level radio: "Match a specific condition (build below)" vs. "Match any price level" (emits the literal string `price_level`). No other presets.
- **Sidebar navigation** (`apps/ui/web/components/app-sidebar.tsx`): add "Custom Channels" under the existing Discord group, linking to `/discord/custom`.
- **No change** to the Python router, the Python Streamlit page, `custom_discord_channels.json`'s on-disk schema, or the alert-dispatch pipeline. The Streamlit page can coexist and write to the same document; the schemas must remain byte-compatible.

### Out of scope

- Per-timeframe custom channels (hourly/daily/weekly splits). Custom channels continue to match all timeframes, as they do today.
- Bulk import/export of custom channels.
- Editing a channel's `name` (the map key). Rename = delete + recreate.
- Migrating custom channels from JSON file to a dedicated Postgres table. If migration is wanted, it's a separate spec.
- Changes to the `DiscordEconomyRouter` Python matching logic (`check_custom_channel_condition`, `is_price_level_condition`, etc.).
- Removing the Streamlit Discord Management page.
- UI for viewing routing history / which alerts fanned out to which custom channel (separate feature).

## Source excerpts

References to existing code that this spec cites as the source of truth for the restored behavior. The Python files are not ephemeral — cite them by path and line number.

- `pages/Discord_Management.py:467–699` — the old "Custom Discord Channels" tab UI this spec restores. Form fields, preset list, existing-channel expander, test button, enable/disable toggle, delete button.
- `src/services/discord_routing.py:221–323` — `check_custom_channel_condition()` (exact-match after `normalize_condition`, plus `price_level` special keyword branch). This is the authoritative matching logic the new UI must stay compatible with.
- `src/services/discord_routing.py:254–260` — `is_price_level_condition()` regex defining what counts as a "price level" condition.
- `src/services/discord_routing.py:207–219` — `_load_custom_channels()` showing the `document_store` key `custom_discord_channels` and fallback path `custom_discord_channels.json`.
- `proto/discord/v1/discord.proto` — the service this spec extends; keep existing messages untouched.
- `apps/ui/web/CLAUDE.md §11` — "How to Add a New Page" — follow this for the new `/discord/custom` route.
- `apps/ui/web/CLAUDE.md §12` — "How to Add a New gRPC Service or RPC" — follow this for wiring the five RPCs.
- `apps/ui/web/app/alerts/add/_components/ConditionBuilder.tsx` / `types.ts` — reused for condition input. The serialized expression string from `conditionEntryToExpressionRaw()` is the value persisted.

## Contract

New messages and RPCs added to `proto/discord/v1/discord.proto`. All timestamps are `google.protobuf.Timestamp`. Booleans default to `false`. The `name` field is the map key; updates and deletes are keyed on it.

```proto
message CustomDiscordChannel {
  string name = 1;              // unique display key (also map key in persisted doc)
  string channel_name = 2;      // Discord display name (e.g. "#harsi-alerts")
  string description = 3;
  string webhook_url = 4;
  string condition = 5;         // canonical expression OR literal "price_level"
  bool enabled = 6;
  google.protobuf.Timestamp created_at = 7;
}

message ListCustomDiscordChannelsRequest {}
message ListCustomDiscordChannelsResponse {
  repeated CustomDiscordChannel channels = 1;
}

message CreateCustomDiscordChannelRequest {
  string name = 1;
  string webhook_url = 2;
  string description = 3;
  string condition = 4;
  bool enabled = 5;
}
message CreateCustomDiscordChannelResponse {
  bool success = 1;
  string error_message = 2;     // e.g. "name already exists", "condition failed to parse"
  CustomDiscordChannel channel = 3;  // populated on success
}

message UpdateCustomDiscordChannelRequest {
  string name = 1;                           // lookup key, immutable
  optional string webhook_url = 2;
  optional string description = 3;
  optional string condition = 4;
  optional bool enabled = 5;
}
message UpdateCustomDiscordChannelResponse {
  bool success = 1;
  string error_message = 2;
  CustomDiscordChannel channel = 3;
}

message DeleteCustomDiscordChannelRequest { string name = 1; }
message DeleteCustomDiscordChannelResponse {
  bool success = 1;
  string error_message = 2;
}

message SendCustomDiscordChannelTestMessageRequest { string name = 1; }
message SendCustomDiscordChannelTestMessageResponse {
  bool success = 1;
  string error_message = 2;
}

service DiscordConfigService {
  // ... existing RPCs unchanged ...
  rpc ListCustomDiscordChannels(ListCustomDiscordChannelsRequest) returns (ListCustomDiscordChannelsResponse);
  rpc CreateCustomDiscordChannel(CreateCustomDiscordChannelRequest) returns (CreateCustomDiscordChannelResponse);
  rpc UpdateCustomDiscordChannel(UpdateCustomDiscordChannelRequest) returns (UpdateCustomDiscordChannelResponse);
  rpc DeleteCustomDiscordChannel(DeleteCustomDiscordChannelRequest) returns (DeleteCustomDiscordChannelResponse);
  rpc SendCustomDiscordChannelTestMessage(SendCustomDiscordChannelTestMessageRequest) returns (SendCustomDiscordChannelTestMessageResponse);
}
```

**Persistence contract (must be preserved):** the Go service reads and writes the `custom_discord_channels` document as a JSON object keyed by `name`. Each value has keys: `webhook_url`, `channel_name`, `description`, `condition`, `enabled`, `created`. The `channel_name` field is auto-derived as `#<lower-kebab-of-name>` on create when not supplied (matching Streamlit behavior at `pages/Discord_Management.py:623`). `created` is an ISO 8601 string. Any unknown keys encountered on read must be preserved on write (do not drop them) to avoid data loss when the Streamlit page owns a key the Go service does not know about.

## Acceptance criteria

1. **Proto contract**
   - `proto/discord/v1/discord.proto` contains the five new RPCs and all new message types listed above, under the existing `DiscordConfigService`.
   - Protobuf code regeneration (`pnpm buf generate` or the project's equivalent) produces `gen/ts/discord/v1/discord.ts` and `gen/go/discord/v1/discord*.pb.go` with the new symbols.
   - `pnpm --filter web typecheck` exits 0 after regen.

2. **Go service implementation**
   - `apps/grpc/discord_service/server.go` implements all five new RPCs. None of them return `codes.Unimplemented`.
   - Create/Update validate `condition` by either (a) exact match `"price_level"` (case-insensitive) or (b) passing `expr.Parse` in the shared `expr/` package — on parse failure the response has `success=false` and `error_message` names the parse error.
   - Create rejects duplicate `name` with `success=false, error_message="channel '<name>' already exists"`.
   - Delete returns `success=false, error_message="channel '<name>' not found"` when the key is missing.
   - SendTest posts a Discord embed (green `0x00FF00`, title "🧪 Test — `<name>`", description includes the stored condition) to the channel's `webhook_url` via the existing Go Discord client; returns `success=true` on HTTP 204, otherwise `success=false` with the status code in `error_message`.
   - All mutations persist via the document-store write path that the Python router reads. Reloading the Streamlit page after a Go mutation shows the new/updated/deleted entry (and vice versa).

3. **Server actions**
   - `apps/ui/web/actions/discord-custom-actions.ts` exports `listCustomDiscordChannels`, `createCustomDiscordChannel`, `updateCustomDiscordChannel`, `deleteCustomDiscordChannel`, `sendCustomDiscordChannelTestMessage`.
   - All actions use `"use server"`. `createdAt` is returned as ISO string (not `Timestamp`).
   - `pnpm --filter web typecheck` exits 0.

4. **Store + hooks**
   - `apps/ui/web/lib/store/discord-custom.ts` exports a `CUSTOM_DISCORD_KEY` and an `atomWithQuery`-based list atom.
   - `apps/ui/web/lib/hooks/useDiscordCustom.ts` exports `useCustomDiscordChannels` (list), `useCreateCustomDiscordChannel`, `useUpdateCustomDiscordChannel`, `useDeleteCustomDiscordChannel`, `useSendCustomDiscordChannelTestMessage`. Each mutation invalidates `CUSTOM_DISCORD_KEY` on success.

5. **Page**
   - `apps/ui/web/app/discord/custom/page.tsx` renders:
     - A "Create custom channel" form with fields: `name` (text), `webhook_url` (password-type input), `description` (textarea), `enabled` (checkbox), and a condition builder block.
     - The condition block is a radio: **"Specific condition"** (shows `ConditionBuilder`) vs. **"Any price level"** (no inputs; submits `"price_level"`).
     - A list of existing channels (one Card per entry) showing: name, `channel_name`, description, condition (in `<code>`), enabled status, test button, enable/disable toggle, delete button.
     - Empty state message when zero channels exist.
   - Submit errors surface via Sonner `toast.error` with the server's `error_message`.
   - Successful mutations call `toast.success` and refetch the list automatically.

6. **Sidebar**
   - `components/app-sidebar.tsx` has a "Custom Channels" entry under the Discord group, linking to `/discord/custom`, with a suitable lucide icon.

7. **End-to-end routing compatibility**
   - Creating a channel with condition `rsi(14)[-1] < 30` and `enabled=true` via the new UI, then triggering a Python-evaluated alert with that exact condition on any ticker, results in the alert being delivered to the configured webhook (manual check — see validation file).
   - Creating a channel with condition `price_level` and triggering an alert with `Close[-1] < 150` delivers to the configured webhook.
   - Disabling the channel via the toggle stops further deliveries on the next scheduler tick.

8. **No regression**
   - `pnpm --filter web typecheck` exits 0.
   - `pnpm --filter web lint` exits 0 (or equivalent).
   - Existing hourly/daily/weekly Discord pages still render and function unchanged.
   - `src/services/discord_routing.py` is not modified.

## Conventions

- **RPC naming**: follow the `<Verb><Resource>` pattern used elsewhere in `DiscordConfigService` (`UpdateHourlyChannelWebhook`, `SendHourlyTestMessage`). `CustomDiscordChannel` is the resource; prefix does not include a timeframe because custom channels span all timeframes.
- **Field presence for partial updates**: use `optional` modifier on updatable fields so proto3 emits field-presence (`HasXxx()` in Go, `.xxx !== undefined` in TS).
- **Persistence**: always round-trip through the `app_documents`/`document_store` abstraction that Python uses. Do not read/write the raw JSON file directly from Go; use the shared document API. If that API is not yet available in Go, add a minimal reader/writer in `database/` that matches the Python contract (same key, same JSON shape).
- **Condition validation**: call `expr.Parse` from the shared `expr/` package. Do not re-implement condition parsing. The `price_level` literal bypasses the parser (match it before calling `expr.Parse`).
- **Timestamps**: store `created` as ISO 8601 UTC. Convert to `google.protobuf.Timestamp` in the gRPC response and to ISO string again in the Next.js action layer.
- **`channel_name` derivation**: if the create request omits `channel_name`, derive it as `"#" + name.toLowerCase().replace(/\s+/g, "-")` (matches `pages/Discord_Management.py:623`).
- **UI primitives**: use existing shadcn components only (`Card`, `Input`, `Textarea`, `Checkbox`, `Button`, `Label`, `Separator`) — no new UI dependencies.
- **Query invalidation key**: `CUSTOM_DISCORD_KEY = ["discord", "custom"] as const`.
- **Condition serialization**: the condition string persisted equals the output of `conditionEntryToExpressionRaw()` from `app/alerts/add/_components/types.ts`, or the literal `"price_level"`. The UI must not post-process or pretty-print the expression — store-what-you-see, send-what-you-stored.
- **Dark mode**: follow `apps/ui/web/CLAUDE.md §13` — use Tailwind semantic tokens only.

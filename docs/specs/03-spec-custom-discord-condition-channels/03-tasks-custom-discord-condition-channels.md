# Tasks: Custom Discord Condition Channels (03)

## Task 01 - Extend the Discord proto with custom-channel RPCs

- Edit `proto/discord/v1/discord.proto`:
  - Add `CustomDiscordChannel` message with fields exactly as in the spec contract.
  - Add request/response pairs for `ListCustomDiscordChannels`, `CreateCustomDiscordChannel`, `UpdateCustomDiscordChannel`, `DeleteCustomDiscordChannel`, `SendCustomDiscordChannelTestMessage`.
  - Use `optional` on all mutable fields in `UpdateCustomDiscordChannelRequest` so proto3 emits field presence.
  - Register all five RPCs on the existing `DiscordConfigService`.
- Regenerate code (e.g. `buf generate` or the project's codegen script referenced in `buf.gen.yaml`).
- Confirm generated files contain the new symbols:
  - `gen/ts/discord/v1/discord.ts` — new message types + service definition.
  - `gen/go/discord/v1/discord.pb.go` and `discord_grpc.pb.go` — new server interface stubs.

**Proof:** 03-proofs/03-task-01-proofs.md

## Task 02 - Implement the Go `DiscordConfigService` methods

- In `apps/grpc/discord_service/server.go` (and a new `custom_channels.go` file alongside if helpful), implement the five RPC methods on `*Server`.
- Read/write the `custom_discord_channels` document via the same storage abstraction used by the Python `document_store`:
  - If a Go equivalent does not exist yet, add a minimal reader/writer in `database/documents.go` (or the closest existing Go data-access package) that targets the same `app_documents` row and preserves unknown JSON keys on round-trip.
- Validate the `condition` string on create/update:
  - If `strings.EqualFold(condition, "price_level")`, accept as-is and normalize to the lowercase literal `"price_level"` on write.
  - Otherwise call `expr.Parse` from the shared `expr/` package. On failure, return `success=false, error_message=fmt.Sprintf("condition failed to parse: %v", err)`.
- Enforce uniqueness on create: reject if `name` already exists.
- `Update`: apply only fields whose proto3 presence is set. Return the full updated channel in the response.
- `Delete`: hard-delete the map entry; error if not present.
- `SendTest`: POST a Discord embed (green `0x00FF00`, title "🧪 Test — <name>", description = channel's stored `condition`) to `webhook_url` using the existing Go Discord sender in `discord/` (add a thin helper if needed).
- Auto-derive `channel_name` as `"#" + kebab(lower(name))` when the create request omits it.
- Persist `created` as RFC3339 UTC. Surface it as `google.protobuf.Timestamp` in the response.

**Proof:** 03-proofs/03-task-02-proofs.md

## Task 03 - Add Next.js server actions

- Create `apps/ui/web/actions/discord-custom-actions.ts` with `"use server"` at the top.
- Export types:
  - `CustomDiscordChannel` — plain object with `name`, `channelName`, `description`, `webhookUrl`, `condition`, `enabled`, `createdAt: string | null`.
  - `MutationResult<T = void>` — `{ success: boolean; errorMessage?: string; channel?: CustomDiscordChannel }` (reuse shape from existing discord actions).
- Export async functions that call `discordClient.<rpc>(...)`:
  - `listCustomDiscordChannels(): Promise<CustomDiscordChannel[]>`
  - `createCustomDiscordChannel(input: Omit<CustomDiscordChannel, "channelName" | "createdAt">): Promise<MutationResult>`
  - `updateCustomDiscordChannel(name: string, patch: Partial<…>): Promise<MutationResult>`
  - `deleteCustomDiscordChannel(name: string): Promise<{ success: boolean; errorMessage?: string }>`
  - `sendCustomDiscordChannelTestMessage(name: string): Promise<{ success: boolean; errorMessage?: string }>`
- Convert `Timestamp` → ISO string in `listCustomDiscordChannels` and wherever `channel` is returned.

**Proof:** 03-proofs/03-task-03-proofs.md

## Task 04 - Add Jotai store and TanStack hooks

- Create `apps/ui/web/lib/store/discord-custom.ts`:
  - `"use client"` header.
  - Export `CUSTOM_DISCORD_KEY = ["discord", "custom"] as const`.
  - Export `customDiscordChannelsQueryAtom` (`atomWithQuery`) calling `listCustomDiscordChannels`.
- Create `apps/ui/web/lib/hooks/useDiscordCustom.ts`:
  - `"use client"` header.
  - Export `useCustomDiscordChannels()` — wraps the atom.
  - Export one mutation hook per action, each invalidating `CUSTOM_DISCORD_KEY` on success via `queryClient.invalidateQueries`.

**Proof:** 03-proofs/03-task-04-proofs.md

## Task 05 - Build `/discord/custom` page

- Create `apps/ui/web/app/discord/custom/page.tsx` and a `_components/` directory beside it.
- Components:
  - `CustomDiscordChannelsContainer.tsx` — top-level client component, orchestrates list + create form.
  - `CustomChannelCreateForm.tsx` — form with `name`, `webhook_url`, `description`, `enabled`, and condition input.
  - `CustomChannelConditionInput.tsx` — radio group: "Specific condition" (renders `ConditionBuilder` in single-row mode) vs. "Any price level" (emits literal `"price_level"`). Returns the serialized condition string to the parent.
  - `CustomChannelListCard.tsx` — per-entry card: shows name, `channel_name`, description, `<code>{condition}</code>`, enable/disable `Checkbox`, "Test" `Button`, "Delete" `Button`.
- Reuse `ConditionBuilder` and `types.ts` from `apps/ui/web/app/alerts/add/_components/`. Do not duplicate condition serialization logic; call `conditionEntryToExpressionRaw` directly.
- Surface all mutation outcomes via Sonner (`toast.success` / `toast.error`).
- Handle the empty-list state with a muted placeholder ("No custom channels yet — create one above.").
- Follow `apps/ui/web/CLAUDE.md §13` for styling (shadcn primitives, semantic tokens, no hardcoded colors).

**Proof:** 03-proofs/03-task-05-proofs.md

## Task 06 - Add sidebar entry

- Edit `apps/ui/web/components/app-sidebar.tsx`:
  - Under the existing Discord group, add `{ title: "Custom Channels", url: "/discord/custom", icon: <FilterIcon /> }` (or another suitable lucide icon already imported, e.g. `ScanSearch`, `StarIcon`).
  - Keep the ordering: Hourly, Daily, Weekly, Custom Channels.

**Proof:** 03-proofs/03-task-06-proofs.md

## Task 07 - Validate and capture proof artifacts

- Run `pnpm --filter web typecheck`; capture output — must exit 0.
- Run `pnpm --filter web lint`; capture output.
- Run the Go service test suite touching `apps/grpc/discord_service/` (or `go test ./apps/grpc/discord_service/...` from repo root); capture output.
- Start the Go discord service and Next.js dev server. Manually:
  - Create a channel with condition `rsi(14)[-1] < 30` + a real webhook. Confirm it appears in the list and that the Streamlit page also lists it (proves shared persistence).
  - Click "Test" — confirm the embed arrives in Discord with the stored condition in the body.
  - Toggle "enabled" off → confirm persisted state flipped in the Streamlit page.
  - Delete the channel → confirm both UIs show it gone.
  - Attempt to create with a garbage condition like `rsi(14[-1]` (invalid) → confirm the UI shows a clear parse-error toast and nothing is persisted.
  - Attempt `price_level` via the radio → confirm the channel is saved with exactly `"price_level"` as its condition.
- Trigger an actual alert (via the scheduler or `scripts/analysis/test_alert_trigger.py`) with a condition matching a saved custom channel. Confirm the webhook fires (check Discord or mock the webhook endpoint).
- Fill in all proof files with real command output and screenshots where appropriate.
- Confirm every acceptance criterion in `03-spec-custom-discord-condition-channels.md` passes; tick the Definition-of-done checklist in the validation file.

**Proof:** 03-proofs/03-task-07-proofs.md

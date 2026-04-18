# Validation: Custom Discord Condition Channels (03)

## Automated verification

From repository root:

```bash
# AC1: Proto contains new RPCs
grep -E "rpc (List|Create|Update|Delete|SendCustomDiscordChannelTestMessage)CustomDiscordChannel" proto/discord/v1/discord.proto
# Expected: 5 matches, one per RPC.

# AC1: Generated TS types contain new symbols
grep -E "CustomDiscordChannel|ListCustomDiscordChannelsRequest" gen/ts/discord/v1/discord.ts | head -10
# Expected: multiple matches (message definitions + service methods).

# AC1: Generated Go code contains new server interface methods
grep -E "ListCustomDiscordChannels|CreateCustomDiscordChannel|SendCustomDiscordChannelTestMessage" gen/go/discord/v1/discord_grpc.pb.go | head -10
# Expected: matches for all five methods in the server interface.

# AC2: Go service implements the RPCs (no Unimplemented*)
grep -RE "func \(s \*Server\) (List|Create|Update|Delete)CustomDiscordChannel|SendCustomDiscordChannelTestMessage" apps/grpc/discord_service/
# Expected: 5 method definitions on *Server.

# AC2: No stub returns for the new RPCs
grep -RE "codes\.Unimplemented" apps/grpc/discord_service/ | grep -i "custom"
# Expected: no output.

# AC3: Server actions file exists and exports the five functions
grep -E "export async function (listCustomDiscordChannels|createCustomDiscordChannel|updateCustomDiscordChannel|deleteCustomDiscordChannel|sendCustomDiscordChannelTestMessage)" apps/ui/web/actions/discord-custom-actions.ts
# Expected: 5 matches.

# AC3 + AC8: Frontend typechecks
pnpm --filter web typecheck
# Expected: exit 0.

# AC4: Store + hook files exist with expected exports
grep -E "CUSTOM_DISCORD_KEY|customDiscordChannelsQueryAtom" apps/ui/web/lib/store/discord-custom.ts
grep -E "useCustomDiscordChannels|useCreateCustomDiscordChannel|useUpdateCustomDiscordChannel|useDeleteCustomDiscordChannel|useSendCustomDiscordChannelTestMessage" apps/ui/web/lib/hooks/useDiscordCustom.ts
# Expected: matches for all listed symbols.

# AC5: Page exists
ls apps/ui/web/app/discord/custom/page.tsx
# Expected: file listed.

# AC5: Page uses ConditionBuilder (reuse, not duplication)
grep -E "from.*alerts/add/_components" apps/ui/web/app/discord/custom/_components/*.tsx
# Expected: at least one import — proving reuse of the alerts condition builder.

# AC6: Sidebar entry added
grep -E "/discord/custom" apps/ui/web/components/app-sidebar.tsx
# Expected: one match.

# AC8: Python router untouched
git diff --stat main -- src/services/discord_routing.py
# Expected: empty output (no modifications).

# AC8: Existing discord pages untouched except where the sidebar is edited
git diff --stat main -- apps/ui/web/app/discord/hourly/ apps/ui/web/app/discord/daily/ apps/ui/web/app/discord/weekly/
# Expected: empty output.

# AC8: Lint passes
pnpm --filter web lint
# Expected: exit 0 (or only pre-existing warnings; no new issues).

# AC2: Go tests
go test ./apps/grpc/discord_service/...
# Expected: exit 0.
```

**Expected outcomes:**
- Proto regeneration produced all five new RPCs in both TS and Go.
- Go service defines all five methods on `*Server` — no `Unimplemented*` fallbacks.
- Next.js server actions, store, hooks, page, and sidebar entry all exist.
- `pnpm --filter web typecheck` and `pnpm --filter web lint` both exit 0.
- `src/services/discord_routing.py` and existing hourly/daily/weekly pages unchanged.

## Traceability

- Feature spec: `03-spec-custom-discord-condition-channels.md`
- Task breakdown: `03-tasks-custom-discord-condition-channels.md`
- Questions and decisions: `03-questions-1-custom-discord-condition-channels.md`
- Per-task evidence: `03-proofs/03-task-NN-proofs.md`
- Upstream specs: none (first spec to extend `DiscordConfigService` beyond economy channels).
- Related existing code: `pages/Discord_Management.py` (old Streamlit UI), `src/services/discord_routing.py` (routing & matching), `proto/discord/v1/discord.proto` (service to extend), `apps/ui/web/app/discord/` (existing pages to sit alongside).

## Manual checks

1. Start the Go discord service and the Next.js dev server (`pnpm --filter web dev`). Navigate to `/discord/custom`.
2. **Create a condition channel:**
   - Enter `name=HARSI Flip Buy Test`, a real webhook, description "Testing HARSI flip".
   - Choose "Specific condition", build `harsi_flip(period=14)[-1] > 0` (or any known condition).
   - Enable = true, submit. Expect success toast; card appears in the list with the stored condition shown verbatim.
3. **Create a price-level channel:**
   - Same page; new entry with `name=All Price Levels`, webhook, description, radio = "Any price level".
   - Submit. Card shows `price_level` as the condition.
4. **Cross-UI sync:** open the old Streamlit `pages/Discord_Management.py` → "Custom Discord Channels" tab. Both channels must appear with identical fields. This proves the shared persistence (R10/O1).
5. **Test button:** click Test on one channel — a green embed arrives in Discord with the channel's condition string.
6. **Toggle enabled off:** in the React UI, uncheck enabled. Refresh Streamlit — the entry shows disabled.
7. **Invalid condition rejection:** try to create a channel with condition `rsi(14[-1]` (bad parens). Expect a toast error naming the parse failure; no persistence.
8. **Duplicate name rejection:** try to create a second channel named `HARSI Flip Buy Test`. Expect a toast error: "channel 'HARSI Flip Buy Test' already exists".
9. **Delete:** delete both test channels. Both UIs show them gone.
10. **End-to-end match:** recreate the HARSI Flip channel. Trigger an alert whose condition exactly matches (via `scripts/analysis/test_alert_trigger.py` or by running the scheduler). Confirm the webhook URL receives the alert (inspect Discord or swap the webhook for an HTTP test endpoint like webhook.site).
11. **Price-level match:** create a `price_level` channel, trigger an alert with `Close[-1] < 200`, confirm delivery.

## Definition of done

- [ ] AC1: Proto extended with five RPCs + messages; codegen produces matching TS and Go symbols.
- [ ] AC2: Go service implements all five methods; shared persistence with Python; `expr` validation; `price_level` bypass; duplicate/not-found errors.
- [ ] AC3: Server actions file exists and `pnpm --filter web typecheck` exits 0.
- [ ] AC4: Store + hooks exist with listed exports; mutations invalidate `CUSTOM_DISCORD_KEY`.
- [ ] AC5: `/discord/custom` page renders list + create form with condition builder + price-level radio; empty state handled; Sonner toasts on success/error.
- [ ] AC6: Sidebar entry added under Discord group.
- [ ] AC7: End-to-end alert delivery confirmed for both a specific-condition channel and a `price_level` channel (manual check #10 and #11).
- [ ] AC8: No changes to `src/services/discord_routing.py` or the existing hourly/daily/weekly React pages. `typecheck` and `lint` both exit 0.
- [ ] Proof artifacts contain real command output and (where applicable) screenshots / Discord message captures — not placeholders.
